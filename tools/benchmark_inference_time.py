#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forward-only inference LATENCY of the flood-SR models, measured UNIFORMLY on one GPU,
at the factor-8 operating point (the main experiment). Companion to compute_flops.py
(#Params + MACs) -- for a hydrology audience wall-clock is often more convincing than MACs.

Measures ONLY the deep-learning forward pass (per patch). BG-Flood coarse/fine wall-clock
comes from the user's own BG-Flood run logs and is combined by hand afterwards:

    speed-up = T_BGflood(fine, one scenario)
               ---------------------------------------------------------
               T_BGflood(coarse, same scenario) + T_model(N_patches)

where N_patches is the WHOLE scenario, NOT one patch (see the dimensional note below).

Methodology (all six matter; a reviewer will check them):
  1. model.eval() + torch.no_grad(), EMA weights, fp32.
  2. One input batch is put on the GPU ONCE and reused every iteration -- the dataloader /
     disk I/O is NEVER in the timed region (it would dwarf the architecture differences).
  3. The first --warmup iterations are DISCARDED (CUDA context, cuDNN autotune, allocator).
  4. torch.cuda.synchronize() straddles every timed region -- CUDA is async; without it you
     time kernel-launch queueing, not execution.
  5. >=50 iterations; report MEDIAN and IQR (p25..p75), not the mean (robust to stragglers).
  6. All models on the SAME named GPU, SAME batch size -- both are printed for the paper.

DIMENSIONAL NOTE (the classic reviewer trap): a full 8 m flood map is many patches, and a
scenario is (patches-per-map x #timesteps) patches. NEVER compare a single-patch time to a
whole-scenario BG-Flood time. Pass --test-index <testdataset_*/index.csv> and this script
reads the real patch/timestep counts and reports the full-map and full-scenario time too.

Run on a GPU node (see tools/benchmark_inference_time.sh). Example:
    python tools/benchmark_inference_time.py --iters 100 --warmup 20 \
        --test-index /nesi/.../AIFloodModel/testdataset_100y42h0c/index.csv \
        --gpu-name A100-40GB --csv tools/model_infer_time.csv
"""
import os
import csv
import glob
import time
import argparse
import statistics
import yaml
import torch

# importing basicsr.archs registers every *_arch via ARCH_REGISTRY and exposes build_network
from basicsr.archs import build_network

# (display_name, train-config yml). SAME four models as compute_flops.py, same factor-8
# forward(coarse_fm, static_f) contract, so latencies are directly comparable.
MODELS = [
    ("FMPFTV8 (ours, main)", "options/train/01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW.yml"),
    ("HeUNet",               "options/train/02_HeUNet_SRx8_Filter_InbaL1BCE_LW.yml"),
    ("SwinFlood",            "options/train/02_SwinFlood_SRx8_Filter_InbaL1BCE_LW.yml"),
    ("RSwinUNet",            "options/train/02_RSwinUNet_SRx8_Filter_InbaL1BCE_LW.yml"),
]

COARSE_HW = 64
UPSCALE = 8
FINE_HW = COARSE_HW * UPSCALE   # 512


def _ckpt_iter(path):
    """net_g_600000.pth -> 600000 ; -1 if it doesn't parse."""
    b = os.path.basename(path)
    try:
        return int(b[len("net_g_"):-len(".pth")])
    except ValueError:
        return -1


def load_model(cfg_path, exp_root, load_weights, device):
    """Build the arch and (optionally) load its LATEST trained EMA checkpoint. The main
    model trained to 1M iters; the baselines are stopped ~600k (still training), so we
    auto-pick the highest-iteration net_g_*.pth per model rather than assuming 1M.
    LATENCY IS ITERATION- AND WEIGHT-INDEPENDENT (same layers/shapes; PFT's top-k count
    is fixed by config, not by learned values) -> 600k vs 1M vs random-init all time the
    same, so the comparison stays valid; we load real weights only for defensibility."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    net_opt = opt["network_g"]
    model = build_network(net_opt)

    loaded = "random-init"
    if load_weights:
        exp_name = opt.get("name") or os.path.splitext(os.path.basename(cfg_path))[0]
        mdir = os.path.join(exp_root, exp_name, "models")
        ckpts = [p for p in glob.glob(os.path.join(mdir, "net_g_*.pth")) if _ckpt_iter(p) >= 0]
        if ckpts:
            ckpt = max(ckpts, key=_ckpt_iter)
            it = _ckpt_iter(ckpt)
            sd = torch.load(ckpt, map_location="cpu")
            key = "params_ema" if isinstance(sd, dict) and "params_ema" in sd else \
                  ("params" if isinstance(sd, dict) and "params" in sd else None)
            state = sd[key] if key else sd
            model.load_state_dict(state, strict=True)
            tag = "EMA" if key == "params_ema" else (key or "raw")
            loaded = f"{tag}@{it // 1000}k"
        else:
            loaded = f"random-init (no net_g_*.pth in {mdir})"

    model = model.to(device).float().eval()
    return model, net_opt, loaded


@torch.no_grad()
def bench(model, coarse, static, warmup, iters, cuda):
    """Return a list of per-forward times in ms (one entry per timed iteration)."""
    for _ in range(warmup):                 # discard: CUDA ctx / cuDNN autotune / allocator
        model(coarse, static)
    if cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(coarse, static)
        if cuda:
            torch.cuda.synchronize()        # CUDA is async -> sync before reading the clock
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def quartiles(xs):
    """(p25, median, p75) -- statistics.quantiles needs >=2 points."""
    if len(xs) < 2:
        m = xs[0] if xs else float("nan")
        return m, m, m
    q = statistics.quantiles(xs, n=4)       # [p25, p50, p75]
    return q[0], statistics.median(xs), q[2]


def scan_test_index(path):
    """Return (n_patches, n_timesteps, patches_per_timestep) for the h-downscaling task
    from a testdataset index.csv, or (None, None, None) if unreadable.

    The factor-8 testdataset lists h/u/v rows (column `var`); the models downscale h, so
    we count ONLY var=='h' rows -- counting all rows would triple the patch count. (h-only
    testdatasets have no u/v rows, so the filter is a no-op there.) Timestep column is `t`
    (make_patches.py header: scenario,var,t,time_index,patch_row,patch_col,...)."""
    if not path or not os.path.isfile(path):
        return None, None, None
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        tcol = next((c for c in ("t", "timestep", "time", "frame") if c in cols), None)
        has_var = "var" in cols
        n_patches = 0
        tset = set()
        for row in reader:
            if has_var and row["var"] != "h":
                continue
            n_patches += 1
            if tcol is not None:
                tset.add(row[tcol])
    n_t = len(tset) if tset else None
    ppt = (n_patches / n_t) if n_t else None
    return n_patches, n_t, ppt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1, help="patches per forward (default 1 = deployment)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100, help=">=50 recommended")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--exp-root", default="experiments", help="dir holding <exp>/models/net_g_1000000.pth")
    ap.add_argument("--no-load-weights", action="store_true", help="skip EMA load (latency is identical)")
    ap.add_argument("--test-index", default="", help="testdataset_*/index.csv -> full-map & full-scenario time")
    ap.add_argument("--gpu-name", default="", help="label printed into the report (e.g. A100-40GB)")
    ap.add_argument("--csv", default="", help="optional path to also write a CSV")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if args.device.startswith("cuda") and not cuda:
        print("[warn] CUDA requested but not available -> falling back to CPU (timing not paper-grade).")
        args.device = "cpu"
    if cuda:
        torch.backends.cudnn.benchmark = True   # autotune fastest kernels; --warmup absorbs it
        gpu = args.gpu_name or torch.cuda.get_device_name(0)
    else:
        gpu = args.gpu_name or "CPU"

    n_patches, n_t, ppt = scan_test_index(args.test_index)

    print(f"Device: {gpu} | batch: {args.batch} | warmup: {args.warmup} | timed iters: {args.iters} | fp32")
    print(f"Input @ factor 8 (per forward): coarse [{args.batch},3,64,64] + static [{args.batch},7,512,512]")
    if n_patches:
        extra = f", {n_t} timesteps, {ppt:.1f} patches/map" if n_t else ""
        print(f"Test set ({os.path.basename(os.path.dirname(args.test_index))}): {n_patches} patches{extra}")
    print("-" * 92)
    print(f"{'model':<26} {'weights':<14} {'med ms/patch':>13} {'IQR':>8} {'mean±std ms':>18}")
    print("(ms/patch = per-forward time at this batch, divided by batch)")
    print("-" * 92)

    rows = []
    for name, cfg_path in MODELS:
        if not os.path.isfile(cfg_path):
            print(f"{name:<26} [skip] config not found: {cfg_path}")
            continue
        model, net_opt, loaded = load_model(
            cfg_path, args.exp_root, load_weights=not args.no_load_weights, device=args.device)

        cc = int(net_opt.get("coarse_in_chans", 3))
        cs = int(net_opt.get("static_in_chans", 7))
        coarse = torch.randn(args.batch, cc, COARSE_HW, COARSE_HW, device=args.device)
        static = torch.randn(args.batch, cs, FINE_HW, FINE_HW, device=args.device)

        per_forward = bench(model, coarse, static, args.warmup, args.iters, cuda)
        per_patch = [t / args.batch for t in per_forward]
        p25, med, p75 = quartiles(per_patch)
        mean = statistics.mean(per_patch)
        std = statistics.stdev(per_patch) if len(per_patch) > 1 else 0.0

        print(f"{name:<26} {loaded:<14} {med:>13.3f} {p75 - p25:>8.3f} {f'{mean:.3f}±{std:.3f}':>18}")
        rows.append((name, loaded, med, p25, p75, mean, std, n_patches, ppt))

        del model, coarse, static
        if cuda:
            torch.cuda.empty_cache()

    # ---- full-map / full-scenario extrapolation (per-patch median x patch count) ----
    if n_patches:
        print("-" * 84)
        hdr = f"{'model':<26} {'full map (s)':>14}" + (f" {'full scenario (s)':>18}" if n_patches else "")
        print(hdr)
        print("-" * 84)
        for name, loaded, med, p25, p75, mean, std, npat, pp in rows:
            map_s = (pp * med / 1e3) if pp else float("nan")
            scen_s = npat * med / 1e3
            line = f"{name:<26} {map_s:>14.2f}" + f" {scen_s:>18.2f}"
            print(line)
        print("\nfull map    = patches-per-map x ms/patch (one 8 m timestep).")
        print("full scenario = total test patches x ms/patch (all timesteps).")
        print("For the speed-up ratio, use the FULL-SCENARIO model time with the coarse/fine")
        print("BG-Flood wall-clock from your own run logs (same scenario, same sim length).")

    if args.csv and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "weights", "ms_per_patch_median", "ms_p25", "ms_p75",
                        "ms_mean", "ms_std", "test_patches", "patches_per_map", "gpu", "batch"])
            for name, loaded, med, p25, p75, mean, std, npat, pp in rows:
                w.writerow([name, loaded, f"{med:.4f}", f"{p25:.4f}", f"{p75:.4f}",
                            f"{mean:.4f}", f"{std:.4f}", npat if npat else "",
                            f"{pp:.2f}" if pp else "", gpu, args.batch])
        print(f"\n[saved] {args.csv}")


if __name__ == "__main__":
    main()
