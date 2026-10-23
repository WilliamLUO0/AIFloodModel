#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_overlap.py -- Paper 2 M4: inference on overlapping tiles + three merges, in ONE pass.

--mode overlap     reads the tiles of tools/make_overlap_tiles.py (patch P, stride = pad = P/2, mirror-padded
                   domain), runs the model once per tile and merges the predictions of every (scenario, t)
                   into three full maps:
                     avg    : uniform average of the 4 predictions covering a cell
                              (Song et al. 2025 DenseUNet; RSwinUNet: overlapping regions averaged)
                     centre : only the central (P/2)x(P/2) of each tile is kept; every cell lies in exactly one
                              tile's centre (cf. He et al. 2023: output patch surrounded by context)
                     hann   : weights w(x) w(y), w(x) = sin^2(pi (x + 0.5) / P) (2-D Hann); with stride P/2 the
                              4 weights of a cell sum to exactly 1
                   Predictions are merged in PHYSICAL units after the same inverse transform + clamp(>=0) as
                   FMSRModel.nondist_validation. The three maps are cut back into the ORIGINAL test tiling and
                   saved exactly like test_flood_map.py saves predictions, so tools/eval_flood.py runs unchanged
                   (and its S16 edge profile is measured relative to the original 512 grid).
--mode nonoverlap  runs the same code on the ORIGINAL test index (stride = P, no padding): validation against
                   the existing test_flood_map predictions (--compare-vis-root) and the timing reference
                   (predictions written only with --save-nonoverlap).

Every AOI cell is checked to be covered by exactly 4 tiles, to lie in exactly one tile centre, and to have
Hann weights summing to 1; any violation aborts.

Outputs (results root, exp = <exp-prefix><merge>):
  <results>/<exp>_eval_<tag><suffix>/visualization/<core>_coarse/<core>_<exp>.npy    (1,1,P,P) float32, metres
  <results>/<exp-prefix>timing_<mode>_eval_<tag><suffix>.json
"""

import os
import sys
import csv
import json
import time
import argparse
from copy import deepcopy

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basicsr.archs import build_network                                    # noqa: E402
from basicsr.data.paired_floodmap_dataset import PairedFloodMapDataset      # noqa: E402
from basicsr.utils import destand_to_physical, denorm_to_physical           # noqa: E402

MERGES = ("avg", "centre", "hann")


def load_net(opt, device):
    net = build_network(deepcopy(opt["network_g"]))
    p = opt["path"]
    ck = torch.load(p["pretrain_network_g"], map_location="cpu")
    key = p.get("param_key_g", "params")
    if key is not None:
        if key not in ck and "params" in ck:
            print(f"[warn] {key} not in checkpoint -> using params")
            key = "params"
        ck = ck[key]
    ck = {(k[7:] if k.startswith("module.") else k): v for k, v in ck.items()}
    net.load_state_dict(ck, strict=bool(p.get("strict_load_g", True)))
    print(f"[info] loaded {p['pretrain_network_g']} [{key}]")
    return net.to(device).eval()


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def core_of(coarse_path):
    b = os.path.splitext(os.path.basename(coarse_path))[0]
    if not b.endswith("_coarse"):
        raise RuntimeError(f"unexpected coarse file name {coarse_path}")
    return b[:-len("_coarse")]


def save_pred(results, exp, tag, suffix, core, arr):
    d = os.path.join(results, f"{exp}_eval_{tag}{suffix}", "visualization", f"{core}_coarse")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"{core}_{exp}.npy"), arr.astype(np.float32)[None, None])


CMP_TOL_M = 1e-3   # nonoverlap re-run vs existing test_flood_map predictions, metres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-opt", required=True, help="eval yml of the model (network, checkpoint, normalisation)")
    ap.add_argument("--mode", required=True, choices=["overlap", "nonoverlap"])
    ap.add_argument("--orig-index", required=True, help="original test index (output tiling, AOI masks)")
    ap.add_argument("--tiles-dir", default="", help="make_overlap_tiles.py output (overlap mode)")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--exp-prefix", required=True, help="e.g. 01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_M4")
    ap.add_argument("--tag", required=True, help="e.g. test100y42h0c")
    ap.add_argument("--suffix", default="_pbs")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--compare-vis-root", default="", help="nonoverlap: existing test_flood_map visualization dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-nonoverlap", action="store_true",
                    help="nonoverlap: also save the predictions (default: validate + time only, nothing written)")
    a = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device(a.device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with open(a.opt) as f:
        opt = yaml.safe_load(f)
    ds_opt = deepcopy(next(iter(opt["datasets"].values())))
    ds_opt["phase"] = "test"
    var = ds_opt.get("target_var", "h")

    orig = [r for r in read_rows(a.orig_index) if r["var"] == var and r["filtered_out"].strip() == "0"]
    P = int(orig[0]["patch_size_fine"])
    by_t_orig = {}
    for r in orig:
        by_t_orig.setdefault((r["scenario"], r["t"]), []).append(r)
    nr = max(int(r["patch_row"]) for r in read_rows(a.orig_index) if r["var"] == var) + 1
    nc = max(int(r["patch_col"]) for r in read_rows(a.orig_index) if r["var"] == var) + 1
    N_y, N_x = nr * P, nc * P

    if a.mode == "overlap":
        with open(os.path.join(a.tiles_dir, "overlap_manifest.json")) as f:
            man = json.load(f)
        pad, stride = man["pad"], man["stride"]
        if man["patch"] != P or pad != stride or stride * 2 != P or man["domain"] != [N_y, N_x]:
            raise SystemExit(f"[fatal] tiles {man} do not match the original index (P={P}, domain {N_y}x{N_x})")
        ds_opt["index_csv"] = os.path.join(a.tiles_dir, "index.csv")
        # AOI mask of the domain (time-invariant) for the coverage checks
        aoi = np.zeros((N_y, N_x), bool)
        seen = set()
        for r in orig:
            rc = (int(r["patch_row"]), int(r["patch_col"]))
            if rc not in seen:
                seen.add(rc)
                aoi[rc[0] * P:rc[0] * P + P, rc[1] * P:rc[1] * P + P] = np.load(r["mask_fine_path"]) > 0.5
    else:
        pad, stride = 0, P
        ds_opt["index_csv"] = a.orig_index

    dataset = PairedFloodMapDataset(ds_opt)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True)
    to_phy = destand_to_physical if dataset.norm == "zscore" else denorm_to_physical
    net = load_net(opt, device)
    exp_of = {m: f"{a.exp_prefix}{m}" for m in (MERGES if a.mode == "overlap" else ("nonoverlap",))}
    if a.compare_vis_root and not os.path.isdir(a.compare_vis_root):     # requested -> must exist (review A4)
        raise SystemExit(f"[fatal] --compare-vis-root {a.compare_vis_root} not found (restore the archived 01 "
                         f"predictions or re-run its test_flood_map eval first, or drop this tag from COMPARE_TAGS)")

    Npy, Npx = N_y + 2 * pad, N_x + 2 * pad
    w1 = np.sin(np.pi * (np.arange(P) + 0.5) / P) ** 2
    hann = np.outer(w1, w1)
    q0, q1 = P // 4, 3 * P // 4
    state = {}

    def new_canvases():
        return dict(s_avg=np.zeros((Npy, Npx)), cnt=np.zeros((Npy, Npx), np.int16),
                    s_hann=np.zeros((Npy, Npx)), w_hann=np.zeros((Npy, Npx)),
                    centre=np.zeros((Npy, Npx)), n_centre=np.zeros((Npy, Npx), np.int16))

    def finalize(key):
        c = state.pop(key)
        sl = (slice(pad, pad + N_y), slice(pad, pad + N_x))
        cnt, ncen, wh = c["cnt"][sl], c["n_centre"][sl], c["w_hann"][sl]
        if not (np.all(cnt[aoi] == 4) and np.all(ncen[aoi] == 1) and np.allclose(wh[aoi], 1.0, atol=1e-9)):
            raise SystemExit(f"[fatal] coverage check failed at {key}: cnt {np.unique(cnt[aoi])}, "
                             f"centre {np.unique(ncen[aoi])}, hann wsum [{wh[aoi].min()}, {wh[aoi].max()}]")
        with np.errstate(invalid="ignore", divide="ignore"):
            maps = {"avg": np.where(cnt > 0, c["s_avg"][sl] / np.maximum(cnt, 1), 0.0),
                    "centre": c["centre"][sl],
                    "hann": np.where(wh > 0, c["s_hann"][sl] / np.where(wh > 0, wh, 1.0), 0.0)}
        for r in by_t_orig[key]:
            ri, ci = int(r["patch_row"]), int(r["patch_col"])
            for m, mp in maps.items():
                save_pred(a.results_root, exp_of[m], a.tag, a.suffix, core_of(r["coarse_path"]),
                          mp[ri * P:ri * P + P, ci * P:ci * P + P])

    done, n_tiles, t_fwd = set(), 0, 0.0
    cmp_max, cmp_n = 0.0, 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            coarse = batch["coarse_flood_map"].to(device, non_blocking=True)
            static = batch["fine_static_feature"].to(device, non_blocking=True)
            sync(); tf = time.time()
            depth, _ = net(coarse, static)
            sync(); t_fwd += time.time() - tf
            phy = to_phy(depth, var, dataset.stats_var, dataset.transform, dataset.h_asinh_q)
            if var == "h":
                phy = phy.clamp_min(0.0)
            phy = phy[:, 0].cpu().numpy().astype(np.float64)
            meta = batch["meta"]
            for k in range(phy.shape[0]):
                key = (meta["scenario"][k], meta["t"][k])
                row, col = int(meta["row"][k]), int(meta["col"][k])
                n_tiles += 1
                if a.mode == "nonoverlap":
                    core = core_of(meta["coarse_path"][k])
                    if a.save_nonoverlap:
                        save_pred(a.results_root, exp_of["nonoverlap"], a.tag, a.suffix, core, phy[k])
                    if a.compare_vis_root:
                        d = os.path.join(a.compare_vis_root, f"{core}_coarse")
                        cand = [f for f in os.listdir(d) if f.startswith(core + "_") and f.endswith(".npy")] if os.path.isdir(d) else []
                        if not cand:
                            raise SystemExit(f"[fatal] --compare-vis-root has no prediction for {core} ({d})")
                        ref = np.squeeze(np.load(os.path.join(d, cand[0])))
                        cmp_max = max(cmp_max, float(np.max(np.abs(ref - phy[k]))))
                        cmp_n += 1
                    continue
                if key in done:
                    raise SystemExit(f"[fatal] tiles of {key} are not contiguous in the index")
                if key not in state:
                    for old in list(state):
                        finalize(old)
                        done.add(old)
                    state[key] = new_canvases()
                c = state[key]
                y0, x0 = row * stride, col * stride
                ys, xs = slice(y0, y0 + P), slice(x0, x0 + P)
                p = phy[k]
                c["s_avg"][ys, xs] += p
                c["cnt"][ys, xs] += 1
                c["s_hann"][ys, xs] += hann * p
                c["w_hann"][ys, xs] += hann
                c["centre"][y0 + q0:y0 + q1, x0 + q0:x0 + q1] = p[q0:q1, q0:q1]
                c["n_centre"][y0 + q0:y0 + q1, x0 + q0:x0 + q1] += 1
        for old in list(state):
            finalize(old)
            done.add(old)
    wall = time.time() - t0

    if a.mode == "overlap" and done != set(by_t_orig):
        raise SystemExit(f"[fatal] merged {len(done)} (scenario,t) maps but the test set has {len(by_t_orig)}")
    info = {"mode": a.mode, "tag": a.tag, "opt": a.opt, "n_tiles_inferred": n_tiles,
            "n_original_patches": len(orig), "tile_ratio_vs_original": n_tiles / float(len(orig)),
            "forward_seconds": t_fwd, "wall_seconds": wall, "batch_size": a.batch_size,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)}
    if a.mode == "nonoverlap" and a.compare_vis_root:
        info["compare_vs_existing"] = {"vis_root": a.compare_vis_root, "patches": cmp_n, "max_abs_diff_m": cmp_max}
        print(f"[validate] {cmp_n} patches vs existing predictions: max |diff| = {cmp_max:.3e} m")
    out = os.path.join(a.results_root, f"{a.exp_prefix}timing_{a.mode}_eval_{a.tag}{a.suffix}.json")
    os.makedirs(a.results_root, exist_ok=True)
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[done] {a.mode}: {n_tiles} tiles, forward {t_fwd:.0f}s, wall {wall:.0f}s -> {out}")
    if a.mode == "nonoverlap" and a.compare_vis_root:                    # review A4: the end-to-end GPU check
        if cmp_n == 0 or cmp_max > CMP_TOL_M:
            raise SystemExit(f"[fatal] validation FAILED: {cmp_n} patches compared, max |diff| = {cmp_max:.3e} m "
                             f"(tolerance {CMP_TOL_M} m) -> the M4 inference path does not reproduce test_flood_map")
        print(f"[validate] PASS (max |diff| {cmp_max:.3e} m <= {CMP_TOL_M} m)")


if __name__ == "__main__":
    main()
