#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute #Params and FLOPs for the flood-SR models, UNIFORMLY, via a real forward
pass counted by fvcore (NOT the per-arch hand-written flops(), which are missing
for HeUNet/SwinFlood and are approximate MACs elsewhere).

All models are measured at the factor-8 operating point (the main experiment):
    coarse_fm : [1, coarse_in_chans, 64, 64]     (coarse-grid input, 64x64)
    static_f  : [1, static_in_chans, 512, 512]   (fine-grid topography, 64*8)
so the numbers are directly comparable across architectures.

CPU-only, no GPU. Run on a login node, or via tools/compute_flops.sh.

NOTE on convention: fvcore's FlopCountAnalysis counts multiply-accumulates (MACs),
which most SR/vision papers report as "FLOPs". True FLOPs ~= 2 x MACs. Both printed.
Elementwise ops fvcore cannot trace (softmax, norms, index gathers) are not counted,
but that undercount is UNIFORM across models so the comparison stays fair.

The ONE model-specific gap is FloodMapPFTV8's custom sparse-attention kernels
(smm_cuda: SMM_QmK / SMM_AmV) -- fvcore counts them as 0, while the baselines use
plain matmul attention that IS counted. We close that gap analytically via smm_macs()
and ADD it to the fvcore total (see that function for the derivation), so the reported
GMac/GFLOPs are the corrected, apples-to-apples numbers.
"""
import os
import argparse
import yaml
import torch

# importing basicsr.archs registers every *_arch via ARCH_REGISTRY and exposes build_network
from basicsr.archs import build_network

# (display_name, train-config yml with the network_g we want to size).
# FMPFTV8 is our main model. The other three are the flood baselines with the same
# forward(coarse_fm, static_f) contract, so all four are directly comparable.
# PFT is intentionally NOT here: the original PFT is a single-input image-SR net (no
# coarse/static flood contract), so its FLOPs are not comparable and it is skipped.
MODELS = [
    ("FMPFTV8 (ours, main)", "options/train/01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW.yml"),
    ("HeUNet",               "options/train/02_HeUNet_SRx8_Filter_InbaL1BCE_LW.yml"),
    ("SwinFlood",            "options/train/02_SwinFlood_SRx8_Filter_InbaL1BCE_LW.yml"),
    ("RSwinUNet",            "options/train/02_RSwinUNet_SRx8_Filter_InbaL1BCE_LW.yml"),
]

# factor-8 operating point (shared by every model)
COARSE_HW = 64
UPSCALE = 8
FINE_HW = COARSE_HW * UPSCALE   # 512


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_macs(model, inputs):
    """Return (MACs, unsupported_ops, by_module) via fvcore; (None, None, None) if
    fvcore is unavailable. We keep stderr clean (warnings off) but RETURN the reports:
      - unsupported_ops {op: count}: ops fvcore could not trace. This VERIFIES (not just
        assumes) that the undercount is uniform across models -- elementwise/norm ops for
        all; the custom smm_cuda kernels show up here for FloodMapPFTV8 only.
      - by_module {qualname: MACs}: authoritative per-component breakdown (so the decoder-
        vs-encoder-vs-attention split is measured, not hand-derived)."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None, None, None
    fca = FlopCountAnalysis(model, inputs)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    total = fca.total()
    return total, dict(fca.unsupported_ops()), dict(fca.by_module())


def smm_macs(net_opt):
    """Analytic MACs for the sparse-attention ops fvcore CANNOT trace.

    FloodMapPFTV8's Progressive-Focusing-Attention uses two custom smm_cuda kernels
    (SMM_QmK for q.k^T over top-k keys, SMM_AmV for attn.v over top-k), wrapped in
    torch.autograd.Function -> fvcore sees an opaque custom op and counts 0 MACs.
    (The 1st layer of each PFA chain is still DENSE q@k / attn@v, which fvcore DOES
    count, so we add ONLY the sparse ops here.) Every baseline uses standard window
    attention (plain matmul) that fvcore counts fully, so this term is V8-only.

    Geometry (factor-8 operating point, patch_size=1):
      PFT runs on the flood_map_size x flood_map_size bottleneck; window W -> n=W*W
      tokens/window, nwin=(fm/W)^2 windows, b_ = batch(1) * nwin, head_dim d=embed/heads.
      One SMM op costs (b_*heads) * n * width * d MACs, where `width` = #candidate keys.

    PFA threading: pfa_indices is a per-shift (0/1) chain; shift alternates 0/1 within
    each stage; layer_id is global and indexes num_topk. Along a chain the candidate
    set shrinks 256->...->topk. Per layer:
      QmK width = incoming candidate count (None=dense, counted -> add 0);
      AmV width = topk after this layer's sparsification (None=dense -> add 0).
    """
    if net_opt.get("type") != "FloodMapPFTV8":
        return 0
    W = int(net_opt.get("window_size", 16))
    n = W * W
    fm = int(net_opt.get("flood_map_size", 64))
    nwin = (fm // W) ** 2
    embed = int(net_opt["embed_dim"]); heads = int(net_opt["num_heads"])
    d = embed // heads
    depths = list(net_opt["depths"]); topk = list(net_opt["num_topk"])
    b_ = 1 * nwin
    U = b_ * heads * n * d                       # MACs per unit `width`

    # per global layer_id: its shift slot (0 for even position within a stage, else 1)
    shift = []
    for dep in depths:
        for i in range(dep):
            shift.append(0 if (i % 2 == 0) else 1)

    total_width = 0
    for s in (0, 1):
        chain = [topk[i] for i in range(len(topk)) if shift[i] == s]
        pfa = None                               # current candidate count (None => dense path)
        for tk in chain:
            if pfa is not None:                  # QmK: sparse over `pfa` candidates
                total_width += pfa
            if tk < n:                           # sparsify -> pfa_indices now holds tk
                pfa = tk
            if pfa is not None:                  # AmV: sparse over `pfa` (= tk if just set)
                total_width += pfa
    return U * total_width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="optional path to also write a CSV")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    rows = []
    print(f"{'model':<26} {'Params(M)':>11} {'GMac':>10} {'GFLOPs(=2xMac)':>16}")
    print("-" * 67)

    for name, cfg_path in MODELS:
        if not os.path.isfile(cfg_path):
            print(f"{name:<26}  [skip] config not found: {cfg_path}")
            continue
        with open(cfg_path, "r", encoding="utf-8") as f:
            opt = yaml.safe_load(f)
        net_opt = opt["network_g"]

        model = build_network(net_opt).eval()

        cc = int(net_opt.get("coarse_in_chans", 3))
        cs = int(net_opt.get("static_in_chans", 7))
        coarse = torch.randn(1, cc, COARSE_HW, COARSE_HW)
        static = torch.randn(1, cs, FINE_HW, FINE_HW)

        total, _ = count_params(model)
        macs, unsup, by_mod = count_macs(model, (coarse, static))
        smm = smm_macs(net_opt)                  # custom SMM ops fvcore can't trace (V8 only)

        params_m = total / 1e6
        gmac_smm = smm / 1e9
        if macs is None:
            gmac_fvcore = float("nan"); gmac = float("nan"); gflops = float("nan")
            note = "  (fvcore not installed -> MACs skipped)"
        else:
            gmac_fvcore = macs / 1e9
            gmac = gmac_fvcore + gmac_smm         # total = fvcore-traced + analytic SMM
            gflops = 2 * gmac
            note = f"  (incl +{gmac_smm:.3f} SMM)" if smm > 0 else ""

        print(f"{name:<26} {params_m:>11.3f} {gmac:>10.2f} {gflops:>16.2f}{note}")
        rows.append((name, params_m, gmac_fvcore, gmac_smm, gmac, gflops))

        # --- per-model diagnostics: what fvcore missed + authoritative component split ---
        if macs:
            ops = ", ".join(f"{k}×{v}" for k, v in sorted(unsup.items(), key=lambda kv: -kv[1]))
            print(f"    [unsupported ops fvcore couldn't trace] {ops if ops else 'none'}")
            top = {k: v for k, v in by_mod.items() if k and "." not in k}  # direct children only
            for k, v in sorted(top.items(), key=lambda kv: -kv[1]):
                if v > 0:
                    print(f"      {k:<26} {v/1e9:>8.3f} GMac  ({100 * v / macs:>5.1f}% of fvcore)")
            if smm > 0:
                print(f"      {'(SMM sparse-attn, analytic)':<26} {gmac_smm:>8.3f} GMac  (not in by_module)")

    if args.csv and rows:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "params_M", "GMac_fvcore", "GMac_smm", "GMac_total", "GFLOPs_total"])
            w.writerows(rows)
        print(f"\n[saved] {args.csv}")

    print("\nInput @ factor 8: coarse [1,3,64,64] + static [1,7,512,512].")
    print("GMac = multiply-accumulates (what most papers call 'FLOPs'); GFLOPs = 2 x GMac.")
    print("GMac(total) = fvcore-traced ops + analytic SMM (FloodMapPFTV8's custom sparse")
    print("attention, which fvcore counts as 0; baselines use plain matmul -> SMM=0).")


if __name__ == "__main__":
    main()
