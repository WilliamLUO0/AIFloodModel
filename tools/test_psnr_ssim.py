#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the Paper 2 PSNR / SSIM in tools/eval_flood.py (run before any table).

    python tools/test_psnr_ssim.py          # exit code 0 = all pass

Spec checks (paper2_experiments.md, "Metrics for every Paper 2 evaluation"):
  1. identical maps -> PSNR = inf, SSIM = 1 (ssim_patch and ssim_map)
  2. GT + Gaussian noise of known sigma -> PSNR = 10 log10(R^2 / sigma^2) within 0.1 dB
  3. constant shift of the whole map -> SSIM < 1, PSNR finite
  4. masked SSIM == skimage on an all-AOI square (ssim_patch crop rule == skimage full=False)
Extra checks:
  5. strip-wise full-map SSIM == whole-map skimage SSIM (exact)
  6. end-to-end eval_flood.py on a synthetic stitched test set: R, psnr, ssim_map, ssim_patch
     against independent re-computation; ssim_map skipped on an incomplete (val-like) set;
     every pre-existing metric unchanged vs --ssim none.
Needs numpy + scikit-image (no torch, no GPU).
"""
import os
import sys
import csv
import json
import tempfile
import subprocess

import numpy as np
from skimage.metrics import structural_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_flood as ef  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def flood_field(rng, H, W):
    """smooth, sparse, non-negative depth-like field (m)."""
    x = rng.normal(size=(H, W))
    from scipy.ndimage import gaussian_filter
    x = gaussian_filter(x, 6.0)
    x = np.clip((x - 0.02) * 60.0, 0.0, None)
    return x


def sk_ssim_map(g, p, R):
    _, s = structural_similarity(g, p, data_range=R, gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False, full=True)
    return s


def main():
    rng = np.random.default_rng(0)
    full_aoi = np.ones((512, 512))

    print("[1] identical maps")
    g = flood_field(rng, 512, 512)
    R = float(g.max())
    s, n = ef.ssim_patch_sums(g, g, full_aoi, R)
    check(ef.psnr_from_mse(0.0, R) == float("inf"), "PSNR(MSE=0) = inf")
    check(abs(s / n - 1.0) < 1e-12, f"ssim_patch = {s / n:.15f}")
    smap = ef.ssim_full_map_strips(g, g, R, strip_rows=100)
    check(np.allclose(smap, 1.0, atol=1e-12), "ssim_map (strips) = 1 everywhere")

    print("[2] GT + Gaussian noise")
    G = flood_field(rng, 2048, 2048)
    R = 5.0
    for sigma in (0.01, 0.05, 0.2):
        P = G + rng.normal(0.0, sigma, size=G.shape)
        mse = float(np.mean((P - G) ** 2))
        got = ef.psnr_from_mse(mse, R)
        exp = 10 * np.log10(R * R / (sigma * sigma))
        check(abs(got - exp) < 0.1, f"sigma={sigma}: PSNR {got:.3f} vs {exp:.3f} dB")

    print("[3] constant shift")
    P = g + 0.3
    s, n = ef.ssim_patch_sums(P, g, full_aoi, float(g.max()))
    ps = ef.psnr_from_mse(0.09, float(g.max()))
    check(s / n < 1.0, f"ssim_patch = {s / n:.6f} < 1")
    check(np.isfinite(ps), f"PSNR finite ({ps:.3f})")

    print("[4] masked SSIM == skimage on an all-AOI square")
    P = g + rng.normal(0, 0.05, g.shape)
    R = float(g.max())
    mssim_sk = structural_similarity(g, P, data_range=R, gaussian_weights=True, sigma=1.5,
                                     use_sample_covariance=False)          # full=False: crops pad
    s, n = ef.ssim_patch_sums(P, g, full_aoi, R)
    check(abs(s / n - mssim_sk) < 1e-12, f"ssim_patch {s / n:.12f} vs skimage {mssim_sk:.12f}")
    check(n == (512 - 2 * ef.SSIM_PAD) ** 2 and ef.SSIM_PAD == 5, f"crop = {ef.SSIM_PAD} cells, n = {n}")
    # AOI handling: cells outside AOI are excluded AND zeroed before filtering
    m = np.zeros_like(g)
    m[100:400, 50:450] = 1
    s, n = ef.ssim_patch_sums(P, g, m, R)
    ref = sk_ssim_map(np.where(m > 0.5, g, 0), np.where(m > 0.5, P, 0), R)[5:-5, 5:-5][m[5:-5, 5:-5] > 0.5]
    check(n == ref.size and abs(s / n - ref.mean()) < 1e-12, "partial AOI: zero-fill + mask + crop")

    print("[5] strips == whole map")
    G = flood_field(rng, 1500, 700)
    P = G + rng.normal(0, 0.05, G.shape)
    whole = sk_ssim_map(G, P, 4.0)
    for sr in (64, 333, 1024):
        st = ef.ssim_full_map_strips(G, P, 4.0, strip_rows=sr, halo=16)
        check(np.array_equal(st, whole), f"strip_rows={sr}: max |diff| = {np.abs(st - whole).max():.2e}")

    print("[6] end-to-end eval_flood.py")
    e2e(rng)

    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 0 if not FAILS else 1


def e2e(rng):
    P = 64                        # fine patch size
    nr, nc, T = 3, 4, 2
    scen = "100y_42h_0c"
    with tempfile.TemporaryDirectory() as td:
        Hf, Wf = nr * P, nc * P
        aoi = np.zeros((Hf, Wf))
        aoi[10:Hf - 7, 5:Wf - 20] = 1
        aoi[:P, :P] = 0           # patch (0,0) fully outside AOI -> filtered_out = 1
        full_gt, full_pr = {}, {}
        rows = []
        vis = os.path.join(td, "vis")
        for ti in range(T):
            t = f"t{ti + 1:04d}"
            gt = flood_field(rng, Hf, Wf) * aoi
            pr = np.clip(gt + rng.normal(0, 0.05, gt.shape), 0, None)
            pr[aoi < 0.5] = rng.uniform(0, 1, int((aoi < 0.5).sum()))   # junk outside AOI
            full_gt[t], full_pr[t] = gt, pr
            for r in range(nr):
                for c in range(nc):
                    core = f"h_{scen}_{t}_r{r:03d}_c{c:03d}_s8"
                    sl = (slice(r * P, r * P + P), slice(c * P, c * P + P))
                    filt = int(aoi[sl].sum() == 0)
                    row = dict(scenario=scen, var="h", t=t, patch_row=r, patch_col=c, downscale=8,
                               patch_size_fine=P, filtered_out=filt, fine_path="", mask_fine_path="",
                               coarse_path="")
                    if not filt:
                        fp = os.path.join(td, f"{core}_fine.npy")
                        mp = os.path.join(td, f"mask_r{r}_c{c}.npy")
                        np.save(fp, gt[sl].astype(np.float32))
                        np.save(mp, aoi[sl].astype(np.uint8))
                        row.update(fine_path=fp, mask_fine_path=mp)
                        d = os.path.join(vis, core + "_coarse")
                        os.makedirs(d, exist_ok=True)
                        np.save(os.path.join(d, f"{core}_evalname.npy"), pr[sl][None, None].astype(np.float32))
                    rows.append(row)
        idx_csv = os.path.join(td, "index.csv")
        with open(idx_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        def run(extra, tag):
            out = os.path.join(td, f"{tag}.json")
            cmd = [sys.executable, os.path.join(HERE, "eval_flood.py"), "--index-csv", idx_csv,
                   "--vis-root", vis, "--out-json", out,
                   "--out-csv-time", os.path.join(td, f"{tag}_time.csv")] + extra
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stdout, res.stderr)
            return json.load(open(out)), res

        js, _ = run([], "auto")
        dg = js["metrics_dataset_global"]
        # independent reference on the stitched (float32-saved) maps
        m = aoi > 0.5
        G = {t: full_gt[t].astype(np.float32).astype(np.float64) for t in full_gt}
        Pm = {t: full_pr[t].astype(np.float32).astype(np.float64) for t in full_pr}
        R = max(G[t][m].max() for t in G)
        mse = np.mean(np.concatenate([(Pm[t][m] - G[t][m]) ** 2 for t in G]))
        check(abs(dg["psnr_ssim_data_range"] - R) < 1e-9, f"R = {dg['psnr_ssim_data_range']:.6f} (ref {R:.6f})")
        check(abs(dg["psnr"] - 10 * np.log10(R * R / mse)) < 1e-9, f"psnr = {dg['psnr']:.6f}")
        ref_map = np.mean([sk_ssim_map(np.where(m, G[t], 0), np.where(m, Pm[t], 0), R)[m].mean() for t in G])
        check(dg["ssim_map_available"] and abs(dg["ssim_map"] - ref_map) < 1e-12,
              f"ssim_map = {dg['ssim_map']:.12f} (ref {ref_map:.12f}, n_maps {dg['ssim_map_n_maps']})")
        ps, pn = 0.0, 0
        for t in G:
            for r in range(nr):
                for c in range(nc):
                    sl = (slice(r * P, r * P + P), slice(c * P, c * P + P))
                    mm = m[sl]
                    if not mm.any():
                        continue
                    sm = sk_ssim_map(np.where(mm, G[t][sl], 0), np.where(mm, Pm[t][sl], 0), R)[5:-5, 5:-5]
                    k = mm[5:-5, 5:-5]
                    ps += sm[k].sum()
                    pn += int(k.sum())
        check(abs(dg["ssim_patch"] - ps / pn) < 1e-12 and dg["ssim_patch_n_cells"] == pn,
              f"ssim_patch = {dg['ssim_patch']:.12f} (ref {ps / pn:.12f})")
        trows = list(csv.DictReader(open(os.path.join(td, "auto_time.csv"))))
        check(all(k in trows[0] for k in ("psnr_global", "ssim_map", "ssim_patch", "psnr_ssim_data_range")),
              "time CSV carries psnr_global / ssim_map / ssim_patch / psnr_ssim_data_range")

        # pre-existing metrics unchanged vs --ssim none
        jn, _ = run(["--ssim", "none"], "none")
        new_keys = {"psnr", "psnr_ssim_data_range", "ssim_patch", "ssim_patch_n_cells", "ssim_map",
                    "ssim_map_n_maps", "ssim_map_available"}
        same = all((dg[k] == jn["metrics_dataset_global"][k]) or
                   (isinstance(dg[k], float) and np.isnan(dg[k]) and np.isnan(jn["metrics_dataset_global"][k]))
                   for k in dg if k not in new_keys)
        check(same, "all pre-existing dataset_global metrics identical with --ssim none")

        # incomplete set (val-like): remove one patch folder -> ssim_map skipped, ssim_patch kept
        import shutil
        shutil.rmtree(os.path.join(vis, f"h_{scen}_t0001_r001_c001_s8_coarse"))
        jv, res = run([], "val")
        dv = jv["metrics_dataset_global"]
        check((not dv["ssim_map_available"]) and np.isnan(dv["ssim_map"]) and np.isfinite(dv["ssim_patch"]),
              "incomplete maps -> ssim_map skipped (NaN), ssim_patch reported")
        check("ssim_map skipped" in res.stdout, "skip is announced in the log")


if __name__ == "__main__":
    sys.exit(main())
