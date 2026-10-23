#!/usr/bin/env python3
"""Paper 2 M4 diagnostic (2026-10-23): WHERE do two prediction sets of the same test set differ?

infer_overlap.py --mode nonoverlap re-runs 01 on the original tiles (batch 8) and compares with the existing
test_flood_map predictions (batch 1) by the max |diff| over ALL cells in metres. A large max can be (A) floating-point
noise (TF32 convolutions / different cuDNN algorithms per batch size) amplified by the sinh of the asinh target at
the deepest cells, or (B) a real inconsistency. This script separates the two from the SAVED predictions (CPU only):
  * distribution of |diff| in metres (log histogram, counts above 1e-6 ... 1e-1), all cells and AOI cells;
  * the same in the MODEL's own space z = (asinh(h/s) - mu)/sigma  (noise is roughly uniform in z; a bug is not);
  * per simulated-depth band: n, mean |diff|, max |diff|, n(|diff| > 1 mm);
  * the 10 patches with the largest |diff| (value of both predictions and of the simulation at the argmax).
Reading: (A) => tiny median / p99.9 everywhere, max |dz| ~1e-3..1e-2, large metre diffs only at deep cells.
         (B) => many cells > 1 mm, large |dz|, not concentrated at deep water.

python tools/m4_compare_preds.py --index-csv <testdataset/index_niwa.csv> --ref-vis <results/A/visualization> \
    --new-vis <results/B/visualization> --stats-json <ds8 split_stats_h_asinh.json> --out-json <out.json>
"""
import argparse, csv, glob, json, math, os, time
import numpy as np


def pick(folder, core):
    c = sorted(f for f in glob.glob(os.path.join(folder, core + "_coarse", "*.npy"))
               if os.path.basename(f).startswith(core + "_"))
    return c[0] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--ref-vis", required=True, help="existing test_flood_map predictions (visualization dir)")
    ap.add_argument("--new-vis", required=True, help="re-run predictions (visualization dir)")
    ap.add_argument("--stats-json", required=True, help="split_stats_h_asinh.json of the training set")
    ap.add_argument("--var", default="h")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    node = json.load(open(a.stats_json))["stats_var"]["asinh_by_q"]["90"]
    s, mu, sig = float(node["s"]), float(node["shared"]["mean"]), float(node["shared"]["std"])
    z = lambda h: (np.arcsinh(h / s) - mu) / sig

    rows = [r for r in csv.DictReader(open(a.index_csv, newline="")) if r.get("var", "") == a.var]
    E = np.concatenate([[0.0], np.logspace(-9, 1, 101)])          # |diff| bins in metres (first bin = exact 0)
    EZ = np.concatenate([[0.0], np.logspace(-9, 1, 101)])
    thr = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    bands = [("dry_or_<0.1", -1.0, 0.1), ("0.1-0.5", 0.1, 0.5), ("0.5-1", 0.5, 1.0), ("1-5", 1.0, 5.0),
             ("5-10", 5.0, 10.0), (">=10", 10.0, 1e9)]
    H_all = np.zeros(len(E) - 1); H_aoi = np.zeros(len(E) - 1); HZ_aoi = np.zeros(len(EZ) - 1)
    n_all = n_aoi = 0
    gt_all = {k: 0 for k in thr}; gt_aoi = {k: 0 for k in thr}
    bstat = {b[0]: dict(n=0, sum=0.0, max=0.0, n_gt_1mm=0) for b in bands}
    top = []
    maxdz = 0.0
    missing = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        core = os.path.splitext(os.path.basename(r["coarse_path"]))[0][:-len("_coarse")]
        fr, fn = pick(a.ref_vis, core), pick(a.new_vis, core)
        if fr is None or fn is None:
            missing += 1
            continue
        ref = np.squeeze(np.load(fr)).astype(np.float64)
        new = np.squeeze(np.load(fn)).astype(np.float64)
        gt = np.squeeze(np.load(r["fine_path"])).astype(np.float64)
        m = np.squeeze(np.load(r["mask_fine_path"])) > 0.5
        d = np.abs(ref - new)
        dz = np.abs(z(np.maximum(ref, 0)) - z(np.maximum(new, 0)))
        H_all += np.histogram(np.clip(d, 0, 9.99), E)[0]; n_all += d.size
        da, dza = d[m], dz[m]
        H_aoi += np.histogram(np.clip(da, 0, 9.99), E)[0]; HZ_aoi += np.histogram(np.clip(dza, 0, 9.99), EZ)[0]
        n_aoi += da.size
        maxdz = max(maxdz, float(dza.max()) if dza.size else 0.0)
        for k in thr:
            gt_all[k] += int((d > k).sum()); gt_aoi[k] += int((da > k).sum())
        g = gt[m]
        for name, lo, hi in bands:
            sel = (g >= lo) & (g < hi)
            if sel.any():
                bs = bstat[name]; v = da[sel]
                bs["n"] += int(sel.sum()); bs["sum"] += float(v.sum()); bs["max"] = max(bs["max"], float(v.max()))
                bs["n_gt_1mm"] += int((v > 1e-3).sum())
        j = int(np.argmax(d))
        top.append((float(d.flat[j]), core, int(j // d.shape[1]), int(j % d.shape[1]), float(ref.flat[j]),
                    float(new.flat[j]), float(gt.flat[j]), bool(m.flat[j]), float(dz.flat[j])))
        top = sorted(top, reverse=True)[:10]
        if (i + 1) % 2000 == 0:
            print(f"[info] {i + 1}/{len(rows)} patches ({time.time() - t0:.0f}s)", flush=True)

    def q(H, edges, p):
        c = np.cumsum(H) / max(H.sum(), 1)
        return float(edges[1:][np.searchsorted(c, p)])

    out = dict(index=a.index_csv, ref_vis=a.ref_vis, new_vis=a.new_vis, patches_compared=len(rows) - missing,
               patches_missing=missing, cells_all=n_all, cells_aoi=n_aoi,
               exact_zero_share_aoi=float(H_aoi[0] / max(n_aoi, 1)),
               absdiff_m_aoi_quantiles={f"p{p}": q(H_aoi, E, p) for p in (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999)},
               absdiff_z_aoi_quantiles={f"p{p}": q(HZ_aoi, EZ, p) for p in (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999)},
               absdiff_z_aoi_max=maxdz,
               n_cells_absdiff_m_gt={"all": {str(k): v for k, v in gt_all.items()},
                                     "aoi": {str(k): v for k, v in gt_aoi.items()}},
               by_simulated_depth_band_aoi={k: dict(n=v["n"], mean_absdiff_m=v["sum"] / v["n"] if v["n"] else float("nan"),
                                                    max_absdiff_m=v["max"], n_absdiff_gt_1mm=v["n_gt_1mm"])
                                            for k, v in bstat.items()},
               top10_patches=[dict(max_absdiff_m=t[0], core=t[1], row=t[2], col=t[3], ref_m=t[4], new_m=t[5],
                                   sim_m=t[6], in_aoi=t[7], absdiff_z=t[8]) for t in top])
    json.dump(out, open(a.out_json, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
