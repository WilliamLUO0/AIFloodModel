"""Sign agreement of the predicted flow velocities (WRR Table 2, round 2).

Over the cells inside the AOI that are FLOWING (|value| >= 0.1 m/s) in BOTH the
simulated and the predicted fine-grid flood map, the percentage whose predicted
velocity has the same sign as the simulated one.  Pooled over every patch and
time step of the set, exactly like the *_global metrics of eval_flood.py: the
same patches (the *_coarse folders under --vis-root), the same loader, the same
AOI mask (mask > 0.5) and the same 0.1 threshold on the magnitude.

Self-check: the number of cells flowing in both maps is the true-positive count
of eval_flood.py's flowing/nonflowing classification, so with --summary-json it
is compared with recall * target_prevalence * num_valid_sum of that summary and
the run fails if they differ.

Inputs are the three that eval_flood.py reads: the per-patch predictions
written by basicsr/test_flood_map.py (options/test/<eval>_pbs.yml, save_flood_map:
true) under results/<eval>_pbs/visualization, the dataset index (fine_path =
simulated map, mask_fine_path = AOI) and the eval summary (cross-check).
Batch for all sets of run 01: tools/sign_agreement_batch.pbs (NIWA, qsub).

Usage (repo root, one variable and one set per call):
  python tools/sign_agreement.py --index-csv <index.csv> --vis-root <results/<eval>/visualization> \
      --var u --summary-json <results/<eval>/eval_<tag>_summary.json> --out-json <out.json>
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_flood as ef  # noqa: E402

LEVELS = [("slow", 0.1, 0.5), ("moderate", 0.5, 1.0), ("fast", 1.0, np.inf)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--vis-root", required=True)
    ap.add_argument("--var", required=True, choices=["u", "v"])
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    idx = ef.build_index_map(a.index_csv)
    cores = [ef.parse_folder_name(os.path.basename(f.rstrip("/")))
             for f in sorted(glob.glob(os.path.join(a.vis_root, "*_coarse")))]
    cores = [c for c in cores if c and ef.parse_core_key(c)["var"] == a.var]
    if not cores:
        raise SystemExit("no %s patches under %s" % (a.var, a.vis_root))

    thr = ef.DEPTH_EVENT_THRESHOLD
    n_valid = n_both = n_agree = 0
    lv_both = [0] * 3; lv_agree = [0] * 3
    for i, core in enumerate(cores):
        pred, gt, mask, _, _ = ef._load_patch_pred_gt_mask(core, idx, a.vis_root, "model")
        m = np.asarray(mask, dtype=np.float64) > 0.5
        pred = np.asarray(pred, dtype=np.float64); gt = np.asarray(gt, dtype=np.float64)
        ga = np.abs(gt)
        both = m & (ga >= thr) & (np.abs(pred) >= thr)
        agree = both & (np.sign(pred) == np.sign(gt))
        n_valid += int(m.sum()); n_both += int(both.sum()); n_agree += int(agree.sum())
        for k, (_, lo, hi) in enumerate(LEVELS):
            sel = both & (ga >= lo) & (ga < hi)
            lv_both[k] += int(sel.sum()); lv_agree[k] += int((sel & agree).sum())
        if (i + 1) % 2000 == 0:
            print("[info] %d / %d patches" % (i + 1, len(cores)), flush=True)

    out = {"var": a.var, "vis_root": a.vis_root, "n_patches": len(cores),
           "num_valid_sum": n_valid, "n_flowing_both": n_both, "n_same_sign": n_agree,
           "sign_agreement_pct": 100.0 * n_agree / n_both if n_both else float("nan"),
           "by_simulated_level": {name: {"n_flowing_both": b, "n_same_sign": g,
                                         "sign_agreement_pct": 100.0 * g / b if b else float("nan")}
                                  for (name, _, _), b, g in zip(LEVELS, lv_both, lv_agree)}}
    if a.summary_json:
        s = json.load(open(a.summary_json))
        g = s["metrics_dataset_global"]
        tp = g["recall"] * g["target_prevalence"] * s["num_valid_sum"]
        out["check_tp_from_summary"] = tp
        out["check_num_valid_sum_summary"] = s["num_valid_sum"]
        if s["num_valid_sum"] != n_valid or abs(tp - n_both) > 0.5:
            json.dump(out, open(a.out_json, "w"), indent=1)
            raise SystemExit("[ERROR] population differs from the summary: num_valid %d vs %d, TP %.1f vs %d"
                             % (n_valid, s["num_valid_sum"], tp, n_both))
    json.dump(out, open(a.out_json, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
