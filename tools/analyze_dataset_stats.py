#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import math
import argparse
import numpy as np
from collections import defaultdict

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def t_sort_key(t_tag: str):
    m = re.match(r"t(\d+)$", (t_tag or "").strip())
    return int(m.group(1)) if m else 10**18


class AggStats:
    """Streaming aggregation for min/max/mean/std using sum/sumsq (masked values only)."""
    __slots__ = ("n", "sum", "sumsq", "min", "max", "num_patches")

    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = float("inf")
        self.max = float("-inf")
        self.num_patches = 0

    def update_from_values(self, vals_1d: np.ndarray):
        """vals_1d: 1D float array of selected values (already masked & finite)."""
        self.num_patches += 1
        if vals_1d is None or vals_1d.size == 0:
            return
        v = vals_1d.astype(np.float64, copy=False)
        cnt = int(v.size)
        s = float(v.sum())
        ss = float(np.square(v).sum())
        mn = float(v.min())
        mx = float(v.max())

        self.n += cnt
        self.sum += s
        self.sumsq += ss
        if mn < self.min:
            self.min = mn
        if mx > self.max:
            self.max = mx

    def mean(self):
        return (self.sum / self.n) if self.n > 0 else float("nan")

    def std(self):
        if self.n <= 0:
            return float("nan")
        mu = self.sum / self.n
        var = self.sumsq / self.n - mu * mu
        if var < 0 and var > -1e-12:
            var = 0.0
        return math.sqrt(var) if var >= 0 else float("nan")


def load_npy(path: str, mmap: bool = False):
    if mmap:
        return np.load(path, mmap_mode="r")
    return np.load(path, allow_pickle=False)


def masked_values(arr: np.ndarray, mask: np.ndarray):
    """
    Return 1D selected values: isfinite(arr) & (mask==1).
    mask shape mismatch: crop to overlap (defensive).
    """
    if arr is None or mask is None:
        return np.array([], dtype=np.float64)

    if arr.shape != mask.shape:
        # defensive: crop to min overlap
        h = min(arr.shape[0], mask.shape[0])
        w = min(arr.shape[1], mask.shape[1])
        arr = arr[:h, :w]
        mask = mask[:h, :w]

    sel = np.isfinite(arr) & (mask == 1)
    if not np.any(sel):
        return np.array([], dtype=np.float64)
    return arr[sel].astype(np.float64, copy=False)


def compute_patch_stats_masked(val_path: str, mask_path: str, mmap: bool = False):
    arr = load_npy(val_path, mmap=mmap)
    m = load_npy(mask_path, mmap=mmap)

    vals = masked_values(arr, m)
    if vals.size == 0:
        return dict(n=0, min=np.nan, max=np.nan, mean=np.nan, std=np.nan)

    vmin = float(vals.min())
    vmax = float(vals.max())
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    return dict(n=int(vals.size), min=vmin, max=vmax, mean=mean, std=std), vals


def analyze(index_csv: str,
            out_dir: str,
            target_var: str = "h",
            grid: str = "coarse",
            max_rows: int = 0,
            mmap: bool = False):

    os.makedirs(out_dir, exist_ok=True)

    patch_stats_csv = os.path.join(out_dir, f"patch_stats_{target_var}_{grid}_masked.csv")
    t_stats_csv = os.path.join(out_dir, f"t_stats_{target_var}_{grid}_masked.csv")
    scenario_stats_csv = os.path.join(out_dir, f"scenario_stats_{target_var}_{grid}_masked.csv")
    dataset_stats_txt = os.path.join(out_dir, f"dataset_stats_{target_var}_{grid}_masked.txt")

    # aggregators
    agg_dataset = AggStats()
    agg_by_t = defaultdict(AggStats)        # key: (scenario, t)
    agg_by_scenario = defaultdict(AggStats) # key: scenario
    scenario_t_set = defaultdict(set)

    # select correct column names
    val_key = "coarse_path" if grid == "coarse" else "fine_path"
    mask_key = "mask_coarse_path" if grid == "coarse" else "mask_fine_path"

    # open writers
    pf = open(patch_stats_csv, "w", newline="")
    pw = csv.writer(pf)
    pw.writerow([
        "scenario", "t", "time_index", "patch_row", "patch_col",
        "grid", "val_path", "mask_path",
        "n_masked", "min", "max", "mean", "std"
    ])

    total = 0
    used_rows = 0
    missing_files = 0

    with open(index_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        iterator = reader
        if tqdm is not None:
            iterator = tqdm(reader, desc="Reading index.csv", unit="rows")

        for row in iterator:
            total += 1
            if max_rows and total > max_rows:
                break

            if row.get("var", "") != target_var:
                continue
            if str(row.get("filtered_out", "0")).strip() == "1":
                continue

            val_path = (row.get(val_key, "") or "").strip()
            mask_path = (row.get(mask_key, "") or "").strip()
            if (not val_path) or (not mask_path):
                continue
            if (not os.path.exists(val_path)) or (not os.path.exists(mask_path)):
                missing_files += 1
                continue

            scenario = row.get("scenario", "unknown")
            t_tag = row.get("t", None) or f"t{int(row.get('time_index','0')):04d}"
            time_index = row.get("time_index", "")
            pr = row.get("patch_row", "")
            pc = row.get("patch_col", "")

            used_rows += 1

            st, vals = compute_patch_stats_masked(val_path, mask_path, mmap=mmap)

            # patch-level output
            pw.writerow([
                scenario, t_tag, time_index, pr, pc,
                grid, val_path, mask_path,
                st["n"], st["min"], st["max"], st["mean"], st["std"]
            ])

            # aggregated updates
            agg_dataset.update_from_values(vals)
            agg_by_t[(scenario, t_tag)].update_from_values(vals)
            agg_by_scenario[scenario].update_from_values(vals)
            scenario_t_set[scenario].add(t_tag)

    pf.close()

    # write t-level stats
    with open(t_stats_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "t", "grid", "num_patches", "n_masked", "min", "max", "mean", "std"])
        items = sorted(agg_by_t.items(), key=lambda kv: (kv[0][0], t_sort_key(kv[0][1])))
        for (scenario, t_tag), agg in items:
            w.writerow([
                scenario, t_tag, grid, agg.num_patches, agg.n,
                agg.min, agg.max, agg.mean(), agg.std()
            ])

    # write scenario-level stats
    with open(scenario_stats_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "grid", "num_t", "num_patches", "n_masked", "min", "max", "mean", "std"])
        for scenario in sorted(agg_by_scenario.keys()):
            agg = agg_by_scenario[scenario]
            w.writerow([
                scenario, grid, len(scenario_t_set[scenario]), agg.num_patches, agg.n,
                agg.min, agg.max, agg.mean(), agg.std()
            ])

    # write dataset-level stats
    with open(dataset_stats_txt, "w") as f:
        f.write(f"var={target_var}\n")
        f.write(f"grid={grid}\n")
        f.write("masked_by=mask==1 & isfinite\n")
        f.write(f"rows_total_in_index={total}\n")
        f.write(f"rows_used={used_rows}\n")
        f.write(f"missing_files_skipped={missing_files}\n")
        f.write(f"num_patches={agg_dataset.num_patches}\n")
        f.write(f"n_masked={agg_dataset.n}\n")
        f.write(f"min={agg_dataset.min}\n")
        f.write(f"max={agg_dataset.max}\n")
        f.write(f"mean={agg_dataset.mean()}\n")
        f.write(f"std={agg_dataset.std()}\n")

    print("✅ Done.")
    print(f"Patch stats   -> {patch_stats_csv}")
    print(f"t stats       -> {t_stats_csv}")
    print(f"Scenario stats-> {scenario_stats_csv}")
    print(f"Dataset stats -> {dataset_stats_txt}")


def parse_args():
    ap = argparse.ArgumentParser("Analyze flood-map patch dataset stats (masked by AOI mask)")
    ap.add_argument("--index-csv", required=True, help="Path to dataset/index.csv produced by make_patches.py")
    ap.add_argument("--out-dir", required=True, help="Output directory for stats CSVs")
    ap.add_argument("--var", default="h", choices=["h", "zs", "u", "v"], help="Which variable to analyze (default: h)")
    ap.add_argument("--grid", default="coarse", choices=["coarse", "fine"], help="Analyze coarse or fine patches")
    ap.add_argument("--max-rows", type=int, default=0, help="Debug: only process first N rows (0 = all)")
    ap.add_argument("--mmap", action="store_true", help="Use mmap to reduce memory spikes when loading npy")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(
        index_csv=args.index_csv,
        out_dir=args.out_dir,
        target_var=args.var,
        grid=args.grid,
        max_rows=args.max_rows,
        mmap=args.mmap
    )
