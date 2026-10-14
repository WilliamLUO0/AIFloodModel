#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_split_stats.py -- Paper 2 M3 Row B, review item A5 (post-run check, costs nothing).

A Row B dataset copies the ds8 train/val split tile for tile and cuts the SAME fine-grid patches, so every
fine-grid quantity of its split_stats JSON must equal the ds8 JSON: static channels (stats.*), the asinh
scale candidates, asinh_by_q.*.{s,fine,shared}, wet ratio / pos_weight, flood-interval counts and zs
fine/shared. Only quantities of the COARSE grid ('coarse' in the key path) may differ, plus bookkeeping
(split row ids, which index into a different CSV; provenance paths; notes). Any other difference means the
tile set or the recipe is not the same -> exit 1.

Usage: python tools/compare_split_stats.py --ref ds8/split_stats_h_asinh.json --new dsX_pf512/split_stats_h_asinh.json
"""

import sys
import json
import math
import argparse

SKIP_KEYS = {"split", "split_reused_from", "split_reused_index_csv", "note", "stats_note", "percentile_note",
             "pos_weight_def"}


def walk(a, b, path, out, rtol):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP_KEYS or k == "coarse":
                continue
            if k not in a or k not in b:
                out.append((f"{path}/{k}", "missing in " + ("ref" if k not in a else "new"), ""))
                continue
            walk(a[k], b[k], f"{path}/{k}", out, rtol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (a == b or (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= rtol * max(abs(a), abs(b)))):
            out.append((path, a, b))
    elif a != b:
        out.append((path, a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--rtol", type=float, default=1e-9, help="relative tolerance (summation order differs)")
    a = ap.parse_args()
    ref, new = json.load(open(a.ref)), json.load(open(a.new))
    out = []
    walk(ref, new, "", out, a.rtol)
    n_coarse = 0

    def count_coarse(x):
        nonlocal n_coarse
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "coarse":
                    n_coarse += 1
                else:
                    count_coarse(v)
    count_coarse(new)
    if out:
        print(f"[FAIL] {len(out)} fine-grid / recipe differences between {a.ref} and {a.new}:")
        for p, x, y in out[:40]:
            print(f"   {p}: ref={x!r}  new={y!r}")
        sys.exit(1)
    print(f"[PASS] every fine-grid quantity and the recipe equal the reference (rtol {a.rtol}); "
          f"{n_coarse} 'coarse' blocks allowed to differ")


if __name__ == "__main__":
    main()
