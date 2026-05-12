import os
import csv
import json
import argparse
import numpy as np
from tqdm import tqdm


def resolve_path(path, root):
    if path is None or path == "":
        return ""
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def safe_percentiles(values, percentiles=(50, 75, 90, 95, 99)):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {f"p{p}": None for p in percentiles}

    return {f"p{p}": float(np.percentile(values, p)) for p in percentiles}


def count_thresholds(values, name, thresholds=(0.0, 0.001, 0.005, 0.01)):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    out = {}

    for thr in thresholds:
        if thr == 0.0:
            key = f"{name}_gt_0"
            out[key] = int(np.sum(values > 0.0))
        else:
            key = f"{name}_ge_{thr:g}"
            out[key] = int(np.sum(values >= thr))

    return out


def compute_interval_stats(
    fine_path,
    mask_path,
    h_slight=0.1,
    h_severe=0.5,
    h_extreme=1.0,
):
    h = np.load(fine_path).astype(np.float32)
    mask = np.load(mask_path).astype(bool)

    if h.shape != mask.shape:
        raise RuntimeError(
            f"Shape mismatch: h={h.shape}, mask={mask.shape}, "
            f"fine_path={fine_path}, mask_path={mask_path}"
        )

    # Only AOI, finite, non-negative depth cells are counted as valid.
    valid = mask & np.isfinite(h) & (h >= 0.0)

    valid_count = int(np.count_nonzero(valid))

    if valid_count == 0:
        return {
            "valid_count": 0,
            "nonflood_count": 0,
            "slight_count": 0,
            "severe_count": 0,
            "extreme_count": 0,
            "wet_count": 0,
            "nonflood_ratio": 0.0,
            "slight_ratio": 0.0,
            "severe_ratio": 0.0,
            "extreme_ratio": 0.0,
            "wet_ratio": 0.0,
        }

    # Left-closed and right-open intervals:
    # nonflood: 0.0 <= h < 0.1
    # slight:   0.1 <= h < 0.5
    # severe:   0.5 <= h < 1.0
    # extreme:  h >= 1.0
    nonflood = valid & (h < h_slight)
    slight = valid & (h >= h_slight) & (h < h_severe)
    severe = valid & (h >= h_severe) & (h < h_extreme)
    extreme = valid & (h >= h_extreme)

    nonflood_count = int(np.count_nonzero(nonflood))
    slight_count = int(np.count_nonzero(slight))
    severe_count = int(np.count_nonzero(severe))
    extreme_count = int(np.count_nonzero(extreme))
    wet_count = slight_count + severe_count + extreme_count

    denom = float(valid_count)

    return {
        "valid_count": valid_count,
        "nonflood_count": nonflood_count,
        "slight_count": slight_count,
        "severe_count": severe_count,
        "extreme_count": extreme_count,
        "wet_count": wet_count,
        "nonflood_ratio": nonflood_count / denom,
        "slight_ratio": slight_count / denom,
        "severe_ratio": severe_count / denom,
        "extreme_ratio": extreme_count / denom,
        "wet_ratio": wet_count / denom,
    }


def load_summary_ids(split_stats_json, summary_split):
    if split_stats_json is None or str(split_stats_json).strip() == "":
        return None, {
            "summary_source": "all_target_patches",
            "split_stats_json": None,
            "summary_split": "all",
        }

    if not os.path.isfile(split_stats_json):
        raise FileNotFoundError(f"split_stats_json not found: {split_stats_json}")

    with open(split_stats_json, "r") as f:
        meta = json.load(f)

    if "split" not in meta:
        raise RuntimeError(f"Invalid split_stats_json: missing key 'split': {split_stats_json}")

    if summary_split == "all":
        return None, {
            "summary_source": "all_target_patches",
            "split_stats_json": os.path.abspath(split_stats_json),
            "summary_split": "all",
        }

    if summary_split not in meta["split"]:
        raise RuntimeError(
            f"split_stats_json does not contain split['{summary_split}']. "
            f"Available keys: {list(meta['split'].keys())}"
        )

    ids = set(int(x) for x in meta["split"][summary_split])

    return ids, {
        "summary_source": f"{summary_split}_split_only",
        "split_stats_json": os.path.abspath(split_stats_json),
        "summary_split": summary_split,
        "num_ids_in_split_stats": len(ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--root", default="")
    parser.add_argument("--target-var", default="h", choices=["h"])

    parser.add_argument("--h-slight", type=float, default=0.1)
    parser.add_argument("--h-severe", type=float, default=0.5)
    parser.add_argument("--h-extreme", type=float, default=1.0)

    # New: use the same split as training/validation.
    parser.add_argument(
        "--split-stats-json",
        default="",
        help="Existing split_stats_*.json. If provided, summary percentiles are computed only on the selected split."
    )
    parser.add_argument(
        "--summary-split",
        default="train",
        choices=["train", "val", "all"],
        help="Which split to use for JSON summary statistics. Usually train."
    )

    args = parser.parse_args()

    root = args.root
    if root == "":
        root = os.path.dirname(os.path.abspath(args.index_csv))

    with open(args.index_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    # Important: reproduce the row ids used by load_index_csv().
    # These ids must match the ids stored in split_stats_json["split"].
    for i, row in enumerate(rows):
        if "_row_id" not in row:
            row["_row_id"] = str(i)

    summary_ids, summary_meta = load_summary_ids(
        args.split_stats_json,
        args.summary_split
    )

    out_rows = []

    # These arrays are for the JSON summary only.
    # If split-stats-json is provided and summary-split=train, they only collect train rows.
    nonflood_ratios = []
    slight_ratios = []
    severe_ratios = []
    extreme_ratios = []
    wet_ratios = []

    total_valid_count = 0
    total_nonflood_count = 0
    total_slight_count = 0
    total_severe_count = 0
    total_extreme_count = 0
    total_wet_count = 0

    num_target_patches_all = 0
    num_target_patches_summary = 0

    for row in tqdm(rows, desc="Computing patch interval stats"):
        is_target = row.get("var", "") == args.target_var
        is_filtered = int(row.get("filtered_out", 0)) == 1

        if (not is_target) or is_filtered:
            empty_stats = {
                "valid_count": "",
                "nonflood_count": "",
                "slight_count": "",
                "severe_count": "",
                "extreme_count": "",
                "wet_count": "",
                "nonflood_ratio": "",
                "slight_ratio": "",
                "severe_ratio": "",
                "extreme_ratio": "",
                "wet_ratio": "",
            }
            row.update(empty_stats)
            out_rows.append(row)
            continue

        row_id = int(row["_row_id"])

        fine_path = resolve_path(row["fine_path"], root)
        mask_path = resolve_path(row["mask_fine_path"], root)

        if fine_path == "" or mask_path == "":
            raise RuntimeError(f"Missing fine_path or mask_fine_path in row: {row}")

        stats = compute_interval_stats(
            fine_path=fine_path,
            mask_path=mask_path,
            h_slight=args.h_slight,
            h_severe=args.h_severe,
            h_extreme=args.h_extreme,
        )

        # Always write patch-level stats to the output CSV for all valid target patches.
        row.update({k: str(v) for k, v in stats.items()})
        out_rows.append(row)

        num_target_patches_all += 1

        # But only use selected split rows for JSON summary statistics.
        use_for_summary = (summary_ids is None) or (row_id in summary_ids)

        if not use_for_summary:
            continue

        num_target_patches_summary += 1

        nonflood_ratios.append(stats["nonflood_ratio"])
        slight_ratios.append(stats["slight_ratio"])
        severe_ratios.append(stats["severe_ratio"])
        extreme_ratios.append(stats["extreme_ratio"])
        wet_ratios.append(stats["wet_ratio"])

        total_valid_count += stats["valid_count"]
        total_nonflood_count += stats["nonflood_count"]
        total_slight_count += stats["slight_count"]
        total_severe_count += stats["severe_count"]
        total_extreme_count += stats["extreme_count"]
        total_wet_count += stats["wet_count"]

    nonflood_ratios = np.asarray(nonflood_ratios, dtype=np.float64)
    slight_ratios = np.asarray(slight_ratios, dtype=np.float64)
    severe_ratios = np.asarray(severe_ratios, dtype=np.float64)
    extreme_ratios = np.asarray(extreme_ratios, dtype=np.float64)
    wet_ratios = np.asarray(wet_ratios, dtype=np.float64)

    nonflood_positive = nonflood_ratios[nonflood_ratios > 0]
    slight_positive = slight_ratios[slight_ratios > 0]
    severe_positive = severe_ratios[severe_ratios > 0]
    extreme_positive = extreme_ratios[extreme_ratios > 0]
    wet_positive = wet_ratios[wet_ratios > 0]

    new_fields = [
        "valid_count",
        "nonflood_count",
        "slight_count",
        "severe_count",
        "extreme_count",
        "wet_count",
        "nonflood_ratio",
        "slight_ratio",
        "severe_ratio",
        "extreme_ratio",
        "wet_ratio",
    ]

    # Do not write _row_id into the output CSV unless it already existed in the input.
    # This avoids changing the expected index schema.
    final_fieldnames = fieldnames + [x for x in new_fields if x not in fieldnames]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    denom = float(max(total_valid_count, 1))

    patch_counts_by_condition = {}
    patch_counts_by_condition.update(count_thresholds(slight_ratios, "slight_ratio"))
    patch_counts_by_condition.update(count_thresholds(severe_ratios, "severe_ratio"))
    patch_counts_by_condition.update(count_thresholds(extreme_ratios, "extreme_ratio"))
    patch_counts_by_condition.update(count_thresholds(wet_ratios, "wet_ratio"))

    summary = {
        "target_var": args.target_var,
        "index_csv": os.path.abspath(args.index_csv),
        "out_csv": os.path.abspath(args.out_csv),
        "summary_meta": summary_meta,
        "thresholds_m": {
            "nonflood": [0.0, args.h_slight],
            "slight": [args.h_slight, args.h_severe],
            "severe": [args.h_severe, args.h_extreme],
            "extreme": [args.h_extreme, None],
        },
        "interval_rule": (
            "left-closed, right-open except extreme: "
            "nonflood=[0,h_slight), slight=[h_slight,h_severe), "
            "severe=[h_severe,h_extreme), extreme=[h_extreme,+inf)"
        ),
        "num_rows_total": len(rows),
        "num_target_patches_all": int(num_target_patches_all),
        "num_target_patches_summary": int(num_target_patches_summary),
        "global_cell_counts": {
            "valid_count": int(total_valid_count),
            "nonflood_count": int(total_nonflood_count),
            "slight_count": int(total_slight_count),
            "severe_count": int(total_severe_count),
            "extreme_count": int(total_extreme_count),
            "wet_count": int(total_wet_count),
        },
        "global_cell_ratios": {
            "nonflood_ratio": total_nonflood_count / denom,
            "slight_ratio": total_slight_count / denom,
            "severe_ratio": total_severe_count / denom,
            "extreme_ratio": total_extreme_count / denom,
            "wet_ratio": total_wet_count / denom,
        },
        "patch_counts_by_condition": patch_counts_by_condition,
        "ratio_percentiles_all_patches": {
            "nonflood_ratio": safe_percentiles(nonflood_ratios),
            "slight_ratio": safe_percentiles(slight_ratios),
            "severe_ratio": safe_percentiles(severe_ratios),
            "extreme_ratio": safe_percentiles(extreme_ratios),
            "wet_ratio": safe_percentiles(wet_ratios),
        },
        "ratio_percentiles_positive_patches": {
            "nonflood_ratio": safe_percentiles(nonflood_positive),
            "slight_ratio": safe_percentiles(slight_positive),
            "severe_ratio": safe_percentiles(severe_positive),
            "extreme_ratio": safe_percentiles(extreme_positive),
            "wet_ratio": safe_percentiles(wet_positive),
        },
        "notes": {
            "csv_output": (
                "The output CSV preserves the original row order and adds patch-level interval stats "
                "for all valid target patches. Non-target or filtered rows receive empty interval fields."
            ),
            "summary_statistics": (
                "The JSON summary statistics are computed only on the selected summary split "
                "if --split-stats-json is provided. This is recommended for sampler reference values."
            ),
            "ratio_percentiles_all_patches": (
                "Percentiles computed over selected summary patches, including patches where the ratio is zero."
            ),
            "ratio_percentiles_positive_patches": (
                "Percentiles computed only over selected summary patches where the corresponding ratio is greater than zero."
            ),
            "recommended_use": (
                "Use train-split positive-patch percentiles, such as p50/p90, as candidate reference ratios "
                "for interval-balanced sampling."
            ),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved CSV:  {args.out_csv}")
    print(f"Saved JSON: {args.out_json}")
    print(f"Summary split: {summary_meta['summary_split']}")
    print(f"Target patches all: {num_target_patches_all}")
    print(f"Target patches used for summary: {num_target_patches_summary}")
    print("Done.")


if __name__ == "__main__":
    main()