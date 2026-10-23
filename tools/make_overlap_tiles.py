#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_overlap_tiles.py -- Paper 2 M4: overlapping inference tiles built from an EXISTING test set.

The existing test-set patches (make_patches.py, stride = patch, filter 1e-6) tile the fine domain without
gaps; a patch is missing only if it holds NO AOI cell, and such a patch would have contained only zeros
(statics are zeroed outside the AOI, coarse inputs outside the coarse AOI, mask = 0). So the full-domain
model inputs are rebuilt exactly by pasting the patches back (missing ones = 0; that this is exact for the
COARSE inputs is proven on NeSI by tools/check_aoi_slivers.py -> --aoi-report, review A3). Then:
  1. ZERO-pad the inputs by --pad fine cells (--pad/scale coarse cells), AOI masks padded with 0 -- i.e.
     the pad ring looks exactly like any out-of-AOI region of the training data ("zero-padded outside the
     AOI"; review B1: the earlier mirror pad was wiped by step 2 anyway, so the numbers are unchanged);
  2. the make_patches masking rule is applied again: fine statics -> 0 where mask_fine == 0, coarse
     h / zs / dem -> 0 where mask_coarse == 0 (a no-op inside the domain and on the zero ring);
  3. tiles of --patch fine cells are cut every --stride cells over the padded domain; a tile is kept if it
     holds at least one AOI cell (as the test set's 1e-6 filter).
Output = a normal dataset (index.csv + .npy) read by PairedFloodMapDataset; fine_path is empty
(inference only; the ground truth stays in the original test set). Static tiles are time-independent
and identical for every test set -> written once to --static-dir (guarded by a content hash). If a test
set's statics differ (hash mismatch, e.g. a different NaN pattern in its flood output entered joint_valid),
the differing channels are printed and that set gets its own static dir <static-dir>__<out-dir name> (C3).

Checks (abort on failure):
  A. coarse alignment: on tiles fully inside the AOI, the 8x8 block mean of the fine DEM patch must equal
     the stored coarse DEM patch (offset 0, tolerance 1e-3 m) -> the pasted coarse windows are contiguous.
  B. AOI slivers: (i) --aoi-report (tools/check_aoi_slivers.py on NeSI, full-domain masks) must say PASS for
     this scale/patch and list exactly this index's kept tiles -> no absent tile can hold a coarse AOI cell;
     (ii) over the stored tiles, coarse cells with mask_coarse = 1 whose fine block has no AOI cell must be 0.
     Failure -> rebuild the test set unfiltered with make_patches and rerun.
  C. the padded ring of every input is exactly 0 after step 3.
  D. tiles that coincide with an original patch (origin = pad + k*patch) reproduce it bit for bit.

Usage:
  python tools/make_overlap_tiles.py --test-index .../testdataset_100y42h0c/index_niwa.csv \
      --out-dir .../overlap_tiles/test100y42h0c --static-dir .../overlap_tiles/static_p512_pad256_s256
"""

import os
import csv
import json
import hashlib
import argparse
from collections import defaultdict

import numpy as np

STATIC_FINE = {   # channel -> (index column, sub-dir, prefix)
    "elev":   ("elev_fine_path",  "elevation/fine", "elevation"),
    "rough":  ("rough_path",      "roughness",      "roughness"),
    "slope":  ("slope_path",      "slope",          "slope"),
    "twi":    ("twi_path",        "twi",            "twi"),
    "asin":   ("aspect_sin_path", "aspect_sin",     "aspect_sin"),
    "acos":   ("aspect_cos_path", "aspect_cos",     "aspect_cos"),
}
INDEX_HEADER = [
    "scenario", "var", "t", "time_index", "patch_row", "patch_col",
    "downscale", "patch_size_coarse", "patch_size_fine",
    "x0_fine", "y0_fine", "x1_fine", "y1_fine", "src_fine_file",
    "coarse_path", "fine_path", "zs_coarse_path", "zs_fine_path",
    "elev_fine_path", "elev_coarse_path", "rough_path",
    "mask_fine_path", "mask_coarse_path",
    "slope_path", "twi_path", "aspect_sin_path", "aspect_cos_path",
    "aoi_ratio", "filtered_out",
    "pad_fine", "stride_fine", "tile_y0_padded", "tile_x0_padded",
]


def load(path, dtype=None):
    a = np.load(path)
    if a.ndim != 2:
        raise RuntimeError(f"{path}: expected 2D, got {a.shape}")
    return a if dtype is None else a.astype(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-index", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--static-dir", required=True)
    ap.add_argument("--pad", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--var", default="h")
    ap.add_argument("--aoi-report", required=True, help="tools/check_aoi_slivers.py JSON (NeSI) for this scale/patch")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    with open(a.test_index, newline="") as f:
        rows = list(csv.DictReader(f))
    allv = [r for r in rows if r["var"] == a.var]
    kept = [r for r in allv if r["filtered_out"].strip() == "0"]
    if not kept:
        raise SystemExit("[fatal] no kept rows")
    P = {int(r["patch_size_fine"]) for r in kept}
    Pc = {int(r["patch_size_coarse"]) for r in kept}
    S = {int(r["downscale"]) for r in kept}
    if len(P) != 1 or len(Pc) != 1 or len(S) != 1:
        raise SystemExit(f"[fatal] mixed patch sizes {P} {Pc} {S}")
    P, Pc, S = P.pop(), Pc.pop(), S.pop()
    if P != Pc * S or a.pad % S or a.stride % S:
        raise SystemExit(f"[fatal] inconsistent sizes P={P} Pc={Pc} s={S} pad={a.pad} stride={a.stride}")
    if not (a.pad == a.stride == P // 2):
        raise SystemExit(f"[fatal] the M4 merges assume pad = stride = patch/2 (= {P // 2}); got pad={a.pad} stride={a.stride}")
    nr = max(int(r["patch_row"]) for r in allv) + 1
    nc = max(int(r["patch_col"]) for r in allv) + 1
    N_y, N_x = nr * P, nc * P
    Nc_y, Nc_x = nr * Pc, nc * Pc
    pad, pc = a.pad, a.pad // S
    Npy, Npx = N_y + 2 * pad, N_x + 2 * pad
    if (Npy - P) % a.stride or (Npx - P) % a.stride:
        raise SystemExit(f"[fatal] padded domain {Npy}x{Npx} not tiled by patch {P} / stride {a.stride}")
    print(f"[info] {a.test_index}: tiles {nr}x{nc} (kept {len({(r['patch_row'], r['patch_col']) for r in kept})}), "
          f"P={P} Pc={Pc} s={S}; domain {N_y}x{N_x} -> padded {Npy}x{Npx}")

    # ---------------- static canvases (time-invariant), from any row of each tile ----------------
    tile_row = {}
    for r in kept:
        tile_row.setdefault((int(r["patch_row"]), int(r["patch_col"])), r)
    fine = {k: np.zeros((N_y, N_x), np.float32) for k in STATIC_FINE}
    mfine = np.zeros((N_y, N_x), np.uint8)
    cdem = np.zeros((Nc_y, Nc_x), np.float32)
    mcoarse = np.zeros((Nc_y, Nc_x), np.uint8)
    for (ri, ci), r in tile_row.items():
        ys, xs = slice(ri * P, ri * P + P), slice(ci * P, ci * P + P)
        ycs, xcs = slice(ri * Pc, ri * Pc + Pc), slice(ci * Pc, ci * Pc + Pc)
        for k, (col, _, _) in STATIC_FINE.items():
            fine[k][ys, xs] = load(r[col], np.float32)
        mfine[ys, xs] = load(r["mask_fine_path"], np.uint8)
        cdem[ycs, xcs] = load(r["elev_coarse_path"], np.float32)
        mcoarse[ycs, xcs] = load(r["mask_coarse_path"], np.uint8)

    # ---------------- check A: coarse alignment ----------------
    errs = {}
    full = [(ri, ci) for (ri, ci) in tile_row
            if mfine[ri * P:ri * P + P, ci * P:ci * P + P].all() and mcoarse[ri * Pc:ri * Pc + Pc, ci * Pc:ci * Pc + Pc].all()
            and (fine["elev"][ri * P:ri * P + P, ci * P:ci * P + P] != 0).all()]
    if len(full) < 3:
        raise SystemExit(f"[fatal] check A: only {len(full)} tiles fully inside the AOI -> cannot verify alignment")
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            e = 0.0
            for (ri, ci) in full:
                bm = fine["elev"][ri * P:ri * P + P, ci * P:ci * P + P].astype(np.float64).reshape(Pc, S, Pc, S).mean(axis=(1, 3))
                y0, x0 = ri * Pc + dy, ci * Pc + dx
                if y0 < 0 or x0 < 0 or y0 + Pc > Nc_y or x0 + Pc > Nc_x:
                    e = np.inf
                    break
                e = max(e, float(np.max(np.abs(bm - cdem[y0:y0 + Pc, x0:x0 + Pc]))))
            errs[(dy, dx)] = e
    best = min(errs, key=errs.get)
    print(f"[check A] coarse alignment on {len(full)} interior tiles: max|blockmean(fine dem) - coarse dem| "
          f"offset(0,0)={errs[(0, 0)]:.2e} m; best offset {best} ({errs[best]:.2e})")
    if best != (0, 0) or errs[(0, 0)] > 1e-3:
        raise SystemExit("[fatal] check A failed: coarse windows are not aligned with the fine tiles")

    # ---------------- check B: AOI slivers ----------------
    with open(a.aoi_report) as f:
        rep = json.load(f)
    rep_tiles = {tuple(t) for t in rep["nonempty_tiles"]}
    if not rep.get("pass") or rep["scale"] != S or rep["patch_fine"] != P or rep["tiles"] != [nr, nc]:
        raise SystemExit(f"[fatal] check B(i): {a.aoi_report} is not a PASS report for s={S}, P={P}, tiles {nr}x{nc}")
    if rep_tiles != set(tile_row):
        raise SystemExit(f"[fatal] check B(i): kept tiles of this index != tiles with a fine AOI cell in the report "
                         f"({len(set(tile_row))} vs {len(rep_tiles)})")
    print(f"[check B] (i) full-domain report PASS: {nr * nc - len(rep_tiles)} absent tiles hold no coarse AOI cell; "
          f"kept tiles == report ({len(rep_tiles)})")
    fine_any = mfine.reshape(Nc_y, S, Nc_x, S).max(axis=(1, 3))
    slivers = int(((mcoarse == 1) & (fine_any == 0)).sum())
    print(f"[check B] (ii) stored tiles: coarse AOI cells without any fine AOI cell (slivers): {slivers}")
    if slivers:
        # Not fatal (fourth review, 2026-10-16): a sliver inside a KEPT tile is harmless here -- that tile's stored coarse
        # patch carries the cell's value, so the pasted canvas is complete. The harmful case, a sliver inside an ABSENT
        # tile, is excluded by check B(i) above (fatal). Kept as a warning so the count is visible in the log.
        print(f"[warn] check B(ii): {slivers} coarse AOI cell(s) in kept tiles have no fine AOI cell in their block "
              f"(sub-cell AOI slivers); their coarse values come from the stored tiles -- no action needed")

    # ---------------- zero pad (masks with 0), re-mask ----------------
    def pad_f(x):
        return np.pad(x, pad, mode="constant", constant_values=0)

    def pad_c(x):
        return np.pad(x, pc, mode="constant", constant_values=0)

    mfine_p = np.pad(mfine, pad, mode="constant", constant_values=0)
    mcoarse_p = np.pad(mcoarse, pc, mode="constant", constant_values=0)
    fine_p = {k: np.where(mfine_p == 1, pad_f(v), 0).astype(np.float32) for k, v in fine.items()}
    cdem_p = np.where(mcoarse_p == 1, pad_c(cdem), 0).astype(np.float32)
    ring_f = np.ones((Npy, Npx), bool); ring_f[pad:pad + N_y, pad:pad + N_x] = False
    ring_c = np.ones(mcoarse_p.shape, bool); ring_c[pc:pc + Nc_y, pc:pc + Nc_x] = False
    for k, v in fine_p.items():                                     # inside the domain: no-op re-mask
        if not np.array_equal(v[pad:pad + N_y, pad:pad + N_x], fine[k]):
            raise SystemExit(f"[fatal] re-masking changed in-domain static {k} (stored patches not AOI-masked?)")
    bad = [k for k, v in fine_p.items() if np.any(v[ring_f] != 0)] + (["cdem"] if np.any(cdem_p[ring_c] != 0) else [])
    if bad or np.any(mfine_p[ring_f]) or np.any(mcoarse_p[ring_c]):
        raise SystemExit(f"[fatal] check C failed: non-zero pad ring in {bad}")
    print("[check C] padded ring of all static inputs and masks is exactly 0")

    # tiles
    nty, ntx = (Npy - P) // a.stride + 1, (Npx - P) // a.stride + 1
    tiles = []
    for i in range(nty):
        for j in range(ntx):
            y0, x0 = i * a.stride, j * a.stride
            n_aoi = int(mfine_p[y0:y0 + P, x0:x0 + P].sum())
            if n_aoi > 0:
                tiles.append((i, j, y0, x0, n_aoi / float(P * P)))
    print(f"[info] tiles {nty}x{ntx} = {nty * ntx}; kept (any AOI cell) {len(tiles)}")

    # ---------------- static tiles (shared across test sets) ----------------
    chan_arrays = {**fine_p, "mask_fine": mfine_p, "elev_coarse": cdem_p, "mask_coarse": mcoarse_p}
    chan_digest = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in sorted(chan_arrays.items())}
    h = hashlib.sha256()
    for k in sorted(fine_p):
        h.update(fine_p[k].tobytes())
    h.update(mfine_p.tobytes()); h.update(cdem_p.tobytes()); h.update(mcoarse_p.tobytes())
    h.update(json.dumps([P, Pc, S, pad, a.stride]).encode())
    digest = h.hexdigest()
    man = os.path.join(a.static_dir, "static_manifest.json")
    if os.path.isfile(man):
        with open(man) as f:
            m = json.load(f)
        if m["sha256"] != digest:                                   # C3: fall back to a per-set static dir
            old = m.get("channel_sha256", {})
            diff = [k for k in chan_digest if old.get(k) != chan_digest[k]] if old else ["(unknown: old manifest)"]
            own = os.path.normpath(a.static_dir) + "__" + os.path.basename(os.path.normpath(a.out_dir))
            print(f"[warn] statics differ from {a.static_dir} in channels {diff} -> this test set uses its own "
                  f"static dir {own}")
            a.static_dir = own
            man = os.path.join(a.static_dir, "static_manifest.json")
    spath = {}
    for (i, j, y0, x0, _) in tiles:
        tag = f"r{i:03d}_c{j:03d}"
        spath[(i, j)] = {**{k: os.path.join(a.static_dir, sub, f"{pre}_{tag}.npy") for k, (_, sub, pre) in STATIC_FINE.items()},
                         "mask_fine": os.path.join(a.static_dir, "mask/fine", f"mask_fine_{tag}.npy"),
                         "elev_coarse": os.path.join(a.static_dir, "elevation/coarse", f"elevation_{tag}.npy"),
                         "mask_coarse": os.path.join(a.static_dir, "mask/coarse", f"mask_coarse_{tag}.npy")}
    if os.path.isfile(man):
        with open(man) as f:
            m = json.load(f)
        if m["sha256"] != digest:
            raise SystemExit(f"[fatal] {a.static_dir} holds statics of a DIFFERENT grid/test set (hash mismatch)")
        print(f"[info] static tiles already in {a.static_dir} (hash match) -> reused")
        missing_static = [t for t in tiles if not os.path.isfile(spath[(t[0], t[1])]["mask_fine"])]
        if missing_static:
            raise SystemExit(f"[fatal] static dir incomplete ({len(missing_static)} tiles missing)")
    else:
        for (i, j, y0, x0, _) in tiles:
            sp = spath[(i, j)]
            yc0, xc0 = y0 // S, x0 // S
            for k in STATIC_FINE:
                os.makedirs(os.path.dirname(sp[k]), exist_ok=True)
                np.save(sp[k], np.ascontiguousarray(fine_p[k][y0:y0 + P, x0:x0 + P]))
            for k, arr, sl in (("mask_fine", mfine_p, (slice(y0, y0 + P), slice(x0, x0 + P))),
                               ("elev_coarse", cdem_p, (slice(yc0, yc0 + Pc), slice(xc0, xc0 + Pc))),
                               ("mask_coarse", mcoarse_p, (slice(yc0, yc0 + Pc), slice(xc0, xc0 + Pc)))):
                os.makedirs(os.path.dirname(sp[k]), exist_ok=True)
                np.save(sp[k], np.ascontiguousarray(arr[sl]))
        with open(man, "w") as f:
            json.dump({"sha256": digest, "channel_sha256": chan_digest, "patch": P, "patch_coarse": Pc, "scale": S, "pad": pad,
                       "stride": a.stride, "n_tiles": len(tiles), "source_index": a.test_index}, f, indent=2)
        print(f"[info] wrote {len(tiles)} static tiles -> {a.static_dir}")

    # ---------------- check D (static): tiles on the original grid reproduce the stored patches ----------------
    def orig_tile(y0, x0):
        q, rem_y = divmod(y0 - pad, P)
        qx, rem_x = divmod(x0 - pad, P)
        return (q, qx) if rem_y == 0 and rem_x == 0 else None

    n_cmp = 0
    for (i, j, y0, x0, _) in tiles:
        o = orig_tile(y0, x0)
        if o is None or o not in tile_row:
            continue
        r = tile_row[o]
        for k, (col, _, _) in STATIC_FINE.items():
            if not np.array_equal(np.load(spath[(i, j)][k]), np.load(r[col]).astype(np.float32)):
                raise SystemExit(f"[fatal] check D: static {k} of tile {(i, j)} != original patch {o}")
        for k, col in (("mask_fine", "mask_fine_path"), ("elev_coarse", "elev_coarse_path"), ("mask_coarse", "mask_coarse_path")):
            if not np.array_equal(np.load(spath[(i, j)][k]), np.load(r[col])):
                raise SystemExit(f"[fatal] check D: {k} of tile {(i, j)} != original patch {o}")
        n_cmp += 1
    print(f"[check D] {n_cmp} on-grid tiles: all static channels identical to the original patches")

    # ---------------- dynamic coarse inputs per time step ----------------
    if os.path.exists(os.path.join(a.out_dir, "index.csv")) and not a.overwrite:
        raise SystemExit(f"[error] {a.out_dir}/index.csv exists -> --overwrite")
    by_t = defaultdict(dict)
    meta_t = {}
    for r in kept:
        by_t[(r["scenario"], r["t"])][(int(r["patch_row"]), int(r["patch_col"]))] = r
        meta_t[(r["scenario"], r["t"])] = r["time_index"]
    idx_path = os.path.join(a.out_dir, "index.csv")
    os.makedirs(a.out_dir, exist_ok=True)
    n_rows = 0
    with open(idx_path, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(INDEX_HEADER)
        for (scen, t) in sorted(by_t):
            tiles_t = by_t[(scen, t)]
            if set(tiles_t) != set(tile_row):
                raise SystemExit(f"[fatal] {scen} {t}: tile set differs from the static tile set")
            ch = np.zeros((Nc_y, Nc_x), np.float32)
            cz = np.zeros((Nc_y, Nc_x), np.float32)
            for (ri, ci), r in tiles_t.items():
                ycs, xcs = slice(ri * Pc, ri * Pc + Pc), slice(ci * Pc, ci * Pc + Pc)
                ch[ycs, xcs] = load(r["coarse_path"], np.float32)
                cz[ycs, xcs] = load(r["zs_coarse_path"], np.float32)
            ch_p = np.where(mcoarse_p == 1, pad_c(ch), 0).astype(np.float32)
            cz_p = np.where(mcoarse_p == 1, pad_c(cz), 0).astype(np.float32)
            if np.any(ch_p[ring_c] != 0) or np.any(cz_p[ring_c] != 0):
                raise SystemExit(f"[fatal] check C failed: coarse h/zs pad ring non-zero at {scen} {t}")
            if not (np.array_equal(ch_p[pc:pc + Nc_y, pc:pc + Nc_x], ch) and np.array_equal(cz_p[pc:pc + Nc_y, pc:pc + Nc_x], cz)):
                raise SystemExit(f"[fatal] re-masking changed in-domain coarse h/zs at {scen} {t}")
            d_h = os.path.join(a.out_dir, a.var, scen, "coarse")
            d_z = os.path.join(a.out_dir, "zs", "coarse")
            os.makedirs(d_h, exist_ok=True); os.makedirs(d_z, exist_ok=True)
            for (i, j, y0, x0, ratio) in tiles:
                yc0, xc0 = y0 // S, x0 // S
                core = f"{a.var}_{scen}_{t}_r{i:03d}_c{j:03d}_s{S}"
                zcore = f"zs_{scen}_{t}_r{i:03d}_c{j:03d}_s{S}"
                p_h, p_z = os.path.join(d_h, f"{core}_coarse.npy"), os.path.join(d_z, f"{zcore}_coarse.npy")
                blk_h = np.ascontiguousarray(ch_p[yc0:yc0 + Pc, xc0:xc0 + Pc])
                blk_z = np.ascontiguousarray(cz_p[yc0:yc0 + Pc, xc0:xc0 + Pc])
                o = orig_tile(y0, x0)                                  # check D (dynamic)
                if o is not None and o in tiles_t:
                    if not (np.array_equal(blk_h, load(tiles_t[o]["coarse_path"])) and
                            np.array_equal(blk_z, load(tiles_t[o]["zs_coarse_path"]))):
                        raise SystemExit(f"[fatal] check D: coarse h/zs of tile {(i, j)} != original {o} at {t}")
                np.save(p_h, blk_h)
                np.save(p_z, blk_z)
                sp = spath[(i, j)]
                w.writerow([scen, a.var, t, meta_t[(scen, t)], i, j, S, Pc, P,
                            "", "", "", "", "",
                            p_h, "", p_z, "",
                            sp["elev"], sp["elev_coarse"], sp["rough"], sp["mask_fine"], sp["mask_coarse"],
                            sp["slope"], sp["twi"], sp["asin"], sp["acos"],
                            f"{ratio:.6f}", 0, pad, a.stride, y0, x0])
                n_rows += 1
    with open(os.path.join(a.out_dir, "overlap_manifest.json"), "w") as f:
        json.dump({"source_index": a.test_index, "patch": P, "patch_coarse": Pc, "scale": S, "pad": pad,
                   "stride": a.stride, "domain": [N_y, N_x], "orig_tiles": [nr, nc],
                   "n_tiles": len(tiles), "n_timesteps": len(by_t), "n_rows": n_rows,
                   "n_orig_patches": len(kept), "static_dir": a.static_dir, "static_sha256": digest,
                   "padding": "zero (AOI mask 0 in the pad ring)", "aoi_report": a.aoi_report,
                   "checks": {"A_alignment_err_m": errs[(0, 0)], "B_slivers": slivers, "B_report_pass": True,
                              "C_pad_ring_zero": True,
                              "D_on_grid_tiles_compared": n_cmp}}, f, indent=2)
    print(f"[done] {n_rows} rows ({len(tiles)} tiles x {len(by_t)} steps; original {len(kept)} patches) -> {idx_path}")


if __name__ == "__main__":
    main()
