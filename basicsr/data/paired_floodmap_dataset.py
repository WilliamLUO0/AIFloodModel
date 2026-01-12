"""
datasets:
  train:
    name: FloodSR-Train-h
    type: PairedFloodMapDataset
    dataroot: /.../dataset_patches
    index_csv: /.../dataset_patches/index.csv

    phase: train
    target_var: h
    scale: 16
    patch_fine: 1024
    patch_coarse: 64
    # wet_threshold: 0.05

    split_cfg:
      by: scenario
      val_ratio: 0.2
      seed: 2025
      split_stats_json: /.../dataset_patches/split_stats.json
    stats:
      calculate_if_missing: false
      use_clip_coarse: false
      use_clip_static: false
      norm: zscore  # or "minmax"

    use_hflip: true
    use_rot: true

    num_worker_per_gpu: 6
    batch_size_per_gpu: 1
    pin_memory: true
    persistent_workers: true

  val_1:
    name: FloodSR-Val-h
    type: PairedFloodMapDataset
    dataroot: /.../dataset_patches
    index_csv: /.../dataset_patches/index.csv

    phase: val
    target_var: h
    scale: 16
    patch_fine: 1024
    patch_coarse: 64
    # wet_threshold: 0.05

    split_cfg:
      by: scenario
      val_ratio: 0.2
      seed: 2025
      split_stats_json: /.../dataset_patches/split_stats.json
    stats:
      calculate_if_missing: false
      use_clip_coarse: false
      use_clip_static: false
      norm: zscore  # or "minmax"

    num_worker_per_gpu: 2
    batch_size_per_gpu: 1
    pin_memory: true
    persistent_workers: false

datasets:
  test_1:
    name: FloodSR-Test-h
    type: PairedFloodMapDataset
    dataroot: /.../realworld_patches
    index_csv: /.../realworld_patches/index.csv

    phase: test
    target_var: h
    scale: 16
    patch_fine: 1024
    patch_coarse: 64

    split_cfg:
      # 这里指向“训练阶段”保存的 split_stats.json（同一个工程/场景）
      split_stats_json: /.../dataset_patches/split_stats.json
      # 下面两个对 test 无实质影响，但可以保留字段完整性
      by: scenario
      val_ratio: 0.2
      seed: 2025
    stats:
      calculate_if_missing: false   # 防止在 test 上重算（避免数据泄露）
      use_clip_coarse: false
      use_clip_static: false
      norm: zscore  # or "minmax"

    # test 不做增强（即使写了 use_hflip/use_rot，也会被 phase=test 屏蔽）
    use_hflip: false
    use_rot: false

    num_worker_per_gpu: 2
    batch_size_per_gpu: 1
    pin_memory: true
    persistent_workers: false

"""

import os
import re
import json
import random
import numpy as np
from copy import deepcopy
import torch
from torch.utils import data as data

from basicsr.utils import (
    cal_log1p, cal_zscore, cal_minmaxnorm, cal_asinh_p90, group_by_scenario,
    percentile_clip, load_npy_shape, load_index_csv,
    cal_stats_on_train, check_required_fields
)
from basicsr.data.transforms import augment_flood_map
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedFloodMapDataset(data.Dataset):
    """
    Paired Flood map dataset (index.csv + .npy).

    Output:
     - input:
        Coarse-grid flood map: [1, patch_coarse, patch_coarse]
            h (water depth): log1p + clip[p1, p99] + z-score
            u (water velocity in x-direction): asinh(x/|x|_p90) + clip[p1, p99] + z-score
            v (water velocity in y-direction): asinh(x/|x|_p90) + clip[p1, p99] + z-score
        Fine-grid static features: [7, patch_fine, patch_fine]
            Elevation  : clip[p1, p99] + z-score
            Roughness  : clip[p1, p99] + z-score
            TWI        : clip[p1, p99] + z-score
            Slope_deg  : / 90
            Aspect_Sin : skip
            Aspect_COS : skip
            Mask       : skip
    - label:
        Fine-grid flood map (simulated / ground truth / target): [1, patch_fine, patch_fine]
    """

    def __init__(self, opt):
        super(PairedFloodMapDataset, self).__init__()
        self.opt = deepcopy(opt)
        self.phase = opt.get('phase', 'train')
        assert self.phase in ('train', 'val', 'test'), f'phase must be train/val/test, got {self.phase}'
        self.target_var = opt.get('target_var', 'h')
        assert self.target_var in ('h', 'u', 'v'), f"target_var should be one of ['h', 'u', 'v'], got {self.target_var}"

        self.scale = int(opt.get('scale', 16))
        self.patch_fine = int(opt.get('patch_fine', 1024))
        self.patch_coarse = int(opt.get('patch_coarse', 64))

        self.root = opt['dataroot']
        self.index_csv = opt.get('index_csv', os.path.join(self.root, 'index.csv'))
        if not os.path.isfile(self.index_csv):
            raise FileNotFoundError(f'[ERROR] index_csv not found: {self.index_csv}')

        self.use_hflip = bool(opt.get('use_hflip', False))
        self.use_rot = bool(opt.get('use_rot', False))

        split_cfg = opt.get('split_cfg', {}) or {}
        self.split_by = split_cfg.get('by', 'scenario')
        self.val_ratio = float(split_cfg.get('val_ratio', 0.2))
        self.seed = int(split_cfg.get('seed', 61))
        self.split_stats_json = split_cfg.get(
            'split_stats_json',
            os.path.join(self.root, 'dataset', f'split_stats_{self.target_var}.json')
        )

        stats_cfg = opt.get('stats', {}) or {}
        self.calculate_if_missing = bool(stats_cfg.get('calculate_if_missing', True))
        self.use_clip_coarse = bool(stats_cfg.get('use_clip_coarse', False))
        self.use_clip_static = bool(stats_cfg.get('use_clip_static', False))
        self.norm = str(stats_cfg.get('norm', 'zscore')).lower()
        assert self.norm in ('zscore', 'minmax'), f"[ERROR] stats.norm must be zscore/minmax, got {self.norm}"

        self.transform = str(stats_cfg.get('transform', 'log1p')).lower().strip()
        self.h_asinh_q = int(stats_cfg.get('h_asinh_q', 90))
        if self.target_var in ('u', 'v'):
            self.transform = 'asinh'
        else:
            assert self.transform in ('log1p', 'asinh'), f"[ERROR] stats.transform must be log1p/asinh for h, got {self.transform}"

        all_rows = load_index_csv(self.index_csv)
        rows = [r for r in all_rows if int(r['filtered_out']) == 0 and str(r.get('var', '')).lower() == self.target_var]

        self.meta = self._ensure_split_stats(rows, var=self.target_var)
        self.stats_static = self.meta['stats']
        self.stats_var = self.meta['stats_var']

        if self.phase in ('train', 'val'):
            ids = set(self.meta['split'][self.phase])
            self.rows = [r for r in rows if int(r['_row_id']) in ids]
        elif self.phase == 'test':
            self.rows = rows
        else:
            raise ValueError(f'[ERROR] Unknown phase: {self.phase}')

        _wet_th = opt.get('wet_threshold', None)
        self.wet_threshold = float(_wet_th) if _wet_th is not None else None

        check_required_fields(self.rows)

    def _ensure_split_stats(self, rows, var: str):
        """
        Read split_stats_json in .yml if existing,
        otherwise use split_cfg to split dataset and calculate stats
        """
        if os.path.isfile(self.split_stats_json):
            with open(self.split_stats_json, 'r') as f:
                meta = json.load(f)
            for k in ('split', 'seed', 'val_ratio'):
                if k not in meta:
                    raise RuntimeError(f'[ERROR] Invalid split_stats_json missing key: {k}')
            if 'stats' not in meta:
                raise RuntimeError(f'[ERROR] Invalid split_stats_json missing "stats"')
            if 'stats_var' not in meta or meta.get('stats_var_for', None) != var:
                # stats_var_for is not for the current task, re-calculate the stats
                ids_train = set(meta['split']['train'])
                stats = meta['stats']
                stats_var = cal_stats_on_train(rows, ids_train, var=var)
                meta['stats_var'] = stats_var
                meta['stats_var_for'] = var
                if self.calculate_if_missing:
                    os.makedirs(os.path.dirname(self.split_stats_json), exist_ok=True)
                    with open(self.split_stats_json, 'w') as f:
                        json.dump(meta, f, indent=2)
            if self.phase == 'test':
                if ('stats' not in meta) or ('stats_var' not in meta):
                    raise RuntimeError(f'[ERROR] split_stats_json (path: {self.split_stats_json}) '
                                       f'missing stats/stats_var for {self.phase}')
                if meta.get('stats_var_for', None) != var:
                    raise RuntimeError(f'[ERROR] split_stats_json (path: {self.split_stats_json}) '
                                       f'var is different with {var}')
            return meta

        rng = random.Random(self.seed)
        buckets = group_by_scenario(rows)
        train_ids, val_ids = [], []
        for sc, items in buckets.items():
            ids = [int(r['_row_id']) for r in items]
            rng.shuffle(ids)
            n_val = max(1, int(round(len(ids) * self.val_ratio)))
            val_ids.extend(ids[:n_val])
            train_ids.extend(ids[n_val:])

        train_ids_set = set(train_ids)
        stats_static_feature = cal_stats_on_train(rows, train_ids_set, var='static')
        stats_var = cal_stats_on_train(rows, train_ids_set, var=var)

        meta = {
            "seed": self.seed,
            "val_ratio": self.val_ratio,
            "split": {"train": train_ids, "val": val_ids},
            "stats": stats_static_feature,
            "stats_var": stats_var,
            "stats_var_for": var,
            "note": "Split by scenario with fixed seed; stats calculated on train only with masks."
        }

        if self.calculate_if_missing:
            os.makedirs(os.path.dirname(self.split_stats_json), exist_ok=True)
            with open(self.split_stats_json, 'w') as f:
                json.dump(meta, f, indent=2)
        return meta

    def _infer_h_fine_path(self, fine_path_uv: str, src_var: str):
        d, fname = os.path.split(fine_path_uv)

        d2 = d.replace(f'{os.sep}{src_var}{os.sep}', f'{os.sep}h{os.sep}', 1)
        d2 = d2.replace('/' + src_var + '/', '/h/', 1)
        d2 = d2.replace('\\' + src_var + '\\', '\\h\\', 1)

        if fname.startswith(src_var + '_'):
            fname2 = 'h_' + fname[len(src_var) + 1:]
        else:
            fname2 = re.sub(r'^[uv]_', 'h_', fname, count=1)

        return os.path.join(d2, fname2)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        Hf = self.patch_fine
        Hc = self.patch_coarse

        coarse_fm = load_npy_shape(r['coarse_path'], expect_shape=(Hc, Hc))
        elev = load_npy_shape(r['elev_path'], expect_shape=(Hf, Hf))
        rough = load_npy_shape(r['rough_path'], expect_shape=(Hf, Hf))
        slope = load_npy_shape(r['slope_path'], expect_shape=(Hf, Hf))
        twi = load_npy_shape(r['twi_path'], expect_shape=(Hf, Hf))
        asin = load_npy_shape(r['aspect_sin_path'], expect_shape=(Hf, Hf))
        acos = load_npy_shape(r['aspect_cos_path'], expect_shape=(Hf, Hf))
        mask_fine = load_npy_shape(r['mask_fine_path'], expect_shape=(Hf, Hf), dtype=np.uint8)

        if (self.phase == 'test') and (not os.path.isfile(r.get('fine_path', ''))):
            fine_fm = np.zeros((Hf, Hf), dtype=np.float32)
        else:
            fine_fm = load_npy_shape(r['fine_path'], expect_shape=(Hf, Hf))

        transform_list = [coarse_fm, fine_fm, elev, rough, slope, twi, asin, acos, mask_fine.astype(np.float32)]

        if (self.target_var in ('u', 'v')) and (self.phase == 'train') and (self.wet_threshold is not None):
            h_fine_path = self._infer_h_fine_path(r['fine_path'], src_var=self.target_var)
            if os.path.isfile(h_fine_path):
                h_fine = load_npy_shape(h_fine_path, expect_shape=(Hf, Hf), dtype=np.float32)
                wet_mask_fine = (h_fine >= self.wet_threshold).astype(np.float32)
                transform_list.append(wet_mask_fine)
            else:
                raise ValueError(f'[ERROR] Cannot find h_fine_path based on {self.target_var} file path: {h_fine_path}')

        transform_list = augment_flood_map(
            transform_list,
            target_var=self.target_var,
            use_hflip=self.use_hflip,
            use_rot=self.use_rot,
            is_train_phase=(self.phase == 'train'),
            idx_coarse=0,
            idx_fine=1,
            idx_asin=6,
            idx_acos=7
        )

        coarse_fm = transform_list[0]
        fine_fm = transform_list[1]
        elev = transform_list[2]
        rough = transform_list[3]
        slope = transform_list[4]
        twi = transform_list[5]
        asin = transform_list[6]
        acos = transform_list[7]
        mask_fine = (transform_list[8] > 0.5).astype(np.float32)

        loss_wet_mask = None
        if (self.target_var in ('u', 'v')) and (self.phase == 'train') and (self.wet_threshold is not None):
            wet_mask_fine = (transform_list[9] > 0.5).astype(np.float32)
            loss_wet_mask = (wet_mask_fine * mask_fine).astype(np.float32)

        S_static = self.stats_static
        S_var = self.stats_var

        if self.target_var == 'h':
            if self.transform == 'log1p':
                S_shared = S_var['shared']
                cfm = cal_log1p(coarse_fm)
                ffm = cal_log1p(fine_fm)
            elif self.transform == 'asinh':
                qk = str(self.h_asinh_q)
                asinh_stats = S_var['asinh_by_q'][qk]
                S_shared = asinh_stats['shared']
                s_val = float(asinh_stats['s'])
                cfm = cal_asinh_p90(coarse_fm, s_val)
                ffm = cal_asinh_p90(fine_fm, s_val)
            else:
                assert self.transform in ('log1p', 'asinh'), f"[ERROR] stats.transform must be log1p/asinh for h, got {self.transform}"

            if self.use_clip_coarse:
                cfm = percentile_clip(cfm, S_shared['p1'], S_shared['p99'])
                ffm = percentile_clip(ffm, S_shared['p1'], S_shared['p99'])
            if self.norm == 'zscore':
                cfm = cal_zscore(cfm, S_shared['mean'], S_shared['std'])
                ffm = cal_zscore(ffm, S_shared['mean'], S_shared['std'])
            else:
                cfm = cal_minmaxnorm(cfm, S_shared['min'], S_shared['max'])
                ffm = cal_minmaxnorm(ffm, S_shared['min'], S_shared['max'])

        else:
            S_shared = S_var['shared']
            s = float(S_shared['asinh_scale_shared'])
            cfm = cal_asinh_p90(coarse_fm, s)
            ffm = cal_asinh_p90(fine_fm, s)

            if self.use_clip_coarse:
                cfm = percentile_clip(cfm, S_shared['p1'], S_shared['p99'])
                ffm = percentile_clip(ffm, S_shared['p1'], S_shared['p99'])
            if self.norm == 'zscore':
                cfm = cal_zscore(cfm, S_shared['mean'], S_shared['std'])
                ffm = cal_zscore(ffm, S_shared['mean'], S_shared['std'])
            else:
                cfm = cal_minmaxnorm(cfm, S_shared['min'], S_shared['max'])
                ffm = cal_minmaxnorm(ffm, S_shared['min'], S_shared['max'])

        e = elev
        if self.use_clip_static:
            e = percentile_clip(e, S_static['elevation']['p1'], S_static['elevation']['p99'])
        if self.norm == 'zscore':
            e = cal_zscore(e, S_static['elevation']['mean'], S_static['elevation']['std'])
        else:
            e = cal_minmaxnorm(e, S_static['elevation']['min'], S_static['elevation']['max'])

        rgh = rough
        if self.use_clip_static:
            rgh = percentile_clip(rgh, S_static['roughness']['p1'], S_static['roughness']['p99'])
        if self.norm == 'zscore':
            rgh = cal_zscore(rgh, S_static['roughness']['mean'], S_static['roughness']['std'])
        else:
            rgh = cal_minmaxnorm(rgh, S_static['roughness']['min'], S_static['roughness']['max'])

        tf = twi
        if self.use_clip_static:
            tf = percentile_clip(tf, S_static['twi']['p1'], S_static['twi']['p99'])
        if self.norm == 'zscore':
            tf = cal_zscore(tf, S_static['twi']['mean'], S_static['twi']['std'])
        else:
            tf = cal_minmaxnorm(tf, S_static['twi']['min'], S_static['twi']['max'])

        slp = (slope / 90.0)
        a_sin = asin
        a_cos = acos
        mask = mask_fine

        fine_static_feature = np.stack([e, rgh, slp, tf, a_sin, a_cos, mask], axis=0).astype(np.float32)
        coarse_flood_map = np.expand_dims(cfm.astype(np.float32), axis=0)
        fine_flood_map = np.expand_dims(ffm.astype(np.float32), axis=0)

        sample = {
            'coarse_flood_map': torch.from_numpy(coarse_flood_map),  # [1, Hc, Hc]
            'fine_static_feature': torch.from_numpy(fine_static_feature),  # [7, Hf, Hf]
            'fine_flood_map': torch.from_numpy(fine_flood_map),  # [1, Hf, Hf]
            'meta': {
                'scenario': r['scenario'],
                't': r['t'],
                'downscale': self.scale,
                'row': int(r['patch_row']),
                'col': int(r['patch_col']),
                'var': self.target_var,
                'coarse_path': r['coarse_path'],
                'fine_path': r['fine_path'],
            }
        }

        if loss_wet_mask is not None:
            sample['wet_mask'] = torch.from_numpy(np.expand_dims(loss_wet_mask, axis=0))

        return sample
