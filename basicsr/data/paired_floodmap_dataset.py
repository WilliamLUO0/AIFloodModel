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
import numpy as np
from copy import deepcopy
import torch
from torch.utils import data as data

from basicsr.utils import (
    cal_log1p, cal_zscore, cal_minmaxnorm, cal_asinh_p90,
    percentile_clip, load_npy_shape, load_index_csv,
    check_required_fields, get_root_logger
)
from basicsr.data.transforms import augment_flood_map
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedFloodMapDataset(data.Dataset):
    """
    Paired Flood map dataset (index.csv + .npy).

    Output:
     - coarse_flood_map:
        default:
            [1, Hc, Hc] -> coarse target map only
        when target_var == 'h' and aux_vars enabled:
            [1 + len(aux_vars), Hc, Hc]
            e.g. target_var=h, aux_var=[zs, dem]
            -> channel order: [h, zs, dem]

    - fine_static_feature:
        [7, Hf, Hf]
        channel order: [elevation, roughness, slope, twi, aspect_sin, aspect_cos, mask]

    - fine_flood_map:
        [1, Hf, Hf]
        main target map (h / u/ v)
    """

    def __init__(self, opt):
        super(PairedFloodMapDataset, self).__init__()
        self.opt = deepcopy(opt)

        self.phase = opt.get('phase', 'train')
        assert self.phase in ('train', 'val', 'test'), f'phase must be train/val/test, got {self.phase}'

        self.target_var = opt.get('target_var', 'h')
        assert self.target_var in ('h', 'u', 'v'), f"target_var should be one of ['h', 'u', 'v'], got {self.target_var}"

        self.aux_vars = self._parse_aux_vars(opt.get('aux_vars', opt.get('aux_var', [])))
        allowed_aux = {'zs', 'dem'}
        unknown_aux = [x for x in self.aux_vars if x not in allowed_aux]
        if len(unknown_aux) > 0:
            raise ValueError(f'[ERROR] Unsupported aux_vars: {unknown_aux}, allowed: {sorted(list(allowed_aux))}')

        if self.target_var != 'h' and len(self.aux_vars) > 0:
            raise ValueError(
                f"[ERROR] aux_vars={self.aux_vars} is currently only supported when target_var == 'h'. "
                f"Got target_var={self.target_var}"
            )

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
        # NOTE: `calculate_if_missing` is kept for backward-compat with old yml,
        # but is now ignored. The in-code stats computation was removed because
        # its output schema did not match the dataset/loss readers and would
        # silently produce broken split_stats JSONs. Pre-compute the JSON via
        # tools/precompute_split_stats*.py and reference it via split_cfg.
        if bool(stats_cfg.get('calculate_if_missing', False)):
            get_root_logger().warning(
                '[PairedFloodMapDataset] stats.calculate_if_missing=True is ignored. '
                'split_stats JSON must be pre-computed via tools/precompute_split_stats*.py.'
            )
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

        self.meta = self._ensure_split_stats(rows, var=self.target_var, aux_vars=self.aux_vars)
        self.stats_static = self.meta['stats']
        self.stats_var = self.meta['stats_var']
        self.aux_stats = self.meta.get('aux_stats', {}) or {}

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

        check_required_fields(self.rows)
        self._check_required_fields_for_aux(self.rows)

    def _parse_aux_vars(self, aux_raw):
        if aux_raw is None:
            return []

        if isinstance(aux_raw, str):
            s = aux_raw.strip()
            if s == '':
                return []
            return [x.strip() for x in s.split(',') if x.strip() != '']

        if isinstance(aux_raw, (list, tuple, set)):
            return [str(x).strip() for x in aux_raw if str(x).strip() != '']

        raise TypeError(f'[ERROR] aux_vars must be str/list/tuple/set, got {type(aux_raw)}')

    def _check_required_fields_for_aux(self, rows):
        if len(rows) == 0:
            return

        if 'zs' in self.aux_vars:
            if 'zs_coarse_path' not in rows[0]:
                raise RuntimeError('[ERROR] index.csv missing field for aux var zs: zs_coarse_path')

        if 'dem' in self.aux_vars:
            if 'elev_coarse_path' not in rows[0]:
                raise RuntimeError('[ERROR] index.csv missing field for aux var dem: elev_coarse_path')

    def _ensure_split_stats(self, rows, var: str, aux_vars=None):
        """
        Load the split_stats JSON referenced by ``split_cfg.split_stats_json``.

        The JSON must be pre-computed via ``tools/precompute_split_stats*.py``
        before training. The legacy in-code fallback (which used to randomly
        split + recompute stats here) has been removed because its output
        schema did not match the dataset / loss readers and would silently
        write a broken JSON.

        The JSON is expected to contain the full schema:
          - ``split.train`` / ``split.val`` (list of row ids)
          - ``seed`` / ``val_ratio``
          - ``stats``     (static feature stats)
          - ``stats_var`` (target variable stats: ``shared`` and/or ``asinh_by_q``,
                          plus ``pos_weight_fine_raw_tau`` when BCE losses are used)
          - ``stats_var_for == var``
          - ``aux_stats[name]`` for every name in ``aux_vars``
        """
        aux_vars = aux_vars or []
        _ = rows  # rows are no longer used here; kept in the signature for back-compat.

        if not os.path.isfile(self.split_stats_json):
            raise FileNotFoundError(
                f'[ERROR] split_stats_json not found: {self.split_stats_json}\n'
                f'Pre-compute it via tools/precompute_split_stats*.py before training.'
            )

        with open(self.split_stats_json, 'r') as f:
            meta = json.load(f)

        for k in ('split', 'seed', 'val_ratio', 'stats', 'stats_var'):
            if k not in meta:
                raise RuntimeError(
                    f'[ERROR] split_stats_json (path: {self.split_stats_json}) missing key: {k}. '
                    f'Re-generate the JSON via tools/precompute_split_stats*.py.'
                )

        if meta.get('stats_var_for', None) != var:
            raise RuntimeError(
                f'[ERROR] split_stats_json (path: {self.split_stats_json}) has '
                f'stats_var_for={meta.get("stats_var_for")}, expected {var}. '
                f'Re-generate the JSON for var={var}.'
            )

        if 'zs' in aux_vars:
            aux_stats = meta.get('aux_stats', {}) or {}
            if 'zs' not in aux_stats:
                raise RuntimeError(
                    f'[ERROR] aux_vars includes "zs", but split_stats_json does not contain '
                    f'meta["aux_stats"]["zs"]. Please regenerate stats json with --aux_vars zs'
                )

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

    def _clip_by_stats(self, arr, stat_dict, enabled, name='unknown'):
        if not enabled:
            return arr
        if ('p1' not in stat_dict) or ('p99' not in stat_dict):
            raise RuntimeError(
                f'[ERROR] use_clip is enabled for {name}, but stats do not contain p1/p99. '
                f'Please regenerate stats with percentile fields, or set use_clip_* = false.'
            )
        return percentile_clip(arr, stat_dict['p1'], stat_dict['p99'])

    def __len__(self):
        return len(self.rows)

    def get_interval_sample_weights(self, sampler_cfg=None):
        """Compute patch-level sampling weights from interval ratios.

        This method is used by IntervalBalancedSampler.

        Required CSV fields:
            slight_ratio
            severe_ratio
            extreme_ratio

        Recommended YAML:
            sampler:
              type: interval_balanced
              score_mode: percentile_band
              summary_json: /path/to/patch_interval_summary_h_train.json
              percentile_source: positive
              low_percentile: p75
              high_percentile: p90
              slight_alpha: 2.0
              severe_alpha: 1.0
              extreme_alpha: 1.0
              base_weight: 1.0
              max_weight: 5.0
        """
        sampler_cfg = sampler_cfg or {}

        score_mode = str(sampler_cfg.get('score_mode', 'percentile_band')).lower().strip()

        base_weight = float(sampler_cfg.get('base_weight', 1.0))
        max_weight = float(sampler_cfg.get('max_weight', 10.0))

        slight_alpha = float(sampler_cfg.get('slight_alpha', 4.0))
        severe_alpha = float(sampler_cfg.get('severe_alpha', 2.0))
        extreme_alpha = float(sampler_cfg.get('extreme_alpha', 2.0))

        eps = 1e-12

        def _to_float(x, default=0.0):
            if x is None:
                return default
            if isinstance(x, str) and x.strip() == '':
                return default
            return float(x)

        def _band_score(ratio, low, high):
            denom = max(high - low, eps)
            score = (ratio - low) / denom
            if score < 0.0:
                return 0.0
            if score > 1.0:
                return 1.0
            return float(score)

        def _ratio_ref_score(ratio, ref):
            return float(min(ratio / max(ref, eps), 1.0))

        def _load_percentile_refs_from_summary(summary_json, percentile_source, low_percentile, high_percentile):
            if summary_json is None or str(summary_json).strip() == '':
                raise ValueError(
                    '[ERROR] sampler.summary_json is required when using percentile_band with percentile settings.'
                )

            summary_json = str(summary_json)
            if not os.path.isfile(summary_json):
                raise FileNotFoundError(f'[ERROR] sampler.summary_json not found: {summary_json}')

            with open(summary_json, 'r') as f:
                summary = json.load(f)

            if percentile_source == 'positive':
                key = 'ratio_percentiles_positive_patches'
            elif percentile_source == 'all':
                key = 'ratio_percentiles_all_patches'
            else:
                raise ValueError(
                    f"[ERROR] percentile_source must be 'positive' or 'all', got {percentile_source}"
                )

            if key not in summary:
                raise RuntimeError(f'[ERROR] summary_json missing key: {key}')

            refs = {}

            for name in ['slight_ratio', 'severe_ratio', 'extreme_ratio']:
                if name not in summary[key]:
                    raise RuntimeError(f'[ERROR] summary_json missing {key}/{name}')

                low = summary[key][name].get(low_percentile, None)
                high = summary[key][name].get(high_percentile, None)

                if low is None or high is None:
                    raise RuntimeError(
                        f'[ERROR] summary_json missing percentile for {name}: '
                        f'{low_percentile}={low}, {high_percentile}={high}'
                    )

                low = float(low)
                high = float(high)

                if not np.isfinite(low) or not np.isfinite(high):
                    raise RuntimeError(
                        f'[ERROR] invalid percentile values for {name}: low={low}, high={high}'
                    )

                if high <= low:
                    raise RuntimeError(
                        f'[ERROR] high percentile must be larger than low percentile for {name}: '
                        f'{high_percentile}={high}, {low_percentile}={low}'
                    )

                refs[name] = {
                    'low': low,
                    'high': high,
                }

            return refs

        weights = []

        if score_mode == 'percentile_band':
            # Preferred mode:
            # Load low/high percentiles directly from patch_interval_summary_h_train.json.
            summary_json = sampler_cfg.get('summary_json', '')
            percentile_source = str(sampler_cfg.get('percentile_source', 'positive')).lower().strip()
            low_percentile = str(sampler_cfg.get('low_percentile', 'p75')).strip()
            high_percentile = str(sampler_cfg.get('high_percentile', 'p90')).strip()

            if summary_json:
                refs = _load_percentile_refs_from_summary(
                    summary_json=summary_json,
                    percentile_source=percentile_source,
                    low_percentile=low_percentile,
                    high_percentile=high_percentile,
                )

                slight_low = refs['slight_ratio']['low']
                slight_high = refs['slight_ratio']['high']
                severe_low = refs['severe_ratio']['low']
                severe_high = refs['severe_ratio']['high']
                extreme_low = refs['extreme_ratio']['low']
                extreme_high = refs['extreme_ratio']['high']

            else:
                # Fallback: allow manual values in YAML.
                slight_low = float(sampler_cfg['slight_low'])
                slight_high = float(sampler_cfg['slight_high'])
                severe_low = float(sampler_cfg['severe_low'])
                severe_high = float(sampler_cfg['severe_high'])
                extreme_low = float(sampler_cfg['extreme_low'])
                extreme_high = float(sampler_cfg['extreme_high'])

            for r in self.rows:
                slight_ratio = _to_float(r.get('slight_ratio', 0.0))
                severe_ratio = _to_float(r.get('severe_ratio', 0.0))
                extreme_ratio = _to_float(r.get('extreme_ratio', 0.0))

                slight_score = _band_score(slight_ratio, slight_low, slight_high)
                severe_score = _band_score(severe_ratio, severe_low, severe_high)
                extreme_score = _band_score(extreme_ratio, extreme_low, extreme_high)

                w = base_weight
                w += slight_alpha * slight_score
                w += severe_alpha * severe_score
                w += extreme_alpha * extreme_score
                w = min(w, max_weight)

                weights.append(w)

        elif score_mode == 'ratio_ref':
            slight_ref = float(sampler_cfg['slight_ref'])
            severe_ref = float(sampler_cfg['severe_ref'])
            extreme_ref = float(sampler_cfg['extreme_ref'])

            for r in self.rows:
                slight_ratio = _to_float(r.get('slight_ratio', 0.0))
                severe_ratio = _to_float(r.get('severe_ratio', 0.0))
                extreme_ratio = _to_float(r.get('extreme_ratio', 0.0))

                slight_score = _ratio_ref_score(slight_ratio, slight_ref)
                severe_score = _ratio_ref_score(severe_ratio, severe_ref)
                extreme_score = _ratio_ref_score(extreme_ratio, extreme_ref)

                w = base_weight
                w += slight_alpha * slight_score
                w += severe_alpha * severe_score
                w += extreme_alpha * extreme_score
                w = min(w, max_weight)

                weights.append(w)

        elif score_mode == 'hard_rich':
            slight_thr = float(sampler_cfg['slight_thr'])
            severe_thr = float(sampler_cfg['severe_thr'])
            extreme_thr = float(sampler_cfg['extreme_thr'])

            for r in self.rows:
                slight_ratio = _to_float(r.get('slight_ratio', 0.0))
                severe_ratio = _to_float(r.get('severe_ratio', 0.0))
                extreme_ratio = _to_float(r.get('extreme_ratio', 0.0))

                w = base_weight

                if slight_ratio >= slight_thr:
                    w += slight_alpha
                if severe_ratio >= severe_thr:
                    w += severe_alpha
                if extreme_ratio >= extreme_thr:
                    w += extreme_alpha

                w = min(w, max_weight)
                weights.append(w)

        else:
            raise ValueError(f'[ERROR] Unknown interval sampler score_mode: {score_mode}')

        weights = torch.as_tensor(weights, dtype=torch.double)

        if weights.numel() != len(self.rows):
            raise RuntimeError(
                f'[ERROR] weight number mismatch: weights={weights.numel()}, rows={len(self.rows)}'
            )

        if torch.any(~torch.isfinite(weights)):
            raise RuntimeError('[ERROR] interval sample weights contain NaN or Inf.')

        if torch.any(weights < 0):
            raise RuntimeError('[ERROR] interval sample weights contain negative values.')

        if torch.sum(weights) <= 0:
            raise RuntimeError('[ERROR] sum of interval sample weights <= 0.')

        return weights

    def __getitem__(self, index):
        r = self.rows[index]
        Hf = self.patch_fine
        Hc = self.patch_coarse

        # -----------------------------
        # load main target inputs
        # -----------------------------
        coarse_fm = load_npy_shape(r['coarse_path'], expect_shape=(Hc, Hc))
        if (self.phase == 'test') and (not os.path.isfile(r.get('fine_path', ''))):
            # Inference-only branch (real-world events without ground-truth fine
            # flood map). We populate fine_fm with zeros so the rest of the
            # pipeline (normalization, batching, model.feed_data) still works,
            # but DOWNSTREAM EVAL METRICS COMPUTED ON THIS BATCH ARE
            # MEANINGLESS. This branch should only ever be reached in pure
            # inference, never in train/val.
            if not getattr(self, '_warned_missing_fine_path', False):
                get_root_logger().warning(
                    f'[PairedFloodMapDataset] phase=test: fine_path missing for '
                    f'row scenario={r.get("scenario")}, t={r.get("t")}, '
                    f'patch=({r.get("patch_row")},{r.get("patch_col")}). '
                    f'Returning zero fine_fm. Eval metrics on these patches '
                    f'will be invalid; only use predictions for inference.'
                )
                self._warned_missing_fine_path = True
            fine_fm = np.zeros((Hf, Hf), dtype=np.float32)
        else:
            fine_fm = load_npy_shape(r['fine_path'], expect_shape=(Hf, Hf))

        # -----------------------------
        # load fine-grid static features
        # -----------------------------
        elev_key = 'elev_fine_path' if 'elev_fine_path' in r else 'elev_path'
        elev = load_npy_shape(r[elev_key], expect_shape=(Hf, Hf))
        rough = load_npy_shape(r['rough_path'], expect_shape=(Hf, Hf))
        slope = load_npy_shape(r['slope_path'], expect_shape=(Hf, Hf))
        twi = load_npy_shape(r['twi_path'], expect_shape=(Hf, Hf))
        asin = load_npy_shape(r['aspect_sin_path'], expect_shape=(Hf, Hf))
        acos = load_npy_shape(r['aspect_cos_path'], expect_shape=(Hf, Hf))
        mask_fine = load_npy_shape(r['mask_fine_path'], expect_shape=(Hf, Hf), dtype=np.uint8)

        # -----------------------------
        # load optional coarse auxiliary variables
        # keep loading in the same order as self.aux_vars
        # -----------------------------
        aux_loaded = {}
        if self.target_var == 'h':
            for aux_name in self.aux_vars:
                if aux_name == 'zs':
                    aux_loaded['zs'] = load_npy_shape(r['zs_coarse_path'], expect_shape=(Hc, Hc))
                elif aux_name == 'dem':
                    aux_loaded['dem'] = load_npy_shape(r['elev_coarse_path'], expect_shape=(Hc, Hc))
                else:
                    raise ValueError(f'[ERROR] Unsupported aux var: {aux_name}')

        # -----------------------------
        # build transform list
        # -----------------------------
        transform_list = [
            coarse_fm,                      # 0
            fine_fm,                        # 1
            elev,                           # 2
            rough,                          # 3
            slope,                          # 4
            twi,                            # 5
            asin,                           # 6
            acos,                           # 7
            mask_fine.astype(np.float32)    # 8
        ]

        aux_transform_indices = {}
        for aux_name in self.aux_vars:
            aux_transform_indices[aux_name] = len(transform_list)
            transform_list.append(aux_loaded[aux_name])

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

        # -----------------------------
        # unpack transformed data
        # -----------------------------
        coarse_fm = transform_list[0]
        fine_fm = transform_list[1]
        elev = transform_list[2]
        rough = transform_list[3]
        slope = transform_list[4]
        twi = transform_list[5]
        asin = transform_list[6]
        acos = transform_list[7]
        mask_fine = (transform_list[8] > 0.5).astype(np.float32)

        for aux_name in self.aux_vars:
            aux_loaded[aux_name] = transform_list[aux_transform_indices[aux_name]]

        loss_wet_mask = None
        if (self.target_var in ('u', 'v')) and (self.phase == 'train') and (self.wet_threshold is not None):
            wet_mask_fine = (transform_list[-1] > 0.5).astype(np.float32)
            loss_wet_mask = (wet_mask_fine * mask_fine).astype(np.float32)

        S_static = self.stats_static
        S_var = self.stats_var

        # -----------------------------
        # normalize main target variable
        # -----------------------------
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
                raise ValueError(f"[ERROR] Unsupported h transform: {self.transform}")

            cfm = self._clip_by_stats(cfm, S_shared, self.use_clip_coarse, name='coarse_h')
            ffm = self._clip_by_stats(ffm, S_shared, self.use_clip_coarse, name='fine_h')

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

            cfm = self._clip_by_stats(cfm, S_shared, self.use_clip_coarse, name='coarse_uv')
            ffm = self._clip_by_stats(ffm, S_shared, self.use_clip_coarse, name='fine_uv')

            if self.norm == 'zscore':
                cfm = cal_zscore(cfm, S_shared['mean'], S_shared['std'])
                ffm = cal_zscore(ffm, S_shared['mean'], S_shared['std'])
            else:
                cfm = cal_minmaxnorm(cfm, S_shared['min'], S_shared['max'])
                ffm = cal_minmaxnorm(ffm, S_shared['min'], S_shared['max'])

        # -----------------------------
        # normalize fine static features
        # -----------------------------
        e = self._clip_by_stats(elev, S_static['elevation'], self.use_clip_static, name='elevation_fine')
        if self.norm == 'zscore':
            e = cal_zscore(e, S_static['elevation']['mean'], S_static['elevation']['std'])
        else:
            e = cal_minmaxnorm(e, S_static['elevation']['min'], S_static['elevation']['max'])

        rgh = self._clip_by_stats(rough, S_static['roughness'], self.use_clip_static, name='roughness')
        if self.norm == 'zscore':
            rgh = cal_zscore(rgh, S_static['roughness']['mean'], S_static['roughness']['std'])
        else:
            rgh = cal_minmaxnorm(rgh, S_static['roughness']['min'], S_static['roughness']['max'])

        tf = self._clip_by_stats(twi, S_static['twi'], self.use_clip_static, name='twi')
        if self.norm == 'zscore':
            tf = cal_zscore(tf, S_static['twi']['mean'], S_static['twi']['std'])
        else:
            tf = cal_minmaxnorm(tf, S_static['twi']['min'], S_static['twi']['max'])

        slp = (slope / 90.0)
        a_sin = asin
        a_cos = acos
        mask = mask_fine

        fine_static_feature = np.stack([e, rgh, slp, tf, a_sin, a_cos, mask], axis=0).astype(np.float32)

        # -----------------------------
        # normalize aux vars and keep order:
        # [target_var] + aux_vars in yaml order
        # -----------------------------
        coarse_channels = [cfm.astype(np.float32)]

        for aux_name in self.aux_vars:
            if aux_name == 'zs':
                if 'zs' not in self.aux_stats:
                    raise RuntimeError('[ERROR] aux_stats["zs"] not found in split_stats_json.')

                S_zs = self.aux_stats['zs']['shared']
                arr = aux_loaded['zs']
                arr = self._clip_by_stats(arr, S_zs, self.use_clip_coarse, name='coarse_zs')

                if self.norm == 'zscore':
                    arr = cal_zscore(arr, S_zs['mean'], S_zs['std'])
                else:
                    arr = cal_minmaxnorm(arr, S_zs['min'], S_zs['max'])

                coarse_channels.append(arr.astype(np.float32))

            elif aux_name == 'dem':
                S_elev = S_static['elevation']
                arr = aux_loaded['dem']
                arr = self._clip_by_stats(arr, S_elev, self.use_clip_static, name='coarse_elevation')

                if self.norm == 'zscore':
                    arr = cal_zscore(arr, S_elev['mean'], S_elev['std'])
                else:
                    arr = cal_minmaxnorm(arr, S_elev['min'], S_elev['max'])

                coarse_channels.append(arr.astype(np.float32))

            else:
                raise ValueError(f'[ERROR] Unsupported aux var during normalization: {aux_name}')

        coarse_flood_map = np.stack(coarse_channels, axis=0).astype(np.float32)
        fine_flood_map = np.expand_dims(ffm.astype(np.float32), axis=0)

        sample = {
            'coarse_flood_map': torch.from_numpy(coarse_flood_map),         # [1 + len(aux_vars), Hc, Hc]
            'fine_static_feature': torch.from_numpy(fine_static_feature),   # [7, Hf, Hf]
            'fine_flood_map': torch.from_numpy(fine_flood_map),             # [1, Hf, Hf]
            'meta': {
                'scenario': r['scenario'],
                't': r['t'],
                'downscale': self.scale,
                'row': int(r['patch_row']),
                'col': int(r['patch_col']),
                'var': self.target_var,
                'aux_vars': list(self.aux_vars),
                'coarse_path': r['coarse_path'],
                'fine_path': r['fine_path'],
            }
        }

        if loss_wet_mask is not None:
            sample['wet_mask'] = torch.from_numpy(np.expand_dims(loss_wet_mask, axis=0))

        return sample
