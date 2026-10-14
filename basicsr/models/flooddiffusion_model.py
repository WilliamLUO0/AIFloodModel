"""
flooddiffusion_model.py -- Paper 2 S8: Flood-DM (pixel-space variant of Flood-LDM, Neo et al. 2026) under our protocol.

The diffusion process follows flood-diff (Apache-2.0, commit d04b751) model/sr3_modules/diffusion.py: SR3 continuous
noise level, linear beta schedule, epsilon prediction, ancestral DDPM sampler with x0 clipping and their truncated start
from the noised coarse map. Specification: paper2_S8/S8_FloodDM.md §5, README paper2_S8 §3 (rulings).

Data side = ours (sampler, flips/rotations, batch, seed, inputs); model side = theirs. Specifics:
  * conditioning: cat[x_t, 3 coarse channels bilinear x8, 7 fine statics] -> FloodSR3UNet(in_channel 11), z-space;
  * diffusion variable: x0 = 2*asinh(h/s)/A - 1 with asinh(h/s) = z*sigma + mu (stats json, never hard-coded) and
    A = diffusion.asinh_range (4.0): dry -> -1; inverse mapped back to z-space for evaluation; x0 := -1 outside the AOI;
  * training: one integer t ~ U{1..T} per batch, per-sample gamma ~ U[sqrt(abar_{t-1}), sqrt(abar_t)],
    x_t = gamma*x0 + sqrt(1-gamma^2)*eps, loss = masked L1 (or L2) on the noise, mean over AOI cells of the batch;
    Adam + linear warm-up then constant lr (train.scheduler MultiStepLR with no milestones); EMA 0.999 (ours);
    autocast (bf16) around the denoiser only;
  * sampling: t_start = 0 -> the coarse map itself; 0 < t_start < T -> their truncated start (coarse map in x0-space
    noised to gamma ~ U[sqrt(abar_{t_start-1}), sqrt(abar_{t_start})]); t_start >= T -> pure noise, T steps. After every
    step the outside-AOI pixels are reset to the forward-noised dry value (ruling). Noise comes from a per-patch
    generator seeded by seed_base + crc32(scenario|t|row|col|var) (+ sample index): reproducible, independent of the
    patch order / subset / GPU split. n_samples > 1 -> mean of the samples in metres (SI row);
  * validation: during training (called on rank 0 only by train_flood_map_v2.py) plain loop; at test time
    (test_flood_map.py calls every rank) the patches are split over the ranks and the metric sums are all-reduced;
  * resume (test only, opt-in, added 2026-10-14 by the discussion-session review): `diffusion.resume_vis_dir` = the
    visualization/ folder of an archived, killed run -> its finished patches are moved over instead of re-sampled
    (a full val split at T_s >= 500 does not fit one 24 h job).
"""
import math
import os
import zlib
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, destand_to_physical
from basicsr.utils.registry import MODEL_REGISTRY
from .fmsr_model import FMSRModel


@MODEL_REGISTRY.register()
class FloodDiffusionModel(FMSRModel):

    def __init__(self, opt):
        self._setup_diffusion(opt['diffusion'], opt)
        super().__init__(opt)

    # ------------------------------------------------------------------ setup
    def _setup_diffusion(self, d, opt):
        sched = d.get('schedule', 'linear')
        if sched != 'linear':
            raise ValueError(f'[FloodDiffusionModel] only the linear beta schedule is used (got {sched})')
        self.T = int(d.get('n_timestep', 1000))
        betas = np.linspace(float(d.get('linear_start', 1e-6)), float(d.get('linear_end', 1e-2)), self.T, dtype=np.float64)
        alphas = 1. - betas
        ac = np.cumprod(alphas)
        ac_prev = np.append(1., ac[:-1])
        self.sqrt_ac_prev = np.sqrt(np.append(1., ac))           # length T+1: index k -> sqrt(abar_{k-1}), [0] = 1
        post_var = betas * (1. - ac_prev) / (1. - ac)
        self._sched_np = dict(
            sqrt_recip_ac=np.sqrt(1. / ac), sqrt_recipm1_ac=np.sqrt(1. / ac - 1),
            post_logvar=np.log(np.maximum(post_var, 1e-20)),
            coef1=betas * np.sqrt(ac_prev) / (1. - ac), coef2=(1. - ac_prev) * np.sqrt(alphas) / (1. - ac))
        self.alphas_cumprod = ac
        self.loss_type = str(d.get('loss_type', 'l1')).lower()
        if self.loss_type not in ('l1', 'l2'):
            raise ValueError(f'[FloodDiffusionModel] loss_type must be l1/l2, got {self.loss_type}')
        self.clip_denoised = bool(d.get('clip_denoised', True))
        self.outside_aoi_replace = bool(d.get('outside_aoi_replace', True))
        self.upsample_mode = str(d.get('upsample_mode', 'bilinear'))
        self.A = float(d['asinh_range'])
        with open(str(d['stats_json']), 'r', encoding='utf-8') as f:
            node = json.load(f)['stats_var']['asinh_by_q'][str(int(d.get('h_asinh_q', 90)))]
        S = node['shared']
        self.mu, self.sigma, self.s_asinh = float(S['mean']), float(S['std']), float(node['s'])
        if not float(S['max']) < self.A:
            raise ValueError(f'[FloodDiffusionModel] asinh_range A={self.A} must exceed the training asinh max {S["max"]}')
        self.amp_dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[str(d.get('amp_dtype', 'bfloat16'))]
        self.use_amp_train = bool(d.get('amp_train', True))
        self.use_amp_sample = bool(d.get('amp_sample', True))
        if d.get('t_start', None) is None:
            raise ValueError('[FloodDiffusionModel] diffusion.t_start is not set: choose T_s with the valsub256 sweep '
                             '(their < 1 % rule) and write it into this yml')
        self.t_start = int(d['t_start'])
        self.n_samples = int(d.get('n_samples', 1))
        self.seed_base = int(d.get('seed_base', opt.get('manual_seed', 0) or 0))
        # Resume a killed inference job (opt-in, test only): BasicSR archives an existing results folder at start-up
        # (make_exp_dirs -> mkdir_and_rename), so point `resume_vis_dir` at the ARCHIVED run's visualization/ folder.
        # Patches whose .npy is found there (same file name, loadable, right shape, finite) are moved into the new
        # folder instead of being re-sampled. The per-patch seeds make a resumed run identical to an uninterrupted one.
        self.resume_vis_dir = d.get('resume_vis_dir', None) or None
        if not 0 <= self.t_start <= self.T or self.n_samples < 1:
            raise ValueError(f'[FloodDiffusionModel] t_start must be in [0, {self.T}] and n_samples >= 1')
        logger = get_root_logger()
        logger.info(f'[FloodDiffusionModel] T={self.T} linear beta {d.get("linear_start")}..{d.get("linear_end")}, '
                    f'loss={self.loss_type}, A={self.A}, mu={self.mu:.6f}, sigma={self.sigma:.6f}, s={self.s_asinh}, '
                    f't_start={self.t_start}, n_samples={self.n_samples}, seed_base={self.seed_base}, '
                    f'outside_aoi_replace={self.outside_aoi_replace}, amp={self.amp_dtype} '
                    f'(train {self.use_amp_train} / sample {self.use_amp_sample})')

    def _sched(self, name, i):
        return float(self._sched_np[name][i])

    def to_x0(self, z):
        return 2. * (z * self.sigma + self.mu) / self.A - 1.

    def from_x0(self, x0):
        return ((x0 + 1.) * self.A / 2. - self.mu) / self.sigma

    def _autocast(self, enabled):
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                              enabled=enabled and self.device.type == 'cuda')

    def _condition(self, coarse, static):
        up = F.interpolate(coarse, size=static.shape[-2:], mode=self.upsample_mode,
                           **({} if self.upsample_mode == 'nearest' else {'align_corners': False}))
        return torch.cat([up, static], dim=1), up

    def init_training_settings(self):
        train_opt = self.opt['train']
        for k in ('pixel_opt', 'ordinal_opt', 'nonflood_opt', 'slight_opt', 'deep_opt', 'flood_bce_opt', 'prev_opt'):
            if train_opt.get(k):
                raise ValueError(f'[FloodDiffusionModel] train.{k} must be absent: the loss is the masked noise loss')
        self.net_g.train()
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            get_root_logger().info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()
            if any(True for _ in self.net_g_ema.buffers()):
                raise RuntimeError('[FloodDiffusionModel] denoiser has buffers; the parameter-only EMA would leave them stale')
        self.use_wet_mask = False
        self.setup_optimizers()
        self.setup_schedulers()

    # ------------------------------------------------------------------ training
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad(set_to_none=True)
        mask = self.mask
        x0 = torch.where(mask > 0.5, self.to_x0(self.fine_fm), torch.full_like(self.fine_fm, -1.))
        B = x0.shape[0]
        t = int(torch.randint(1, self.T + 1, (1,)).item())            # one t per batch, as released
        lo, hi = self.sqrt_ac_prev[t - 1], self.sqrt_ac_prev[t]
        gamma = (hi + (lo - hi) * torch.rand(B, 1, device=self.device)).float()   # U[sqrt(abar_t), sqrt(abar_{t-1})]
        noise = torch.randn_like(x0)
        g4 = gamma.view(B, 1, 1, 1)
        x_t = g4 * x0 + (1. - g4 ** 2).sqrt() * noise
        cond, _ = self._condition(self.coarse_fm, self.static_f)
        with self._autocast(self.use_amp_train):
            eps = self.net_g(torch.cat([x_t, cond], dim=1), gamma)
        l_eps = self.noise_loss(eps.float(), noise, mask)
        if not torch.isfinite(l_eps):
            raise RuntimeError(f'[FloodDiffusionModel] non-finite loss at iter {current_iter}')
        l_eps.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(OrderedDict(l_eps=l_eps.detach(),
                                                          t_frac=torch.tensor(t / self.T, device=self.device)))
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def noise_loss(self, eps_hat, noise, mask):
        """Masked L1 / L2 on the noise: mean over the AOI cells of the batch (0, not NaN, for an empty mask)."""
        err = eps_hat - noise
        err = err.abs() if self.loss_type == 'l1' else err ** 2
        return (err * mask).sum() / mask.sum().clamp_min(1.)

    # ------------------------------------------------------------------ sampling
    def p_step(self, net, img, cond, i, noise):
        """One ancestral step x_i -> x_{i-1} (= p_sample of the release, with an explicit noise tensor)."""
        level = torch.full((img.shape[0], 1), float(self.sqrt_ac_prev[i + 1]), device=img.device)
        with self._autocast(self.use_amp_sample):
            eps = net(torch.cat([img, cond], dim=1), level)
        x0_hat = self._sched('sqrt_recip_ac', i) * img - self._sched('sqrt_recipm1_ac', i) * eps.float()
        if self.clip_denoised:
            x0_hat = x0_hat.clamp(-1., 1.)
        mean = self._sched('coef1', i) * x0_hat + self._sched('coef2', i) * img
        return mean + math.exp(0.5 * self._sched('post_logvar', i)) * noise if i > 0 else mean

    @staticmethod
    def _meta_item(meta, key, i):
        v = meta[key]
        if isinstance(v, (list, tuple)):
            v = v[i]
        if torch.is_tensor(v):
            v = v.view(-1)[i].item() if v.numel() > 1 else v.item()
        return v

    def patch_seed(self, meta, i, k=0):
        ident = '|'.join(str(self._meta_item(meta, key, i)) for key in ('scenario', 't', 'row', 'col', 'var'))
        return (self.seed_base * 1000003 + zlib.crc32(ident.encode()) + k * 1000000007) % (2 ** 62)

    @torch.no_grad()
    def sample(self, net, coarse, static, t_start, generators):
        """One reverse chain per sample; returns x0 in [-1, 1] (diffusion space). generators: one per batch item."""
        dev = static.device
        B, _, H, W = static.shape
        cond, up = self._condition(coarse, static)
        mask = static[:, -1:] > 0.5
        x_start = torch.where(mask, self.to_x0(up[:, :1]).clamp(-1., 1.), torch.full_like(up[:, :1], -1.))
        randn = lambda: torch.stack([torch.randn((1, H, W), generator=g, device=dev) for g in generators])
        rand = lambda: torch.stack([torch.rand((1,), generator=g, device=dev) for g in generators])
        if t_start == 0:
            return x_start
        if t_start >= self.T:
            n_steps, img = self.T, randn()
        else:
            n_steps = t_start
            lo, hi = self.sqrt_ac_prev[t_start - 1], self.sqrt_ac_prev[t_start]
            g4 = (hi + (lo - hi) * rand()).float().view(B, 1, 1, 1)
            img = g4 * x_start + (1. - g4 ** 2).sqrt() * randn()
        for i in reversed(range(n_steps)):
            img = self.p_step(net, img, cond, i, randn() if i > 0 else None)
            if self.outside_aoi_replace:
                if i > 0:
                    a = float(self.sqrt_ac_prev[i])                   # level of x_{i}: sqrt(abar_{i-1})
                    out = -a + math.sqrt(1. - a * a) * randn()
                else:
                    out = torch.full_like(img, -1.)
                img = torch.where(mask, img, out)
        return img

    def test(self):
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.get_bare_model(self.net_g)
        was_training = net.training
        net.eval()
        B = self.static_f.shape[0]
        h_sum = None
        for k in range(self.n_samples):
            gens = [torch.Generator(device=self.device).manual_seed(self.patch_seed(self.meta, i, k)) for i in range(B)]
            z = self.from_x0(self.sample(net, self.coarse_fm, self.static_f, self.t_start, gens))
            if self.n_samples == 1:
                self.output = z
                break
            h = self.s_asinh * torch.sinh(z * self.sigma + self.mu)               # mean of the samples in metres
            h_sum = h if h_sum is None else h_sum + h
        if self.n_samples > 1:
            self.output = (torch.asinh(h_sum / self.n_samples / self.s_asinh) - self.mu) / self.sigma
        self.output_flood_logit = None
        if was_training:
            net.train()

    # ------------------------------------------------------------------ validation
    def _map_path(self, current_iter):
        """(save_dir, file name) of this patch's prediction -- the FMSRModel naming used by eval_flood.py."""
        m = self.meta
        flood_map_name = osp.splitext(osp.basename(self.first_scalar(m.get('coarse_path'), name='coarse_path')))[0]
        fname = (f"{self.first_scalar(m.get('var'), name='var')}_{self.first_scalar(m.get('scenario'), name='scenario')}_"
                 f"{self.first_scalar(m.get('t'), name='t')}_r{int(self.first_scalar(m.get('row'), name='row')):03d}_"
                 f"c{int(self.first_scalar(m.get('col'), name='col')):03d}_s{int(self.first_scalar(m.get('downscale'), name='downscale'))}"
                 f"_{current_iter}.npy")
        return osp.join(self.opt['path']['visualization'], flood_map_name), fname

    def _take_resumed(self, save_dir, fname):
        """Move a finished prediction of an archived run into this run's folder; None if absent or unusable."""
        src = osp.join(self.resume_vis_dir, osp.basename(save_dir), fname)
        if not osp.isfile(src):
            return None
        try:
            arr = np.load(src)
        except Exception:
            return None
        if arr.shape != tuple(self.fine_fm.shape) or not np.isfinite(arr).all():
            return None
        os.makedirs(save_dir, exist_ok=True)
        os.replace(src, osp.join(save_dir, fname))
        return torch.from_numpy(arr).to(self.device)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_flood_map):
        if self.is_train:   # train_flood_map_v2.py calls validation on rank 0 only -> no collectives
            if self.opt['rank'] == 0:
                self.nondist_validation(dataloader, current_iter, tb_logger, save_flood_map)
            return
        self.nondist_validation(dataloader, current_iter, tb_logger, save_flood_map,
                                shard=(self.opt['rank'], self.opt['world_size']))

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_flood_map, shard=(0, 1)):
        rank, world = shard
        ds = dataloader.dataset
        dataset_name = ds.opt['name']
        if self.opt['val'].get('logit_metrics') is not None:
            raise ValueError('[FloodDiffusionModel] val.logit_metrics must be absent (no flood-logit head)')
        metric_opts = self.opt['val'].get('metrics') or {}
        keys = list(metric_opts.keys())
        if keys:
            self._initialize_best_metric_results(dataset_name)
        sums = torch.zeros(len(keys), dtype=torch.float64)
        counts = torch.zeros(len(keys), dtype=torch.float64)
        stats_var, var_name = ds.stats_var, ds.target_var
        transform, h_asinh_q = str(ds.transform).lower(), ds.h_asinh_q
        if str(ds.norm).lower() != 'zscore' or var_name != 'h' or transform != 'asinh':
            raise ValueError('[FloodDiffusionModel] expects h with asinh + zscore')
        idx = list(range(rank, len(ds), world))
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=1, shuffle=False, num_workers=0)
        get_root_logger().info(f'[FloodDiffusionModel] {dataset_name}: {len(idx)} of {len(ds)} patches on rank {rank}/{world}, '
                               f't_start={self.t_start}, n_samples={self.n_samples}, seed_base={self.seed_base}')
        pbar = tqdm(total=len(idx), unit='patch') if (self.opt['val'].get('pbar', False) and rank == 0) else None
        if self.resume_vis_dir is not None and (self.is_train or not save_flood_map):
            raise ValueError('[FloodDiffusionModel] diffusion.resume_vis_dir is for test jobs that save maps only')
        n_resumed = 0
        for val_data in loader:
            self.feed_data(val_data)
            pred_phy = None
            if save_flood_map:
                save_dir, fname = self._map_path(current_iter)
                if self.resume_vis_dir is not None:
                    pred_phy = self._take_resumed(save_dir, fname)
                    n_resumed += pred_phy is not None
            with torch.no_grad():
                if pred_phy is None:
                    self.test()
                    pred_phy = destand_to_physical(self.output, var_name, stats_var, transform, h_asinh_q).clamp_min(0.0)
                simu_phy = destand_to_physical(self.fine_fm, var_name, stats_var, transform, h_asinh_q).clamp_min(0.0)
            eval_mask = self.static_f[:, -1:, ...]
            for j, name in enumerate(keys):
                opt_clean = {k: v for k, v in metric_opts[name].items() if k != 'better'}
                val = calculate_metric({'pred': pred_phy, 'target': simu_phy, 'mask': eval_mask}, opt_clean)
                v = val.detach().view(-1).double().cpu() if torch.is_tensor(val) else torch.tensor([float(val)], dtype=torch.float64)
                fin = torch.isfinite(v)
                sums[j] += v[fin].sum()
                counts[j] += fin.sum()
            if save_flood_map and not osp.isfile(osp.join(save_dir, fname)):
                os.makedirs(save_dir, exist_ok=True)
                np.save(osp.join(save_dir, fname), pred_phy.detach().cpu().numpy().astype(np.float32))
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        if self.resume_vis_dir is not None:
            get_root_logger().info(f'[FloodDiffusionModel] rank {rank}: {n_resumed} of {len(idx)} patches taken from '
                                   f'{self.resume_vis_dir}, {len(idx) - n_resumed} sampled')
        if world > 1:
            packed = torch.cat([sums, counts]).to(self.device)
            torch.distributed.all_reduce(packed)
            sums, counts = packed[:len(keys)].cpu(), packed[len(keys):].cpu()
        if keys:
            self.metric_results = {k: (float(sums[j] / counts[j]) if counts[j] > 0 else float('nan')) for j, k in enumerate(keys)}
            for k, v in self.metric_results.items():
                self._update_best_metric_result(dataset_name, k, v, current_iter)
            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
