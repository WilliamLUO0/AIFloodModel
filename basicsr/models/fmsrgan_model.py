"""
fmsrgan_model.py -- Paper 2 S8: adversarial training for the CBAM-GAN row (Shen et al. 2026), on top of FMSRModel.

What it adds to FMSRModel (paper2_S8/S8_ShenCBAMGAN.md §5.3(c), README §3):
  * net_d (`network_d`), its optimiser (`train.optim_d`, appended to self.optimizers so the warm-up and MultiStepLR
    of `train.scheduler` apply to it as to G -> lr_d tied to lr_g), saved as net_d_<iter>.pth and reloaded from
    `path.pretrain_network_d` (check_resume sets it on resume). EMA only on G.
  * iteration = ONE generator update. On iterations where current_iter % d_every == 0 the discriminator is updated
    first, on the same batch (G:D = 2:1 for d_every = 2; published order D then G).
  * D input: real = target; fake = M*G(x) + (1-M)*target (real map pasted outside the AOI, S8 ruling). real and fake
    go through D as ONE concatenated batch (D has only per-sample InstanceNorm, so this equals two forwards, and
    keeps one DDP forward per backward).
  * gradient clipping clip_g / clip_d (max-norm). During the G step D is frozen (requires_grad False) and called as
    the BARE module (not through DDP): no D gradient is needed there, and a DDP forward whose parameters receive no
    gradient would leave DDP's reducer waiting.
  * the loss (`train.shen_gan_opt`, ShenHydrologicalGANLoss) replaces pixel_opt / ordinal_opt, which must be absent.
    Loss values fed to the adaptive weights are averaged over GPUs first, so every rank uses the same scales.
  * non-finite G or D loss -> RuntimeError (the release substitutes a constant, which would hang DDP).
Evaluation configs use the plain FMSRModel (generator only).
"""
import torch
from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .fmsr_model import FMSRModel


@MODEL_REGISTRY.register()
class FMSRGANModel(FMSRModel):

    def init_training_settings(self):
        train_opt = self.opt['train']
        logger = get_root_logger()
        for k in ('pixel_opt', 'ordinal_opt', 'nonflood_opt', 'slight_opt', 'deep_opt', 'flood_bce_opt', 'prev_opt'):
            if train_opt.get(k):
                raise ValueError(f'[FMSRGANModel] train.{k} must be absent: the generator loss is train.shen_gan_opt')

        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True),
                              self.opt['path'].get('param_key_d', 'params'))
        self.net_g.train()
        self.net_d.train()

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay} (generator only)')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()
            n_buf = sum(1 for _ in self.net_g_ema.buffers())
            if n_buf:   # model_ema averages parameters only -> buffers (e.g. BN running stats) would go stale
                raise RuntimeError(f'[FMSRGANModel] generator has {n_buf} buffers; parameter-only EMA would leave them '
                                   f'stale (use GroupNorm/InstanceNorm, gate_norm: gn)')

        self.cri_gan = build_loss(train_opt['shen_gan_opt']).to(self.device)
        self.use_wet_mask = False
        self.d_every = int(train_opt.get('d_every', 2))
        self.clip_g = float(train_opt.get('clip_g', 1.0))
        self.clip_d = float(train_opt.get('clip_d', 0.5))
        if self.d_every < 1:
            raise ValueError(f'[FMSRGANModel] d_every must be >= 1, got {self.d_every}')
        self.last_d_log = None

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        super().setup_optimizers()                                    # optimizer_g = self.optimizers[0]
        train_opt = self.opt['train']
        lr_g, lr_d = float(train_opt['optim_g']['lr']), float(train_opt['optim_d']['lr'])
        if lr_d != lr_g:
            raise ValueError(f'[FMSRGANModel] optim_d.lr {lr_d} != optim_g.lr {lr_g} (S8 ruling: lr_d tied to lr_g)')
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, list(self.net_d.parameters()), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def _mean_over_ranks(self, t):
        t = t.detach().clone()
        if self.opt['dist']:
            torch.distributed.all_reduce(t)
            t /= self.opt['world_size']
        return t.item()

    @staticmethod
    def _check_finite(loss, what, current_iter):
        if not torch.isfinite(loss):
            raise RuntimeError(f'[FMSRGANModel] non-finite {what} loss at iter {current_iter}: {loss.item()}')

    def optimize_parameters(self, current_iter):
        M, y = self.mask, self.fine_fm
        loss_dict = OrderedDict()

        # ---- discriminator step (every d_every-th iteration, same batch, before the G step) ----
        if current_iter % self.d_every == 0:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = self.net_g(self.coarse_fm, self.static_f)[0]
                fake_d = M * fake + (1 - M) * y
            pred = self.net_d(torch.cat([y, fake_d], dim=0))
            real_pred, fake_pred = pred[:y.shape[0]], pred[y.shape[0]:]
            l_d, d_parts = self.cri_gan.d_loss(real_pred, fake_pred)
            self._check_finite(l_d, 'discriminator', current_iter)
            l_d.backward()
            torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), max_norm=self.clip_d)
            self.optimizer_d.step()
            self.cri_gan.record_d(self._mean_over_ranks(l_d))
            self.last_d_log = OrderedDict(l_d=l_d.detach(), **{k: v.detach() for k, v in d_parts.items()})

        # ---- generator step (every iteration) ----
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad(set_to_none=True)
        self.output, self.output_flood_logit = self.net_g(self.coarse_fm, self.static_f)
        fake_d = M * self.output + (1 - M) * y
        l_g, g_parts = self.cri_gan.g_loss(self.get_bare_model(self.net_d)(fake_d), y, self.output, M)
        self._check_finite(l_g, 'generator', current_iter)
        l_g.backward()
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=self.clip_g)
        self.optimizer_g.step()
        self.cri_gan.record_g(self._mean_over_ranks(l_g))

        loss_dict['l_g'] = l_g.detach()
        for k, v in g_parts.items():
            loss_dict[k] = v.detach()
        if self.last_d_log is not None:                               # most recent D step (keys stable in the log)
            loss_dict.update(self.last_d_log)
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        self.save_network(self.net_d, 'net_d', current_iter)          # before the .state file that marks the iter
        super().save(epoch, current_iter)
