import os
import numpy as np
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, destand_to_physical
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class FMSRModel(BaseModel):
    """Flood Model Super-Resolution Model"""

    def __init__(self, opt):
        super(FMSRModel, self).__init__(opt)
        self.opt = deepcopy(opt)
        self.net_g = build_network(self.opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.ema_decay = 0.0
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        if train_opt.get('pixel_opt'):
            self.use_wet_mask = train_opt['pixel_opt'].pop('use_wet_mask')
            self.loss_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            raise RuntimeError(f'[ERROR] pixel_opt (MaskL1Loss) is not configured.')

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.coarse_fm = data['coarse_flood_map'].to(self.device)
        self.static_f = data['fine_static_feature'].to(self.device)
        self.fine_fm = data['fine_flood_map'].to(self.device)
        if self.is_train and ('wet_mask' in data) and self.use_wet_mask:
            self.mask = data['wet_mask'].to(self.device)
        else:
            self.mask = self.static_f[:, -1:, ...]
        self.meta = data.get('meta', None)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # forward: net_g(coarse_flood_map, fine_static_feature) -> predicted_fine_flood_map
        self.output = self.net_g(self.coarse_fm, self.static_f)

        l_total = 0
        loss_dict = OrderedDict()

        l_pix = self.loss_pix(self.output, self.fine_fm, mask=self.mask)
        l_total += l_pix
        loss_dict['l_pix'] = l_pix

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.coarse_fm, self.static_f)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.coarse_fm, self.static_f)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_flood_map):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_flood_map)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_flood_map):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                # execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            # reset current metrics
            self.metric_results = {metric: 0 for metric in self.metric_results}

        stats_var = getattr(dataloader.dataset, 'stats_var', None)
        var_name = getattr(dataloader.dataset, 'target_var', 'h')
        if stats_var is None:
            raise RuntimeError(f'[ERROR] Validation needs dataset.stats_var for de-standardization.')

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='flood_map')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()
            if self.meta is not None and isinstance(self.meta, dict):
                flood_map_name = osp.splitext(osp.basename(self.meta.get('coarse_path')))[0]

            eval_mask = self.static_f[:, -1:, ...]

            with torch.no_grad():
                pred_phy = destand_to_physical(self.output, var_name, stats_var)
                simu_phy = destand_to_physical(self.fine_fm, var_name, stats_var)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    val = calculate_metric(
                        {'pred': pred_phy, 'target': simu_phy, 'mask': eval_mask},
                        opt_
                    )
                    if torch.is_tensor(val):
                        val = val.mean().item()
                    self.metric_results[name] += float(val)

            if save_flood_map:
                if self.meta is not None and isinstance(self.meta, dict):
                    save_dir = osp.join(self.opt['path']['visualization'], flood_map_name)
                    os.makedirs(save_dir, exist_ok=True)

                    scenario = self.meta.get('scenario')
                    t = self.meta.get('t')
                    row = self.meta.get('row')
                    col = self.meta.get('col')
                    var = self.meta.get('var')
                    downscale = self.meta.get('downscale')
                    save_flood_map_name = f'{var}_{scenario}_t{t}_r{row:03d}_c{col:03d}_s{downscale}_{current_iter}.npy'
                else:
                    raise RuntimeError(f'[ERROR] Save flood map needs val_data.meta information.')

                np.save(osp.join(save_dir, save_flood_map_name), pred_phy.detach().cpu().numpy().astype(np.float32))

            del self.coarse_fm, self.static_f, self.fine_fm, self.output
            torch.cuda.empty_cache()

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {flood_map_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            n = idx + 1
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= n
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.6f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.6f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if hasattr(self, 'coarse_fm'):
            out_dict['input_coarse_flood_map'] = self.coarse_fm.detach().cpu()
        if hasattr(self, 'static_f'):
            out_dict['input_fine_static_feature'] = self.static_f.detach().cpu()
        if hasattr(self, 'output'):
            out_dict['predicted_fine_flood_map'] = self.output.detach().cpu()
        if hasattr(self, 'fine_fm'):
            out_dict['simulated_fine_flood_map'] = self.fine_fm.detach().cpu()
        if hasattr(self, 'mask'):
            out_dict['mask_fine'] = self.mask.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)