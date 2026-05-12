import math
import torch
from torch.utils.data.sampler import Sampler

from basicsr.utils import get_root_logger


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class IntervalBalancedSampler(Sampler):
    """Distributed weighted sampler for interval-balanced flood-map training.

    This sampler supports:
      - DDP rank splitting
      - dataset_enlarge_ratio
      - set_epoch(epoch)
      - weighted sampling with replacement

    It expects the dataset to provide:
      dataset.get_interval_sample_weights(sampler_cfg)
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1, sampler_cfg=None):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.ratio = int(ratio)
        self.epoch = 0
        self.sampler_cfg = sampler_cfg or {}

        if not hasattr(dataset, 'get_interval_sample_weights'):
            raise RuntimeError(
                '[ERROR] Dataset does not implement get_interval_sample_weights(), '
                'which is required by IntervalBalancedSampler.'
            )

        self.weights = dataset.get_interval_sample_weights(self.sampler_cfg)
        self.weights = torch.as_tensor(self.weights, dtype=torch.double).cpu()

        if self.weights.ndim != 1:
            raise RuntimeError(f'[ERROR] weights should be 1D, got shape {self.weights.shape}')

        if len(self.weights) != len(self.dataset):
            raise RuntimeError(
                f'[ERROR] weights length {len(self.weights)} != dataset length {len(self.dataset)}'
            )

        if torch.any(~torch.isfinite(self.weights)):
            raise RuntimeError('[ERROR] weights contain NaN or Inf.')

        if torch.any(self.weights < 0):
            raise RuntimeError('[ERROR] weights contain negative values.')

        if torch.sum(self.weights) <= 0:
            raise RuntimeError('[ERROR] sum(weights) <= 0.')

        self.num_samples = math.ceil(len(self.dataset) * self.ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        logger = get_root_logger()
        if logger is not None:
            logger.info(
                '[IntervalBalancedSampler] '
                f'n={len(self.weights)}, '
                f'min={self.weights.min().item():.6f}, '
                f'mean={self.weights.mean().item():.6f}, '
                f'max={self.weights.max().item():.6f}, '
                f'sum={self.weights.sum().item():.6f}, '
                f'num_samples_per_rank={self.num_samples}, '
                f'total_size={self.total_size}'
            )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # Draw a global weighted sequence, then split it across ranks.
        # replacement=True means high-weight patches may appear multiple times
        # in one epoch.
        indices = torch.multinomial(
            self.weights,
            num_samples=self.total_size,
            replacement=True,
            generator=g
        ).tolist()

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch