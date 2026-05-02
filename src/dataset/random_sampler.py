from torch.utils.data import Sampler
import numpy as np


class FixedSizeRandomSampler(Sampler):
    """
    Randomly sample a fixed number of samples every epoch.
    """

    def __init__(self, dataset, num_samples, seed=0, shuffle=True, log=False):
        self.dataset = dataset
        self.num_samples = num_samples
        self.base_seed = seed
        self.shuffle = shuffle
        self.log = log

        self.indices = self._sample_once(epoch=0)

    def _sample_once(self, epoch):
        rng = np.random.default_rng(self.base_seed + epoch)

        dataset_size = len(self.dataset)

        # sample WITHOUT replacement
        if self.num_samples <= dataset_size:
            indices = rng.choice(dataset_size, size=self.num_samples, replace=False)
        else:
            # fallback if dataset smaller (rare)
            indices = rng.choice(dataset_size, size=self.num_samples, replace=True)

        if self.shuffle:
            rng.shuffle(indices)

        return indices

    def set_epoch(self, epoch):
        self.indices = self._sample_once(epoch)
        
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples