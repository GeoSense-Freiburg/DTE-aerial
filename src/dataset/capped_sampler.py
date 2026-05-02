from torch.utils.data import Sampler
import numpy as np


class CappedSampler(Sampler):
    """
    Uniform capped sampling per stratification group.
    """

    def __init__(
        self,
        dataframe,
        stratify_col,
        shuffle=True,
        seed=0,
        log=False,
        max_samples_per_group=None
    ):
        self.strat_col = stratify_col
        self.shuffle = shuffle
        self.base_seed = seed
        self.log = log

        # Precompute group → indices
        df = dataframe.copy()
        df[self.strat_col] = df[self.strat_col].astype(str)
        self.df = df
        
        
        self.group_to_indices = (
            self.df.reset_index()
                   .groupby(self.strat_col)["index"]
                   .apply(list)
                   .to_dict()
        )
        # --------------------------------------------------
        # 🔹 Dynamic cap = smallest group
        # --------------------------------------------------
        group_sizes = {
            g: len(idxs) for g, idxs in self.group_to_indices.items()
        }
        
        #print all groups and their sizes
        print("Group sizes:")
        for g, size in group_sizes.items():
            print(f"  Group '{g}': {size} samples")

        
        if len(group_sizes) == 0:
            raise ValueError("No groups found for sampling.")
        
        self.max_samples_per_group = max_samples_per_group if max_samples_per_group is not None else min(group_sizes.values())
                
        
        self.group_caps = {}
        for g in self.group_to_indices.keys():
            self.group_caps[g] = self.max_samples_per_group


        self.n_groups = len(self.group_to_indices)

        self.indices = self._sample_once(epoch=0)
        
        """
        Uniform capped sampling per stratification group.
        Logs indices each epoch so you can verify resampling.
        """

        # if self.log:
        #     print(f"[Sampler] Epoch 0 → {len(self.indices)} samples")
        #     print(self.indices[:50], "...")
        self.n_groups = len(self.group_to_indices)
        print(f'num of groups:{self.n_groups}')

    def _sample_once(self, epoch):
        rng = np.random.default_rng(self.base_seed + epoch)

        epoch_indices = []

        for group, idx_list in self.group_to_indices.items():

            idx_list = np.array(idx_list)

            cap = self.group_caps[group]

            if len(idx_list) >= cap:
                chosen = rng.choice(idx_list, size=cap, replace=False)
            else:
                chosen = idx_list

            epoch_indices.append(chosen)

        epoch_indices = np.concatenate(epoch_indices)

        if self.shuffle:
            rng.shuffle(epoch_indices)

        return epoch_indices

    def set_epoch(self, epoch):
        """Call at each new epoch to trigger resampling + logging."""
        self.indices = self._sample_once(epoch)


    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.n_groups * self.max_samples_per_group
