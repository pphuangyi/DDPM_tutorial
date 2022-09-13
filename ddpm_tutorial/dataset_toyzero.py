"""
Load toyzero paired data
"""

import random
import numpy as np

import torch
from torch.utils.data import Dataset

class ToyzeroAligned(Dataset):
    """
    Load toyzero paired data
    """
    def __init__(self,
                 dataroot,
                 partition='train',
                 domain=None,
                 max_dataset_size=float('inf'),):
        super().__init__()

        assert partition in ['train', 'test']
        partition_folder = dataroot/partition

        if domain is None:
            fake_fnames = sorted(list((partition_folder/'fake').glob('*npz')))
            real_fnames = sorted(list((partition_folder/'real').glob('*npz')))
            fnames = fake_fnames + real_fnames
        elif domain == 'real':
            fnames = sorted(list((partition_folder/'real').glob('*npz')))
        elif domain == 'fake':
            fnames = sorted(list((partition_folder/'fake').glob('*npz')))
        else:
            raise ValueError("choose domain from [None, 'real', 'fake']")

        length = len(fnames)
        assert length > 0

        indices = list(length)
        random.shuffle(indices)
        if max_dataset_size < length:
            indices = indices[:max_dataset_size]

        self.fnames = [fnames[i] for i in indices]
        self.length = len(self.fnames)

    def __len__(self,):
        return self.length

    @staticmethod
    def _load(fname):
        with np.load(fname) as file_handle:
            image = file_handle[file_handle.files[0]]
        # The image is short type
        image = torch.tensor(image, dtype=torch.float)
        return image

    def __getitem__(self, index):
        return self._load(self.fnames[index % self.length])
