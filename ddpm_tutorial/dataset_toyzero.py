"""
Load toyzero paired data
"""

import random
import numpy as np

import torch
from torch.utils.data import Dataset

class Toyzero(Dataset):
    """
    Load toyzero paired data
    """
    def __init__(self,
                 dataroot,
                 partition='train',
                 domain=None,
                 max_dataset_size=float('inf'),
                 clamp=100):
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

        indices = list(range(length))
        random.shuffle(indices)
        if max_dataset_size < length:
            indices = indices[:max_dataset_size]

        self.fnames = [fnames[i] for i in indices]
        self.length = len(self.fnames)

        self.clamp = clamp

    def __len__(self,):
        return self.length

    def _load(self, fname):
        with np.load(fname) as file_handle:
            image = file_handle[file_handle.files[0]]
        # The image is short type
        image = torch.tensor(image, dtype=torch.float)
        image = torch.clamp(image, min=-self.clamp, max=self.clamp)
        image /= self.clamp
        image = image.unsqueeze(0)
        return image

    def __getitem__(self, index):
        fname = self.fnames[index % self.length]
        return self._load(fname), str(fname)
