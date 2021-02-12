import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class AisDataset(Dataset):
    """Custom dataset for processed ais data"""

    def __init__(self, npy_file_paths):
        self.data_files = npy_file_paths

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index: int):
        x = np.load(self.data_files[index])
        x = torch.from_numpy(x)
        return
