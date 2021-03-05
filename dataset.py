import numpy as np
import os
import joblib

import torch
from torch.utils.data import Dataset, DataLoader

script_dir = os.path.abspath(os.path.dirname(__file__))


class PortDataset(Dataset):
    def __init__(self, port_dir: str):
        self.port_dir = port_dir
        self.loader = joblib.load(os.path.join(script_dir, "mmsi_data_file_loader.pkl"))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index: int):
        x = np.load(self.data_files[index])
        x = torch.from_numpy(x)
        return
