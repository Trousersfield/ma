import numpy as np
import os
import pandas as pd
import re
import torch

from typing import List

script_dir = os.path.abspath(os.path.dirname(__file__))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_empty(df: pd.DataFrame) -> bool:
    return len(df.index) == 0


def get_destination_file_name(name: any) -> str:
    dest = str(name)
    return re.sub(r'\W', '', dest).upper()


def data_file(mmsi: float) -> str:
    mmsi_str = str(mmsi)
    return f"data_{mmsi_str}.npy"


def obj_file(file_type: str, mmsi: float) -> str:
    mmsi_str = str(mmsi)
    if file_type == "labeler":
        return f"labeler_{mmsi_str}.pkl"
    elif file_type == "train_scaler":
        return f"train_scaler_{mmsi_str}.pkl"
    elif file_type == "test_scaler":
        return f"test_scaler_{mmsi_str}.pkl"
    elif file_type == "validate_scaler":
        return f"validate_scaler{mmsi_str}.pkl"
    elif file_type == "ship_type":
        return f"ship_type_encoder_{mmsi_str}.pkl"
    elif file_type == "nav_status":
        return f"nav_status_encoder_{mmsi_str}.pkl"
    elif file_type == "data_unlabeled":
        return f"data_df_{mmsi_str}.pkl"
    else:
        raise ValueError(f"Unknown file type '{file_type}'")


def npy_file_len(file_path: str) -> int:
    file = np.load(file_path, mmap_mode="r")
    return file.shape[0]


def write_to_console(message):
    filler = "*" * len(message)
    print(f"\t********** {filler} ***********")
    print(f"\t********** {message} ***********")
    print(f"\t********** {filler} ***********")


def compute_mae(y_true: List[torch.Tensor], y_pred: List[torch.Tensor]) -> float:
    assert len(y_true) == len(y_pred)
    err = 0
    for i in range(len(y_true)):
        # TODO: .sum() might not be necessary
        err += abs(y_true[i].sum().to(torch.float32) - y_pred[i].sum().to(torch.float32))
    mae = err / len(y_true)
    return mae


def compute_mse(y_true: List[torch.Tensor], y_pred: List[torch.Tensor]) -> float:
    assert len(y_true) == len(y_pred)
    err = 0
    for i in range(len(y_true)):
        err += (y_true[i].sum().to(torch.float32) - y_pred[i].sum().to(torch.float32))**2
    mse = err / len(y_true)
    return mse
