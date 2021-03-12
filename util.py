import numpy as np
import os
import pandas as pd
import re

script_dir = os.path.abspath(os.path.dirname(__file__))


def is_empty(df: pd.DataFrame) -> bool:
    return len(df.index) == 0


def get_destination_file_name(name: any) -> str:
    dest = str(name)
    return re.sub(r'\W', '', dest).upper()


def data_f(mmsi: float) -> str:
    mmsi_str = str(mmsi)
    return "data_{}.npy".format(mmsi_str)


def label_f(mmsi: float) -> str:
    mmsi_str = str(mmsi)
    return "label_{}".format(mmsi_str)


def scaler_f(scaler_type: str, mmsi: float) -> str:
    mmsi_str = str(mmsi)
    if scaler_type == "labeler":
        return "labeler_{}.pkl".format(mmsi_str)
    elif scaler_type == "normalize":
        return "normalize_scaler_{}.pkl".format(mmsi_str)
    elif scaler_type == "ship_type":
        return "ship_type_encoder_{}.pkl".format(mmsi_str)
    elif scaler_type == "nav_status":
        return "nav_status_encoder_{}.pkl".format(mmsi_str)
    else:
        raise ValueError("Unknown scaler type '{}'".format(scaler_type))


def npy_file_len(file_path: str) -> int:
    file = np.load(file_path, mmap_mode="r")
    return file.shape[0]


def write_to_console(message):
    filler = "*" * len(message)
    print("\t********** {} ***********".format(filler))
    print("\t********** {} ***********".format(message))
    print("\t********** {} ***********".format(filler))
