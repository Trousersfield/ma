import numpy as np
import os
import pandas as pd
import re
import torch

from datetime import datetime
from pytz import timezone
from typing import List, Tuple

script_dir = os.path.abspath(os.path.dirname(__file__))

mc_to_dk = {"BaseDateTime": "# Timestamp", "LAT": "Latitude", "LON": "Longitude", "Status": "Navigational Status",
            "Draft": "Draught", "VesselType": "Ship Type"}
data_ranges = {
    "Latitude": {"min": -90., "max": 90.},
    "Longitude": {"min": -180., "max": 180.},
    "SOG": {"min": 0., "max": 110},                  # max = 102 from (1)
    "COG": {"min": 0., "max": 359.9},                # (1) max from data: 359.9
    "Heading": {"min": 0., "max": 511.},             # (1)
    "Width": {"min": 0., "max": 80},                 # {3)
    "Length": {"min": 0., "max": 500.},              # (2)
    "Draught": {"min": 0., "max": 40.},              # assume some value that seems high enough
    "time_scaled": {"min": 0., "max": 31622400.},    # max value for seconds per year is dependant on year
    "label": {"min": 0., "max": 31622400.},          # same range as time within a year
    "Cargo": {"min": 0., "max": 1.},                 # One-Hot-Encoded features from this point
    "Dredging": {"min": 0., "max": 1.},
    "HSC": {"min": 0., "max": 1.},
    "Fishing": {"min": 0., "max": 1.},
    "Passenger": {"min": 0., "max": 1.},
    "Pleasure": {"min": 0., "max": 1.},
    "Reserved": {"min": 0., "max": 1.},
    "Sailing": {"min": 0., "max": 1.},
    "Tanker": {"min": 0., "max": 1.},
    "Towing": {"min": 0., "max": 1.},
    "Tug": {"min": 0., "max": 1.},
    "Other ship type": {"min": 0., "max": 1.},
    "Default ship type": {"min": 0., "max": 1.},
    "Under way using engine": {"min": 0., "max": 1.},
    "At anchor": {"min": 0., "max": 1.},
    "Not under command": {"min": 0., "max": 1.},
    "Restricted maneuverability": {"min": 0., "max": 1.},
    "Constrained by her draught": {"min": 0., "max": 1.},
    "Moored": {"min": 0., "max": 1.},
    "Aground": {"min": 0., "max": 1.},
    "Engaged in fishing": {"min": 0., "max": 1.},
    "Under way sailing": {"min": 0., "max": 1.},
    "Reserved for future amendment of Navigational Status for HSC": {"min": 0., "max": 1.},
    "Reserved for future amendment of Navigational Status for WIG": {"min": 0., "max": 1.},
    "Reserved fof future use": {"min": 0., "max": 1.},
    "AIS-SART is active": {"min": 0., "max": 1.},
    "Other navigational status": {"min": 0., "max": 1.},
    "Default navigational status": {"min": 0., "max": 1.}
}
# year with 365 days: 31536000
# year with 366 days: 31622400
# sources:
# (1) https://www.sostechnic.com/epirbs/ais/aisinformationenglish/index.php
# assume the biggest vessel in the world in service (+ some more):
# (2) https://en.wikipedia.org/wiki/List_of_longest_ships
# (3) https://gcaptain.com/emma-maersk-engine/worlds-largest-tanker-knock-nevis/

categorical_values = {
    "ship type": [
        "Cargo",
        "Dredging",
        "HSC",
        "Fishing",
        "Passenger",
        "Pleasure",
        "Reserved",
        "Sailing",
        "Tanker",
        "Towing",
        "Tug"
    ],
    "navigational status": [
        "Under way using engine",
        "At anchor",
        "Not under command",
        "Restricted maneuverability",
        "Constrained by her draught",
        "Moored",
        "Aground",
        "Engaged in fishing",
        "Under way sailing",
        "Reserved for future amendment of Navigational Status for HSC",
        "Reserved for future amendment of Navigational Status for WIG",
        "Reserved fof future use",
        "AIS-SART is active"
    ]
}


def now() -> datetime:
    ger = timezone("Europe/Berlin")
    return datetime.now(ger)


def as_str(time: datetime) -> str:
    return datetime.strftime(time, "%Y%m%d-%H%M%S")


def as_float(time: datetime) -> float:
    return datetime.timestamp(time)


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


def obj_file(file_type: str, mmsi: float, suffix: str = None) -> str:
    file_str = str(mmsi)
    if suffix is not None:
        file_str += f"-{suffix}"
    if file_type == "labeler":
        return f"labeler-{file_str}.pkl"
    elif file_type == "train_scaler":
        return f"train_scaler-{file_str}.pkl"
    elif file_type == "test_scaler":
        return f"test_scaler-{file_str}.pkl"
    elif file_type == "validate_scaler":
        return f"validate_scaler-{file_str}.pkl"
    elif file_type == "ship_type":
        return f"ship_type_encoder-{file_str}.pkl"
    elif file_type == "nav_status":
        return f"nav_status_encoder-{file_str}.pkl"
    elif file_type == "data_unlabeled":
        return f"u_data-{file_str}.pkl"
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


def encode_pm_file(time: str) -> str:
    return f"pm-ports_{time}.pkl"


def decode_pm_file(file_name: str) -> str:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1]


def encode_loss_plot(port_name: str, time: str) -> str:
    return f"loss_{port_name}_{time}.png"


def decode_loss_plot(file_name: str) -> Tuple[str, str, str]:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1], result[2]


def encode_loss_file(port: str, time: str) -> str:
    return f"loss_{port}_{time}_loss.npy"


def decode_loss_file(file_name: str) -> Tuple[str, str, str]:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1], result[2]


def encode_model_file(port_name: str, start_time: str, end_time: str, is_checkpoint: bool = False) -> str:
    model_type = "checkpoint" if is_checkpoint else "model"
    return f"{model_type}_{port_name}_{start_time}_{end_time}.pt"


def decode_model_file(file_name: str) -> Tuple[str, str, str, str]:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1], result[2], result[3]


def num_total_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def num_total_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def debug_data(data_tensor: torch.Tensor, target_tensor: torch.Tensor, data_idx: int, loader, logger,
               log_prefix: str = "Training") -> None:
    # use fact that 'nan != nan' for NaN detection
    if data_tensor != data_tensor:
        logger.write(f"{log_prefix}: Detected NaN in data-tensor at index {data_idx}. Window width "
                     f"{loader.window_width}")
    if target_tensor != target_tensor:
        logger.write(f"{log_prefix}: Detected NaN in target-tensor at index {data_idx}. Window width "
                     f"{loader.window_width}")
