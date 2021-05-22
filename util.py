import math
import numpy as np
import os
import pandas as pd
import re
import torch

from datetime import datetime, timedelta
from pytz import timezone
from torch import nn
from typing import Dict, List, Tuple, Union

script_dir = os.path.abspath(os.path.dirname(__file__))

mc_to_dma = {"BaseDateTime": "# Timestamp", "LAT": "Latitude", "LON": "Longitude", "Status": "Navigational Status",
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
time_format = "%Y%m%d-%H%M%S"


def now() -> datetime:
    ger = timezone("Europe/Berlin")
    return datetime.now(ger)


def as_str(time: datetime) -> str:
    return datetime.strftime(time, time_format)


def as_float(time: datetime) -> float:
    return datetime.timestamp(time)


def as_datetime(time: str) -> datetime:
    return datetime.strptime(time, time_format)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_empty(df: pd.DataFrame) -> bool:
    return len(df.index) == 0


def verify_output_dir(output_dir: str, port_name: str) -> Dict[str, str]:
    output_dirs = {}
    for kind in ["data", "debug", "model", "plot", "log", "eval"]:
        curr_dir = os.path.join(output_dir, kind, port_name)
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
        output_dirs[kind] = curr_dir
    return output_dirs


def get_destination_file_name(name: any) -> str:
    dest = str(name)
    return re.sub(r'\W', '', dest).upper()


def encode_data_file(mmsi: float, data_dir: str = None, join: bool = False) -> str:
    mmsi_str = str(mmsi)
    file_name = f"data_{mmsi_str}"
    if data_dir is not None and os.path.exists(data_dir):
        route_files = list(filter(lambda file: file.startswith(file_name), os.listdir(data_dir)))
        if len(route_files) > 0:
            file_name = f"{file_name}_{len(route_files)}"
    file_name = f"{file_name}.npy"
    return os.path.join(data_dir, file_name) if join else file_name


def descale_mae(scaled_mae: float, as_str_duration: bool = False) -> Union[float, str]:
    data_range = data_ranges["time_scaled"]["max"] - data_ranges["time_scaled"]["min"]
    seconds = scaled_mae * data_range
    if as_str_duration:
        return str(timedelta(seconds=seconds))
    return seconds


def as_duration(mae: float, as_seconds: bool = False) -> Union[str, float]:
    seconds = mae * data_ranges["label"]["max"]
    if as_seconds:
        return seconds
    return str(timedelta(seconds=seconds))


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
    file = np.load(file_path, mmap_mode="r", allow_pickle=True)
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


def decode_pm_file(file_name: str) -> Tuple[str, str]:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1]


def encode_dataset_config_file(time: str, file_type: str) -> str:
    return f"dataset-config-{file_type}_{time}.pkl"


def decode_dataset_config_file(file_name: str) -> Tuple[str, str]:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1]


def decode_debug_file(file_name: str) -> Tuple[str, str, str]:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1], result[2]


def decode_log_file(file_name: str) -> Tuple[str, str, str]:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1], result[2]


def encode_loss_plot(port_name: str, time: str, file_type: str, scale: str = "linear") -> str:
    return f"loss-{file_type}_{port_name}_{time}_{scale}.png"


def decode_loss_plot(file_name: str) -> Tuple[str, str, str, str]:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1], result[2], result[3]


def encode_loss_file(port: str, time: str, file_type: str) -> str:
    return f"loss-{file_type}_{port}_{time}.npy"


def decode_loss_file(file_name: str) -> Tuple[str, str, str]:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1], result[2]


def encode_meta_file(port: str, time: str, file_type: str) -> str:
    return f"meta-{file_type}_{port}_{time}.npy"


def decode_meta_file(file_name: str) -> Tuple[str, str, str]:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1], result[2]


def encode_model_file(port_name: str, start_time: str, end_time: str, file_type: str,
                      is_checkpoint: bool = False, base_port_name: str = None) -> str:
    model_type = "checkpoint" if is_checkpoint else "model"
    if base_port_name is not None:
        port_name = f"{port_name}-{base_port_name}"
    return f"{model_type}-{file_type}_{port_name}_{start_time}_{end_time}.pt"


def decode_model_file(file_name: str, times_as_datetime: bool = False) -> Tuple[str, str, Union[str, datetime],
                                                                                Union[str, datetime], str]:
    """
    Decode a model file name
    :param file_name: Name of file to decode
    :param times_as_datetime: If True, return times as datetime types. Str otherwise
    :return: [file_type, port, start_time, end_time, "base_port_name" if transferred else ""]
    """
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    port_name = result[1]
    base_port_name = ""
    if "-" in port_name:  # transferred model: two ports are encoded
        port_names = port_name.split("-")
        port_name = port_names[0]
        base_port_name = port_names[1]
    start_time = result[2]
    end_time = result[3]
    if times_as_datetime:
        start_time = as_datetime(start_time)
        end_time = as_datetime(end_time)
    return result[0], port_name, start_time, end_time, base_port_name


def encode_checkpoint_file(port_name: str, start_time: str, end_time: str, file_type: str,
                           is_checkpoint: bool = False, base_port_name: str = None) -> str:
    # checkpoint_type = "checkpoint" if is_checkpoint else "model"
    # return f"{checkpoint_type}-{file_type}_{port_name}_{start_time}_{end_time}.tar"
    file_name = encode_model_file(port_name, start_time, end_time, file_type, is_checkpoint, base_port_name)
    file_name_no_ext = os.path.splitext(file_name)[0]
    return f"{file_name_no_ext}.tar"


def decode_checkpoint_file(file_name: str, times_as_datetime: bool = False) -> Tuple[str, str, Union[str, datetime],
                                                                                     Union[str, datetime], str]:
    return decode_model_file(file_name, times_as_datetime)


def encode_transfer_result_file(start_time: str, end_time: str) -> str:
    return f"transfer-result_{start_time}_{end_time}"


def decode_transfer_result_file(file_name: str) -> Tuple[str, str, str]:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1], result[2]


def encode_grouped_mae_plot(start_time: str, file_type: str) -> str:
    return f"grouped-mae-{file_type}_{start_time}.png"


def decode_grouped_mae_plot(file_name: str) -> Tuple[str, str]:
    file_no_ext = os.path.splitext(file_name)[0]
    result = file_no_ext.split("_")
    return result[0], result[1]


def num_total_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def num_total_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_latest_checkpoint_file_path(checkpoint_dir: str, checkpoint_type: str = "base") -> Tuple[str, str]:
    """
    Look for the latest checkpoint file of type 'checkpoint_type' within the given directory.
    :param checkpoint_dir: Directory to search for checkpoint file
    :param checkpoint_type: Type of checkpoint file: [base, transfer]
    :return: (path_to_checkpoint_file, type_of_returned_checkpoint)
    """
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Unable to find checkpoint file: No such directory: {checkpoint_dir}")
    if checkpoint_type not in ["base", "transfer"]:
        raise ValueError(f"Unknown checkpoint type '{checkpoint_type}' not in [base, transfer]")
    checkpoint_file_path, model_file_path = "", ""
    latest_start, latest_start_model = None, None
    for file in filter(lambda f: f.endswith(".tar") and (f.startswith(f"model-{checkpoint_type}") or
                                                         f.startswith(f"checkpoint-{checkpoint_type}")),
                       os.listdir(checkpoint_dir)):
        cp_type, _, start, _ = decode_checkpoint_file(file, True)
        if file.startswith("checkpoint") and (latest_start is None or latest_start < start):
            latest_start = start
            checkpoint_file_path = os.path.join(checkpoint_dir, file)
        # handle if latest checkpoint is current optimum --> saved as model-file
        elif latest_start_model is None or latest_start_model < start:
            latest_start_model = start
            model_file_path = os.path.join(checkpoint_dir, file)
    return (checkpoint_file_path, "checkpoint") if checkpoint_file_path != "" else (model_file_path, "model")


def debug_data(data_tensor: torch.Tensor, target_tensor: torch.Tensor, data_idx: int, loader, logger,
               log_prefix: str = "Training") -> None:
    # use fact that 'nan != nan' for NaN detection
    if data_tensor != data_tensor:
        logger.write(f"{log_prefix}: Detected NaN in data-tensor at index {data_idx}. Window width "
                     f"{loader.window_width}")
    if target_tensor != target_tensor:
        logger.write(f"{log_prefix}: Detected NaN in target-tensor at index {data_idx}. Window width "
                     f"{loader.window_width}")
