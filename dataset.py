import argparse
import bisect
import joblib
import numpy as np
import os

import torch

from port import Port, PortManager
from torch.utils.data import Dataset
from typing import List, Tuple
from util import npy_file_len

script_dir = os.path.abspath(os.path.dirname(__file__))


class MmsiDataFile:
    def __init__(self, path: str, length: int):
        self.path = path
        self.length = length

    def __len__(self):
        return self.length


class RoutesDirectoryDataset(Dataset):
    def __init__(self, data_dir: str, start: int = 0, end: int = None, window_width: int = 100) -> None:
        if end is not None and end < start:
            raise ValueError(f"Invalid data indices: start ({start}) < end ({end})")
        self.data_dir = data_dir
        self.config_file_name = "dataset_config.pkl"
        self.config_path = os.path.join(self.data_dir, self.config_file_name)
        self.start = start
        self.end = end
        self.window_width = window_width
        self.size = None
        self.data_files: List[MmsiDataFile] = []
        self.access_matrix: np.ndarray = np.array([[-1, -1]])
        self.shuffled_data_indices: List[int] = []
        self.load_config()

    def __len__(self):
        return 0 if self.end is None else self.end - self.start

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.check_initialized()
        if len(self) <= idx:
            raise ValueError(f"Training example with index {idx} out of range [0, {len(self) - 1}]!")

        shuffled_idx = self.shuffled_data_indices[idx]
        file_vec = self.access_matrix[shuffled_idx]
        # print(f"file vec: {file_vec}")
        data_file = self.data_files[file_vec[0]]
        # print(f"data file: {data_file.path}")
        local_file_idx = file_vec[1]
        # print(f"local file idx: {local_file_idx}")

        # TODO: alles in Arbeitsspeicher laden
        data = np.load(os.path.join(data_file.path))
        # print("data shape for item {}: {}".format(idx, data.shape))

        # generate index-matrix to extract window from data
        index_vector = (np.expand_dims(np.arange(self.window_width), axis=0) + local_file_idx)
        # print("index-vector: \n", index_vector)

        # print(f"index_vector: {index_vector}")
        # print(f"data: {data}")

        window = data[index_vector][0]
        target = torch.from_numpy(np.array(window[:, -1][len(window) - 1])).float()
        data = torch.from_numpy(np.array(window[:, :-1])).float()
        # print(f"window:\n{window}")
        # print(f"data:\n{data}")
        # print(f"target:\n{target}")
        return data, target

    def check_initialized(self) -> None:
        if len(self.data_files) == 0:
            raise ValueError("Route directory dataset either has no data or is not fit! Run fit() first!")

    def load_config(self) -> None:
        if os.path.exists(self.config_path):
            rdd: RoutesDirectoryDataset = joblib.load(self.config_path)
            self.data_dir = rdd.data_dir
            # self.start = rdd.start
            # self.end = rdd.end
            self.window_width = rdd.window_width
            self.size = rdd.size
            self.data_files = rdd.data_files
            self.access_matrix = rdd.access_matrix
            self.shuffled_data_indices = rdd.shuffled_data_indices
            if self.end is None:
                self.end = self.size
            assert self.end <= self.size
            print(f"---- Route directory dataset loaded! ----\n"
                  f"Data dir: {self.data_dir}\n"
                  f"Start: {self.start} End: {self.end} Files: {len(self.data_files)} Training Examples: {len(self)}")
        else:
            print(f"No config found at {self.config_path}. "
                  f"Run 'fit()' to initialize Dataset for directory {self.data_dir}")

    def fit(self) -> None:
        for idx, file in enumerate(os.listdir(self.data_dir)):
            # print(f"idx: {idx}")
            if not file.startswith("data_"):
                continue
            data_file_path = os.path.join(self.data_dir, file)
            data_file = MmsiDataFile(data_file_path, npy_file_len(data_file_path))
            self.data_files.append(data_file)

            num_train_examples = len(data_file) - self.window_width + 1
            if num_train_examples < 1:  # skip data file if no train example can be extracted
                continue

            file_idx = np.empty(shape=(num_train_examples, 1), dtype=int)   # array of index of file in folder
            file_idx.fill(idx)
            local_file_indices = np.arange(num_train_examples).reshape(-1, 1)   # indices within the current file
            access_matrix = np.hstack((file_idx, local_file_indices))

            if self.access_matrix[0][0] == -1:
                self.access_matrix = access_matrix
            else:
                self.access_matrix = np.concatenate([self.access_matrix, access_matrix])
            # print("concatenated: \n", self.access_matrix)
            # print("shape: ", self.access_matrix.shape)

        self.size = self.access_matrix.shape[0]
        # generate random training, validation and testing data-indices
        self.shuffled_data_indices = np.arange(self.size)
        np.random.shuffle(self.shuffled_data_indices)
        joblib.dump(self, self.config_path)
        print(f"Dataset config has been fit on directory {self.data_dir}")


def main(args) -> None:
    if args.command == "fit":
        print("Fitting Directory Dataset")
        pm = PortManager()
        pm.load()
        if len(pm.ports) == 0:
            raise ValueError("Port Manager has no ports. Is it initialized?")
        port = pm.find_port(args.port_name)
        dataset = RoutesDirectoryDataset(os.path.join(args.data_dir, "routes", port.name))
        dataset.fit()
    elif args.command == "load":
        pm = PortManager()
        pm.load()
        if len(pm.ports) == 0:
            raise ValueError("Port Manager has no ports. Is it initialized?")
        port = pm.find_port(args.port_name)
        dataset = RoutesDirectoryDataset(os.path.join(args.data_dir, "routes", port.name))
    elif args.command == "test":
        print(f"Testing Directory Dataset for port {args.port_name} at index {args.data_idx}")
        if args.port_name is None:
            raise ValueError("No port name found in 'args.port_name'. Specify a port name for testing.")
        pm = PortManager()
        pm.load()
        if len(pm.ports) == 0:
            raise LookupError("Unable to load ports! Make sure port manager is fit")
        port = pm.find_port(args.port_name)
        if port is None:
            raise ValueError(f"Unable to associate '{args.port_name}' with any port")
        dataset = RoutesDirectoryDataset(os.path.join(args.data_dir, "routes", port.name))
        print(f"Dataset length: {len(dataset)}")
        data, target = dataset[args.data_idx]
        print(f"Data at pos {args.data_idx} of shape {data.shape}:\n{data}")
        print(f"Target at pos {args.data_idx} of shape {target.shape}:\n{target}")
    elif args.command == "test_range":
        print("Testing Directory Dataset in directory 'routes'")
        pm = PortManager()
        pm.load()
        if len(pm.ports) == 0:
            raise LookupError("Unable to load ports! Make sure port manager is fit")
        for port in pm.ports.values():
            port_dir = os.path.join(args.data_dir, "routes", port.name)
            if os.path.exists(port_dir):
                print(f"Testing port {port.name}")
                dataset = RoutesDirectoryDataset(port_dir)
                print(f"Dataset length: {len(dataset)}")
                for i in range(len(dataset)):
                    try:
                        _ = dataset[i]
                    except IndexError:
                        print(f"Original Exception: {IndexError}")
                        print(f"Occurred in Directory Dataset while accessing index: {i}")
                        break
            else:
                print(f"Directory {port_dir} does not exist for port {port.name}")
        print("Done!")
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual testing and validating Directory Dataset!")
    parser.add_argument("command", choices=["fit", "load", "test", "test_range"])
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, "data"),
                        help="Path to data files")
    parser.add_argument("--window_width", type=int, default=100, help="Sliding window width of training examples")
    parser.add_argument("--data_idx", type=int, default=0, help="Data index to retrieve (for testing only)")
    parser.add_argument("--port_name", type=str,
                        help="Name of port to fit Dataset Directory. Make sure Port Manager is initialized")
    main(parser.parse_args())
