import argparse
import bisect
import joblib
import numpy as np
import os

from port import PortManager
from typing import List, Tuple
from util import npy_file_len

script_dir = os.path.abspath(os.path.dirname(__file__))


class MmsiDataFile:
    def __init__(self, path: str, length: int):
        self.path = path
        self.length = length

    def __len__(self):
        return self.length


class TrainingExampleLoader:
    def __init__(self, data_dir: str = "", train_ratio: float = 0.8, window_width: int = 10) -> None:
        if 0 < train_ratio < 1:
            raise ValueError(f"Train ratio of '{train_ratio}' out of range [0, 1]")
        if data_dir == "":
            self.data_dir = os.path.join(script_dir, "data", "routes", "ROSTOCK")
        else:
            self.data_dir = data_dir
        self.loader_file_name = "data_loader.pkl"
        self.loader_dir = os.path.join(self.data_dir, self.loader_file_name)
        self.window_width = window_width
        self.data_files: List[MmsiDataFile] = []
        self.access_matrix: np.ndarray = np.array([[-1, -1]])
        self.train_ratio: float = train_ratio
        self.train_indices: List[int] = []
        self.validate_indices: List[int] = []
        self.test_indices: List[int] = []

    def __len__(self):
        if self.access_matrix[0][0] == -1:
            return 0
        return self.access_matrix.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        self.check_initialized()
        if len(self) <= idx:
            raise ValueError("Training example with index {} out of range [0, {}]!".format(idx, len(self)-1))

        file_vec = self.access_matrix[idx]
        # print(f"file vec: {file_vec}")
        data_file = self.data_files[file_vec[0]]
        # print(f"data file: {data_file.path}")
        local_file_idx = file_vec[1]
        # print(f"local file idx: {local_file_idx}")

        data = np.load(os.path.join(data_file.path))
        # print("data shape for item {}: {}".format(idx, data.shape))

        # generate index-matrix to extract window from data
        index_vector = (np.expand_dims(np.arange(self.window_width), axis=0) + local_file_idx)
        # print("index-vector: \n", index_vector)

        # print(f"index_vector: {index_vector}")
        # print(f"data: {data}")

        window = data[index_vector][0]
        target = np.array([window[:, -1][len(window) - 1]])
        data = np.array([window[:, :-1]])
        # print(f"data: \n{}")
        # print(f"window: \n{window}")
        # print(f"target: \n{target}")
        return data, target

    def check_initialized(self) -> None:
        if len(self.data_files) == 0:
            raise ValueError("MMSI Data File loader either has no data or is not fit! Run fit() first!")

    def load(self) -> None:
        if os.path.exists(self.loader_dir):
            loader: TrainingExampleLoader = joblib.load(self.loader_dir)
            self.data_dir = loader.data_dir
            self.window_width = loader.window_width
            self.data_files = loader.data_files
            self.access_matrix = loader.access_matrix
            print("---- Data loader loaded! ----\nData dir: {}\nFiles: {} Training Examples: {}"
                  .format(self.data_dir, len(self.data_files), len(self)))
        else:
            print("No loader definition found at {}. Run fit() first.".format(self.loader_dir))

    def fit(self) -> None:
        for idx, data_file in enumerate(os.listdir(self.data_dir)):
            # print(f"idx: {idx}")
            if data_file == self.loader_file_name:
                continue
            data_file_path = os.path.join(self.data_dir, data_file)
            data_file = MmsiDataFile(data_file_path, npy_file_len(data_file_path))
            self.data_files.append(data_file)

            num_train_examples = len(data_file) - self.window_width + 1
            # print(f"num train examples for file {data_file.path}: {num_train_examples}")
            if num_train_examples < 1:  # skip data file if no train example can be extracted
                continue

            # self.data_files.append(data_file)
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

        # generate random training, validation and testing data-indices
        rand_indices = np.arange(len(self))
        np.random.shuffle(rand_indices)
        self.train_indices, remain = np.split(rand_indices, [self.train_ratio * len(self)])
        self.validate_indices, self.test_indices = remain.split(remain, 2)

        joblib.dump(self, self.loader_dir)
        print(f"Data Loader has been fit on directory {self.data_dir}")


def main(args) -> None:
    if args.command == "fit":
        print("Fitting Data Loader")
        pm = PortManager()
        pm.load()
        if len(pm.ports) == 0:
            raise ValueError("Port Manager has no ports. Is it initialized?")
        port = pm.find_port(args.port_name)
        loader = TrainingExampleLoader(os.path.join(args.data_dir, "routes", port.name))
        loader.fit()
    elif args.command == "load":
        print("Loading Data Loader!")
        loader = TrainingExampleLoader(args.data_dir)
        loader.load()
    elif args.command == "test":
        print(f"Testing Data Loader for port {args.port_name} at index {args.data_idx}")
        if args.port_name is None:
            raise ValueError("No port name found in 'args.port_name'. Specify a port name for testing.")
        pm = PortManager()
        pm.load()
        if len(pm.ports) == 0:
            raise LookupError("Unable to load ports! Make sure port manager is fit")
        port = pm.find_port(args.port_name)
        if port is None:
            raise ValueError(f"Unable to associate '{args.port_name}' with any port")
        # loader = TrainingExampleLoader(os.path.join(args.data_dir, "routes", "ROSTOCK"))
        loader = TrainingExampleLoader(os.path.join(args.data_dir, "routes", port.name))
        loader.load()
        print(f"DataLoader length: {len(loader)}")
        print(f"Window at pos {args.data_idx}:\n{loader[args.data_idx]}")
    elif args.command == "test_range":
        print("Testing Data Loader for directory 'routes'")
        pm = PortManager()
        pm.load()
        if len(pm.ports) == 0:
            raise LookupError("Unable to load ports! Make sure port manager is fit")
        for port in pm.ports.values():
            port_dir = os.path.join(args.data_dir, "routes", port.name)
            if os.path.exists(port_dir):
                print(f"Testing port {port.name}")
                loader = TrainingExampleLoader(port_dir)
                loader.load()
                print(f"DataLoader length: {len(loader)}")
                for i in range(len(loader)):
                    try:
                        _ = loader[i]
                    except IndexError:
                        print(f"Original Exception: {IndexError}")
                        print(f"Occurred at loader while accessing index: {i}")
                        break
            else:
                print(f"Directory {port_dir} does not exist for port {port.name}")
        print("Done!")
    else:
        raise ValueError("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual testing and validating Data Loader!")
    parser.add_argument("command", choices=["fit", "load", "test", "test_range"])
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, "data"),
                        help="Path to data files")
    parser.add_argument("--window_width", type=int, default=10, help="Sliding window width of training examples")
    parser.add_argument("--data_idx", type=int, default=0, help="Data index to retrieve (for testing only)")
    parser.add_argument("--port_name", type=str,
                        help="Name of port to fit data loader. Make sure Port Manager is initialized")
    main(parser.parse_args())
