import argparse
import bisect
import joblib
import numpy as np
import os

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
    def __init__(self, data_dir: str = "", window_width: int = 10) -> None:
        if data_dir == "":
            self.data_dir = os.path.join(script_dir, "data", "train", "COPENHAGEN")
        else:
            self.data_dir = data_dir
        self.loader_dir = os.path.join(self.data_dir, "data_loader.pkl")
        self.window_width = window_width
        self.data_files: List[MmsiDataFile] = []
        self.access_matrix: np.ndarray = np.array([[-1, -1]])

    def __len__(self):
        if self.access_matrix[0][0] == -1:
            return 0
        return self.access_matrix.shape[0]

    def __getitem__(self, idx: int):
        self.check_initialized()
        if len(self) < idx:
            raise ValueError("Training example with index {} out of range [0, {}]!".format(idx, len(self)))

        file_vec = self.access_matrix[idx]
        data_file = self.data_files[file_vec[0]]
        local_file_idx = file_vec[1]

        data = np.load(os.path.join(data_file.path))
        print("data shape: ", data.shape)

        # generate index-matrix to extract window from data
        index_vector = np.expand_dims(np.arange(self.window_width)) + local_file_idx
        print("index-vector: ", index_vector)

        training_example = data[index_vector]
        print("training_example {}".format(training_example))

        return training_example

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
        else:
            print("No loader definition found at {}. Run fit() first.".format(self.loader_dir))

    def fit(self) -> None:
        for idx, data_file in enumerate(os.listdir(self.data_dir)):
            data_file_path = os.path.join(self.data_dir, data_file)
            data_file = MmsiDataFile(data_file_path, npy_file_len(data_file_path))

            num_train_examples = len(data_file) - self.window_width + 1
            if num_train_examples < 1:  # skip data file if no train example can be extracted
                continue

            self.data_files.append(data_file)
            file_idx = np.empty(shape=(num_train_examples, 1), dtype=int)   # array of index of file in folder
            file_idx.fill(idx)
            local_file_indices = np.arange(num_train_examples).reshape(-1, 1)   # indices within the current file
            access_matrix = np.hstack((file_idx, local_file_indices))

            if self.access_matrix[0][0] == -1:
                self.access_matrix = access_matrix
            else:
                self.access_matrix = np.concatenate([self.access_matrix, access_matrix])
            # print("concatenated: \n", self.access_matrix)
        joblib.dump(self, self.loader_dir)


def main(args) -> None:
    if args.command == "init":
        print("Initializing Data Loader!")
        TrainingExampleLoader()
    elif args.command == "load":
        print("Loading Data Loader!")
        loader = TrainingExampleLoader()
        loader.load()
    elif args.command == "fit":
        print("Fitting Data Loader")
        loader = TrainingExampleLoader()
        loader.fit()
    else:
        raise ValueError("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual testing and validating Data Loader!")
    parser.add_argument("command", choices=["init", "load", "fit"])
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, "data", "train", "COPENGAHEN"),
                        help="Path to data files")
    parser.add_argument("--window_width", type=int, default=10, help="Sliding window width of training examples")
    main(parser.parse_args())
