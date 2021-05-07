import argparse
import numpy as np
import os
import torch

from datetime import datetime
from torch.utils.data import Dataset
from typing import List, Tuple
from tqdm import tqdm

from port import Port, PortManager
from util import npy_file_len, encode_dataset_config_file, decode_dataset_config_file, as_datetime

script_dir = os.path.abspath(os.path.dirname(__file__))

# torch.set_printoptions(precision=10)


class RoutesDirectoryDataset(Dataset):
    def __init__(self, data_dir: str, start_time: str, start: int = 0, end: int = None, window_width: int = 100,
                 batch_size: int = 1, shuffled_data_indices: List[int] = None) -> None:
        """
        Index-based access for route files within a given directory
        :param data_dir: Directory to data
        :param start_time: Identification to associate with output data from training
        :param start: Index for first batch (inclusive)
        :param end: Index for last batch (exclusive)
        :param window_width: Number of data-points to retrieve within one batch-index
        :param batch_size: Number of windows within a single batch
        :param shuffled_data_indices: Given an already initialized dataset: Shuffled batch indices to keep track of
            training, validation and test data split
        """
        if end is not None and end < start:
            raise ValueError(f"Invalid data indices: start ({start}) < end ({end})")
        self.data_dir = data_dir
        self.config_file_name = encode_dataset_config_file(start_time)
        self.config_path = os.path.join(self.data_dir, self.config_file_name)
        self.start = start
        self.window_width = window_width
        self.batch_size = batch_size
        self.window_vector = np.expand_dims(np.arange(self.window_width), axis=0)
        self.route_file_paths = list(map(lambda file_name: os.path.join(data_dir, file_name),
                                         filter(lambda file_name: file_name.startswith("data_"), os.listdir(data_dir))))
        self.offsets = []
        self.size = sum(tqdm(map(self._count_data_length, self.route_file_paths), desc=f"Counting data in files"))
        if end is None:
            self.end = self.size
        else:
            assert end <= self.size
            self.end = end
        print(f"start: {self.start} end: {self.end}")
        assert self.start < self.end
        self.data = list(tqdm(map(self._read_file, self.route_file_paths), desc=f"Loading data"))
        self.access_matrix = self._generate_access_matrix()
        if shuffled_data_indices is None:
            self.shuffled_data_indices = self._shuffle_data_indices()
        else:
            self.shuffled_data_indices = shuffled_data_indices

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self) <= batch_idx:
            raise ValueError(f"Batch with index {batch_idx} out of range [0, {len(self) - 1}]!")
        batch_idx = self.start + batch_idx  # transform to start
        # print(f"batch_idx: {batch_idx}")
        batch_idx = self.shuffled_data_indices[batch_idx]  # use shuffled indices
        # print(f"shuffled: {batch_idx}")
        # start and end (exclusive) entry that belong to batch
        start = batch_idx * self.batch_size
        end = (batch_idx + 1) * self.batch_size
        # print(f"start: {start} stop: {end}")
        if end > len(self.access_matrix):
            # print(f"len access matrix: {len(self.access_matrix)}")
            end = len(self.access_matrix - 1)
        file_vectors = self.access_matrix[start:end, :]
        # print(f"file vectors:\n{file_vectors}")
        data_idx = file_vectors[0][0]
        file_start = file_vectors[0][1]
        # data_tensors: List[torch.Tensor] = []
        # target_tensors: List[torch.Tensor] = []
        # print(f"file vectors: {file_vectors}")
        #
        # start = datetime.now()
        # for file_vector in file_vectors:
        #     # generate index-based access vector to extract window from data with offset
        #     # print(f"offset: {self.offsets[file_vector[0]]}")
        #     index_vector = (self.window_vector + file_vector[1])
        #     # print(f"index vector:\n{index_vector}")
        #     # print(f"data size: {self.data[file_vector[0]].shape}")
        #     window = self.data[file_vector[0]][index_vector][0]
        #     # print(f"window:\n{window}")
        #     # print(f"data type: {window.dtype}")
        #     data_tensors.append(torch.from_numpy(window[:, :-1]).float())
        #     # print(f"target: {window[-1][window[-1].shape[0] - 1]}")
        #     target_tensors.append(torch.from_numpy(np.array([window[-1][window[-1].shape[0] - 1]])).float())
        # end = datetime.now()
        # print(f"old method took {end.microsecond - start.microsecond} microseconds")
        #
        # data = torch.stack(data_tensors, dim=0)
        # target = torch.stack(target_tensors, dim=0)
        # # print(f"window:\n{window}")
        # print(f"data:\n{data.shape}")
        # print(f"target:\n{target.shape}")

        data_tensors = []
        target_tensors = []

        # start = datetime.now()
        for i in range(self.batch_size):
            window = self.data[data_idx][file_start + i:file_start + self.window_width + i]
            data_tensors.append(torch.from_numpy(window[:, :-1]).float())
            # print(f"target: {window[-1][window[-1].shape[0] - 1]}")
            target_tensors.append(torch.from_numpy(np.array([window[-1][window[-1].shape[0] - 1]])).float())
        # end = datetime.now()
        # print(f"new method took {end.microsecond - start.microsecond} microseconds")

        data = torch.stack(data_tensors, dim=0)
        target = torch.stack(target_tensors, dim=0)
        # print(f"window:\n{window}")
        # print(f"data:\n{data.shape}")
        # print(f"target:\n{target.shape}")
        return data, target

    def _batches(self, data_idx, start) -> Tuple[torch.Tensor, torch.Tensor]:
        for i in range(self.batch_size):
            window = self.data[data_idx][start + i:start + self.window_width + i]
            yield torch.from_numpy(window[:, :-1]).float(), \
                torch.from_numpy(np.array([window[-1][window[-1].shape[0] - 1]])).float()

    def _count_data_length(self, file_path: str) -> int:
        data_len = npy_file_len(file_path) - self.window_width + 1
        # print(f"num raw rows: {data_len}")
        if data_len < 1:
            data_len = 0
        # trade-off: keep data at the end a series rather than at beginning
        num_batches, offset = divmod(data_len, self.batch_size)
        self.offsets.append(offset)
        return num_batches

    @staticmethod
    def _read_file(file_path: str) -> np.ndarray:
        data = np.load(os.path.join(file_path))
        return data

    @staticmethod
    def load_from_config(config_path: str, start: int = None,
                         end: int = None, new_data_dir: str = None) -> 'RoutesDirectoryDataset':
        if os.path.exists(config_path):
            file_name = os.path.split(config_path)[1]
            _, start_time = decode_dataset_config_file(file_name)
            # print(f"Loading dataset from config {config_path}")
            config = torch.load(config_path)
            dataset = RoutesDirectoryDataset(data_dir=config["data_dir"] if new_data_dir is None else new_data_dir,
                                             start_time=start_time,
                                             start=config["start"] if start is None else start,  # start-index
                                             end=config["end"] if end is None else end,  # end-index
                                             window_width=config["window_width"],
                                             batch_size=config["batch_size"],
                                             shuffled_data_indices=config["shuffled_data_indices"])
            return dataset
        else:
            print(f"No config found at {config_path}")

    def save_config(self) -> None:
        print(f"Saving dataset config at {self.config_path}")
        torch.save({
            "data_dir": self.data_dir,
            "start": self.start,
            "end": self.end,
            "window_width": self.window_width,
            "batch_size": self.batch_size,
            "shuffled_data_indices": self.shuffled_data_indices
        }, self.config_path)

    def _generate_access_matrix(self) -> np.ndarray:
        """
        Generate constant access by indices fit on self.data. access_matrix of n-rows with each row consisting of
        two columns:
        [[file_index, internal_data_index],
         [file_index, internal_data_index],
         ...]
        :return: 2-dim array of indices
        """
        access_matrix = np.array([[-1, -1]])
        for index, data in enumerate(tqdm(self.data, desc="Generating access matrix")):
            # data_len = data.shape[0] - self.window_width + 1
            data_len = data.shape[0] - self.window_width + 1  # - self.offsets[index]
            # print(f"file ({index}) source data len: {data.shape[0]} after window ({self.window_width}) "
            #       f"and offset ({self.offsets[index]}): {data_len}")
            if data_len < 1:  # skip data file if no train example can be extracted
                continue

            data_index = np.empty(shape=(data_len - self.offsets[index], 1), dtype=int)
            data_index.fill(index)
            internal_data_indices = np.arange(self.offsets[index], data_len).reshape(-1, 1)
            # print(f"internal data indices: {internal_data_indices}")
            data_matrix = np.hstack((data_index, internal_data_indices))

            if access_matrix[0][0] == -1:
                access_matrix = data_matrix
            else:
                access_matrix = np.concatenate([access_matrix, data_matrix])
        return access_matrix

    def _shuffle_data_indices(self) -> np.ndarray:
        shuffled = np.arange(self.size)
        np.random.shuffle(shuffled)
        return shuffled


def find_latest_dataset_config_path(dataset_dir: str) -> str:
    max_time, path = None, None
    for file in os.listdir(dataset_dir):
        if file.startswith("dataset-config"):
            _, start_time = decode_dataset_config_file(file)
            start_time = as_datetime(start_time)
            if max_time is None or max_time < start_time:
                max_time = start_time
                path = file
    return path if path is None else os.path.join(dataset_dir, path)


def test_dataset(dataset) -> None:
    config_file_name = os.path.split(dataset.config_path)[1]
    for batch_idx in range(len(dataset)):
        try:
            _, _ = dataset[batch_idx]
        except (IndexError, RuntimeError) as e:
            print(f"Original Exception: {e}")
            print(f"Occurred in Directory Dataset from config '{config_file_name}' while accessing index: {batch_idx}")
            break
    print(f"Route Dataset from config'{config_file_name}' checked!")


def main(args) -> None:
    if args.command == "generate":
        print("Generating Directory Dataset")
        pm = PortManager()
        pm.load()
        if len(pm.ports) == 0:
            raise ValueError("Port Manager has no ports. Is it initialized?")
        port = pm.find_port(args.port_name)
        data_dir = os.path.join(args.data_dir, "routes", port.name)
        batch_size = int(args.batch_size)
        dataset = RoutesDirectoryDataset(data_dir, batch_size=batch_size, start=0)
        dataset.save_config()
        end_train = int(.8 * len(dataset))
        if not (len(dataset) - end_train) % 2 == 0 and end_train < len(dataset):
            end_train += 1
        end_validate = int(len(dataset) - ((len(dataset) - end_train) / 2))

        train_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, start=0,
                                                                end=end_train)
        validate_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, start=end_train,
                                                                   end=end_validate)
        eval_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, start=end_validate)

        print(f"- - - - - Generated Datasets - - - - - -")
        print(f"Dataset: {len(dataset)}")
        print(f"Train: {len(train_dataset)}")
        print(f"Validate: {len(validate_dataset)}")
        print(f"Eval: {len(eval_dataset)}")

        data, target = dataset[args.data_idx]
        print(f"Dataset at pos {args.data_idx} has shape {data.shape}. Target shape: {target.shape}")
        data, target = train_dataset[args.data_idx]
        print(f"Train at pos {args.data_idx} has shape {data.shape}. Target shape: {target.shape}")
        data, target = validate_dataset[args.data_idx]
        print(f"Validate at pos {args.data_idx} has shape {data.shape}. Target shape: {target.shape}")
        data, target = eval_dataset[args.data_idx]
        print(f"Validate at pos {args.data_idx} has shape {data.shape}. Target shape: {target.shape}")
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
        dataset_dir = os.path.join(args.data_dir, "routes", port.name)
        dataset = RoutesDirectoryDataset.load_from_config(find_latest_dataset_config_path(dataset_dir))
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
        port = pm.find_port(args.port_name)
        if port is None:
            raise ValueError(f"Unable to associate '{args.port_name}' with any port")

        dataset_dir = os.path.join(args.data_dir, "routes", port.name)
        dataset = RoutesDirectoryDataset.load_from_config(find_latest_dataset_config_path(dataset_dir))
        end_train = int(.8 * len(dataset))
        if not (len(dataset) - end_train) % 2 == 0 and end_train < len(dataset):
            end_train += 1
        end_validate = int(len(dataset) - ((len(dataset) - end_train) / 2))

        train_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, start=0, end=end_train)
        validate_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, start=end_train,
                                                                   end=end_validate)
        eval_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, start=end_validate)

        test_dataset(train_dataset)
        test_dataset(validate_dataset)
        test_dataset(eval_dataset)
    elif args.command == "change_data_dir":
        print("Changing Directory Dataset Config's data directory")
        pm = PortManager()
        pm.load()
        if len(pm.ports) == 0:
            raise LookupError("Unable to load ports! Make sure port manager is fit")
        port = pm.find_port(args.port_name)
        if port is None:
            raise ValueError(f"Unable to associate '{args.port_name}' with any port")
        routes_dir = os.path.join(args.data_dir, "routes", port.name)
        config_path = os.path.join(routes_dir, args.config_file_name)
        if not os.path.exists(config_path):
            raise ValueError(f"No config file found at '{config_path}'")
        dataset = RoutesDirectoryDataset.load_from_config(config_path, new_data_dir=routes_dir)
        dataset.save_config()
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual testing and validating Directory Dataset!")
    parser.add_argument("command", choices=["generate", "test", "test_range", "change_data_dir"])
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, "data"),
                        help="Path to data files")
    parser.add_argument("--window_width", type=int, default=100, help="Sliding window width of training examples")
    parser.add_argument("--data_idx", type=int, default=0, help="Data index to retrieve (for testing only)")
    parser.add_argument("--port_name", type=str,
                        help="Name of port to fit Dataset Directory. Make sure Port Manager is initialized")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to extract from individual files")
    parser.add_argument("--config_file_name", type=str, help="File name including extension for dataset config")
    main(parser.parse_args())
