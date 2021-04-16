import math
import numpy as np
import os

from loader import MmsiDataFile
from port import Port

from typing import Dict, List


class PortData:
    def __init__(self, port: Port):
        self.port = port
        self.data_length = 0
        self.route_files: List[MmsiDataFile] = []


class DataSplitter:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.ports: Dict[str, PortData] = {}

    def register_route(self, file_path: str, port: Port, length: int):
        file = MmsiDataFile(file_path, length)
        if port.name in self.ports:
            self.ports[port.name].data_length += length
            self.ports[port.name].route_files.append(file)
        else:
            port_data = PortData(port)
            port_data.data_length += length
            port_data.route_files.append(file)
            self.ports[port.name] = port_data

    def split(self) -> None:
        for port_data in self.ports.values():
            rand_indices = np.arange(len(port_data.route_files))
            np.random.shuffle(rand_indices)
            num_val = (len(rand_indices)*0.1) / 2
            num_val = math.ceil(num_val) if num_val < 2. else int(num_val)
            num_train = len(rand_indices) - num_val

            # desired number of training data-points
            n_train_data = int(port_data.data_length*0.8)  # choose bigger threshold due to adding "one more"
            n_val_data = int((port_data.data_length - n_train_data) / 2)
            n_test_data = n_train_data - n_val_data

            num_categorized_data = 0
            i = 0
            while i < len(rand_indices) and num_categorized_data < n_train_data:
                num_categorized_data += port_data.route_files[rand_indices[i]].length
                i += 1

            # move route to validate-folder
            while i < len(rand_indices) and num_categorized_data < n_val_data:
                path = port_data.route_files[i].path
                os.replace(path, self.to_path(path, "validate", port_data.port.name))
                num_categorized_data += port_data.route_files[rand_indices[i]].length
                i += 1

            # move route to test-folder
            while i < len(rand_indices) and num_categorized_data < n_test_data:
                path = port_data.route_files[i].path
                os.replace(path, self.to_path(path, "test", port_data.port.name))
                num_categorized_data += port_data.route_files[rand_indices[i]].length
                i += 1

            # for i in np.arange(num_train, num_train + num_val):
            #     path = port_data.route_files[i].path
            #     os.replace(path, self.to_path(path, "validate", port_data.port.name))
            #
            # for i in np.arange(num_train + num_val, num_train + 2*num_val):
            #     path = port_data.route_files[i].path
            #     os.replace(path, self.to_path(path, "test", port_data.port.name))

    def to_path(self, path: str, target_folder: str, port_name) -> str:
        _, file_name = os.path.split(path)
        return os.path.join(self.data_dir, target_folder, port_name, file_name)
