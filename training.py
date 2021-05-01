import os

from typing import Union


class TrainingIteration:
    def __init__(self, start_time: str, end_time: str, data_path: str, log_path: str, model_path: str,
                 plot_path: str, debug_path: str = None):
        for kind in [("data", data_path), ("log", log_path), ("model", model_path), ("plot", plot_path)]:
            if not os.path.exists(kind[1]):
                self._raise_path_err(kind[0], kind[1])
        if debug_path is not None and not os.path.exists(debug_path):
            self._raise_path_err("debug", debug_path)
        self.start_time = start_time
        self.end_time = end_time
        self.data_path = data_path
        self.log_path = log_path
        self.model_path = model_path
        self.plot_path = plot_path
        self.debug_path = debug_path

    @staticmethod
    def _raise_path_err(kind: str, path: str) -> None:
        raise ValueError(f"Path for '{kind}' at '{path}' does not exist")
