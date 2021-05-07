import os

from typing import List

from util import decode_loss_plot


class TrainingIteration:
    def __init__(self, start_time: str, end_time: str, data_path: str, log_path: str, model_path: str,
                 plot_paths: List[str], dataset_config_path: str, debug_path: str = None):
        for kind in [("data", data_path), ("log", log_path), ("model", model_path), ("debug", debug_path)]:
            if kind[1] is not None and not os.path.exists(kind[1]):
                self._raise_path_err(kind[0], kind[1])
        if plot_paths is None:
            plot_paths = []
        for plot_path in plot_paths:
            if not os.path.exists(plot_path):
                self._raise_path_err("plot", plot_path)
        if debug_path is not None and not os.path.exists(debug_path):
            self._raise_path_err("debug", debug_path)
        self.start_time = start_time
        self.end_time = end_time
        self.data_path = data_path
        self.log_path = log_path
        self.model_path = model_path
        self.plot_paths = plot_paths
        self.debug_path = debug_path
        self.dataset_config_path = dataset_config_path

    @staticmethod
    def _raise_path_err(kind: str, path: str) -> None:
        raise ValueError(f"Path for '{kind}' at location '{path}' does not exist")

    def find_plot_path(self, kind: str = "linear") -> str:
        if kind not in ["linear", "log"]:
            raise ValueError(f"Unable to find plot: Unknown plot kind '{kind}' not in [log]")
        kinds = []
        for plot_path in self.plot_paths:
            _, file_name = os.path.split(plot_path)
            _, _, start, file_kind = decode_loss_plot(file_name)
            kinds.append(file_kind)
            if kind == file_kind:
                return plot_path
        raise ValueError(f"No plot of kind 'kind' found. Got {kinds}")
