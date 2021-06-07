import os

from typing import List

from util import decode_loss_history_plot


class TrainingIteration:
    def __init__(self, start_time: str, model_path: str, data_paths: List[str], log_paths: List[str],
                 plot_paths: List[str], eval_paths: List[str], debug_paths: List[str],
                 source_port: str = None, config_uid: int = None) -> None:
        if plot_paths is None:
            plot_paths = []
        self.start_time = start_time
        self.source_port = source_port
        self.config_uid = config_uid
        self.model_path = model_path
        self.data_paths = data_paths
        self.log_paths = log_paths
        self.plot_paths = plot_paths
        self.debug_paths = debug_paths
        self.eval_paths = eval_paths

    def find_plot_path(self, kind: str = "linear") -> str:
        if kind not in ["linear", "log"]:
            raise ValueError(f"Unable to find plot: Unknown plot kind '{kind}' not in [log]")
        kinds = []
        for plot_path in self.plot_paths:
            _, file_name = os.path.split(plot_path)
            _, _, start, file_kind, source_port, config_uid = decode_loss_history_plot(file_name)
            kinds.append(file_kind)
            if kind == file_kind:
                return plot_path
        raise ValueError(f"No plot of kind 'kind' found. Got {kinds}")
