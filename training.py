import os

from typing import List

from util import decode_loss_history_plot


class TrainingIteration:
    def __init__(self, start_time: str, config_uid: int, data_path: str, log_path: str, model_path: List[str],
                 plot_paths: List[str], eval_paths: List[str], debug_path: str = None) -> None:
        if plot_paths is None:
            plot_paths = []
        self.start_time = start_time
        self.config_uid = config_uid
        self.data_path = data_path
        self.log_path = log_path
        self.model_path = model_path
        self.plot_paths = plot_paths
        self.debug_path = debug_path
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
