import os

from typing import Union


class EvaluationResult:
    def __init__(self, mae: float, mse: float):
        self.mae = mae
        self.mse = mse


class TrainingIteration:
    def __init__(self, start_time: float, end_time: float, data_path: str, log_path: str, model_path: str,
                 plot_path: str, debug_path: str = None):
        self.start_time = start_time
        self.end_time = end_time
        self.data_path = data_path
        self.log_path = log_path
        self.model_path = model_path
        self.plot_path = plot_path
        self.debug_path = debug_path
        self.eval_result: Union[EvaluationResult, None] = None

    def __eq__(self, other: any) -> bool:
        if other is str:
            return str(self.start_time) == other or str(int(self.start_time)) == other
        if other is float:
            return self.start_time == other
        elif other is TrainingIteration:
            return self.start_time == other.start_time
        return False


def make_training_iteration(start_time: float, end_time: float, data_path: str, log_path: str, model_path: str,
                            plot_path: str, debug_path: str = None) -> TrainingIteration:
    for kind in [("data", data_path), ("log", log_path), ("model", model_path), ("plot", plot_path)]:
        if not os.path.exists(kind[1]):
            raise_path_err(kind[0], kind[1])
    if debug_path is not None and not os.path.exists(debug_path):
        raise_path_err("debug", debug_path)
    return TrainingIteration(start_time, end_time, data_path, log_path, model_path, plot_path, debug_path)


def raise_path_err(kind: str, path: str) -> None:
    raise ValueError(f"Path for '{kind}' at '{path}' does not exist")
