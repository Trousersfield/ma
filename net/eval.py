import argparse
import os
import torch
from torch import nn

from dataset import MmsiDataFile, TrainingExampleLoader
from net.model import InceptionTimeModel
from plotter import plot_bars
from port import Port, PortManager
from training import EvaluationResult
from util import compute_mae, compute_mse

from typing import List, Tuple, Union

script_dir = os.path.abspath(os.path.dirname(__file__))

# Evaluationsmethoden: https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234


def eval_all(data_dir: str, output_dir: str, eval_latest: bool = True) -> None:
    """
    Entry point for evaluating all available ports
    :param data_dir: Directory to 'general' data without port-folder
    :param output_dir: Directory to output results
    :param eval_latest: Specify if only latest training iteration shall be evaluated. Default = True
    :return: None
    """
    pm = PortManager()
    pm.load()
    if len(pm.ports.keys()) < 1:
        raise ValueError("No port data available")

    maes = []
    port_names = []
    # evaluate all ports
    for _, port in pm.ports.items():
        eval_port(port, os.path.join(data_dir, port.name), eval_latest)

        train_times = port.trainings.keys()
        times_to_evaluate = sorted(train_times)
        if eval_latest:
            times_to_evaluate = [times_to_evaluate[-1]]

        for time in times_to_evaluate:
            maes.append(port.trainings[time].eval_result.mae)
            port_names.append(port.name)

    plot_bars(maes, port_names, title="MAE by port", y_label="MAE", path=os.path.join(output_dir, "eval", "eval.png"))


def eval_port(port: Union[str, Port], data_dir: str, eval_latest: bool = True) -> None:
    if port is str:
        pm = PortManager()
        pm.load()
        if len(pm.ports.keys()) < 1:
            raise ValueError("No port data available")
        port = pm.find_port(port)

    if len(port.trainings.keys()) > 0:
        print(f"Evaluating training iteration(s) for port {port.name}")
        train_times = port.trainings.keys()
        times_to_evaluate = sorted(train_times)
        if eval_latest:
            times_to_evaluate = [times_to_evaluate[-1]]
        for time in times_to_evaluate:
            er = eval_model(port.trainings[time].model_path, data_dir)
            port.trainings[time].eval_result = er
    else:
        print(f"No training iteration for port {port.name} available")


def eval_model(model_path: str, data_dir: str) -> EvaluationResult:
    loader = TrainingExampleLoader(data_dir)
    loader.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionTimeModel.load(model_path, device).to(device)
    model.eval()

    outputs = []
    labels = []
    test_indices = loader.shuffled_data_indices(kind="test")
    with torch.no_grad():
        for test_idx in test_indices:
            test_data, target = loader[test_idx]
            data_tensor = torch.Tensor(test_data).to(device)
            target_tensor = torch.Tensor(target).unsqueeze(-1).to(device)

            output = model(data_tensor)
            outputs.append(output)
            labels.append(target_tensor)

    # print(f"labels:\n{labels}")
    # print(f"outputs:\n{outputs}")
    mae = compute_mae(labels, outputs)
    mse = compute_mse(labels, outputs)
    print(f"mae: {mae}")
    print(f"mse: {mse}")

    return EvaluationResult(mae=mae, mse=mse)


def main(args) -> None:
    command = args.command
    if command == "evaluate":
        eval_model(args.model_path, args.data_dir)
    else:
        raise ValueError(f"Unknown command '{command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["evaluate"])
    parser.add_argument("--model_path", type=str, default=os.path.join(script_dir, os.pardir, "output", "model",
                                                                       "BREMERHAVEN_20210416-092158.pt"),
                        help="Path to model")
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, os.pardir, "data", "validate",
                                                                     "BREMERHAVEN"),
                        help="Directory to evaluation data")
    main(parser.parse_args())
