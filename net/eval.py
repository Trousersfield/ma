import argparse
import os
import torch

from torch import nn
from torch.utils.data import DataLoader

from dataset import RoutesDirectoryDataset
from net.model import InceptionTimeModel
from plotter import plot_bars
from port import Port, PortManager
from training import EvaluationResult
from util import descale_mae

from typing import List, Tuple, Union

script_dir = os.path.abspath(os.path.dirname(__file__))

# Evaluationsmethoden: https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234


def eval_all(data_dir: str, output_dir: str) -> None:
    """
    Entry point for evaluating all available ports
    :param data_dir: Directory to 'general' data without port-folder
    :param output_dir: Directory to output results
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
        eval_port(port, os.path.join(data_dir, port.name))
        maes.append(port.training.eval_result.mae)
        port_names.append(port.name)

    plot_bars(maes, port_names, title="MAE by port", y_label="MAE", path=os.path.join(output_dir, "eval", "eval.png"))


def eval_port(port: Union[str, Port], routes_dir: str) -> EvaluationResult:
    if port is str:
        pm = PortManager()
        pm.load()
        if len(pm.ports.keys()) < 1:
            raise ValueError("No port data available")
        port = pm.find_port(port)

    if port.training is not None:
        print(f"Evaluating training iteration(s) for port {port.name}")
        dataset_dir = os.path.join(routes_dir, port.name)
        return eval_model(port.training.model_path, dataset_dir)
    else:
        print(f"No training iteration for port {port.name} found")


def eval_model(model_path: str, dataset_dir: str) -> EvaluationResult:
    dataset = RoutesDirectoryDataset.load_from_config(os.path.join(dataset_dir, "default_dataset_config.pkl"))
    end_train = int(.8 * len(dataset))
    if not (len(dataset) - end_train) % 2 == 0 and end_train < len(dataset):
        end_train += 1
    end_validate = int(len(dataset) - ((len(dataset) - end_train) / 2))

    # use initialized dataset's config for consistent split
    eval_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, kind="eval", start=end_validate)

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=None, drop_last=False, pin_memory=True,
                                              num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionTimeModel.load(model_path, device).to(device)
    model.eval()

    x = []
    y = []
    criterion = nn.L1Loss(reduction="mean")
    print(f"len data: {len(eval_loader)}")
    with torch.no_grad():
        for eval_idx, (data, target) in enumerate(eval_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            x.append(output)
            y.append(target)

    # print(f"labels:\n{labels}")
    # print(f"outputs:\n{outputs}")
    outputs = torch.cat(x, dim=0)
    targets = torch.cat(y, dim=0)
    loss = criterion(outputs, targets)
    mae = loss.item()
    print(f"mae loss: {mae}")
    mae_eta = descale_mae(mae)
    print(f"descaled mae: {mae_eta}")
    print(f"formatted: {descale_mae(mae, as_duration=True)}")

    return EvaluationResult(mae=mae, mse=0)


def main(args) -> None:
    command = args.command
    if command == "eval":
        # eval_model(args.model_path, args.data_dir)
        eval_all(args.routes_dir, args.output_dir)
    elif command == "eval_port":
        eval_port(args.port_name, args.routes_dir)
    else:
        raise ValueError(f"Unknown command '{command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["eval", "eval_port"])
    parser.add_argument("--routes_dir", type=str, default=os.path.join(script_dir, os.pardir, "data", "routes"),
                        help="Directory to routes")
    parser.add_argument("--port_name", type=str, help="Port name to evaluate trained model")
    main(parser.parse_args())
