import argparse
import os
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RoutesDirectoryDataset
from net.model import InceptionTimeModel
from plotter import plot_bars, plot_grouped_maes
from port import Port, PortManager
from util import descale_mae, encode_dataset_config_file, mae_by_duration

from typing import Dict, List, Tuple, Union

script_dir = os.path.abspath(os.path.dirname(__file__))

# Evaluationsmethoden: https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234


class Evaluation:
    def __init__(self, port_name: str, mae: float) -> None:
        self.port_name = port_name
        self.mae = mae


class Evaluator:
    def __init__(self, output_dir: str, routes_dir: str, results: Dict[str, float] = None) -> None:
        self.output_dir = output_dir
        self.routes_dir = routes_dir
        self.model_dir = os.path.join(output_dir, "model")
        self.eval_dir = os.path.join(output_dir, "eval")
        self.path = os.path.join(self.eval_dir, "evaluator.tar")
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        self.pm = PortManager()
        self.pm.load()
        if len(self.pm.ports.keys()) < 1:
            raise ValueError("No port data available")
        self.results = results
        if results is None:
            self.results = {}

    def save(self):
        torch.save({
            "path": self.path,
            "output_dir": self.output_dir,
            "routes_dir": self.routes_dir,
            "results": self.results
        }, self.path)

    @staticmethod
    def load(eval_dir_or_path: str) -> 'Evaluator':
        path = eval_dir_or_path
        eval_dir, file = os.path.split(eval_dir_or_path)
        if not file.endswith(".tar"):
            path = os.path.join(path, "evaluator.tar")
        state_dict = torch.load(path)
        evaluator = Evaluator(
            output_dir=state_dict["output_dir"],
            routes_dir=state_dict["routes_dir"],
            results=state_dict["results"]
        )
        return evaluator

    @staticmethod
    def _encode_result_key(port_name: str, start_time: str) -> str:
        return f"{port_name}_{start_time}"

    @staticmethod
    def _decode_result_key(key: str) -> Tuple[str, str]:
        result = key.split("_")
        return result[0], result[1]

    def set_result(self, port: Port, start_time: str, mae: float) -> None:
        self.results[self._encode_result_key(port.name, start_time)] = mae

    def remove_result(self, port: Port) -> None:
        del self.results[port.name]

    def eval_port(self, port: Union[str, Port]) -> float:
        if isinstance(port, str):
            port = self.pm.find_port(port)

        trainings = self.pm.load_trainings(port, self.output_dir, self.routes_dir)
        if len(trainings) < 1:
            print(f"Skipping evaluation for port '{port.name}': No training found")
            return -1.

        dataset = RoutesDirectoryDataset.load_from_config(trainings[-1].dataset_config_path)
        end_train = int(.8 * len(dataset))
        if not (len(dataset) - end_train) % 2 == 0 and end_train < len(dataset):
            end_train += 1
        end_validate = int(len(dataset) - ((len(dataset) - end_train) / 2))

        # use initialized dataset's config for consistent split
        eval_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, start=end_validate)

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=None, drop_last=False, pin_memory=True,
                                                  num_workers=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = InceptionTimeModel.load(trainings[-1].model_path, device).to(device)
        model.eval()

        x = []
        y = []
        criterion = nn.L1Loss(reduction="mean")
        with torch.no_grad():
            for eval_idx, (data, target) in enumerate(tqdm(eval_loader, desc="Evaluation progress")):
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
        print(f"formatted: {descale_mae(mae, as_str_duration=True)}")

        mae_groups = mae_by_duration(outputs, targets)
        print(f"Mae by duration:\n{mae_groups}")
        plot_grouped_maes(mae_groups, os.path.join(self.eval_dir, port.name, "test.png"))

        self.set_result(port, start_time, mae)
        return mae

    def eval_all(self) -> None:
        """
        Entry point for evaluating all available ports
        :return: None
        """
        # evaluate all ports
        for port in self.pm.ports.values():
            self.eval_port(port)

        plot_bars(list(self.results.values()), list(self.results.keys()), title="MAE by port", y_label="MAE",
                  path=os.path.join(self.output_dir, "eval", "eval.png"))


def main(args) -> None:
    command = args.command
    if command == "init":
        e = Evaluator(args.output_dir, args.routes_dir)
        e.save()
    elif command == "eval":
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))
        e.eval_all()
    elif command == "eval_port":
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))
        e.eval_port(args.port_name)
    else:
        raise ValueError(f"Unknown command '{command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "eval", "eval_port"])
    parser.add_argument("--routes_dir", type=str, default=os.path.join(script_dir, os.pardir, "data", "routes"),
                        help="Directory to routes")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, os.pardir, "output"),
                        help="Directory to outputs")
    parser.add_argument("--port_name", type=str, help="Port name to evaluate trained model")
    parser.add_argument("--init_new", type=bool, default=False, help="Set 'True' to reset evaluation results")
    main(parser.parse_args())
