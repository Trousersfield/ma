import argparse
import csv
import os
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RoutesDirectoryDataset
from net.model import InceptionTimeModel
from plotter import plot_bars, plot_grouped_maes
from port import Port, PortManager
from util import encode_grouped_mae_plot, mae_by_duration, as_duration

from typing import Dict, List, Tuple, Union

script_dir = os.path.abspath(os.path.dirname(__file__))

# Evaluationsmethoden: https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234


class Evaluator:
    def __init__(self, output_dir: str, routes_dir: str,
                 mae_base: Dict[str, float] = None, mae_transfer: Dict[str, float] = None) -> None:
        self.output_dir = output_dir
        self.routes_dir = routes_dir
        self.data_dir = os.path.join(output_dir, "data")
        self.eval_dir = os.path.join(output_dir, "eval")
        self.model_dir = os.path.join(output_dir, "model")
        self.path = os.path.join(self.eval_dir, "evaluator.tar")
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        self.pm = PortManager()
        self.pm.load()
        if len(self.pm.ports.keys()) < 1:
            raise ValueError("No port data available")
        self.mae_base = mae_base if mae_base is not None else {}
        self.mae_transfer = mae_transfer if mae_transfer is not None else {}
        # self.results = results
        # if results is None:
        #     self.results = {}

    def save(self):
        torch.save({
            "path": self.path,
            "output_dir": self.output_dir,
            "routes_dir": self.routes_dir,
            "mae_base": self.mae_base,
            "mae_transfer": self.mae_transfer
        }, self.path)

    @staticmethod
    def load(eval_dir_or_path: str) -> 'Evaluator':
        path = eval_dir_or_path
        eval_dir, file = os.path.split(eval_dir_or_path)
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        if not file.endswith(".tar"):
            path = os.path.join(path, "evaluator.tar")
        state_dict = torch.load(path)
        evaluator = Evaluator(
            output_dir=state_dict["output_dir"],
            routes_dir=state_dict["routes_dir"],
            mae_base=state_dict["mae_base"],
            mae_transfer=state_dict["mae_transfer"]
        )
        return evaluator

    @staticmethod
    def _encode_base_key(port_name: str, start_time: str) -> str:
        return f"{port_name}_{start_time}"

    @staticmethod
    def _decode_base_key(key: str) -> Tuple[str, str]:
        result = key.split("_")
        return result[0], result[1]

    @staticmethod
    def _encode_transfer_key(base_port: str, transfer_port: str, start_time: str) -> str:
        return f"{base_port}_{transfer_port}_{start_time}"

    @staticmethod
    def _decode_transfer_key(key: str) -> Tuple[str, str, str]:
        result = key.split("_")
        return result[0], result[1], result[2]

    def _find_transfer_ports(self, base_port: str):

    def export(self) -> None:
        base_keys = sorted(self.mae_base.keys())
        with open("evaluation_results.csv", "w", newline="") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(["Base Port", "Start Time", "Base Port MAE", "Target Port", "Transfer Port MAE"])
            for base_key in base_keys:
                base_port, start_time = self._decode_base_key(base_key)
                transfer_key = self._encode_transfer_key(base_port, transfer_port)
                transfer_port, transfer_mae = "", .0
                if transfer_key in self.mae_transfer:
                    transfer_mae = self.mae_transfer[transfer_key]
                    # TODO: find all transfers for base port
                writer.writerow([port, start_time, self.mae_base[key], transfer_port, transfer_mae])

    def set_result(self, port: Port, start_time: str, mae: float, training_type: str) -> None:
        if training_type not in ["base", "transfer"]:
            raise ValueError(f"Unable to set result for training type '{training_type}'. Not in [base, transfer]")
        base_key = self._encode_base_key(port.name, start_time)
        transfer_key = self._encode_transfer_key(base_port=port.name, transfer_port=, start_time)
        self.mae_base[result_key] = mae
        self.mae_transfer[result_key] = mae

    def remove_result(self, port: Port, start_time: str) -> None:
        key = self._encode_result_key(port.name, start_time)
        if key in self.results:
            del self.results[key]
        else:
            print(f"No evaluation result found for port '{port.name}' and start time '{start_time}'")

    def eval_port(self, port: Union[str, Port], training_type: str) -> float:
        if isinstance(port, str):
            port = self.pm.find_port(port)

        trainings = self.pm.load_trainings(port, self.output_dir, self.routes_dir, training_type=training_type)
        if len(trainings) < 1:
            print(f"Skipping evaluation for port '{port.name}': No training found")
            return -1.

        training = trainings[-1]
        dataset = RoutesDirectoryDataset.load_from_config(training.dataset_config_path)
        end_train = int(.8 * len(dataset))
        if not (len(dataset) - end_train) % 2 == 0 and end_train < len(dataset):
            end_train += 1
        end_validate = int(len(dataset) - ((len(dataset) - end_train) / 2))

        # use initialized dataset's config for consistent split
        eval_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, start=end_validate)

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=None, drop_last=False, pin_memory=True,
                                                  num_workers=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = InceptionTimeModel.load(training.model_path, device).to(device)
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
        print(f"Mae loss: {mae} || {as_duration(mae)}")

        mae_groups = mae_by_duration(outputs, targets)
        print(f"Mae by duration:\n{mae_groups}")
        plot_path = os.path.join(self.eval_dir, port.name, encode_grouped_mae_plot(training.start_time,
                                                                                   file_type="base"))
        plot_grouped_maes(mae_groups, port_name=port.name, path=plot_path)

        self.set_result(port, training.start_time, mae)
        return mae

    def eval_all(self) -> None:
        """
        Entry point for evaluating all available ports
        :return: None
        """
        # evaluate all ports
        for port in self.pm.ports.values():
            [self.eval_port(port, training_type=t) for t in ["base", "transfer"]]

        # plot mae of each port in one plot
        plot_bars(list(self.results.values()), list(self.results.keys()), title="MAE by port", y_label="MAE",
                  path=os.path.join(self.output_dir, "eval", "ports-mae.png"))


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
        training_type = args.training_type
        training_types = []
        if training_type == "all":
            training_types = ["base", "transfer"]
        elif args.training_type not in ["base", "transfer"]:
            raise ValueError(f"Unknown parameter --training_type '{args.training_type}'. Not in [all, base, transfer]")
        else:
            training_types = [training_type]

        for t in training_types:
            e.eval_port(args.port_name, training_type=t)
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
    parser.add_argument("--training_type", type=str, choices=["all", "base", "transfer"])
    main(parser.parse_args())
