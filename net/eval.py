import argparse
import csv
import os
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RoutesDirectoryDataset
from net.model import InceptionTimeModel
from plotter import plot_ports_by_mae, plot_grouped_maes, plot_transfer_effect, plot_transfer_effects
from port import Port, PortManager
from training import TrainingIteration
from util import encode_grouped_mae_plot, as_duration, data_ranges, decode_model_file

from typing import Dict, List, Tuple, Union

script_dir = os.path.abspath(os.path.dirname(__file__))


class Evaluator:
    def __init__(self, output_dir: str, routes_dir: str, mae_base: Dict[str, float] = None,
                 mae_transfer: Dict[str, float] = None,
                 mae_base_groups: Dict[str, List[Tuple[int, int, int, float, str, str]]] = None,
                 mae_transfer_groups: Dict[str, List[Tuple[int, int, int, float, str, str]]] = None) -> None:
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
        self.mae_base_groups = mae_base_groups if mae_base_groups is not None else {}
        self.mae_transfer_groups = mae_transfer_groups if mae_transfer_groups is not None else {}

    def save(self):
        torch.save({
            "path": self.path,
            "output_dir": self.output_dir,
            "routes_dir": self.routes_dir,
            "mae_base": self.mae_base if self.mae_base else None,
            "mae_transfer": self.mae_transfer if self.mae_transfer else None,
            "mae_base_groups": self.mae_base_groups if self.mae_base_groups else None,
            "mae_transfer_groups": self.mae_transfer_groups if self.mae_transfer_groups else None
        }, self.path)

    def reset(self):
        self.mae_base = {}
        self.mae_base_groups = {}
        self.mae_transfer = {}
        self.mae_transfer_groups = {}

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
            mae_transfer=state_dict["mae_transfer"],
            mae_base_groups=state_dict["mae_base_groups"],
            mae_transfer_groups=state_dict["mae_transfer_groups"]
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

    def _get_mae_base(self, transfer_key: str, group: bool) -> float:
        base_port, transfer_port, start_time = self._decode_transfer_key(transfer_key)
        base_key = self._encode_base_key(base_port, start_time)
        return self.mae_base_groups[base_key] if group else self.mae_base[base_key]

    def export(self) -> None:
        base_keys = sorted(self.mae_base.keys())
        transfer_keys = sorted(self.mae_transfer.keys())
        decoded_transfer_keys = [self._decode_transfer_key(k) for k in transfer_keys]
        with open("evaluation_results.csv", "w", newline="") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(["Base Port", "Target Port", "Start Time", "Base Port MAE", "Transfer Port MAE"])
            for base_key in base_keys:
                base_port, start_time = self._decode_base_key(base_key)
                curr_decoded_transfer_keys = filter(lambda decoded_key: decoded_key[0] == base_port,
                                                    decoded_transfer_keys)

                for entry in curr_decoded_transfer_keys:
                    transfer_key = self._encode_transfer_key(base_port, entry[1], start_time)
                    if transfer_key in self.mae_transfer:
                        writer.writerow([base_port, entry[1], start_time, self.mae_base[base_key],
                                         self.mae_transfer[transfer_key]])
                    else:
                        raise ValueError(f"Unable to retrieve transfer result base port '{base_port}' to '{entry[1]}. "
                                         f"No such transfer key '{transfer_key}' (base key: '{base_key}')")

    def set_mae(self, base_port: Port, start_time: str, mae: Union[float, List[Tuple[int, int, int, float, str, str]]],
                transfer_port: Port = None, grouped: bool = False) -> None:
        if transfer_port is not None:
            transfer_key = self._encode_transfer_key(base_port.name, transfer_port.name, start_time)
            if grouped:
                self.mae_transfer_groups[transfer_key] = mae
            else:
                self.mae_transfer[transfer_key] = mae
        else:
            base_key = self._encode_base_key(base_port.name, start_time)
            if grouped:
                self.mae_base_groups[base_key] = mae
            else:
                self.mae_base[base_key] = mae

    def remove_mae(self, base_port: Port, start_time: str, transfer_port: Port = None, grouped: bool = False) -> None:
        if transfer_port is not None:
            transfer_key = self._encode_transfer_key(base_port.name, transfer_port.name, start_time)#
            if grouped:
                if transfer_key in self.mae_transfer_groups:
                    del self.mae_transfer_groups[transfer_key]
                else:
                    print(f"No grouped transfer result found for base_port '{base_port.name}', "
                          f"transfer_port '{transfer_port.name}' and start time '{start_time}'")
            else:
                if transfer_key in self.mae_transfer:
                    del self.mae_transfer[transfer_key]
                else:
                    print(f"No transfer result found for base_port '{base_port.name}', "
                          f"transfer_port '{transfer_port.name}' and start time '{start_time}'")
        else:
            base_key = self._encode_base_key(base_port.name, start_time)

            if grouped:
                if base_key in self.mae_base_groups:
                    del self.mae_base_groups[base_key]
                else:
                    print(f"No grouped base result found for port '{base_port.name}' and start time '{start_time}'")
            else:
                if base_key in self.mae_base:
                    del self.mae_base[base_key]
                else:
                    print(f"No base result found for port '{base_port.name}' and start time '{start_time}'")

    def eval_port(self, port: Union[str, Port], training_type: str, plot: bool = True) -> None:
        if isinstance(port, str):
            port = self.pm.find_port(port)

        trainings = self.pm.load_trainings(port, self.output_dir, self.routes_dir, training_type=training_type)
        if len(trainings) < 1:
            print(f"Skipping evaluation for port '{port.name}': No training found")
            return

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

        outputs = torch.cat(x, dim=0)
        targets = torch.cat(y, dim=0)
        loss = criterion(outputs, targets)
        mae = loss.item()
        print(f"Mae loss: {mae} || {as_duration(mae)}")

        mae_groups = mae_by_duration(outputs, targets)
        print(f"Mae by duration:\n{mae_groups}")

        self.set_mae(port, training.start_time, mae, grouped=False)
        self.set_mae(port, training.start_time, mae_groups, grouped=True)
        self.save()
        if plot:
            self.plot_grouped_mae(port, training_type=training_type, training=training)

    def eval_all(self) -> None:
        """
        Entry point for evaluating all available ports
        :return: None
        """
        # evaluate all ports
        for port in self.pm.ports.values():
            for t in ["base", "transfer"]:
                self.eval_port(port, training_type=t, plot=True)

        self.plot_ports_by_mae(training_type="base")
        self.plot_ports_by_mae(training_type="transfer")

    def plot_grouped_mae(self, port: Union[str, Port], training_type: str, training: TrainingIteration = None) -> None:
        if port is str:
            orig_port = port
            port = self.pm.find_port(port)
            if port is None:
                raise ValueError(f"Unable to associate port with port name '{orig_port}'")

        if training is None:
            trainings = self.pm.load_trainings(port, output_dir=self.output_dir, routes_dir=self.routes_dir,
                                               training_type=training_type)
            if len(trainings) > 0:
                training = trainings[-1]
            else:
                raise ValueError(f"No training of type '{training_type}' found for port '{port.name}'")

        transfer_port_name = None
        if training_type == "base":
            base_key = self._encode_base_key(port.name, training.start_time)
            mae_groups = self.mae_base_groups[base_key]
        else:
            model_file_name = os.path.split(training.model_path)[1]
            _, _, _, _, transfer_port_name = decode_model_file(model_file_name)
            transfer_key = self._encode_transfer_key(port.name, transfer_port_name, training.start_time)
            mae_groups = self.mae_transfer_groups[transfer_key]

        plot_path = os.path.join(self.eval_dir, port.name, encode_grouped_mae_plot(training.start_time,
                                                                                   file_type=training_type))
        title = f"Grouped MAE {training_type}-training: Port {port.name}"
        if training_type == "transfer":
            title = f"{title} (Base: {transfer_port_name})"
        plot_grouped_maes(mae_groups, port_name=port.name, path=plot_path, title=title)

    def plot_ports_by_mae(self, training_type: str) -> None:
        result = []
        for key, mae in self.mae_base.items():
            port_name, start_time = self._decode_base_key(key)
            result.append((mae, port_name))

        result.sort(key=lambda r: r[0])  # sort by mae
        result = list(map(list, zip(*result)))

        plot_ports_by_mae(result[0], result[1], title=f"MAE from {training_type}-training by port",
                          path=os.path.join(self.output_dir, "eval", "ports-mae.png"))

    def plot_transfer_effect(self, port: Union[str, Port]) -> None:
        """
        What's the cost of transferring a certain port's model to another port?
        MAE of transferred- vs. base-model
        :param port: Port, that has a transferred model from another port
        :return: None
        """
        if port is str:
            orig_port = port
            port = self.pm.find_port(port)
            if port is None:
                raise ValueError(f"Unable to associate port with port name '{orig_port}'")
        transfer_trainings = self.pm.load_trainings(port, output_dir=self.output_dir, routes_dir=self.routes_dir,
                                                    training_type="transfer")
        transfer_training = transfer_trainings[-1]
        _, start_time, _, _, base_port_name = decode_model_file(os.path.split(transfer_training.model_path)[1])

        base_trainings = self.pm.load_trainings(base_port_name, output_dir=self.output_dir, routes_dir=self.routes_dir,
                                                training_type="base")
        base_trainings = [t for t in base_trainings if t.start_time == start_time]
        if len(base_trainings) != 1:
            raise ValueError(f"Unable to identify base-training for start_time '{start_time}': "
                             f"Got {len(base_trainings)}, expected exactly 1")
        base_training = base_trainings[0]
        base_key = self._encode_base_key(port.name, base_training.start_time)
        transfer_key = self._encode_transfer_key(base_port_name, port.name, start_time)
        base_data = self.mae_base_groups[base_key]
        transfer_data = self.mae_transfer_groups[transfer_key]
        path = os.path.join(self.output_dir, "eval", f"transfer-effect_{base_port_name}-{port.name}.png")
        plot_transfer_effect(base_data, transfer_data, base_port_name, port.name, path)

    def plot_transfer_effects(self, sort: str = "mae_base") -> None:
        """
        MAE of transferred- vs base-model for all ports with matching trainings of type 'base' and 'transfer'
        :param sort: How to sort result data. Options [mae_base, num_data]
        :return: None
        """
        tmp = {}
        for transfer_key, mae_transfer in self.mae_transfer.items():
            base_port_name, transfer_port_name, _ = self._decode_transfer_key(transfer_key)
            mae_base = self._get_mae_base(transfer_key, group=False)

            if transfer_port_name in tmp:
                tmp[transfer_port_name][0].append(base_port_name)
                tmp[transfer_port_name][1].append(mae_base)
                tmp[transfer_port_name][2].append(mae_transfer)
            else:
                tmp[transfer_port_name] = ([base_port_name], [mae_base], [mae_transfer])

        def compute_metrics(key, val: Tuple[List[str], List[float], List[float]]) -> Tuple[str, str, float, str, float,
                                                                                           str, float, str, float,
                                                                                           float, float]:
            """
            :return: Tuple in form of
                transfer_port_name,
                max_mae_base_port_name, max_mae_base,
                min_mae_base_port_name, min_mae_base,
                max_mae_transfer_port_name, max_mae_transfer,
                min_mae_transfer_port_name, min_mae_transfer,
                avg_mae_base,
                avg_mae_transfer
            """
            max_mae_base = max(val[1])
            max_mae_base_port_name = val[0][val[1].index(max_mae_base)]
            min_mae_base = min(val[1])
            min_mae_base_port_name = val[0][val[1].index(min_mae_base)]
            max_mae_transfer = max(val[2])
            max_mae_transfer_port_name = val[0][val[2].index(max_mae_transfer)]
            min_mae_transfer = min(val[2])
            min_mae_transfer_port_name = val[0][val[2].index(min_mae_transfer)]
            return (key, max_mae_base_port_name, max_mae_base, min_mae_base_port_name, min_mae_base,
                    max_mae_transfer_port_name, max_mae_transfer, min_mae_transfer_port_name, min_mae_transfer,
                    sum(val[1]) / len(val[1]), sum(val[2]) / len(val[2]))

        result = [compute_metrics(key, val) for key, val in tmp.items()]

        if sort == "mae_base":
            result.sort(key=lambda r: r[0])
        result = list(map(list, zip(*result)))

        # plot_transfer_effects(base_data, transfer_data, base_port_names, transfer_port_names, path)
        path = os.path.join(self.output_dir, "eval", f"transfer-effects_{sort}.png")
        plot_transfer_effects(result[0], result[1], result[2], result[3], path)


def mae_by_duration(outputs: torch.Tensor, targets: torch.Tensor) -> List[Tuple[int, int, int, float, str, str]]:
    """
    Compute multiple maes for each target duration group
    :param outputs: Predicted values
    :param targets: Target values
    :return: List of tuples. Each tuple represents one group
        [(group_start, group_end, num_data, scaled_mae, descaled_mae, group_description), ...]
    """
    groups = [
        (-1, 1800, "0-0.5h"),
        (1800, 3600, "0.5-1h"),
        (3600, 7200, "1-2h"),
        (7200, 10800, "2-3h"),
        (10800, 14400, "3-4h"),
        (14400, 18000, "4-5h"),
        (18000, 21600, "5-6h"),
        (21600, 25200, "6-7h"),
        (25200, 28800, "7-8h"),
        (28800, 32400, "8-9h"),
        (32400, 36000, "9-10h"),
        (36000, 39600, "10-11h"),
        (39600, 43200, "11-12"),
        (43200, 86400, "12h - 1 day"),
        (86400, 172800, "1 day - 2 days"),
        (172800, 259200, "2 days - 3 days"),
        (259200, 345600, "3 days - 4 days"),
        (345600, 432000, "4 days - 5 days"),
        (432000, 518400, "5 days - 6 days"),
        (518400, 604800, "6 days - 1 week"),
        (604800, 155520000, "1 week - 1 month"),
        (155520000, int(data_ranges["label"]["max"]), "> 1 month")
    ]

    def scale(seconds: int) -> float:
        # half_range = (data_ranges["label"]["max"] - data_ranges["label"]["min"]) / 2
        # result = seconds / half_range
        # return -1 + result if seconds < half_range else result
        label_range = data_ranges["label"]["max"]
        return seconds / label_range

    def process_group(x: torch.Tensor, y: torch.Tensor, group: Tuple[int, int, str]) -> Tuple[int, int, int, float,
                                                                                              str, str]:
        criterion = nn.L1Loss(reduction="mean")
        mask = (y > scale(group[0])) & (y <= scale(group[1]))
        # mask = (y > group[0]) & (y <= group[1])
        x = x[mask]
        y = y[mask]
        mae = 0.
        num_data = x.shape[0]
        if num_data > 0:
            loss = criterion(x, y)
            mae = loss.item()
        # return group[0], group[1], num_data, mae, descale_mae(mae, as_str_duration=True), group[2]
        return group[0], group[1], num_data, mae, as_duration(mae), group[2]

    mae_groups = [process_group(outputs, targets, group) for group in groups]
    return mae_groups


def main(args) -> None:
    command = args.command
    if command == "init":
        e = Evaluator(args.output_dir, args.routes_dir)
        e.save()
    elif command == "reset":
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))
        e.reset()
        e.save()
    elif command == "eval":
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))
        e.eval_all()
    elif command == "eval_port":
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))
        training_types = []
        if args.training_type == "all":
            training_types = ["base", "transfer"]
        elif args.training_type not in ["base", "transfer"]:
            raise ValueError(f"Unknown parameter --training_type '{args.training_type}'. Not in [all, base, transfer]")
        else:
            training_types = [args.training_type]
        for t in training_types:
            e.eval_port(args.port_name, training_type=t, plot=True)
    elif command == "plot":  # plot saved evaluations
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))
        for t in ["base", "transfer"]:
            e.plot_grouped_mae(args.port_name, training_type=t)

    else:
        raise ValueError(f"Unknown command '{command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "eval", "eval_port", "plot"])
    parser.add_argument("--routes_dir", type=str, default=os.path.join(script_dir, os.pardir, "data", "routes"),
                        help="Directory to routes")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, os.pardir, "output"),
                        help="Directory to outputs")
    parser.add_argument("--port_name", type=str, help="Port name")
    parser.add_argument("--training_type", type=str, choices=["all", "base", "transfer"])
    main(parser.parse_args())
