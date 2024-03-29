import argparse
import csv
import os
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from captum.attr import IntegratedGradients

from dataset import RoutesDirectoryDataset
from net.model import InceptionTimeModel
from plotter import plot_ports_by_mae, plot_grouped_maes, plot_transfer_effect, plot_transfer_effects
from port import Port, PortManager
from training import TrainingIteration
from util import encode_grouped_mae_plot, as_duration, data_ranges, decode_model_file, dataset_cols

from typing import Dict, List, Tuple, Union

script_dir = os.path.abspath(os.path.dirname(__file__))


class Evaluator:
    def __init__(self, output_dir: str, routes_dir: str, mae_base: Dict[str, float] = None,
                 mae_transfer: Dict[str, float] = None,
                 mae_base_groups: Dict[str, List[Tuple[int, int, int, float, str]]] = None,
                 mae_transfer_groups: Dict[str, List[Tuple[int, int, int, float, str]]] = None) -> None:
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
    def load(eval_dir_or_path: str, output_dir: str = None, routes_dir: str = None) -> 'Evaluator':
        path = eval_dir_or_path
        eval_dir, file = os.path.split(eval_dir_or_path)
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        if not file.endswith(".tar"):
            path = os.path.join(path, "evaluator.tar")
        state_dict = torch.load(path)
        evaluator = Evaluator(
            output_dir=state_dict["output_dir"] if output_dir is None else output_dir,
            routes_dir=state_dict["routes_dir"] if routes_dir is None else routes_dir,
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
    def _encode_transfer_key(source_port: str, target_port: str, start_time: str) -> str:
        return f"{source_port}_{target_port}_{start_time}"

    @staticmethod
    def _decode_transfer_key(key: str) -> Tuple[str, str, str]:
        result = key.split("_")
        return result[0], result[1], result[2]

    def _get_mae_base(self, transfer_key: str, group: bool) -> float:
        source_port, _, start_time = self._decode_transfer_key(transfer_key)
        base_key = self._encode_base_key(source_port, start_time)
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

                for decoded_transfer_key in curr_decoded_transfer_keys:
                    transfer_key = self._encode_transfer_key(base_port, decoded_transfer_key[1], start_time)
                    if transfer_key in self.mae_transfer:
                        writer.writerow([base_port, decoded_transfer_key[1], start_time, self.mae_base[base_key],
                                         self.mae_transfer[transfer_key]])
                    else:
                        raise ValueError(f"Unable to retrieve transfer result base port '{base_port}' to "
                                         f"'{decoded_transfer_key[1]}. No such transfer key '{transfer_key}' "
                                         f"(base key: '{base_key}')")

    def set_mae(self, port: Port, start_time: str, mae: Union[float, List[Tuple[int, int, int, float, str]]],
                source_port: Port = None, grouped: bool = False) -> None:
        if source_port is not None:
            transfer_key = self._encode_transfer_key(source_port.name, port.name, start_time)
            if grouped:
                self.mae_transfer_groups[transfer_key] = mae
            else:
                self.mae_transfer[transfer_key] = mae
        else:
            base_key = self._encode_base_key(port.name, start_time)
            if grouped:
                self.mae_base_groups[base_key] = mae
            else:
                self.mae_base[base_key] = mae

    def remove_mae(self, port: Port, start_time: str, source_port: Port = None, grouped: bool = False) -> None:
        if source_port is not None:
            transfer_key = self._encode_transfer_key(source_port.name, port.name, start_time)
            # print(f"transfer key: {transfer_key}")
            # print(f"transfer keys: {self.mae_transfer.keys()}")
            # print(f"transfer group keys: {self.mae_transfer_groups.keys()}")
            if grouped:
                if transfer_key in self.mae_transfer_groups:
                    del self.mae_transfer_groups[transfer_key]
                else:
                    print(f"No grouped transfer result found for port '{port.name}', "
                          f"source_port '{source_port.name}' and start time '{start_time}'")
            else:
                if transfer_key in self.mae_transfer:
                    del self.mae_transfer[transfer_key]
                else:
                    print(f"No transfer result found for port '{port.name}', "
                          f"source_port '{source_port.name}' and start time '{start_time}'")
        else:
            base_key = self._encode_base_key(port.name, start_time)
            if grouped:
                if base_key in self.mae_base_groups:
                    del self.mae_base_groups[base_key]
                else:
                    print(f"No grouped base result found for port '{port.name}' and start time '{start_time}'")
            else:
                if base_key in self.mae_base:
                    del self.mae_base[base_key]
                else:
                    print(f"No base result found for port '{port.name}' and start time '{start_time}'")
        # print(f"base keys: {self.mae_base.keys()}")
        # print(f"base group keys: {self.mae_base_groups.keys()}")
        # print(f"transfer keys: {self.mae_transfer.keys()}")
        # print(f"transfer group keys: {self.mae_transfer_groups.keys()}")

    def eval_port(self, port: Union[str, Port], training_type: str, plot: bool = True) -> None:
        if isinstance(port, str):
            orig_port = port
            port = self.pm.find_port(port)
            if port is None:
                raise ValueError(f"Unable to associate port with port name '{orig_port}'")

        trainings = self.pm.load_trainings(port, self.output_dir, self.routes_dir, training_type=training_type)
        if len(trainings) < 1:
            print(f"Skipping evaluation for port '{port.name}': No {training_type}-training found")
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
        print(f">->->->->->-> STARTED EVALUATION OF PORT {port.name} | TRAINING TYPE {training_type} <-<-<-<-<-<-<-<-<")
        criterion = nn.L1Loss(reduction="mean")
        x_in = []
        with torch.no_grad():
            for eval_idx, (data, target) in enumerate(tqdm(eval_loader, desc="Evaluation progress")):
                data = data.to(device)
                target = target.to(device)
                output = model(data)

                x.append(output)
                y.append(target)
                x_in.append(data)

        x_input = torch.cat(x_in, dim=0)
        outputs = torch.cat(x, dim=0)
        targets = torch.cat(y, dim=0)
        loss = criterion(outputs, targets)
        mae = loss.item()
        print(f"Mae loss: {mae} || {as_duration(mae)}")

        mae_groups = self.group_mae(outputs, targets)
        print(f"Mae by duration:\n{mae_groups}")

        if training_type == "transfer":
            model_file = os.path.split(training.model_path)[1]
            _, _, _, _, source_port_name = decode_model_file(model_file)
            source_port = self.pm.find_port(source_port_name)
            if source_port is None:
                raise ValueError(f"Unable to associate port with port name '{source_port_name}")
            self.set_mae(port, training.start_time, mae, source_port=source_port, grouped=False)
            self.set_mae(port, training.start_time, mae_groups, source_port=source_port, grouped=True)
        else:
            self.set_mae(port, training.start_time, mae, grouped=False)
            self.set_mae(port, training.start_time, mae_groups, grouped=True)
        self.save()

        # Obtain Feature Attributions: https://arxiv.org/pdf/1703.01365.pdf
        ig = IntegratedGradients(model)
        ig_attr_test = ig.attribute(x_input, n_steps=50)

        if plot:
            self.plot_grouped_mae(port, training_type=training_type, training=training)
            # self.plot_ig_attr_test(ig_attr_test)

    def eval_all(self, plot: bool = True) -> None:
        """
        Entry point for evaluating all available ports
        :return: None
        """
        # evaluate all ports
        for port in self.pm.ports.values():
            for t in ["base", "transfer"]:
                self.eval_port(port, training_type=t, plot=plot)

        if plot:
            self.plot_ports_by_mae(training_type="base")
            self.plot_ports_by_mae(training_type="transfer")

    def plot(self, port_name: str = None) -> None:
        """
        Generate all general and specific plots for specified/all available ports.
        :param port_name: If specified, plot this port. If not, plot all
        :return: None
        """
        if port_name is not None:
            self.plot_port(port_name)
        else:
            for port in self.pm.ports.values():
                self.plot_port(port)
        self.plot_transfer_effects()

    def plot_port(self, port: Union[str, Port]):
        for t in ["base", "transfer"]:
            self.plot_grouped_mae(port, training_type=t)
            self.plot_ports_by_mae(training_type=t)
        self.plot_transfer_effect(port)

    def plot_grouped_mae(self, port: Union[str, Port], training_type: str, training: TrainingIteration = None) -> None:
        if isinstance(port, str):
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
                print(f"No training of type '{training_type}' found for port '{port.name}'. Skipping plot_grouped_mae")
                return

        source_port_name = None
        if training_type == "base":
            base_key = self._encode_base_key(port.name, training.start_time)
            mae_groups = self.mae_base_groups[base_key]
        else:
            model_file_name = os.path.split(training.model_path)[1]
            _, _, _, _, source_port_name = decode_model_file(model_file_name)
            transfer_key = self._encode_transfer_key(source_port_name, port.name, training.start_time)
            mae_groups = self.mae_transfer_groups[transfer_key]

        plot_path = os.path.join(self.eval_dir, port.name, encode_grouped_mae_plot(training.start_time,
                                                                                   file_type=training_type))
        title = f"Grouped MAE {training_type}-training: Port {port.name}"
        if training_type == "transfer":
            title = f"{title} (Source port: {source_port_name})"
        plot_grouped_maes(mae_groups, title=title, path=plot_path)

    def plot_ports_by_mae(self, training_type: str) -> None:
        result = []
        if training_type == "base":
            for key, mae in self.mae_base.items():
                port_name, start_time = self._decode_base_key(key)
                result.append((mae, port_name))
        elif training_type == "transfer":
            tmp = {}
            for key, mae in self.mae_transfer.items():
                source_port_name, target_port_name, start_time = self._decode_transfer_key(key)
                if target_port_name in tmp:
                    tmp[target_port_name].append(mae)
                else:
                    tmp[target_port_name] = [mae]
            result = [(sum(v) / len(v), k) for k, v in tmp.items()]
        else:
            raise ValueError(f"Unknown training-type '{training_type}'")

        result.sort(key=lambda r: r[0])  # sort by mae
        result = list(map(list, zip(*result)))

        title = f"MAE from {training_type}-training by port"
        if training_type == "transfer":
            title = f"Average {title}"
        plot_ports_by_mae(result[0], result[1], title=title,
                          path=os.path.join(self.output_dir, "eval", f"ports-mae_{training_type}-training.png"))

    def plot_transfer_effect(self, port: Union[str, Port]) -> None:
        """
        What's the cost of transferring a certain port's model to another port?
        MAE of transferred- vs. base-model
        :param port: Port, that has a transferred model from another port
        :return: None
        """
        if isinstance(port, str):
            orig_port = port
            port = self.pm.find_port(port)
            if port is None:
                raise ValueError(f"Unable to associate port with port name '{orig_port}'")
        transfer_trainings = self.pm.load_trainings(port, output_dir=self.output_dir, routes_dir=self.routes_dir,
                                                    training_type="transfer")
        if len(transfer_trainings) < 1:
            print(f"No training of type 'transfer' found for port {port.name}. Skipping plot_transfer_effect")
            return

        transfer_training = transfer_trainings[-1]
        _, _, start_time, _, source_port_name = decode_model_file(os.path.split(transfer_training.model_path)[1])

        base_trainings = self.pm.load_trainings(source_port_name, output_dir=self.output_dir,
                                                routes_dir=self.routes_dir, training_type="base")
        base_trainings = [t for t in base_trainings if t.start_time == start_time]
        if len(base_trainings) != 1:
            raise ValueError(f"Unable to identify base-training for start_time '{start_time}': "
                             f"Got {len(base_trainings)}, expected exactly 1")
        base_training = base_trainings[0]
        base_key = self._encode_base_key(source_port_name, base_training.start_time)
        # print(f"normal keys: {self.mae_base.keys()}")
        # print(f"grouped keys: {self.mae_base_groups.keys()}")
        # print(f"transferred normal keys: {self.mae_transfer.keys()}")
        # print(f"transferred grouped keys: {self.mae_transfer_groups.keys()}")
        transfer_key = self._encode_transfer_key(source_port_name, port.name, start_time)
        base_data = self.mae_base_groups[base_key]
        transfer_data = self.mae_transfer_groups[transfer_key]
        path = os.path.join(self.output_dir, "eval", f"transfer-effect_{source_port_name}-{port.name}.png")
        plot_transfer_effect(base_data, transfer_data, source_port_name, port.name, path)

    def plot_transfer_effects(self, sort: str = "mae_base") -> None:
        """
        MAE of transferred- vs base-model for all ports with matching trainings of type 'base' and 'transfer'
        :param sort: How to sort result data. Options [mae_base, num_data]
        :return: None
        """
        tmp = {}
        for transfer_key, mae_transfer in self.mae_transfer.items():
            source_port_name, target_port_name, _ = self._decode_transfer_key(transfer_key)
            mae_source_base = self._get_mae_base(transfer_key, group=False)

            if target_port_name in tmp:
                tmp[target_port_name][0].append(source_port_name)
                tmp[target_port_name][1].append(mae_source_base)
                tmp[target_port_name][2].append(mae_transfer)
            else:
                tmp[target_port_name] = ([source_port_name], [mae_source_base], [mae_transfer])

        def compute_metrics(key, val: Tuple[List[str], List[float], List[float]]) -> Tuple[str, str, float, str, float,
                                                                                           str, float, str, float,
                                                                                           float, float]:
            """
            :return: Tuple in form of
                transfer_port_name,
                max_mae_source_port_name, max_mae_source_base,
                min_mae_source_port_name, min_mae_source_base,
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

        path = os.path.join(self.output_dir, "eval", f"transfer-effects_{sort}.png")
        plot_transfer_effects(result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7],
                              result[8], result[9], result[10], path)

    def plot_ig_attr_test(self, result: List[float]) -> None:
        # labels =
        return

    @staticmethod
    def group_mae(outputs: torch.Tensor, targets: torch.Tensor) -> List[Tuple[int, int, int, float, str]]:
        """
        Compute multiple maes for each target duration group
        :param outputs: Predicted values
        :param targets: Target values
        :return: List of tuples. Each tuple represents one group
            [(group_start, group_end, num_data, scaled_mae, group_description), ...]
        """
        # groups = [
        #     (-1, 1800, "0-0.5h"),
        #     (1800, 3600, "0.5-1h"),
        #     (3600, 7200, "1-2h"),
        #     (7200, 10800, "2-3h"),
        #     (10800, 14400, "3-4h"),
        #     (14400, 18000, "4-5h"),
        #     (18000, 21600, "5-6h"),
        #     (21600, 25200, "6-7h"),
        #     (25200, 28800, "7-8h"),
        #     (28800, 32400, "8-9h"),
        #     (32400, 36000, "9-10h"),
        #     (36000, 39600, "10-11h"),
        #     (39600, 43200, "11-12"),
        #     (43200, 86400, "12h - 1 day"),
        #     (86400, 172800, "1 day - 2 days"),
        #     (172800, 259200, "2 days - 3 days"),
        #     (259200, 345600, "3 days - 4 days"),
        #     (345600, 432000, "4 days - 5 days"),
        #     (432000, 518400, "5 days - 6 days"),
        #     (518400, 604800, "6 days - 1 week"),
        #     (604800, 155520000, "1 week - 1 month"),
        #     (155520000, int(data_ranges["label"]["max"]), "> 1 month")
        # ]
        groups = [
            (-1, 1800, "0-0.5h"),
            (1800, 3600, "0.5-1h"),
            (3600, 7200, "1-2h"),
            (7200, 10800, "2-3h"),
            (10800, 14400, "3-4h"),
            (14400, 21600, "4-6h"),
            (21600, 28800, "6-8h"),
            (28800, 36000, "8-10h"),
            (36000, 43200, "10-12h"),
            (43200, 50400, "12-16h"),
            (50400, 64800, "16-20h"),
            (64800, 86400, "20-24h"),
            (86400, 172800, "1-2d"),
            (172800, 259200, "2-3d"),
            (259200, 345600, "3-4d"),
            (345600, 432000, "4-5d"),
            (432000, 518400, "5-6d"),
            (518400, 604800, "6-7d"),
            (604800, 1209600, "1-2w"),
            (1209600, 2419200, "2-4w"),
            (2419200, int(data_ranges["label"]["max"]), "> 4w")
        ]

        def scale(seconds: int) -> float:
            # half_range = (data_ranges["label"]["max"] - data_ranges["label"]["min"]) / 2
            # result = seconds / half_range
            # return -1 + result if seconds < half_range else result
            label_range = data_ranges["label"]["max"]
            return seconds / label_range

        def process_group(x: torch.Tensor, y: torch.Tensor, group: Tuple[int, int, str]) -> Tuple[int, int, int, float,
                                                                                                  str]:
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
            return group[0], group[1], num_data, mae, group[2]

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
    elif command == "reset_mae":
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))
        port = e.pm.find_port(args.port_name)
        if port is None:
            raise ValueError(f"Unable to associate port with port name '{args.port_name}'")
        source_port = None
        if args.source_port is not None:
            source_port = e.pm.find_port(args.source_port)
            if source_port is None:
                raise ValueError(f"Unable to associate port with port name '{args.source_port}'")
        print(f"source port: {source_port}")
        e.remove_mae(port=port, start_time=args.start_time, source_port=source_port, grouped=args.group)
        e.save()
    elif command == "eval":
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))
        e.eval_all(plot=args.plot)
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
            e.eval_port(args.port_name, training_type=t, plot=args.plot)
    elif command == "make_plots":  # plot saved evaluations
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))
        e.plot(port_name=args.port_name)
    elif command == "adapt_paths":
        e = Evaluator.load(eval_dir_or_path=os.path.join(args.output_dir, "eval"), output_dir=args.output_dir,
                           routes_dir=args.routes_dir)
        # print(f"path: {e.path}")
        # print(f"output dir: {e.output_dir}")
        # print(f"routes dir: {e.routes_dir}")
        e.save()
    else:
        raise ValueError(f"Unknown command '{command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "eval", "eval_port", "make_plots", "adapt_paths", "reset_mae"])
    parser.add_argument("--routes_dir", type=str, default=os.path.join(script_dir, os.pardir, "data", "routes"),
                        help="Directory to routes")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, os.pardir, "output"),
                        help="Directory to outputs")
    parser.add_argument("--port_name", type=str, help="Port name")
    parser.add_argument("--training_type", type=str, choices=["all", "base", "transfer"])
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--no_plot", dest="plot", action="store_false")
    parser.add_argument("--source_port", type=str, default=None)
    parser.add_argument("--start_time", type=str)
    parser.add_argument("--group", dest="group", action="store_true")
    parser.add_argument("--no_group", dest="group", action="store_false")
    parser.set_defaults(plot=True)
    main(parser.parse_args())
