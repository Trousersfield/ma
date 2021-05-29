import argparse
import csv
import os
import torch

from sklearn.metrics import mean_absolute_error

# from captum.attr import IntegratedGradients

from plotter import plot_ports_by_mae, plot_grouped_maes, plot_transfer_effect, plot_transfer_effects
from port import Port, PortManager
from training import TrainingIteration
from util import encode_grouped_mae_plot, decode_keras_model, SECONDS_PER_YEAR

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
    def _encode_transfer_key(source_port: str, target_port: str, start_time: str, config_uid: int) -> str:
        return f"{source_port}_{target_port}_{start_time}_{config_uid}"

    @staticmethod
    def _decode_transfer_key(key: str) -> Tuple[str, str, str, int]:
        result = key.split("_")
        return result[0], result[1], result[2], int(result[3])

    def _get_mae_base(self, transfer_key: str, group: bool) -> float:
        source_port, _, start_time, _ = self._decode_transfer_key(transfer_key)
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
                    target_port = decoded_transfer_key[1]
                    config_uid = decoded_transfer_key[3]
                    transfer_key = self._encode_transfer_key(base_port, target_port, start_time, config_uid)
                    if transfer_key in self.mae_transfer:
                        writer.writerow([base_port, target_port, start_time, self.mae_base[base_key],
                                         self.mae_transfer[transfer_key]])
                    else:
                        raise ValueError(f"Unable to retrieve transfer result base port '{base_port}' to "
                                         f"'{decoded_transfer_key[1]}. No such transfer key '{transfer_key}' "
                                         f"(base key: '{base_key}')")

    def set_mae(self, port: Port, start_time: str, mae: Union[float, List[Tuple[int, int, int, float, str]]],
                source_port: Port = None, config_uid: int = None, save: bool = True) -> None:
        if source_port is not None:
            if config_uid is None:
                raise ValueError(f"Missing required parameter config_uid for setting transfer result")
            transfer_key = self._encode_transfer_key(source_port.name, port.name, start_time, config_uid)
            if mae is float:
                self.mae_transfer[transfer_key] = mae
            else:
                self.mae_transfer_groups[transfer_key] = mae
        else:
            base_key = self._encode_base_key(port.name, start_time)
            if mae is float:
                self.mae_base[base_key] = mae
            else:
                self.mae_base_groups[base_key] = mae
        if save:
            self.save()

    def remove_mae(self, port: Port, start_time: str, grouped: bool = False, source_port: Port = None,
                   config_uid: bool = None, save: bool = True) -> None:
        if source_port is not None:
            transfer_key = self._encode_transfer_key(source_port.name, port.name, start_time, config_uid)
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
        if save:
            self.save()
        # print(f"base keys: {self.mae_base.keys()}")
        # print(f"base group keys: {self.mae_base_groups.keys()}")
        # print(f"transfer keys: {self.mae_transfer.keys()}")
        # print(f"transfer group keys: {self.mae_transfer_groups.keys()}")

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

    def plot_grouped_mae(self, port: Union[str, Port], training_type: str, start_time: str = None) -> None:
        if isinstance(port, str):
            orig_port = port
            port = self.pm.find_port(port)
            if port is None:
                raise ValueError(f"Unable to associate port with port name '{orig_port}'")

        trainings = self.pm.load_trainings(port, output_dir=self.output_dir, routes_dir=self.routes_dir,
                                           training_type=training_type)
        if trainings:
            if start_time is None:
                trainings = list(trainings.values())[-1]
            elif start_time in trainings:
                trainings = trainings[start_time]
            else:
                raise ValueError(f"No {training_type}-training for start time '{start_time}' found")
        else:
            print(f"No training of type '{training_type}' found for port '{port.name}'. Skipping plot_grouped_mae")
            return

        def _plot(ti: TrainingIteration, c_uid: int = None) -> None:
            plot_path = os.path.join(self.eval_dir, port.name,
                                     encode_grouped_mae_plot(training_type, ti.start_time, c_uid))
            title = f"Grouped MAE {training_type}-training: Port {port.name}"
            if training_type == "transfer":
                title = f"{title} (Source port: {source_port_name})"
            plot_grouped_maes(mae_groups, title=title, path=plot_path)

        source_port_name = None
        if training_type == "base":
            assert len(trainings) == 1
            base_key = self._encode_base_key(port.name, trainings[0].start_time)
            mae_groups = self.mae_base_groups[base_key]
            _plot(trainings[0])
        else:
            for training in trainings:
                model_file_name = os.path.split(training.model_path)[1]
                _, _, _, source_port_name, config_uid = decode_keras_model(model_file_name)
                transfer_key = self._encode_transfer_key(source_port_name, port.name, training.start_time, config_uid)
                mae_groups = self.mae_transfer_groups[transfer_key]
                _plot(training, config_uid)

    def plot_ports_by_mae(self, training_type: str) -> None:
        results = [(-1, [])]
        if training_type == "base":
            for key, mae in self.mae_base.items():
                port_name, start_time = self._decode_base_key(key)
                results[0][1].append((mae, port_name))
        elif training_type == "transfer":
            tmp = {}
            for key, mae in self.mae_transfer.items():
                source_port_name, target_port_name, start_time, config_uid = self._decode_transfer_key(key)
                if config_uid in tmp:
                    if target_port_name in tmp[config_uid]:
                        tmp[config_uid][target_port_name].append(mae)
                    else:
                        tmp[config_uid][target_port_name] = [mae]
                else:
                    tmp[config_uid] = {}
                    tmp[config_uid][target_port_name] = [mae]
                # if target_port_name in tmp:
                #     tmp[target_port_name].append(mae)
                # else:
                #     tmp[target_port_name] = [mae]
            # result = [(sum(v) / len(v), k) for k, v in tmp.items()]
            results = [(c_uid, [(sum(v) / len(v), k) for k, v in item.items()]) for c_uid, item in tmp.items()]
        else:
            raise ValueError(f"Unknown training-type '{training_type}'")

        for result in results:
            result[1].sort(key=lambda r: r[0])  # sort by mae
            data = list(map(list, zip(*result[1])))

            title = f"MAE from {training_type}-training by port"
            if training_type == "transfer":
                title = f"Average {title} (Config {result[0]})"
            plot_ports_by_mae(data[0], data[1], title=title,
                              path=os.path.join(self.output_dir, "eval", f"ports-mae_{training_type}-training.png"))

    def plot_transfer_effect(self, port: Union[str, Port], start_time: str = None) -> None:
        """
        What's the cost of transferring a certain port's model to another port?
        MAE of transferred- vs. base-model
        :param port: Port, that has a transferred model from another port
        :param start_time: corresponding start time
        :return: None
        """
        if isinstance(port, str):
            orig_port = port
            port = self.pm.find_port(port)
            if port is None:
                raise ValueError(f"Unable to associate port with port name '{orig_port}'")
        trainings = self.pm.load_trainings(port, output_dir=self.output_dir, routes_dir=self.routes_dir,
                                           training_type="transfer")
        if len(trainings.keys()) < 1:
            raise ValueError(f"No transfer-training found for port {port.name}")

        if start_time is None:
            transfer_trains = list(trainings.values())[-1]
        elif start_time in trainings:
            transfer_trains = trainings[start_time]
        else:
            raise ValueError(f"Unable to find transfer-training for start time '{start_time}'")

        _, _, _, source_port_name, _ = decode_keras_model(os.path.split(transfer_trains[0].model_path)[1])
        base_key = self._encode_base_key(source_port_name, start_time)
        base_data = self.mae_base_groups[base_key]

        # trainings = self.pm.load_trainings(source_port_name, output_dir=self.output_dir, routes_dir=self.routes_dir,
        #                                    training_type="base")
        # if len(trainings.keys()) < 1:
        #     raise ValueError(f"No corresponding base-training found for port {source_port_name}")

        # if start_time in trainings:
        #     assert len(trainings[start_time]) == 1
        #     base_train = trainings[start_time][0]
        # else:
        #     raise ValueError(f"Unable to find corresponding base-training for port {source_port_name} and start "
        #                      f"time {start_time}")

        # print(f"normal keys: {self.mae_base.keys()}")
        # print(f"grouped keys: {self.mae_base_groups.keys()}")
        # print(f"transferred normal keys: {self.mae_transfer.keys()}")
        # print(f"transferred grouped keys: {self.mae_transfer_groups.keys()}")

        # plot effect for each transfer-kind
        for transfer_train in transfer_trains:
            _, _, _, _, config_uid = decode_keras_model(os.path.split(transfer_train.model_path)[1])
            transfer_key = self._encode_transfer_key(source_port_name, port.name, start_time, config_uid)
            transfer_data = self.mae_transfer_groups[transfer_key]
            path = os.path.join(self.output_dir, "eval",
                                f"transfer-effect-{config_uid}_{source_port_name}-{port.name}.png")
            plot_transfer_effect(base_data, transfer_data, source_port_name, port.name, path)

    def plot_transfer_effects(self, sort: str = "mae_base") -> None:
        """
        MAE of transferred- vs base-model for all ports with matching trainings of type 'base' and 'transfer'
        :param sort: How to sort result data. Options [mae_base, num_data]
        :return: None
        """
        tmp = {}
        for transfer_key, mae_transfer in self.mae_transfer.items():
            source_port_name, target_port_name, _, config_uid = self._decode_transfer_key(transfer_key)
            mae_source_base = self._get_mae_base(transfer_key, group=False)
            config_uid = str(config_uid)

            if config_uid in tmp:
                if target_port_name in tmp[config_uid]:
                    tmp[config_uid][target_port_name][0].append(source_port_name)
                    tmp[config_uid][target_port_name][1].append(mae_source_base)
                    tmp[config_uid][target_port_name][2].append(mae_transfer)
                else:
                    tmp[config_uid][target_port_name] = ([source_port_name], [mae_source_base], [mae_transfer])
            else:
                tmp[config_uid] = {}
                tmp[config_uid][target_port_name] = ([source_port_name], [mae_source_base], [mae_transfer])

        def compute_metrics(port, val: Tuple[List[str], List[float], List[float]]) -> Tuple[str, str, float, str, float,
                                                                                            str, float, str, float,
                                                                                            float, float]:
            """
            :return: Tuple in form of
                config_uid,
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
            return (port, max_mae_base_port_name, max_mae_base, min_mae_base_port_name, min_mae_base,
                    max_mae_transfer_port_name, max_mae_transfer, min_mae_transfer_port_name, min_mae_transfer,
                    sum(val[1]) / len(val[1]), sum(val[2]) / len(val[2]))

        results = [(c_uid, [compute_metrics(port, val) for port, val in item]) for c_uid, item in tmp.items()]

        for result in results:
            if sort == "mae_base":
                result[1].sort(key=lambda r: r[0])
            data = list(map(list, zip(*result[1])))

            path = os.path.join(self.output_dir, "eval", f"transfer-effects-{result[0]}_{sort}.png")
            plot_transfer_effects(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
                                  data[9], data[10], path)

    def plot_ig_attr_test(self, result: List[float]) -> None:
        # labels =
        return

    @staticmethod
    # def group_mae(outputs: torch.Tensor, targets: torch.Tensor) -> List[Tuple[int, int, int, float, str]]:
    def group_mae(y_true, y_pred) -> List[Tuple[int, int, int, float, str]]:
        """
        Compute multiple maes for each target duration group
        :param y_true: Labels
        :param y_pred: Predictions
        :return: List of tuples. Each tuple represents one group
            [(group_start, group_end, num_data, scaled_mae, group_description), ...]
        """
        groups = [
            (-1., 15., "0-15min"),
            (15., 30., "15-30min"),
            (30., 45., "30-45min"),
            (45., 60., "45-60min"),
            (60., 120., "1-2h"),
            (120., 180., "2-3h"),
            (180., 240., "3-4h"),
            (240., 360., "4-6h"),
            (360., 600., "6-10h"),
            (600., 960., "10-16h"),
            (960., 1200., "16-20h"),
            (1200., 1440., "20-24h"),
            (1440., 1880., "1-2d"),
            (1880., 5760., "2-4d"),
            (5760., 10080., "4-7d"),
            (10080., 20160., "1-2w"),
            (20160., 40320., "2-4w"),
            (40320., SECONDS_PER_YEAR / 60, "> 4w")
        ]

        def process_group(group: Tuple[int, int, str]) -> Tuple[int, int, int, float, str]:
            mask = (y_true > group[0]) & (y_true <= group[1])
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            mae = 0.
            num_y_true = y_true_group.shape[0]
            # print(f"group {group[0]} shapes - y_true: {num_y_true} y_pred: {y_pred_group.shape[0]}")
            if num_y_true > 0 and y_pred_group.shape[0] > 0:
                mae = mean_absolute_error(y_true_group, y_pred_group)
            return group[0], group[1], num_y_true, mae, group[2]

        mae_groups = [process_group(group) for group in groups]
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
        raise ValueError(f"No implemented: config_uid")
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
