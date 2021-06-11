import argparse
import csv
import glob
import numpy as np
import os
import torch

from datetime import datetime
from sklearn.metrics import mean_absolute_error

# from captum.attr import IntegratedGradients

from plotter import plot_ports_by_mae, plot_grouped_maes, plot_transfer_effect, plot_grouped_transfer_effect,\
    plot_transfer_effects, plot_mae_heatmap
from port import Port, PortManager
from training import TrainingIteration
from util import encode_grouped_mae_plot, decode_keras_model, SECONDS_PER_YEAR, validate_params, as_datetime, as_str

from typing import Dict, List, Tuple, Union

script_dir = os.path.abspath(os.path.dirname(__file__))


class Evaluator:
    def __init__(self, output_dir: str, routes_dir: str, mae_base: Dict[str, float] = None,
                 mae_transfer: Dict[str, float] = None,
                 mae_base_groups: Dict[str, List[Tuple[int, int, int, float, str]]] = None,
                 mae_transfer_groups: Dict[str, List[Tuple[int, int, int, float, str]]] = None,
                 naive_baselines: Dict[str, float] = None) -> None:
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
        self.naive_baselines = naive_baselines if naive_baselines is not None else {}

    def save(self):
        torch.save({
            "path": self.path,
            "output_dir": self.output_dir,
            "routes_dir": self.routes_dir,
            "mae_base": self.mae_base if self.mae_base else None,
            "mae_transfer": self.mae_transfer if self.mae_transfer else None,
            "mae_base_groups": self.mae_base_groups if self.mae_base_groups else None,
            "mae_transfer_groups": self.mae_transfer_groups if self.mae_transfer_groups else None,
            "naive_baselines": self.naive_baselines if self.naive_baselines else None
        }, self.path)

    def reset(self, options: List[str] = None):
        if options is not None:
            if "transfer" in options:
                self.mae_transfer = {}
                self.mae_transfer_groups = {}
        else:
            self.mae_base = {}
            self.mae_base_groups = {}
            self.mae_transfer = {}
            self.mae_transfer_groups = {}
        self.save()

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
            mae_transfer_groups=state_dict["mae_transfer_groups"],
            naive_baselines=state_dict["naive_baselines"] if "naive_baselines" in state_dict else None
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

                for _, target_port, _, config_uid in curr_decoded_transfer_keys:
                    # target_port = decoded_transfer_key[1]
                    # config_uid = decoded_transfer_key[3]
                    transfer_key = self._encode_transfer_key(base_port, target_port, start_time, config_uid)
                    if transfer_key in self.mae_transfer:
                        writer.writerow([base_port, target_port, start_time, self.mae_base[base_key],
                                         self.mae_transfer[transfer_key]])
                    else:
                        raise ValueError(f"Unable to retrieve transfer result base port '{base_port}' to "
                                         f"'{target_port}. No such transfer key '{transfer_key}' "
                                         f"(base key: '{base_key}')")

    def set_naive_baseline(self, port: Port, mae: float) -> None:
        self.naive_baselines[port.name] = mae
        self.save()

    def set_mae(self, port: Port, start_time: str, mae: Union[float, List[Tuple[int, int, int, float, str]]],
                source_port: Port = None, config_uid: int = None, save: bool = True) -> None:
        if source_port is not None:
            if config_uid is None:
                raise ValueError(f"Missing required parameter config_uid for setting transfer result")
            transfer_key = self._encode_transfer_key(source_port.name, port.name, start_time, config_uid)
            if isinstance(mae, float):
                self.mae_transfer[transfer_key] = mae
            else:
                self.mae_transfer_groups[transfer_key] = mae
        else:
            base_key = self._encode_base_key(port.name, start_time)
            if isinstance(mae, float):
                self.mae_base[base_key] = mae
            else:
                self.mae_base_groups[base_key] = mae
        if save:
            self.save()

    def remove_mae(self, port: Port, start_time: str, grouped: bool = False, source_port: Port = None,
                   config_uid: bool = None, save: bool = True) -> None:
        if source_port is not None:
            transfer_key = self._encode_transfer_key(source_port.name, port.name, start_time, config_uid)
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

    def get_groups(self, port: Port, start_time: str, source_port: Port = None,
                   config_uid: int = None) -> List[Tuple[int, int, int, float, str]]:
        if source_port is not None and config_uid is not None:
            transfer_key = self._encode_transfer_key(source_port.name, port.name, start_time, config_uid)
            return self.mae_transfer_groups[transfer_key]
        else:
            base_key = self._encode_base_key(port.name, start_time)
            return self.mae_base_groups[base_key]

    def plot_grouped_mae(self, port: Union[str, Port], source_port: Union[str, Port] = None,
                         config_uid: int = None, start_time: str = None) -> None:
        if isinstance(port, str):
            orig_port = port
            port = self.pm.find_port(port)
            if port is None:
                raise ValueError(f"Unable to associate port with port name '{orig_port}'")
        if isinstance(source_port, str):
            orig_port = source_port
            source_port = self.pm.find_port(source_port)
            if source_port is None:
                raise ValueError(f"Unable to associate port with port name '{orig_port}'")
        training_type = validate_params(source_port, config_uid)

        def _plot(eval_dir, _ti: TrainingIteration, _mae_groups: List[Tuple[int, int, int, float, str]],
                  _config_uid: int = None) -> None:
            path = os.path.join(eval_dir, port.name, encode_grouped_mae_plot(training_type, _ti.start_time,
                                                                             _config_uid))
            title = f"Grouped MAE {training_type}-training: Port {port.name}"
            if training_type == "transfer":
                title = f"{title} (Source port: {_ti.source_port})"
            plot_grouped_maes(_mae_groups, title=title, path=path)

        trainings = self.pm.load_trainings(port, output_dir=self.output_dir, routes_dir=self.routes_dir,
                                           training_type=training_type, filter_config_uid=config_uid)
        if not trainings:
            print(f"No {training_type}-training and config_uid {config_uid} found for port '{port.name}'")
            return

        if training_type == "transfer":
            ti = [t for t in trainings.values() if t.source_port == source_port.name]
        else:
            ti = list(trainings.values())

        if len(ti) > 1:
            print(f"Multiple {training_type}-trainings found for port {port.name}, source_port {source_port} and "
                  f"config {config_uid}. Using latest")
        ti = ti[-1]

        if training_type == "base":
            base_key = self._encode_base_key(port.name, ti.start_time)
            mae_groups = self.mae_base_groups[base_key]
            _plot(self.eval_dir, ti, mae_groups)
        else:
            transfer_key = self._encode_transfer_key(source_port.name, port.name, ti.start_time, config_uid)
            mae_groups = self.mae_transfer_groups[transfer_key]
            _plot(self.eval_dir, ti, mae_groups, config_uid)

    def plot_ports_by_mae(self, config_uid: int = None) -> None:
        training_type = "base" if config_uid is None else "transfer"
        if training_type == "base":
            result = []
            for key, mae in self.mae_base.items():
                port_name, start_time = self._decode_base_key(key)
                result.append((port_name, mae))
        elif training_type == "transfer":
            tmp = {}
            for key, mae in self.mae_transfer.items():
                _, target_port, start_time, c_uid = self._decode_transfer_key(key)
                if config_uid == c_uid:
                    if target_port in tmp:
                        tmp[target_port].append(mae)
                    else:
                        tmp[target_port] = [mae]
            result = [(k, sum(v) / len(v)) for (k, v) in tmp.items()]
        else:
            raise ValueError(f"Unknown training-type '{training_type}'")

        result.sort(key=lambda r: r[1])  # sort by mae
        title = f"MAE from {training_type}-training by port"
        file_name = f"ports-mae_{training_type}-training"
        if training_type == "transfer":
            title = f"Average {title} (Config {config_uid})"
            file_name = f"{file_name}_{config_uid}"
        path = os.path.join(self.output_dir, "eval", f"{file_name}.png")
        plot_ports_by_mae(result, title=title, path=path)

    def plot_transfer_effect(self, port: Union[str, Port], config_uid: int = None, source_port: Union[str, Port] = None,
                             by: str = "source_port") -> None:
        """
        What's the cost of transferring a certain port's model to another port?
        For same port: MAE of transferred- vs. base-model for given config
        :param port: Port, that has a transferred model from another port
        :param config_uid: plot for certain config_uid
        :param source_port: plot for certain source_port
        :param by: specify how to aggregate transfer-bars [source_port, config_uid, mae_groups]
        :return: None
        """
        if isinstance(port, str):
            orig_port = port
            port = self.pm.find_port(port)
            if port is None:
                raise ValueError(f"Unable to associate port with port name '{orig_port}'")
        # load base training
        base_training = self.pm.load_trainings(port, output_dir=self.output_dir, routes_dir=self.routes_dir,
                                               training_type="base")
        assert len(base_training) == 1
        base_training = list(base_training.values())[0]
        base_key = self._encode_base_key(port.name, base_training.start_time)
        base_data = (port.name, self.mae_base[base_key])

        # load all transfer-trainings
        transfer_trainings = self.pm.load_trainings(port, output_dir=self.output_dir, routes_dir=self.routes_dir,
                                                    training_type="transfer", filter_config_uid=config_uid)

        if not transfer_trainings:
            print(f"No transfer-trainings with config '{config_uid}' found for port '{port.name}'")
            return

        def _plot_transfer_effect(_base_data, _transfer_data, _path) -> None:
            _transfer_data.sort(key=lambda r: r[1])  # sort by mae
            plot_transfer_effect(_base_data, _transfer_data, _path, by)

        if by == "source_port" and config_uid is not None:
            transfer_data = []
            for ti in transfer_trainings.values():
                # _, _, _, _, config_uid = decode_keras_model(os.path.split(transfer_train.model_path)[1])
                transfer_key = self._encode_transfer_key(ti.source_port, port.name, ti.start_time, config_uid)
                # transfer_data = self.mae_transfer_groups[transfer_key]
                transfer_data.append((ti.source_port, self.mae_transfer[transfer_key]))
            path = os.path.join(self.output_dir, "eval", f"transfer-effect_{config_uid}_{port.name}.png")
            _plot_transfer_effect(base_data, transfer_data, path)

        elif by == "source_port":
            tmp = {}
            for ti in transfer_trainings.values():
                transfer_key = self._encode_transfer_key(ti.source_port, port.name, ti.start_time, ti.config_uid)
                if ti.source_port in tmp:
                    tmp[ti.source_port].append(self.mae_transfer[transfer_key])
                else:
                    tmp[ti.source_port] = [self.mae_transfer[transfer_key]]
            transfer_data = [(k, sum(v) / len(v)) for k, v in tmp.items()]
            path = os.path.join(self.output_dir, "eval", f"transfer-effect-avg-configs_{port.name}.png")
            _plot_transfer_effect(base_data, transfer_data, path)

        elif by == "config_uid":
            tmp = {}
            for ti in transfer_trainings.values():
                transfer_key = self._encode_transfer_key(ti.source_port, port.name, ti.start_time, ti.config_uid)
                if ti.config_uid in tmp:
                    tmp[ti.config_uid].append(self.mae_transfer[transfer_key])
                else:
                    tmp[ti.config_uid] = [self.mae_transfer[transfer_key]]
            transfer_data = [(str(k), sum(v) / len(v)) for k, v in tmp.items()]
            path = os.path.join(self.output_dir, "eval", f"transfer-effect-avg-source-ports_{port.name}.png")
            _plot_transfer_effect(base_data, transfer_data, path)

        elif by == "mae_groups" and config_uid is not None and source_port is not None:
            if isinstance(source_port, str):
                orig_port = source_port
                source_port = self.pm.find_port(source_port)
                if source_port is None:
                    raise ValueError(f"Unable to associate port with port name '{orig_port}'")
            ti = [t for t in transfer_trainings.values() if t.source_port == source_port.name]
            assert len(ti) == 1
            ti = ti[0]
            transfer_key = self._encode_transfer_key(ti.source_port, port.name, ti.start_time, config_uid)
            transfer_data = self.mae_transfer_groups[transfer_key]
            path = os.path.join(self.output_dir, "eval",
                                f"transfer-effect-{config_uid}_{ti.source_port}-{port.name}.png")
            plot_grouped_transfer_effect(self.mae_base_groups[base_key], transfer_data, ti.source_port, port.name, path)

    def plot_transfer_effects(self, sort: str = "mae_base") -> None:
        """
        MAE of transferred- vs base-model for all ports with matching trainings of type 'base' and 'transfer'
        :param sort: How to sort result data. Options [mae_base, num_data]
        :return: None
        """
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

        results = [(c_uid, [compute_metrics(port, val) for port, val in item.items()]) for c_uid, item in tmp.items()]

        for result in results:
            if sort == "mae_base":
                result[1].sort(key=lambda r: r[0])
            data = list(map(list, zip(*result[1])))

            path = os.path.join(self.output_dir, "eval", f"transfer-effects-{result[0]}_{sort}.png")
            plot_transfer_effects(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
                                  data[9], data[10], path)

    def plot_mae_heatmap(self, config_uid: int, start_time: str = None) -> None:
        # TODO: Select certain training with specified start_time
        def map_base_data(_data: Tuple[str, float],
                          time_as_datetime: bool = True) -> Tuple[str, Union[datetime, str], float]:
            _port, start = self._decode_base_key(_data[0])
            if time_as_datetime:
                start = as_datetime(start)
            return _port, start, _data[1]

        tmp = {}
        for item in self.mae_base.items():
            port, start_time, mae = map_base_data(item)
            if port in tmp:
                if tmp[port][0] < start_time:
                    tmp[port] = (start_time, mae)
            else:
                tmp[port] = (start_time, mae)

        # find corresponding transfer-mae
        reverse_ports = list(tmp.keys())
        reverse_ports.reverse()
        data = []
        for i, (port, val) in enumerate(tmp.items()):
            row = []
            for target_port in reverse_ports:
                if target_port == port:
                    row.append(val[1])
                else:
                    transfer_key = self._encode_transfer_key(port, target_port, as_str(val[0]), config_uid)
                    row.append(self.mae_transfer[transfer_key])
            data.append(row)
        data = np.array(data)
        data = data.transpose()

        path = os.path.join(self.output_dir, "eval", f"transfer-matrix_{config_uid}.png")
        plot_mae_heatmap(data, list(tmp.keys()), config_uid, path)

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
        e.reset(options=["transfer"])
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
    elif command == "adapt_paths":
        e = Evaluator.load(eval_dir_or_path=os.path.join(args.output_dir, "eval"), output_dir=args.output_dir,
                           routes_dir=args.routes_dir)
        # print(f"path: {e.path}")
        # print(f"output dir: {e.output_dir}")
        # print(f"routes dir: {e.routes_dir}")
        e.save()
    elif command == "plot":
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))

        # ports by base-training mae
        e.plot_ports_by_mae()

        # ports by transfer-training mae for each config_uid
        for i in [1, 5, 6, 7, 8]:
            e.plot_ports_by_mae(i)

        # transfer effect for port X
        port = e.pm.find_port("rostock")
        e.plot_transfer_effect(port)

        # transfer effect by aggregation methods
        config_uid = 1
        source_port = e.pm.find_port("skagen")
        e.plot_transfer_effect(port, config_uid, source_port, by="source_port")
        e.plot_transfer_effect(port, source_port=source_port, by="source_port")
        e.plot_transfer_effect(port, source_port=source_port, by="config_uid")
        e.plot_transfer_effect(port, config_uid, source_port, by="mae_groups")

        e.plot_grouped_mae(port)
        # e.plot_grouped_mae(port, source_port=source_port)  # test, if err
        for i in [1, 5, 6, 7, 8]:
            e.plot_grouped_mae(port, source_port=source_port, config_uid=i)
        e.plot_mae_heatmap(7)
        e.plot_transfer_effects()
    elif command == "set":  # set mae manually in case sth went wrong
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))
        ports = ["ESBJERG", "ROSTOCK", "KIEL", "SKAGEN", "TRELLEBORG", "THYBORON", "HIRTSHALS", "HVIDESANDE",
                 "AALBORG", "GOTEBORG", "COPENHAGEN", "GRENAA"]
        maes = [65., 359., 476., 26., 6., 63., 6., 12., 53., 184., 30., 57.]
        starts = ["20210531-151223", "20210531-151917", "20210531-153035", "20210531-154831", "20210531-155238",
                  "20210531-160259", "20210531-161357", "20210531-162304", "20210531-163250", "20210531-164118",
                  "20210531-164917", "20210531-165908"]
        assert len(maes) == len(ports) == len(starts)
        for i, p in enumerate(ports):
            port = e.pm.find_port(p)
            if port is None:
                raise ValueError(f"Unable to associate port with port name '{p}'")
            e.set_mae(port, starts[i], maes[i], save=False)
        # save after everything is done, in case something goes wrong in the process
        print(f"{e.mae_base}")
        # e.save()
    elif command == "reload_base_mae":
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))

        for key in e.mae_base.keys():
            port, start_time = e._decode_base_key(key)
            port = e.pm.find_port(port)
            # load history
            path = os.path.join(args.output_dir, "data", port.name, f"history_base_{port.name}_{start_time}.npy")
            history = np.load(path, allow_pickle=True)
            mae = min(history.item().get("val_mae"))
            e.set_mae(port, start_time, mae, save=False)
        e.save()
    elif command == "reload_transfer_mae":
        e = Evaluator.load(os.path.join(args.output_dir, "eval"))

        for key in e.mae_transfer.keys():
            source_port, target_port, start_time, config_uid = e._decode_transfer_key(key)
            source_port = e.pm.find_port(source_port)
            target_port = e.pm.find_port(target_port)
            # load history
            path = os.path.join(args.output_dir, "data", target_port.name,
                                f"history_transfer-{config_uid}_{target_port.name}-{source_port.name}_{start_time}.npy")
            history = np.load(path, allow_pickle=True)
            mae = min(history[0]["val_mae"])
            tune_mae = min(history[1]["val_mae"])
            mae = mae if mae < tune_mae else tune_mae
            e.set_mae(target_port, start_time, mae, source_port, config_uid, save=False)
        e.save()
    else:
        raise ValueError(f"Unknown command '{command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "eval", "adapt_paths", "reset_mae", "plot", "set", "reset",
                                            "reload_base_mae", "reload_transfer_mae"])
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
