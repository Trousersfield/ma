import argparse
import numpy as np
import os
import torch

from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model

from dataset import RoutesDirectoryDataset
from logger import Logger
# from net.model import InceptionTimeModel
# from net.trainer import train_loop, validate_loop, make_train_step, make_training_checkpoint, conclude_training
from inception.eval import Evaluator
from inception.trainer import load_data
from inception.model import inception_time
from port import Port, PortManager
from util import as_datetime, as_str, decode_model_file, encode_transfer_result_file, verify_output_dir,\
    encode_model_file, num_total_trainable_parameters, num_total_parameters, encode_dataset_config_file, read_json,\
    encode_keras_model, decode_keras_model, encode_history_file

script_dir = os.path.abspath(os.path.dirname(__file__))


class TransferConfig:
    def __init__(self, uid: int, desc: str, nth_subset: int, train_layers: List[str]) -> None:
        self.uid = uid,
        self.desc = desc,
        self.nth_subset = nth_subset,
        self.train_layers = train_layers


class TransferDefinition:
    def __init__(self, base_port_name: str, base_model_path: str, target_port_name: str,
                 target_routes_dir: str, target_model_dir: str, target_output_data_dir: str, target_plot_dir: str,
                 target_log_dir: str) -> None:
        self.base_port_name = base_port_name
        self.base_model_path = base_model_path
        self.target_port_name = target_port_name
        self.target_routes_dir = target_routes_dir
        self.target_model_dir = target_model_dir
        self.target_output_data_dir = target_output_data_dir
        self.target_plot_dir = target_plot_dir
        self.target_log_dir = target_log_dir


class TransferResult:
    def __init__(self, path: str, transfer_definition: TransferDefinition, start: datetime,
                 loss_history_path: str, model_path: str, plot_path: str) -> None:
        self.path = path
        self.transfer_definition = transfer_definition
        self.start = start
        self.loss_history_path = loss_history_path
        self.model_path = model_path
        self.plot_path = plot_path

    def save(self) -> None:
        torch.save({
            "path": self.path,
            "transfer_definition": self.transfer_definition,
            "start": self.start,
            "loss_history_path": self.loss_history_path,
            "model_path": self.model_path,
            "plot_path": self.plot_path
        }, self.path)

    @staticmethod
    def load(path: str) -> 'TransferResult':
        if os.path.exists(path):
            file = torch.load(path)
            result = TransferResult(path=file["path"],
                                    transfer_definition=file["transfer_definition"],
                                    start=file["start"],
                                    loss_history_path=file["loss_history_path"],
                                    model_path=file["model_path"],
                                    plot_path=file["plot_path"])
            return result


class TransferManager:
    def __init__(self, config_path: str, routes_dir: str, output_dir: str, transfers: Dict[str, List[str]] = None):
        self.path = os.path.join(script_dir, "TransferManager.tar")
        self.config_path = config_path
        self.routes_dir = routes_dir
        self.output_dir = output_dir
        self.pm = PortManager()
        self.pm.load()
        if len(self.pm.ports.keys()) < 1:
            raise ValueError("No port data available")
        self.transfer_defs = self._generate_transfers()
        self.transfer_configs = self._generate_configs()
        self.transfers = {} if transfers is None else transfers

    def save(self) -> None:
        torch.save({
            "config_path": self.config_path,
            "routes_dir": self.routes_dir,
            "output_dir": self.output_dir,
            "transfers": self.transfers if self.transfers else None
        }, self.path)

    @staticmethod
    def load(path: str) -> 'TransferManager':
        if not os.path.exists(path):
            raise ValueError(f"No TransferManager.tar found at '{path}'")
        state_dict = torch.load(path)
        tm = TransferManager(
            config_path=state_dict["config_path"],
            routes_dir=state_dict["routes_dir"],
            output_dir=state_dict["output_dir"],
            transfers=state_dict["transfers"]
        )
        return tm

    def _is_transferred(self, base_port_name: str, target_port_name: str) -> bool:
        return base_port_name in self.transfers and target_port_name in self.transfers[base_port_name]

    def reset(self, base_port: Union[str, Port] = None, target_port: Union[str, Port] = None) -> None:
        if base_port is not None:
            if isinstance(base_port, str):
                orig_name = base_port
                base_port = self.pm.find_port(base_port)
                if base_port is None:
                    raise ValueError(f"Unable to associate port with port name '{orig_name}'")
            if target_port is not None:
                if isinstance(target_port, str):
                    orig_name = target_port
                    target_port = self.pm.find_port(target_port)
                    if target_port is None:
                        raise ValueError(f"Unable to associate port with port name '{orig_name}'")
                self.transfers[base_port.name].remove(target_port.name)
            else:
                del self.transfers[base_port.name]
        else:
            self.transfers = {}
        self.save()

    # def transfer(self, source_port_name: str) -> None:
    def transfer(self, target_port: Port, evaluator: Evaluator) -> None:
        """
        Transfer models to target port
        :param target_port: port for which to train transfer-model
        :param evaluator: evaluator instance to store results
        :return: None
        """
        if target_port.name not in self.transfer_defs:
            print(f"No transfer definition found for target port '{target_port.name}'")
            return
        tds = self.transfer_defs[target_port.name]
        output_dir = os.path.join(script_dir, os.pardir, "output")
        training_type = "transfer"
        print(f"Transferring models to target port '{target_port.name}'")
        print(f"Loading data...")
        window_width = 50
        num_epochs = 25
        lr = 0.01
        batch_size = 1024
        X_ts, y_ts = load_data(target_port, window_width)
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_ts, y_ts, test_size=0.2,
                                                                                random_state=42, shuffle=False)

        for td in tds:
            print(f".:'`!`':. TRANSFERRING PORT {td.base_port_name} TO {td.target_port_name} .:'`!`':.")
            print(f"- - Epochs {num_epochs} </> Training shape {X_train_orig.shape} </> Learning rate {lr} - -")
            print(f"- - Window width {window_width} </> Batch size {batch_size} - -")
            # print(f"- - Number of model's parameters {num_total_trainable_parameters(model)} device {device} - -")
            base_port = self.pm.find_port(td.base_port_name)
            if base_port is None:
                raise ValueError(f"Unable to associate port with port name '{td.base_port_name}'")

            # model = inception_time(input_shape=(window_width, 37))
            # print(model.summary())

            # apply transfer config
            for config in self.transfer_configs:
                print(f"\n.:'':. APPLYING CONFIG {config.uid} ::'':.")
                _, _, start_time, _, _ = decode_keras_model(os.path.split(td.base_model_path)[1])
                model_file_name = encode_keras_model(td.target_port_name, start_time, td.base_port_name, config.uid)
                file_path = os.path.join(output_dir, "model", td.target_port_name, model_file_name)

                checkpoint = ModelCheckpoint(file_path, monitor='val_mae', mode='min', verbose=2, save_best_only=True)
                early = EarlyStopping(monitor="val_mae", mode="min", patience=10, verbose=2)
                redonplat = ReduceLROnPlateau(monitor="val_mae", mode="min", patience=3, verbose=2)
                callbacks_list = [checkpoint, early, redonplat]

                # optimizer = Adam(learning_rate=lr)
                #
                # # configure model
                # model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

                # load base model
                model = load_model(td.base_model_path)
                # del model

                X_train = X_train_orig
                X_test = X_test_orig
                y_train = y_train_orig
                y_test = y_test_orig

                # apply transfer configuration
                if config.uid == 1:
                    # method 1: cut data
                    X_train = X_train_orig[0::10]
                model.trainable = False
                for layer in config.train_layers:
                    l = model.get_layer(layer).output
                    l.trainable = True

                # transfer model
                result = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1,
                                   validation_data=(X_test, y_test), callbacks=callbacks_list)
                print(f"history:\n{result.history.keys()}")
                train_loss = result.history["loss"]
                train_mae = result.history["mae"]
                val_loss = result.history["val_loss"]
                val_mae = result.history["val_mae"]
                model.load_weights(file_path)

                baseline = mean_absolute_error(y_ts, np.full_like(y_ts, np.mean(y_ts)))
                print(f"naive baseline: {baseline}")

                # set evaluation
                # evaluator.set_mae(port, start_time, val_mae)
                evaluator.set_mae(target_port, start_time, val_mae, base_port, config.uid)
                y_pred = model.predict(X_test)
                print(f"types - y_pred: {type(y_pred)} y_test: {type(y_test)}")
                grouped_mae = evaluator.group_mae(y_test, y_pred)
                evaluator.set_mae(port, start_time, grouped_mae)

                # save history
                history_file_name = encode_history_file(training_type, port.name, start_time, config.uid)
                history_path = os.path.join(output_dir, "data", port.name, history_file_name)
                np.save(history_path, result.history)

                # plot history
                plot_dir = os.path.join(output_dir, "plot")
                plot_history(train_mae, val_mae, plot_dir, port.name, start_time, training_type)
                evaluator.plot_grouped_mae(port, training_type, start_time)
                plot_predictions(y_pred, y_test, plot_dir, port.name, start_time, training_type)

            # if transfer_def.base_port_name in self.transfers:
            #     self.transfers[transfer_def.base_port_name].append(transfer_def.target_port_name)
            # else:
            #     self.transfers[transfer_def.base_port_name] = [transfer_def.target_port_name]
            # self.save()

    def _generate_transfers(self) -> Dict[str, List[TransferDefinition]]:
        """
        Generate TransferDefinitions based on transfer-config.json, containing those ports that have a base training for
        transferring to another port
        :return: Dict of key = target_port_name, val = List of TransferDefinition
        """
        config = read_json(self.config_path)
        transfer_defs = {}

        def _permute(ports: List[str]) -> List[Tuple[str, str]]:
            return [(ports[i], ports[j]) for i in range(len(ports)) for j in range(i+1, len(ports))]
        permutations = _permute(config["ports"])
        print(f"permutations:\n{permutations}")

        for pair in _permute(config["ports"]):
            base_port = self.pm.find_port(pair[0])
            trainings = self.pm.load_trainings(base_port, self.output_dir, self.routes_dir, training_type="base")

            if len(trainings.keys()) < 1:
                print(f"No base-training found for port '{base_port.name}'")
                continue
            print(f"Port {base_port.name} has {len(trainings)} base-trainings. Using latest")
            training = list(trainings.values())[-1]
            for target_port_name in pair[1]:
                target_port = self.pm.find_port(target_port_name)
                if target_port is None:
                    raise ValueError(f"Unable to transfer from port '{base_port.name}'. "
                                     f"No port for '{target_port_name}' found")
                verify_output_dir(self.output_dir, target_port.name)
                td = TransferDefinition(base_port_name=base_port.name,
                                        base_model_path=training.model_path,
                                        target_port_name=target_port.name,
                                        target_routes_dir=os.path.join(self.routes_dir, target_port.name),
                                        target_model_dir=os.path.join(self.output_dir, "model", target_port.name),
                                        target_output_data_dir=os.path.join(self.output_dir, "data", target_port.name),
                                        target_plot_dir=os.path.join(self.output_dir, "plot", target_port.name),
                                        target_log_dir=os.path.join(self.output_dir, "log", target_port.name))
                if base_port.name in transfer_defs:
                    transfer_defs[target_port.name].append(td)
                else:
                    transfer_defs[target_port.name] = [td]
        return transfer_defs

    def _generate_configs(self) -> List[TransferConfig]:
        config = read_json(self.config_path)
        transfer_configs = []
        for conf in config["configs"]:
            c = TransferConfig(uid=conf["uid"],
                               desc=conf["desc"],
                               nth_subset=conf["nth_subset"],
                               train_layers=conf["train_layers"])
            transfer_configs.append(c)
        return transfer_configs


def get_tm(config_path: str, routes_dir: str, output_dir: str) -> 'TransferManager':
    tm_state_path = os.path.join(script_dir, "TransferManager.tar")
    if os.path.exists(tm_state_path):
        return TransferManager.load(tm_state_path)
    else:
        return TransferManager(config_path, routes_dir, output_dir)


def transfer(config_path: str, port_name: str, routes_dir: str, output_dir: str, skip_transferred: bool) -> None:
    tm = get_tm(config_path, routes_dir, output_dir)
    tm.transfer(source_port_name=port_name, skip_transferred=skip_transferred)


def transfer_all(config_path: str, routes_dir: str, output_dir: str, skip_transferred: bool) -> None:
    tm = get_tm(config_path, routes_dir, output_dir)
    for port_name in tm.transfers.keys():
        tm.transfer(source_port_name=port_name, skip_transferred=skip_transferred)


def main(args) -> None:
    if args.command == "transfer":
        transfer_all(args.config_path, args.routes_dir, args.output_dir, args.skip_transferred)
    elif args.command == "transfer_port":
        transfer(args.config_path, args.port_name, args.routes_dir, args.output_dir, args.skip_transferred)
    else:
        raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("command", choices=["transfer", "transfer_port"])
    parser.add_argument("--port_name", type=str, default="all", help="Port to transfer from")
    parser.add_argument("--config_path", type=str, default=os.path.join(script_dir, "transfer-config.json"),
                        help="Path to file for transfer definition generation")
    parser.add_argument("--routes_dir", type=str, default=os.path.join(script_dir, os.pardir, "data", "routes"),
                        help="Path to routes-data directory without port")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, os.pardir, "output"),
                        help="Path to output directory without port")
    parser.add_argument("--skip_transferred", dest="skip_transferred", action="store_true")
    parser.add_argument("--no_skip_transferred", dest="skip_transferred", action="store_false")
    parser.set_defaults(skip_transferred=True)
    main(parser.parse_args())

