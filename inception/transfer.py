import argparse
import gc
import itertools
import numpy as np
import os
import torch

from datetime import datetime
from typing import Dict, List, Tuple, Union

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.layer_utils import count_params

from inception.eval import Evaluator
from inception.trainer import load_data, plot_history, plot_predictions
from port import Port, PortManager
from util import verify_output_dir, read_json, encode_keras_model, decode_keras_model, encode_history_file

script_dir = os.path.abspath(os.path.dirname(__file__))


class TransferConfig:
    def __init__(self, uid: int, desc: str, nth_subset: int, train_layers: List[str], tune: bool) -> None:
        self.uid = uid
        self.desc = desc
        self.nth_subset = nth_subset
        self.train_layers = train_layers
        self.tune = tune


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
    def __init__(self, config_path: str, routes_dir: str, output_dir: str,
                 transfers: Dict[str, List[Tuple[str, int]]] = None):
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

    def _is_transferred(self, target_port: str, source_port: str, config_uid: int) -> bool:
        if target_port in self.transfers:
            return len([t for t in self.transfers[target_port] if t[0] == source_port and t[1] == config_uid]) == 1
        return False

    def set_transfer(self, target_port: str, source_port: str, config_uid: int) -> None:
        if target_port in self.transfers:
            self.transfers[target_port].append((source_port, config_uid))
        else:
            self.transfers[target_port] = [(source_port, config_uid)]
        self.save()

    def reset_transfer(self, target_port: str = None, source_port: str = None, config_uid: int = None) -> None:
        if target_port is not None:
            if source_port is not None:
                if config_uid is not None:
                    indices = [i for i, t in enumerate(self.transfers) if t[0] == source_port and t[1] == config_uid]
                else:
                    indices = [i for i, t in enumerate(self.transfers) if t[0] == source_port]
                [self.transfers[target_port].pop(i) for i in indices]
            else:
                self.transfers[target_port] = []
        else:
            self.transfers = {}
        self.save()

    # def transfer(self, source_port_name: str) -> None:
    def transfer(self, target_port: Port, evaluator: Evaluator, config_uids: List[int] = None) -> None:
        """
        Transfer models to target port
        :param target_port: port for which to train transfer-model
        :param evaluator: evaluator instance to store results
        :param config_uids: specify config_uids to transfer. If none, transfer all
        :return: None
        """
        if target_port.name not in self.transfer_defs:
            print(f"No transfer definition found for target port '{target_port.name}'")
            return
        # transfer definitions for specified target port
        tds = self.transfer_defs[target_port.name]
        output_dir = os.path.join(script_dir, os.pardir, "output")
        training_type = "transfer"
        print(f"TRANSFERRING MODELS TO TARGET PORT '{target_port.name}'")
        if config_uids is not None:
            print(f"Transferring configs -> {config_uids} <-")
        window_width = 50
        num_epochs = 20
        train_lr = 0.01
        fine_tune_lr = 1e-5
        batch_size = 1024

        # skip transferred
        for td in tds:
            for config in self.transfer_configs:
                if self._is_transferred(target_port.name, td.base_port_name, config.uid):
                    print(f"Already transferred ({config.uid}): {td.base_port_name} -> {target_port.name}. Skipping")
                    return
        print(f"Loading data ...")
        X_ts, y_ts = load_data(target_port, window_width)
        # X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_ts, y_ts, test_size=0.2,
        #                                                                         random_state=42, shuffle=False)
        # train_optimizer = Adam(learning_rate=train_lr)
        # fine_tune_optimizer = Adam(learning_rate=fine_tune_lr)

        for td in tds:
            print(f".:'`!`':. TRANSFERRING PORT {td.base_port_name} TO {td.target_port_name} .:'`!`':.")
            print(f"- - Epochs {num_epochs} </>  </> Learning rate {train_lr} - -")
            print(f"- - Window width {window_width} </> Batch size {batch_size} - -")
            # print(f"- - Number of model's parameters {num_total_trainable_parameters(model)} device {device} - -")
            base_port = self.pm.find_port(td.base_port_name)
            if base_port is None:
                raise ValueError(f"Unable to associate port with port name '{td.base_port_name}'")

            # model = inception_time(input_shape=(window_width, 37))
            # print(model.summary())

            # apply transfer config
            for config in self.transfer_configs:
                if config_uids is not None and config.uid not in config_uids:
                    continue
                print(f"\n.:'':. APPLYING CONFIG {config.uid} ::'':.")
                print(f"-> -> {config.desc} <- <-")
                print(f"-> -> nth_subset: {config.nth_subset} <- <-")
                print(f"-> -> trainable layers: {config.train_layers} <- <-")
                _, _, start_time, _, _ = decode_keras_model(os.path.split(td.base_model_path)[1])
                model_file_name = encode_keras_model(td.target_port_name, start_time, td.base_port_name, config.uid)
                file_path = os.path.join(output_dir, "model", td.target_port_name, model_file_name)

                X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_ts, y_ts, test_size=0.2,
                                                                                        random_state=42, shuffle=False)
                print(f"Training shape {X_train_orig.shape}")
                train_optimizer = Adam(learning_rate=train_lr)
                fine_tune_optimizer = Adam(learning_rate=fine_tune_lr)

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
                # if config.uid == 0:
                #     print(model.summary())
                # else:
                #     print(model.summary())
                # del model

                X_train = X_train_orig
                X_test = X_test_orig
                y_train = y_train_orig
                y_test = y_test_orig

                # apply transfer configuration
                if config.nth_subset > 1:
                    if X_train.shape[0] < config.nth_subset:
                        print(f"Unable to apply nth-subset. Not enough data")
                    X_train = X_train_orig[0::config.nth_subset]
                    X_test = X_test_orig[0::config.nth_subset]
                    y_train = y_train_orig[0::config.nth_subset]
                    y_test = y_test_orig[0::config.nth_subset]
                    print(f"Orig shape: {X_train_orig.shape} {config.nth_subset} th-subset shape: {X_train.shape}")
                    print(f"Orig shape: {X_test_orig.shape} {config.nth_subset} th-subset shape: {X_test.shape}")
                    print(f"Orig shape: {y_train_orig.shape} {config.nth_subset} th-subset shape: {y_train.shape}")
                    print(f"Orig shape: {y_test_orig.shape} {config.nth_subset} th-subset shape: {y_test.shape}")
                modified = False
                # freeze certain layers
                for layer in model.layers:
                    if layer.name not in config.train_layers:
                        modified = True
                        layer.trainable = False
                if modified:
                    # re-compile
                    model.compile(optimizer=train_optimizer, loss="mse", metrics=["mae"])
                # trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
                # non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
                trainable_count = count_params(model.trainable_weights)
                non_trainable_count = count_params(model.non_trainable_weights)
                print(f"Total params: {trainable_count + non_trainable_count}")
                print(f"Trainable params: {trainable_count}")
                print(f"Non trainable params: {non_trainable_count}")

                # transfer model
                result = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1,
                                   validation_data=(X_test, y_test), callbacks=callbacks_list)
                train_mae = result.history["mae"]
                val_mae = result.history["val_mae"]

                if config.tune:
                    print(f"Tuning transferred model")
                    # apply fine-tuning: unfreeze all but batch-normalization layers!
                    for layer in model.layers:
                        if not layer.name.startswith("batch_normalization"):
                            layer.trainable = True
                    model.compile(optimizer=fine_tune_optimizer, loss="mse", metrics=["mae"])
                    print(f"model for fine tuning")
                    print(model.summary())
                    result = model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose=1,
                                       validation_data=(X_test, y_test), callbacks=callbacks_list)

                tune_train_mae = result.history["mae"] if config.tune else None
                tune_val_mae = result.history["val_mae"] if config.tune else None
                model.load_weights(file_path)

                baseline = mean_absolute_error(y_ts, np.full_like(y_ts, np.mean(y_ts)))
                print(f"naive baseline: {baseline}")

                # set evaluation
                # evaluator.set_mae(port, start_time, val_mae)
                evaluator.set_mae(target_port, start_time, val_mae, base_port, config.uid)
                y_pred = model.predict(X_test)
                print(f"types - y_pred: {type(y_pred)} y_test: {type(y_test)}")
                grouped_mae = evaluator.group_mae(y_test, y_pred)
                evaluator.set_mae(target_port, start_time, grouped_mae, base_port, config.uid)

                # save history
                history_file_name = encode_history_file(training_type, target_port.name, start_time, config.uid)
                history_path = os.path.join(output_dir, "data", target_port.name, history_file_name)
                np.save(history_path, result.history)

                # plot history
                plot_dir = os.path.join(output_dir, "plot")
                plot_history(train_mae, val_mae, plot_dir, target_port.name, start_time, training_type,
                             td.base_port_name, config.uid, tune_train_mae, tune_val_mae)
                evaluator.plot_grouped_mae(target_port, training_type, start_time, config.uid)
                plot_predictions(y_pred, y_test, plot_dir, target_port.name, start_time, training_type,
                                 td.base_port_name, config.uid)
                self.set_transfer(target_port.name, td.base_port_name, config.uid)
                del X_train_orig, X_test_orig, y_train_orig, y_test_orig, model, X_train, y_train, X_test, y_test
        del X_ts, y_ts
        gc.collect()

    def _generate_transfers(self) -> Dict[str, List[TransferDefinition]]:
        """
        Generate TransferDefinitions based on transfer-config.json, containing those ports that have a base training for
        transferring to another port
        :return: Dict of key = target_port_name, val = List of TransferDefinition
        """
        config = read_json(self.config_path)
        transfer_defs = {}

        # def _permute(ports: List[str]) -> List[Tuple[str, str]]:
        #     return [(ports[i], ports[j]) for i in range(len(ports)) for j in range(len(ports))]
        # permutations = _permute(config["ports"])
        ports = list(config["ports"])
        permutations = list(itertools.permutations(ports, r=2))

        # for pair in _permute(config["ports"]):
        for pair in permutations:
            base_port, target_port = self.pm.find_port(pair[0]), self.pm.find_port(pair[1])
            if target_port is None:
                raise ValueError(f"No port found: Unable to transfer from base-port with name '{base_port.name}'")
            if target_port is None:
                raise ValueError(f"No port found: Unable to transfer to target-port with name '{pair[1]}'")

            trainings = self.pm.load_trainings(base_port, self.output_dir, self.routes_dir, training_type="base")
            if len(trainings.keys()) < 1:
                print(f"No base-training found for port '{base_port.name}'. Skipping")
                continue

            training = list(trainings.values())[-1][0]
            # print(f"Pair {base_port.name} ({len(trainings)} base-trains) -> {target_port.name}. "
            #       f"Using latest at '{training.start_time}'")
            verify_output_dir(self.output_dir, target_port.name)
            td = TransferDefinition(base_port_name=base_port.name,
                                    base_model_path=training.model_path,
                                    target_port_name=target_port.name,
                                    target_routes_dir=os.path.join(self.routes_dir, target_port.name),
                                    target_model_dir=os.path.join(self.output_dir, "model", target_port.name),
                                    target_output_data_dir=os.path.join(self.output_dir, "data", target_port.name),
                                    target_plot_dir=os.path.join(self.output_dir, "plot", target_port.name),
                                    target_log_dir=os.path.join(self.output_dir, "log", target_port.name))
            # print(f"keys: {transfer_defs.keys()}")
            name = target_port.name
            # print(f"check: {target_port.name in transfer_defs} vs. {name in transfer_defs}")
            if name in transfer_defs:
                transfer_defs[target_port.name].append(td)
            else:
                transfer_defs[target_port.name] = [td]
        return transfer_defs

    def _generate_configs(self) -> List[TransferConfig]:
        config = read_json(self.config_path)

        def _make_config(uid: str, desc: str, nth_subset: str, train_layers: List[str], tune: bool) -> TransferConfig:
            uid = int(uid)
            nth_subset = int(nth_subset)
            return TransferConfig(uid, desc, nth_subset, train_layers, tune)
        configs = [_make_config(c["uid"], c["desc"], c["nth_subset"], c["train_layers"], c["tune"])
                   for c in config["configs"]]
        return configs


def get_tm(config_path: str, routes_dir: str, output_dir: str) -> 'TransferManager':
    tm_state_path = os.path.join(script_dir, "TransferManager.tar")
    if os.path.exists(tm_state_path):
        return TransferManager.load(tm_state_path)
    else:
        tm = TransferManager(config_path, routes_dir, output_dir)
        tm.save()
        return tm


def transfer_all(config_path: str, routes_dir: str, output_dir: str, config_uids: List[int] = None) -> None:
    tm = get_tm(config_path, routes_dir, output_dir)
    e = Evaluator.load(os.path.join(output_dir, "eval", "evaluator.tar"))
    for target_port in tm.transfer_defs.keys():
        tm.transfer(tm.pm.find_port(target_port), e)


def main(args) -> None:
    if args.command == "transfer":
        transfer_all(args.config_path, args.routes_dir, args.output_dir, args.config_uids)
    else:
        raise ValueError(f"Unknown command '{args.command}'")


def parse_list_arg(arg) -> List[int]:
    return list(map(int, arg.split(",")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("command", choices=["transfer"])
    parser.add_argument("--config_path", type=str, default=os.path.join(script_dir, "transfer-config.json"),
                        help="Path to file for transfer definition generation")
    parser.add_argument("--routes_dir", type=str, default=os.path.join(script_dir, os.pardir, "data", "routes"),
                        help="Path to routes-data directory without port")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, os.pardir, "output"),
                        help="Path to output directory without port")
    parser.add_argument("--config_uids", type=parse_list_arg)
    main(parser.parse_args())
