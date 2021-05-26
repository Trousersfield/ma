import argparse
import os
import torch

from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, List, Union

from dataset import RoutesDirectoryDataset
from logger import Logger
from net.model import InceptionTimeModel
from net.trainer import train_loop, validate_loop, make_train_step, make_training_checkpoint, conclude_training
from port import Port, PortManager
from util import as_datetime, as_str, decode_model_file, encode_transfer_result_file, verify_output_dir,\
    encode_model_file, num_total_trainable_parameters, num_total_parameters, encode_dataset_config_file, read_json

script_dir = os.path.abspath(os.path.dirname(__file__))


class TransferDefinition:
    def __init__(self, base_port_name: str, base_model_path: str, target_port_name: str, target_routes_dir: str,
                 target_model_dir: str, target_output_data_dir: str, target_plot_dir: str, target_log_dir: str,
                 learning_rate: float = .00025):
        if learning_rate > .001:
            raise ValueError(f"Max value for parameter learning_rate is 0.001 (got {learning_rate})")
        self.base_port_name = base_port_name
        self.base_model_path = base_model_path
        self.target_port_name = target_port_name
        self.target_routes_dir = target_routes_dir
        self.target_model_dir = target_model_dir
        self.target_output_data_dir = target_output_data_dir
        self.target_plot_dir = target_plot_dir
        self.target_log_dir = target_log_dir
        self.learning_rate = learning_rate


class TransferResult:
    def __init__(self, path: str, transfer_definition: TransferDefinition, start: datetime, end: datetime,
                 loss_history_path: str, model_path: str, plot_path: str) -> None:
        self.path = path
        self.transfer_definition = transfer_definition
        self.start = start
        self.end = end
        self.loss_history_path = loss_history_path
        self.model_path = model_path
        self.plot_path = plot_path

    def save(self) -> None:
        torch.save({
            "path": self.path,
            "transfer_definition": self.transfer_definition,
            "start": self.start,
            "end": self.end,
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
                                    end=file["end"],
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

    def transfer(self, source_port_name: str, skip_transferred: bool = True) -> None:
        source_port = self.pm.find_port(source_port_name)
        if source_port is None:
            print(f"No port found for port name '{source_port_name}'")
            return
        if source_port.name in self.transfer_defs:
            transfer_defs = self.transfer_defs[source_port.name]
        else:
            raise ValueError(f"No transfer definition found for port '{source_port.name}'. Make sure config contains "
                             f"transfer definition for '{source_port.name}' and has a base-training model")

        # transfer base model to each port specified in transfer definition
        for transfer_def in transfer_defs:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            training_type = "transfer"
            logger = Logger(training_type, transfer_def.target_log_dir)
            batch_size = 64
            window_width = 128
            # load start_time according to base model for association of models
            _, _, start_time, _, _ = decode_model_file(os.path.split(transfer_def.base_model_path)[1])
            port = self.pm.find_port(transfer_def.target_port_name)
            if port is None:
                raise ValueError(f"Unable to associate port with port name '{transfer_def.target_port_name}'")
            if not os.path.exists(transfer_def.target_routes_dir):
                print(f"Skipping transfer {transfer_def.base_port_name} -> {transfer_def.target_port_name}: No routes")
                continue
            if skip_transferred and self._is_transferred(transfer_def.base_port_name, transfer_def.target_port_name):
                print(f"Skipping transfer {transfer_def.base_port_name} -> {transfer_def.target_port_name}: "
                      f"Already transferred")
                continue

            dataset = RoutesDirectoryDataset(data_dir=transfer_def.target_routes_dir, start_time=start_time,
                                             training_type=training_type, batch_size=batch_size, start=0,
                                             window_width=window_width)
            dataset_file_name = encode_dataset_config_file(start_time, file_type="transfer")
            dataset_config_path = os.path.join(transfer_def.target_routes_dir, dataset_file_name)
            if not os.path.exists(dataset_config_path):
                dataset.save_config()
            else:
                if not os.path.exists(dataset_config_path):
                    raise FileNotFoundError(f"Unable to transfer: No dataset config found at {dataset_config_path}")
                dataset = RoutesDirectoryDataset.load_from_config(dataset_config_path)
            end_train = int(.8 * len(dataset))
            if not (len(dataset) - end_train) % 2 == 0 and end_train < len(dataset):
                end_train += 1
            end_validate = int(len(dataset) - ((len(dataset) - end_train) / 2))

            # use initialized dataset's config for consistent split
            train_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, start=0, end=end_train)
            validate_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, start=end_train,
                                                                       end=end_validate)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, drop_last=False, pin_memory=True,
                                                       num_workers=1)
            validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=None, drop_last=False,
                                                          pin_memory=True, num_workers=1)

            model = InceptionTimeModel.load(transfer_def.base_model_path, device=device)
            model.freeze_inception()
            # TODO: optimizer
            # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
            #                                                lr=transfer_def.learning_rate)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=transfer_def.learning_rate)

            num_epochs = 10
            loss_history = ([], [])
            elapsed_time_history = []
            criterion: torch.nn.MSELoss = torch.nn.MSELoss()
            min_val_idx = 0

            print(f".:'`!`':. TRANSFERRING PORT {transfer_def.base_port_name} TO {transfer_def.target_port_name} .:'`!`"
                  f"':.")
            print(f"- - Epochs {num_epochs} </> Training examples {len(train_loader)} </> "
                  f"Learning rate {transfer_def.learning_rate} - -")
            print(f"- - Weight decay {0} Window width {window_width} </> Batch size {batch_size} - -")
            print(f"- - Number of model's parameters {num_total_trainable_parameters(model)} device {device} - -")
            logger.write(f"{port.name}-model\n"
                         f"Number of epochs: {num_epochs}\n"
                         f"Learning rate: {transfer_def.learning_rate}\n"
                         f"Total number of parameters: {num_total_parameters(model)}\n"
                         f"Total number of trainable parameters: {num_total_trainable_parameters(model)}")
            # transfer loop
            for epoch in range(num_epochs):
                # re-train model-parameters with requires_grad == True
                print(f"->->->->-> Epoch ({epoch + 1}/{num_epochs}) <-<-<-<-<-<-")
                avg_train_loss, elapsed_time = train_loop(criterion=criterion, model=model, device=device,
                                                          optimizer=optimizer, loader=train_loader)
                loss_history[0].append(avg_train_loss)
                elapsed_time_history.append(elapsed_time)

                # validate model
                avg_validation_loss = validate_loop(criterion=criterion, device=device, model=model,
                                                    optimizer=optimizer, loader=validate_loader)
                loss_history[1].append(avg_validation_loss)

                # check if current model has lowest validation loss (= is current optimal model)
                if loss_history[1][epoch] < loss_history[1][min_val_idx]:
                    min_val_idx = epoch

                logger.write(f"Epoch {epoch + 1}/{num_epochs}:\n"
                             f"\tAvg train loss {avg_train_loss}\n"
                             f"\tAvg val   loss {avg_validation_loss}")

                make_training_checkpoint(model=model, model_dir=transfer_def.target_model_dir, port=port,
                                         start_time=start_time, num_epochs=num_epochs,
                                         learning_rate=transfer_def.learning_rate, weight_decay=.0,
                                         num_train_examples=len(train_loader), loss_history=loss_history,
                                         elapsed_time_history=elapsed_time_history, optimizer=optimizer,
                                         is_optimum=min_val_idx == epoch, base_port_name=transfer_def.base_port_name)
                print(f">>>> Avg losses (MSE) - Train: {avg_train_loss} Validation: {avg_validation_loss} <<<<\n")

            # conclude transfer
            conclude_training(loss_history=loss_history, data_dir=transfer_def.target_output_data_dir,
                              plot_dir=transfer_def.target_plot_dir, port=port, start_time=start_time,
                              elapsed_time_history=elapsed_time_history, plot_title="Transfer loss",
                              training_type=training_type)

            if transfer_def.base_port_name in self.transfers:
                self.transfers[transfer_def.base_port_name].append(transfer_def.target_port_name)
            else:
                self.transfers[transfer_def.base_port_name] = [transfer_def.target_port_name]
            self.save()

    def _generate_transfers(self) -> Dict[str, List[TransferDefinition]]:
        """
        Generate TransferDefinitions based on config.json, containing those ports that have a base training for
        transferring to another port
        :return: Dict of key = port_name, val = List of TransferDefinition
        """
        config = read_json(self.config_path)
        transfer_defs = {}

        for transfer_def in config:
            base_port = self.pm.find_port(transfer_def["base_port"])
            base_port_trainings = self.pm.load_trainings(base_port, self.output_dir, self.routes_dir,
                                                         training_type="base")

            if len(base_port_trainings) == 0:
                print(f"No base-training found for port '{base_port.name}'")
                continue
            print(f"Port {base_port.name} has {len(base_port_trainings)} base-trainings. Using latest")
            base_train = base_port_trainings[-1]
            for target_port_name in transfer_def["target_ports"]:
                target_port = self.pm.find_port(target_port_name)
                if target_port is None:
                    raise ValueError(f"Unable to transfer from port '{base_port.name}'. "
                                     f"No port for '{target_port_name}' found")
                verify_output_dir(self.output_dir, target_port.name)
                td = TransferDefinition(base_port_name=base_port.name,
                                        base_model_path=base_train.model_path,
                                        target_port_name=target_port.name,
                                        target_routes_dir=os.path.join(self.routes_dir, target_port.name),
                                        target_model_dir=os.path.join(self.output_dir, "model", target_port.name),
                                        target_output_data_dir=os.path.join(self.output_dir, "data", target_port.name),
                                        target_plot_dir=os.path.join(self.output_dir, "plot", target_port.name),
                                        target_log_dir=os.path.join(self.output_dir, "log", target_port.name))
                if base_port.name in transfer_defs:
                    transfer_defs[base_port.name].append(td)
                else:
                    transfer_defs[base_port.name] = [td]
        return transfer_defs


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

