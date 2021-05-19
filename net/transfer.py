import argparse
import joblib
import json
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
    encode_model_file, num_total_trainable_parameters, num_total_parameters

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
    def __init__(self, config_path: str, routes_dir: str, output_dir: str):
        self.config_path = config_path
        self.routes_dir = routes_dir
        self.output_dir = output_dir
        self.pm = PortManager()
        self.pm.load()
        if len(self.pm.ports.keys()) < 1:
            raise ValueError("No port data available")
        self.transfer_definitions = self._generate_transfers()
        self.completed_transfers: List[TransferResult] = []

    def transfer(self, port_name: str) -> None:
        port = self.pm.find_port(port_name)
        if port is None:
            print(f"No port found for port name '{port_name}'")
            return
        transfer_def = self._find_transfer(port)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_type = "transfer"
        logger = Logger(training_type, transfer_def.target_log_dir)
        batch_size = 64
        window_width = 128
        _, _, start_time, _ = decode_model_file(os.path.split(transfer_def.base_model_path)[1])
        # start_time = datetime.now()
        # start_time = as_str(start_time)
        port = self.pm.find_port(transfer_def.target_port_name)
        if port is None:
            raise ValueError(f"Unable to associate transfer-target port name {transfer_def.target_port_name}")

        dataset = RoutesDirectoryDataset(data_dir=transfer_def.target_routes_dir, start_time=start_time,
                                         training_type=training_type, batch_size=batch_size, start=0,
                                         window_width=window_width)
        dataset_config_path = os.path.join(transfer_def.target_routes_dir, "transfer_dataset_config.pkl")
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

        # data, target = train_dataset[0]
        # input_dim = data.size(-1)
        # output_dim = 1
        # start_epoch = 0
        num_epochs = 50
        loss_history = ([], [])
        criterion: torch.nn.MSELoss = torch.nn.MSELoss()
        min_val_idx = 0

        print(f".:'`!`':. TRANSFERRING PORT {transfer_def.base_port_name} TO {transfer_def.target_port_name} .:'`!`':.")
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
            avg_train_loss = train_loop(criterion=criterion, model=model, device=device, optimizer=optimizer,
                                        loader=train_loader)
            loss_history[0].append(avg_train_loss)

            # validate model
            avg_validation_loss = validate_loop(criterion=criterion, device=device, model=model, optimizer=optimizer,
                                                loader=validate_loader)
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
                                     loss_history=loss_history, optimizer=optimizer, is_optimum=min_val_idx == epoch,
                                     is_transfer=True)
            print(f">>>> Avg losses - Train: {avg_train_loss} Validation: {avg_validation_loss} <<<<\n")

        # conclude transfer
        end = datetime.now()
        loss_history_path, plot_path = conclude_training(loss_history=loss_history, end=end,
                                                         model_dir=transfer_def.target_model_dir,
                                                         data_dir=transfer_def.target_output_data_dir,
                                                         plot_dir=transfer_def.target_plot_dir, port=port,
                                                         start_time=start_time)

        tr_path = os.path.join(transfer_def.target_model_dir, encode_transfer_result_file(start_time, as_str(end)))
        model_path = os.path.join(transfer_def.target_model_dir, encode_model_file(port.name, start_time, as_str(end),
                                                                                   file_type=training_type))
        result = TransferResult(path=tr_path, transfer_definition=transfer_def, start=as_datetime(start_time), end=end,
                                loss_history_path=loss_history_path, model_path=model_path, plot_path=plot_path)
        self.completed_transfers.append(result)

    # def load(self, transfer_definition_path: str) -> None:
    #     if os.path.exists(transfer_definition_path):
    #         self.transfer_definitions = joblib.load(transfer_definition_path)
    #     else:
    #         raise FileNotFoundError(f"No transfer definition file found at '{transfer_definition_path}'. "
    #                                 f"Make sure to generate definitions are generated.")

    def _generate_transfers(self) -> List[TransferDefinition]:
        config = read_json(self.config_path)
        transfers: List[TransferDefinition] = []

        for transfer_def in config:
            base_port = self.pm.find_port(transfer_def["base_port"])
            base_port_trainings = self.pm.load_trainings(base_port, self.output_dir, self.routes_dir,
                                                         training_type="base")

            if len(base_port_trainings) == 0:
                print(f"No training found for port '{base_port.name}'")
                continue
            print(f"{len(base_port_trainings)} trainings for port {base_port.name} found")

            base_train = base_port_trainings[-1]
            print(f"base training iteration:\n{base_train}")
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
                transfers.append(td)
        return transfers

    def _find_transfer(self, port: Port) -> TransferDefinition:
        for td in self.transfer_definitions:
            if port.name == td.base_port_name:
                return td


def read_json(path: str) -> json:
    if os.path.exists(path):
        with open(path) as json_file:
            result = json.load(json_file)
            return result
    else:
        raise ValueError(f"Unable to read .json file from {path}")


# entry point for transferring models
def transfer(config_path: str, port_name: str, routes_dir: str, output_dir: str) -> None:
    tm = TransferManager(config_path, routes_dir, output_dir)
    tm.transfer(port_name=port_name)


def main(args) -> None:
    if args.command == "transfer":
        transfer(args.config_path, args.port_name, args.routes_dir, args.output_dir)
    else:
        raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("command", choices=["transfer"])
    parser.add_argument("--port_name", type=str, default="all", help="Port to transfer from")
    parser.add_argument("--config_path", type=str, default=os.path.join(script_dir, "transfer", "config.json"),
                        help="Path to file for transfer definition generation")
    parser.add_argument("--routes_dir", type=str, default=os.path.join(script_dir, os.pardir, "data", "routes"),
                        help="Path to routes-data directory without port")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, os.pardir, "output"),
                        help="Path to output directory without port")
    main(parser.parse_args())

