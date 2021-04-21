import argparse
import joblib
import numpy as np
import os
import torch

from datetime import datetime
from typing import Dict, List, Tuple

from loader import MmsiDataFile, TrainingExampleLoader
from logger import Logger
from plotter import plot_series
from port import Port, PortManager
from net.model import InceptionTimeModel
from util import debug_data, encode_model_file, encode_loss_file, encode_loss_plot, as_str, num_total_parameters,\
    num_total_trainable_parameters, decode_model_file

script_dir = os.path.abspath(os.path.dirname(__file__))
torch.set_printoptions(precision=10)


class TrainingCheckpoint:
    def __init__(self, model_path: str, start_time: str, epoch: int, num_epochs: int, learning_rate: float,
                 loss_history: Tuple[List[float], List[float]], optimizer: torch.optim.Adam) -> None:
        self.model_path = model_path
        self.save_path = os.path.join(os.path.split(model_path)[0], "checkpoint.pkl")
        self.start_time = start_time
        self.epoch = epoch
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_history = loss_history
        self.optimizer = optimizer

    def safe(self):
        joblib.dump(self, self.save_path)


# entry point for training models for each port defined in port manager
def train_all(data_dir: str, output_dir: str, debug: bool = False) -> None:
    pm = PortManager()
    pm.load()
    if len(pm.ports.keys()) < 1:
        raise ValueError("No port data available")

    for _, port in pm.ports.items():
        train(port_name=port.name, data_dir=data_dir, output_dir=output_dir, num_epochs=5, learning_rate=.0001, pm=pm,
              resume_checkpoint=False, debug=debug)


def train(port_name: str, data_dir: str, output_dir: str, num_epochs: int = 50, learning_rate: float = .0001,
          pm: PortManager = None, resume_checkpoint: bool = False, debug: bool = False) -> None:
    start_datetime = datetime.now()
    start_time = as_str(start_datetime)
    torch.autograd.set_detect_anomaly(True)
    if pm is None:
        pm = PortManager()
        pm.load()
        if len(pm.ports.keys()) < 1:
            raise ValueError("No port data available")
    port = pm.find_port(port_name)

    output_dirs: Dict[str, str] = {}
    for kind in ["data", "debug", "model", "plot", "log"]:
        curr_dir = os.path.join(output_dir, kind, port.name)
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
        output_dirs[kind] = curr_dir

    routes_path = os.path.join(data_dir, "routes", port.name)
    # eval_dir = os.path.join(output_dir, "eval")

    log_file_name = f"train-log_{port.name}_{start_time}"
    train_logger = Logger(log_file_name, output_dirs["log"])
    if debug:
        debug_logger = Logger(f"train-log_{port.name}_{start_time}_debug", output_dirs["log"])

    # set device: use gpu if available
    # more options: https://pytorch.org/docs/stable/notes/cuda.html
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {port.name}-model. Device: {device}")

    loader = TrainingExampleLoader(routes_path)
    loader.load()

    if len(loader) == 0:
        raise ValueError(f"Unable to load data from directory {loader.data_dir}\n"
                         f"Make sure Data loader is fit!")

    # tc: TrainingCheckpoint = TrainingCheckpoint("", 0)
    data, target = loader[0]
    input_dim = data.shape[2]
    output_dim = 1
    start_epoch = 0
    loss_history = [[], []]
    criterion: torch.nn.MSELoss = torch.nn.MSELoss()

    # resume from a checkpoint if training was aborted
    if resume_checkpoint:
        tc = load_checkpoint(output_dirs["model"])
        model = InceptionTimeModel.load(tc.model_path, device).to(device)
        start_epoch = tc.epoch
        start_time = tc.start_time
        num_epochs = tc.num_epochs
        learning_rate = tc.learning_rate
        loss_history = tc.loss_history
        optimizer: torch.optim.Adam = tc.optimizer
    else:
        model = InceptionTimeModel(num_inception_blocks=1, in_channels=input_dim, out_channels=32,
                                   bottleneck_channels=8, use_residual=True, output_dim=output_dim).to(device)
        # test what happens if using "weight_decay" e.g. with 1e-4
        optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # model.to(device)
    print(f"model:\n{model}")
    print(f"Starting training from directory {loader.data_dir} on {len(loader)} training examples")
    # print(f"First training example: \n{train_loader[0]}")
    train_logger.write(f"{port.name}-model num_epochs: {num_epochs} learning_rate: {learning_rate}")

    # training loop
    for epoch in range(start_epoch, num_epochs):
        loss_train = 0
        loss_validation = 0

        # train model
        model.train()
        train_indices = loader.shuffled_data_indices(kind="train")

        for loop_idx, train_idx in enumerate(train_indices):
            train_data, target = loader[train_idx]
            # print(f"train data:\n{train_data}")
            # print(f"target data:\n{target}")
            data_tensor = torch.Tensor(train_data).to(device)
            target_tensor = torch.Tensor(target).unsqueeze(-1).to(device)
            # print(f"train tensor:\n{data_tensor}")
            # print(f"target tensor:\n{target_tensor}")

            if debug:
                debug_data(data_tensor, target_tensor, train_idx, loader, debug_logger)

            batch_loss = make_train_step(data_tensor, target_tensor, optimizer, model, criterion)
            loss_train += batch_loss

            if loop_idx % 1000 == 0:
                print(f"train_tensor: {data_tensor}")
                print(f"target_tensor: {target_tensor}")
                print(f"loop idx: {loop_idx} batch loss: {batch_loss}")
                print(f"loop idx: {loop_idx} loss_train: {loss_train}")

        avg_train_loss = loss_train / len(loader.train_indices)
        loss_history[0].append(avg_train_loss)

        print(f"epoch: {epoch} avg train loss: {avg_train_loss}")

        # validate model
        model.eval()
        with torch.no_grad():
            validate_indices = loader.shuffled_data_indices(kind="validate")

            for validate_idx in validate_indices:
                validate_data, target = loader[validate_idx]
                validate_tensor = torch.Tensor(validate_data).to(device)
                target_tensor = torch.Tensor(target).unsqueeze(-1).to(device)
                if debug:
                    debug_data(validate_tensor, target_tensor, validate_idx, loader,
                               debug_logger, "Validation")

                batch_loss = make_train_step(validate_tensor, target_tensor, optimizer, model, criterion,
                                             training=False)
                loss_validation += batch_loss
        avg_validation_loss = loss_validation / len(loader.validate_indices)
        loss_history[1].append(avg_validation_loss)

        make_training_checkpoint(model=model, model_dir=output_dirs["model"], port=port, start_time=start_time,
                                 epoch=epoch, num_epochs=num_epochs, learning_rate=learning_rate,
                                 loss_history=loss_history, optimizer=optimizer)
        print(f"epoch: {epoch} avg val loss: {avg_validation_loss}")

        train_logger.write(f"Epoch {epoch + 1}/{num_epochs}:\n"
                           f"\tAvg train loss {avg_train_loss}\n"
                           f"\tAvg val loss   {avg_validation_loss}")

    end_datetime = datetime.now()
    end_time = as_str(end_datetime)
    model_path = os.path.join(output_dirs["model"], encode_model_file(port.name, start_time, end_time))
    model.save(model_path)
    loss_history_path = os.path.join(output_dirs["data"], encode_loss_file(port.name, end_time))
    np.save(loss_history_path, loss_history)
    pm.add_training(port, start_datetime, end_datetime, model_path, loss_history_path,
                    os.path.join(output_dirs["log"], f"{log_file_name}.txt"))

    train_logger.write(f"Total number of parameters: {num_total_parameters(model)}\n"
                       f"Total number of trainable parameters: {num_total_trainable_parameters(model)}")
    plot_series(series=loss_history, x_label="Epoch", y_label="Loss",
                legend_labels=["Training", "Validation"], x_ticks=1.,
                path=os.path.join(output_dirs["plot"], encode_loss_plot(port.name, end_time)))

    remove_training_checkpoint(output_dirs["model"], port, start_time, end_time)


def make_train_step(data_tensor: torch.Tensor, target_tensor: torch.Tensor, optimizer, model, criterion,
                    training: bool = True):
    output = model(data_tensor)
    loss = criterion(output, target_tensor)
    if training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def make_training_checkpoint(model, model_dir: str, port: Port, start_time: str, epoch: int, num_epochs: int,
                             learning_rate: float, loss_history: Tuple[List[float], List[float]],
                             optimizer: torch.optim.Adam) -> None:
    current_time = as_str(datetime.now())
    model_path = os.path.join(model_dir, encode_model_file(port.name, start_time, current_time, is_checkpoint=True))
    model.save(model_path)
    tc = TrainingCheckpoint(model_path=model_path, epoch=epoch, start_time=start_time, num_epochs=num_epochs,
                            learning_rate=learning_rate, loss_history=loss_history, optimizer=optimizer)
    tc.safe()
    # remove previous checkpoint
    remove_training_checkpoint(model_dir, port, start_time, current_time)


def remove_training_checkpoint(model_dir: str, port: Port, start_time: str, current_time: str) -> None:
    for file in os.listdir(model_dir):
        if file.startswith("checkpoint_"):
            _, cp_port_name, cp_start_time, cp_end_time = decode_model_file(file)
            if cp_port_name == port.name and cp_start_time == start_time and cp_end_time < current_time:
                os.rmdir(file)


def load_checkpoint(model_dir: str) -> TrainingCheckpoint:
    max_start_time, max_end_time, file_path = None, None, None
    for file in os.listdir(model_dir):
        if file.startswith("checkpoint_"):
            _, port_name, start_time, end_time = decode_model_file(file)
            if file_path is None or max_start_time < start_time and max_end_time < end_time:
                max_start_time = start_time
                max_end_time = end_time
                file_path = os.path.join(model_dir, file)
    if file_path is not None:
        tc: TrainingCheckpoint = joblib.load(file_path)
        return tc
    else:
        raise FileNotFoundError(f"Unable to load training checkpoint from directory {model_dir}")


def main(args) -> None:
    if args.command == "train":
        train_all(args.data_dir, args.output_dir)
    if args.command == "train_port":
        if not args.port_name:
            raise ValueError(f"No port name found! Use '--port_name=' to train specific model")
        print(f"Training single model for port {args.port_name}")
        train(args.port_name, args.data_dir, args.output_dir)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training endpoint")
    parser.add_argument("command", choices=["train", "train_port"])
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, os.pardir, "data"),
                        help="Path to data file directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, os.pardir, "output"),
                        help="Path to output directory")
    parser.add_argument("--port_name", type=str, help="Name of port to train model")
    main(parser.parse_args())
