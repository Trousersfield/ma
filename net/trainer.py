import argparse
import numpy as np
import os
import torch

from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union

from dataset import MmsiDataFile, RoutesDirectoryDataset
from logger import Logger
from net.model import InceptionTimeModel
from plotter import plot_series
from port import Port, PortManager
from util import debug_data, encode_model_file, encode_loss_file, encode_loss_plot, as_str, num_total_parameters,\
    num_total_trainable_parameters, decode_model_file

script_dir = os.path.abspath(os.path.dirname(__file__))
torch.set_printoptions(precision=10)


class TrainingCheckpoint:
    def __init__(self, model_path: str, start_time: str, epoch: int, num_epochs: int, learning_rate: float,
                 loss_history: Tuple[List[float], List[float]], optimizer: torch.optim.Adam,
                 is_optimum: bool) -> None:
        self.model_path = model_path
        self.save_path = os.path.join(os.path.split(model_path)[0], "checkpoint.tar")
        self.start_time = start_time
        self.epoch = epoch
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_history = loss_history
        self.is_optimum = is_optimum
        self.optimizer = optimizer

    def safe(self):
        torch.save({
            "model_path": self.model_path,
            "save_path": self.save_path,
            "start_time": self.start_time,
            "epoch": self.epoch,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "loss_history": self.loss_history,
            "is_optimum": self.is_optimum,
            "optimizer_state_dict": self.optimizer.state_dict()
        }, self.save_path)

    @staticmethod
    def load(checkpoint_path: str, device) -> Tuple['TrainingCheckpoint', InceptionTimeModel]:
        print(f"Loading training checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model = InceptionTimeModel.load(checkpoint["model_path"], device)  # .to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=checkpoint["learning_rate"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        tc = TrainingCheckpoint(model_path=checkpoint["model_path"],
                                start_time=checkpoint["start_time"],
                                epoch=checkpoint["epoch"],
                                num_epochs=checkpoint["num_epochs"],
                                learning_rate=checkpoint["learning_rate"],
                                loss_history=checkpoint["loss_history"],
                                is_optimum=checkpoint["is_optimum"],
                                optimizer=optimizer)
        return tc, model


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
    # torch.autograd.set_detect_anomaly(True)
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

    log_file_name = f"train-log_{port.name}_{start_time}"
    train_logger = Logger(log_file_name, output_dirs["log"])
    debug_logger = Logger(f"train-log_{port.name}_{start_time}_debug", output_dirs["log"]) if debug else train_logger

    if port is None:
        train_logger.write(f"Training skipped: Unable to find port based on port_name {port_name}")
        return

    # set device: use gpu if available
    # more options: https://pytorch.org/docs/stable/notes/cuda.html
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {port.name}-model. Device: {device}")

    # loader = TrainingExampleLoader(routes_path)
    # loader.load()

    batch_size = 32
    window_width = 100
    dataset_dir = os.path.join(data_dir, "routes", port.name)
    # dataset = RoutesDirectoryDataset(data_dir=dataset_dir, train_ratio=.8, window_width=100)
    dataset = RoutesDirectoryDataset(data_dir=dataset_dir, window_width=window_width)
    # eval_dir = os.path.join(output_dir, "eval")

    if len(dataset) == 0:
        print(f"No data for port {port.name} available! Make sure Directory Dataset is fit")
        return

    # rand_indices = np.arange(len(dataset))
    # np.random.shuffle(rand_indices)
    end_train = int(.8 * len(dataset))
    if not (len(dataset) - end_train) % 2 == 0 and end_train < len(dataset):
        end_train += 1
    end_validate = int(len(dataset) - ((len(dataset) - end_train) / 2))

    train_dataset = RoutesDirectoryDataset(data_dir=dataset_dir, end=end_train, window_width=window_width)
    validate_dataset = RoutesDirectoryDataset(data_dir=dataset_dir, start=end_train, end=end_validate,
                                              window_width=window_width)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False, pin_memory=True,
                                               num_workers=1)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, drop_last=False,
                                                  pin_memory=True, num_workers=1)

    # if len(loader) == 0:
    #     raise ValueError(f"Unable to load data from directory {loader.data_dir}\n"
    #                      f"Make sure Data loader is fit!")

    # data, target = loader[0]
    # input_dim = data.shape[2]
    data, target = dataset[0]
    input_dim = data.size(-1)
    output_dim = 1
    start_epoch = 0
    loss_history = [[], []]
    criterion: torch.nn.MSELoss = torch.nn.MSELoss()

    # resume from a checkpoint if training was aborted
    if resume_checkpoint:
        tc, model = load_checkpoint(output_dirs["model"], device)
        start_epoch = tc.epoch
        start_time = tc.start_time
        num_epochs = tc.num_epochs
        learning_rate = tc.learning_rate
        loss_history = tc.loss_history
        optimizer: torch.optim.Adam = tc.optimizer
    else:
        model = InceptionTimeModel(num_inception_blocks=3, in_channels=input_dim, out_channels=32,
                                   bottleneck_channels=16, use_residual=True, output_dim=output_dim).to(device)
        # test what happens if using "weight_decay" e.g. with 1e-4
        optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # model.to(device)
    # print(f"model:\n{model}")
    print(f"Starting training from directory {dataset_dir} on {len(train_loader)} training examples")
    # print(f"First training example: \n{train_loader[0]}")
    train_logger.write(f"{port.name}-model\n"
                       f"Number of epochs: {num_epochs}\n"
                       f"Learning rate: {learning_rate}\n"
                       f"Total number of parameters: {num_total_parameters(model)}\n"
                       f"Total number of trainable parameters: {num_total_trainable_parameters(model)}")

    min_val_idx = 0
    # training loop
    for epoch in range(start_epoch, num_epochs):
        loss_train = 0
        loss_validation = 0

        # train model
        model.train()
        # train_indices = loader.shuffled_data_indices(kind="train")

        # for loop_idx, train_idx in enumerate(train_indices):
        for batch_idx, (train_data, target) in enumerate(train_loader):
            # TODO: Tensoren
            # train_data, target = loader[train_idx]
            # print(f"train data:\n{train_data}")
            # print(f"target data:\n{target}")
            # TODO: from_numpy
            # data_tensor = torch.Tensor(train_data).to(device)
            # target_tensor = torch.Tensor(target).unsqueeze(-1).to(device)
            train_data = train_data.to(device)
            target = target.unsqueeze(-1).to(device)
            # print(f"train tensor:\n{train_data.shape}")
            # print(f"target tensor:\n{target.shape}")

            if debug:
                # debug_data(data_tensor, target_tensor, train_idx, loader, debug_logger)
                debug_data(train_data, target, batch_idx, train_loader, debug_logger)

            # batch_loss = make_train_step(data_tensor, target_tensor, optimizer, model, criterion)
            batch_loss = make_train_step(train_data, target, optimizer, model, criterion)
            loss_train += batch_loss

            if batch_idx % 1000 == 0:
                # print(f"train_tensor: {data_tensor}")
                # print(f"target_tensor: {target_tensor}")
                print(f"Loop idx: {batch_idx} batch loss: {batch_loss}")
                print(f"Loop idx: {batch_idx} loss_train: {loss_train}")

        avg_train_loss = loss_train / len(train_loader)
        loss_history[0].append(avg_train_loss)

        print(f"epoch: {epoch} avg train loss: {avg_train_loss}")

        # validate model
        model.eval()
        with torch.no_grad():
            # validate_indices = loader.shuffled_data_indices(kind="validate")

            # for validate_idx in validate_indices:
            for batch_idx, (validate_data, target) in enumerate(train_loader):
                # validate_data, target = loader[validate_idx]
                # validate_tensor = torch.Tensor(validate_data).to(device)
                # target_tensor = torch.Tensor(target).unsqueeze(-1).to(device)
                validate_data = validate_data.to(device)
                target = target.unsqueeze(-1).to(device)

                if debug:
                    # debug_data(validate_tensor, target_tensor, validate_idx, loader,
                    #            debug_logger, "Validation")
                    debug_data(validate_data, target, batch_idx, validate_loader, debug_logger, "Validation")

                # batch_loss = make_train_step(validate_tensor, target_tensor, optimizer, model, criterion,
                #                              training=False)
                batch_loss = make_train_step(validate_data, target, optimizer, model, criterion, training=False)

                loss_validation += batch_loss
        avg_validation_loss = loss_validation / len(validate_loader)
        loss_history[1].append(avg_validation_loss)

        # check if current model has lowest validation loss (= is current optimal model)
        if loss_history[1][epoch] < loss_history[1][min_val_idx]:
            min_val_idx = epoch

        train_logger.write(f"Epoch {epoch + 1}/{num_epochs}:\n"
                           f"\tAvg train loss {avg_train_loss}\n"
                           f"\tAvg val loss   {avg_validation_loss}")

        make_training_checkpoint(model=model, model_dir=output_dirs["model"], port=port, start_time=start_time,
                                 epoch=epoch, num_epochs=num_epochs, learning_rate=learning_rate,
                                 loss_history=loss_history, optimizer=optimizer, is_optimum=min_val_idx == epoch)
        print(f"epoch: {epoch} avg val loss: {avg_validation_loss}")

    # save data concerning training
    end_datetime = datetime.now()
    end_time = as_str(end_datetime)
    remove_previous_checkpoint_model(output_dirs["model"], port, start_time, end_time)
    # model_path = os.path.join(output_dirs["model"], encode_model_file(port.name, start_time, end_time))
    # model.save(model_path)
    loss_history_path = os.path.join(output_dirs["data"], encode_loss_file(port.name, end_time))
    np.save(loss_history_path, loss_history)
    model_path = find_model_path(output_dirs["model"], start_time=start_time)
    pm.add_training(port, start_datetime, end_datetime, model_path, loss_history_path,
                    os.path.join(output_dirs["log"], f"{log_file_name}.txt"))

    plot_series(series=loss_history, x_label="Epoch", y_label="Loss",
                legend_labels=["Training", "Validation"], x_ticks=1.,
                path=os.path.join(output_dirs["plot"], encode_loss_plot(port.name, end_time)))


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
                             optimizer: torch.optim.Adam, is_optimum: bool) -> None:
    current_time = as_str(datetime.now())
    model_path = os.path.join(model_dir, encode_model_file(port.name, start_time, current_time,
                                                           is_checkpoint=False if is_optimum else True))
    model.save(model_path)
    tc = TrainingCheckpoint(model_path=model_path, epoch=epoch, start_time=start_time, num_epochs=num_epochs,
                            learning_rate=learning_rate, loss_history=loss_history, optimizer=optimizer,
                            is_optimum=is_optimum)
    tc.safe()
    # remove previous checkpoint's model
    remove_previous_checkpoint_model(model_dir, port, start_time, current_time, is_optimum)


def remove_previous_checkpoint_model(model_dir: str, port: Port, start_time: str, current_time: str,
                                     new_optimum: bool = False) -> None:
    for file in os.listdir(model_dir):
        if file.startswith("checkpoint_") or (new_optimum and file.startswith("model_")):
            _, cp_port_name, cp_start_time, cp_end_time = decode_model_file(file)
            if cp_port_name == port.name and cp_start_time == start_time and cp_end_time < current_time:
                os.rmdir(os.path.join(model_dir, file))


def load_checkpoint(model_dir: str, device) -> Tuple[TrainingCheckpoint, InceptionTimeModel]:
    max_start_time, max_end_time, file_path = None, None, None
    for file in os.listdir(model_dir):
        if file.startswith("checkpoint_") or file.startswith("model_"):
            _, _, start_time, end_time = decode_model_file(file)
            if file_path is None or (max_start_time < start_time and max_end_time < end_time):
                max_start_time = start_time
                max_end_time = end_time
                file_path = os.path.join(model_dir, file)
    if file_path is not None:
        tc, model = TrainingCheckpoint.load(file_path, device)
        return tc, model
    else:
        raise FileNotFoundError(f"Unable to load training checkpoint from directory {model_dir}")


def find_model_path(model_dir: str, start_time: str) -> str:
    result = ""
    for file in os.listdir(model_dir):
        if file.startswith("model_"):
            _, _, file_start_time, file_end_time = decode_model_file(file)
            if file_start_time == start_time:
                if result != "":
                    print(f"Warning: Multiple models in directory {model_dir} with same training-time '{start_time}' "
                          f"detected. Returning latest")
                result = os.path.join(model_dir, file)
    if result is not None:
        return result
    else:
        raise FileNotFoundError(f"Unable to retrieve model path from directory {model_dir} at '{start_time}'")


def main(args) -> None:
    if args.command == "train":
        train_all(args.data_dir, args.output_dir)
    if args.command == "train_port":
        if not args.port_name:
            raise ValueError(f"No port name found! Use '--port_name=' to train specific model")
        print(f"Training single model for port {args.port_name}")
        train(args.port_name, args.data_dir, args.output_dir, resume_checkpoint=args.resume_checkpoint)
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
    parser.add_argument("--resume_checkpoint", type=bool, default=False,
                        help="Specifies training shall recover from previous checkpoint")
    main(parser.parse_args())
