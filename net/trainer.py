import argparse
import numpy as np
import os
import torch

from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Union

from dataset import RoutesDirectoryDataset
from logger import Logger
from net.model import InceptionTimeModel
from plotter import plot_series
from port import Port, PortManager
from util import debug_data, encode_model_file, encode_loss_file, encode_loss_plot, as_str, num_total_parameters,\
    num_total_trainable_parameters, decode_model_file, find_latest_checkpoint_file_path, encode_checkpoint_file,\
    decode_checkpoint_file, as_datetime, verify_output_dir

script_dir = os.path.abspath(os.path.dirname(__file__))
torch.set_printoptions(precision=10)


class TrainingCheckpoint:
    def __init__(self, path: str, model_path: str, start_time: str, epoch: int, num_epochs: int,
                 learning_rate: float, loss_history: Tuple[List[float], List[float]], optimizer: torch.optim.Adam,
                 is_optimum: bool) -> None:
        self.path = path
        self.model_path = model_path
        self.start_time = start_time
        self.epoch = epoch
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_history = loss_history
        self.is_optimum = is_optimum
        self.optimizer = optimizer

    def safe(self):
        torch.save({
            "path": self.path,
            "model_path": self.model_path,
            "start_time": self.start_time,
            "epoch": self.epoch,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "loss_history": self.loss_history,
            "is_optimum": self.is_optimum,
            "optimizer_state_dict": self.optimizer.state_dict()
        }, self.path)

    @staticmethod
    def load(path: str, device) -> Tuple['TrainingCheckpoint', InceptionTimeModel]:
        print(f"Loading training checkpoint from {path}")
        checkpoint = torch.load(path, device)
        model = InceptionTimeModel.load(checkpoint["model_path"], device)  # .to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=checkpoint["learning_rate"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        tc = TrainingCheckpoint(path=checkpoint["path"],
                                model_path=checkpoint["model_path"],
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


# lr = .0001  --> some jumps
# lr = .00001 --> slow but steady updating: but after 50 epochs still 9x worse than above
# lr = .00005 --> ?
def train(port_name: str, data_dir: str, output_dir: str, num_epochs: int = 100, learning_rate: float = .00005,
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

    output_dirs = verify_output_dir(output_dir, port.name)

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

    batch_size = 32
    window_width = 100
    dataset_dir = os.path.join(data_dir, "routes", port.name)
    dataset_config_path = os.path.join(dataset_dir, "default_dataset_config.pkl")

    # init dataset on directory
    dataset = RoutesDirectoryDataset(dataset_dir, batch_size=batch_size, start=0, window_width=window_width)
    if not resume_checkpoint:
        dataset.save_config()
    else:
        if not os.path.exists(dataset_config_path):
            raise FileNotFoundError(f"Unable to recoder training: No dataset config found at {dataset_config_path}")
        dataset = RoutesDirectoryDataset.load_from_config(dataset_config_path)
    end_train = int(.8 * len(dataset))
    if not (len(dataset) - end_train) % 2 == 0 and end_train < len(dataset):
        end_train += 1
    end_validate = int(len(dataset) - ((len(dataset) - end_train) / 2))

    # use initialized dataset's config for consistent split
    train_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, kind="train", start=0,
                                                            end=end_train)
    validate_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, kind="validate",
                                                               start=end_train, end=end_validate)
    # eval_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, kind="eval", start=end_validate)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, drop_last=False, pin_memory=True,
                                               num_workers=1)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=None, drop_last=False,
                                                  pin_memory=True, num_workers=1)

    # if len(loader) == 0:
    #     raise ValueError(f"Unable to load data from directory {loader.data_dir}\n"
    #                      f"Make sure Data loader is fit!")

    # data, target = loader[0]
    # input_dim = data.shape[2]
    data, target = train_dataset[0]
    input_dim = data.size(-1)
    output_dim = 1
    start_epoch = 0
    loss_history = ([], [])
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
        model = InceptionTimeModel(num_inception_blocks=2, in_channels=input_dim, out_channels=32,
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
        # train model
        avg_train_loss = train_loop(criterion=criterion, model=model, device=device, optimizer=optimizer,
                                    loader=train_loader, debug=debug, debug_logger=debug_logger)
        loss_history[0].append(avg_train_loss)
        print(f"epoch: {epoch} avg train loss: {avg_train_loss}")

        # validate model
        avg_validation_loss = validate_loop(criterion=criterion, device=device, model=model, optimizer=optimizer,
                                            loader=validate_loader, debug=debug, debug_logger=debug_logger)
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
        print(f"epoch: {epoch + 1} avg val loss: {avg_validation_loss}")

    # conclude training
    end_datetime = datetime.now()
    conclude_training(loss_history=loss_history, end=end_datetime, model_dir=output_dirs["model"],
                      data_dir=output_dirs["data"], plot_dir=output_dirs["plot"], port=port, start_time=start_time)


def conclude_training(loss_history: Tuple[List[float], List[float]], end: datetime, model_dir: str, data_dir: str,
                      plot_dir: str, port: Port, start_time: str,
                      plot_title: str = "Training loss") -> Tuple[str, str, str]:
    end_time = as_str(end)
    remove_previous_checkpoint(model_dir, port, start_time, end_time)
    loss_history_path = os.path.join(data_dir, encode_loss_file(port.name, end_time))
    np.save(loss_history_path, loss_history)
    model_path = find_model_path(model_dir, start_time=start_time)
    plot_path = os.path.join(plot_dir, encode_loss_plot(port.name, end_time))
    plot_series(series=loss_history, title=plot_title, x_label="Epoch", y_label="Loss",
                legend_labels=["Training", "Validation"], path=plot_path)
    plot_series(series=loss_history, title=plot_title, x_label="Epoch", y_label="Loss",
                legend_labels=["Training", "Validation"], y_scale="log",
                path=os.path.join(plot_dir, encode_loss_plot(f"{port.name}_LOG-SCALE", end_time)))
    return loss_history_path, model_path, plot_path


def train_loop(criterion, device, model, optimizer, loader, debug=False, debug_logger=None) -> float:
    loss = 0
    model.train()
    for batch_idx, (inputs, target) in enumerate(tqdm(loader)):
        inputs = inputs.to(device)
        target = target.to(device)
        # print(f"train tensor:\n{train_data.shape}")
        # print(f"target tensor:\n{target.shape}")

        if debug and debug_logger is not None:
            debug_data(inputs, target, batch_idx, loader, debug_logger)

        batch_loss = make_train_step(inputs, target, optimizer, model, criterion)
        loss += batch_loss

        # if batch_idx % 1000 == 0:
        #     print(f"Loop idx: {batch_idx} batch loss: {batch_loss}")
        #     print(f"Loop idx: {batch_idx} loss_train: {loss}")
    return loss / len(loader)


def validate_loop(criterion, device, model, optimizer, loader, debug=False, debug_logger=None) -> float:
    loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
            inputs = inputs.to(device)
            target = target.to(device)

            if debug and debug_logger is not None:
                debug_data(inputs, target, batch_idx, loader, debug_logger, "Validation")

            batch_loss = make_train_step(inputs, target, optimizer, model, criterion, training=False)

            loss += batch_loss
    return loss / len(loader)


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
                             optimizer: torch.optim.Adam, is_optimum: bool, is_transfer: bool = False) -> None:
    current_time = as_str(datetime.now())
    model_path = os.path.join(model_dir, encode_model_file(port.name, start_time, current_time,
                                                           is_checkpoint=False if is_optimum else True,
                                                           is_transfer=is_transfer))
    model.save(model_path)
    tc_path = os.path.join(os.path.split(model_path)[0],
                           encode_checkpoint_file(port.name, start_time, current_time,
                                                  is_checkpoint=False if is_optimum else True, is_transfer=is_transfer))
    tc = TrainingCheckpoint(path=tc_path, model_path=model_path, epoch=epoch, start_time=start_time,
                            num_epochs=num_epochs, learning_rate=learning_rate, loss_history=loss_history,
                            optimizer=optimizer, is_optimum=is_optimum)
    tc.safe()
    # remove previous checkpoint's model
    remove_previous_checkpoint(model_dir, port, start_time, current_time, is_optimum)


def remove_previous_checkpoint(model_dir: str, port: Port, start_time: str, current_time: str,
                               new_optimum: bool = False) -> None:
    """
    Remove a previous checkpoint's model- and checkpoint-file. Identification via encoded start_time: Files with the
    same start_time as given in the parameters are considered to be the same training
    :param model_dir: Directory to models for given port (including port name)
    :param port: Port
    :param start_time: Main identification for files that belong to the same training within the given directory
    :param current_time: Current time for identifying older checkpoints
    :param new_optimum: Remove files from a previous optimum
    :return: None
    """
    current_time = as_datetime(current_time)
    for file in os.listdir(model_dir):
        _, ext = os.path.splitext(file)
        if file.startswith("checkpoint_") or (new_optimum and file.startswith("model_")):
            if ext in [".pt", ".tar"]:  # model or checkpoint file
                _, cp_port_name, cp_start_time, cp_end_time = \
                    decode_model_file(file) if ext == ".pt" else decode_checkpoint_file(file)
                cp_end_time = as_datetime(cp_end_time)
                if cp_port_name == port.name and cp_start_time == start_time and cp_end_time < current_time:
                    os.remove(os.path.join(model_dir, file))


def load_checkpoint(model_dir: str, device) -> Tuple[TrainingCheckpoint, InceptionTimeModel]:
    file_path, checkpoint_type = find_latest_checkpoint_file_path(model_dir)
    if checkpoint_type == "model":
        inp = input(f"Latest checkpoint is of type 'model' at {file_path}\nEnter 'Y' to continue")
        if inp not in ["y", "Y"]:
            raise ValueError(f"Training stopped by user! Undesired latest checkpoint of type 'model' at {file_path}")
    tc, model = TrainingCheckpoint.load(file_path, device)
    return tc, model


def find_model_path(model_dir: str, start_time: str, transfer: bool = False) -> str:
    result = ""
    file_kind = "transfer_" if transfer else "model_"
    for file in os.listdir(model_dir):
        if file.startswith(file_kind):
            _, _, file_start_time, file_end_time = decode_model_file(file)
            if file_start_time == start_time:
                if result != "":
                    print(f"Warning: Multiple models of kind '{file_kind}' in directory {model_dir} with same"
                          f"training-time '{start_time}' detected. Returning latest")
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
