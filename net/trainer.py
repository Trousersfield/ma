import argparse
import numpy as np
import os
import time
import torch

from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Union

from dataset import RoutesDirectoryDataset, find_latest_dataset_config_path
from logger import Logger
from net.model import InceptionTimeModel
from plotter import plot_series
from port import Port, PortManager
from util import debug_data, encode_model_file, encode_loss_file, encode_loss_plot, as_str, num_total_parameters,\
    num_total_trainable_parameters, decode_model_file, find_latest_checkpoint_file_path, as_datetime,\
    verify_output_dir, encode_dataset_config_file, decode_dataset_config_file, as_duration, encode_meta_file,\
    encode_checkpoint_file, decode_checkpoint_file

script_dir = os.path.abspath(os.path.dirname(__file__))
torch.set_printoptions(precision=10)


class TrainingCheckpoint:
    def __init__(self, path: str, model_path: str, start_time: str, num_epochs: int, learning_rate: float,
                 weight_decay: float, num_train_examples: int, loss_history: Tuple[List[float], List[float]],
                 elapsed_time_history: List[float], optimizer: Union[torch.optim.Adam, torch.optim.AdamW],
                 is_optimum: bool) -> None:
        self.path = path
        self.model_path = model_path
        self.start_time = start_time
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_train_examples = num_train_examples
        self.loss_history = loss_history
        self.elapsed_time_history = elapsed_time_history
        self.is_optimum = is_optimum
        self.optimizer = optimizer

    def safe(self):
        torch.save({
            "path": self.path,
            "model_path": self.model_path,
            "start_time": self.start_time,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_train_examples": self.num_train_examples,
            "loss_history": self.loss_history,
            "elapsed_time_history": self.elapsed_time_history,
            "is_optimum": self.is_optimum,
            "optimizer_state_dict": self.optimizer.state_dict()
        }, self.path)

    @staticmethod
    def load(path: str, device, new_lr: float = None,
             new_num_epochs: int = None) -> Tuple['TrainingCheckpoint', InceptionTimeModel]:
        print(f"Loading training checkpoint from {path}")
        checkpoint = torch.load(path, device)
        if new_lr is not None:
            print(f"Updated learning rate: {new_lr}")
        if new_num_epochs is not None:
            epoch = checkpoint["epoch"]
            if new_num_epochs > int(epoch):
                print(f"Updated number of epochs from {epoch} to {new_num_epochs}")
            else:
                raise ValueError(f"Invalid argument: 'new_num_epochs' ({new_num_epochs}) must be larger than "
                                 f"currently trained epochs ({epoch})")
        num_epochs = new_num_epochs if new_num_epochs is not None else checkpoint["num_epochs"]
        lr = new_lr if new_lr is not None else checkpoint["learning_rate"]
        # backward compatibility
        wd = checkpoint["weight_decay"] if "weight_decay" in checkpoint else .0
        eth = checkpoint["elapsed_time_history"] if "elapsed_time_history" in checkpoint else \
            np.zeros(len(checkpoint["loss_history"]))
        nte = checkpoint["num_train_examples"] if "num_train_examples" in checkpoint else 0

        model = InceptionTimeModel.load(checkpoint["model_path"], device)  # .to(device)
        # TODO: optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        tc = TrainingCheckpoint(path=checkpoint["path"],
                                model_path=checkpoint["model_path"],
                                start_time=checkpoint["start_time"],
                                num_epochs=num_epochs,
                                learning_rate=lr,
                                weight_decay=wd,
                                num_train_examples=nte,
                                loss_history=checkpoint["loss_history"],
                                elapsed_time_history=eth,
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
        train(port_name=port.name, data_dir=data_dir, output_dir=output_dir, num_epochs=100, learning_rate=.0004, pm=pm,
              resume_checkpoint=None, debug=debug)


# THIS IS GOOD:
# lr = 0.0003 batch size 64 window width 128
# lr = 0.00025 batch size 64 window width 128

# lr = 0.0004 batch size 128 window 200 --> very slow
# lr = 0.0003 batch size 64 window 200 -->
# lr = .0001  --> some jumps, but pretty good results with batch-size 32
# lr = .00001 --> slow but steady updating: but after 50 epochs still 9x worse than above
# lr = .00005 --> seems to be still too slow, similar to above
def train(port_name: str, data_dir: str, output_dir: str, num_epochs: int = 100, learning_rate: float = .00025,
          weight_decay: float = .0001, pm: PortManager = None, resume_checkpoint: str = None,
          debug: bool = False) -> None:
    # TODO: Make sure dataset does not overwrite if (accidently) new training is started
    start_datetime = datetime.now()
    start_time = as_str(start_datetime)
    # set device: use gpu if available
    # more options: https://pytorch.org/docs/stable/notes/cuda.html
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.autograd.set_detect_anomaly(True)
    if pm is None:
        pm = PortManager()
        pm.load()
        if len(pm.ports.keys()) < 1:
            raise ValueError("No port data available")
    port = pm.find_port(port_name)

    output_dirs = verify_output_dir(output_dir, port.name)

    log_file_name = f"train-log-base_{port.name}_{start_time}"
    train_logger = Logger(log_file_name, output_dirs["log"], save=False)
    debug_logger = Logger(f"{log_file_name}_debug", output_dirs["log"]) if debug else train_logger

    if port is None:
        train_logger.write(f"Training skipped: Unable to find port based on port_name {port_name}")
        return

    training_type = "base"
    batch_size = 64
    window_width = 128
    dataset_dir = os.path.join(data_dir, "routes", port.name)

    # init dataset on directory
    dataset = RoutesDirectoryDataset(dataset_dir, start_time=start_time, training_type=training_type,
                                     batch_size=batch_size, start=0, window_width=window_width)
    if resume_checkpoint is not None:
        dataset_config_path = encode_dataset_config_file(resume_checkpoint, training_type) \
            if resume_checkpoint != "latest" else find_latest_dataset_config_path(dataset_dir,
                                                                                  training_type=training_type)
        if not os.path.exists(dataset_config_path):
            latest_config_path = find_latest_dataset_config_path(dataset_dir, training_type=training_type)
            use_latest = input(f"Unable to find dataset config for start time '{resume_checkpoint}'. "
                               f"Continue with latest config (Y) at '{latest_config_path}' or abort")
            if use_latest not in ["Y", "y", "YES", "yes"]:
                print(f"Training aborted")
                return
            dataset_config_path = latest_config_path
        if dataset_config_path is None or not os.path.exists(dataset_config_path):
            raise FileNotFoundError(f"Unable to recover training: No dataset config found at {dataset_config_path}")
        dataset = RoutesDirectoryDataset.load_from_config(dataset_config_path)
    else:
        dataset.save_config()
    end_train = int(.8 * len(dataset))
    if not (len(dataset) - end_train) % 2 == 0 and end_train < len(dataset):
        end_train += 1
    end_validate = int(len(dataset) - ((len(dataset) - end_train) / 2))

    # use initialized dataset's config for consistent split
    train_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, start=0, end=end_train)
    validate_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, start=end_train, end=end_validate)
    # eval_dataset = RoutesDirectoryDataset.load_from_config(dataset.config_path, kind="eval", start=end_validate)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, drop_last=False, pin_memory=True,
                                               num_workers=2)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=None, drop_last=False,
                                                  pin_memory=True, num_workers=2)
    train_logger.write(f"Dataset lengths:\n"
                       f"all: {len(dataset)}\ntrain: {len(train_dataset)}\nvalidate: {len(validate_dataset)}")

    data, target = train_dataset[0]
    input_dim = data.size(-1)
    output_dim = 1
    start_epoch = 0
    loss_history = ([], [])
    elapsed_time_history = []
    criterion: torch.nn.MSELoss = torch.nn.MSELoss()

    # resume from a checkpoint if training was aborted
    if resume_checkpoint is not None:
        tc, model = load_checkpoint(output_dirs["model"], device)
        start_epoch = len(tc.loss_history[1])
        start_time = tc.start_time
        num_epochs = tc.num_epochs
        learning_rate = tc.learning_rate
        weight_decay = tc.weight_decay
        loss_history = tc.loss_history
        elapsed_time_history = tc.elapsed_time_history
        # TODO: optimizer
        optimizer = tc.optimizer
    else:
        model = InceptionTimeModel(num_inception_blocks=3, in_channels=input_dim, out_channels=32,
                                   bottleneck_channels=16, use_residual=True, output_dim=output_dim).to(device)
        # test what happens if using "weight_decay" e.g. with 1e-4
        # TODO: optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f".:'`!`':. TRAINING FOR PORT {port_name} STARTED .:'`!`':.")
    print(f"- - Epochs {num_epochs} </> Training examples {len(train_loader)} </> Learning rate {learning_rate} - -")
    print(f"- - Weight decay {weight_decay} Window width {window_width} </> Batch size {batch_size} - -")
    print(f"- - Number of model's parameters {num_total_trainable_parameters(model)} device {device} - -")
    train_logger.write(f"{port.name}-model\n"
                       f"Number of epochs: {num_epochs}\n"
                       f"Learning rate: {learning_rate}\n"
                       f"Total number of parameters: {num_total_parameters(model)}\n"
                       f"Total number of trainable parameters: {num_total_trainable_parameters(model)}")

    min_val_idx = 0
    if resume_checkpoint is not None:
        min_val_idx = loss_history[1].index(min(loss_history[1]))
    # training loop
    print(f"loss history:\n{loss_history}")
    print(f"min index:\n{min_val_idx}")
    for epoch in range(start_epoch, num_epochs):
        # train model
        print(f"->->->->-> Epoch ({epoch + 1}/{num_epochs}) <-<-<-<-<-<-")
        avg_train_loss, elapsed_time = train_loop(criterion=criterion, model=model, device=device, optimizer=optimizer,
                                                  loader=train_loader, debug=debug, debug_logger=debug_logger)
        loss_history[0].append(avg_train_loss)
        elapsed_time_history.append(elapsed_time)

        # validate model
        avg_validation_loss = validate_loop(criterion=criterion, device=device, model=model, optimizer=optimizer,
                                            loader=validate_loader, debug=debug, debug_logger=debug_logger)
        loss_history[1].append(avg_validation_loss)

        # check if current model has lowest validation loss (= is current optimal model)
        if avg_validation_loss < loss_history[1][min_val_idx]:
            min_val_idx = epoch

        train_logger.write(f"Epoch {epoch + 1}/{num_epochs}:\n"
                           f"\tAvg train loss {avg_train_loss}\n"
                           f"\tAvg val   loss {avg_validation_loss}")

        make_training_checkpoint(model=model, model_dir=output_dirs["model"], port=port, start_time=start_time,
                                 num_epochs=num_epochs, learning_rate=learning_rate,
                                 weight_decay=weight_decay, num_train_examples=len(train_loader),
                                 loss_history=loss_history, elapsed_time_history=elapsed_time_history,
                                 optimizer=optimizer, is_optimum=min_val_idx == epoch)
        save_intermediate(data_dir=output_dirs["data"], elapsed_time_history=elapsed_time_history,
                          loss_history=loss_history, port=port, start_time=start_time, training_type="base")
        print(f">>>> Avg losses (MSE) - Train: {avg_train_loss} Validation: {avg_validation_loss} <<<<\n")

    # conclude training
    conclude_training(loss_history=loss_history, data_dir=output_dirs["data"], plot_dir=output_dirs["plot"],
                      port=port, start_time=start_time, elapsed_time_history=elapsed_time_history,
                      plot_title="Training loss", training_type="base")


def conclude_training(loss_history: Tuple[List[float], List[float]], data_dir: str,
                      plot_dir: str, port: Port, start_time: str, elapsed_time_history: List[float],
                      plot_title: str, training_type: str) -> None:
    save_intermediate(data_dir, elapsed_time_history, loss_history, port, start_time, training_type)
    plot_training(loss_history, plot_dir, plot_title, port, start_time, training_type)


def save_intermediate(data_dir: str, elapsed_time_history: List[float], loss_history: Tuple[List[float], List[float]],
                      port: Port, start_time: str, training_type: str) -> None:
    loss_history_path = os.path.join(data_dir, encode_loss_file(port.name, start_time, file_type=training_type))
    meta_data_path = os.path.join(data_dir, encode_meta_file(port.name, start_time, file_type=training_type))
    np.save(loss_history_path, loss_history)
    np.save(meta_data_path, elapsed_time_history)


def plot_training(loss_history: Tuple[List[float], List[float]], plot_dir: str, plot_title: str, port: Port,
                  start_time: str, training_type: str):
    plot_linear_path = os.path.join(plot_dir, encode_loss_plot(port.name, start_time, file_type=training_type))
    plot_log_path = os.path.join(plot_dir, encode_loss_plot(port.name, start_time, file_type=training_type,
                                                            scale="log"))
    plot_series(series=loss_history, title=plot_title, x_label="Epoch", y_label="Loss",
                legend_labels=["Training", "Validation"],
                path=plot_linear_path)
    plot_series(series=loss_history, title=plot_title, x_label="Epoch", y_label="Loss",
                legend_labels=["Training", "Validation"], y_scale="log",
                path=plot_log_path)


def train_loop(criterion, device, model, optimizer, loader, debug=False, debug_logger=None) -> Tuple[float, float]:
    loss = 0
    model.train()
    t_start = time.time()
    for batch_idx, (inputs, target) in enumerate(tqdm(loader, desc="Training-loop progress")):
        inputs = inputs.to(device)
        target = target.to(device)

        if debug and debug_logger is not None:
            debug_data(inputs, target, batch_idx, loader, debug_logger)

        batch_loss = make_train_step(inputs, target, optimizer, model, criterion)
        loss += batch_loss
    return loss / len(loader), time.time() - t_start


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


def make_training_checkpoint(model, model_dir: str, port: Port, start_time: str, num_epochs: int,
                             learning_rate: float, weight_decay: float, num_train_examples: int,
                             loss_history: Tuple[List[float], List[float]],
                             elapsed_time_history: List[float],
                             optimizer: Union[torch.optim.Adam, torch.optim.AdamW],
                             is_optimum: bool, base_port_name: str = None) -> None:
    training_type = "base" if base_port_name is None else "transfer"
    current_time = as_str(datetime.now())
    model_file_name = encode_model_file(port.name, start_time, current_time, is_checkpoint=not is_optimum,
                                        file_type=training_type, base_port_name=base_port_name)
    model_path = os.path.join(model_dir, model_file_name)
    model.save(model_path)

    cp_file_name = encode_checkpoint_file(port.name, start_time, current_time, is_checkpoint=not is_optimum,
                                          file_type=training_type, base_port_name=base_port_name)
    tc_path = os.path.join(os.path.split(model_path)[0], cp_file_name)
    tc = TrainingCheckpoint(path=tc_path, model_path=model_path, start_time=start_time,
                            num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                            num_train_examples=num_train_examples, loss_history=loss_history,
                            elapsed_time_history=elapsed_time_history, optimizer=optimizer,
                            is_optimum=is_optimum)
    tc.safe()
    # remove previous checkpoint's model
    remove_previous_checkpoint(model_dir, port, start_time, current_time, is_optimum, base_port_name)


def remove_previous_checkpoint(model_dir: str, port: Port, start_time: str, current_time: str,
                               new_optimum: bool, base_port: str = None) -> None:
    """
    Remove a previous checkpoint's model- and checkpoint-file. Identification via file_type and encoded start_time:
    Files with the same start_time as given in the parameters are considered to be the same training
    :param model_dir: Directory to models for given port (including port name)
    :param port: Port
    :param start_time: Main identification for files that belong to the same training within the given directory
    :param current_time: Current time for identifying older checkpoints
    :param new_optimum: Specify if new optimum (= model) was found, so previous model gets removed
    :param base_port: Base port name if checkpoint is made during transfer
    :return: None
    """
    current_time = as_datetime(current_time)
    cp_type = "checkpoint-base" if base_port is None else "checkpoint-transfer"
    model_type = "model-base" if base_port is None else "model-transfer"
    for file in os.listdir(model_dir):
        _, ext = os.path.splitext(file)
        if file.startswith(cp_type) or (new_optimum and file.startswith(model_type)):
            if ext in [".pt", ".tar"]:  # model or checkpoint file
                _, cp_port_name, cp_start_time, cp_end_time, _ = \
                    decode_model_file(file) if ext == ".pt" else decode_checkpoint_file(file)
                cp_end_time = as_datetime(cp_end_time)
                if cp_port_name == port.name and cp_start_time == start_time and cp_end_time < current_time:
                    os.remove(os.path.join(model_dir, file))


def load_checkpoint(model_dir: str, device) -> Tuple[TrainingCheckpoint, InceptionTimeModel]:
    file_path, checkpoint_type = find_latest_checkpoint_file_path(model_dir)
    lr, num_epochs = None, None
    if checkpoint_type == "model":
        inp = input(f"Latest checkpoint is of type 'model' at {file_path}\nEnter 'Y' to continue")
        if inp not in ["y", "Y"]:
            raise ValueError(f"Recovering aborted! Undesired latest checkpoint of type 'model' at {file_path}")
        lr = input(f"Type in new learning rate if desired. Press ENTER to skip")
        if lr is not None and lr not in ["", "n", "N"]:
            try:
                lr = float(lr)
            except ValueError:
                raise ValueError(f"Unable to cast entered learning rate '{lr}' to float")
        else:
            lr = None
        num_epochs = input(f"Type in new number of epochs. Press ENTER to skip")
        if num_epochs is not None and num_epochs not in ["", "n", "N"]:
            try:
                num_epochs = int(num_epochs)
            except ValueError:
                raise ValueError(f"Unable to cast entered number of epochs '{num_epochs}' to int")
        else:
            num_epochs = None
    tc, model = TrainingCheckpoint.load(file_path, device, new_lr=lr, new_num_epochs=num_epochs)
    return tc, model


def main(args) -> None:
    if args.command == "train":
        train_all(args.data_dir, args.output_dir)
    elif args.command == "train_port":
        if not args.port_name:
            raise ValueError(f"No port name found! Use '--port_name=' to train specific model")
        print(f"Attempting to train single model for port name '{args.port_name}'")
        train(args.port_name, args.data_dir, args.output_dir, resume_checkpoint=args.resume_checkpoint)
    elif args.command == "edit_checkpoint":
        if not os.path.exists(args.checkpoint_path):
            raise ValueError(f"Checkpoint file not found: checkpoint_path at '{args.checkpint_path}' does not exist")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tc, model = TrainingCheckpoint.load(path=args.checkpoint_path, device=device)
        new_tc = TrainingCheckpoint(path=tc.path,
                                    model_path=tc.model_path,
                                    start_time=tc.start_time,
                                    num_train_examples=90193,
                                    num_epochs=tc.num_epochs,
                                    learning_rate=tc.learning_rate,
                                    weight_decay=tc.weight_decay,
                                    loss_history=tc.loss_history,
                                    elapsed_time_history=[4009., 4011., 4005., 4008., 3991., 3981., 3975., 3991., 3979., 3975., 3981, 3956., 4056., 4068., 4073., 4073., 4074.],
                                    optimizer=tc.optimizer,
                                    is_optimum=tc.is_optimum)
        # new_tc.safe()
        # SKAGEN:
        # 101482 data entries
        # [4144., 4135., 4116., 4125., 4130., 4115., 4116., 4124., 4126., 4133., 4132., 4132., 4131., 4125., 4139., 4163., 4151., 4148., 4125.]
        # 24,5 it/sec

        # TRELLEBORG:
        # 24788 data entries
        # [273., 274., 275., 269., 269., 276., 277., 275., 275., 273.,
        # 274., 275., 274., 276., 266., 268., 270., 269., 269., 270.,
        # 268., 272., 283., 282., 275., 273., 274., 274., 273., 275.,
        # 275., 269., 267., 266., 268., 266., 267., 268., 266., 273.,
        # 274., 276., 274., 274., 275., 273., 273., 274., 275., 275.,
        # 267., 268., 268., 266., 271., 267., 276., 274., 274., 275.,
        # 275., 274., 271., 268., 274., 274., 278., 278., 277., 275.,
        # 279., 289., 293., 291., 279., 273., 271., 275., 277., 277.,
        # 278., 282., 281., 281., 280., 281., 284., 281., 279., 280.,
        # 278., 273., 268., 269., 271., 263., 274., 274., 275., 281.]
        # 90 it/sec

        # ESBJERG
        # 2296 files
        # 140349 data entries
        # 112316 train
        # 126355-112316=14039 eval
        # Obergrenze RAM fast erreicht: 11,17/12,69 GB
        # 24,5 it/sec

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training endpoint")
    parser.add_argument("command", choices=["train", "train_port", "edit_checkpoint"])
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, os.pardir, "data"),
                        help="Path to data file directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, os.pardir, "output"),
                        help="Path to output directory")
    parser.add_argument("--port_name", type=str, help="Name of port to train model")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Specify if training shall recover from previous checkpoint")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    main(parser.parse_args())
