import argparse
import numpy as np
import os
import torch

from datetime import datetime
from loader import MmsiDataFile, TrainingExampleLoader
from logger import Logger
from plotter import plot_series
from port import PortManager
from net.model import InceptionTimeModel
from util import debug_data, encode_model_file, encode_loss_file, encode_loss_plot, as_str, now

script_dir = os.path.abspath(os.path.dirname(__file__))


# entry point for training models for each port defined in port manager
def train_all(data_dir: str, output_dir: str, debug: bool = False) -> None:
    pm = PortManager()
    pm.load()
    if len(pm.ports.keys()) < 1:
        raise ValueError("No port data available")

    for _, port in pm.ports.items():
        train(port_name=port.name, data_dir=data_dir, output_dir=output_dir, num_epochs=5, learning_rate=.01, pm=pm,
              debug=debug)


def train(port_name: str, data_dir: str, output_dir: str, num_epochs: int = 5, learning_rate: float = .01,
          pm: PortManager = None, debug: bool = False) -> None:
    start_time = now()
    torch.autograd.set_detect_anomaly(True)
    if pm is None:
        pm = PortManager()
        pm.load()
        if len(pm.ports.keys()) < 1:
            raise ValueError("No port data available")
    port = pm.find_port(port_name)

    routes_path = os.path.join(data_dir, "routes", port.name)
    model_dir = os.path.join(output_dir, "model")
    eval_dir = os.path.join(output_dir, "eval")

    log_file_name = f"train-log_{port.name}_{as_str(start_time)}"
    train_logger = Logger(log_file_name, eval_dir)
    debug_logger = Logger(f"train-log_{port.name}_{as_str(start_time)}_debug", eval_dir)

    # set device: use gpu if available
    # more options: https://pytorch.org/docs/stable/notes/cuda.html
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {port.name}-model. Device: {device}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    loader = TrainingExampleLoader(routes_path)
    loader.load()

    if len(loader) == 0:
        raise ValueError(f"Unable to load data from directory {loader.data_dir}\n"
                         f"Make sure Data loader is fit!")

    data, target = loader[0]
    input_dim = data.shape[2]
    output_dim = 1

    model = InceptionTimeModel(num_inception_blocks=1, in_channels=input_dim, out_channels=32,
                               bottleneck_channels=8, use_residual=True, output_dim=output_dim)
    model.to(device)
    print(f"model:\n{model}")

    # train & validation loss
    loss_history = [[], []]
    criterion: torch.nn.MSELoss = torch.nn.MSELoss()
    # test what happens if using "weight_decay" e.g. with 1e-4
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training from directory {loader.data_dir} on {len(loader)} training examples")
    # print(f"First training example: \n{train_loader[0]}")

    # get learnable parameters
    params = list(model.parameters())
    print("number of params: ", len(params))
    print("first param's weight: ", params[0].size())
    train_logger.write(f"{port.name}-model num_epochs: {num_epochs} learning_rate: {learning_rate}, "
                       f"num_params: {len(params)}")

    # training loop
    for epoch in range(num_epochs):
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
            # print(f"train tensor:\n{data_tensor.shape}")
            # print(f"target tensor:\n{target_tensor.shape}")

            if debug:
                debug_data(data_tensor, target_tensor, train_idx, loader, debug_logger)

            batch_loss = make_train_step(data_tensor, target_tensor, optimizer, model, criterion)
            loss_train += batch_loss

            if loop_idx == 0 or loop_idx == len(train_indices) - 1:
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

        print(f"epoch: {epoch} avg val loss: {avg_validation_loss}")

        if epoch % 20 == 1:
            print(f"epoch: {epoch} average validation loss: {avg_validation_loss}")
        # loss_history.append([avg_train_loss, avg_validation_loss])
        print(f"loss history:\n{loss_history}")

        train_logger.write(f"Epoch {epoch + 1}/{num_epochs}:\n"
                           f"\tAvg train loss {avg_train_loss}\n"
                           f"\tAvg val loss   {avg_validation_loss}")

    curr_datetime = now()
    end_time = as_str(curr_datetime)
    model_path = os.path.join(model_dir, encode_model_file(port.name, end_time))
    model.save(model_path)
    loss_history_path = os.path.join(eval_dir, encode_loss_file(port.name, end_time))
    np.save(loss_history_path, loss_history)
    pm.add_training(port, curr_datetime, model_path, loss_history_path, os.path.join(eval_dir, f"{log_file_name}.txt"))

    plot_series(series=loss_history, x_label="Epoch", y_label="Loss",
                legend_labels=["Training", "Validation"], x_ticks=1., y_ticks=.2,
                path=os.path.join(eval_dir, encode_loss_plot(port.name, end_time)))


def make_train_step(data_tensor: torch.Tensor, target_tensor: torch.Tensor, optimizer, model, criterion,
                    training: bool = True):
    output = model(data_tensor)
    loss = criterion(output, target_tensor)
    if training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


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
