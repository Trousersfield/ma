import argparse
import os
import torch

from datetime import datetime
from loader import MmsiDataFile, TrainingExampleLoader
from plotter import plot_series
from net.model import InceptionTimeModel

script_dir = os.path.abspath(os.path.dirname(__file__))


def train(data_dir: str, output_dir: str, num_epochs: int = 3, learning_rate: float = .01) -> None:
    torch.autograd.set_detect_anomaly(True)

    train_path = os.path.join(data_dir, "train", "ROSTOCK")
    validation_path = os.path.join(data_dir, "validate", "ROSTOCK")
    model_dir = os.path.join(output_dir, "model")
    eval_dir = os.path.join(output_dir, "eval")
    # set device: use gpu is available
    # more options: https://pytorch.org/docs/stable/notes/cuda.html
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    train_loader = TrainingExampleLoader(train_path)
    validation_loader = TrainingExampleLoader(validation_path)
    train_loader.load()
    validation_loader.load()

    if len(train_loader) == 0:
        raise ValueError("Unable to load data from directory {}\nData loader must be initialized first!"
                         .format(train_loader.data_dir))
    if len(validation_loader) == 0:
        raise ValueError("Unable to load data from directory {}\nData loader must be initialized first!"
                         .format(validation_loader.data_dir))

    data, target = train_loader[0]
    input_dim = data.shape[2]
    output_dim = 1

    model = InceptionTimeModel(num_inception_blocks=1, in_channels=input_dim, out_channels=32,
                               bottleneck_channels=8, use_residual=True, output_dim=output_dim)
    model.to(device)
    print(f"model: \n{model}")

    train_loss_history = []
    validate_loss_history = []
    criterion: torch.nn.MSELoss = torch.nn.MSELoss()
    # test what happens if using "weight_decay" e.g. with 1e-4
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training from directory {train_loader.data_dir} on {len(train_loader)} training examples")
    # print(f"First training example: \n{train_loader[0]}")

    # get learnable parameters
    params = list(model.parameters())
    print("number of params: ", len(params))
    print("first param's weight: ", params[0].size())

    # training loop
    for epoch in range(num_epochs):
        loss_train = 0
        loss_validation = 0

        # train model
        model.train()
        for train_idx in range(len(train_loader)):
            train_data, target = train_loader[train_idx]
            data_tensor = torch.Tensor(train_data).to(device)
            target_tensor = torch.Tensor(target).unsqueeze(-1).to(device)

            batch_loss = make_train_step(data_tensor, target_tensor, optimizer, model, criterion)
            loss_train += batch_loss

            if train_idx == 0 or train_idx % 2000 == 1 or train_idx == len(train_loader) - 1:
                # print(f"train_tensor: {data_tensor}")
                # print(f"target_tensor: {target_tensor}")
                print(f"idx: {train_idx} batch loss: {batch_loss}")
                print(f"idx: {train_idx} loss_train: {loss_train}")

            # if torch.isnan(data_tensor):
            #    print(f"found tensor with nan values: {data_tensor}")

        avg_train_loss = loss_train / len(train_loader)
        train_loss_history.append(avg_train_loss)

        print(f"epoch: {epoch} avg train loss: {avg_train_loss}")

        # validate model
        model.eval()
        for validation_idx in range(len(validation_loader)):
            validation_data, target = validation_loader[validation_idx]
            validation_tensor = torch.Tensor(validation_data).to(device)
            target_tensor = torch.Tensor(target).unsqueeze(-1).to(device)

            batch_loss = make_train_step(validation_tensor, target_tensor, optimizer, model, criterion, training=False)
            loss_validation += batch_loss
        avg_validation_loss = loss_validation / len(validation_loader)
        validate_loss_history.append(avg_validation_loss)

        print(f"epoch: {epoch} avg train loss: {avg_train_loss}")

        if epoch % 20 == 1:
            print(f"epoch: {epoch} average validation loss: {avg_validation_loss}")
        # loss_history.append([avg_train_loss, avg_validation_loss])
        print(f"train loss history:\n{train_loss_history}")
        print(f"validate loss history:\n{validate_loss_history}")

    timestamp = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
    model.save(os.path.join(model_dir), f"{timestamp}_model.pt")
    plot_series(series=[train_loss_history, validate_loss_history], x_label="Epoch", y_label="Loss",
                legend_labels=["Training", "Validation"], ticks=1.,
                path=os.path.join(eval_dir, f"{timestamp}_loss.png"))


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
        print("Training a model!")
        train(args.data_dir, args.output_dir)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training endpoint")
    parser.add_argument("command", choices=["train"])
    # path = os.path.join("C:\\", "Users", "benja", "myProjects", "ma", "data")  # , "train", "ROSTOCK")
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, os.pardir, "data"),
                        help="Path to data file directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, os.pardir, "output"),
                        help="Path to output directory")
    main(parser.parse_args())
