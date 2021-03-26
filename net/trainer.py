import argparse
import os
import torch

from ..loader import TrainingExampleLoader
from ..plotter import plot_loss
from net.model import InceptionTimeModel

script_dir = os.path.abspath(os.path.dirname(__file__))


def train(train_dir: str, validation_dir: str, num_epochs: int = 10, learning_rate: float = .01) -> None:
    train_loader = TrainingExampleLoader(train_dir)
    validation_loader = TrainingExampleLoader(validation_dir)
    train_loader.load()
    validation_loader.load()

    if len(train_loader) == 0:
        raise ValueError("Unable to load data from directory {}!\nData loader must be initialized first!"
                         .format(train_loader.data_dir))
    if len(validation_loader) == 0:
        raise ValueError("Unable to load data from directory {}!\nData loader must be initialized first!"
                         .format(validation_loader.data_dir))

    data, target = train_loader[0]
    input_dim = data.shape[1]
    output_dim = target.shape[1]

    model = InceptionTimeModel(num_inception_blocks=1, in_channels=input_dim, out_channels=64,
                               num_bottleneck_channels=input_dim // 2, use_residual=True,
                               num_dense_blocks=1, dense_in_channels=1,
                               output_dim=output_dim)

    print("model: \n".format(model))

    loss_history = []
    criterion: torch.nn.MSELoss = torch.nn.MSELoss()
    # test what happens if using "weight_decay" e.g. with 1e-4
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training from directory {train_loader.data_dir} on {len(train_loader)} training examples")
    print(f"First training example: \n{train_loader[0]}")

    # get learnable parameters
    params = list(model.parameters())
    print("number of params: ", len(params))
    print("first param's weight: ", params[0].size)

    # training loop
    for epoch in range(num_epochs):
        loss_train = 0
        loss_validation = 0

        # train model
        model.train()
        for train_idx in range(len(train_loader)):
            train_data, target = train_loader[train_idx]
            data_tensor = torch.Tensor(train_data)
            target_tensor = torch.Tensor(target)

            batch_loss = make_train_step(data_tensor, target_tensor, optimizer, model, criterion)
            loss_train += batch_loss
        loss_train_avg = loss_train / len(train_loader)

        # validate model
        model.eval()
        for validation_idx in range(len(validation_loader)):
            validation_data, target = validation_loader[validation_idx]
            validation_tensor = torch.Tensor(validation_data)
            target_tensor = torch.Tensor(target)

            batch_loss = make_train_step(validation_tensor, target_tensor, optimizer, model, criterion, training=False)
            loss_validation += batch_loss
        loss_validation_avg = loss_validation / len(validation_loader)

        if epoch % 20 == 1:
            print(f"epoch: {epoch} average validation loss: {loss_validation_avg}")
        loss_history.append([loss_train_avg, loss_validation_avg])
    plot_loss(loss_history, labels=["Training", "Validation"])


def make_train_step(data_tensor: torch.Tensor, target_tensor: torch.Tensor, optimizer, model, criterion,
                    training: bool = True):
    output = model(data_tensor)
    loss = criterion(output, target_tensor)
    if training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def test(data_dir) -> None:
    loader = TrainingExampleLoader(data_dir)
    loader.load()
    window = loader[0]
    print("window: ", window)


def main(args) -> None:
    if args.command == "train":
        print("Training a model!")
        train(args.data_dir, args.validation_dir)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training endpoint")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--train_dir", type=str, default=os.path.join(script_dir, "data", "train", "ROSTOCK"),
                        help="Path to data files")
    parser.add_argument("--validation_dir", type=str, default=os.path.join(script_dir, "data", "test", "ROSTOCK"),
                        help="Path to validation files")
    main(parser.parse_args())

