import argparse
import os
import torch

from ..loader import TrainingExampleLoader
from net.model import InceptionTimeModel

script_dir = os.path.abspath(os.path.dirname(__file__))


def train(data_dir, num_epochs: int = 10, learning_rate: float = .01) -> None:
    model = InceptionTimeModel(num_inception_blocks=1, in_channels=9, out_channels=9, kernel_sizes=9)
    loader = TrainingExampleLoader(data_dir)
    loader.load()

    if len(loader) == 0:
        raise ValueError("Unable to load data from directory {}!\nData loader must be initialized first!"
                         .format(loader.data_dir))

    print("model: \n".format(model))

    loss_history = []
    criterion: torch.nn.MSELoss = torch.nn.MSELoss()
    optimizer: torch.optim.Adamax = torch.optim.Adamax(model.parameters(), lr=learning_rate)

    # get learnable parameters
    params = list(model.parameters())
    print("number of params: ", len(params))
    print("first param's weight: ", params[0].size)

    # data stuff
    print("Starting training from directory {} on {} training examples".format(loader.data_dir, len(loader)))
    print("First example: \n{}".format(loader[0]))

    # training loop
    for epoch in range(num_epochs):
        loss_epoch = []
        model.train()

        # iterate training examples
        for train_idx in range(len(loader)):
            data, target = loader[train_idx]
            data_tensor = torch.Tensor(data)
            target_tensor = torch.Tensor(target)
            optimizer.zero_grad()

            output = model(data_tensor)
            loss = criterion(output, target_tensor)
            loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()

        loss_history.append(loss_epoch)


def test() -> None:
    loader = TrainingExampleLoader()
    loader.load()
    training_example = loader[0]
    print("training example: ", training_example)


def main(args) -> None:
    if args.command == "train":
        print("Training a model!")
    elif args.command == "test":
        print("Testing a model!")
        test()
    else:
        raise ValueError("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training endpoint")
    parser.add_argument("command", choices=["train", "test"])
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, "data", "train", "COPENGAHEN"),
                        help="Path to data files")
    main(parser.parse_args())

