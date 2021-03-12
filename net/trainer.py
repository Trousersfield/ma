import argparse
import os

from import
from net.model import InceptionTimeModel

script_dir = os.path.abspath(os.path.dirname(__file__))


def train(data, num_epochs: int, learning_rate: int) -> None:
    model = InceptionTimeModel(num_inception_blocks=1, in_channels=9, out_channels=1, kernel_sizes=1)

    # get learnable parameters
    params = list(model.parameters())
    print("number of params: ", len(params))
    print("first param's weight: ", params[0].size)


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

