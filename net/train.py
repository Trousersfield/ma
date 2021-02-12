import argparse
import os

from torch.utils.data import DataLoader, Dataset


def train(dataset_path, batch_size, num_epochs, learning_rate):


    # TODO: What does pin_memory do?
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    train(args.dataset_path, batch_size, num_epochs, learning_rate)

    return

def main(args):
    batch_size = 64
    num_epochs = 50
    learning_rate = 5e-5

    train(path, batch_size, num_epochs, learning_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    main(parser.parse_args())
