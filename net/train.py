import argparse
import os

def train(dataset_path, batch_size, num_epochs, learning_rate):
    return

def main(args):
    batch_size = 64
    learning_rate = 5e-5
    num_epochs = 50

    train(args.dataset_path, batch_size, num_epochs, learning_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    main(parser.parse_args())
