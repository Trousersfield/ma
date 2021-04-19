import argparse
import os
import torch
from torch import nn

from loader import MmsiDataFile, TrainingExampleLoader
from net.model import InceptionTimeModel
from util import compute_mae, compute_mse

from typing import List, Tuple

script_dir = os.path.abspath(os.path.dirname(__file__))

# Evaluationsmethoden: https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234


def evaluate(model_path: str, data_dir: str) -> None:
    loader = TrainingExampleLoader(data_dir)
    loader.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionTimeModel.load(model_path, device).to(device)
    model.eval()

    print(f"Model has {num_parameters(model)} Parameters")

    outputs = []
    labels = []
    test_indices = loader.shuffled_data_indices(kind="test")
    for test_idx in test_indices:
        test_data, target = loader[test_idx]
        data_tensor = torch.Tensor(test_data).to(device)
        target_tensor = torch.Tensor(target).unsqueeze(-1).to(device)

        output = model(data_tensor)
        outputs.append(output)
        labels.append(target_tensor)

    print(f"labels:\n{labels}")
    print(f"outputs:\n{outputs}")
    mae = compute_mae(labels, outputs)
    mse = compute_mse(labels, outputs)
    print(f"mae: {mae}")
    print(f"mse: {mse}")


def num_parameters(model: nn.Module) -> float:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main(args) -> None:
    command = args.command
    if command == "evaluate":
        evaluate(args.model_path, args.data_dir)
    else:
        raise ValueError(f"Unknown command '{command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["evaluate"])
    parser.add_argument("--model_path", type=str, default=os.path.join(script_dir, os.pardir, "output", "model",
                                                                       "BREMERHAVEN_20210416-092158.pt"),
                        help="Path to model")
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, os.pardir, "data", "validate",
                                                                     "BREMERHAVEN"),
                        help="Directory to evaluation data")
    main(parser.parse_args())
