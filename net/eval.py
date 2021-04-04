import argparse
import os
import torch

from datetime import datetime
from loader import MmsiDataFile, TrainingExampleLoader
from net.model import InceptionTimeModel

from typing import List, Tuple


# Evaluationsmethoden: https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234


def evaluate(model_path: str, data_path: str) -> None:
    eval_loader = TrainingExampleLoader(data_path)
    eval_loader.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionTimeModel.load(model_path).to(device)
    model.eval()

    print(f"Number of parameters: {5}")

    eval_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for eval_idx in range(len(eval_loader)):
        eval_data, target = eval_loader[eval_idx]
        data_tensor = torch.Tensor(eval_data).to(device)
        target_tensor = torch.Tensor(target).to(device)

        output = model(data_tensor)
        eval_outputs.append((output, target_tensor))


def main(args) -> None:
    evaluate(args.model_path, args.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to model")
    parser.add_argument("data_path", type=str, help="Path to data")
    main(parser.parse_args())
