import argparse
import joblib
import json
import os

from datetime import datetime
from typing import Dict, List, Union

from util import decode_model_file

script_dir = os.path.abspath(os.path.dirname(__file__))


class TransferDefinition:
    def __init__(self, source_model_path: str, target_model_dir: str, target_port: str):
        self.source_model_path = source_model_path
        self.target_port = target_port
        self.target_model_dir = target_model_dir


class TransferResult:
    def __init__(self, transfer_definition: TransferDefinition, start_time: datetime = None):
        self.transfer_definition = transfer_definition
        self.start_time = start_time
        self.end_time = datetime.now()
        self.result = "Result of transfer"

    def str_start_time(self) -> str:
        return "no-start-time" if self.start_time is None else datetime.strftime(self.start_time, "%Y%m%d-%H%M%S")

    def str_end_time(self) -> str:
        return datetime.strftime(self.end_time, "%Y%m%d-%H%M%S")


class ModelTransactor:
    def __init__(self):
        self.data = "Test"
        self.completed_transfers: List[TransferResult] = []

    def transfer(self, transfer_definition: TransferDefinition) -> TransferResult:
        result = TransferResult(transfer_definition)
        self.completed_transfers.append(result)
        return result

    @staticmethod
    def load_transfer_definitions(td_file_path: str) -> List[TransferDefinition]:
        result: List[TransferDefinition] = []
        if os.path.exists(td_file_path):
            result = joblib.load(td_file_path)
        return result


def generate_transfers(td_gen_file_path: str, source_model_dir: str, output_dir: str = None) -> None:
    if output_dir is None:
        output_dir, _ = os.path.split(td_gen_file_path)

    td_defs = read_json(td_gen_file_path)
    transfers: List[TransferDefinition] = []

    # gather existing models
    model_files: Dict[str, str] = {}
    if os.path.exists(source_model_dir):
        for idx, model_file in enumerate(os.listdir(source_model_dir)):
            port, _ = decode_model_file(model_file)
            model_files[port] = model_file

    for td_def in td_defs:
        source_port = td_def["source_port"]
        if source_port in model_files:
            sm_path = os.path.join(source_model_dir, model_files[source_port])
            transfers.append(TransferDefinition(source_model_path=sm_path, target_model_dir=output_dir,
                                                target_port=td_def["target_port"]))
    joblib.dump(transfers, os.path.join(output_dir, "transfer_definitions.pkl"))


def read_json(path: str) -> json:
    if os.path.exists(path):
        with open(path) as json_file:
            result = json.load(json_file)
            return result
    else:
        raise ValueError(f"Unable to read .json file from {path}")


def transfer(td_dir: str) -> None:
    mt = ModelTransactor()
    tds = mt.load_transfer_definitions(td_dir)

    for td in tds:
        result = mt.transfer(td)


def main(args) -> None:
    if args.command == "transfer":
        transfer(args.td_dir)
    elif args.command == "generate":
        generate_transfers(args.td_gen_path)
    else:
        raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("command", choices=["read_ports"])
    parser.add_argument("--td_dir", type=str, default=os.path.join(script_dir, "td"),
                        help="Directory to transfer definition files")
    parser.add_argument("--td_gen_path", type=str, default=os.path.join(script_dir, "td", "td_gen.json"),
                        help="Path to file for transfer definition generation")
    main(parser.parse_args())

