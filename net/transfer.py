import argparse
import joblib
import json
import os

from datetime import datetime
from typing import Dict, List, Union

from port import Port, PortManager
from util import decode_model_file

script_dir = os.path.abspath(os.path.dirname(__file__))


class TransferDefinition:
    def __init__(self, base_model_path: str, target_model_dir: str, target_port: str):
        self.base_model_path = base_model_path
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
        self.transfer_definitions: List[TransferDefinition] = []
        self.completed_transfers: List[TransferResult] = []

    def transfer(self, transfer_definition: TransferDefinition) -> TransferResult:
        result = TransferResult(transfer_definition)
        self.completed_transfers.append(result)
        return result

    def load(self, transfer_definition_path: str) -> None:
        if os.path.exists(transfer_definition_path):
            self.transfer_definitions = joblib.load(transfer_definition_path)
        else:
            raise FileNotFoundError(f"No transfer definition file found at '{transfer_definition_path}'. "
                                    f"Make sure to generate definitions are generated.")


def generate_transfers(config_path: str, transfer_latest_model: bool = True) -> None:
    pm = PortManager()
    pm.load()
    if len(pm.ports.keys()) < 1:
        raise ValueError("No port data available")
    config = read_json(config_path)
    transfers: List[TransferDefinition] = []

    for transfer_def in config:
        base_port = pm.find_port(transfer_def["base_port"])
        for target_port_name in transfer_def["target_ports"]:
            target_port = pm.find_port(target_port_name)

            if transfer_latest_model:
                base_model_path = base_port.trainings[-1].model_path
                target_model_dir = os.path.split(base_model_path)[0]
                if not os.path.exists(target_model_dir):
                    os.makedirs(target_model_dir)

                transfers.append(TransferDefinition(base_model_path=base_model_path,
                                                    target_model_dir=target_model_dir,
                                                    target_port=target_port.name))
    # dump in same directory as config
    joblib.dump(transfers, os.path.join(os.path.split(config_path)[0], "transfer_definitions.pkl"))


def read_json(path: str) -> json:
    if os.path.exists(path):
        with open(path) as json_file:
            result = json.load(json_file)
            return result
    else:
        raise ValueError(f"Unable to read .json file from {path}")


# entry point for transferring models
def transfer(td_dir: str) -> None:
    mt = ModelTransactor()
    tds = mt.load_transfer_definitions(td_dir)

    for td in tds:
        result = mt.transfer(td)


def main(args) -> None:
    if args.command == "transfer":
        transfer(args.transfer_dir)
    elif args.command == "generate":
        generate_transfers(args.config_path)
    else:
        raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("command", choices=["read_ports"])
    parser.add_argument("--transfer_dir", type=str, default=os.path.join(script_dir, "transfer"),
                        help="Directory to transfer definition files")
    parser.add_argument("--config_path", type=str, default=os.path.join(script_dir, "transfer", "config.json"),
                        help="Path to file for transfer definition generation")
    main(parser.parse_args())

