import argparse
import numpy as np
import os
import pandas as pd
import re

from typing import Dict
from util import is_empty

script_dir = os.path.abspath(os.path.dirname(__file__))


class UData:
    def __init__(self, path: str, min_time: float, max_time):
        self.path = path
        self.min_time = min_time
        self.max_time = max_time


class RouteCombiner:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files: Dict[str, UData] = {}

    def fit(self) -> None:
        print(f"Fitting data combiner on directory {self.data_dir}")
        for idx, data_file in enumerate(os.listdir(self.data_dir)):
            if data_file.startswith("u_data_"):
                f_mmsi = re.findall(r'\d+', data_file)[0]
                # TODO: Make sure there always is only one file for one MMSI
                if self.has_match(f_mmsi):
                    raise ValueError(f"Error fitting directory '{self.data_dir}': A file for MMSI '{f_mmsi}'"
                                     f"already exists")
                self.files.update({f_mmsi: UData(os.path.join(self.data_dir, data_file), min_time=0., max_time=0.)})

    def unfit(self, mmsi: str) -> None:
        if not self.has_match(mmsi):
            raise ValueError(f"Unable to unfit: No file for MMSI '{mmsi}' found")
        del self.files[mmsi]

    def has_match(self, mmsi: str) -> bool:
        return mmsi in self.files

    def match(self, mmsi: str, new_data: pd.DataFrame) -> pd.DataFrame:
        print(f"Matching data")
        if new_data.shape[0] == 0 or not self.has_match(mmsi):
            return new_data
        match = self.files[mmsi]
        match_data = pd.read_pickle(match.path)
        # first entry of a row is normalized timestamp
        min_time = new_data.iloc[0]["# Timestamp"]
        max_time = new_data.iloc[-1]["# Timestamp"]
        # desired case: new data continues unlabeled data
        if match.max_time <= min_time:
            self.unfit(mmsi)
            return pd.concat([match_data, new_data])
        # unusual case TODO: handle it somehow better
        else:
            self.unfit(mmsi)
            return pd.concat([match_data, match_data])


def main(args) -> None:
    if args.command == "fit":
        rc = RouteCombiner(args.data_dir)
        rc.fit()
    else:
        raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("command", choices=["fit"])
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, "data", "unlabeled", "ROSTOCK"),
                        help="Directory with unlabeled data")
    main(parser.parse_args())
