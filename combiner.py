import argparse
import json
import numpy as np
import os
import pandas as pd
import re

from typing import Dict, List, Tuple
from util import is_empty
from logger import Logger

script_dir = os.path.abspath(os.path.dirname(__file__))
logger = Logger("combiner")


class UnlabeledDataFile:
    def __init__(self, path: str, min_time: float, max_time):
        self.path = path
        self.min_time = min_time
        self.max_time = max_time


class RouteCombiner:
    def __init__(self, data_dir: str, csv_map_path: str):
        self.data_dir = data_dir
        self.csv_map_path = csv_map_path
        self.files: Dict[str, UnlabeledDataFile] = {}
        self.source_map: json = {}
        self.id_re = r'\d+'

    def fit(self) -> None:
        # print(f"Fitting data combiner on directory {self.data_dir}")
        for idx, data_file in enumerate(os.listdir(self.data_dir)):
            if data_file.startswith("u_data"):
                f_mmsi, f_source_date = self.mmsi_and_date_from_u_file(data_file)
                f_key = self.accessor(f_mmsi, f_source_date)
                if self.has_match(f_mmsi, f_source_date):
                    raise ValueError(f"Error fitting directory '{self.data_dir}': Multiple files exist for MMSI"
                                     f"'{f_mmsi}' and date '{f_source_date}'")
                self.files.update({f_key: UnlabeledDataFile(os.path.join(self.data_dir, data_file), min_time=0.,
                                                            max_time=0.)})
        # print(f"Done! Files:\n{self.files}")
        # print(f"Loading mapping from file {self.csv_map_path}")
        self.source_map = self.read_csv_map()

    def unfit(self, file_key: str) -> None:
        if file_key not in self.files:
            raise KeyError(f"Unable to unfit: No file with key '{file_key}' in self.files")
        del self.files[file_key]

    def has_match(self, mmsi: str, source_date: str) -> bool:
        return len(self.get_keys_and_matches(mmsi, source_date)) > 0

    def get_keys_and_matches(self, mmsi: str, source_date: str) -> List[Tuple[str, UnlabeledDataFile]]:
        result = []
        if source_date in self.source_map:  # check if mappable other csv has been processed
            for date in self.source_map[source_date]:
                f_key = self.accessor(mmsi, date)
                if f_key in self.files:
                    result.append((f_key, self.files[f_key]))
        return result

    def match(self, mmsi: str, source_date: str, new_data: pd.DataFrame) -> pd.DataFrame:
        if is_empty(new_data):
            # print(f"Unable to match data: 'new_data' is empty")
            return new_data
        # print(f"valid mappings:\n{self.source_map}")

        matches = self.get_keys_and_matches(mmsi, source_date)
        # print(f"input\n{new_data}")
        result = pd.DataFrame
        for f_key, match in matches:
            match_data = pd.read_pickle(match.path)
            if is_empty(match_data):
                self.remove_u_file(f_key)
                continue

            min_time = new_data.iloc[0]["time"]
            # max_time = new_data.iloc[-1]["# Timestamp"]
            match_min_time = match_data.iloc[0]["time"]
            if match_min_time <= min_time:  # unlabeled data is older than current data
                result = pd.concat([match_data, new_data])
                logger.write(f"Data match: MMSI {mmsi} | match min time: {match_min_time} new data time: {min_time}\n"
                             f"\tmatch data lenght: {len(match_data)} new data length: {len(new_data.index)}"
                             f"result data length: {len(result.index)}")
            else:  # unlabeled data is newer than current data
                logger.write(f"Detected match where unlabeled data is newer than current data! Not combining!")
                # result = pd.concat([new_data, match_data])
            self.remove_u_file(f_key)
        # print(f"output:\n{result}")
        return result

    def date_from_source_csv(self, file_name: str) -> str:
        source_date = re.findall(self.id_re, file_name)
        if source_date[0] is not None:
            return source_date[0]
        else:
            return ""

    def mmsi_and_date_from_u_file(self, file_name: str) -> Tuple[str, str]:
        re_result = re.findall(self.id_re, file_name)
        if re_result[0] is not None and re_result[1] is not None:
            return re_result[0], re_result[1]

    def read_csv_map(self) -> json:
        if os.path.exists(self.csv_map_path):
            with open(self.csv_map_path) as json_file:
                source_map = json.load(json_file)
                return source_map
        else:
            raise ValueError(f"Unable to read .csv mapping file from {self.csv_map_path}")

    def remove_u_file(self, file_key):
        file = self.files[file_key]
        if os.path.exists(file.path):
            os.remove(file.path)
        self.unfit(file_key)

    @staticmethod
    def accessor(mmsi: str, source_date: str) -> str:
        return f"{mmsi}-{source_date}"


def main(args) -> None:
    if args.command == "fit":
        rc = RouteCombiner(args.data_dir, args.csv_map_path)
        rc.fit()
    else:
        raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("command", choices=["fit"])
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, "data", "unlabeled", "ROSTOCK"),
                        help="Directory with unlabeled data")
    parser.add_argument("--csv_map_path", type=str, default=os.path.join(script_dir, "data", "raw", "dma",
                                                                         "csv_to_route.json"))
    main(parser.parse_args())
