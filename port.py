import argparse
import joblib
import json
import numpy as np
import os
import pandas as pd

from logger import Logger
from training import TrainingIteration, make_training_iteration
from util import as_float, as_str, encode_pm_file, decode_pm_file, is_empty, get_destination_file_name

from datetime import datetime
from math import radians, cos, sin, asin, sqrt, degrees
from typing import Dict, List, Tuple, Union

script_dir = os.path.abspath(os.path.dirname(__file__))


class Port:
    KM_TO_LAT_FACTOR = 110.574
    KM_TO_LONG_FACTOR = 111.230

    def __init__(self, name: str, latitude: float, longitude: float, radius: float) -> None:
        self.name = name
        self.latitude = latitude    # degrees
        self.longitude = longitude  # degrees
        self.radius = radius        # km
        # approximate maximum lat and long distances from center point of the maximum square within radius r
        self.r_lat = self.km_to_lat(radius)   # r in degrees for lat
        self.r_long = self.km_to_long(radius, self.latitude)  # r in degrees for long
        hypotenuse = haversine(self.latitude+self.r_lat, self.longitude, self.latitude, self.longitude+self.r_long) # km
        self.inner_square_lat_radius = self.km_to_lat(hypotenuse/2)
        self.inner_square_long_radius = self.km_to_long(hypotenuse/2, self.latitude)
        self.trainings: Dict[float, TrainingIteration] = dict()

    def km_to_lat(self, km: float) -> float:
        return 1/(self.KM_TO_LAT_FACTOR*km) if km > 0 else 0

    def km_to_long(self, km: float, latitude: float) -> float:
        return 1/(self.KM_TO_LONG_FACTOR*km*cos(radians(latitude))) if km > 0 else 0


class PortManager:
    def __init__(self, init_new: bool = False, port_path: str = "", alias_path: str = "") -> None:
        self.ports: Dict[str, Port] = dict()
        self.alias: Dict[str, str] = dict()
        if (port_path != "") and os.path.exists(port_path):
            self.port_path = port_path
        else:
            if init_new:
                now = as_str(datetime.now())
                self.port_path = os.path.join(script_dir, encode_pm_file(now))
            else:
                times = []
                for idx, data_file in enumerate(os.listdir(script_dir)):
                    if data_file.startswith("pm-ports_"):
                        _, time = decode_pm_file(data_file)
                        times.append(time)
                if len(times) > 0:
                    times = sorted(times)
                    self.port_path = os.path.join(script_dir, encode_pm_file(times[-1]))
                else:
                    raise ValueError(f"Unable to initialize Port Manager on directory '{script_dir}': "
                                     f"No 'pm-ports'-file found. Initialize new if desired.")
        if os.path.exists(alias_path):
            self.alias_path = alias_path
        else:
            self.alias_path = os.path.join(script_dir, "alias.pkl")

    def generate_from_source(self, source_path: str = "", load: bool = False) -> None:
        if source_path == "":
            source_path = os.path.join(script_dir, "port.json")
        if os.path.exists(source_path):
            with open(source_path) as json_file:
                # ports: Dict[str, Port] = dict()
                data = json.load(json_file)

                for port_row in data["rows"]:
                    self.ports[port_row["name"]] = Port(port_row["name"], float(port_row["latitude"]),
                                                        float(port_row["longitude"]),
                                                        float(port_row["radius"]))
                self.save()

            if load:
                self.load()
        else:
            print(f"Unable to locate port source file at {source_path}")

    def load(self) -> None:
        if os.path.exists(self.port_path):
            ports = joblib.load(self.port_path)
            self.ports = ports
        else:
            print(f"No port definition found at {self.port_path}. Generate from source first.")
        if os.path.exists(self.alias_path):
            self.alias = joblib.load(self.alias_path)
        else:
            print("No alias definition found at {}".format(self.alias_path))
        print(f"Port Manager loaded: {len(self.ports.keys())} ports; {len(self.alias.keys())} alias'")

    def save(self) -> None:
        print(f"ports:\n{self.ports}")
        joblib.dump(self.ports, self.port_path)

    def add_port(self, port: Port) -> None:
        # TODO: check if port name already in place and ask for confirmation if values overwrite
        # (over)write port information
        if port.name != "":
            self.ports[port.name] = port
            self.save()

    def add_alias(self, port_name: str, alias: str, overwrite: bool = False) -> None:
        alias = alias.upper()
        if port_name in self.ports.keys():
            if overwrite or (alias not in self.alias.keys()):
                self.alias[alias] = port_name
                joblib.dump(self.alias, self.alias_path)
                print(f"Alias '{alias}' added. Associated port: {self.alias[alias]}")
            else:
                print(f"Alias '{alias}' already contained! Associated port: {self.alias[alias]}")
        else:
            print(f"Unable to associate '{port_name}' with loaded port data. Make sure port data is loaded "
                  f"before adding alias.")

    def find_port(self, name: str) -> Port:
        # print("Searching match for name {}".format(name))
        name_upper = name.upper()
        if name_upper in self.ports:
            return self.ports[name_upper]

        if (name_upper in self.alias) and (self.alias[name_upper] in self.ports):
            return self.ports[self.alias[name_upper]]
        # print("No match found!")

    @staticmethod
    def identify_arrival_times(port: Port, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # separation by inner square of port area
        lat_mask = ((df["Latitude"] > (port.latitude - port.inner_square_lat_radius)) &
                    (df["Latitude"] < (port.latitude + port.inner_square_lat_radius)))

        df_outside_square: pd.DataFrame = df.loc[~lat_mask]
        df_inside_square: pd.DataFrame = df.loc[lat_mask]

        long_mask = ((df_inside_square["Longitude"] > (port.longitude - port.inner_square_long_radius)) &
                     (df_inside_square["Longitude"] < (port.longitude + port.inner_square_long_radius)))

        df_outside_square.append(df_inside_square.loc[~long_mask])
        df_inside_square = df_inside_square.loc[long_mask]

        # print("lat interval [{}; {}] ".format(str(port.latitude - port.inner_square_lat_radius),
        #                                      str(port.latitude + port.inner_square_lat_radius)))
        # print("long interval [{}; {}] ".format(str(port.longitude - port.inner_square_long_radius),
        #                                       str(port.longitude + port.inner_square_long_radius)))

        # accurate separation outside of inner square but within port's radius
        radius_mask = df_outside_square.apply(radius_filter, args=(port,), axis=1)

        df_outside_circle: pd.DataFrame = df_outside_square[radius_mask]  # training data series
        df_inside_circle: pd.DataFrame = df_outside_square[~radius_mask]

        # minimum timestamp of inside port area data-points is arrival time
        arrival_times: pd.DataFrame = get_minimum_time(df_inside_square, df_inside_circle)

        if is_empty(arrival_times):
            arrival_times = pd.DataFrame(columns=df_outside_circle.columns)

        return df_outside_circle, arrival_times

    def add_training(self, port: Union[str, Port], start_time: Union[float, datetime], end_time: Union[float, datetime],
                     model_path: str, loss_path: str, log_path: str) -> None:
        if port is str:
            port = self.find_port(port)
            if port is None:
                return
        if start_time is datetime:
            start_time = as_float(start_time)
        if end_time is datetime:
            end_time = as_float(end_time)
        port.trainings[end_time] = make_training_iteration(start_time, end_time, model_path, loss_path, log_path)
        self.save()

    def remove_training(self, port: Union[str, Port], training_iteration: TrainingIteration) -> None:
        if port is str:
            port = self.find_port(port)
            if port is None:
                return
        if training_iteration.end_time in port.trainings:
            del port.trainings[training_iteration.end_time]
            self.save()
        else:
            print(f"Training iteration '{training_iteration}' not found for port {port.name}")

    def reset_training(self, ports: Union[List[str], List[Port]]) -> None:
        for port in ports:
            if port is str:
                port = self.find_port(port)
            if port is not None:
                port.trainings = {}
        self.save()


def haversine(lat1: float, long1: float, lat2: float, long2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])

    # formula itself
    d_long = long2 - long1
    d_lat = lat2 - lat1
    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_long / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # radius of the earth in km
    return c * r


def radius_filter(row, port: Port) -> bool:
    if haversine(row["Latitude"], row["Longitude"], port.latitude, port.longitude) <= port.radius:
        return False
    else:
        return True


def get_minimum_time(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    if is_empty(df_1) and is_empty(df_2):
        return pd.DataFrame()

    mmsis_1 = df_1["MMSI"].unique().tolist()
    mmsis_2 = df_2["MMSI"].unique().tolist()
    mmsis = list(set(mmsis_1) | set(mmsis_2))
    # preallocate result
    result_df = pd.DataFrame(index=np.arange(0, len(mmsis)), columns=df_1.columns)

    for idx, mmsi in enumerate(mmsis):
        mmsi_df_1 = df_1.loc[df_1["MMSI"] == mmsi]
        mmsi_df_2 = df_2.loc[df_2["MMSI"] == mmsi]
        min_df_1 = mmsi_df_1.loc[mmsi_df_1["time"] == mmsi_df_1["time"].min()]
        min_df_2 = mmsi_df_2.loc[mmsi_df_2["time"] == mmsi_df_2["time"].min()]
        # print("min_df_1: \n", min_df_1)
        # print("min_df_2: \n", min_df_2)

        # make sure no duplicate arrival times occur in case of identical min timestamps
        len_df_1 = len(min_df_1.index)
        len_df_2 = len(min_df_2.index)
        if len_df_1 > 1:
            min_df_1 = min_df_1.drop(min_df_1.index[[1, len_df_1-1]])
        if len_df_2 > 1:
            min_df_1 = min_df_2.drop(min_df_2.index[[1, len_df_2-1]])

        # assign minimum of both DataFrames to result
        if len_df_1 > 0:
            if len_df_2 > 0:
                min_time_1 = min_df_1["time"].iloc[0]
                min_time_2 = min_df_2["time"].iloc[0]
                result_df.loc[idx] = min_df_1.iloc[0] if min_time_1 < min_time_2 else min_df_2.iloc[0]
            else:
                result_df.loc[idx] = min_df_1.iloc[0]
        elif len_df_2 > 0:
            result_df.loc[idx] = min_df_2.iloc[0]
        else:
            result_df.drop([idx])
    # print("min df for each mmsi: \n", result_df)

    return result_df


def analyze_csv(input_dir: str, file_name: str = None) -> None:
    print("Reading ports...")
    logger = Logger(f"existing_ports_{file_name}")
    port_info: Dict[str, int] = {}

    if file_name is not None:
        print(f"Processing file at {os.path.join(input_dir, file_name)}")
        df = pd.read_csv(os.path.join(input_dir, file_name), ",", None)
        port_info = agg_ports_info(df, port_info)
    else:
        for idx, data_file in enumerate(os.listdir(input_dir)):
            if data_file.startswith("aisdk_"):
                print(f"Processing file at {os.path.join(input_dir, data_file)}")
                df = pd.read_csv(os.path.join(input_dir, data_file), ",", None)
                port_info = agg_ports_info(df, port_info)

    output = ""
    for port, rows in sorted(port_info.items(), key=lambda item: item[1], reverse=True):
        output += f"Port {port}: {rows} rows\n"
    logger.write(output)
    print(f"Done!")


def agg_ports_info(df: pd.DataFrame, port_info: Dict[str, int] = None) -> Dict[str, int]:
    ports: List[str] = df["Destination"].unique()
    if port_info is None:
        port_info = {}

    for port_column_header in ports:
        if pd.isnull(port_column_header):
            continue

        port: str = get_destination_file_name(port_column_header)
        if port in port_info:
            port_info[port] += len(df.loc[df["Destination"] == port_column_header].index)
        else:
            port_info[port] = len(df.loc[df["Destination"] == port_column_header].index)
    return port_info


def main(args) -> None:
    if args.command == "init_ports":
        pm = PortManager(init_new=True)
        pm.generate_from_source(load=True)
    elif args.command == "analyze_csv":
        analyze_csv(args.input_dir, args.file_name)
    elif args.command == "test":
        pm = PortManager()
        pm.load()
        port = pm.find_port("hamburg")
        print(f"Port Manager:\n{port.trainings}")
    else:
        raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handle ports")
    parser.add_argument("command", choices=["init_ports", "analyze_csv", "test"])
    parser.add_argument("--input_dir", type=str, default=os.path.join(script_dir, "data", "raw", "dma"),
                        help="Path to directory of AIS .csv files")
    parser.add_argument("--file_name", type=str,
                        help="Specify single file name if ports shall get read from specific .csv file")
    main(parser.parse_args())
