import argparse
import joblib
import json
import numpy as np
import os
import pandas as pd

from logger import Logger
from output_collector import OutputCollector
from training import TrainingIteration
from util import as_float, as_str, encode_pm_file, decode_pm_file, is_empty, get_destination_file_name,\
    decode_loss_file, decode_loss_plot, decode_checkpoint_file, decode_model_file, encode_dataset_config_file

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
        self.r_lat = self._km_to_lat(radius)   # r in degrees for lat
        self.r_long = self._km_to_long(radius, self.latitude)  # r in degrees for long
        hypotenuse = haversine(self.latitude+self.r_lat, self.longitude, self.latitude, self.longitude+self.r_long) # km
        self.inner_square_lat_radius = self._km_to_lat(hypotenuse/2)
        self.inner_square_long_radius = self._km_to_long(hypotenuse/2, self.latitude)

    def _km_to_lat(self, km: float) -> float:
        return 1/(self.KM_TO_LAT_FACTOR*km) if km > 0 else 0

    def _km_to_long(self, km: float, latitude: float) -> float:
        return 1/(self.KM_TO_LONG_FACTOR*km*cos(radians(latitude))) if km > 0 else 0


class PortManager:
    def __init__(self, init_new: bool = False, port_path: str = "", alias_path: str = "") -> None:
        self.ports: Dict[str, Port] = dict()
        self.alias: Dict[str, str] = dict()
        self.port_path = self._init_port_path(init_new, port_path)
        if os.path.exists(alias_path):
            self.alias_path = alias_path
        else:
            self.alias_path = os.path.join(script_dir, "alias.pkl")

    @staticmethod
    def _init_port_path(init_new: bool, port_path: str) -> str:
        if (port_path != "") and os.path.exists(port_path):
            return port_path
        elif init_new:
            now = as_str(datetime.now())
            return os.path.join(script_dir, encode_pm_file(now))
        else:  # look for latest pm-ports file
            times = []
            for idx, data_file in enumerate(os.listdir(script_dir)):
                if data_file.startswith("pm-ports_"):
                    _, time = decode_pm_file(data_file)
                    times.append(time)
            if len(times) > 0:
                times = sorted(times)
                return os.path.join(script_dir, encode_pm_file(times[-1]))
            else:
                raise ValueError(f"Unable to initialize Port Manager on directory '{script_dir}': "
                                 f"No 'pm-ports'-file found. Initialize new if desired.")

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
            print(f"No alias definition found at {self.alias_path}")
        print(f"Port Manager loaded: {len(self.ports.keys())} ports; {len(self.alias.keys())} alias'")
        # print(f"Ports: {self.ports.keys()}")

    def save(self) -> None:
        joblib.dump(self.ports, self.port_path)

    def add_port(self, port: Port) -> None:
        # TODO: check if port name already in place and ask for confirmation if values overwrite
        # (over)write port information
        if port.name != "":
            self.ports[port.name] = port
            self.save()

    def add_alias(self, port_name: str, alias: List[str], overwrite: bool = False) -> None:
        for al in alias:
            al = al.upper()
            if port_name in self.ports.keys():
                if overwrite or (al not in self.alias.keys()):
                    self.alias[al] = port_name
                    joblib.dump(self.alias, self.alias_path)
                    print(f"Alias '{al}' added. Associated port: {self.alias[al]}")
                else:
                    print(f"Alias '{al}' already contained! Associated port: {self.alias[al]}")
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

    def load_trainings(self, port: Union[str, Port], output_dir: str, routes_dir: str,
                       training_type: str) -> List[TrainingIteration]:
        if not os.path.exists(output_dir):
            raise ValueError(f"No such directory: {output_dir}")
        if not os.path.exists(routes_dir):
            raise ValueError(f"No such directory: {routes_dir}")
        if training_type not in ["base", "transfer"]:
            raise ValueError(f"Unknown training type '{training_type}': Not in [base, transfer]")
        orig_port = port
        if isinstance(port, str):
            port = self.find_port(port)
        if port is None:
            raise ValueError(f"Unable to associate port with '{orig_port}'")

        oc = OutputCollector(output_dir)
        data_paths = oc.collect_data(port.name, file_type=training_type, group=True)
        debug_paths = oc.collect_debug(port.name, file_type=training_type, group=True)
        log_paths = oc.collect_log(port.name, file_type=training_type, group=True)
        model_paths = oc.collect_model(port.name, file_type=training_type, group=True)
        plot_paths = oc.collect_plot(port.name, file_type=training_type, group=True)
        eval_paths = oc.collect_eval(port.name, file_type=training_type, group=True)

        trainings = []
        start_times = model_paths.keys()
        for start in sorted(start_times):
            _, _, _, end, _ = decode_model_file(model_paths[start])
            dataset_config_path = os.path.join(routes_dir, port.name, encode_dataset_config_file(start, training_type))
            ti = TrainingIteration(start_time=start, end_time=end,
                                   data_path=data_paths[start] if start in data_paths else None,
                                   log_path=log_paths[start] if start in log_paths else None,
                                   model_path=model_paths[start] if start in model_paths else None,
                                   plot_paths=plot_paths[start] if start in plot_paths else None,
                                   debug_path=debug_paths[start] if start in debug_paths else None,
                                   eval_paths=eval_paths[start] if start in eval_paths else None,
                                   dataset_config_path=dataset_config_path)

            trainings.append(ti)
        return trainings


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
    elif args.command == "add_alias":
        pm = PortManager()
        pm.load()
        pm.add_alias("ANTWERPEN", ["ANTWERPBEL", "ANTWERPEN", "ANTWERP"])
        pm.add_alias("BREMERHAVEN", ["BREMENGERMANY", "BREMEN", "BREMENDE", "DEBREMEN", "BREMEHAVEN", "BREMERNHAVEN"])
        pm.add_alias("CUXHAVEN", ["GERCUXHAVEN", "CUXHAVENGER"])
        pm.add_alias("ESBJERG", ["ESBJERGDK", "DKESBJERG", "ESBJERGDENMARK", "DENMARKESBJERG"])
        pm.add_alias("FREDERICIA", ["FREDERICIA_DK", "DKFREDERICIA", "FREDERICIADK", "DENMARKESBJERG", "DKFREDERICIA"])
        pm.add_alias("FREDERIKSHAVN", ["FREDERIKSHAVEN"])
        pm.add_alias("GRENAA", ["GRENAADK", "GRENAADMK", "DKGRENAA", "GRENA", "GRENAADENMARK", "DENMARKGRENAA"])
        pm.add_alias("HAMBURG", ["HAMBURGGER", "GERHAMBURG"])
        pm.add_alias("HIRTSHALS", ["HIRTSHALSDMK", "DKHIRTHALS", "DMKHIRTSHALS", "HIRTSHALSDK"])
        pm.add_alias("HARLINGEN", ["HARLINGENX"])
        pm.add_alias("HVIDESANDE", ["HVIDE SANDE", "HVIDESANDEDK", "HVSANDE"])
        pm.add_alias("KALUNDBORG", ["DKKALUNDBORG", "KALUNDBORGDK", "KALUNDBORGDENMARK", "DENMARKKALUNDBORG"])
        pm.add_alias("KIEL", ["KIELDE", "DEKIEL", "KIELGER", "GERKIEL"])
        pm.add_alias("MALMO", ["MALMOSWE", "SWEMALMO", "SWEDENMALMO", "MALMOSWEDEN"])
        pm.add_alias("ODENSE", ["ODENSEDENMARK", "DENMARKODENSE", "ODENSEDK", "DKODENSE"])
        pm.add_alias("PETERSBURG", ["ST.PETERSBURG", "STPETERSBURG", "SAITPETERSBURG"])
        pm.add_alias("RIGA", ["RIGALATIVIA", "RIGALAT", "LATRIGA"])
        pm.add_alias("SKAGEN", ["SKAGENDK", "DKSKAGEN", "SKAGENDENMARK", "DENMARKSKAGEN"])
        pm.add_alias("THYBORON", ["THYBOROEN"])
        pm.add_alias("VARBERG", ["VARBERGV"])
        pm.add_alias("GOTEBORG", ["GOTHENBURG", "GOTHEBORG", "GOTHEBORGSWE", "SWEGOTHEBORG", "GOTEBORGSWE",
                                  "SWEGOTEBORG", "SWEDENGOTEBORG", "GOTEBORGSWEDEN"])
    elif args.command == "analyze_csv":
        analyze_csv(args.csv_dir, args.file_name)
    elif args.command == "load_trainings":
        pm = PortManager()
        pm.load()
        port = pm.find_port(args.port_name)
        if port is None:
            raise ValueError(f"Unable to associate port with name '{args.port_name}'")
        trainings = pm.load_trainings(port, output_dir=args.output_dir, routes_dir=args.routes_dir,
                                      training_type="base")
        print(f"Base trainings: {trainings}")
        transfers = pm.load_trainings(port, output_dir=args.output_dir, routes_dir=args.routes_dir,
                                      training_type="transfer")
        print(f"Transfer trainings: {transfers}")
    else:
        raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handle ports")
    parser.add_argument("command", choices=["init_ports", "add_alias", "analyze_csv", "test"])
    parser.add_argument("--csv_dir", type=str, default=os.path.join(script_dir, "data", "raw", "dma"),
                        help="Path to directory of AIS .csv files")
    parser.add_argument("--file_name", type=str,
                        help="Specify single file name if ports shall get read from specific .csv file")
    parser.add_argument("--port_name", type=str)
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "output"))
    parser.add_argument("--routes_dir", type=str, default=os.path.join(script_dir, "data", "routes"))
    main(parser.parse_args())
