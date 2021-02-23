import argparse
import joblib
import json
import os
import pandas as pd

from math import radians, cos, sin, asin, sqrt, degrees
from typing import Dict, List, Tuple

from util import is_empty

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

    def km_to_lat(self, km: float) -> float:
        return 1/(self.KM_TO_LAT_FACTOR*km) if km > 0 else 0

    def km_to_long(self, km: float, latitude: float) -> float:
        return 1/(self.KM_TO_LONG_FACTOR*km*cos(radians(latitude))) if km > 0 else 0


class PortManager:
    def __init__(self, port_dir: str = "", alias_dir: str = "") -> None:
        self.ports: Dict[str, Port] = dict()
        self.alias: Dict[str, str] = dict()
        if (port_dir != "") and os.path.exists(port_dir):
            self.port_dir = port_dir
        else:
            self.port_dir = os.path.join(script_dir, "port.pkl")
        if os.path.exists(alias_dir):
            self.alias_dir = alias_dir
        else:
            self.alias_dir = os.path.join(script_dir, "alias.pkl")

    def generate_from_source(self, source_dir: str = "", load: bool = False) -> None:
        if source_dir == "":
            source_dir = os.path.join(script_dir, "port.json")
        if os.path.exists(source_dir):
            with open(source_dir) as json_file:
                ports: Dict[str, Port] = dict()
                data = json.load(json_file)

                for port_row in data["rows"]:
                    ports[port_row["name"]] = Port(port_row["name"], float(port_row["latitude"]),
                                                   float(port_row["longitude"]),
                                                   float(port_row["radius"]))

                joblib.dump(ports, self.port_dir)

            if load:
                self.load()
        else:
            print("Unable to locate port source file at {}".format(source_dir))

    def load(self) -> None:
        if os.path.exists(self.port_dir):
            self.ports = joblib.load(self.port_dir)
        else:
            print("No port definition found at {}. Generate from source first.".format(self.port_dir))
        if os.path.exists(self.alias_dir):
            self.alias = joblib.load(self.alias_dir)
        else:
            print("No alias definition found at {}".format(self.alias_dir))
        print("Port Manager loaded: {} Ports; {} Alias".format(len(self.ports.keys()), len(self.alias.keys())))

    def add_port(self, port: Port) -> None:
        # TODO: check if port name already in place and ask for confirmation if values overwrite
        # (over)write port information
        if port.name != "":
            self.ports[port.name] = port
            joblib.dump(self.ports, os.path.join(script_dir, "port.pkl"))

    def add_alias(self, port_name: str, alias: str, overwrite: bool = False) -> None:
        alias = alias.upper()
        if port_name in self.ports.keys():
            if overwrite or (alias not in self.alias.keys()):
                self.alias[alias] = port_name
                joblib.dump(self.alias, self.alias_dir)
                print("Alias {} added. Associated port: {}".format(alias, self.alias[alias]))
            else:
                print("Alias {} already contained! Associated port: {}".format(alias, self.alias[alias]))
        else:
            print("Unable to associate {} with loaded port data. Make sure port data is loaded before adding alias."
                  .format(port_name))

    def find_port(self, name: str) -> Port:
        print("Searching match for name {}".format(name))
        name_upper = name.upper()
        if name_upper in self.ports:
            return self.ports[name]

        if (name_upper in self.alias) and (self.alias[name_upper] in self.ports):
            return self.ports[self.alias[name_upper]]
        print("No match found!")

    @staticmethod
    def identify_label(port: Port, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # separation by inner square of port area
        print("Port lat/long: ", port.latitude, port.longitude)
        print("r_lat / r_long: ", port.r_lat, port.r_long)
        print("df: \n", df)
        lat_mask = ((df["Latitude"] > (port.latitude - port.inner_square_lat_radius)) &
                    (df["Latitude"] < (port.latitude + port.inner_square_lat_radius)))

        df_outside_square: pd.DataFrame = df.loc[~lat_mask]
        df_inside_square: pd.DataFrame = df.loc[lat_mask]

        long_mask = ((df_inside_square["Longitude"] > (port.longitude - port.inner_square_long_radius)) &
                     (df_inside_square["Longitude"] < (port.longitude + port.inner_square_long_radius)))

        # df_outside_square = df_outside_square.loc[long_mask]
        # df_inside_square = df_inside_square[~long_mask]
        df_outside_square.append(df_inside_square.loc[~long_mask])
        df_inside_square = df_inside_square.loc[long_mask]

        print("lat interval [{}; {}] ".format(str(port.latitude - port.inner_square_lat_radius),
                                              str(port.latitude + port.inner_square_lat_radius)))
        print("long interval [{}; {}] ".format(str(port.longitude - port.inner_square_long_radius),
                                               str(port.longitude + port.inner_square_long_radius)))
        print("df outside square: \n", df_outside_square)
        print("df inside square: \n", df_inside_square)

        # accurate separation outside of inner square but within port's radius
        radius_mask = df_outside_square.apply(radius_filter, args=(port,), axis=1)

        df_outside_circle: pd.DataFrame = df_outside_square[radius_mask]  # training data series
        df_inside_circle: pd.DataFrame = df_outside_square[~radius_mask]

        # minimum timestamp of inside port area data-points is output label
        min_square: pd.DataFrame = get_minimum_row(df_inside_square, "time")
        min_circle: pd.DataFrame = get_minimum_row(df_inside_circle, "time")

        print("min square: ", min_square["time"])
        print("min circle: ", min_circle["time"])

        port_label = pd.DataFrame()
        if not is_empty(min_square):
            if not is_empty(min_circle):
                port_label = min_square if min_square["time"].iloc[0] < min_circle["time"].iloc[0] else min_circle
            else:
                port_label = min_square

        df_train = df_outside_circle.append(port_label)

        return df_train, port_label


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


def get_minimum_row(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    result_df = pd.DataFrame()
    if not is_empty(df):
        if column_name not in df:
            raise ValueError("Column {} not contained in dataframe. Got {}.".format(column_name, str(df.columns)))
        result_df = df[df[column_name] == df[column_name].min()]
    return result_df

