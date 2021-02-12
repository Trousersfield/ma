import argparse
import joblib
import os
import pandas as pd
import re

from math import radians, cos, sin, asin, sqrt
from typing import Dict, Tuple


script_dir = os.path.abspath(os.path.dirname(__file__))


class Port:
    def __init__(self, name: str, latitude: float, longitude: float, radius: float) -> None:
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.radius = radius
        # approximate maximum inner square
        max_corner_dist = haversine(latitude, longitude, latitude+radius, longitude+radius)
        self.inner_square_distance = sqrt((max_corner_dist-radius)**2 / 2)


def get_destination_file_name(destination_name):
    dest = str(destination_name)
    return re.sub(r'\W', '', dest).upper()


def write_to_console(message):
    filler = "*" * len(message)
    print("\t********** {} ***********".format(filler))
    print("\t********** {} ***********".format(message))
    print("\t********** {} ***********".format(filler))


def is_empty(df: pd.DataFrame) -> bool:
    return len(df.index) == 0


def get_minimum_row(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    result_df = pd.DataFrame()
    if not is_empty(df):
        result_df = df[df[column_name] == df[column_name].min()]
    return result_df


def port_radius_filter(row, port: Port) -> bool:
    if haversine(row["Latitude"], row["Longitude"], port.latitude, port.longitude) < port.radius:
        return False
    else:
        return True


def haversine(lat1: float, long1: float, lat2: float, long2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])

    # formula itself
    d_long = long2 - long1
    d_lat = lat2 - lat1
    a = sin(d_lat/2)**2 + cos(lat1) * cos(lat2) * sin(d_long/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371    # radius of the earth in km
    return c * r


def generate_label(port_name: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ports = load_ports()
    port = ports.get(port_name)

    if not port:
        raise ValueError("No information for port {} available".format(port_name))

    # separation by inner square of port area
    inner_square_mask = (((df["Latitude"] > (port.latitude - port.inner_square_distance)) &
                          (df["Latitude"] < (port.latitude + port.inner_square_distance))) &
                         ((df["Longitude"] > (port.longitude - port.inner_square_distance)) &
                          (df["Longitude"] < (port.longitude + port.inner_square_distance))))

    df_outside_square: pd.DataFrame = df[inner_square_mask]
    df_inside_square: pd.DataFrame = df[~inner_square_mask]

    # accurate separation outside of inner square but within port's radius
    radius_mask = df_outside_square.apply(port_radius_filter, args=(port,), axis=1)

    df_outside_circle: pd.DataFrame = df_outside_square[radius_mask]  # training data series
    df_inside_circle: pd.DataFrame = df_outside_square[~radius_mask]

    # minimum timestamp of inside port area data-points is output label
    min_square: pd.DataFrame = get_minimum_row(df_inside_square, "time")
    min_circle: pd.DataFrame = get_minimum_row(df_inside_circle, "time")

    port_label = pd.DataFrame()
    if not is_empty(min_square):
        if not is_empty(min_circle):
            port_label = min_square if min_square["time"] < min_circle["time"] else min_circle
        else:
            port_label = min_square

    df_train = df_outside_circle.append(port_label)

    return df_train, port_label


def load_ports() -> Dict[str, Port]:
    port_path = os.path.join(script_dir, "data", "port", "ports.pkl")
    ports = {}

    if os.path.exists(port_path):
        ports = joblib.load(port_path)

    return ports


def make_port(name: str, latitude: float, longitude: float, radius: float) -> None:
    ports = load_ports()

    # (over)write port information
    ports[name] = Port(name, latitude, longitude, radius)

    joblib.dump(ports, os.path.join(script_dir, "data", "port", "ports.pkl"))


def print_port(port="all"):

    return port


def main(args):
    if args.command == "set_port":
        make_port(args.name, args.latitude, args.longitude, args.radius)
    elif args.command == "print_all":
        print_port()
    else:
        raise ValueError("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["set_port"])
    parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument("--name", type=str, required=True)
    required.add_argument("--latitude", type=float, required=True)
    required.add_argument("--longitude", type=float, required=True)
    optional.add_argument("--radius", type=float, default=1.0)
    main(parser.parse_args())
