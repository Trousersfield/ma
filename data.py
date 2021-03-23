import argparse
import os
import pandas as pd
import numpy as np
import joblib
import re
# import random
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from typing import List, Tuple

from logger import Logger
from port import PortManager
from labeler import DurationLabeler
from util import get_destination_file_name, is_empty, data_file, obj_file, write_to_console

script_dir = os.path.abspath(os.path.dirname(__file__))
data_folders = ("encode", "test", "train", "unlabeled")
LAT = {"min": -90, "max": 90}
LONG = {"min": -180, "max": 180}


def initialize(output_dir:  str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for folder in data_folders:
        if not os.path.exists(os.path.join(output_dir, folder)):
            os.makedirs(os.path.join(output_dir, folder))

    if not os.path.exists(os.path.join(output_dir, "raw")):
        os.makedirs(os.path.join(output_dir, "raw"))


def correlate(raw_data_frame):
    print("Computing correlations ...")

    correlations = raw_data_frame.corr()
    print(correlations)

    np.save(os.path.join(script_dir, "data", "correlations.npy"), correlations)


def generate_label(df: pd.DataFrame, arrival_time: int) -> Tuple[pd.DataFrame, DurationLabeler]:
    labeler = DurationLabeler()
    labeled_df = labeler.fit_transform(df, arrival_time)
    return labeled_df, labeler


def descale_from_label(df: pd.DataFrame, scaler: DurationLabeler) -> pd.DataFrame:
    return scaler.inverse_transform(df)


def normalize(data: np.ndarray, scaler: MinMaxScaler = None) -> Tuple[np.ndarray, MinMaxScaler]:
    if scaler is None:
        # copy = False, if input already is numpy array
        scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
        scaler.fit(data)
    normalized_data = scaler.transform(data)
    return normalized_data, scaler


def denormalize(data: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    return scaler.inverse_transform(data)


# create one-hot encoded DataFrame: one column for each category
def one_hot_encode(data: pd.Series) -> Tuple[pd.DataFrame, OneHotEncoder]:
    # print("one hot encoding. number of data-points: ", len(data))
    categories = data.unique()
    nan_idx = -1
    # prevent typing error by replacing NaN with custom decorator "nan_type"
    for idx, cat in enumerate(categories):
        if type(cat) is float:
            nan_idx = idx

    if nan_idx > -1:
        categories[nan_idx] = "nan_entry"

    data = data.values.reshape(-1, 1)   # shape as column
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(data)
    data_ohe = encoder.transform(data)
    df_ohe = pd.DataFrame(data_ohe, columns=[categories[i] for i in range(len(categories))])

    return df_ohe, encoder


def format_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(time=pd.to_datetime(df["# Timestamp"], format='%d/%m/%Y %H:%M:%S').values.astype(np.int64) // 10**9)
    df = df.drop(columns=["# Timestamp"])
    return df


def generate_dataset(input_dir: str, output_dir: str) -> None:
    print("Generating dataset from {} ...".format(input_dir))
    min_number_of_rows = 10000
    numerical_features = ["time", "Latitude", "Longitude", "SOG", "COG", "Heading", "Width", "Length", "Draught"]
    categorical_features = ["Ship type", "Navigational status"]

    logger = Logger()

    # initialize port manager
    pm = PortManager()
    pm.load()

    if len(pm.ports.keys()) < 1:
        raise ValueError("No port data available")

    df = pd.read_csv(input_dir, ",", None)

    # drop undesired columns
    df = df.drop(columns=["Type of mobile", "ROT", "IMO", "Callsign", "Name", "Cargo type",
                          "Type of position fixing device", "ETA", "Data source type", "A", "B", "C", "D"])

    # filter out of range values
    df = df.loc[(df["Latitude"] >= LAT["min"]) & (df["Latitude"] <= LAT["max"])]
    df = df.loc[(df["Longitude"] >= LONG["min"]) & (df["Longitude"] <= LONG["max"])]

    # assert if enough data remains
    if len(df.index) < min_number_of_rows:
        logger.write("Required {} rows of data, got {}".format(str(min_number_of_rows), len(df.index)))

    initialize(output_dir)

    """
    Find unique routes of a ship to a destination from data pool
    1) Group by destination
    2) Group by ship (MMSI)
    """
    destinations: List[str] = df["Destination"].unique()

    for dest_column_header in destinations:
        if pd.isnull(dest_column_header):
            continue

        dest_name: str = get_destination_file_name(dest_column_header)
        port = pm.find_port(dest_name)

        # skip if no port data is set
        if port is None:
            continue
        print("\n\nPort match: {}".format(port.name))

        for folder in data_folders:
            if not os.path.exists(os.path.join(output_dir, folder, port.name)):
                os.makedirs(os.path.join(output_dir, folder, port.name))

        dest_df = df.loc[df["Destination"] == dest_column_header]

        dest_df = dest_df.drop(columns=["Destination"])
        dest_df = format_timestamp_col(dest_df)

        # extract data-points that are sent while sitting in port to compute label
        x_df, arrival_times_df = pm.identify_arrival_times(port, dest_df)

        # skip port if ship is hanging out in port area only
        if is_empty(x_df):
            print("Data for port {} only within port area. {} data, {} labels".format(port.name, len(x_df.index),
                                                                                      len(arrival_times_df.index)))
            logger.write("No Data for port {} outside of port area. {} data, {} labels".format(port.name,
                                                                                               len(x_df.index),
                                                                                               len(arrival_times_df.index)))
            continue

        # handle categorical data
        x_ship_types, ship_type_encoder = one_hot_encode(x_df.pop("Ship type"))
        x_nav_states, nav_status_encoder = one_hot_encode(x_df.pop("Navigational status"))
        # x_cargo_types, cargo_types_encoder = one_hot_encode(x_df.pop("Cargo type"))

        # arrival_times_df = arrival_times_df.drop(columns=["Ship type", "Navigational status", "Cargo type"])
        arrival_times_df = arrival_times_df.drop(columns=["Ship type", "Navigational status"])
        # print("all arrival times: \n", arrival_times_df)

        mmsi_col = x_df["MMSI"]
        # Add MMSI identification to categorical features
        # x_ship_types["MMSI"], x_nav_states["MMSI"], x_cargo_types["MMSI"] = mmsi_col, mmsi_col, mmsi_col
        x_ship_types["MMSI"], x_nav_states["MMSI"] = mmsi_col, mmsi_col

        mmsis = x_df["MMSI"].unique()

        for idx, mmsi in enumerate(mmsis):
            # TODO: Handle ships that head to the same port more than once within the dataset
            ship_df = x_df.loc[x_df["MMSI"] == mmsi]
            arrival_time_df = arrival_times_df.loc[arrival_times_df["MMSI"] == mmsi]
            # print("arrival_time_df for mmsi: \n", arrival_time_df)
            arrival_time = -1
            if not is_empty(arrival_time_df):
                arrival_time = arrival_time_df.iloc[0]["time"]

                # drop rows sent after ship left the port
                ship_df = ship_df[ship_df["time"] <= arrival_time]

                if not is_empty(ship_df):
                    print("- - - - - - - - - - - - - - - - - - - - - - -")
                    print("- - T A R G E T  &  D A T A   F O U N D - - -")
                    print("- - - - - - - - - - - - - - - - - - - - - - -")

            if is_empty(ship_df):
                print(f"No data for MMSI {mmsi}")
                continue

            print("arrival time for mmsi {}: {}".format(mmsi, arrival_time))
            # TODO: Consider not dropping MMSI. But keep in mind: MMSI number has no natural order but is float
            ship_df = ship_df.drop(columns=["MMSI"])
            # print("data: \n", ship_df)

            # ship_categorical_df = x_categorical_df.loc[x_categorical_df["MMSI"] == mmsi]
            # ship_categorical_df = ship_categorical_df.drop(columns=["MMSI"])

            if arrival_time == -1:
                print(f"No label found for MMSI {mmsi}. Saving as unlabeled.")
                ship_df.to_pickle(os.path.join(output_dir, "unlabeled", port.name, obj_file("data_unlabeled", mmsi)))
                continue

            ship_df, labeler = generate_label(ship_df, arrival_time)
            print("df with added label: \n", ship_df)

            data = ship_df.to_numpy()
            print("Shape of data for MMSI {}: {} Type: {}".format(mmsi, data.shape, data.dtype))
            # print("data: ", data)
            # label = ship_categorical_df.to_numpy()

            train, test = split(data)

            # TODO: Check if label needs to be added after normalization
            # Intuition: MSE loss is huge if target has large values like duration in seconds
            train_normalized, train_scaler = normalize(train)
            test_normalized, test_scaler = normalize(test)
            print(f"normalized train shape: {train_normalized.shape}")
            print(f"normalized test shape: {test_normalized.shape}")

            np.save(os.path.join(output_dir, "train", port.name, data_file(mmsi)), train)
            np.save(os.path.join(output_dir, "test", port.name, data_file(mmsi)), test)

            joblib.dump(train_scaler, os.path.join(output_dir, "encode", port.name, obj_file("train_scaler", mmsi)))
            joblib.dump(test_scaler, os.path.join(output_dir, "encode", port.name, obj_file("test_scaler", mmsi)))
            joblib.dump(labeler, os.path.join(output_dir, "encode", port.name, obj_file("labeler", mmsi)))
            joblib.dump(ship_type_encoder, os.path.join(output_dir, "encode", port.name, obj_file("ship_type", mmsi)))
            joblib.dump(nav_status_encoder, os.path.join(output_dir, "encode", port.name, obj_file("nav_status", mmsi)))

    print("Done.")


def split(data: np.ndarray, train_ratio: float = .9) -> Tuple[np.ndarray, np.ndarray]:
    if 0 < train_ratio < 1:
        return np.split(data, [int(train_ratio*data.shape[0])])
    else:
        print("Unable to split by {}. Use a value within (0, 1)".format(train_ratio))
        return data, np.array([])


def main(args) -> None:
    if args.command == "init":
        write_to_console("Initializing repo")
        initialize(args.output_dir)
    elif args.command == "generate":
        write_to_console("Generating data")
        generate_dataset(args.input_dir, args.output_dir)
    elif args.command == "check_ports":
        pm = PortManager()
        pm.generate_from_source(load=True)
    elif args.command == "add_alias":
        pm = PortManager()
        pm.load()
        pm.add_alias("PETERSBURG", "ST.PETERSBURG")
        pm.add_alias("THYBORON", "THYBOROEN")
        pm.add_alias("ANTWERPEN", "ANTWERP")
        pm.add_alias("GRENAA", "GRENA")
        pm.add_alias("GOTEBORG", "GOTHENBURG")
    else:
        raise ValueError("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    small = "small.csv"
    medium = "medium.csv"
    big = "aisdk_20181101.csv"
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("command", choices=["init", "generate", "cc", "add_alias", "check_ports"])
    parser.add_argument("--port", type=str, default="COPENGAHEN", help="Name of port to load dataset")
    parser.add_argument("--window_width", type=int, default=20, help="Sliding window width of training examples")
    parser.add_argument("--input_dir", type=str, default=os.path.join(script_dir, "data", "raw", big),
                        help="Path to AIS .csv file")
    # parser.add_argument("--input_dir", type=str, default=os.path.join(script_dir, "data", "raw" "aisdk_20181101.csv"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "data"),
                        help="Output directory path")
    parser.add_argument("--ship_type", type=str, default="Cargo", choices=["Cargo", "Fishing", "Passenger", "Military",
                                                                           "Tanker"])
    # parser.add_argument("--name", type=str, required=True)
    # parser.add_argument("--latitude", type=float, required=True)
    # parser.add_argument("--longitude", type=float, required=True)
    # parser.add_argument("--radius", type=float, default=1.0)
    main(parser.parse_args())
