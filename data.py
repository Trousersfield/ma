import argparse
import os
import datetime
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
from util import get_destination_file_name, is_empty, write_to_console

script_dir = os.path.abspath(os.path.dirname(__file__))
data_folders = ("encode", "test", "train")
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


def generate_training_examples(df: pd.DataFrame, sequence_len: int):
    input_seqs, output = [], []

    # create sequences where, starting from sequence_len'th index, each data-point represents one output
    for i in range(len(data)):

        end_index = i + sequence_len

        # abort if sequence exceeds data
        if end_index > len(data)-1:
            break

        input_seqs.append(data[i:end_index])
        output.append(data[end_index][0])

    # expand output to behave as a matrix -> one-dim vector as each entry of a column
    return np.array(input_seqs), np.expand_dims(np.array(output), axis=-1)


def normalize(data: np.ndarray, scaler: MinMaxScaler = None) -> Tuple[np.ndarray, MinMaxScaler]:
    if scaler is None:
        # copy = False, if input already is numpy array
        scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
        scaler.fit(data)
    normalized_data = scaler.transform(data)
    return normalized_data, scaler


# create one-hot encoded DataFrame: one column for each category
def one_hot_encode(data: pd.Series) -> Tuple[pd.DataFrame, OneHotEncoder]:
    print("one hot encoding. number of data-points: ", len(data))
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


def denormalize(data: List[List[List[float]]], scaler: MinMaxScaler) -> List[List[List[float]]]:
    denormalized_data = scaler.inverse_transform(data)
    return denormalized_data


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
    df = df.drop(columns=["Type of mobile", "ROT", "Type of position fixing device", "ETA", "Name", "Callsign", "IMO",
                          "Data source type", "A", "B", "C", "D"])

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
        if port.name == "RANDERS":
            print("dest_df: \n", dest_df["Ship type"])
        dest_df = dest_df.drop(columns=["Destination"])
        dest_df = format_timestamp_col(dest_df)

        # extract data-points that are sent while sitting in port to compute label
        x_df, label_df = pm.identify_label(port, dest_df)

        # skip port if ship is hanging out in port area only
        if is_empty(x_df):
            print("Data for port {} only within port area. {} data, {} labels".format(port.name, len(x_df.index),
                                                                                      len(label_df.index)))
            logger.write("No Data for port {} outside of port area. {} data, {} labels".format(port.name,
                                                                                               len(x_df.index),
                                                                                               len(label_df.index)))
            continue

        # handle categorical data
        if port.name == "RANDERS":
            print("x_df: \n", x_df)
        x_ship_types, ship_type_encoder = one_hot_encode(x_df.pop("Ship type"))
        x_nav_states, nav_status_encoder = one_hot_encode(x_df.pop("Navigational status"))
        x_cargo_types, cargo_types_encoder = one_hot_encode(x_df.pop("Cargo type"))

        label_df = label_df.drop(columns=["Ship type", "Navigational status", "Cargo type"])

        mmsi_col = x_df["MMSI"]
        # Add MMSI identification to categorical features
        x_ship_types["MMSI"], x_nav_states["MMSI"], x_cargo_types["MMSI"] = mmsi_col, mmsi_col, mmsi_col

        mmsis = x_df["MMSI"].unique()
        # print("data: \n" , x_df)
        # print("label: \n", label_df)
        # print("data__mmsis: ", mmsis)
        # print("label_mmsis: ", label_df["MMSI"].unique())

        for mmsi in mmsis:
            # TODO: Handle ships that head to the same port more than once within the dataset
            ship_df = x_df.loc[x_df["MMSI"] == mmsi]
            label_ship_df = label_df.loc[label_df["MMSI"] == mmsi]

            # TODO: Consider not dropping MMSI. But keep in mind: MMSI number has no natural order but is float
            ship_df = ship_df.drop(columns=["MMSI"])
            label_ship_df = label_ship_df.drop(columns=["MMSI"])
            # print("data: \n", ship_df)

            # ship_categorical_df = x_categorical_df.loc[x_categorical_df["MMSI"] == mmsi]
            # ship_categorcal_df = ship_categorical_df.drop(columns=["MMSI"])

            data = ship_df.to_numpy()
            print("Shape of data for MMSI {}: {} Type: {}".format(mmsi, data.shape, data.dtype))
            # label = ship_categorical_df.to_numpy()
            labels = label_ship_df.to_numpy()
            print("Shape of label for MMSI {}: {} Type: {}".format(mmsi, labels.shape, labels.dtype))
            if len(labels) > 0:
                print("-----------------")
                print("!!!LABEL FOUND!!!")
                print("-----------------")

            data_normalized, scaler = normalize(data)
            labels_normalized, _ = normalize(labels, scaler) if len(labels) == 1 else [np.array([]), None]

            print("normalized data: \n", data_normalized)

            separate train and test data keeping the order of entries
            split_arg = [int(.80*data_normalized.shape[0])]
            train, test = np.split(data_normalized, split_arg)
            labels_train, labels_test = np.split(labels_normalized, split_arg)

            print("train shape: ", train.shape)
            print("test shape: ", test.shape)
            break

            np.save(os.path.join(output_dir, "train", dest_name, "data.npy"), train)
            np.save(os.path.join(output_dir, "train", dest_name, "labels.npy"), labels_train)

            np.save(os.path.join(output_dir, "test", dest_name, "data.npy"), test)
            np.save(os.path.join(output_dir, "test", dest_name, "labels.npy"), labels_test)

            joblib.dump(scaler, os.path.join(output_dir, "encode", dest_name, "scaler.pkl"))
            joblib.dump(ship_type_encoder, os.path.join(output_dir, "encode", dest_name, "ship_type_encoder.pkl"))
            joblib.dump(nav_status_encoder, os.path.join(output_dir, "encode", dest_name, "nav_status_encoder.pkl"))
            break

    print("Done.")


def load_dataset(output_dir: str, destination_name: str, window_length: int):
    write_to_console("Loading data for {} with windows of {}"
                     .format(destination_name, window_length))

    file_name = get_destination_file_name(destination_name)

    data = np.load(os.path.join(output_dir, "{}", "data.npy".format(file_name)))

    scaler = joblib.load(os.path.join(output_dir, "{}", "scalers.pkl".format(file_name)))

    return data, scaler


def main(args) -> None:
    if args.command == "init":
        write_to_console("Initializing repo")
        initialize(args.output_dir)
    elif args.command == "load":
        write_to_console("Loading data")
        port = args.port.upper()
        load_dataset(args.output_dir, port, args.window_width)
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
    big = "aisdk_20181101.csv"
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("command", choices=["init", "generate", "load", "cc", "add_alias", "check_ports"])
    parser.add_argument("--port", type=str, default="COPENGAHEN", help="Name of port to load dataset")
    parser.add_argument("--window_width", type=int, default=20, help="Sliding window width of training examples")
    parser.add_argument("--input_dir", type=str, default=os.path.join(script_dir, "data", "raw", small),
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
