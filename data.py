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

# from port import generate as generate_ports, load as load_ports, identify_label, find_match as find_port_match
from port import PortManager
from util import get_destination_file_name, write_to_console

script_dir = os.path.abspath(os.path.dirname(__file__))
data_folders = ("encode", "test", "train")

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


# normalize numeric data on interval [-1,1]
def normalize(data: np.ndarray, scaler: MinMaxScaler = None) -> Tuple[np.ndarray, MinMaxScaler]:
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data)
    normalized_data = scaler.transform(data)
    return normalized_data, scaler


# create one-hot encoded vector for arbitrary categorical data
def one_hot_encode(data: pd.Series) -> Tuple[List[List[int]], OneHotEncoder]:
    encoder = OneHotEncoder()
    encoder.fit(data)
    encoded_data = encoder.transform(data)
    return encoded_data, encoder


def denormalize(data: List[List[List[float]]], scaler: MinMaxScaler) -> List[List[List[float]]]:
    denormalized_data = scaler.inverse_transform(data)
    return denormalized_data


def format_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    df.assign(time=pd.to_datetime(df.pop("Timestamp"), format='%d/%m/%Y %H:%M:%S')
              .map(datetime.datetime.timestamp))
    return df


def generate_dataset(input_dir: str, output_dir: str) -> None:
    print("Generating dataset from {} ...".format(input_dir))
    min_number_of_rows = 10000
    numerical_features = ["time", "Latitude", "Longitude", "SOG", "COG", "Heading", "Width", "Length", "Draught"]
    categorical_features = ["Ship type", "Navigational status"]

    # initialize port manager
    pm = PortManager()
    pm.load()

    if len(pm.ports.keys()) < 1:
        raise ValueError("No port data available")

    df = pd.read_csv(input_dir, ",", None)

    # drop undesired columns
    df.drop(columns=['Type of mobile', 'ROT', 'Type of position fixing deice', 'ETA',
                     'Data source type', 'A', 'B', 'C', 'D'], inplace=True)

    # group rows by ship type
    # df = df.loc[df["Ship type"] == ship_type]

    # assert if enough data remains
    if len(df) < min_number_of_rows:
        raise ValueError("Required {} rows of data, got {}.".format(str(min_number_of_rows), len(df)))

    initialize(output_dir)

    """
    Find unique routes of a ship to a destination from data pool
    1) Group by destination
    2) Group by ship (MMSI)
    """
    destinations = df["Destination"].unique()

    for dest_column_header in destinations:
        if pd.isnull(dest_column_header):
            continue

        dest_name = get_destination_file_name(dest_column_header)
        port = pm.find_port(dest_name)
        print("Port match: {}".format(port.name))

        # skip if no port data is set
        if port is None:
            continue

        for folder in data_folders:
            if not os.path.exists(os.path.join(output_dir, folder, port.name)):
                os.makedirs(os.path.join(output_dir, folder, port.name))

        dest_df = df.loc[df["Destination"] == dest_column_header]

        # filter data-points that are sent while sitting in port and compute label
        x_df, y_df = pm.identify_label(dest_column_header, dest_df)

        x_df = format_timestamp_col(x_df)

        # handle categorical data
        x_categorical_df = pd.DataFrame()
        x_categorical_df["MMSI"] = x_df["MMSI"]
        x_categorical_df["Ship type"], ship_type_encoder = one_hot_encode(x_df.pop("Ship type"))
        x_categorical_df["Navigational status"], nav_status_encoder = one_hot_encode(x_df.pop("Navigational status"))
        ships = x_df["MMSI"].unique()
        x_data = np.array([])
        labels = np.array([])

        for ship in ships:
            # TODO: Handle ships that head to the same port more than once within the dataset
            ship_df = x_df.loc[x_df["MMSI"] == ship]
            ship_df.drop(columns=["MMSI"], inplace=True)
            ship_df_typed = ship_df.astype("float64")
            ship_categorical_df = x_categorical_df.loc[x_categorical_df["MMSI"] == ship]
            ship_categorical_df.drop(columns=["MMSI"], inplace=True)

            # data = []
            # label = []
            # for num_feat in numerical_features:
            #    np.append(data, ship_df_typed.pop(num_feat), axis=1)

            # for cat_feat in categorical_features:
            #    np.append(data, ship_categorical_df.pop(cat_feat), axis=1)

            # np.append(label, y_df.pop("time"), axis=1)

            data = ship_df_typed.to_numpy()
            label = ship_categorical_df.to_numpy()

            # x_data.append(data)
            # labels.append(label)

            np.append(x_data, data, axis=0)
            np.append(labels, label, axis=0)

        # TODO: Take care if label is allowed to be in normalized data set (currently it is the last row)
        x_normalized, scaler = normalize(x_data)
        labels_normalized, _ = normalize(labels, scaler)

        shape = x_normalized.shape()

        # create train and test data
        x_train, x_test = np.split(x_normalized, [int(.80*len(x_normalized))])
        labels_train, labels_test = np.split(labels_normalized, [int(.80*len(labels_normalized))])

        # np.save(os.path.join(output_dir, "train", dest_name, "columns.npy"), columns)
        np.save(os.path.join(output_dir, "train", dest_name, "data.npy"), x_train)
        np.save(os.path.join(output_dir, "train", dest_name, "labels.npy"), labels_train)

        np.save(os.path.join(output_dir, "test", dest_name, "data.npy"), x_test)
        np.save(os.path.join(output_dir, "test", dest_name, "labels.npy"), labels_test)

        joblib.dump(scaler, os.path.join(output_dir, "encode", dest_name, "scaler.pkl"))
        joblib.dump(ship_type_encoder, os.path.join(output_dir, "encode", dest_name, "ship_type_encoder.pkl"))
        joblib.dump(nav_status_encoder, os.path.join(output_dir, "encode", dest_name, "nav_status_encoder.pkl"))

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
    elif args.command == "add_port":
        # add(args.name, args.latitude, args.longitude, args.radius)
        print("add_port command: TODO")
    else:
        raise ValueError("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("command", choices=["init", "generate", "load", "cc", "add_port", "check_ports"])
    parser.add_argument("--port", type=str, default="COPENGAHEN", help="Name of port to load dataset")
    parser.add_argument("--window_width", type=int, default=20, help="Sliding window width of training examples")
    parser.add_argument("--input_dir", type=str, default=os.path.join(script_dir, "data", "raw", "small.csv"),
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
