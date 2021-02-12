import argparse
import os
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
# import random
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from typing import List, Tuple

from util import get_destination_file_name, write_to_console, generate_label

script_dir = os.path.abspath(os.path.dirname(__file__))


def correlate(raw_data_frame):
    print("Computing correlations ...")

    correlations = raw_data_frame.corr()
    print(correlations)

    np.save(os.path.join(script_dir, "data", "correlations.npy"), correlations)


def compute_missing_entries(df):
    # find entry where vessel information is not empty
    unique_mmsi_df = df['MMSI'].unique()

    for mmsi in unique_mmsi_df:
        df_single_mmsi = df['MMSI'] == mmsi


def make_numpy_dataset(df):
    features_float32 = ['Latitude', 'Longitude', 'SOG', 'COG', 'Draught']
    features_uint16 = ['Heading', 'Width', 'Length']
    print("features_float34: {}".format(features_float32))
    print("features_uint16: {}".format(features_uint16))

    data = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(df[features_float32].values, tf.float32),
            tf.cast(df[features_uint16].values, tf.uint16)
        )
    )

    features = features_float32.extend(features_uint16)
    meta_df = df.drop(columns=features, inplace=True)

    return data, features_float32, features_uint16, meta_df


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
def normalize(data: List[List[List[int]]], scaler: MinMaxScaler = None) -> Tuple[List[List[List[int]]], MinMaxScaler]:
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


def denormalize(data, scaler: MinMaxScaler):
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
    df = pd.read_csv(input_dir, ",", None)

    # drop undesired columns
    df.drop(columns=['Type of mobile', 'ROT', 'Type of position fixing device', 'ETA',
                     'Data source type', 'A', 'B', 'C', 'D'], inplace=True)

    # group rows by ship type
    # df = df.loc[df["Ship type"] == ship_type]

    # assert if enough data remains
    if len(df) < min_number_of_rows:
        raise ValueError("Required {} rows of data, got {}.".format(str(min_number_of_rows), len(df)))

    """
    Find unique routes of a ship to a destination from data pool
    1) Group by destination
    2) Group by ship (MMSI)
    """
    destinations = df["Destination"].unique()

    for dest in destinations:
        dest_df = df.loc[df["Destination"] == dest]

        # filter data-points that are sent while sitting in port and compute label
        x_df, y_df = generate_label(dest, dest_df)

        x_df = format_timestamp_col(x_df)

        # handle categorical data
        x_categorical_df = pd.DataFrame()
        x_categorical_df["MMSI"] = x_df["MMSI"]
        x_categorical_df["Ship type"], ship_type_encoder = one_hot_encode(x_df.pop("Ship type"))
        x_categorical_df["Navigational status"], nav_status_encoder = one_hot_encode(x_df.pop("Navigational status"))
        ships = x_df["MMSI"].unique()
        x_data = []
        labels = []

        for ship in ships:
            # TODO: Handle ships that head to the same port more than once within the dataset
            ship_df = x_df.loc[x_df["MMSI"] == ship]
            ship_categorical_df = x_categorical_df.loc[x_categorical_df["MMSI"] == ship]

            data = []
            label = []
            for num_feat in numerical_features:
                np.append(data, ship_df.pop(num_feat), axis=1)

            for cat_feat in categorical_features:
                np.append(data, ship_categorical_df.pop(cat_feat), axis=1)

            x_data.append(data)
            np.append(label, y_df.pop("time"), axis=1)
            labels.append(label)

        # TODO: Take care if label is allowed to be in normalized data set (currently it is the last row)
        x_normalized, scaler = normalize(x_data)
        labels_normalized = normalize(labels, scaler)

        dest_name = get_destination_file_name(dest)

        np.save(os.path.join(output_dir, dest_name, "data.npy"), x_normalized)
        np.save(os.path.join(output_dir, dest_name, "labels.npy"), labels_normalized)

        joblib.dump(scaler, os.path.join(output_dir, dest_name, "scaler.pkl"))
        joblib.dump(ship_type_encoder, os.path.join(output_dir, dest_name, "ship_type_encoder.pkl"))
        joblib.dump(nav_status_encoder, os.path.join(output_dir, dest_name, "nav_status_encoder.pkl"))

    print("Done.")


def load_dataset(destination_name, window_length):
    write_to_console("Loading data for {} with windows of {}"
                     .format(destination_name, window_length))

    dest_dir = get_destination_file_name(destination_name)

    data = np.load(os.path.join(script_dir, "data", "{}", "data.npy".format(dest_dir)))

    scaler = joblib.load(os.path.join(script_dir, "data", "{]", "scalers.pkl"))

    return data, scaler


def main(args):
    write_to_console("Pre-processing data")

    if args.command == "load":
        load_dataset("COPENHAGEN", args.window_length)
    elif args.command == "generate":
        generate_dataset(args.input_dir, args.output_dir)
    else:
        raise ValueError("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "load", "cc"])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--ship_type", type=str, default="Cargo",
                        choices=["Cargo", "Fishing", "Passenger", "Military", "Tanker"])
    parser.add_argument("--input_dir", type=str, default=os.path.join(script_dir, "data", "csv", "small.csv"))
    # parser.add_argument("--input_dir", type=str, default=os.path.join(script_dir, "data", "csv" "aisdk_20181101.csv"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "data", "destination"))
    parser.add_argument("--window_length", type=int, default=20)
    main(parser.parse_args())
