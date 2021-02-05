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

from util import get_destination_file_name, write_to_console

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


def generate_sequences(data, sequence_len):
    input_seqs, labels = [], []

    # create sequences where, starting from sequence_len'th index, each data-point represents one output
    for i in range(len(data)):

        end_index = i + sequence_len

        # abort if sequence exceeds data
        if end_index > len(data)-1:
            break

        input_seqs.append(data[i:end_index])
        labels.append(data[end_index])

    # expand labels to behave as a matrix -> vector of outputs as column
    return np.array(input_seqs), np.expand_dims(np.array(labels), axis=-1)


# normalize numeric data on interval [-1,1]
def normalize(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data)
    normalized_data = scaler.transform(data)
    return normalized_data, scaler


# create one-hot encoded vector for arbitrary categorical data
def one_hot_encode(data):
    encoder = OneHotEncoder()
    encoder.fit(data)
    encoded_data = encoder.transform(data)
    return encoded_data, encoder


def denormalize(data, scaler):
    denormalized_data = scaler.inverse_transform(data)
    return denormalized_data


def generate_dataset(input_dir, sequence_len, output_dir, ship_type):
    min_number_of_rows = 10000

    print("Generating dataset from {} ...".format(input_dir))

    df = pd.read_csv(input_dir, ",", None)

    # drop undesired columns
    df.drop(columns=['Type of mobile', 'ROT', 'Type of position fixing device', 'ETA',
                     'Data source type', 'A', 'B', 'C', 'D'], inplace=True)

    # group rows by ship type
    # df = df.loc[df["Ship type"] == ship_type]

    # assert if enough data remains
    if len(df) < min_number_of_rows:
        raise ValueError("Required {} rows of data, got {} for ship of type {}."
                         .format(str(min_number_of_rows), str(len(df)), ship_type))

    # group by destination
    destinations = df["Destination"].unique()

    for dest in destinations:
        dest_df = df.loc[df["Destination"] == dest]

        """Group by ship to extract continuous time series data-points
        One ship on its way to a destination port represents a single time-series
        All of those series per ship can be synchronous = data-points at the same time
        """
        ships = df["MMSI"].unique()
        dest_df.assign(timestamp_formatted=pd.to_datetime(dest_df.pop("Timestamp"), format='%d/%m/%Y %H:%M:%S')
                       .map(datetime.datetime.timestamp))

        normalized_dest_df, scaler = normalize(dest_df)

        input_series = []
        output = []

        for ship in ships:
            # TODO: Handle ships that head to the same port more than once within the dataset
            ship_df = normalized_dest_df.loc[dest_df["MMSI"] == ship]

            # numerical
            timestamp = ship_df.pop("timestamp_formatted")
            latitude = ship_df.pop("Latitude")
            longitude = ship_df.pop("Longitude")
            sog = ship_df.pop("SOG")
            cog = ship_df.pop("COG")
            heading = ship_df.pop("Heading")
            # IMO, Callsign, Name, Ship type, Cargo type
            width = ship_df.pop("Width")
            length = ship_df.pop("Length")
            draught = ship_df.pop("Draught")
            # Destination

            # categorical
            # ship_type_one_hot_encoded, ship_type_encoder = one_hot_encode(ship_df.pop["Ship type"])
            # nav_status_one_hot_encoded, nav_status_encoder = one_hot_encode(ship_df.pop["Navigational status"])

            data = np.array([timestamp, latitude, longitude, sog, cog, heading, width, length, draught])
            input_data, output_data = generate_sequences(data, sequence_len)

            input_series.append(input_data)
            output.append(output_data)

        dest_name = get_destination_file_name(dest)

        np.save(os.path.join(output_dir, dest_name, "input_series.npy"), input_series)
        np.save(os.path.join(output_dir, dest_name, "output.npy"), output)

        joblib.dump(scaler, os.path.join(output_dir, dest_name, "scaler.pkl"))

    print("Done.")


def load_dataset(destination_name):
    write_to_console("Loading data for {}".format(destination_name))

    dest_dir = get_destination_file_name(destination_name)

    data = np.load(os.path.join(script_dir, "data", "{}", "data.npy".format(dest_dir)))

    scaler = joblib.load(os.path.join(script_dir, "data", "{]", "scalers.pkl"))

    return data, scaler


def main(args):
    write_to_console("Pro-processing data")

    if args.command == "load":
        load_dataset("COPENHAGEN")
    elif args.command == "generate":
        generate_dataset(args.input_dir, args.output_dir)
    else:
        raise ValueError("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["load", "generate", "cc"])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--ship_type", type=str, default="Cargo",
                        choices=["Cargo", "Fishing", "Passenger", "Military", "Tanker"])
    parser.add_argument("--input_dir", type=str, default=os.path.join(script_dir, "data", "csv", "small.csv"))
    # parser.add_argument("--input_dir", type=str, default=os.path.join(script_dir, "data", "csv" "aisdk_20181101.csv"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "data", "destination"))
    main(parser.parse_args())
