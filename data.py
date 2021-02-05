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


def generate_sequences(series, sequence_len):
    sequences = []

    x = np.arange(sequence_len)

    return np.array()


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

    # df = pd.read_csv(raw_data_path, ",", None)
    # load raw AIS data from .csv
    df = pd.read_csv(input_dir, ",", None)
    print("Csv read. Number of lines: {}".format(len(df.index)))

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
        All of those series are generally parallel = at the same time (multiple parallel series)
        """
        ships = df["MMSI"].unique()
        data = []
        scalers = []

        for ship in ships:
            ship_df = dest_df.loc[dest_df["MMSI"] == ship]

            # numerical
            timestamp = pd.to_datetime(dest_df.pop("Timestamp"), format='%d/%m/%Y %H:%M:%S')\
                .map(datetime.datetime.timestamp)
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

            features = np.array([timestamp, latitude, longitude, sog, cog, heading, width, length, draught])

            features_normalized, scaler = normalize(features)

            series = generate_sequences(features_normalized)

            data.append(series)
            scalers.append(scaler)

        dest_name = get_destination_file_name(dest)
        # np.save(os.path.join(output_dir, "{}.npy".format(file_name)), dest_df)
        np.save(os.path.join(output_dir, dest_name, "data.npy"), data)

        joblib.dump(timestamp_scaler, os.path.join(output_dir, dest_name, "timestamp_scaler.pkl"))
    # np.save(os.path.join(output_dir, "destination_mapping.npy"), file_names)

    print("Done.")


def load_dataset(destination_name):
    write_to_console("Loading data for {}".format(destination_name))

    dest_dir = get_destination_file_name(destination_name)

    timestamp_normalized = np.load(os.path.join(script_dir, "data", "{}", "timestamp.npy".format(dest_dir)))
    latitude_normalized = np.load(os.path.join(script_dir, "data", "{}", "latitude.npy".format(dest_dir)))
    longitude_normalized = np.load(os.path.join(script_dir, "data", "{}", "longitude.npy".format(dest_dir)))
    sog_normalized = np.load(os.path.join(script_dir, "data", "{}", "sog.npy".format(dest_dir)))
    cog_normalized = np.load(os.path.join(script_dir, "data", "{}", "cog.npy".format(dest_dir)))
    heading_normalized = np.load(os.path.join(script_dir, "data", "{}", "heading.npy".format(dest_dir)))
    width_normalized = np.load(os.path.join(script_dir, "data", "{}", "width.npy".format(dest_dir)))
    length_normalized = np.load(os.path.join(script_dir, "data", "{}", "length.npy".format(dest_dir)))
    draught_normalized = np.load(os.path.join(script_dir, "data", "{}", "draught.npy".format(dest_dir)))

    timestamp_scaler = joblib.load(os.path.join(script_dir, "data", "{]", "timestamp_scaler.pkl"))
    latitude_scaler = joblib.load(os.path.join(script_dir, "data", "{]", "latitude_scaler.pkl"))
    longitude_scaler = joblib.load(os.path.join(script_dir, "data", "{]", "longitude_scaler.pkl"))
    sog_scaler = joblib.load(os.path.join(script_dir, "data", "{]", "sog_scaler.pkl"))
    cog_scaler = joblib.load(os.path.join(script_dir, "data", "{]", "cog_scaler.pkl"))
    heading_scaler = joblib.load(os.path.join(script_dir, "data", "{]", "heading_scaler.pkl"))
    width_scaler = joblib.load(os.path.join(script_dir, "data", "{]", "width_scaler.pkl"))
    length_scaler = joblib.load(os.path.join(script_dir, "data", "{]", "length_scaler.pkl"))
    draught_scaler = joblib.load(os.path.join(script_dir, "data", "{]", "draught_scaler.pkl"))

    return timestamp_normalized, latitude_normalized, longitude_normalized, sog_normalized, cog_normalized,\
        heading_normalized, width_normalized, length_normalized, draught_normalized, timestamp_scaler,\
        latitude_scaler, longitude_scaler, sog_scaler, cog_scaler, heading_scaler, width_scaler, length_scaler,\
        draught_scaler


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
