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

from combiner import RouteCombiner
from labeler import DurationLabeler
from logger import Logger
from port import PortManager
from util import get_destination_file_name, is_empty, data_file, obj_file, write_to_console, mc_to_dk

script_dir = os.path.abspath(os.path.dirname(__file__))
logger = Logger()
data_folders = ("encode", "test", "train", "validate", "unlabeled")
data_ranges = {"Latitude": {"min": -90., "max": 90.},
               "Longitude": {"min": -180., "max": 180.},
               "SOG": {"min": 0., "max": 110},                  # max = 102 from (1)
               "COG": {"min": 0., "max": 359.9},                # (1) max from data: 359.9
               "Heading": {"min": 0., "max": 511.},             # (1)
               "Width": {"min": 0., "max": 80},                 # {3)
               "Length": {"min": 0., "max": 500.},              # (2)
               "time_scaled": {"min": 0., "max": 31622400.},    # max value for seconds per year is dependant on year
               "label": {"min": 0., "max": 31622400.}}          # same range as time within a year
# year with 365 days: 31536000
# year with 366 days: 31622400
# sources:
# (1) https://www.sostechnic.com/epirbs/ais/aisinformationenglish/index.php
# assume the biggest vessel in the world in service (+ some more):
# (2) https://en.wikipedia.org/wiki/List_of_longest_ships
# (3) https://gcaptain.com/emma-maersk-engine/worlds-largest-tanker-knock-nevis/


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


def init_scaler(source_df: pd.DataFrame, target_columns: List[str]) -> MinMaxScaler:
    # use min and max from source data if no definition is available. definitions see above: data_ranges
    source_min_df = source_df.min().to_frame().T
    source_max_df = source_df.max().to_frame().T
    # print(f"columns:\n{target_columns}")
    # print(f"min df:\n{source_min_df}")
    # print(f"max df:\n{source_max_df}")
    target_min_df = pd.DataFrame(index=np.arange(0, 1), columns=target_columns)
    target_max_df = pd.DataFrame(index=np.arange(0, 1), columns=target_columns)

    # add min and max range for certain columns
    for column in target_columns:
        if column in data_ranges:
            # check if real data point is within defined data range. adapt accordingly
            if column in source_min_df and source_min_df.iloc[0][column] < data_ranges[column]["min"]:  # "bad" case 1
                logger.write("Scaler init warning: Defined data range for column {}: [{}, {}], got minimum of {}"
                             .format(column, data_ranges[column]["min"], data_ranges[column]["max"],
                                     source_min_df.iloc[0][column]))
                target_min_df.iloc[0][column] = source_min_df.iloc[0][column]
            else:
                target_min_df.loc[0][column] = data_ranges[column]["min"]

            if column in source_max_df and source_max_df.iloc[0][column] > data_ranges[column]["max"]:  # "bad" case 2
                logger.write("Scaler init warning: Defined data range for column {}: [{}, {}], got maximum of {}"
                             .format(column, data_ranges[column]["min"], data_ranges[column]["max"],
                                     source_max_df.iloc[0][column]))
                target_max_df.iloc[0][column] = source_max_df.iloc[0][column]
            else:
                target_max_df.loc[0][column] = data_ranges[column]["max"]

        elif column in source_min_df:
            target_min_df.loc[0][column] = source_min_df.iloc[0][column]
            target_max_df.loc[0][column] = source_min_df.iloc[0][column]
        else:
            raise ValueError(f"Unknown column {column}! No min and max values available!")

    min_max_df = pd.concat([target_min_df, target_max_df])
    min_max_data = min_max_df.to_numpy()
    # copy = False, if input already is numpy array
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    scaler.fit(min_max_data)
    return scaler


def normalize(data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    normalized_data = scaler.transform(data)
    return normalized_data


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


def format_timestamp_col(df: pd.DataFrame, data_source: str) -> pd.DataFrame:
    # data from danish maritime authority
    # https://www.dma.dk/SikkerhedTilSoes/Sejladsinformation/AIS/Sider/default.aspx
    if data_source == "dk":
        df = df.assign(time=pd.to_datetime(df["# Timestamp"], format='%d/%m/%Y %H:%M:%S')
                       .values.astype(np.int64) // 10**9)
        df = df.drop(columns=["# Timestamp"])
    # data from marine cadastre: https://marinecadastre.gov/ais/
    elif data_source == "mc":
        df = df.assign(time=pd.to_datetime(df["BaseDateTime"], format='%Y-%m-%d %H:%M:%S')
                       .values.astype(np.int64) // 10**9)
        df = df.drop(columns=["# Timestamp"])
    return df


def generate(data_dir: str, output_dir: str, data_source: str) -> None:
    print(f"Generating dataset from directory '{data_dir}'")
    # initialize port manager
    pm = PortManager()
    pm.load()
    if len(pm.ports.keys()) < 1:
        raise ValueError("No port data available")

    # iterate all raw .csv files in given directory
    files = sorted(os.listdir(data_dir))
    for idx, file in enumerate(files):
        if file.startswith("aisdk_"):
            generate_dataset(os.path.join(data_dir, file), output_dir, data_source, pm)


def generate_dataset(file_path: str, output_dir: str, data_source: str, pm: PortManager) -> None:
    print(f"Extracting file from '{file_path}' of type '{data_source}'")
    min_number_of_rows = 10000
    numerical_features = ["time", "Latitude", "Longitude", "SOG", "COG", "Heading", "Width", "Length", "Draught"]
    categorical_features = ["Ship type", "Navigational status"]
    df = pd.read_csv(file_path, ",", None)

    if data_source == "dma":
        df = df.drop(columns=["Type of mobile", "ROT", "IMO", "Callsign", "Name", "Cargo type",
                              "Type of position fixing device", "ETA", "Data source type", "A", "B", "C", "D"])

    # unify data sources to 'dma' source
    if data_source == "mc":
        df = df.rename(columns=mc_to_dk)
        df = df.drop(columns={"VesselName", "IMO", "Callsign", "Cargo", "TransceiverClass"})

    # fill NaN values with their defaults from official AIS documentation
    # https://api.vtexplorer.com/docs/response-ais.html
    df.fillna(value={"Heading": 511})

    # filter out of range values
    df = df.loc[(df["Latitude"] >= data_ranges["Latitude"]["min"])
                & (df["Latitude"] <= data_ranges["Latitude"]["max"])]
    df = df.loc[(df["Longitude"] >= data_ranges["Longitude"]["min"])
                & (df["Longitude"] <= data_ranges["Longitude"]["max"])]

    # assert if enough data remains
    if len(df.index) < min_number_of_rows:
        logger.write(f"Required {min_number_of_rows} rows of data, got {len(df.index)}")

    initialize(output_dir)
    scaler = None

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
        print(f"Port match: {port.name}")

        for folder in data_folders:
            if not os.path.exists(os.path.join(output_dir, folder, port.name)):
                os.makedirs(os.path.join(output_dir, folder, port.name))

        dest_df = df.loc[df["Destination"] == dest_column_header]

        dest_df = dest_df.drop(columns=["Destination"])
        dest_df = format_timestamp_col(dest_df)
        # print(f"dest_df:\n{dest_df}")

        # extract data-points that are sent while sitting in port to compute label
        x_df, arrival_times_df = pm.identify_arrival_times(port, dest_df)

        # skip port if all ships are hanging out in port area only
        if is_empty(x_df):
            print("Data for port {} only within port area. {} data, {} labels".format(port.name, len(x_df.index),
                                                                                      len(arrival_times_df.index)))
            logger.write("No Data for port {} outside of port area. {} data, {} labels".format(port.name,
                                                                                               len(x_df.index),
                                                                                               len(arrival_times_df.index)))
            continue

        # init route combiner on existing unlabeled data for current port
        rc = RouteCombiner(os.path.join(output_dir, "unlabeled", port.name))
        rc.fit()

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

            # print("arrival time for mmsi {}: {}".format(mmsi, arrival_time))
            # TODO: Consider not dropping MMSI. But keep in mind: MMSI number has no natural order
            ship_df = ship_df.drop(columns=["MMSI"])
            # print("data: \n", ship_df)

            # ship_categorical_df = x_categorical_df.loc[x_categorical_df["MMSI"] == mmsi]
            # ship_categorical_df = ship_categorical_df.drop(columns=["MMSI"])

            if arrival_time == -1:
                print(f"No label found for MMSI {mmsi}. Saving as unlabeled.")
                if rc.has_match(mmsi):
                    ship_df = rc.match(mmsi, ship_df)
                ship_df.to_pickle(os.path.join(output_dir, "unlabeled", port.name, obj_file("data_unlabeled", mmsi)))
                continue

            ship_df, labeler = generate_label(ship_df, arrival_time)
            # print("df with added label: \n", ship_df)

            if scaler is None:
                scaler = init_scaler(x_df, ship_df.columns.tolist())
                joblib.dump(scaler, os.path.join(output_dir, "encode", "normalizer.pkl"))

            if rc.has_match(mmsi):
                ship_df = rc.match(mmsi, ship_df)
            data = ship_df.to_numpy()
            # print("Shape of data for MMSI {}: {} Type: {}".format(mmsi, data.shape, data.dtype))
            # print("data: ", data)
            # label = ship_categorical_df.to_numpy()

            train, test, val = split(data)

            # Intuition: MSE loss is huge if target has large values like duration in seconds
            train_normalized = normalize(train, scaler)
            test_normalized = normalize(test, scaler)
            val_normalized = normalize(val, scaler)
            # print(f"normalized train data:\n{train_normalized}")
            # print(f"normalized train shape: {train_normalized.shape}")
            # print(f"normalized test shape: {test_normalized.shape}")
            # print(f"normalized validate shape: {val_normalized.shape}")

            np.save(os.path.join(output_dir, "train", port.name, data_file(mmsi)), train_normalized)
            np.save(os.path.join(output_dir, "test", port.name, data_file(mmsi)), test_normalized)
            np.save(os.path.join(output_dir, "validate", port.name, data_file(mmsi)), val_normalized)

            joblib.dump(labeler, os.path.join(output_dir, "encode", port.name, obj_file("labeler", mmsi)))
            joblib.dump(ship_type_encoder, os.path.join(output_dir, "encode", port.name, obj_file("ship_type", mmsi)))
            joblib.dump(nav_status_encoder, os.path.join(output_dir, "encode", port.name, obj_file("nav_status", mmsi)))

    print("Done.")


def split(data: np.ndarray, train_ratio: float = .9, test_val_ratio: float = .5) -> Tuple[np.ndarray, np.ndarray,
                                                                                          np.ndarray]:
    if 0 < train_ratio < 1:
        train, remain = np.split(data, [int(train_ratio*data.shape[0])])
        test, val = np.split(remain, [int(test_val_ratio*remain.shape[0])])
        return train, test, val
    else:
        print("Unable to split by {}. Use a value within (0, 1)".format(train_ratio))
        return data, np.array([]), np.array([])


def main(args) -> None:
    if args.command == "init":
        write_to_console("Initializing repo")
        initialize(args.output_dir)
    elif args.command == "generate":
        write_to_console("Generating data")
        generate(args.input_path, args.output_dir, args.data_source)
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
    data_path = os.path.join(script_dir, "data", "raw", big)
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("command", choices=["init", "generate", "cc", "add_alias", "check_ports"])
    parser.add_argument("--data_source", choices=["dma", "mc"], default="dma",
                        help="Source type for raw dataset: 'dma' - Danish Marine Authority, 'mc' - MarineCadastre")
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, "data", "raw", "dma"),
                        help="Path to directory of AIS .csv files")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "data"),
                        help="Output directory path")
    parser.add_argument("--ship_type", type=str, default="Cargo", choices=["Cargo", "Fishing", "Passenger", "Military",
                                                                           "Tanker"])
    main(parser.parse_args())
