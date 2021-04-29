import argparse
import os
import pandas as pd
import numpy as np
import joblib
import re
# import random
# import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from typing import List, Tuple

from combiner import RouteCombiner
from labeler import DurationLabeler
from logger import Logger
from port import Port, PortManager
from util import data_ranges, categorical_values, get_destination_file_name, is_empty, encode_data_file, obj_file,\
    write_to_console, mc_to_dk, as_str

script_dir = os.path.abspath(os.path.dirname(__file__))
logger = Logger(file_name=f"log_data-generation_{as_str(datetime.now())}")
output_folders = ["encode", "routes", "unlabeled"]


def initialize(output_dir:  str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for folder in output_folders + ["raw"]:
        if not os.path.exists(os.path.join(output_dir, folder)):
            os.makedirs(os.path.join(output_dir, folder))


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
    scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
    scaler.fit(min_max_data)
    return scaler


def normalize(data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    normalized_data = scaler.transform(data)
    return normalized_data


def denormalize(data: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    return scaler.inverse_transform(data)


# create one-hot encoded DataFrame: one column for each category
def one_hot_encode(series: pd.Series, ais_col_name: str) -> Tuple[pd.DataFrame, OneHotEncoder]:
    series_categories = series.unique()  # in order of appearance
    ais_col_name = ais_col_name.lower()
    default_alias = ["Default", "Unknown", "Undefined"]

    if ais_col_name in categorical_values:
        categories = categorical_values[ais_col_name] + default_alias
        nan_idx = -1
        # prevent typing error by replacing NaN with custom decorator "nan_type"
        for idx, cat in enumerate(series_categories):
            if type(cat) is float:
                nan_idx = idx

        if nan_idx > -1:
            series_categories[nan_idx] = "default"

        for s_cat in series_categories:
            i = 0
            found = False
            while i < len(categories) and not found:
                cat = categories[i]
                if cat.lower() in s_cat.lower():
                    if cat in default_alias:
                        series = series.replace(s_cat, f"Default {ais_col_name}")
                    else:
                        series = series.replace(s_cat,  cat)
                    found = True
                i += 1
            if not found or s_cat.lower() == "other":
                # print(f"One hot encoding: Unable to map value '{s_cat}' to known category from '{ais_col_name}'")
                series = series.replace(s_cat, f"Other {ais_col_name}")

        # print(f"unique values after mapping:\n{series.unique()}")
        desired_categories = categorical_values[ais_col_name] + [f"Other {ais_col_name}", f"Default {ais_col_name}"]
        # print(f"desired categories:\n{desired_categories}")

        series = series.values.reshape(-1, 1)  # shape as column
        encoder = OneHotEncoder(sparse=False, categories=[desired_categories])
        encoder.fit(series)
        data_ohe = encoder.transform(series)
        df_ohe = pd.DataFrame(data_ohe, columns=[desired_categories[i] for i in range(len(desired_categories))])
        return df_ohe, encoder
        # return data_ohe, desired_categories, encoder

    else:
        raise ValueError(f"Error while one hot encoding: Cannot find column {ais_col_name} "
                         f"in {categorical_values.keys()}")


def format_timestamp_col(df: pd.DataFrame, data_source: str) -> pd.DataFrame:
    # data from danish maritime authority
    # https://www.dma.dk/SikkerhedTilSoes/Sejladsinformation/AIS/Sider/default.aspx
    if data_source == "dma":
        df = df.assign(time=pd.to_datetime(df["# Timestamp"], format='%d/%m/%Y %H:%M:%S')
                       .values.astype(np.int64) // 10**9)
        df = df.drop(columns=["# Timestamp"])
    # data from marine cadastre: https://marinecadastre.gov/ais/
    elif data_source == "mc":
        df = df.assign(time=pd.to_datetime(df["BaseDateTime"], format='%Y-%m-%d %H:%M:%S')
                       .values.astype(np.int64) // 10**9)
        df = df.drop(columns=["BaseDateTime"])
    return df


def generate(input_dir: str, output_dir: str, data_source: str) -> None:
    print(f"Generating dataset from directory '{input_dir}'")
    initialize(output_dir)

    # initialize port manager
    pm = PortManager()
    pm.load()
    if len(pm.ports.keys()) < 1:
        raise ValueError("No port data available")

    # iterate all raw .csv files in given directory
    files = sorted(os.listdir(input_dir))
    for idx, file in enumerate(files):
        if file.startswith("aisdk_"):
            generate_dataset(os.path.join(input_dir, file), output_dir, data_source, pm)
    print("Data generation complete!")


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
    # COG = 0 and SOG = 0 might not be the best default values
    df = df.fillna(value={"Heading": 511, "SOG": 0, "COG": 0, "Width": 0, "Length": 0, "Draught": 0})

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

        for folder in output_folders:
            if not os.path.exists(os.path.join(output_dir, folder, port.name)):
                os.makedirs(os.path.join(output_dir, folder, port.name))

        dest_df = df.loc[df["Destination"] == dest_column_header]

        dest_df = dest_df.drop(columns=["Destination"])
        dest_df = format_timestamp_col(dest_df, data_source)
        # print(f"dest_df:\n{dest_df}")

        # extract data-points that are sent while sitting in port to compute label
        x_df, arrival_times_df = pm.identify_arrival_times(port, dest_df)

        # skip port if all ships are hanging out in port area only
        if is_empty(x_df):
            # print(f"Data for port {port.name} only within port area. {len(x_df.index)} data,"
            #      f"{len(arrival_times_df.index)} labels")
            logger.write(f"No data for port {port.name} outside of port area. "
                         f"{x_df.index} number of data-points, {arrival_times_df.index} number of labels")
            continue

        # init route combiner on existing unlabeled data for current port
        # rc = RouteCombiner(data_dir=os.path.join(output_dir, "unlabeled", port.name),
        #                    csv_map_path=os.path.join(output_dir, "raw", "dma", "csv_to_route.json"))
        rc = RouteCombiner(data_dir=os.path.join(output_dir, "unlabeled", port.name),
                           csv_map_path=os.path.join(output_dir, "dma", "combinations.json"))
        rc.fit()

        # handle categorical data
        # ship_types_df, ship_type_encoder = one_hot_encode(x_df.pop("Ship type"), "Ship type")
        ship_types_df, ship_type_encoder = one_hot_encode(x_df.pop("Ship type"), "Ship type")
        nav_states_df, nav_status_encoder = one_hot_encode(x_df.pop("Navigational status"), "Navigational Status")
        # df_cargo_types, cargo_types_encoder = one_hot_encode(x_df.pop("Cargo type"))
        # print(f"ohe ship_types:\n{ship_types_df}")
        # print(f"ohe nav_states:\n{nav_states_df}")

        arrival_times_df = arrival_times_df.drop(columns=["Ship type", "Navigational status"])

        # print(f"x_df: {x_df.shape} ship_types_df: {ship_types_df.shape}")
        # unite source df with one hot encoded data
        x_df = pd.concat([x_df.reset_index(drop=True), ship_types_df.reset_index(drop=True),
                          nav_states_df.reset_index(drop=True)], axis=1)
        # print(f"concatenated x_df:\n{x_df}")

        # mmsi_col = x_df["MMSI"]
        # Add MMSI identification to categorical features
        # x_ship_types["MMSI"], x_nav_states["MMSI"], x_cargo_types["MMSI"] = mmsi_col, mmsi_col, mmsi_col
        # ship_types_df["MMSI"], nav_states_df["MMSI"] = mmsi_col, mmsi_col

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

                # if not is_empty(ship_df):
                    # print("- - - - - - - - - - - - - - - - - - - - - - -")
                    # print("- - T A R G E T  &  D A T A   F O U N D - - -")
                    # print("- - - - - - - - - - - - - - - - - - - - - - -")
                    # print(f"MMSI {mmsi}")

            # print(f"ship_df length before 'is_empty': {len(ship_df.index)}")
            if is_empty(ship_df):
                continue

            # print("arrival time for mmsi {}: {}".format(mmsi, arrival_time))
            ship_df = ship_df.drop(columns=["MMSI"])
            # print("data: \n", ship_df)

            _, file_name = os.path.split(file_path)
            file_date = rc.date_from_source_csv(file_name)

            if arrival_time == -1:
                # print(f"No label found for MMSI {mmsi}. Saving as unlabeled.")
                if rc.has_match(str(mmsi), file_date):
                    ship_df = rc.match(str(mmsi), file_date, ship_df)
                f_path = os.path.join(output_dir, "unlabeled", port.name, obj_file("data_unlabeled", mmsi, file_date))
                # print(f"ship_df:\n{ship_df}")
                # print(f"path for pickle: {f_path}")
                ship_df.to_pickle(f_path)
                continue

            # print(f"ship_df before matching (MMSI {mmsi})\n{ship_df}")
            if rc.has_match(str(mmsi), file_date):
                ship_df = rc.match(str(mmsi), file_date, ship_df)
            # print(f"ship_df after matching:\n{ship_df}")

            ship_df, labeler = generate_label(ship_df, arrival_time)
            # print("df with added label: \n", ship_df)

            if scaler is None:
                scaler = init_scaler(x_df, ship_df.columns.tolist())
                joblib.dump(scaler, os.path.join(output_dir, "encode", "normalizer.pkl"))

            data = ship_df.to_numpy()
            # print(f"Shape after all: {data.shape}")
            # print(f"Data:\n{data}")

            data_normalized = normalize(data, scaler) if data.shape[0] > 0 else data
            # print(f"normalized data:\n{data_normalized}")

            routes_dir = os.path.join(output_dir, "routes", port.name)
            data_file_path = encode_data_file(mmsi, routes_dir, join=True)

            np.save(data_file_path, data_normalized)

            joblib.dump(labeler, os.path.join(output_dir, "encode", port.name, obj_file("labeler", mmsi)))
            joblib.dump(ship_type_encoder, os.path.join(output_dir, "encode", port.name, obj_file("ship_type", mmsi)))
            joblib.dump(nav_status_encoder, os.path.join(output_dir, "encode", port.name, obj_file("nav_status", mmsi)))


def main(args) -> None:
    if args.command == "generate":
        write_to_console("Generating data")
        generate(args.input_dir, args.output_dir, args.data_source)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    small = "small.csv"
    medium = "medium.csv"
    big = "aisdk_20181101.csv"
    data_path = os.path.join(script_dir, "data", "raw", big)
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("command", choices=["generate"])
    parser.add_argument("--data_source", choices=["dma", "mc"], default="dma",
                        help="Source type for raw dataset: 'dma' - Danish Marine Authority, 'mc' - MarineCadastre")
    parser.add_argument("--input_dir", type=str, default=os.path.join(script_dir, "data", "raw", "dma", "2020"),
                        help="Path to directory of AIS .csv files")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "data"),
                        help="Output directory path")
    parser.add_argument("--ship_type", type=str, default="Cargo", choices=["Cargo", "Fishing", "Passenger", "Military",
                                                                           "Tanker"])
    main(parser.parse_args())
