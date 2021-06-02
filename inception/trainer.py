import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tqdm import tqdm
from typing import List

from inception.eval import Evaluator
from inception.model import inception_time
from plotter import plot_series
from port import Port, PortManager
from util import as_str, encode_keras_model, encode_loss_history_plot, encode_history_file, encode_x_y_plot, \
    SECONDS_PER_YEAR, read_json, verify_output_dir

# FILES = glob.glob("/content/drive/MyDrive/ma/data/routes/SKAGEN/data_*.npy")[:]
script_dir = os.path.abspath(os.path.dirname(__file__))


def explore():
    # print(pd.DataFrame(np.load(FILES[0])))
    return


def route_to_ts(f, window_len, max_len=250):
    route = np.load(f).astype("float32")[-max_len:]

    X = route[1:, :-1]
    y = route[:-1, -1]

    data_gen = TimeseriesGenerator(X, y, length=window_len, batch_size=len(X))

    X_ts = data_gen[0][0]
    y_ts = data_gen[0][1] * SECONDS_PER_YEAR / 60

    return X_ts, y_ts


def load_data(port: Port, window_len: int):
    data_path = os.path.join(script_dir, os.pardir, "data", "routes", port.name, "data_*.npy")
    # files_old = glob.glob(data_path)[:]
    files = glob.glob(data_path)
    X_ts_list = []
    y_ts_list = []

    for f in tqdm(files, desc="Loading files"):
        # if np.load(f).shape[0] < window_len:
        file_len = np.load(f, mmap_mode="r", allow_pickle=True).shape[0]
        if file_len < (window_len + 2):
            continue
        # print(f"f len: {file_len} window len: {window_len}")
        X_ts_tmp, y_ts_tmp = route_to_ts(f, window_len)
        X_ts_list.append(X_ts_tmp)
        y_ts_list.append(y_ts_tmp.reshape(-1, 1))

    X_ts = np.vstack(X_ts_list)
    y_ts = np.vstack(y_ts_list)
    plt.plot(y_ts)
    return X_ts, y_ts


def train(output_dir: str, port_name: str = None) -> None:
    pm = PortManager()
    pm.load()
    if len(pm.ports.keys()) < 1:
        raise ValueError("No port data available")
    e = Evaluator.load(os.path.join(output_dir, "eval"))

    if isinstance(port_name, str):
        port = pm.find_port(port_name)
        if port is None:
            raise ValueError(f"Unable to associate port with port name '{port_name}'")
        train_port(port, e)
    else:
        # train ports required for transfer
        config = read_json(os.path.join(script_dir, "transfer-config.json"))
        ports = [pm.find_port(port_name) for port_name in config["ports"]]
        if None in ports:
            raise ValueError(f"Found type None in list of ports to train: {ports}")
        for port in ports:
            train_port(port, e)


def train_port(port: Port, evaluator: Evaluator) -> None:
    start_datetime = datetime.now()
    start_time = as_str(start_datetime)
    output_dir = os.path.join(script_dir, os.pardir, "output")
    verify_output_dir(output_dir, port.name)
    training_type = "base"

    print(f"\nStarted training for port '{port.name}'")
    print(f"Loading data...")
    window_len = 50
    X_ts, y_ts = load_data(port, window_len)
    X_train, X_test, y_train, y_test = train_test_split(X_ts, y_ts, test_size=0.2, random_state=42, shuffle=False)

    print(f"Train shape - X:{X_train.shape} y:{y_train.shape} Test shape - X:{X_test.shape} y:{y_test.shape}")

    model = inception_time(input_shape=(window_len, 37))
    # print(model.summary())

    model_file_name = encode_keras_model(port.name, start_time)
    # file_path = "/content/drive/MyDrive/ma/output/model/SKAGEN/inception_time.h5"
    file_path = os.path.join(output_dir, "model", port.name, model_file_name)

    checkpoint = ModelCheckpoint(file_path, monitor='val_mae', mode='min', verbose=2, save_best_only=True)
    early = EarlyStopping(monitor="val_mae", mode="min", patience=10, verbose=2)
    redonplat = ReduceLROnPlateau(monitor="val_mae", mode="min", patience=3, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]

    optimizer = Adam(learning_rate=0.01)

    # configure model
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    # train model
    result = model.fit(X_train, y_train, epochs=50, batch_size=1024, verbose=1, validation_data=(X_test, y_test),
                       callbacks=callbacks_list)
    print(f"history:\n{result.history.keys()}")
    train_loss = result.history["loss"]
    train_mae = result.history["mae"]
    val_loss = result.history["val_loss"]
    val_mae = result.history["val_mae"]
    model.load_weights(file_path)
    # plt.plot(y_ts)
    # plt.plot(model.predict(X_ts))

    baseline = mean_absolute_error(y_ts, np.full_like(y_ts, np.mean(y_ts)))
    print(f"naive baseline: {baseline}")

    # set evaluation
    print(f"Setting evaluation results...")
    evaluator.set_mae(port, start_time, val_mae)
    y_pred = model.predict(X_test)
    grouped_mae = evaluator.group_mae(y_test, y_pred)
    evaluator.set_mae(port, start_time, grouped_mae)

    # save history
    history_path = os.path.join(output_dir, "data", port.name, encode_history_file(training_type, port.name,
                                                                                   start_time))
    np.save(history_path, result.history)

    # plot history
    plot_dir = os.path.join(output_dir, "plot")
    plot_history(train_mae, val_mae, plot_dir, port.name, start_time, training_type)
    evaluator.plot_grouped_mae(port, training_type, start_time)
    plot_predictions(y_pred, y_test, plot_dir, port.name, start_time, training_type)

    # plt.plot(y_test)
    # plt.plot(model.predict(X_test))


def plot_history(train_history: List[float], val_history: List[float], plot_dir: str, port_name: str, start_time: str,
                 training_type: str, source_port_name: str = None, config_uid: int = None,
                 tune_train_history: List[float] = None, tune_val_history: List[float] = None) -> None:
    path = os.path.join(plot_dir, port_name,
                        encode_loss_history_plot(training_type, port_name, start_time, source_port_name, config_uid))
    title = f"Training loss ({training_type})"
    if tune_train_history is not None and tune_val_history is not None:
        history = (train_history + tune_train_history, val_history + tune_val_history)
    else:
        history = (train_history, val_history)
    plot_series(series=history, title=title, x_label="Epoch", y_label="Loss", legend_labels=["Training", "Validation"],
                path=path, x_vline=len(train_history), x_vline_label="Start fine tuning", mark_min=[1])


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, plot_dir: str, port_name: str, start_time: str,
                     training_type: str, base_port_name: str = None, config_uid: int = None) -> None:
    path = os.path.join(plot_dir, port_name,
                        encode_x_y_plot(training_type, port_name, start_time, base_port_name, config_uid))
    title = f"Labels and Predictions Port {port_name} ({training_type})"
    plot_series(series=(list(y_true), list(y_pred)), title=title, x_label="Training Example",
                y_label="Target Variable: ETA in Minutes", legend_labels=["Label", "Prediction"], path=path)


def main(args):
    if args.command == "explore":
        explore()
    elif args.command == "train":
        train(args.output_dir, args.port_name)
    else:
        raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training endpoint")
    parser.add_argument("command", choices=["train", "explore"])
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, os.pardir, "data"),
                        help="Path to data file directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, os.pardir, "output"),
                        help="Path to output directory")
    parser.add_argument("--port_name", type=str, help="Name of port to train model")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Specify if training shall recover from previous checkpoint")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    main(parser.parse_args())
