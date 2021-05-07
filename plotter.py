import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import ticker
from typing import List, Tuple, Union

script_dir = os.path.abspath(os.path.dirname(__file__))

series_colors = ["b", "r", "g"]


def plot_series(series: Union[List[float], List[List[float]], Tuple[List[float], List[float]]], x_label: str, y_label: str,
                title: str = None, legend_labels: Union[str, List[str]] = None, x_ticks: float = None,
                y_ticks: float = None, x_scale: str = None, y_scale: str = None, path: str = None) -> None:
    """
    :param series:
        One or more series to plot. Each list must contain x-values and optional y-values. If no y-values are given,
        these values are assumed as ascending numbers 1, 2, 3, ..., n
    :param x_label:
        Label for x-axis
    :param y_label:
        Label for y-axis
    :param title:
        Title of figure
    :param legend_labels:
        Label for each series
    :param x_ticks:
        Ticks for x-axis
    :param y_ticks:
        Ticks for y-axis
    :param x_scale
        Custom scale for x-axis
    :param y_scale
        Custom scale for y-axis
    :param path:
        Path for saving the plot
    :return: None
    """
    scales = ["linear", "log", "symlog", "logit"]

    num_series = 1 if type(series[0]) == float else len(series)
    if legend_labels is not None:
        num_legend_labels = 1 if type(legend_labels) == str else len(legend_labels)
        assert num_legend_labels == num_series
    plt.switch_backend("agg")   # set backend explicitly to run on Jupyter notebooks
    fix, ax = plt.subplots()
    if x_ticks is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=x_ticks))
    if y_ticks is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=y_ticks))
    if x_scale is not None:
        ax.set_xscale(x_scale) if x_scale in scales else print(f"Unable to apply scaling '{x_scale}' to x-axis")
    if y_scale is not None:
        ax.set_yscale(y_scale) if y_scale in scales else print(f"Unable to apply scaling '{y_scale}' to y-axis")

    if num_series == 1 and type(series[0]) == float:
        ax.plot(series)
    else:
        for i in range(num_series):
            curr_series = series[i]
            if type(curr_series[0]) == list and type(curr_series[1] == list):
                ax.plot(curr_series[0], curr_series[1])
            else:
                ax.plot(curr_series)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend_labels is not None:
        ax.legend(legend_labels)
    if title is not None:
        plt.title(title)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def plot_bars(data: List[float], bar_labels: List[str], title: str, y_label: str, path: str = None) -> None:
    fix, ax = plt.subplots()
    indices = np.arange(len(data))

    p = ax.bar(indices, label="MAEs")

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(indices)
    ax.set_xticklabels(bar_labels)
    # ax.legend()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def plot_grouped_maes(data: List[Tuple[int, int, int, float, str, str]], path: str = None) -> None:
    """
    Generate plot with groups of maes
    :param data: Result from 'util.mae_by_duration': List of Tuples
        [(group_start, group_end, num_data, scaled_mae, descaled_mae, group_description), ...]
    :param path: Output path for plot
    :return: None
    """
    data = list(map(list, zip(*data)))

    x = np.arange(len(data[0]))
    num_range = max(data[2]) - min(data[2])
    widths = [round(0.35 + n / num_range * 0.35, 2) for n in data[2]]  # wrong inspection i guess

    fig, ax = plt.subplots()
    bars = ax.bar(x, data[3], widths)
    ax.set_title("MAE by duration until arrival")
    ax.set_ylabel("MAE - ETA")
    ax.set_xticks(x)
    ax.set_xticklabels(data[5], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()

    # for i, entry in enumerate(data):
    #     ax.text(x=i, y=entry[3], s=entry[4], ha="center", va="center")

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def main(args) -> None:
    command = args.command
    if command == "test":
        test_series_1 = [0., 1., 4., 6., 8., 10., 12., 14., 16., 18.]
        test_series_2 = [[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
                         [0., 1., 4., 6., 8., 10., 12., 14., 16., 18.]]
        test_series_3 = [[[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
                          [0., 1., 4., 9., 16., 25., 36., 49., 64., 81.]]]
        test_series_4 = [[[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
                          [0., 1., 4., 9., 16., 25., 36., 49., 64., 81.]],
                         [[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
                          [81., 64., 49., 36., 25., 16., 9., 4., 1., 0.]]]
        plot_series(series=test_series_1, x_label="X", y_label="Y", title="y given, one series",
                    legend_labels=["One"], path=os.path.join(args.output_path, "test1.png"))
        plot_series(series=test_series_2, x_label="X", y_label="Y", title="y given, two series",
                    legend_labels=["One", "Two"], path=os.path.join(args.output_path, "test2.png"))
        plot_series(series=test_series_3, x_label="X", y_label="Y", title="x and y given, one series",
                    legend_labels=["One"], path=os.path.join(args.output_path, "test3.png"))
        plot_series(series=test_series_4, x_label="X", y_label="Y", title="x and y given, two series",
                    legend_labels=["One", "Two"], path=os.path.join(args.output_path, "test4.png"))
    else:
        raise ValueError(f"Unknown command '{command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"plot", "test"})
    parser.add_argument("--output_path", type=str, default=os.path.join(script_dir, "output"), help="Path to output")
    parser.add_argument("--input_dir", type=str, default=os.path.join(script_dir, "output", "plots"), help="Directory to training plit data ")
    main(parser.parse_args())
