import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

from matplotlib import ticker
from typing import Dict, List, Tuple, Union

from port import Port, PortManager
from util import data_ranges

script_dir = os.path.abspath(os.path.dirname(__file__))

series_colors = ["b", "r", "g"]
cm = 1/2.54  # centimeters in inches


def plot_series(series: Union[List[float], List[List[float]], Tuple[List[float], List[float]]], x_label: str,
                y_label: str, title: str = None, legend_labels: Union[str, List[str]] = None, x_ticks: float = None,
                y_ticks: float = None, x_scale: str = None, y_scale: str = None, path: str = None,
                x_vline: int = None, x_vline_label: str = None, mark_min: List[int] = [-1]) -> None:
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
    :param x_vline:
        x-pos for vertical line
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

    def _mark_min(_series) -> None:
        y = min(_series)
        x = [pos for pos, x in enumerate(_series) if x == y][0]
        ax.plot(x, y, "o--")
        ax.text(x=y, y=y, s=str(int(y)), ha="center", va="bottom")

    ymax = 0.
    if num_series == 1 and type(series[0]) == float:
        ax.plot(series)
        ymax = max(series)
        if len(mark_min) > 0 and mark_min[0] != -1:
            _mark_min(series)
    else:
        for i in range(num_series):
            curr_series = series[i]
            if type(curr_series[0]) == list and type(curr_series[1] == list):
                ax.plot(curr_series[0], curr_series[1])
                max_curr = max(curr_series[0])
                if max(curr_series[1]) > max_curr:
                    max_curr = max(curr_series[1])
                if ymax < max_curr:
                    ymax = max_curr
                if i in mark_min:
                    _mark_min(curr_series)
            elif type(curr_series) == list:
                ax.plot(curr_series)
                if ymax < max(curr_series):
                    ymax = max(curr_series)
                if i in mark_min:
                    _mark_min(curr_series)
    if x_vline is not None:
        plt.vlines(x_vline, ymin=0, ymax=ymax, linestyle="dashed", linewidth=1.5, color="green", label=x_vline_label)

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


def plot_ports_by_mae(mae: List[float], ports: List[str], title: str, path: str = None) -> None:
    assert len(mae) == len(ports)
    fix, ax = plt.subplots(figsize=(30*cm, 20*cm))
    x = np.arange(len(mae))
    width = .35

    bars = ax.bar(x, mae, width)
    ax.set_title(title)
    ax.set_ylabel("MAE in minutes")
    ax.set_xticks(x)
    ax.set_xticklabels(ports, rotation=45, ha="center")
    for i, val in enumerate(mae):
        # ax.text(x=i, y=val, s=data[4][i], ha="center", va="bottom")
        ax.text(x=i, y=val, s=val, ha="center", va="bottom")
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(_y_format))
    # ax.legend()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def plot_grouped_maes(data: List[Tuple[int, int, int, float, str]], title: str, path: str = None) -> None:
    """
    Generate plot with groups of maes
    :param data: Result from 'util.mae_by_duration': List of Tuples
        [(group_start, group_end, num_data, mae, group_description), ...]
    :param title: Plot title
    :param path: Output path for plot
    :return: None
    """
    data = list(map(list, zip(*data)))

    x = np.arange(len(data[0]))
    max_num = max(data[2])
    num_range = max_num - min(data[2])
    widths = [round(0.35 + n / num_range * 0.35, 2) for n in data[2]]
    avg = sum(data[3]) / len(data[3])

    greens = plt.get_cmap("Greens")

    def _scaled_num_data(n: int) -> float:
        return n / max_num
    colors = [greens(_scaled_num_data(n)) for n in data[2]]

    fig, ax = plt.subplots(figsize=(30*cm, 15*cm))
    bars = ax.bar(x=x, height=data[3], width=widths, color=colors, edgecolor="black", linewidth=.5)
    plt.axhline(avg, linestyle="dashed", linewidth=1.5, color="black")
    plt.text(x=0, y=avg, s=f"Avg {_y_minutes(avg)}", ha="right", va="bottom")
    ax.set_title(title)
    ax.set_ylabel("MAE - ETA in minutes")
    ax.set_xticks(x)
    ax.set_xticklabels(data[4], rotation=45, ha="right")
    for i, val in enumerate(data[3]):
        # ax.text(x=i, y=val, s=data[3][i], ha="center", va="bottom")
        ax.text(x=i, y=val, s=_y_minutes(data[3][i]), ha="center", va="bottom")
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(_y_format))
    fig.tight_layout()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def plot_transfer_effect(base_data: List[Tuple[int, int, int, float, str]],
                         transfer_data: List[Tuple[int, int, int, float, str]],
                         port_name_base: str, port_name_target: str, path: str) -> None:
    assert len(base_data) == len(transfer_data)
    title = f"Transfer effect on MAE\nBase: {port_name_base} -> Target: {port_name_target}"
    base_data = list(map(list, zip(*base_data)))
    transfer_data = list(map(list, zip(*transfer_data)))
    x = np.arange(len(base_data[0]))
    width = 0.35

    base_avg = sum(base_data[3]) / len(base_data[3])
    transfer_avg = sum(transfer_data[3]) / len(transfer_data[3])

    fig, ax = plt.subplots(figsize=(30*cm, 15*cm))
    # bars
    ax.bar(x - width/2, base_data[3], width, color="blue", label="Base Training")
    ax.bar(x + width/2, transfer_data[3], width, color="orange", label="Transfer Training")
    # average h-lines
    plt.axhline(base_avg, linestyle="dashed", linewidth=1.5, color="blue")
    plt.text(x=-1, y=base_avg, s=f"Avg base {_y_minutes(base_avg)}", ha="left", va="bottom")
    plt.axhline(transfer_avg, linestyle="dashed", linewidth=1.5, color="orange")
    plt.text(x=-1, y=transfer_avg, s=f"Avg transfer {_y_minutes(transfer_avg)}", ha="left", va="bottom")
    ax.set_title(title)
    ax.set_ylabel("MAE - ETA in minutes")
    ax.set_xticks(x)
    ax.set_xticklabels(base_data[4], rotation=45, ha="right")
    for i in range(len(base_data[3])):
        ax.text(x=i - width / 2, y=base_data[3][i], s=_y_minutes(base_data[3][i]), ha="center", va="bottom")
        ax.text(x=i + width / 2, y=transfer_data[3][i], s=_y_minutes(transfer_data[3][i]), ha="center", va="bottom")
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(_y_format))
    ax.legend()
    fig.tight_layout()

    plt.savefig(path)


def plot_transfer_effects(transfer_port_names: List[str],
                          max_mae_base_port_names: List[str], max_mae_base: List[float],
                          min_mae_base_port_names: List[str], min_mae_base: List[float],
                          max_mae_transfer_port_names: List[str], max_mae_transfer: List[float],
                          min_mae_transfer_port_names: List[str], min_mae_transfer: List[float],
                          avg_mae_base: List[float], avg_mae_transfer: List[float],
                          path: str) -> None:
    """
    Plots the 'cost' or 'benefits' of multiple transfers: Reference: Base port mae in comparison to transfer port mae
    :param transfer_port_names: Names of transferred ports
    :param max_mae_base_port_names: Names of ports with max mae from all transfers of specific port(param1)
    :param max_mae_base: Max mae values from all transfers according to prev param's port name
    :param min_mae_base_port_names: Names of ports with min mae from all transfers of specific port (param1)
    :param min_mae_base: Min mae values from all transfers according to prev param's port name
    :param max_mae_transfer_port_names: Names of ports with max mae from
    :param max_mae_transfer: Max mae values
    :param min_mae_transfer_port_names: Name of ports with min mae
    :param min_mae_transfer: Min mae values
    :param avg_mae_base: Average mae of all source ports (base-training)
    :param avg_mae_transfer: Average mae of all transfers
    :param path: Save path
    :return:
    """
    assert len(transfer_port_names) == len(max_mae_base_port_names) == len(max_mae_base) == \
           len(min_mae_base_port_names) == len(min_mae_base) == len(max_mae_transfer_port_names) == \
           len(max_mae_transfer) == len(min_mae_transfer_port_names) == len(min_mae_transfer) == len(avg_mae_base) == \
           len(avg_mae_transfer)
    title = f"Transfer effects on ports"
    x = np.arange(len(avg_mae_base))
    width = .35

    fig, ax = plt.subplots(figsize=(30*cm, 15*cm))
    ax.minorticks_on()

    def compute_diff(mae_b, mae_t) -> Tuple[float, float]:
        return (mae_t - mae_b, 0.) if mae_b > mae_t else (0., mae_b - mae_t)

    yerr_base = [min_mae_base, max_mae_base]
    yerr_transfer = [min_mae_transfer, max_mae_transfer]

    # bars
    diffs = [compute_diff(avg_mae_transfer[i], avg_mae_base[i]) for i in range(len(avg_mae_transfer))]
    diffs = list(map(list, zip(*diffs)))
    base_bars = ax.bar(x - width/2, avg_mae_base, color="gray")
    base_bars_diff = ax.bar(x - width/2, diffs[0], yerr=yerr_base, bottom=avg_mae_base, color="red")
    transfer_bars = ax.bar(x + width/2, avg_mae_base, color="blue")
    transfer_bars_diff = ax.bar(x + width/2, diffs[1], yerr=yerr_transfer, bottom=avg_mae_base, color="green")

    # h-lines
    # plt.hlines(y=0., xmin=0., xmax=0., linestyles="dashed")

    ax.set_title(title)
    ax.set_ylabel("MAE - ETA in minutes")
    ax.set_xticks(x)
    for i in range(len(avg_mae_base)):
        ax.text(x=i - width, y=avg_mae_base[i], s=_y_minutes(avg_mae_base[i]), ha="center", va="bottom")
        ax.text(x=i + width, y=avg_mae_transfer[i], s=_y_minutes(avg_mae_transfer[i]), ha="center", va="bottom")

    # plot table for min an max data values
    def compute_table_col(min_b: Tuple[float, str], max_b: Tuple[float, str], min_t: Tuple[float, str],
                          max_t: Tuple[float, str]) -> List[str]:
        min_b_cell = f"{min_b[1]} {min_b[0]}"
        max_b_cell = f"{max_b[1]} {max_b[0]}"
        min_t_cell = f"{min_t[1]} {min_t[0]}"
        max_t_cell = f"{max_t[1]} {max_t[0]}"
        return [min_b_cell, max_b_cell, min_t_cell, max_t_cell]

    cell_text = [compute_table_col((min_mae_base[i], min_mae_base_port_names[i]),
                                   (max_mae_base[i], max_mae_base_port_names[i]),
                                   (min_mae_transfer[i], min_mae_transfer_port_names[i]),
                                   (max_mae_transfer[i], max_mae_transfer_port_names[i]))
                 for i in range(len(avg_mae_base))]
    cell_text = list(map(list, zip(*cell_text)))
    # print(f"Cell text:\n{cell_text}")
    # print(f"{len(cell_text)} x {len(cell_text[0])}")
    rows = ["Min source", "Max source", "Min transfer", "Max transfer"]
    colors = ["gray", "gray", "gray", "gray"]
    plt.table(cellText=cell_text, rowLabels=rows, rowColours=colors, colLabels=transfer_port_names, loc="bottom")
    plt.subplots_adjust(left=0.2, bottom=0.2)

    # format y-labels
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(_y_format))
    fig.tight_layout()

    plt.savefig(path)


def _y_minutes(val, pos = None) -> str:
    return str(int(val))


def plot_port_visits(mae_data: Dict[str, float]) -> None:
    ports = ["Esbjerg", "Rostock", "Kiel", "Skagen", "Trelleborg", "Thyboron", "Hirthals", "Hvidesande", "Aalborg",
             "Goteborg", "Copenhagen", "Grenaa", "Malmo", "Helsingborg", "Hanstholm", "Fredericia", "Horsens",
             "Kalundborg", "Frederikshavn", "Varberg", "Mukran", "Randers", "Hamburg"]
    visits = [2349, 2232, 1529, 1053, 787, 768, 664, 662, 648, 637, 632, 511, 474, 470, 445, 429, 402, 390, 377, 369,
              335, 309, 268]

    mae = [mae_data[port.upper()] if port in mae_data else 0 for port in ports]
    x = np.arange(len(ports))
    fig, ax = plt.subplots(fizsize=(30*cm, 15*cm))
    bars = ax.bar(x, visits)
    ax.plot(x, mae, color="red")
    ax.set_title("MAE compared to number of unique visits per port")
    ax.set_xticks(x)
    ax.set_xticklabels(ports, rotation=45, ha="right")
    for i, n in enumerate(visits):
        ax.text(x=i, y=n, s=n, ha="center", va="center")


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
                    legend_labels=["One"], path=os.path.join(args.output_dir, "test1.png"))
        plot_series(series=test_series_2, x_label="X", y_label="Y", title="y given, two series",
                    legend_labels=["One", "Two"], path=os.path.join(args.output_dir, "test2.png"))
        plot_series(series=test_series_3, x_label="X", y_label="Y", title="x and y given, one series",
                    legend_labels=["One"], path=os.path.join(args.output_dir, "test3.png"))
        plot_series(series=test_series_4, x_label="X", y_label="Y", title="x and y given, two series",
                    legend_labels=["One", "Two"], path=os.path.join(args.output_dir, "test4.png"))
    elif command == "plot_loss":
        pm = PortManager()
        pm.load()
        if len(pm.ports.keys()) < 1:
            raise ValueError("No port data available")
        if args.port_name == "all":
            asdf = "all"
        port = pm.find_port(args.port_name)
        if port is None:
            raise ValueError(f"Unable to find port for name '{args.port_name}'")
        trainings = pm.load_trainings(port=port, output_dir=args.output_dir, routes_dir=args.routes_dir,
                                      training_type=args.training_type)

    else:
        raise ValueError(f"Unknown command '{command}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"plot_loss", "test"})
    parser.add_argument("--routes_dir", type=str, default=os.path.join(script_dir, "data", "routes"),
                        help="Directory to routes")
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "output"), help="Path to output")
    parser.add_argument("--port_name", type=str, help="Name of port to plot loss history")
    parser.add_argument("--training_type", type=str, choices=["training", "transfer"], default="training",
                        help="Types: training or transfer")
    parser.add_argument("--latest", type=bool, default=True, help="Plot latest training/transfer or all")
    main(parser.parse_args())
