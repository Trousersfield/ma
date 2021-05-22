import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import ticker
from typing import Dict, List, Tuple, Union

from port import Port, PortManager

script_dir = os.path.abspath(os.path.dirname(__file__))

series_colors = ["b", "r", "g"]
cm = 1/2.54  # centimeters in inches


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


def plot_ports_by_mae(maes: List[float], ports: List[str], title: str, path: str = None) -> None:
    assert len(maes) == len(ports)
    fix, ax = plt.subplots()
    x = np.arange(len(maes))

    p = ax.bar(x, maes)
    ax.set_title(title)
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(ports, rotation=45, ha="right")
    # ax.legend()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def plot_grouped_maes(data: List[Tuple[int, int, int, float, str, str]], port_name: str = None,
                      path: str = None, title: str = None) -> None:
    """
    Generate plot with groups of maes
    :param data: Result from 'util.mae_by_duration': List of Tuples
        [(group_start, group_end, num_data, mae, mas_as_str, group_description), ...]
    :param port_name: Name of port
    :param path: Output path for plot
    :param title: Plot title
    :return: None
    """
    if title is None:
        title = "MAE by label duration until arrival"
    if port_name is not None:
        title = f"{title} ({port_name})"
    data = list(map(list, zip(*data)))

    x = np.arange(len(data[0]))
    max_num = max(data[2])
    num_range = max_num - min(data[2])
    widths = [round(0.35 + n / num_range * 0.35, 2) for n in data[2]]

    greens = plt.get_cmap("Greens")

    def _scaled_num_data(n: int) -> float:
        return n / max_num
    colors = [greens(_scaled_num_data(n)) for n in data[2]]

    fig, ax = plt.subplots(figsize=(30*cm, 15*cm))
    bars = ax.bar(x, data[3], widths, colors)
    ax.set_title(title)
    ax.set_ylabel("MAE - ETA in seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(data[5], rotation=45, ha="right")
    # TODO: ax.set_ytickslabels mit formattierten etas
    # ax.bar_labels(bars, padding=3)
    for i, val in enumerate(data[3]):
        ax.text(x=i, y=val, s=data[4][i], ha="center", va="center")
    # ax.legend()
    fig.tight_layout()

    # for i, entry in enumerate(data):
    #     ax.text(x=i, y=entry[3], s=entry[4], ha="center", va="center")

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def plot_transfer_effect(base_data: List[Tuple[int, int, int, float, str, str]],
                         transfer_data: List[Tuple[int, int, int, float, str, str]],
                         port_name_base: str, port_name_target: str, path: str) -> None:
    assert len(base_data) == len(transfer_data)
    title = f"Transfer effect on MAE\nBase: {port_name_base} -> Target: {port_name_target}"
    base_data = list(map(list, zip(*base_data)))
    transfer_data = list(map(list, zip(*transfer_data)))
    x = np.arange(len(base_data[0]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(30*cm, 15*cm))
    base_bars = ax.bar(x - width/2, base_data[3])
    transfer_bars = ax.bar(x + width/2, transfer_data[3])
    ax.set_title(title)
    ax.set_ylabel("MAE - ETA in seconds")
    ax.set_xticks(x)
    ax.set_xtickslabels(base_data[5], rotattion=45, ha="right")
    for i in range(len(base_data[3])):
        ax.text(x=i - width/2, y=base_data[3][i], s=base_data[4][i], ha="center", va="center")
        ax.text(x=i + width/2, y=transfer_data[3][i], s=transfer_data[4][i], ha="center", va="center")
    fig.tight_layout()

    plt.savefig(path)


def plot_transfer_effects(avg_base_mae: List[float], avg_transfer_mae: List[float],
                          transfer_port_names: List[str], path: str) -> None:
    assert len(avg_base_mae) == len(avg_transfer_mae) == len(transfer_port_names)
    title = f"Average transfer effects on ports"
    x = np.arange(len(avg_base_mae)*3)
    width = .35

    fig, ax = plt.subplots(figsize=(30*cm, 15*cm))
    ax.minorticks_on()
    base_bars = ax.bar(x - width/2, avg_base_mae)
    transfer_bars = ax.bar(x - width/2, avg_transfer_mae)
    ax.set_title(title)
    ax.set_ylabel("NAE - ETA in seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(transfer_port_names, ha="right")
    fig.tight_layout()

    plt.savefig(path)


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
