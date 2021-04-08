import matplotlib.pyplot as plt

from matplotlib import ticker
from typing import List

series_colors = ["b", "r", "g"]


def plot_series(series: List[List[float]], x_label: str, y_label: str, title: str = None,
                legend_labels: List[str] = None, path: str = None) -> None:
    num_series = len(series)
    if legend_labels is not None:
        assert len(legend_labels) == num_series
    plt.switch_backend("agg")   # set backend explicitly to run on Jupyter notebooks
    fix, ax = plt.subplots()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.2))

    for i in range(num_series):
        ax.plot([series_i[i] for series_i in series])
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
