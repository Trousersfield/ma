import matplotlib.pyplot as plt

from matplotlib import ticker
from typing import List


def plot_loss(loss_series: List[List[float]], labels: List[str] = None, path: str = None) -> None:
    num_loss_series = len(loss_series)
    assert len(labels) == num_loss_series
    plt.switch_backend("agg")   # explicit set backend to run on Jupyter notebooks
    fix, ax = plt.subplots()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.2))

    for i in range(num_loss_series):
        ax.plot([value[i] for value in loss_series])
    ax.legend(labels)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss History")

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
