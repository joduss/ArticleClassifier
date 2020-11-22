from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plotLosses(x, yAndLabels, fig=None):
    if fig is None:
        fig = plt.figure()

    plt.clf()
    plt.ion()
    ax = fig.subplots()

    i = 0
    for y in yAndLabels:
        ax.plot(x, y[0], label=y[1])
        i = i+1

    plt.xticks(x)
    fig.autofmt_xdate()

    ax.set_title("Losses")
    ax.legend()

    plt.get_current_fig_manager().show()
    plt.pause(0.05)

    return fig


def plot_score_per_theme(x, vector: List[float], ):

    fig: Figure = plt.figure("plot_score_per_theme")

    plt.clf()
    plt.ion()

    f, ax = fig.subplots(ncols=col + 1, nrows=row + 1)
    ax[col, row].plot(x, vector, label=legend)

    plt.xticks(x)
    fig.autofmt_xdate()

    ax.set_title(title)
    ax.legend()

    plt.get_current_fig_manager().show()
    plt.pause(0.05)

    return fig