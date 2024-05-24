"""
Different types of plots I often make
"""


from torch import Tensor
from numpy import ndarray
from matplotlib import cm
from typing import List, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class FigureArguments:
    """Class for keeping track of an item in inventory."""

    suptitle: str = "Default Title"
    width: float = 5.
    height: float = 3.


def _apply_figure_arguments(
    figure_arguments: FigureArguments,
    figure: plt.Figure
):
    figure.suptitle(figure_arguments.suptitle)
    figure.set_size_inches(figure_arguments.width, figure_arguments.height)


def plot_groups(
    xs: List[Union[List[float], Tensor, ndarray]],
    ys: List[Union[List[float], Tensor, ndarray]],
    x_labels: List[str],
    y_labels: List[str],
    color_map: str,
    **extra_args
):
    """Plot groups of data.

    Will return a figure which can be shown. Axes are shared. 

    Arguments:
        xs: List[array[float]]
            The x coordinates for each collumn of graphs
        ys: List[array[float]]
            The y coordinates for each row of graphs
        x_labels: List[str]
            The x label for each collumn of graphs
        y_labels: List[str]
            The y label for each row of graphs
        color_map: str
            The colormap's name used to choose colors
        **extra_args: FigureArguments
            The rest of the arguments will be used to make arguments for the figure

    Example:
        x = np.linspace(0, 2 * np.pi, 400)
        y = np.sin(x ** 2)
        plot_groups(
            xs=[x, -x],
            ys=[y, -y],
            x_labels=["x", "-x"],
            y_labels=["y","-y"],
            color_map="gist_rainbow_r"
        )
    """
    figure, axs = plt.subplots(len(ys), len(xs), sharex=True, sharey=True, squeeze=False)
    figure_arguments = FigureArguments(**extra_args)
    _apply_figure_arguments(figure_arguments, figure)

    # Set axes labels
    for row in range(len(ys)):
        axs[row][0].set_ylabel(y_labels[row])
    for col in range(len(xs)):
        axs[-1][col].set_xlabel(x_labels[col])

    # Plot data!
    max_count = len(ys) * len(xs)
    iteration = 0
    color_map = getattr(cm, color_map)
    for row in range(len(ys)):
        for col in range(len(xs)):
            
            axs[row][col].plot(xs[col], ys[row], color=color_map(iteration/max_count))
            iteration += 1