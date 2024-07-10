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
    figure_args: FigureArguments = FigureArguments(suptitle="Default Plot Groups")
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
        figure_args: FigureArguments
            Used to arrange the figure of the plot

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
    figure, axs = plt.subplots(
        len(ys), len(xs), 
        sharex=True, sharey=True, squeeze=False
    )
    _apply_figure_arguments(figure_args, figure)

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


def plot_images(
    image_groups: List[List[ndarray]],
    group_labels: List[str],
    figure_args: FigureArguments = FigureArguments(suptitle="Default Plot Groups")
):
    """Plot groups of data.

    Plot images in groups split up by columns

    Arguments:
        images: List[List[Image]]
            Groups of images
        group_labels: List[str]
            The x label for each collumn of graphs
        **extra_args: FigureArguments
            The rest of the arguments will be used to make arguments for the figure

    Requirements:
    - Each image must be in "image shape" or (height, width, channel)

    Notes:
    - If an image has three channels it will be plotted in color, one channel implies greyscale
    
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
    max_rows = max(len(group) for group in image_groups)
    figure, axs = plt.subplots(
        max_rows, len(group_labels),
        sharex=True, sharey=True, squeeze=False
    )
    _apply_figure_arguments(figure_args, figure)

    for col in range(len(group_labels)):
        axs[-1][col].set_xlabel(group_labels[col])
    # Apply Images!
    for group_num in range(len(image_groups)):
        for row in range(len(image_groups[group_num])):
            image_data = image_groups[group_num][row]
            assert len(image_data.shape) == 3, \
                f"Image must have 3 dims but got {image_data.shape}"
            if image_data.shape[-1] == 1:
                # Greyscale
                axs[row][group_num].imshow(
                    image_data, 
                    cmap='gray', vmin=image_data.min(), vmax=image_data.max()
                )
            elif image_data.shape[-1] == 3:
                # Color
                axs[row][group_num].imshow(
                    image_data,
                    vmin=image_data.min(), vmax=image_data.max()
                )
            else:
                raise ValueError(f"Image channels must be 1 or 3, found:{image_data.shape[-1]}")