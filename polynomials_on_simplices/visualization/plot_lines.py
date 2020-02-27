"""Functionality for plotting lines."""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa  # Avoid warning about unused import (req. for 3d plots to work)


def plot_curve(vertices, *args, **kwargs):
    """
    Plot a discrete curve consisting of a set of connected vertices.

    :param vertices: The curve vertices (num_vertices by [23] array of floats).
    :param args: Further arguments passed to the plot command.
    :param kwargs: Keyword arguments. 'fig': Figure to populate. A new figure will be generated
        if this argument is not given. Further keyword arguments are passed to the plot command.
    """
    if vertices.shape[1] == 2:
        return plot_plane_curve(vertices, *args, **kwargs)
    if vertices.shape[1] == 3:
        return plot_space_curve(vertices, *args, **kwargs)
    raise ValueError("Can only plot curves in 2d or 3d")


def plot_space_curve(vertices, *args, **kwargs):
    """
    Plot a discrete space curve consisting of a set of connected vertices.

    :param vertices: The curve vertices (num_vertices by 3 array of floats).
    :param args: Further arguments passed to the plot command.
    :param kwargs: Keyword arguments. 'fig': Figure to populate. A new figure will be generated
        if this argument is not given. Further keyword arguments are passed to the plot command.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    fig = kwargs.pop("fig", None)
    unique = (fig is None)
    if fig is None:
        fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, zs=z, *args, **kwargs)
    if unique:
        plt.show()


def plot_plane_curve(vertices, *args, **kwargs):
    """
    Plot a discrete plane curve consisting of a set of connected vertices.

    :param vertices: The curve vertices (num_vertices by 2 array of floats).
    :param args: Further arguments passed to the plot command.
    :param kwargs: Keyword arguments. 'fig': Figure to populate. A new figure will be generated
        if this argument is not given. Further keyword arguments are passed to the plot command.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    fig = kwargs.pop("fig", None)
    unique = (fig is None)
    if fig is None:
        plt.figure()
    plt.plot(x, y, *args, **kwargs)
    if unique:
        plt.show()
