"""Functionality for plotting triangles."""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa  # Avoid warning about unused import (req. for 3d plots to work)
import numpy as np

from polynomials_on_simplices.visualization.plot_lines import plot_space_curve


def plot_triangle_mesh(triangles, vertices, *args, **kwargs):
    """
    Plot a triangle mesh.

    :param triangles: The triangles in the mesh (num_triangles by 3 array of integers).
    :param vertices: The vertices in the mesh (num_vertices by [23] array of floats).
    :param args: Further arguments passed to the plot command.
    :param kwargs: Keyword arguments. 'fig': Figure to populate. A new figure will be generated
        if this argument is not given. Further keyword arguments are passed to the plot command.
    """
    if vertices.shape[1] == 2:
        plot_triangle_mesh_2d(triangles, vertices, *args, **kwargs)
    else:
        plot_triangle_mesh_3d(triangles, vertices, *args, **kwargs)


def plot_triangle_mesh_2d(triangles, vertices, *args, **kwargs):
    """
    Plot a triangle mesh.

    :param triangles: The triangles in the mesh (num_triangles by 3 array of integers).
    :param vertices: The vertices in the mesh (num_vertices by 2 array of floats).
    :param args: Further arguments passed to the plot command.
    :param kwargs: Keyword arguments. 'fig': Figure to populate. A new figure will be generated
        if this argument is not given. Further keyword arguments are passed to the plot command.
    """
    fig = kwargs.pop("fig", None)
    unique = (fig is None)
    if fig is None:
        fig = plt.figure()
    c = np.zeros(len(vertices))
    kwargs["figure"] = fig
    plt.tripcolor(vertices[:, 0], vertices[:, 1], triangles, c, *args, **kwargs)
    kwargs.pop("figure")
    kwargs["fig"] = fig
    plot_triangle_mesh_wireframe_2d(triangles, vertices, *args, **kwargs)
    if unique:
        plt.show()


def plot_triangle_mesh_3d(triangles, vertices, *args, **kwargs):
    """
    Plot a triangle mesh.

    :param triangles: The triangles in the mesh (num_triangles by 3 array of integers).
    :param vertices: The vertices in the mesh (num_vertices by 3 array of floats).
    :param args: Further arguments passed to the plot command.
    :param kwargs: Keyword arguments. 'fig': Figure to populate. A new figure will be generated
        if this argument is not given. Further keyword arguments are passed to the plot command.
    """
    fig = kwargs.pop("fig", None)
    unique = (fig is None)
    if fig is None:
        fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], triangles, vertices[:, 2], *args, **kwargs)
    if unique:
        plt.show()


def plot_triangle_mesh_wireframe(triangles, vertices, *args, **kwargs):
    """
    Plot a triangle mesh as a wireframe.

    :param triangles: The triangles in the mesh (num_triangles by 3 array of integers).
    :param vertices: The vertices in the mesh (num_vertices by [23] array of floats).
    :param args: Further arguments passed to the plot command.
    :param kwargs: Keyword arguments passed to the plot command.
    """
    if vertices.shape[1] == 2:
        plot_triangle_mesh_wireframe_2d(triangles, vertices, *args, **kwargs)
    else:
        plot_triangle_mesh_wireframe_3d(triangles, vertices, *args, **kwargs)


def plot_triangle_mesh_wireframe_2d(triangles, vertices, *args, **kwargs):
    """
    Plot a triangle mesh as a wireframe.

    :param triangles: The triangles in the mesh (num_triangles by 3 array of integers).
    :param vertices: The vertices in the mesh (num_vertices by 2 array of floats).
    :param args: Further arguments passed to the plot command.
    :param kwargs: Keyword arguments. 'fig': Figure to populate. A new figure will be generated
        if this argument is not given. 'color': Color of the wireframe. Further keyword arguments are
        passed to the plot command.
    """
    fig = kwargs.pop("fig", None)
    color = kwargs.pop("color", u'k')
    unique = (fig is None)
    if fig is None:
        fig = plt.figure()
    kwargs["figure"] = fig
    args = (color + '-',) + args
    plt.triplot(vertices[:, 0], vertices[:, 1], triangles, *args, **kwargs)
    if unique:
        plt.show()


def plot_triangle_mesh_wireframe_3d(triangles, vertices, *args, **kwargs):
    """
    Plot a triangle mesh as a wireframe.

    :param triangles: The triangles in the mesh (num_triangles by 3 array of integers).
    :param vertices: The vertices in the mesh (num_vertices by 3 array of floats).
    :param args: Further arguments passed to the plot command.
    :param kwargs: Keyword arguments. 'fig': Figure to populate. A new figure will be generated
        if this argument is not given. 'color': Color of the wireframe. Further keyword arguments are
        passed to the plot command.
    """
    fig = kwargs.pop("fig", None)
    color = kwargs.pop("color", u'k')
    unique = (fig is None)
    if fig is None:
        fig = plt.figure()
    kwargs["fig"] = fig
    args += (color + '-',)
    for i in range(len(triangles)):
        for j in range(3):
            line_vertices = np.empty((2, 3))
            v0 = triangles[i][j]
            v1 = triangles[i][(j + 1) % 3]
            line_vertices[0] = vertices[v0]
            line_vertices[1] = vertices[v1]
            plot_space_curve(line_vertices, *args, **kwargs)
    if unique:
        plt.show()
