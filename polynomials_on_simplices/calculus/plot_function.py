"""Plotting of analytic expressions (in the form of callables).
"""

import numpy as np

from polynomials_on_simplices.geometry.mesh.basic_meshes.triangle_meshes import (
    general_triangle_vertices, triangle_triangulation)
import polynomials_on_simplices.visualization.plot_lines
import polynomials_on_simplices.visualization.plot_triangles


def plot_function(f, x_start=0.0, x_stop=1.0, *args, **kwargs):
    """
    Plot a univariate function.

    :param f: Function expression to plot (univariate real-valued function).
    :type f: Callable f(x)
    :param float x_start: Beginning of x-range to plot.
    :param float x_stop: End of x-range to plot.
    :param args: Additional arguments passed to the :func:`plot_curve` command.
    :param kwargs: Keyword arguments passed to the :func:`plot_curve` command.
    """
    def g(x):
        return [x, f(x)]
    return plot_curve(g, x_start, x_stop, *args, **kwargs)


def evaluate_curve(gamma, t):
    r"""
    Evaluate a parametric curve in 2d (:math:`\gamma(t) = (\gamma_1(t), \gamma_2(t))`)
    or 3d (:math:`\gamma(t) = (\gamma_1(t), \gamma_2(t), \gamma_3(t))`) at a number of locations.

    :param gamma: Parametric curve (vector valued (2d or 3d) univariate function).
    :type gamma: Callable :math:`\gamma(t)`
    :param t: Positions where the curve should be evaluated (array of floats).
    :return: Array of curve values, one for each location in the input t array.
    """
    n = len(gamma(t[0]))
    vertices = np.empty((len(t), n))
    for i in range(len(t)):
        vertices[i] = gamma(t[i])
    return vertices


def plot_curve(gamma, t_start=0.0, t_stop=1.0, *args, **kwargs):
    r"""
    Plot a parametric curve in 2d (:math:`\gamma(t) = (\gamma_1(t), \gamma_2(t))`)
    or 3d (:math:`\gamma(t) = (\gamma_1(t), \gamma_2(t), \gamma_3(t))`).

    :param gamma: Parametric curve (vector valued (2d or 3d) univariate function).
    :type gamma: Callable :math:`\gamma(t)`
    :param float t_start: Beginning of range to plot for the curve parameter t.
    :param float t_stop: End of range to plot for the curve parameter t.
    :param args: Additional arguments passed to the :func:`~polynomials_on_simplices.visualization.plot_lines.plot_curve`
        command.
    :param kwargs: Keyword arguments. 'num': Number of discrete points in the plot. Further keyword arguments are
        passed to the :func:`~polynomials_on_simplices.visualization.plot_lines.plot_curve` command.
    """
    num = kwargs.pop("num", 50)
    t = np.linspace(t_start, t_stop, num=num)
    vertices = evaluate_curve(gamma, t)
    polynomials_on_simplices.visualization.plot_lines.plot_curve(vertices, *args, **kwargs)


def plot_bivariate_function(f, tri_vertices, *args, **kwargs):
    """
    Plot a bivariate function on a triangular domain.

    :param f: Bivariate function to plot (bivariate real-valued function).
    :type f: Callable f(x, y)
    :param tri_vertices: Vertices of the triangular domain (3 by 2 matrix where each row is a vertex in the triangle).
    :param args: Additional arguments passed to the
        :func:`~polynomials_on_simplices.visualization.plot_triangles.plot_triangle_mesh` command.
    :param kwargs: 'edge_resolution': Number of discrete points in the plot along each edge in the triangle mesh.
        Further keyword arguments are passed to the
        :func:`~polynomials_on_simplices.visualization.plot_triangles.plot_triangle_mesh` command.
    """
    # Generate discretization of the triangle
    edge_resolution = kwargs.pop("edge_resolution", 50)
    triangles = triangle_triangulation(edge_resolution)
    vertices = general_triangle_vertices(tri_vertices, edge_resolution)
    # Add third column for z-values
    vertices = np.concatenate((vertices, np.zeros((len(vertices), 1))), axis=1)
    # Compute z-values
    for i in range(len(vertices)):
        vertices[i][2] = f(vertices[i][0], vertices[i][1])
    polynomials_on_simplices.visualization.plot_triangles.plot_triangle_mesh(triangles, vertices, *args, **kwargs)
