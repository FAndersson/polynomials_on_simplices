"""Functionality for plotting discontinuous Galerkin finite elements (piecewise polynomials) on a simplicial domain
(triangulation).
"""

import matplotlib.pyplot as plt
import numpy as np

from polynomials_on_simplices.geometry.mesh.basic_meshes.triangle_meshes import (
    general_triangle_vertices, triangle_triangulation)
from polynomials_on_simplices.geometry.mesh.point_clouds import embed_point_cloud_in_rn
import polynomials_on_simplices.geometry.mesh.triangle_mesh as tri_mesh
from polynomials_on_simplices.visualization.plot_lines import plot_plane_curve
from polynomials_on_simplices.visualization.plot_triangles import plot_triangle_mesh_3d


def plot_univariate_piecewise_polynomial(p, *args, **kwargs):
    """
    Plot a univariate scalar valued piecewise polynomial.

    :param p: Piecewise polynomial that we want to plot.
    :type p: Implementation of
        :class:`~polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial.PiecewisePolynomialBase`
    :param args: Additional arguments passed to the plot command.
    :param kwargs: Keyword arguments. 'edge_resolution': Number of discrete points in the plot on each line in the mesh
        the piecewise polynomial is defined on. Further keyword arguments are passed to the plot_curve command.
    """
    assert p.domain_dimension() == 1
    assert p.target_dimension() == 1

    fig = kwargs.pop("fig", None)
    unique = (fig is None)
    if fig is None:
        fig = plt.figure()
    edge_resolution = kwargs.pop("edge_resolution", 15)
    t = np.linspace(0, 1, edge_resolution)
    c = kwargs.pop("color", np.random.rand(3))

    for line_idx in range(len(p.triangles)):
        line = p.triangles[line_idx]
        v0 = p.vertices[line[0]]
        v1 = p.vertices[line[1]]
        try:
            v0 = v0[0]
            v1 = v1[0]
        except IndexError:
            pass
        vertices = np.empty((edge_resolution, 2))
        for i in range(len(t)):
            x = (1 - t[i]) * v0 + t[i] * v1
            vertices[i][0] = x
            vertices[i][1] = p.evaluate_on_simplex(line_idx, x)

        plot_plane_curve(vertices, *args, color=c, fig=fig, **kwargs)

    if unique:
        plt.show()


def plot_bivariate_piecewise_polynomial(p, *args, **kwargs):
    """
    Plot a bivariate scalar valued piecewise polynomial.

    :param p: Piecewise polynomial that we want to plot.
    :type p: Implementation of
        :class:`~polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial.PiecewisePolynomialBase`
    :param args: Additional arguments passed to the plot command.
    :param kwargs: Keyword arguments. 'edge_resolution': Number of discrete points in the plot along each edge in the
        mesh the piecewise polynomial is defined on. Further keyword arguments are passed to the plot_curve command.
    """
    assert p.domain_dimension() == 2
    assert p.target_dimension() == 1

    fig = kwargs.pop("fig", None)
    unique = (fig is None)
    if fig is None:
        fig = plt.figure()
    edge_resolution = kwargs.pop("edge_resolution", 15)
    triangles = triangle_triangulation(edge_resolution)
    c = np.random.rand(3)

    for tri_idx in range(len(p.triangles)):
        triangle = p.triangles[tri_idx]
        plot_vertices = general_triangle_vertices(tri_mesh.vertices(triangle, p.vertices), edge_resolution)
        plot_vertices = embed_point_cloud_in_rn(plot_vertices, 3)
        for i in range(len(plot_vertices)):
            x = plot_vertices[i, 0:2]
            plot_vertices[i][2] = p.evaluate_on_simplex(tri_idx, x)

        plot_triangle_mesh_3d(triangles, plot_vertices, *args, color=c, fig=fig, **kwargs)

    if unique:
        plt.show()
