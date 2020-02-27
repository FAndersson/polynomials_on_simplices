"""Functionality for plotting Lagrange finite elements (continuous piecewise polynomials) on a simplicial domain
(triangulation).
"""

import matplotlib.pyplot as plt

from polynomials_on_simplices.geometry.mesh.basic_meshes.triangle_meshes import (
    triangle_mesh_triangulation, triangle_mesh_vertices)
from polynomials_on_simplices.geometry.mesh.point_clouds import embed_point_cloud_in_rn
from polynomials_on_simplices.visualization.plot_triangles import plot_triangle_mesh_3d


def plot_bivariate_continuous_piecewise_polynomial(p, *args, **kwargs):
    """
    Plot a bivariate scalar valued continuous piecewise polynomial.

    :param p: Continuous piecewise polynomial that we want to plot.
    :type p: Implementation of
        :class:`~polynomials_on_simplices.piecewise_polynomial.continuous_piecewise_polynomial.ContinuousPiecewisePolynomialBase`
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
    edge_resolution = kwargs.pop("edge_resolution", 50)
    triangles = triangle_mesh_triangulation(p.triangles, edge_resolution)
    vertices = triangle_mesh_vertices(p.triangles, p.vertices, edge_resolution)
    vertices = embed_point_cloud_in_rn(vertices, 3)
    for i in range(len(vertices)):
        x = vertices[i][0:2]
        vertices[i][2] = p(x)

    plot_triangle_mesh_3d(triangles, vertices, *args, fig=fig, **kwargs)

    if unique:
        plt.show()
