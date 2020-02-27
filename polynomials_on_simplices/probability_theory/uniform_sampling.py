"""Uniform sampling of random points in different geometries."""

import numpy as np


def closed_unit_interval_sample():
    """
    Generate a random number sampled from the uniform distribution over the closed unit interval [0, 1].

    :return: Random number.
    """
    x = np.random.rand()
    while x > 0.875:
        x = np.random.rand()
    # x is now a random number in the interval [0, 0.875]
    return x / 0.875


def open_unit_interval_sample():
    """
    Generate a random number sampled from the uniform distribution over the open unit interval (0, 1).

    :return: Random number.
    """
    x = np.random.rand()
    while x <= 0.125:
        x = np.random.rand()
    # x is now a random number in the interval (0.125, 1)
    x -= 0.125
    # x is now a random number in the interval (0, 0.875)
    return x / 0.875


def left_closed_interval_sample():
    """
    Generate a random number sampled from the uniform distribution over the left closed unit interval [0, 1).

    :return: Random number.
    """
    return np.random.rand()


def right_closed_interval_sample():
    """
    Generate a random number sampled from the uniform distribution over the right closed unit interval (0, 1].

    :return: Random number.
    """
    x = np.random.rand()
    x *= -1
    # x is now a random number in the interval (-1, 0]
    x += 1
    return x


def unit_interval_sampling(num_points):
    """
    Uniform random sampling of points in the unit interval [0, 1).

    :param num_points: Number of points to sample.
    :return: List of random points.
    """
    return np.random.rand(num_points)


def closed_unit_interval_sampling(num_points):
    """
    Uniform random sampling of points in the closed unit interval [0, 1].

    :param num_points: Number of points to sample.
    :return: List of random points.
    """
    points = np.empty(num_points)
    for i in range(num_points):
        points[i] = closed_unit_interval_sample()
    return points


def open_unit_interval_sampling(num_points):
    """
    Uniform random sampling of points in the unit interval (0, 1).

    :param num_points: Number of points to sample.
    :return: List of random points.
    """
    points = np.empty(num_points)
    for i in range(num_points):
        points[i] = open_unit_interval_sample()
    return points


def left_closed_unit_interval_sampling(num_points):
    """
    Uniform random sampling of points in the unit interval [0, 1).

    :param num_points: Number of points to sample.
    :return: List of random points.
    """
    return unit_interval_sampling(num_points)


def right_closed_unit_interval_sampling(num_points):
    """
    Uniform random sampling of points in the unit interval (0, 1].

    :param num_points: Number of points to sample.
    :return: List of random points.
    """
    points = np.empty(num_points)
    for i in range(num_points):
        points[i] = right_closed_interval_sample()
    return points


def unit_square_sampling(num_points):
    """
    Uniform random sampling of points in the unit square [0, 1) x [0, 1).

    :param num_points: Number of points to sample.
    :return: List of random points.
    """
    xy = np.random.rand(2 * num_points)
    return np.reshape(xy, (-1, 2))


def unit_disc_sampling(num_points):
    r"""
    Uniform random sampling of points in the unit disc :math:`\{x \in \mathbb{R}^2 : \|x\| \leq 1\}`.

    :param num_points: Number of points to sample.
    :return: List of random points.
    """
    sample_points = np.empty((num_points, 2))
    accepted_samples = 0
    while accepted_samples < num_points:
        # Rejection sampling. Sample in square [-1, 1]x[-1, 1] and reject the sample if it's norm is greater than one
        sample = np.random.rand(2) * 2 - 1
        if not np.dot(sample, sample) > 1:
            sample_points[accepted_samples] = sample
            accepted_samples += 1
    return sample_points


def unit_circle_sampling(num_points):
    """
    Uniform random sampling of points on the unit circle (:math:`S^1`).

    :param num_points: Number of points to sample.
    :return: List of random points.
    """
    return nsphere_surface_sampling(2, num_points)


def ncube_sampling(n, num_points):
    r"""
    Uniform random sampling of points in the n-dimensional unit cube :math:`[0, 1)^n`.

    :param n: Dimension of cube.
    :param num_points: Number of points to sample.
    :return: List of random points.
    """
    points = np.random.rand(n * num_points)
    return np.reshape(points, (num_points, n))


def nsphere_sampling(n, num_points):
    r"""
    Uniform random sampling of points in the n-dimensional unit sphere :math:`\{x \in \mathbb{R}^n : \|x\| \leq 1\}`.

    :param n: Dimension of sphere.
    :param num_points: Number of points to sample.
    :return: List of random points.
    """
    sample_points = np.empty((num_points, n))
    accepted_samples = 0
    while accepted_samples < num_points:
        # Rejection sampling. Sample in unit cube and reject the sample if it's norm is greater than one
        sample = np.random.rand(n) * 2 - 1
        if not np.dot(sample, sample) > 1:
            sample_points[accepted_samples] = sample
            accepted_samples += 1
    return sample_points


def nsphere_surface_sampling(n, num_points):
    r"""
    Uniform random sampling of points on the surface of the n-dimensional unit sphere (:math:`\partial B^n = S^{n-1}`).

    :param n: Dimension of sphere.
    :param num_points: Number of points to sample.
    :return: List of random points.
    """
    # See http://mathworld.wolfram.com/SpherePointPicking.html (Muller 1959, Marsaglia 1972)
    sample_points = np.random.randn(num_points, n)
    for i in range(num_points):
        sample_points[i, :] /= np.linalg.norm(sample_points[i, :])
    return sample_points


def nsimplex_sampling(n, num_points):
    """
    Uniform random sampling of points inside the n-dimensional unit simplex.

    See :func:`polynomials_on_simplices.geometry.primitives.simplex.unit()`.

    :param n: Dimension of the simplex.
    :param num_points: Number of points to sample.
    :return: List of random points.
    """
    sample_points = np.empty((num_points, n))
    for i in range(num_points):
        sample_points[i] = ncube_sampling(n, 1)[0]
        while sum(sample_points[i]) > 1.0:
            sample_points[i] = ncube_sampling(n, 1)[0]
    return sample_points
