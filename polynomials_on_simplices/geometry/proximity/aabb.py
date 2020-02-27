"""Routines for dealing with axis aligned bounding boxes (AABB).

An AABB is represented with two n:d vectors giving the min and max point of the AABB,
i.e.,
aabb = (min, max) = ((x_min, y_min, z_min), (x_max, y_max, z_max)) (for an AABB in 3D).
"""

import math

import numpy as np

from polynomials_on_simplices.calculus.real_interval import in_closed_range


def empty(n=3):
    """
    Create an empty AABB.

    :param n: Dimension of the AABB.
    :return: Empty AABB.
    """
    return float("inf") * np.ones(n), float("-inf") * np.ones(n)


def unit(n=3):
    """
    Create the unit AABB (side lengths 1 and with the min point in the origin).

    :param n: Dimension of the AABB.
    :return: The unit AABB.
    """
    return np.zeros(n), np.ones(n)


def full(n=3):
    r"""
    Create an AABB covering all of :math:`\mathbb{R}^n`.

    :param n: Dimension of the AABB.
    :return: The AABB containing all of :math:`\mathbb{R}^n`.

    .. rubric:: Examples

    >>> full(1)
    (array([-inf]), array([inf]))
    >>> full(2)
    (array([-inf, -inf]), array([inf, inf]))
    """
    return float("-inf") * np.ones(n), float("inf") * np.ones(n)


def create(points):
    """
    Create the minimum AABB containing a set of points.

    :param points: Array of points (num_of_points by n).
    :return: Minimum AABB including all input points.
    """
    min_point = np.amin(points, axis=0)
    max_point = np.amax(points, axis=0)
    return min_point, max_point


def is_empty(aabb):
    """
    Check if an AABB is empty.

    .. note::

        An AABB is empty if max[i] < min[i] for some i in [0, 1, 2].
        An AABB with min = max is not considered empty since it contains a single point.

    :param aabb: AABB defined by its min and max point.
    :type aabb: Pair of n-dimensional vectors
    :return: True/False depending on whether or not the AABB is empty.
    """
    return volume(aabb) is None


def is_valid(aabb):
    """
    Check if an AABB is valid (has valid min and max points, and min[i] <= max[i] for all i in [0, 1, 2]).

    :param aabb: AABB defined by its min and max point.
    :type aabb: Pair of n-dimensional vectors
    :return: True/False depending on whether or not the AABB is valid.
    """
    for i in range(dimension(aabb)):
        if aabb[0][i] > aabb[1][i]:
            return False
    return True


def is_equal(aabb1, aabb2, rel_tol=1e-9, abs_tol=1e-7):
    """
    Check if two AABBs aabb1 and aabb2 are approximately equal.

    The two AABBs are considered equal if the min and max points of the AABBs are componentwise approximately equal.
    For the componentwise equality check, the standard function :func:`math.isclose <python:math.isclose>` is used,
    with the given relative and absolute tolerances.
    """
    for i in range(len(aabb1[0])):
        if not math.isclose(aabb1[0][i], aabb2[0][i], rel_tol=rel_tol, abs_tol=abs_tol):
            return False
        if not math.isclose(aabb1[1][i], aabb2[1][i], rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True


def dimension(aabb):
    """
    Get the dimension of an AABB.

    :param aabb: AABB defined by its min and max point.
    :type aabb: Pair of n-dimensional vectors
    :return: Dimension of the AABB.
    """
    return len(aabb[0])


def midpoint(aabb):
    """
    Compute the midpoint of an AABB.

    :param aabb: AABB defined by its min and max point.
    :type aabb: Pair of n-dimensional vectors
    :return: Midpoint of the AABB (nd vector).
    """
    if not is_valid(aabb):
        return None
    n = dimension(aabb)
    mp = np.empty(n)
    for i in range(n):
        mp[i] = 0.5 * (aabb[0][i] + aabb[1][i])
    return mp


def half_diagonal(aabb):
    """
    Compute the half-diagonal of an AABB.

    :param aabb: AABB defined by its min and max point.
    :type aabb: Pair of n-dimensional vectors
    :return: Half-diagonal of the AABB (vector from the AABB midpoint to the max point).
    """
    if not is_valid(aabb):
        return None
    n = dimension(aabb)
    hd = np.empty(n)
    for i in range(n):
        hd[i] = 0.5 * (aabb[1][i] - aabb[0][i])
    return hd


def diameter(aabb):
    """
    Compute the length of the diameter of an AABB.

    :param aabb: AABB defined by its min and max point.
    :type aabb: Pair of n-dimensional vectors
    :return: Length of the diameter of the AABB.
    """
    if not is_valid(aabb):
        return None
    return np.linalg.norm(aabb[1] - aabb[0])


def volume(aabb):
    """
    Compute the volume of an AABB.

    :param aabb: AABB defined by its min and max point.
    :type aabb: Pair of n-dimensional vectors
    :return: Volume of the AABB.
    """
    if not is_valid(aabb):
        return None
    n = dimension(aabb)
    vol = 1
    for i in range(n):
        vol *= aabb[1][i] - aabb[0][i]
    return vol


def union(aabb1, aabb2):
    """
    Compute the union of two AABBs.

    :param aabb1: First AABB.
    :param aabb2: Second AABB.
    :return: Minimum AABB including all points in either aabb1 or aabb2.
    """
    n = dimension(aabb1)
    aabb = empty(n)
    for i in range(n):
        aabb[0][i] = min(aabb1[0][i], aabb2[0][i])
        aabb[1][i] = max(aabb1[1][i], aabb2[1][i])
    return aabb


def intersection(aabb1, aabb2):
    """
    Compute the intersection of two AABBs.

    :param aabb1: First AABB.
    :param aabb2: Second AABB.
    :return: Minimum AABB including all points in both aabb1 and aabb2.
    """
    n = dimension(aabb1)
    aabb = empty(n)
    for i in range(n):
        aabb[0][i] = max(aabb1[0][i], aabb2[0][i])
        aabb[1][i] = min(aabb1[1][i], aabb2[1][i])
    return aabb


def corner(aabb, i):
    """
    Get the i:th corner of the AABB. Corners are ordered based on their x_n coordinate, with the x_{n-1}
    coordinate as first tiebreaker, x_{n-2} coordinate as second tiebreaker, and so on.
    For example the corners of a 3D AABB are ordered on their z-value, with y-value as first tiebreaker,
    and x-value as second tiebreaker.

    :param aabb: AABB defined by its min and max point.
    :type aabb: Pair of n-dimensional vectors
    :param i: Index of the corner (in the range 0, 1, ..., 2^n-1 where n is the dimension of the AABB).
    :return: AABB corner (nD vector).
    """
    n = dimension(aabb)
    if n == 1:
        return aabb[i]
    if i >= 2**(n - 1):
        upc = corner((aabb[0][0:n - 1], aabb[1][0:n - 1]), i - (2**(n - 1)))
        return np.append(upc, np.array(aabb[1][n - 1]))
    else:
        lpc = corner((aabb[0][0:n - 1], aabb[1][0:n - 1]), i)
        return np.append(lpc, np.array(aabb[0][n - 1]))


def point_inside(aabb, p):
    """
    Check if a point is inside an AABB.

    :param aabb: AABB defined by its min and max point.
    :type aabb: Pair of n-dimensional vectors
    :param p: Point (n dimensional vector).
    :return: True/False whether or not the point is inside the AABB.
    """
    n = len(p)
    for i in range(n):
        if not in_closed_range(p[i], aabb[0][i], aabb[1][i]):
            return False
    return True
