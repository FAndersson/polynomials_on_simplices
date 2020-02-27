"""Projection routines."""

import numpy as np

from polynomials_on_simplices.calculus.affine_map import create_affine_map
from polynomials_on_simplices.calculus.real_interval import in_closed_range
from polynomials_on_simplices.linalg.basis import gram_schmidt_orthonormalization_rn


def interval_projection(p, interval):
    r"""
    Project a point in :math:`\mathbb{R}` to an interval [a, b].

    :param p: Point to project (scalar).
    :param interval: Interval to project to (pair of two scalars).
    :return: Point in [a, b] closest to p (scalar).
    """
    a, b = interval
    if in_closed_range(p, a, b):
        return p
    if p > b:
        return b
    return a


def vector_projection(a, b):
    """
    Orthogonal projection of vector a on b.

    :param a: Vector to project.
    :param b: Vector to project onto.
    :return: Projection of vector a on b.
    """
    return np.dot(a, b) / np.dot(b, b) * b


def vector_rejection(a, b):
    """
    Rejection of vector a from b (the component of a orthogonal to b).

    :param a: Vector to reject.
    :param b: Vector to reject from.
    :return: Rejection of vector a from b.
    """
    return a - vector_projection(a, b)


def vector_oblique_projection_2(a, b, c):
    """
    Projection of vector a on b, along vector c (oblique projection).
    Only applicable to two dimensional vectors.

    :param a: Vector to project.
    :param b: Vector to project onto.
    :param c: Vector defining the direction to project along.
    :return: Oblique projection of a onto b along c.
    """
    # Range of projection
    ma = b
    # Orthogonal complement of the projection null space
    mb = np.dot(np.array([[0, -1], [1, 0]]), c)
    # Projection matrix
    mp = np.outer(ma, mb) / np.dot(ma, mb)
    return np.dot(mp, a)


def vector_plane_projection(v, n):
    """
    Projection of a vector onto a plane with normal n.

    :param v: Vector to project onto the plane.
    :param n: Normal of the plane.
    :return: Vector projected onto the plane.
    """
    return vector_rejection(v, n)


def subspace_projection_transformation(basis, origin=None):
    r"""
    Generate the affine transformation :math:`P_V : \mathbb{R}^n \to \mathbb{R}^n, P(x) = Ax + b` which projects
    a point :math:`x \in \mathbb{R}^n` onto an m-dimensional vector subspace V of :math:`\mathbb{R}^n, m \leq n`.

    We have

    .. math:: A = BB^T \in \mathbb{R}^{n \times n},

    .. math:: b = (I - BB^T) o_V \in \mathbb{R}^n,

    where :math:`B` is the Gram-Schmidt orthonormalization of the basis spanning V and :math:`o_V` is the origin of
    V.

    :param basis: Basis for the subspace V (matrix with the basis vectors as columns).
    :type basis: Element in :math:`\mathbb{R}^{n \times m}`.
    :param origin: Origin of the subspace V. Optional, the n-dimensional zero-vector is used if not specified.
    :type origin: Element in :math:`\mathbb{R}^n`.
    :return: Tuple of A and b.
    """
    n, m = basis.shape
    assert m <= n
    if origin is None:
        origin = np.zeros(n)
    b = gram_schmidt_orthonormalization_rn(basis)
    a = np.dot(b, b.T)
    b = np.dot(np.identity(n) - a, origin)
    return a, b


def subspace_projection_map(basis, origin=None):
    r"""
    Generate the affine map :math:`P_V : \mathbb{R}^n \to \mathbb{R}^n, P(x) = Ax + b` which projects
    a point :math:`x \in \mathbb{R}^n` onto an m-dimensional vector subspace V of :math:`\mathbb{R}^n, m \leq n`.

    We have

    .. math:: A = BB^T \in \mathbb{R}^{n \times n},

    .. math:: b = (I - BB^T) o_V \in \mathbb{R}^n,

    where :math:`B` is the Gram-Schmidt orthonormalization of the basis spanning V and :math:`o_V` is the origin of
    V.

    :param basis: Basis for the subspace V (matrix with the basis vectors as columns).
    :type basis: Element in :math:`\mathbb{R}^{n \times m}`.
    :param origin: Origin of the subspace V. Optional, the n-dimensional zero-vector is used if not specified.
    :type origin: Element in :math:`\mathbb{R}^n`.
    :return: Function which takes an n-dimensional vector as input and returns an n-dimensional vector in V.
    :rtype: Callable :math:`P_V(x)`.
    """
    a, b = subspace_projection_transformation(basis, origin)
    return create_affine_map(a, b)
