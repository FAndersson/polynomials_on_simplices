"""Functionality for dealing with bases."""

import copy

import numpy as np


def cartesian_basis(dimension=3):
    r"""
    Get the standard Cartesian basis in the specified dimension.

    :param int dimension: Dimension of Euclidean space.
    :return: The identity matrix in :math:`\mathbb{R}^n`, where n is the desired dimension.
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    >>> cartesian_basis(2)
    array([[1., 0.],
           [0., 1.]])
    """
    return np.identity(dimension)


def cartesian_basis_vector(i, dimension=3):
    """
    Get the i:th standard Cartesian basis vector in the specified dimension (3 by default).

    :param int i: Index of the basis vector (in [0, 1, ..., dimension - 1]).
    :param int dimension: Dimension of Euclidean space.
    :return: The i:th Cartesian basis vector.
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    >>> cartesian_basis_vector(1)
    array([0., 1., 0.])
    """
    return cartesian_basis(dimension=dimension)[i]


def basis_matrix(basis_vectors_array):
    """
    Construct the matrix representation of a basis from a list of basis vectors.

    :param basis_vectors_array: List of basis vectors.
    :return: Matrix representation of basis (square matrix with basis vectors as columns).
    """
    dim = len(basis_vectors_array)
    basis = np.empty((dim, dim))
    for i in range(dim):
        basis[:, i] = basis_vectors_array[i]
    return basis


def basis_vectors(basis):
    """
    Get a list of basis vectors from the matrix representation of a basis.

    :param basis: Matrix representation of basis (square matrix with basis vectors as columns).
    :return: List of basis vectors.
    """
    return basis.T


def metric_tensor(basis):
    """
    Compute the metric tensor for a basis.

    :param basis: Basis (square matrix with basis vectors as columns).
    :return: Metric tensor.
    """
    if isinstance(basis, np.ndarray):
        return np.dot(basis.T, basis)
    b = np.array(basis)
    return np.dot(b.T, b)


def inverse_metric_tensor(basis):
    """
    Compute the inverse metric tensor for a basis.

    :param basis: Basis (square matrix with basis vectors as columns).
    :return: Inverse metric tensor.
    """
    g = metric_tensor(basis)
    if not isinstance(g, np.ndarray):
        # Assume scalar
        return 1 / g
    return np.linalg.inv(g)


def dual_basis(basis):
    r"""
    Compute the dual basis of a given basis, i.e., the basis :math:`e^i` such that
    :math:`\langle e_i, e^j \rangle = \delta_i^j`.

    :param basis: Basis (square matrix with basis vectors as columns).
    :return: Dual basis (square matrix with dual basis vectors as columns).
    """
    if not isinstance(basis, np.ndarray):
        # Assume scalar
        return 1 / basis
    # B^T * B_d = I => B_d = (B^T)^-1 = (B^-1)^T = B^-T
    return np.linalg.inv(basis).T


def transform_to_basis(v, basis):
    """
    Transform a vector or set of vectors from Cartesian coordinates to the supplied
    coordinate system.

    :param v: Vector(s) expressed in Cartesian coordinates (vector or matrix with vectors as columns).
    :param basis: Basis to express the vector in (matrix with basis vectors as columns).
    :return: Vector representation(s) in the supplied basis.
    """
    # v = b * u => u = b^+ v
    return np.dot(np.linalg.pinv(basis), v)


def transform_from_basis(v, basis):
    """
    Transform a vector or set of vectors in a given coordinate system to
    Cartesian coordinates.

    :param v: Vector(s) expressed in the supplied basis (vector or matrix with vectors as columns).
    :param basis: Basis the vector is expressed in (matrix with basis vectors as columns).
    :return: Vector(s) in Cartesian coordinates.
    """
    return np.dot(basis, v)


def transform(v, basis0, basis1):
    """
    Transform a vector or set of vectors from one basis to another.

    :param v: Representation of the vector(s) in the initial basis (vector or matrix with vectors as columns).
    :param basis0: Initial basis to transform from (matrix with basis vectors as columns).
    :param basis1: Final basis to transform to (matrix with basis vectors as columns).
    :return: Representation of the vector(s) in the final basis.
    """
    return np.dot(np.linalg.pinv(basis1), np.dot(basis0, v))


def transform_bilinear_form(b, basis0, basis1):
    """
    Transform a bilinear form (matrix) from one basis to another.

    :param b: Representation of the bilinear form in the initial basis (n by n matrix).
    :param basis0: Initial basis to transform from (matrix with basis vectors as columns).
    :param basis1: Final basis to transform to (matrix with basis vectors as columns).
    :return: Representation of the bilinear form in the final basis (n by n matrix).
    """
    if np.abs(np.linalg.det(basis0) - 1.0) > 1e-10:
        raise ValueError("Only orthonormal coordinate systems supported")
    if np.abs(np.linalg.det(basis1) - 1.0) > 1e-10:
        raise ValueError("Only orthonormal coordinate systems supported")
    # Compute the transformation matrix which transforms basis0 into basis1
    # basis1 = basis0 * Q
    Q = np.dot(basis0.T, basis1)
    # Transform the bilinear form
    return np.dot(Q.T, np.dot(b, Q))


def gram_schmidt_orthonormalization_rn(basis):
    r"""
    Create an orthonormal basis from a general basis of :math:`\mathbb{R}^n` or an n dimensional subspace of
    :math:`\mathbb{R}^m` using the Gram-Schmidt process.

    :param basis: Input basis (m by n matrix with the basis vectors as columns).
    :type basis: :class:`Numpy array <numpy.ndarray>`
    :return: Orthonormal basis (m by n matrix with the basis vectors as columns), where the basis vectors is
        obtained by Gram-Schmidt orthonormalization of the input basis vectors.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    q, r = np.linalg.qr(basis)
    for i in range(basis.shape[1]):
        if np.dot(q[:, i], basis[:, i]) < 0:
            q[:, i] *= -1
    return q


def gram_schmidt_orthonormalization(basis, inner_product=np.dot):
    r"""
    Create an orthonormal basis for an inner product space V from a general basis using the Gram-Schmidt process.

    :param basis: Input array of basis vectors.
    :param inner_product: Inner product for the inner product space (a map :math:`V \times V \to \mathbb{R}` satisfying
        the inner product properties).
    :return: Orthonormal basis (matrix with the basis vectors as columns), where the basis vectors is
        obtained by Gram-Schmidt orthonormalization of the input basis vectors.
    """
    assert len(basis) > 0

    on_basis = copy.deepcopy(basis)
    for i in range(len(on_basis)):
        for j in range(i):
            on_basis[i] -= inner_product(on_basis[i], on_basis[j]) * on_basis[j]
        on_basis[i] /= np.sqrt(inner_product(on_basis[i], on_basis[i]))
    return on_basis


if __name__ == "__main__":
    import doctest
    doctest.testmod()
