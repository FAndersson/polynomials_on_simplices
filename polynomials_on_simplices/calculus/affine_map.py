r"""
Functionality for dealing with affine maps :math:`\Phi : \mathbb{R}^m \to \mathbb{R}^n` with

.. math:: \Phi(x) = Ax + b,

where :math:`A \in \mathbb{R}^{n \times m}, b \in \mathbb{R}^n`.
"""

import numbers

import numpy as np


def create_affine_map(a, b, multiple_arguments=False):
    r"""
    Generate the affine map :math:`\Phi : \mathbb{R}^m \to \mathbb{R}^n, \Phi(x) = Ax + b` from a matrix A and
    a vector b (or scalars a and b in the case m = n = 1).

    :param a: Matrix or scalar defining the linear part of the affine map.
    :param b: Vector or scalar defining the translation part of the affine map.
    :param bool multiple_arguments: For a multivariate affine map, this argument determines if the generated map should
        take m scalar arguments or 1 m-dimensional vector as argument. For example with a 2-dimensional domain the
        generated map could have the signature :math:`\Phi([x, y])` (if `multiple_arguments` is False) or
        :math:`\Phi(x, y)` (if `multiple_arguments` is True).
    :return: Map :math:`\Phi` which takes an m-dimensional vector as input and returns an n-dimensional vector (or
        scalar input and output for m = n = 1).
    :rtype: Callable :math:`\Phi(x)`.
    """
    if isinstance(a, numbers.Number):
        # R -> R
        assert isinstance(b, numbers.Number)

        def phi(x):
            return a * x + b
        return phi

    if len(a.shape) == 1:
        try:
            len(b)
            # R -> R^n, n > 1

            def phi(x):
                return a * x + b
            return phi
        except TypeError:
            # R^n -> R, n > 1
            assert isinstance(b, numbers.Number)

            if multiple_arguments:
                def phi(*x):
                    return np.dot(a, np.array(x)) + b
            else:
                def phi(x):
                    return np.dot(a, x) + b
            return phi

    assert len(a.shape) == 2
    n, m = a.shape

    if m == 1:
        if n == 1:
            # Invalid input, pass a and b as scalars instead
            assert False
        else:
            assert len(b) == n

            def phi(x):
                return a.flatten() * x + b
            return phi
    else:
        if n == 1:
            assert isinstance(b, numbers.Number)

            if multiple_arguments:
                def phi(*x):
                    return np.dot(a, np.array(x))[0] + b
            else:
                def phi(x):
                    return np.dot(a, x)[0] + b
            return phi
        else:
            assert len(b) == n

            if multiple_arguments:
                def phi(*x):
                    return np.dot(a, np.array(x)) + b
            else:
                def phi(x):
                    return np.dot(a, x) + b
            return phi


def inverse_affine_transformation(a, b):
    r"""
    Generate the matrix and vector defining the inverse of a given affine map
    :math:`\Phi : \mathbb{R}^n \to \mathbb{R}^n`,

    .. math:: \Phi(x) = Ax + b.

    The inverse of the affine map :math:`\Phi(x)` is given by :math:`\Phi^{-1}(x) = A^{-1}x - A^{-1}b`.
    This function returns the matrix :math:`A^{-1}` and vector :math:`-A^{-1}b` defining the inverse map (or
    scalars 1 / A and -b / A for scalar input A, b).

    :param a: Matrix or scalar defining the linear part of the affine map we want to invert.
    :param b: Vector or scalar defining the translation part of the affine map we want to invert.
    :return: Tuple of A and b.
    """
    # Handle 1d case separately
    if isinstance(a, numbers.Number):
        a_inv = 1 / a
        b_inv = -b * a_inv
        return a_inv, b_inv
    a_inv = np.linalg.inv(a)
    b_inv = -np.dot(a_inv, b)
    return a_inv, b_inv


def _is_invertible(a):
    """
    Check if a given matrix is invertible.

    :param a: Square matrix.
    :return: Whether or not the matrix is invertible
    :rtype: bool
    """
    try:
        np.linalg.inv(a)
        return True
    except np.linalg.LinAlgError:
        return False


def pseudoinverse_affine_transformation(a, b):
    r"""
    Generate the matrix and vector defining the pseudoinverse of a given affine map.

    The inverse of an affine map :math:`\Phi(x) = Ax + b` is given by :math:`\Phi^{-1}(x) = A^{-1}x - A^{-1}b`.
    Assuming the matrix A is not invertible, but :math:`A^T A` is, so that it has pseudoinverse
    :math:`A^+ = (A^T A)^{-1} A^T`. Then the affine map :math:`\Phi^+(x) = A^+x - A^+b` satisfies
    :math:`\Phi^+(\Phi(x)) = x`.
    This function returns the matrix :math:`A^+` and vector :math:`-A^+b` defining this pseudoinverse map.

    :param a: Matrix or scalar defining the linear part of the affine map we want to invert.
    :param b: Vector or scalar defining the translation part of the affine map we want to invert.
    :return: Tuple of A and b.
    """
    assert _is_invertible(np.dot(a.T, a)) or isinstance(np.dot(a.T, a), numbers.Number)
    if len(a.shape) == 1:
        return pseudoinverse_affine_transformation(np.reshape(a, (len(a), 1)), b)
    n, m = a.shape
    a_inv = np.linalg.pinv(a)
    if m == 1:
        b_inv = -np.dot(a_inv, b)[0]
    else:
        b_inv = -np.dot(a_inv, b)
    return a_inv, b_inv


def affine_composition(phi1, phi2):
    r"""
    Compute the matrix and vector defining the composition of two affine maps.

    :math:`\Phi_1(x) = A_1x + b_1, \Phi_2(x) = A_2x + b_2, \Phi(x) = (\Phi_2 \circ \Phi_1)(x) = Ax + b`.
    :math:`A = A_2 A_1, b = A_2 b_1 + b_2`.

    :param phi1: Tuple of A matrix and b vector for the first affine transformation.
    :param phi2: Tuple of A matrix and b vector for the second affine transformation.
    :return: Tuple of A and b.
    """
    a1, b1 = phi1
    a2, b2 = phi2
    return np.dot(a2, a1), np.dot(a2, b1) + b2
