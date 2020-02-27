"""Functionality for computing the error between exact and approximate values."""


def absolute_error(x0, x):
    """
    Compute absolute error between a value `x` and its expected value `x0`.

    :param x0: Expected value.
    :param x: Actual value.
    :return: Absolute error between the actual and expected value.
    :rtype: float
    """
    return abs(x0 - x)


def relative_error(x0, x, zero_tol=1e-5):
    """
    Compute relative error between a value `x` and its expected value `x0`.

    :param x0: Expected value.
    :param x: Actual value.
    :param zero_tol: If x0 is smaller than this value, the absolute error is returned instead.
    :return: Relative error between the actual and expected value.
    :rtype: float

    .. rubric:: Examples

    >>> abs(relative_error(0.1, 0.4) - 3) < 1e-10
    True
    >>> abs(relative_error(0.4, 0.1) - 0.75) < 1e-10
    True

    For small values the absolute error is used
    >>> abs(relative_error(1e-6, 1e-6 + 1e-12) - 1e-12) < 1e-20
    True
    """
    if abs(x0) > zero_tol:
        return absolute_error(x0, x) / abs(x0)
    return absolute_error(x0, x)


def relative_error_symmetric(x1, x2, zero_tol=1e-5):
    r"""
    Compute relative error between two values `x1` and `x2`.

    .. note::

        The :func:`relative_error` function is not symmetric, i.e. in general
        `relative_error(a, b) != relative_error(b, a)`, which makes sense when comparing to a reference value. However
        when just checking the error between two values (without one of them being more correct than the other) it makes
        more sense with a symmetric error, which is what this function returns.

        .. math:: \varepsilon = \frac{|x_1 - x_2|}{\max(|x_1|, |x_2|)}.

    :param x1: First value.
    :param x2: Second value.
    :param zero_tol: If max(abs(x1), abs(x2)) is smaller than this value, the absolute error is returned instead.
    :return: Relative error between the two values.
    :rtype: float

    .. rubric:: Examples

    >>> relative_error_symmetric(0.1, 0.2)
    0.5
    >>> relative_error_symmetric(0.1, 0.2) == relative_error_symmetric(0.2, 0.1)
    True
    """
    denom = max(abs(x1), abs(x2))
    if denom > zero_tol:
        return absolute_error(x1, x2) / denom
    return absolute_error(x1, x2)
