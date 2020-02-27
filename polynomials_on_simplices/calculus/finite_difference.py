"""Functions used to compute derivatives using finite difference."""

import numpy as np


def forward_difference(f, x, h=1e-8):
    r"""
    Compute numerical gradient of the scalar valued function f using forward finite difference.

    .. math:: f : \mathbb{R}^n \to \mathbb{R},

    .. math:: \nabla f(x)_i = \frac{\partial f(x)}{\partial x^i},

    .. math:: \nabla f_{\Delta}(x)_i = \frac{f(x + he_i) - f(x)}{h}.

    :param f: Scalar valued function.
    :type f: Callable f(x)
    :param x: Point where the gradient should be evaluated.
    :type x: float or Iterable[float]
    :param float h: Step size in the finite difference method.
    :return: Approximate gradient of f.
    :rtype: float or length n :class:`Numpy array <numpy.ndarray>`.
    """
    try:
        n = len(x)
    except TypeError:
        # Univariate function
        return (f(x + h) - f(x)) / h
    # Multivariate function
    g_fd = np.empty(n)
    f0 = f(x)
    for i in range(n):
        x0 = x[i]
        x[i] += h
        f1 = f(x)
        x[i] = x0
        g_fd[i] = (f1 - f0) / h
    return g_fd


def forward_difference_jacobian(f, n, x, h=1e-8):
    r"""
    Compute numerical jacobian of the vector valued function f using forward finite difference.

    .. math:: f : \mathbb{R}^m \to \mathbb{R}^n,

    .. math:: J_f (x)^i_j = \frac{\partial f(x)^i}{\partial x^j},

    .. math:: J_{f, \Delta}(x, h)^i_j = \frac{f(x + he_j)^i - f(x)^i}{h},

    with :math:`i = 1, 2, \ldots, n, j = 1, 2, \ldots, m`.

    :param f: Vector valued function.
    :type f: Callable f(x)
    :param int n: Dimension of the target of f.
    :param x: Point where the jacobian should be evaluated.
    :type x: float or Iterable[float]
    :param float h: Step size in the finite difference method.
    :return: Approximate jacobian of f.
    :rtype: n by m :class:`Numpy matrix <numpy.ndarray>`.
    """
    # f : R^m -> R^n
    # Get dimension of the domain
    try:
        m = len(x)
    except TypeError:
        # Univariate function
        j_fd = np.empty((n, 1))
        j_fd[:, 0] = (f(x + h) - f(x)) / h
        return j_fd
    # Multivariate function
    j_fd = np.empty((n, m))
    f0 = f(x)
    for j in range(m):
        x0 = x[j]
        x[j] += h
        f1 = f(x)
        x[j] = x0
        j_fd[:, j] = (f1 - f0) / h
    return j_fd


def discrete_forward_difference(f0, x0, f1, x1):
    """
    Compute numerical derivative of a scalar valued, univariate function f using two discrete point evaluations.

    :param float f0: Function value at x0.
    :param float x0: First point where the function has been evaluated.
    :param float f1: Function value at x1.
    :param float x1: Second point where the function has been evaluated.
    :return: Numerical approximation of the derivative.
    :rtype: float
    """
    return (f1 - f0) / (x1 - x0)


def central_difference(f, x, h=1e-8):
    r"""
    Compute numerical gradient of the scalar valued function f using central finite difference.

    .. math:: f : \mathbb{R}^n \to \mathbb{R},

    .. math:: \nabla f(x)_i = \frac{\partial f(x)}{\partial x^i},

    .. math:: \nabla f_{\delta}(x)_i = \frac{f(x + \frac{h}{2}e_i) - f(x - \frac{h}{2}e_i)}{h}.

    :param f: Scalar valued function.
    :type f: Callable f(x)
    :param x: Point where the gradient should be evaluated.
    :type x: float or Iterable[float]
    :param float h: Step size in the central difference method.
    :return: Approximate gradient of f.
    :rtype: float or length n :class:`Numpy array <numpy.ndarray>`.
    """
    try:
        n = len(x)
    except TypeError:
        # Univariate function
        return (f(x + 0.5 * h) - f(x - 0.5 * h)) / h
    # Multivariate function
    g_fd = np.empty(n)
    for i in range(n):
        x0 = x[i]
        x[i] += 0.5 * h
        fp = f(x)
        x[i] = x0 - 0.5 * h
        fm = f(x)
        x[i] = x0
        g_fd[i] = (fp - fm) / h
    return g_fd


def central_difference_jacobian(f, n, x, h=1e-8):
    r"""
    Compute numerical jacobian of the vector valued function f using central finite difference.

    .. math:: f : \mathbb{R}^m \to \mathbb{R}^n,

    .. math:: J_f (x)^i_j = \frac{\partial f(x)^i}{\partial x^j},

    .. math:: J_{f, \delta}(x, h)^i_j = \frac{f(x + \frac{h}{2}e_j)^i - f(x - \frac{h}{2}e_j)^i}{h},

    with :math:`i = 1, 2, \ldots, n, j = 1, 2, \ldots, m`.

    :param f: Vector valued function.
    :type f: Callable f(x)
    :param int n: Dimension of the target of f.
    :param x: Point where the jacobian should be evaluated.
    :type x: float or Iterable[float]
    :param float h: Step size in the finite difference method.
    :return: Approximate jacobian of f.
    :rtype: n by m :class:`Numpy matrix <numpy.ndarray>`.
    """
    # f : R^m -> R^n
    # Get dimension of the domain
    try:
        m = len(x)
    except TypeError:
        # Univariate function
        j_fd = np.empty((n, 1))
        j_fd[:, 0] = (f(x + 0.5 * h) - f(x - 0.5 * h)) / h
        return j_fd
    # Multivariate function
    j_fd = np.empty((n, m))
    for j in range(m):
        x0 = x[j]
        x[j] += 0.5 * h
        fp = f(x)
        x[j] = x0 - 0.5 * h
        fm = f(x)
        x[j] = x0
        j_fd[:, j] = (fp - fm) / h
    return j_fd


def second_forward_difference(f, x, h=1e-5):
    r"""
    Compute the numerical Hessian of the scalar valued function f using second order forward finite difference.

    .. math:: f : \mathbb{R}^n \to \mathbb{R},

    .. math:: H_f(x)_{ij} = \frac{\partial^2 f(x)}{\partial x^i \partial x^j},

    .. math:: H_{f, \Delta}(x)_{ij} = \frac{f(x + h (e_i + e_j)) - f(x + h e_i) - f(x + h e_j) + f(x)}{h^2}.

    :param f: Scalar valued function.
    :type f: Callable f(x)
    :param x: Point where the Hessian should be evaluated.
    :type x: float or Iterable[float]
    :param float h: Step size in the finite difference method.
    :return: Hessian (full matrix, i.e., not utilizing the symmetry or any sparsity structure of the Hessian).
    :rtype: float or n by n :class:`Numpy matrix <numpy.ndarray>`.
    """
    try:
        n = len(x)
    except TypeError:
        # Univariate function
        return (f(x + 2 * h) - 2 * f(x + h) + f(x)) / h**2
    # Multivariate function
    h_fd = np.empty((n, n))
    e_i = np.zeros(n)
    e_j = np.zeros(n)
    h2_inv = 1 / h**2
    for i in range(n):
        e_i[i] = 1.0
        for j in range(i, n):
            e_j[j] = 1.0
            h_fd[i][j] = (f(x + h * (e_i + e_j))
                          - f(x + h * e_i)
                          - f(x + h * e_j)
                          + f(x)) * h2_inv
            if i != j:
                h_fd[j][i] = h_fd[i][j]
            e_j[j] = 0.0
        e_i[i] = 0.0
    return h_fd


def second_central_difference(f, x, h=2e-5):
    r"""
    Compute the numerical Hessian of the scalar valued function f using second order central finite difference.

    .. math:: f : \mathbb{R}^n \to \mathbb{R},

    .. math:: H_f(x)_{ij} = \frac{\partial^2 f(x)}{\partial x^i \partial x^j},

    .. math::

        H_{f, \delta}(x)_{ij} = \bigg[ &f(x + \frac{h}{2} (e_i + e_j)) - f(x + \frac{h}{2} (e_i - e_j))

        &- f(x + \frac{h}{2} (-e_i + e_j)) + f(x + \frac{h}{2} (-e_i - e_j)) \bigg] / h^2.

    :param f: Scalar valued function.
    :type f: Callable f(x)
    :param x: Point where the Hessian should be evaluated.
    :type x: float or Iterable[float]
    :param float h: Step size in the finite difference method.
    :return: Hessian (full matrix, i.e., not utilizing the symmetry or any sparsity structure of the Hessian).
    :rtype: float or n by n :class:`Numpy matrix <numpy.ndarray>`.
    """
    try:
        n = len(x)
    except TypeError:
        # Univariate function
        return (f(x + h) - 2 * f(x) + f(x - h)) / h**2
    # Multivariate function
    h_fd = np.empty((n, n))
    e_i = np.zeros(n)
    e_j = np.zeros(n)
    h2_inv = 1 / h**2
    for i in range(n):
        e_i[i] = 1.0
        for j in range(i, n):
            e_j[j] = 1.0
            h_fd[i][j] = (f(x + 0.5 * h * (e_i + e_j))
                          - f(x + 0.5 * h * (e_i - e_j))
                          - f(x + 0.5 * h * (e_j - e_i))
                          + f(x - 0.5 * h * (e_i + e_j))) * h2_inv
            if i != j:
                h_fd[j][i] = h_fd[i][j]
            e_j[j] = 0.0
        e_i[i] = 0.0
    return h_fd
