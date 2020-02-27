r"""Operations on multi-indices (elements of :math:`\mathbb{N}_0^n`)."""

import math
from operator import add, sub

import numpy as np

from polynomials_on_simplices.algebra.modular_arithmetic import IntegerModuloN


class MultiIndex:
    r"""
    A multi-index (element of :math:`\mathbb{N}_0^n`).

    This class defines the basic algebraic operations on multi-indices:

    **Addition:**

    .. math:: + : \mathbb{N}_0^n \times \mathbb{N}_0^n \to \mathbb{N}_0^n,

    .. math:: \alpha + \beta = (\alpha_1 + \beta_1, \alpha_2 + \beta_2, \ldots, \alpha_n + \beta_n).

    **Power:**

    .. math:: \operatorname{pow} : R^n \times \mathbb{N}_0^n \to R,

    .. math:: \operatorname{pow}(x, \alpha) \equiv x^{\alpha} = x_1^{\alpha_1} x_2^{\alpha_2} \ldots x_n^{\alpha_n},

    where :math:`R` is any ring.
    """

    def __init__(self, *components):
        """
        :param components: Component(s) (indices) for the multi-index.
        :type components: int or Iterable[int]
        """
        if len(components) > 1:
            self._components = list(components)
        else:
            if isinstance(components[0], list):
                self._components = components[0]
            else:
                if _is_iterable(components[0]):
                    self._components = list(components[0])
                else:
                    self._components = list([components[0]])
        for i in range(len(self._components)):
            if self._components[i] < 0:
                raise ValueError("Multi-index component cannot be negative.")

    def __repr__(self):
        """
        Unambiguous string representation of object.

        :return: Unambiguous string which can be used to create an identical multi-index.
        :rtype: str
        """
        return "polynomials_on_simplices.algebra.multiindex.MultiIndex(" + str(self._components) + ")"

    def __str__(self):
        """
        Human readable string representation of object.

        :return: String representation of the object.
        :rtype: str
        """
        return str(tuple(self._components))

    def __len__(self):
        """
        Get number of multi-index components.

        :return: Length of the multi-index.
        :rtype: int
        """
        return len(self._components)

    def __getitem__(self, i):
        """
        Get a component (index) of the multi-index.

        :param int i: Component to get.
        :return: The i:th component of the multi-index.
        """
        return self._components[i]

    def __setitem__(self, i, val):
        """
        Set a component (index) of the multi-index.

        :param int i: Index of component to set.
        :param val: New value for the component.
        """
        if val < 0:
            raise ValueError("Multi-index component cannot be negative.")
        self._components[i] = val

    def __iter__(self):
        return iter(self._components)

    def __hash__(self):
        """
        Get hash value for the multi-index.
        """
        # A multi-index is a tuple of natural numbers, so it makes sense
        # to use the same hash value as for a tuple
        return hash(tuple(self._components))

    def __eq__(self, other):
        r"""
        Check for equality between self and another multi-index, self == other.
        Let :math:`a, b \in \mathbb{N}_0^n`. Then :math:`a = b` if :math:`a_i = b_i, i = 1, 2, \ldots, n`.

        :param other: Other multi-index to compare with.
        :return: Whether or not this multi-index is equal to the other multi-index.
        :rtype: bool
        """
        if len(self) != len(other):
            raise ValueError("Cannot compare multi-indices with different dimensions.")
        return all([x == y for (x, y) in zip(self, other)])

    def __ne__(self, other):
        r"""
        Check for difference between self and another multi-index, self != other.
        Let :math:`a, b \in \mathbb{N}_0^n`. Then :math:`a \neq b` if :math:\exists i \in \{1, 2, \ldots, n\}` such that
        :math:`a_i \neq b_i`.

        :param other: Other multi-index to compare with.
        :return: Whether or not this multi-index is not equal to the other multi-index.
        :rtype: bool
        """
        return not self == other

    def __add__(self, other):
        r"""
        Addition of this multi-index with another multi-index, self + other.
        Let :math:`a, b \in \mathbb{N}_0^n`. Then :math:`a + b \in \mathbb{N}_0^n` is given by :math:`(a + b)_i
        = a_i + b+i`.

        :param other: Other multi-index.
        :return: Sum of this multi-index with the other multi-index.
        :rtype: :class:`MultiIndex`
        """
        if isinstance(other, MultiIndex):
            components = list(map(add, self._components, other.components()))
            return MultiIndex(components)
        components = list(map(add, self._components, other))
        return MultiIndex(components)

    def __radd__(self, other):
        r"""
        Addition of this multi-index with another multi-index, other + self.
        Let :math:`a, b \in \mathbb{N}_0^n`. Then :math:`a + b \in \mathbb{N}_0^n` is given by :math:`(a + b)_i
        = a_i + b+i`.

        :param other: Other multi-index.
        :return: Sum of this multi-index with the other multi-index.
        :rtype: :class:`MultiIndex`
        """
        components = list(map(add, self._components, other))
        return MultiIndex(components)

    def __sub__(self, other):
        r"""
        Subtraction of this multi-index with another multi-index, self - other.
        Let :math:`a, b \in \mathbb{N}_0^n`. Then :math:`a - b \in \mathbb{N}_0^n` is given by :math:`(a - b)_i
        = a_i - b+i`.

        :param other: Other multi-index.
        :return: Difference of this multi-index with the other multi-index.
        :rtype: :class:`MultiIndex`
        """
        if isinstance(other, MultiIndex):
            components = list(map(sub, self._components, other.components()))
            return MultiIndex(components)
        components = list(map(sub, self._components, other))
        return MultiIndex(components)

    def __rsub__(self, other):
        r"""
        Subtraction of this multi-index with another multi-index, other - self.
        Let :math:`a, b \in \mathbb{N}_0^n`. Then :math:`a - b \in \mathbb{N}_0^n` is given by :math:`(a - b)_i
        = a_i - b+i`.

        :param other: Other multi-index.
        :return: Difference of the other multi-index with this multi-index.
        :rtype: :class:`MultiIndex`
        """
        components = list(map(sub, other, self._components))
        return MultiIndex(components)

    def __rpow__(self, x):
        r"""
        Raise x to the power of this multi-index.
        Let :math:`a, b \in \mathbb{N}_0^n`. Then :math:`x^a = x_1^{a_1} x_2^{a_2} \ldots x_n^{a_n}`

        :param x: Iterable of same length as this multi-index.
        :return: x raised to the power of this multi-index.
        """
        return power(x, self)

    def components(self):
        """
        Multi-index components/indices.

        :return: The multi-index components.
        :rtype: tuple[int]
        """
        return tuple(self._components)

    def to_tuple(self):
        """
        Multi-index converted to a tuple.

        :return: Tuple containing the multi-index components (indices).
        :rtype: tuple[int]
        """
        return self.components()


def zero_multiindex(n):
    r"""
    Generate the n-dimensional zero multi-index (element of :math:`\mathbb{N}_0^n` with all entries equal to zero).

    :param int n: Dimension of the multi-index.
    :return: The n-dimensional zero multi-index.
    :rtype: :class:`MultiIndex`

    .. rubric:: Examples

    >>> print(zero_multiindex(2))
    (0, 0)
    """
    return MultiIndex(np.zeros(n, dtype=int))


def unit_multiindex(n, i):
    r"""
    Generate the n-dimensional multi-index (element of :math:`\mathbb{N}_0^n`) with all entries equal to zero except
    the i:th entry which is equal to 1.

    :param int n: Dimension of the multi-index.
    :param int i: Entry of the multi-index which should be equal to 1.
    :return: The i:th n-dimensional unit multi-index.
    :rtype: :class:`MultiIndex`

    .. rubric:: Examples

    >>> print(unit_multiindex(3, 0))
    (1, 0, 0)
    >>> print(unit_multiindex(2, 1))
    (0, 1)
    """
    a = MultiIndex(np.zeros(n, dtype=int))
    a[i] = 1
    return a


def norm(a):
    r"""
    Absolute value of a multi-index, :math:`|a| = a_1 + a_2 + \ldots + a_n`.

    :param a: Multi-index.
    :return: Absolute value of the multi-index.
    """
    return sum(a)


def factorial(a):
    r"""
    Factorial of a multi-index, :math:`a! = a_1! a_2! \ldots a_n!`.

    :param a: Multi-index.
    :return: Factorial of the multi-index.
    """
    f = 1
    for i in range(len(a)):
        f *= math.factorial(a[i])
    return f


def power(x, a):
    r"""
    Raise a vector to the power of a multi-index, :math:`x^a = x_1^{a_1} x_2^{a_2} \ldots x_n^{a_n}`.

    :param x: Iterable of same length as the multi-index `a`.
    :param a: Multi-index.
    :return: x raised to the power a.
    """
    assert len(x) == len(a)
    p = 1
    for i in range(len(a)):
        p *= x[i]**a[i]
    return p


def binom(a, b):
    r"""
    Binomial coefficient of two multi-indices, a over b,

    .. math:: \binom{a}{b} = \frac{a!}{b!(a - b)!}.

    See :func:`factorial`.

    :param a: Multi-index.
    :param b: Multi-index.
    :return: a choose b.
    :rtype: int
    """
    return factorial(a) / (factorial(b) * factorial(a - b))


def multinom(a):
    r"""
    Multinomial coefficient of a multi-index.

    Number of ways to put :math:`|a|` elements in n boxes with :math:`a_i` elements in box i (where n is the number
    of elements in :math:`a`),

    .. math:: \binom{|a|}{a} = \frac{|a|!}{a!}.

    :param a: Multi-index.
    :return: Multinomial coefficient, :math:`\frac{|a|!}{a!}`.
    """
    return math.factorial(norm(a)) / factorial(a)


def multinom_general(r, a):
    r"""
    Multinomial coefficient of a multi-index.

    Number of ways to put :math:`r` elements in n boxes with :math:`a_i` elements in box i,
    :math:`i = 1, 2, \ldots, n - 1` and :math:`r - |a|` elements in box n (where n - 1 is the number of elements
    in :math:`a`),

    .. math:: \binom{r}{a} = \frac{r!}{a!(r - |a|)!}.

    This is equal to the multinomial coefficient of the multi-index a converted to exact form with norm r.

    :param a: Multi-index.
    :param int r: Total number of elements (or norm of the exact multi-index).
    :return: Multinomial coefficient, :math:`\frac{r!}{a!(r - |a|)!}`.
    """
    return multinom(general_to_exact_norm(a, r))


def is_increasing(a):
    r"""
    Check if the indices of a multi-index form an increasing sequence, i.e. :math:`a_i < a_j` if :math:`i < j`.

    :param a: Multi-index.
    :return: Whether or not the indices of the multi-index are increasing.

    .. rubric:: Examples

    >>> is_increasing((1, 2, 3))
    True
    >>> is_increasing((1, 1))
    False
    """
    n = len(a)
    if n == 1:
        return True
    for i in range(n - 1):
        if a[i + 1] <= a[i]:
            return False
    return True


def is_non_decreasing(a):
    r"""
    Check if the indices of a multi-index form a non-decreasing sequence, i.e. :math:`a_i \leq a_j` if :math:`i < j`.

    :param a: Multi-index.
    :return: Whether or not the indices of the multi-index are non-decreasing.

    .. rubric:: Examples

    >>> is_non_decreasing((1, 2, 3))
    True
    >>> is_non_decreasing((1, 1))
    True
    >>> is_non_decreasing((1, 3, 2))
    False
    """
    n = len(a)
    if n == 1:
        return True
    for i in range(n - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def generate_all(n, r):
    """
    Generate the sequence of all n-dimensional multi-indices with norm <= r.

    For ordering of the multi-indices see :func:`generate`.

    :param int n: Dimension of multi-indices.
    :param int r: Maximum norm of multi-indices.
    :return: List of all multi-indices.
    :rtype: List[:class:`MultiIndex`].
    """
    assert r >= 0
    return [mi for mi in MultiIndexIterator(n, r)]


def generate_all_multi_cap(r):
    r"""
    Generate all n-dimensional multi-indices :math:`a` such that :math:`a_i \leq r_i, i = 1, 2, \ldots, n`, where n
    is the length of r.

    :param r: Maximum value for each entry of the multi-indices.
    :type r: Iterable[int]
    :return: List of all multi-indices.
    :rtype: List[:class:`MultiIndex`].
    """
    return [mi for mi in MultiIndexIteratorMultiCap(len(r), r)]


def generate_all_exact_norm(n, r):
    """
    Generate all n-dimensional multi-indices with norm r.

    :param int n: Dimension of multi-indices.
    :param r: Norm of each multi-index.
    :return: List of all multi-indices with norm r.
    :rtype: List[:class:`MultiIndex`].
    """
    return [general_to_exact_norm(mi, r) for mi in generate_all(n - 1, r)]


def generate_all_increasing(n, r):
    """
    Generate all increasing (see :func:`is_increasing`) n-dimensional multi-indices such that each component is
    less than or equal to r.

    :param int n: Dimension of multi-indices.
    :param r: Max value for each component of the multi-indices.
    :return: List of increasing multi-indices.
    :rtype: List[:class:`MultiIndex`].
    """
    return [mi for mi in generate_all_multi_cap(n * [r]) if is_increasing(mi)]


def generate_all_non_decreasing(n, r):
    """
    Generate all non-decreasing (see :func:`is_non_decreasing`) n-dimensional multi-indices such that each component
    is less than or equal to r.

    :param int n: Dimension of multi-indices.
    :param r: Max value for each component of the multi-indices.
    :return: List of non-creasing multi-indices.
    :rtype: List[:class:`MultiIndex`].
    """
    return [mi for mi in generate_all_multi_cap(n * [r]) if is_non_decreasing(mi)]


def generate(n, r, i):
    r"""
    Generate the i:th multi-index in the sequence of all n-dimensional multi-indices with norm <= r.

    There is a natural ordering of the multi-indices in the sense that a multi-index :math:`a` of
    norm <= r can be identified with a natural number :math:`n(a)` by
    :math:`n(a) = \sum_{k = 0}^{\dim \nu} a_k r^k` (interpreting the indices of a as digits of a number in base r),
    and this number is strictly increasing with i.

    :param int n: Dimension of multi-indices.
    :param int r: Maximum norm of multi-indices.
    :param int i: Which multi-index to generate. Need to be in the range [0, num_multiindices(n, r) - 1].
    :return: The i:th multi-index.
    :rtype: :class:`MultiIndex`
    """
    mi_iter = MultiIndexIterator(n, r)
    for j in range(i):
        next(mi_iter)
    return next(mi_iter)


def generate_multi_cap(r, i):
    r"""
    Generate the i:th multi-index among all n-dimensional multi-indices :math:`a` such that
    :math:`a_i \leq r_i, i = 1, 2, \ldots, n`, where n is the length of r.

    The ordering of the multi-indices is natural in the sense that each generated multi-index can be identified
    with a natural number expressed in the base :math:`\max_i r_i`, and this number is strictly increasing with i.

    :param r: Maximum value for each entry of the multi-indices.
    :type r: Iterable[int]
    :param int i: Which multi-index to generate. Need to be in the range [0, :math:`(\Pi_i (r_i + 1)) - 1`].
    :return: The i:th multi-index.
    :rtype: :class:`MultiIndex`
    """
    try:
        len(r)
    except TypeError:
        # Univariate case
        return MultiIndex((i,))
    mi = len(r) * [0]
    for j in range(len(r)):
        mi[j] = i % (r[j] + 1)
        i = i // (r[j] + 1)
    return MultiIndex(mi)


def get_index(mi, r):
    """
    Get the index of a multi-index in the sequence of all multi-indices of the same dimension and with norm <= r
    (as given by :func:`generate_all`).

    :param mi: Multi-index.
    :param int r: Maximum norm of multi-indices.
    :return: Index of multi-index.
    :rtype: int
    :raise: ValueError if the given multi-index doesn't belong to the sequence of multi-indices with the specified
        dimension and with norm <= r.
    """
    assert norm(mi) <= r

    from polynomials_on_simplices.algebra.multiindex_order_cache import multiindex_order_cache
    n = len(mi)
    if (n, r) in multiindex_order_cache:
        return multiindex_order_cache[(n, r)][mi]

    idx = 0
    for mi2 in MultiIndexIterator(n, r):
        if mi == mi2:
            return idx
        idx += 1
    raise ValueError("Failed to find matching multi-index among all multi-indices of dimension "
                     + str(n) + " and norm <= " + str(r))


def num_multiindices(n, r):
    """
    Compute the number of n-dimensional multi-indices with norm <= r.

    :param int n: Dimension of multi-indices.
    :param int r: Maximum norm of multi-indices.
    :return: Number of unique multi-indices.
    :rtype: int
    """
    from scipy.special import binom
    return int(binom(n + r, r))


def general_to_exact_norm(a, r):
    r"""
    Conversion of a multi-index from general to exact form.

    Convert a general n-dimensional multi-index to an exact n+1-dimensional multi-index
    (exact meaning that the multi-index has norm r). Let :math:`a \in \mathbb{N}_0^n`. Then this function returns
    :math:`b \in \mathbb{N}_0^{n + 1}` with :math:`b_1 = r - |a|` and :math:`b_i = a_{i - 1}, i = 2, 3, \ldots, n + 1`.

    :param a: Multi-index.
    :param int r: Desired norm of exact multi-index.
    :return: Multi-index with norm r.
    :rtype: :class:`MultiIndex`

    .. rubric:: Examples

    >>> general_to_exact_norm((1, 2), 4).to_tuple()
    (1, 1, 2)
    >>> general_to_exact_norm((0, 0), 2).to_tuple()
    (2, 0, 0)
    """
    assert norm(a) <= r
    return MultiIndex((r - norm(a),) + tuple(a))


def exact_norm_to_general(a):
    """
    Conversion of a multi-index from exact to general form.

    Convert an n-dimensional exact multi-index to a general n-1-dimensional multi-index by removing the first number
    in the multi-index (exact meaning that the multi-index has norm r).

    :param a: Multi-index.
    :return: Multi-index.
    :rtype: :class:`MultiIndex`
    """
    return MultiIndex(a[1:])


def random_multiindex(n, r):
    """
    Generate a random multi-index from the set of all n-dimensional multi-indices with norm <= r, with uniform
    sampling.

    :param int n: Dimension of multi-index.
    :param int r: Maximum norm of multi-index.
    :return: Random n-dimensional multi-index with norm <= r.
    :rtype: :class:`MultiIndex`
    """
    dim = num_multiindices(n, r)
    i = np.random.randint(0, dim)
    return generate(n, r, i)


class MultiIndexIterator:
    """Iterate over all n-dimensional multi-indices with norm <= r.
    """

    def __init__(self, n, r):
        """
        :param int n: Dimension of the multi-indices we iterate over.
        :param int r: Maximum norm of the multi-indices we iterate over.
        """
        self._n = n
        self._r = r
        self._norm = 0
        self._components = None
        self._help_components = [IntegerModuloN(0, r + 1)] * n

    def __iter__(self):
        return self

    def __next__(self):
        if self._components is None:
            self._components = np.zeros(self._n, dtype=int)
            return zero_multiindex(self._n)
        self._increase_components()
        return MultiIndex(self._components)

    def next(self):
        """Proceed to next multi-index."""
        return self.__next__()

    def _increase_components(self):
        self._increase_component(0)

    def _increase_component(self, i):
        if i >= self._n:
            raise StopIteration
        if self._norm == self._r:
            # Can't increase component further
            # Set it to zero and increase next component instead
            self._norm -= self._components[i]
            self._help_components[i] = IntegerModuloN(0, self._r + 1)
            self._components[i] = 0
            self._increase_component(i + 1)
            return
        self._help_components[i] += 1
        if self._help_components[i] == 0:
            # Component maxed out, increase next component instead
            self._norm -= self._components[i]
            self._components[i] = 0
            self._increase_component(i + 1)
        else:
            self._norm += 1
            self._components[i] += 1


class MultiIndexIteratorMultiCap:
    """Iterate over all n-dimensional multi-indices with satisfying a_i <= r_i.
    """

    def __init__(self, n, r):
        """
        :param int n: Dimension of the multi-indices we iterate over.
        :param r: Maximum value for each component of the multi-indices we iterate over.
        :type r: Iterable[int]
        """
        assert n == len(r)
        self._n = n
        self._r = r
        self._components = None
        self._help_components = [IntegerModuloN(0, r[i] + 1) for i in range(len(r))]

    def __iter__(self):
        return self

    def __next__(self):
        if self._components is None:
            self._components = np.zeros(self._n, dtype=int)
            return zero_multiindex(self._n)
        self._increase_components()
        return MultiIndex(self._components)

    def next(self):
        """Proceed to next multi-index."""
        return self.__next__()

    def _increase_components(self):
        self._increase_component(0)

    def _increase_component(self, i):
        if i >= self._n:
            raise StopIteration
        self._help_components[i] += 1
        if self._help_components[i] == 0:
            # Component maxed out, increase next component instead
            self._components[i] = 0
            self._increase_component(i + 1)
        else:
            self._components[i] += 1


def _is_iterable(a):
    try:
        iter(a)
        return True
    except TypeError:
        return False


if __name__ == "__main__":
    import doctest
    doctest.testmod()
