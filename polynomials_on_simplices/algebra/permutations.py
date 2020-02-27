r"""Permutations module.

Operations on permutations (elements of :math:`S_n`) and functions for permuting a sequence of objects.
The set of permutations :math:`S_n` is defined as the set of bijections from the set :math:`\{0, 1, \ldots, n-1\}`
(or :math:`\{1, 2, \ldots, n\}`) onto itself, see https://en.wikipedia.org/wiki/Permutation and
https://en.wikipedia.org/wiki/Permutation_group for an introduction.

There are two natural ways for defining how a permutation :math:`\sigma` acts on a general sequence of elements
:math:`x = (x_1, x_2, \ldots, x_n)`.

1. The value :math:`x_i` is mapped to the value :math:`x_{\sigma(i)}`,

    .. math:: (x_1, \ldots, x_n) \mapsto \sigma(x) = (x_{\sigma(1)}, \ldots, x_{\sigma(n)}).

2. The element at position i is mapped to the position :math:`\sigma(i)`, i.e. :math:`\sigma(x)_{\sigma(i)} = x_i
   \iff \sigma(x)_i = x_{\sigma^{-1}(i)} \iff x_i \mapsto x_{\sigma^{-1}(i)}`.

    .. math:: (x_1, \ldots, x_n) \mapsto \sigma(x) = (x_{\sigma^{-1}(1)}, \ldots, x_{\sigma^{-1}(n)}).

As an example consider permutation of ['a', 'b', 'c', 'd'] with the permutation

.. math:: \begin{pmatrix} 1 & 2 & 3 & 4 \\ 3 & 2 & 4 & 1 \end{pmatrix}.

1. will then give ['c', 'b', 'd', 'a'] (x[1] ('a') is mapped to (replaced by) x[s(1)] = x[3] ('c') and so on)
2. will then give ['d', 'b', 'a', 'c'] (x[1] ('a') is mapped to position s(1) = 3 and so on)

Obviously these two interpretations are the inverses of each other. In the code we refer to alternative 1 as
permutation by value and alternative 2 as permutation by position.

.. note::

    These two conventions for permuting a generic
    sequence behaves differently under composition. For alternative 1 we have

    .. math:: (\sigma \circ \pi)(x) = \pi(\sigma(x)),

    whereas for alternative 2 we have

    .. math:: (\sigma \circ \pi)(x) = \sigma(\pi(x)).
"""

from copy import deepcopy
import math
import random

import numpy as np
from scipy.special import binom


class Permutation:
    r"""
    A permutation (element of :math:`S_n`).

    This class implements the group structure of the set of n-dimensional permutations.

    **Composition:**

    .. math:: \circ : S_n \times S_n \to S_n, (\sigma \circ \pi)(x) = \sigma(\pi(x)).

    **Inverse:**

    .. math:: (\cdot)^{-1} : S_n \to S_n, \sigma^{-1}(y) = x,

    where :math:`x` is the unique element that satisfies :math:`\sigma(x) = y`.

    .. rubric:: Examples

    >>> Permutation(4, 3, 2, 1, 0) * Permutation(1, 3, 0, 2, 4) == Permutation(3, 1, 4, 2, 0)
    True
    >>> Permutation(1, 4, 3, 2, 0)**(-1) == Permutation(4, 0, 3, 2, 1)
    True
    """

    def __init__(self, *values):
        r"""
        :param values: Sequence of the elements :math:`\{0, 1, \ldots, n - 1 \}` defining the values of the
            permutation (e.g. :math:`0 \mapsto \text{ values[0]}, 1 \mapsto \text{ values[1]}, \ldots`.
        :type values: Iterable[int]
        """
        if len(values) > 1:
            self._values = tuple(values)
        else:
            if isinstance(values[0], tuple):
                self._values = values[0]
            else:
                if _is_iterable(values[0]):
                    self._values = tuple(values[0])
                else:
                    raise ValueError("Permutation: Unknown input.")

    def __repr__(self):
        """
        Unambiguous string representation of object.

        :return: Unambiguous string which can be used to create an identical permutation.
        :rtype: str
        """
        return "polynomials_on_simplices.algebra.permutations.Permutation(" + str(self.to_tuple()) + ")"

    def __str__(self):
        """
        Human readable string representation of object.

        :return: String representation of the object.
        :rtype: str
        """
        return str(to_one_based(self.to_tuple()))

    def __len__(self):
        """
        Get dimension of the permutation.

        :return: Length of the permutation.
        :rtype: int
        """
        return len(self.to_tuple())

    def __getitem__(self, i):
        """
        Evaluate the permutation for value i.

        :param int i: Element in the permutation domain.
        :return: Value of the permutation.
        :rtype: int
        """
        return self(i)

    def __iter__(self):
        return iter(self._values)

    def __eq__(self, other):
        r"""
        Check for equality between self and another permutation, self == other.
        Let :math:`\sigma, \pi \in S_n`. Then :math:`\sigma = \pi` if :math:`\sigma(i) = \pi(i), i = 1, 2, \ldots, n`.

        :param other: Other permutation to compare with.
        :return: Whether or not this permutation is equal to the other permutation.
        :rtype: bool
        """
        if len(self) != len(other):
            raise ValueError("Cannot compare permutations with different dimensions.")
        return all([x == y for (x, y) in zip(self, other)])

    def __ne__(self, other):
        r"""
        Check for difference between self and another permutation, self != other.
        Let :math:`\sigma, \pi \in S_n`. Then :math:`\sigma \neq \pi` if :math:\exists i \in \{1, 2, \ldots, n\}`
        such that :math:`\sigma(i) \neq \pi(i)`.

        :param other: Other permutation to compare with.
        :return: Whether or not this permutation is not equal to the other permutation.
        :rtype: bool
        """
        return not self == other

    def __mul__(self, other):
        r"""
        Composition of this permutation after another permutation.

        :param other: Other permutation to apply before this one.
        :return: Composition of this permutation with another permutation.
        :rtype: :class:`Permutation`.
        """
        return Permutation(composition(self, other))

    def __pow__(self, exp):
        r"""
        Compute the inverse of this permutation (raise it to the power -1).

        :param exp: Exponent. Can only be -1.
        :return: Inverse of this permutation.
        :rtype: :class:`Permutation`.
        """
        if exp != -1:
            raise ValueError("Invalid argument. Permutation can only be raised to the power -1.")
        return Permutation(inverse(self))

    def __call__(self, i):
        r"""
        Evaluate permutation (:math:`\pi(i)`).

        :param int i: Value where the permutation should be evaluated (in :math:`\{ 0, 1, \ldots, n - 1 \}` where
            :math:`n` is the dimension of the permutation).
        :return: Value of the permutation.
        :rtype: int
        """
        assert i >= 0
        assert i < len(self)
        return self._values[i]

    def index(self, value):
        """
        Return first index of value.

        Raises ValueError if the value is not present.
        """
        return self._values.index(value)

    def to_tuple(self):
        """
        Permutation converted to a tuple (one-line notation).

        :return: Tuple containing the permutation values.
        :rtype: Tuple[int]
        """
        return self._values


def to_two_line_notation(permutation):
    r"""
    Convert a permutation from one-line notation to two line notation.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :return: Permutation expressed in two line notations (two length n tuples of the numbers 0, 1, ..., n-1,
        where the first lists the elements :math:`x_i`, and the second lists the image under the permutation
        :math:`\sigma(x_i)`.

    .. rubric:: Examples

    >>> to_two_line_notation((0, 3, 2, 1))
    ((0, 1, 2, 3), (0, 3, 2, 1))
    """
    n = len(permutation)
    return tuple(range(n)), permutation


def from_two_line_notation(permutation):
    r"""
    Convert a permutation from two-line notation to one-line notation.

    :param permutation: Permutation in two line notations (two length n tuples of the numbers 0, 1, ..., n-1,
        where the first lists the elements :math:`x_i`, and the second lists the image under the permutation
        :math:`\sigma(x_i)`.
    :return: Permutation expressed in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).

    .. rubric:: Examples

    >>> from_two_line_notation(((0, 2, 1, 3), (0, 2, 3, 1)))
    (0, 3, 2, 1)
    """
    # Make sure that the first line is sorted
    permutation_sorted = tuple(zip(*sorted(zip(permutation[0], permutation[1]))))
    # Return the second line
    return permutation_sorted[1]


def cycle(permutation, start):
    """
    Compute a cycle of a permutation.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :param start: Permutation element to start with.
    :return: Tuple of elements we pass until we cycle back to the start element.

    .. rubric:: Examples

    >>> cycle((2, 3, 0, 1), 0)
    (0, 2)
    >>> cycle((2, 3, 0, 1), 1)
    (1, 3)
    """
    cycle_list = [start]
    next_elem = permutation[start]
    while next_elem != start:
        cycle_list.append(next_elem)
        next_elem = permutation[next_elem]
    return tuple(cycle_list)


def to_cycle_notation(permutation):
    """
    Convert a permutation from one-line notation to cycle notation.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :return: Permutation in cycle notation (list of tuples of cycles in the permutation).

    .. rubric:: Examples

    >>> to_cycle_notation((1, 4, 3, 2, 0))
    [(0, 1, 4), (2, 3)]
    """
    n = len(permutation)
    visited = [False] * n
    cycles = []
    for i in range(n):
        if visited[i]:
            continue
        perm_cycle = cycle(permutation, i)
        for j in range(len(perm_cycle)):
            visited[perm_cycle[j]] = True
        cycles.append(perm_cycle)
    return cycles


def from_cycle_notation(permutation, n):
    """
    Convert a permutation from cycle notation to one-line notation.

    :param permutation: Permutation in cycle notation (list of tuples fo cycles in the permutation).
    :param n: Length of the permutation (needed since length 1 cycles are omitted in the cycle notation).
    :return: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).

    .. rubric:: Examples

    >>> from_cycle_notation([(0, 1, 4), (2, 3)], 6)
    (1, 4, 3, 2, 0, 5)
    """
    image = [-1] * n
    # Record image of all cycles
    for perm_cycle in permutation:
        for i in range(len(perm_cycle) - 1):
            image[perm_cycle[i]] = perm_cycle[i + 1]
        image[perm_cycle[len(perm_cycle) - 1]] = perm_cycle[0]
    # Handle elements mapping to them selves (length one cycles)
    for i in range(len(image)):
        if image[i] == -1:
            image[i] = i
    return tuple(image)


def to_transpositions(permutation):
    """
    Convert a permutation from one-line notation to a composition of transpositions.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :return: Permutation as a composition of transpositions (length 2 cycles, applied from right to left).

    .. rubric:: Examples

    >>> to_transpositions((2, 0, 1))
    [(0, 2), (0, 1)]
    """
    cycles = to_cycle_notation(permutation)
    transpositions = []
    for perm_cycle in cycles:
        for i in range(1, len(perm_cycle)):
            transpositions.append((perm_cycle[0], perm_cycle[i]))
    return transpositions


def from_transpositions(transpositions, n):
    """
    Convert a permutation from a composition of transpositions to one-line notation.

    :param transpositions: Permutation as a composition of transpositions (length 2 cycles, applied from right to left).
    :param n: Length of the permutation (needed since elements mapping to themselves are not represented
        in the composition of transpositions).
    :return: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).

    .. rubric:: Examples

    >>> from_transpositions([(0, 2), (0, 1)], 3)
    (2, 0, 1)
    """
    permutation = tuple(range(n))
    for transposition in reversed(transpositions):
        permutation = swap(permutation, transposition)
    return permutation


def to_one_based(permutation):
    r"""
    Convert a permutation using zero based elements to one based elements.

    In Python code it's most natural to represent :math:`S_n` by the numbers :math:`\{0, 1, \ldots, n-1\}`,
    while in mathematical literature it's standard to use the numbers :math:`\{1, 2, \ldots, n\}`.

    :param permutation: Zero based permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :return: One based permutation in one line notation (length n tuple of the numbers 1, 2, ..., n).

    .. rubric:: Examples

    >>> to_one_based((0, 1, 2))
    (1, 2, 3)
    """
    n = len(permutation)
    one_based_permutation = [0] * n
    for i in range(n):
        one_based_permutation[i] = permutation[i] + 1
    return tuple(one_based_permutation)


def from_one_based(permutation):
    r"""
    Convert a permutation using one based elements to zero based elements.

    In Python code it's most natural to represent :math:`S_n` by the numbers :math:`\{0, 1, \ldots, n-1\}`,
    while in mathematical literature it's standard to use the numbers :math:`\{1, 2, \ldots, n\}`.

    :param permutation: One based permutation in one-line notation (length n tuple of the numbers 1, 2, ..., n-1).
    :return: One based permutation in one line notation (length n tuple of the numbers 0, 1, ..., n).

    .. rubric:: Examples

    >>> from_one_based((1, 2, 3))
    (0, 1, 2)
    """
    n = len(permutation)
    zero_based_permutation = [0] * n
    for i in range(n):
        zero_based_permutation[i] = permutation[i] - 1
    return tuple(zero_based_permutation)


def identity(n):
    """
    Get the identity permutation in :math:`S_n`.

    :param n: Length of the permutation.
    :return: Identity permutation.

    .. rubric:: Examples

    >>> identity(3)
    (0, 1, 2)
    """
    return tuple(range(n))


def inverse(permutation):
    """
    Compute the inverse of a permutation.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :return: Inverse permutation in one line notation (length n tuple of the numbers 0, 1, ..., n-1).

    .. rubric:: Examples

    >>> inverse((1, 4, 3, 2, 0))
    (4, 0, 3, 2, 1)
    """
    # Inverse is given by swapping the two lines in the two-line notation
    permutation_two_lines = to_two_line_notation(permutation)
    return from_two_line_notation((permutation_two_lines[1], permutation_two_lines[0]))


def composition(sigma, pi):
    r"""
    Compute the composition of two permutations, :math:`\sigma \circ \pi, x \mapsto \sigma(\pi(x))`.

    :param sigma: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :param pi: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :return: Composition of the two permutations, which again is a permutation in one-line notation
        (length n tuple of the numbers 0, 1, ..., n-1).
    """
    sigma_two_lines = to_two_line_notation(sigma)

    # Sort sigma according to order in pi
    def sort_key(x):
        return pi.index(x[0])
    permutation = tuple(zip(*sorted(zip(sigma_two_lines[0], sigma_two_lines[1]), key=sort_key)))
    return permutation[1]


def sign(permutation):
    """
    Compute the sign of a permutation.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :return: Sign of the permutation.

    .. rubric:: Examples

    >>> sign((0, 1, 2))
    1
    >>> sign((0, 2, 1))
    -1
    """
    cycles = to_cycle_notation(permutation)
    # The sign of the permutation is equal to -1 to the power of the number of even cycles
    num_even_cycles = 0
    for perm_cycle in cycles:
        if len(perm_cycle) % 2 == 0:
            num_even_cycles += 1
    return (-1)**num_even_cycles


def num_fixed_points(permutation):
    """
    Compute the number of fixed points (elements mapping to themselves) of a permutation.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :return: Number of fixed points in the permutation.

    .. rubric:: Examples

    >>> num_fixed_points((0, 2, 1))
    1
    """
    n = 0
    for i in range(len(permutation)):
        if permutation[i] == i:
            n += 1
    return n


def swap(permutation, transposition):
    r"""
    Swap image of two elements in a permutation (apply a transposition to the permutation).

    Let :math:`\sigma` be the input permutation, :math:`i, j` be the elements of the transposition, and :math:`\pi`
    be the output permutation after the applied transposition. Then we have
    :math:`\pi(i) = \sigma(j), \pi(j) = \sigma(i), \pi(k) = \sigma(k), k \neq i, j`.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :param transposition: Tuple of two elements that should be switched in the permutation.
    :return: Permutation after the applied transposition in one line notation
        (length n tuple of the numbers 0, 1, ..., n-1).

    .. rubric:: Examples

    >>> swap((0, 1, 2, 3), (1, 3))
    (0, 3, 2, 1)
    >>> swap((0, 3, 2, 1), (2, 3))
    (0, 3, 1, 2)
    """
    transposed_permutation = list(permutation)
    i, j = transposition
    transposed_permutation[i], transposed_permutation[j] = permutation[j], permutation[i]
    return tuple(transposed_permutation)


def permute_values(permutation, sequence):
    r"""
    Permute the elements of a sequence with a specified permutation.

    The permutation specifies how each element of the input sequence maps into another element of the input sequence.
    Mathematically we have:
    :math:`\sigma((x_0, x_1, \ldots, x_{n-1})) = (x_{\sigma(0)}, x_{\sigma(1)}, \ldots, x_{\sigma(n-1)})`.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :param sequence: Sequence of elements we wish to permute.
    :return: The permuted sequence.

    .. rubric:: Examples

    >>> permute_values((2, 0, 1), ['a', 'b', 'c'])
    ['c', 'a', 'b']
    """
    permuted_sequence = deepcopy(sequence)
    n = len(permutation)
    for i in range(n):
        permuted_sequence[i] = sequence[permutation[i]]
    return permuted_sequence


def permute_positions(permutation, sequence):
    r"""
    Permute the elements of a sequence with a specified permutation.

    The permutation specifies what position each element in the input sequence is mapped to in the permuted sequence.
    Mathematically we have:
    :math:`\sigma((x_0, x_1, \ldots, x_n)) = (x_{\sigma^{-1}(0)}, x_{\sigma^{-1}(1)}, \ldots, x_{\sigma^{-1}(n-1)})`.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :param sequence: Sequence of elements we wish to permute.
    :return: The permuted sequence.

    .. rubric:: Examples

    >>> permute_positions((2, 0, 1), ['a', 'b', 'c'])
    ['b', 'c', 'a']
    """
    return permute_values(inverse(permutation), sequence)


def permutation_matrix_values(permutation):
    """
    Compute permutation matrix for permutation by value.

    Compute the permutation matrix which when multiplied with a column vector permutes the values of the vector
    according to the given permutation.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :return: n by n permutation matrix (orthogonal matrix containing a single 1 in each row and column).
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    >>> permutation_matrix_values((0, 3, 1, 4, 2))
    array([[1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0]])
    """
    n = len(permutation)
    m = np.zeros((n, n), dtype=int)
    for i in range(n):
        m[i][permutation[i]] = 1
    return m


def permutation_matrix_positions(permutation):
    """
    Compute permutation matrix for permutation by position.

    Compute the permutation matrix which when multiplied with a column vector permutes the position of the elements
    in the vector according to the given permutation.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :return: n by n permutation matrix (orthogonal matrix containing a single 1 in each row and column).
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    >>> permutation_matrix_positions((0, 3, 1, 4, 2))
    array([[1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1],
           [0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0]])
    """
    return permutation_matrix_values(permutation).T


def cyclic_permutations(permutation):
    """
    Get list of all cyclic permutations equivalent to a given permutation.

    :param permutation: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :return: List of tuples containing all cyclic permutations of the input permutation.

    .. rubric:: Examples

    >>> cyclic_permutations((0, 1, 2))
    [(0, 1, 2), (2, 0, 1), (1, 2, 0)]
    """
    n = len(permutation)
    return list(tuple(permutation[i - j] for i in range(n)) for j in range(n))


def circularly_equivalent(permutation1, permutation2):
    """
    Check if two permutations are circularly equivalent.

    Check whether or not two permutations belong to the same circular equivalence class (whether or not we can
    reach the second permutation by successively moving the last element of the first permutation to the front).

    :param permutation1: First permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :param permutation2: Second permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    :return: True/False whether or not the permutations belong to the same equivalence class.

    .. rubric:: Examples

    >>> circularly_equivalent((0, 1, 2), (2, 0, 1))
    True
    """
    return permutation1 in cyclic_permutations(permutation2)


def permutations(n):
    """
    Get a list of all permutations of length n (elements of :math:`S_n`).

    :param n: Length of permutations.
    :return: List of tuples containing all permutations of the elements 0, 1, ..., n-1.

    .. rubric:: Examples

    >>> permutations(3)
    [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    """
    return subset_permutations(n, n)


def num_permutations(n):
    """
    Get the number of length n permutations, n! (number of permutations of n elements = number of elements in
    :math:`S_n`).

    :param n: Length of permutations.
    :return: Number of length n permutations.
    :rtype: int
    """
    return math.factorial(n)


def subset_permutations(n, k):
    """
    Get a list of all permutations of any k elements of the elements 0, 1, ..., n-1.

    :param n: Number of total elements.
    :param k: Number of elements to pick.
    :return: List of tuples containing all permutations of subsets of k elements from the elements 0, 1, ..., n-1.

    .. rubric:: Examples

    >>> subset_permutations(3, 2)
    [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    """
    if k <= 0 or k > n:
        return []
    return list(generate_subset_permutations(list(range(n)), k))


def num_subset_permutations(n, k):
    r"""
    Get the number of permutations of any k elements of the elements 0, 1, ..., n - 1.

    This is given by :math:`\frac{n!}{(n - k)!}`

    :param n: Number of total elements.
    :param k: Number of elements to pick.
    :return: Number of length k permutations of n elements.
    :rtype: int
    """
    return math.factorial(n) / math.factorial(n - k)


def increasing_subset_permutations(n, k):
    """
    Get increasing subset permutations.

    Get a list of all permutations of any k elements of the elements 0, 1, ..., n-1, where the elements appear in
    increasing order.

    :param n: Number of total elements.
    :param k: Number of elements to pick.
    :return: List of tuples containing all permutations of subsets of k increasing elements from the elements
        0, 1, ..., n-1.

    .. rubric:: Examples

    >>> increasing_subset_permutations(3, 2)
    [(0, 1), (0, 2), (1, 2)]
    """
    if k <= 0 or k > n:
        return []
    return list(generate_increasing_subset_permutations(list(range(n)), k))


def num_increasing_subset_permutations(n, k):
    r"""
    Get the number of permutations of any k elements of the elements 0, 1, ..., n - 1, where the elements appear in
    increasing order.

    This is given by :math:`\binom{n}{k} = \frac{n!}{k! (n - k)!}`.

    :param n: Number of total elements.
    :param k: Number of elements to pick.
    :return: Number of length k permutations of n elements.
    :rtype: int
    """
    return int(binom(n, k))


def generate_subset_permutations(array, k):
    """
    Generate all permutations of any k elements of an array.

    :param array: Input list of elements.
    :param k: Number of elements to pick from the array.
    :return: Generator generating all permutations of subsets of k elements from the input array.
    """
    if k == 0:
        yield tuple()
    else:
        for i in range(len(array)):
            for perm in generate_subset_permutations(array[0:i] + array[i + 1:len(array)], k - 1):
                yield (array[i],) + perm


def generate_increasing_subset_permutations(array, k):
    """
    Generate all increasing subset permutations.

    Generate all permutations of any k elements of an array, where the elements appear in
    increasing order, based on the element order in the input array.

    :param array: Input list of elements.
    :param k: Number of elements to pick from the array.
    :return: Generator generating all permutations of subsets of k increasing elements from the input array.
    """
    if k == 0:
        yield tuple()
    else:
        for i in range(len(array) - k + 1):
            for perm in generate_increasing_subset_permutations(array[i + 1:], k - 1):
                yield (array[i],) + perm


def generate_random_permutation(n):
    """
    Generate a random length n permutation (element of :math:`S_n`).

    :param n: Length of permutation.
    :return: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).
    """
    # Knuth shuffle
    array = list(range(n))
    for i in range(n - 1):
        j = random.randint(0, n - 1 - i) + i
        array[i], array[j] = array[j], array[i]
    return tuple(array)


def generate_random_subset_permutation(n, k):
    """
    Generate a random length k permutation of the elements 0, 1, ..., n-1.

    :param n: Number of elements to pick from.
    :param k: Length of permutation.
    :return: Permutation in one-line notation (length k tuple of the numbers 0, 1, ..., n-1).
    """
    return generate_random_permutation(n)[0:k]


def generate_random_increasing_subset_permutation(n, k):
    """
    Generate random increasing subset permutation.

    Generate a random length k permutation of the elements 0, 1, ..., n-1 where the elements appear
    in increasing order.

    :param n: Number of elements to pick from.
    :param k: Length of permutation.
    :return: Permutation in one-line notation (length k tuple of increasing numbers from the set 0, 1, ..., n-1).
    """
    p = generate_random_subset_permutation(n, k)
    return sorted(p)


def construct_permutation(domain, codomain, n):
    r"""
    Construct a permutation satisfying given constraints.

    Construct a length n permutation which maps the given domain into the given codomain.
    Let :math:`(x_0, x_1, \ldots, x_k)` be the given domain and :math:`(y_0, y_1, \ldots, y_k)` be the given codomain.
    Then the output permutation :math:`\sigma` should satisfy :math:`\sigma(x_i) = y_i, i = 0, 1, \ldots, k`.

    :param domain: Domain for which the permutation is prescribed. Subset of the set {0, 1, ..., n - 1}.
    :param codomain: Image for each element in the prescribed domain. Subset of the set {0, 1, ..., n - 1}. Need to
        be the same length as the domain.
    :param n: Length of permutation.
    :return: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).

    .. rubric:: Examples

    >>> construct_permutation([0, 1], [1, 0], 3)
    (1, 0, 2)
    """
    assert len(domain) == len(codomain)
    permutation = list(range(n))
    for i in range(len(domain)):
        if permutation[domain[i]] != codomain[i]:
            idx = permutation.index(codomain[i])
            permutation[domain[i]], permutation[idx] = permutation[idx], permutation[domain[i]]
    return tuple(permutation)


def construct_permutation_general(domain, codomain, by_value=True):
    r"""
    Construct a permutation from two general sequences of (the same) elements.

    Construct a length n permutation which maps the given domain into the given codomain.
    Let :math:`x = (x_0, x_1, \ldots, x_k)` be the given domain and :math:`y = (y_0, y_1, \ldots, y_k)` be the given
    codomain. Then the output permutation :math:`\sigma` should satisfy :math:`\sigma(x_i) = y_i, i = 0, 1, \ldots, k`.

    :param domain: Domain for which the permutation is prescribed.
    :param codomain: Image for each element in the prescribed domain. Need to be the same length as the domain.
    :param by_value: Whether to use the "by value" or "by position" interpretation of the permutation of an array of
        objects.
    :return: Permutation in one-line notation (length n tuple of the numbers 0, 1, ..., n-1).

    .. rubric:: Examples

    >>> construct_permutation_general(['a', 'b', 'c', 'd'], ['d', 'b', 'a', 'c'])
    (3, 1, 0, 2)
    >>> construct_permutation_general(['a', 'b', 'c', 'd'], ['d', 'b', 'a', 'c'], by_value=False)
    (2, 1, 3, 0)
    """
    # Enumerate domain elements
    domain_element_index = {}
    for i in range(len(domain)):
        domain_element_index[domain[i]] = i

    # Construct permutation
    permutation = []
    for c in codomain:
        permutation.append(domain_element_index[c])

    permutation = tuple(permutation)
    if by_value:
        return permutation
    return inverse(permutation)


def _is_iterable(a):
    try:
        iter(a)
        return True
    except TypeError:
        return False


if __name__ == "__main__":
    import doctest
    doctest.testmod()
