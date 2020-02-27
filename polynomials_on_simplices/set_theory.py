"""Basic set theoretic calculations."""


import itertools


def cartesian_product(s, t):
    r"""
    Compute the Cartesian product of two finite sets, :math:`S \times T`.

    :param set s: First set in the Cartesian product.
    :param set t: Second set in the Cartesian product.
    :return: The Cartesian product of the two sets.
    :rtype: set

    >>> prod = cartesian_product({1, 2}, {1, 2})
    >>> prod == {(1, 1), (1, 2), (2, 1), (2, 2)}
    True
    """
    return set(itertools.product(s, t))


def nfold_cartesian_product(s, n):
    r"""
    Compute the n-fold Cartesian product of a finite set,
    :math:`\underbrace{S \times S \times \ldots \times S}_{n \text{ times}}`.

    :param s: Set we take the n-fold Cartesian product of.
    :param int n: Number of times we take the Cartesian product.
    :return: n-fold Cartesian product of the given set.
    :rtype: set

    >>> prod = nfold_cartesian_product({1, 2}, 3)
    >>> prod == {(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)}
    True
    """
    # Handle special cases
    if n == 0:
        return set()
    if n == 1:
        return s
    if n == 2:
        return cartesian_product(s, s)
    # Compute n-fold Cartesian product
    p0 = nfold_cartesian_product(s, n - 1)
    product = set()
    for sub_tuple in p0:
        for element in s:
            product.add((element,) + sub_tuple)
    return product


if __name__ == "__main__":
    import doctest
    doctest.testmod()
