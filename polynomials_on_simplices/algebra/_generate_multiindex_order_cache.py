"""
Functions used to generate the cache of the ordering of multiindices found in multiindex_order_cache.py.
"""

from polynomials_on_simplices.algebra.multiindex import generate_all
from polynomials_on_simplices.generic_tools.code_generation_utils import CodeWriter


def generate_multiindex_order_cache():
    """
    Generate a cache of all multiindices and their order in the sequence of all n-dimensional multiindices of norm
    <= r, for a collection of (n, r) combinations.

    :return: Nested dictionaries mapping a (n, r) tuple and a multi-index to its position index in the sequence of all
        multiindices, encoded as a Python code string.
    :rtype: str
    """
    cache = CodeWriter()
    cache.wl('{')
    cache.inc_indent()
    for n in [2, 3, 4]:
        for r in range(1, 7):
            cache.wl("# (n, r) = " + str((n, r)))
            cache.wl(str((n, r)) + ": {")
            cache.inc_indent()
            mis = generate_all(n, r)
            for i in range(len(mis)):
                cache.bl("MultiIndex(" + str(mis[i]) + "): " + str(i))
                if i < len(mis) - 1:
                    cache.el(',')
                else:
                    cache.el("")
            cache.dec_indent()
            if n == 4 and r == 6:
                cache.wl('}')
            else:
                cache.wl('},')
    cache.dec_indent()
    cache.wl('}')
    return cache.code


if __name__ == "__main__":
    print(generate_multiindex_order_cache())
