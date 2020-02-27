"""
Functions used to generate the cache of coefficients for Lagrange basis polynomials found in
polynomials_unit_simplex_lagrange_basis_cache.py.
"""

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.generic_tools.code_generation_utils import CodeWriter
from polynomials_on_simplices.generic_tools.str_utils import split_long_str, str_number
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import (
    generate_lagrange_base_coefficients)


def _generate_lagrange_basis_coefficients_cache():
    cache = CodeWriter()
    cache.wl("[")
    cache.inc_indent()
    for n in range(1, 5):
        cache.wl("# n = " + str(n))
        cache.wl("[")
        cache.inc_indent()
        for r in range(1, 5):
            coeffs = _generate_lagrange_basis_coefficients_str(n, r)
            mis = multiindex.generate_all(n, r)
            cache.wl("# r = " + str(r))
            cache.wl("{")
            cache.inc_indent()
            for a, m in zip(coeffs, mis):
                m_str = str(tuple(m))
                a_str = str(a)
                if 4 * cache.indent_depth + len(m_str) + 2 + len(a_str) + 1 > 120:
                    threshold = 120 - 4 * cache.indent_depth - len(m_str) - 3
                    a_str_parts = split_long_str(a_str, sep=", ", long_str_threshold=threshold + 1,
                                                 sep_replacement_before=",")
                    cache.wl(m_str + ": " + a_str_parts[0])
                    offset_length = len(m_str) + 3
                    offset = str(" ") * offset_length
                    for i in range(1, len(a_str_parts) - 1):
                        cache.wl(offset + a_str_parts[i])
                    cache.wl(offset + a_str_parts[-1] + ",")
                else:
                    cache.wl(m_str + ": " + a_str + ",")
            cache.dec_indent()
            cache.wl("}, ")
        cache.dec_indent()
        cache.wl("], ")
    cache.dec_indent()
    cache.wl("]")

    s = cache.code
    s = s.replace("1 * ", "")
    s = s.replace(" * 1", "")
    s = s.replace("+ -", "- ")
    return s


def _generate_lagrange_basis_coefficients_str(n, r):
    """
    Generate list of strings of all coefficients for all Lagrange base polynomials of degree r on the
    n-dimensional unit simplex.

    :param int n: Dimension of the unit simplex.
    :param int r: Degree of the polynomial space.
    :return: List of strings, where row i is a string representation of a list containing all coefficients of the
        i:th Lagrange basis function.
    """
    a = generate_lagrange_base_coefficients(r, n)
    a_str = []
    for i in range(len(a)):
        a_str_i_fractions = ", ".join([str_number(a[j][i]) for j in range(len(a[i]))])
        a_str_i = "[" + a_str_i_fractions + "]"
        a_str.append(a_str_i)
    return a_str


if __name__ == "__main__":
    print(_generate_lagrange_basis_coefficients_cache())
