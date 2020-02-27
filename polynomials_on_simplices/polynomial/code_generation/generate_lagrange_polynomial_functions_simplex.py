"""
Functionality for generating Python code used to evaluate Lagrange polynomials.
"""

import numpy as np

from polynomials_on_simplices.algebra.multiindex import generate_all
from polynomials_on_simplices.generic_tools.code_generation_utils import CodeWriter
from polynomials_on_simplices.generic_tools.str_utils import split_long_str, str_dot_product, str_number, str_sequence
from polynomials_on_simplices.polynomial.code_generation.generate_monomial_polynomial_functions import \
    generate_function_eval_specific_scalar_valued as generate_monomial_polynomial_function_eval_specific_scalar_valued
from polynomials_on_simplices.polynomial.code_generation.generate_polynomial_functions import (
    generate_docstring, generate_function_specific_name)
from polynomials_on_simplices.polynomial.polynomials_base import get_dimension


def generate_function_general(m, r):
    r"""
    Generate code for evaluating a general degree r Lagrange polynomial on the m-dimensional unit simplex.

    .. math:: l(x) = \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} l_{\nu, r}(x),

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the domain of the polynomial.
    :param int r: Degree of the polynomial space.
    :return: Python code for evaluating the polynomial
    :rtype: str
    """
    from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import (
        unique_identifier_lagrange_basis)
    code = CodeWriter()
    code.wl("def lagrange_polynomial_" + str(m) + str(r) + "(a, x):")
    code.inc_indent()
    latex_str = _generate_lagrange_polynomial_latex_str(m, r)
    code.wc(generate_docstring(m, r, unique_identifier_lagrange_basis(), latex_str))
    code.wc(_generate_function_body_general_scalar_valued(m, r))
    code.dec_indent()
    return code.code


def generate_function_specific(m, r, a):
    r"""
    Generate code for evaluating the degree r Lagrange polynomial on an m-dimensional domain with given
    basis coefficients a.

    .. math:: p(x) = \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} l_{\nu, r}(x),

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the domain of the polynomial.
    :param int r: Degree of the polynomial space.
    :param a: Coefficients for the polynomial in the Lagrange basis for :math:`\mathcal{P}_r (\Delta_c^m)`.
        :math:`\text{a}[i] = a_{\nu_i}`, where :math:`\nu_i` is the i:th multi-index in the sequence of all
        multi-indices of dimension m with norm :math:`\leq r`
        (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
    :type a: Union[Iterable[float], Iterable[n-dimensional vector]]
    :return: Python code for evaluating the polynomial
    :rtype: str
    """
    code = CodeWriter()
    fn_name = "eval_lagrange_polynomial_" + str(m) + str(r) + '_' + generate_function_specific_name(a)
    code.wl("def " + fn_name + "(x):")
    code.inc_indent()
    code.wc(_generate_function_body(m, r, a))
    code.dec_indent()
    return code.code, fn_name


def generate_lagrange_basis_fn(nu, r):
    r"""
    Generate code for evaluating a Lagrange basis polynomial on the m-dimensional unit simplex, where m is
    equal to the length of nu.

    :param nu: Multi-index indicating which Lagrange basis polynomial code should be generated for.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :return: Python code for evaluating the Lagrange base polynomial as specified by nu.
    :rtype: str

    .. rubric:: Examples

    >>> generate_lagrange_basis_fn(1, 1)
    'x'
    >>> generate_lagrange_basis_fn((1, 1), 2)
    '4 * x[0] * x[1]'
    """
    try:
        m = len(nu)
        if m == 1:
            nu = nu[0]
    except TypeError:
        m = 1
    from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import (
        get_lagrange_basis_fn_coefficients)
    from polynomials_on_simplices.polynomial.code_generation.generate_monomial_polynomial_functions import (
        generate_monomial_basis)
    coeffs = [str_number(a) for a in get_lagrange_basis_fn_coefficients(nu, r)]
    return str_dot_product(coeffs, generate_monomial_basis(m, r), "*")


def generate_lagrange_basis(m, r):
    r"""
    Generate code for evaluating all Lagrange base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^m)`.

    :param int m: Dimension of the domain.
    :param int r: Degree of the polynomial space.
    :return: List of codes for evaluating each of the base polynomials.
    :rtype: List[str]
    """
    basis = []
    for mi in generate_all(m, r):
        basis.append(generate_lagrange_basis_fn(mi, r))
    return basis


def generate_function_eval_general_scalar_valued(m, r):
    r"""
    Generate code for evaluating a general scalar valued degree r polynomial on the m-dimensional unit simplex
    (:math:`\Delta_c^m`), expressed in the Lagrange basis.

    .. math::

        p_{\nu, r}(x)=\sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} x^{\nu_i},

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the domain.
    :param int r: Degree of the polynomial space.
    :return: Python code for evaluating a general degree r Lagrange polynomial on an m-dimensional domain.
    :rtype: str
    """
    coeff_strs = str_sequence("a", get_dimension(r, m), indexing='c', index_style="list")
    basis_strs = generate_lagrange_basis(m, r)
    for i in range(len(basis_strs)):
        if basis_strs[i].find("+") != -1 or basis_strs[i].find("-") != -1:
            basis_strs[i] = "(" + basis_strs[i] + ")"
    return str_dot_product(coeff_strs, basis_strs, multiplication_character="*")


def generate_function_eval_specific_scalar_valued(m, r, a, prettify_coefficients=False):
    r"""
    Generate code for evaluating a specific scalar valued degree r polynomial on the m-dimensional unit simplex
    (:math:`\Delta_c^m`), expressed in the Lagrange basis.

    .. math::

        p : \Delta_c^m \to \mathbb{R},

        p_{\nu, r}(x)=\sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} x^{\nu_i},

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the domain.
    :param int r: Degree of the polynomial space.
    :param a: Coefficients for the polynomial in the Lagrange basis for :math:`\mathcal{P}_r (\mathbb{R}^m)`.
        :math:`\text{a}[i] = a_{\nu_i}`, where :math:`\nu_i` is the i:th multi-index in the sequence of all
        multi-indices of dimension m with norm :math:`\leq r`
        (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
    :type a: Iterable[float]
    :param bool prettify_coefficients: Whether or not coefficients in the a array should be prettified in the
        generated code (e.g. converting 0.25 -> 1 / 4).
    :return: Python code for evaluating the Lagrange polynomial as specified by m, r and a.
    :rtype: str
    """
    from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import (
        get_lagrange_base_coefficients)
    l = get_lagrange_base_coefficients(r, m)
    al = np.dot(l, a)
    return generate_monomial_polynomial_function_eval_specific_scalar_valued(
        m, r, al, prettify_coefficients=prettify_coefficients)


def generate_function_eval_specific_vector_valued(m, r, a):
    r"""
    Generate code for evaluating a specific vector valued degree r polynomial on the m-dimensional unit simplex
    (:math:`\Delta_c^m`), expressed in the Lagrange basis.

    .. math::

        p : \Delta_c^m \to \mathbb{R}^n, n > 1,

        p_{\nu, r}(x)=\sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} x^{\nu_i},

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the domain.
    :param int r: Degree of the polynomial space.
    :param a: Coefficients for the polynomial in the Lagrange basis for :math:`\mathcal{P}_r (\mathbb{R}^m)`.
        :math:`\text{a}[i] = a_{\nu_i}`, where :math:`\nu_i` is the i:th multi-index in the sequence of all
        multi-indices of dimension m with norm :math:`\leq r`
        (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
    :type a: Iterable[n-dimensional vector]
    :return: Python code for evaluating the Lagrange base polynomial as specified by m, r and a.
    :rtype: str
    """
    code = ""

    code += "np.array(["
    code += generate_function_eval_specific_scalar_valued(m, r, a[:, 0])
    for i in range(1, len(a[0])):
        code += ", " + generate_function_eval_specific_scalar_valued(m, r, a[:, i])
    code += "])"

    return code


def _generate_lagrange_polynomial_latex_str(m, r):
    coeff_strs = str_sequence("a", get_dimension(r, m), indexing='c', index_style='latex_subscript')
    from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import lagrange_basis_latex
    basis_strs = ["(" + basis + ")" for basis in lagrange_basis_latex(r, m)]
    latex_str = str_dot_product(coeff_strs, basis_strs)
    parts = split_long_str(latex_str, "+", 100)
    latex_str = "\\\\\n&\\quad +".join(parts)
    return "p(x) = " + latex_str


def _generate_function_body_general_scalar_valued(m, r):
    code = generate_function_eval_general_scalar_valued(m, r)
    parts = split_long_str(code, " +", 100)
    multi_line = (len(parts) > 1)
    code = "\n        +".join(parts)
    if multi_line:
        return "return (" + code + ")"
    else:
        return "return " + code


def _generate_function_body(m, r, a):
    try:
        len(a[0])
        return _generate_function_body_vector_valued(m, r, a)
    except TypeError:
        return _generate_function_body_scalar_valued(m, r, a)


def _generate_function_body_scalar_valued(m, r, a):
    code = generate_function_eval_specific_scalar_valued(m, r, a)
    if len(code) > 100:
        parts = split_long_str(code, " +", 100)
        multi_line = (len(parts) > 1)
        code = "\n        +".join(parts)
        if multi_line:
            return "return (" + code + ")"
        else:
            return "return " + code
    return "return " + code


def _generate_function_body_vector_valued(m, r, a):
    code = generate_function_eval_specific_vector_valued(m, r, a)
    if len(code) > 100:
        parts = split_long_str(code, ", ", 100)
        code = ",\n                 ".join(parts)
    return "return " + code


if __name__ == "__main__":
    import doctest
    doctest.testmod()
