"""
Functionality for generating Python code used to evaluate barycentric polynomials.
"""

from polynomials_on_simplices.algebra.multiindex import generate_all, norm
from polynomials_on_simplices.generic_tools.code_generation_utils import CodeWriter
from polynomials_on_simplices.generic_tools.str_utils import (
    split_long_str, str_dot_product, str_exponent, str_multi_product, str_multi_sum, str_number, str_product,
    str_sequence)
from polynomials_on_simplices.polynomial.code_generation.generate_polynomial_functions import (
    generate_docstring, generate_function_specific_name)
from polynomials_on_simplices.polynomial.polynomials_base import get_dimension


def generate_function_general(m, r):
    r"""
    Generate code for evaluating a general degree r barycentric polynomial on the m-dimensional unit simplex.

    .. math:: p(x) = \sum_{i = 0}^{\dim(\mathcal{P}_r(\Delta_c^m)) - 1} a_{\nu_i} x^{\nu} (1 - |x|)^{r - |\nu|},

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the domain of the polynomial.
    :param int r: Degree of the polynomial space.
    :return: Python code for evaluating the polynomial
    :rtype: str
    """
    code = CodeWriter()
    code.wl("def barycentric_polynomial_" + str(m) + str(r) + "(a, x):")
    code.inc_indent()
    latex_str = _generate_barycentric_polynomial_latex_str(m, r)
    code.wc(generate_docstring(m, r, "barycentric", latex_str))
    coeff_strs = str_sequence("a", get_dimension(r, m), indexing='c', index_style="list")
    code.wc(_generate_function_body_scalar_valued(m, r, coeff_strs))
    code.dec_indent()
    return code.code


def generate_function_specific(m, r, a):
    r"""
    Generate code for evaluating the degree r barycentric polynomial on the m-dimensional unit simplex with given
    basis coefficients a.

    .. math:: p(x) = \sum_{i = 0}^{\dim(\mathcal{P}_r(\Delta_c^m)) - 1} a_{\nu_i} x^{\nu} (1 - |x|)^{r - |\nu|},

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the domain of the polynomial.
    :param int r: Degree of the polynomial space.
    :param a: Coefficients for the polynomial in the barycentric basis for :math:`\mathcal{P}_r (\Delta_c^m)`.
        :math:`\text{a}[i] = a_{\nu_i}`, where :math:`\nu_i` is the i:th multi-index in the sequence of all
        multi-indices of dimension m with norm :math:`\leq r`
        (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
    :type a: Union[Iterable[float], Iterable[n-dimensional vector]]
    :return: Python code for evaluating the polynomial
    :rtype: str
    """
    code = CodeWriter()
    fn_name = "eval_barycentric_polynomial_" + str(m) + str(r) + '_' + generate_function_specific_name(a)
    code.wl("def " + fn_name + "(x):")
    code.inc_indent()
    code.wc(_generate_function_body(m, r, a))
    code.dec_indent()
    return code.code, fn_name


def generate_barycentric_basis_fn(nu, r, str_type="python"):
    r"""
    Generate string representation for a barycentric basis polynomial on the m-dimensional unit simplex
    (:math:`\Delta_c^m`), :math:`p_{\nu, r} (x) = x^{\nu} (1 - |x|)^{r - |\nu|}`, where m is equal to the length of nu.

    :param nu: Multi-index indicating which barycentric basis polynomial should be generated.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :param str_type: What kind of string should be generated. Can be either "python" or "latex".
    :return: Python code for evaluating the barycentric base polynomial as specified by nu and r.
    :rtype: str

    .. rubric:: Examples

    >>> generate_barycentric_basis_fn(1, 2)
    'x * (1 - x)'
    >>> generate_barycentric_basis_fn((1, 1), 3)
    'x[0] * x[1] * (1 - x[0] - x[1])'
    >>> generate_barycentric_basis_fn((1, 1), 3, str_type="latex")
    'x_1 x_2 (1 - x_1 - x_2)'
    """
    try:
        m = len(nu)
        if m == 1:
            nu = nu[0]
    except TypeError:
        m = 1
    if str_type == "python":
        multiplication_character = "*"
    else:
        multiplication_character = ""
    if m == 1:
        # x**nu * (1 - x)**(r - nu)
        p1 = str_exponent("x", str_number(nu), str_type)
        p2 = str_exponent("(1 - x)", str_number(r - nu), str_type)
        return str_product(p1, p2, multiplication_character)
    else:
        # x1**nu[0] * x2**nu[1] * ... * (1 - x1 - x2 - ...)**(r - |nu|)
        if str_type == "python":
            variables = str_sequence("x", m, indexing="c", index_style="list")
        else:
            variables = str_sequence("x", m, index_style="latex_subscript")
        factors = [str_exponent(variables[i], str_number(nu[i]), str_type) for i in range(m)]
        p1 = str_multi_product(factors, multiplication_character)

        summands = ["-" + variables[i] for i in range(m)]
        summands = ["1"] + summands
        base = "(" + str_multi_sum(summands) + ")"
        exp = str_number(r - norm(nu))
        p2 = str_exponent(base, exp, str_type)

        return str_product(p1, p2, multiplication_character)


def generate_barycentric_basis(m, r, str_type="python"):
    r"""
    Generate string representation for all barycentric base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^m)`.

    :param int m: Dimension of the unit simplex.
    :param int r: Degree of the polynomial space.
    :param str_type: What kind of strings should be generated. Can be either "python" or "latex".
    :return: List of codes for evaluating each of the base polynomials.
    :rtype: List[str]
    """
    basis = []
    for mi in generate_all(m, r):
        basis.append(generate_barycentric_basis_fn(mi, r, str_type))
    return basis


def generate_function_eval_specific_scalar_valued(m, r, a, prettify_coefficients=False):
    r"""
    Generate code for evaluating a specific scalar valued degree r polynomial on the m-dimensional unit simplex
    (:math:`\Delta_c^m`), expressed in the barycentric basis.

    .. math::

        p_{\nu, r} : \Delta_c^m \to \mathbb{R},

        p_{\nu, r}(x)=\sum_{i = 0}^{\dim(\mathcal{P}_r(\Delta_c^m)) - 1} a_{\nu_i} x^{\nu_i} (1 - |x|)^{r - |\nu_i|},

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the unit simplex.
    :param int r: Degree of the polynomial space.
    :param a: Coefficients for the polynomial in the barycentric basis for :math:`\mathcal{P}_r (\Delta_c^m)`.
        :math:`\text{a}[i] = a_{\nu_i}`, where :math:`\nu_i` is the i:th multi-index in the sequence of all
        multi-indices of dimension m with norm :math:`\leq r`
        (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
    :type a: Iterable[float]
    :param bool prettify_coefficients: Whether or not coefficients in the a array should be prettified in the
        generated code (e.g. converting 0.25 -> 1 / 4).
    :return: Python code for evaluating the barycentric base polynomial as specified by m, r and a.
    :rtype: str
    """
    if isinstance(a[0], str):
        coeff_strs = [c for c in a]
    else:
        coeff_strs = [str_number(c, prettify_fractions=prettify_coefficients) for c in a]
    basis_strs = generate_barycentric_basis(m, r)
    return str_dot_product(coeff_strs, basis_strs, multiplication_character="*")


def generate_function_eval_specific_vector_valued(m, r, a):
    r"""
    Generate code for evaluating a specific vector valued degree r polynomial on the m-dimensional unit simplex
    (:math:`\Delta_c^m`), expressed in the barycentric basis.

    .. math::

        p_{\nu, r} : \Delta_c^m \to \mathbb{R}^n, n > 1,

        p_{\nu, r}(x)=\sum_{i = 0}^{\dim(\mathcal{P}_r(\Delta_c^m)) - 1} a_{\nu_i} x^{\nu_i} (1 - |x|)^{r - |\nu_i|},

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the unit simplex.
    :param int r: Degree of the polynomial space.
    :param a: Coefficients for the polynomial in the barycentric basis for :math:`\mathcal{P}_r (\Delta_c^m)`.
        :math:`\text{a}[i] = a_{\nu_i}`, where :math:`\nu_i` is the i:th multi-index in the sequence of all
        multi-indices of dimension m with norm :math:`\leq r`
        (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
    :type a: Iterable[n-dimensional vector]
    :return: Python code for evaluating the barycentric base polynomial as specified by m, r and a.
    :rtype: str
    """
    code = ""

    code += "np.array(["
    code += generate_function_eval_specific_scalar_valued(m, r, a[:, 0])
    for i in range(1, len(a[0])):
        code += ", " + generate_function_eval_specific_scalar_valued(m, r, a[:, i])
    code += "])"

    return code


def _generate_barycentric_polynomial_latex_str(m, r):
    coeff_strs = str_sequence("a", get_dimension(r, m), indexing='c', index_style='latex_subscript')
    basis_strs = generate_barycentric_basis(m, r, str_type="latex")
    latex_str = str_dot_product(coeff_strs, basis_strs)
    parts = split_long_str(latex_str, "+", 100)
    latex_str = "\\\\\n&\\quad +".join(parts)
    return "b(x) = " + latex_str


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
