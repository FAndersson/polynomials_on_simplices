"""
Functionality for generating Python code used to evaluate monomial polynomials.
"""


from polynomials_on_simplices.algebra.multiindex import generate_all
from polynomials_on_simplices.generic_tools.code_generation_utils import CodeWriter
from polynomials_on_simplices.generic_tools.str_utils import (
    split_long_str, str_dot_product, str_exponent, str_multi_product, str_number, str_product, str_sequence, str_sum)
from polynomials_on_simplices.polynomial.code_generation.generate_polynomial_functions import (
    generate_docstring, generate_function_specific_name)
from polynomials_on_simplices.polynomial.polynomials_base import get_dimension


def generate_function_general(m, r):
    r"""
    Generate code for evaluating a general degree r monomial polynomial on an m-dimensional domain.

    .. math:: p(x) = \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} x^{\nu},

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the domain of the polynomial.
    :param int r: Degree of the polynomial space.
    :return: Python code for evaluating the polynomial
    :rtype: str
    """
    from polynomials_on_simplices.polynomial.polynomials_monomial_basis import unique_identifier_monomial_basis
    code = CodeWriter()
    code.wl("def monomial_polynomial_" + str(m) + str(r) + "(a, x):")
    code.inc_indent()
    latex_str = _generate_monomial_polynomial_latex_str(m, r)
    code.wc(generate_docstring(m, r, unique_identifier_monomial_basis(), latex_str))
    coeff_strs = str_sequence("a", get_dimension(r, m), indexing='c', index_style="list")
    code.wc(_generate_function_body_scalar_valued(m, r, coeff_strs))
    code.dec_indent()
    return code.code


def generate_function_specific(m, r, a):
    r"""
    Generate code for evaluating the degree r monomial polynomial on an m-dimensional domain with given
    basis coefficients a.

    .. math:: p(x) = \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} x^{\nu},

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the domain of the polynomial.
    :param int r: Degree of the polynomial space.
    :param a: Coefficients for the polynomial in the monomial basis for :math:`\mathcal{P}_r (\mathbb{R}^m)`.
        :math:`\text{a}[i] = a_{\nu_i}`, where :math:`\nu_i` is the i:th multi-index in the sequence of all
        multi-indices of dimension m with norm :math:`\leq r`
        (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
    :type a: Union[Iterable[float], Iterable[n-dimensional vector]]
    :return: Python code for evaluating the polynomial
    :rtype: str
    """
    code = CodeWriter()
    fn_name = "eval_monomial_polynomial_" + str(m) + str(r) + '_' + generate_function_specific_name(a)
    code.wl("def " + fn_name + "(x):")
    code.inc_indent()
    code.wc(_generate_function_body(m, r, a))
    code.dec_indent()
    return code.code, fn_name


_monomials_basis_fns_code_cache = [
    {
        0: '1',
        1: 'x',
        2: 'x**2',
        3: 'x**3',
        4: 'x**4'
    },
    {
        (0, 0): '1',
        (1, 0): 'x[0]',
        (2, 0): 'x[0]**2',
        (3, 0): 'x[0]**3',
        (4, 0): 'x[0]**4',
        (0, 1): 'x[1]',
        (1, 1): 'x[0] * x[1]',
        (2, 1): 'x[0]**2 * x[1]',
        (3, 1): 'x[0]**3 * x[1]',
        (0, 2): 'x[1]**2',
        (1, 2): 'x[0] * x[1]**2',
        (2, 2): 'x[0]**2 * x[1]**2',
        (0, 3): 'x[1]**3',
        (1, 3): 'x[0] * x[1]**3',
        (0, 4): 'x[1]**4',
    },
    {
        (0, 0, 0): "1",
        (1, 0, 0): "x[0]",
        (2, 0, 0): "x[0]**2",
        (3, 0, 0): "x[0]**3",
        (4, 0, 0): "x[0]**4",
        (0, 1, 0): "x[1]",
        (1, 1, 0): "x[0] * x[1]",
        (2, 1, 0): "x[0]**2 * x[1]",
        (3, 1, 0): "x[0]**3 * x[1]",
        (0, 2, 0): "x[1]**2",
        (1, 2, 0): "x[0] * x[1]**2",
        (2, 2, 0): "x[0]**2 * x[1]**2",
        (0, 3, 0): "x[1]**3",
        (1, 3, 0): "x[0] * x[1]**3",
        (0, 4, 0): "x[1]**4",
        (0, 0, 1): "x[2]",
        (1, 0, 1): "x[0] * x[2]",
        (2, 0, 1): "x[0]**2 * x[2]",
        (3, 0, 1): "x[0]**3 * x[2]",
        (0, 1, 1): "x[1] * x[2]",
        (1, 1, 1): "x[0] * x[1] * x[2]",
        (2, 1, 1): "x[0]**2 * x[1] * x[2]",
        (0, 2, 1): "x[1]**2 * x[2]",
        (1, 2, 1): "x[0] * x[1]**2 * x[2]",
        (0, 3, 1): "x[1]**3 * x[2]",
        (0, 0, 2): "x[2]**2",
        (1, 0, 2): "x[0] * x[2]**2",
        (2, 0, 2): "x[0]**2 * x[2]**2",
        (0, 1, 2): "x[1] * x[2]**2",
        (1, 1, 2): "x[0] * x[1] * x[2]**2",
        (0, 2, 2): "x[1]**2 * x[2]**2",
        (0, 0, 3): "x[2]**3",
        (1, 0, 3): "x[0] * x[2]**3",
        (0, 1, 3): "x[1] * x[2]**3",
        (0, 0, 4): "x[2]**4",
    }
]


def generate_monomial_basis_fn(nu):
    r"""
    Generate code for evaluating a monomial basis polynomial on an m-dimensional domain,
    :math:`p_{\nu} (x) = x^{\nu}`, where m is equal to the length of nu.

    :param nu: Multi-index indicating which monomial basis polynomial code should be generated for.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :return: Python code for evaluating the monomial base polynomial as specified by nu.
    :rtype: str

    .. rubric:: Examples

    >>> generate_monomial_basis_fn(0)
    '1'
    >>> generate_monomial_basis_fn(1)
    'x'
    >>> generate_monomial_basis_fn((1, 1))
    'x[0] * x[1]'
    """
    try:
        m = len(nu)
        if m == 1:
            nu = nu[0]
    except TypeError:
        m = 1
    if m == 1:
        if nu in _monomials_basis_fns_code_cache[m - 1]:
            return _monomials_basis_fns_code_cache[m - 1][nu]
        else:
            return str_exponent("x", str(nu))
    else:
        # x[0]**nu[0] * x[1]**nu[1] * ... * x[m - 1]**nu[m - 1]
        if m <= 3:
            if tuple(nu) in _monomials_basis_fns_code_cache[m - 1]:
                return _monomials_basis_fns_code_cache[m - 1][tuple(nu)]
        variables = str_sequence("x", m, indexing="c", index_style="list")
        factors = [str_exponent(variables[i], str(nu[i])) for i in range(m)]
        return str_multi_product(factors, "*")


def generate_monomial_basis(m, r):
    r"""
    Generate code for evaluating all monomial base polynomials for the space :math:`\mathcal{P}_r(\mathbb{R}^m)`.

    :param int m: Dimension of the domain.
    :param int r: Degree of the polynomial space.
    :return: List of codes for evaluating each of the base polynomials.
    :rtype: List[str]
    """
    basis = []
    for mi in generate_all(m, r):
        basis.append(generate_monomial_basis_fn(mi))
    return basis


def generate_function_eval_specific_scalar_valued(m, r, a, prettify_coefficients=False):
    r"""
    Generate code for evaluating a specific scalar valued degree r polynomial on an m-dimensional domain,
    expressed in the monomial basis.

    .. math::

        p : \mathbb{R}^m \to \mathbb{R},

        p_{\nu, r}(x)=\sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} x^{\nu_i},

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the domain.
    :param int r: Degree of the polynomial space.
    :param a: Coefficients for the polynomial in the monomial basis for :math:`\mathcal{P}_r (\mathbb{R}^m)`.
        :math:`\text{a}[i] = a_{\nu_i}`, where :math:`\nu_i` is the i:th multi-index in the sequence of all
        multi-indices of dimension m with norm :math:`\leq r`
        (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
    :type a: Iterable[float]
    :param bool prettify_coefficients: Whether or not coefficients in the a array should be prettified in the
        generated code (e.g. converting 0.25 -> 1 / 4).
    :return: Python code for evaluating the monomial base polynomial as specified by m, r and a.
    :rtype: str
    """
    if isinstance(a[0], str):
        coeff_strs = [c for c in a]
    else:
        coeff_strs = [str_number(c, prettify_fractions=prettify_coefficients) for c in a]
    if m == 1:
        return _horner_evaluation(coeff_strs)
    basis_strs = generate_monomial_basis(m, r)
    return str_dot_product(coeff_strs, basis_strs, multiplication_character="*")


def generate_function_eval_specific_vector_valued(m, r, a):
    r"""
    Generate code for evaluating a specific vector valued degree r polynomial on an m-dimensional domain,
    expressed in the monomial basis.

    .. math::

        p : \mathbb{R}^m \to \mathbb{R}^n, n > 1,

        p_{\nu, r}(x)=\sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} x^{\nu_i},

    where :math:`\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
    :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).

    :param int m: Dimension of the domain.
    :param int r: Degree of the polynomial space.
    :param a: Coefficients for the polynomial in the monomial basis for :math:`\mathcal{P}_r (\mathbb{R}^m)`.
        :math:`\text{a}[i] = a_{\nu_i}`, where :math:`\nu_i` is the i:th multi-index in the sequence of all
        multi-indices of dimension m with norm :math:`\leq r`
        (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
    :type a: Iterable[n-dimensional vector]
    :return: Python code for evaluating the monomial base polynomial as specified by m, r and a.
    :rtype: str
    """
    code = ""

    code += "np.array(["
    code += generate_function_eval_specific_scalar_valued(m, r, a[:, 0])
    for i in range(1, len(a[0])):
        code += ", " + generate_function_eval_specific_scalar_valued(m, r, a[:, i])
    code += "])"

    return code


def _generate_monomial_polynomial_latex_str(m, r):
    coeff_strs = str_sequence("a", get_dimension(r, m), indexing='c', index_style='latex_subscript')
    from polynomials_on_simplices.polynomial.polynomials_monomial_basis import monomial_basis_latex
    basis_strs = monomial_basis_latex(r, m)
    latex_str = str_dot_product(coeff_strs, basis_strs)
    parts = split_long_str(latex_str, "+", 100)
    latex_str = "\\\\\n&\\quad +".join(parts)
    return "p(x) = " + latex_str


def _horner_evaluation_recursive(a):
    if len(a) > 1:
        nested_part = str_sum(a[0], _horner_evaluation_recursive(a[1:]))
        if nested_part != "0":
            if nested_part == "1":
                return "x"
            else:
                if "+" in nested_part or "-" in nested_part:
                    return str_product("x", "(" + nested_part + ")", "*")
                else:
                    return str_product("x", nested_part, "*")
        else:
            return "0"
    return str_product("x", a[0], "*")


def _horner_evaluation(a):
    if len(a) == 1:
        return a[0]
    else:
        return str_sum(a[0], _horner_evaluation_recursive(a[1:]))


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
