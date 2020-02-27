"""
Generic functionality for generating Python code used to evaluate polynomials.
"""

import numpy as np


def generate_function_specific_name(a):
    """
    Generate name for a general function evaluating a polynomial.

    :param a: Coefficients for the polynomial used to generate a unique name.
    :type a: Iterable[float]
    :return: Name for the function.
    :rtype: str
    """
    if isinstance(a, np.ndarray):
        coeff_hash = hash(a.tostring())
    else:
        coeff_hash = hash(str(a))
    if coeff_hash < 0:
        # Cannot have minus sign in name
        coeff_hash *= -1
    return str(coeff_hash)


def generate_docstring(m, r, basis, latex_str):
    """
    Generate a docstring for a general function evaluating a polynomial given polynomial coefficients and
    a point of evaluation.

    :param int m: Dimension of the domain of the polynomial.
    :param int r: Degree of the polynomial space.
    :param str basis: Name for the polynomial basis used.
    :param str latex_str: Latex str for the calculation done by the function.
    :return: Docstring for the function.
    :rtype: str
    """
    if m == 1:
        docstring = _template_docstring_univariate
    else:
        docstring = _template_docstring_multivariate
        docstring = docstring.replace("<MULTIVARIATE>", _multivariate(m))
        docstring = docstring.replace("<DIM>", str(m))
    docstring = docstring.replace("<R>", str(r))
    docstring = docstring.replace("<BASE>", basis)
    if basis == "monomial":
        docstring = docstring.replace("<DOMAIN>", "")
    else:
        docstring = docstring.replace("<DOMAIN>", " on the " + _domain(m))
    if latex_str.find('\n') == -1:
        docstring = docstring.replace("<LATEX_STR>", " " + latex_str)
    else:
        latex_str = "\n\n    " + "\n    ".join(latex_str.split('\n'))
        docstring = docstring.replace("<LATEX_STR>", latex_str)
    return docstring


_template_docstring_univariate = """r\"\"\"
Evaluate a univariate degree <R> <BASE> polynomial<DOMAIN>.

.. math::<LATEX_STR>.

:param a: Coefficient in front of each <BASE> base polynomial, :math:`\\text{a}[i] = a_i`.
:param x: Point where the polynomial should be evaluated.
:return: Value of the polynomial.
\"\"\""""


_template_docstring_multivariate = """r\"\"\"
Evaluate a <MULTIVARIATE> degree <R> <BASE> polynomial<DOMAIN>.

.. math::<LATEX_STR>.

:param a: Coefficient in front of each <BASE> base polynomial, :math:`\\text{a}[i] = a_{\\nu_i}`, where
    :math:`\\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension <DIM> with norm
    :math:`\\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
:param x: Point where the polynomial should be evaluated.
:return: Value of the polynomial.
\"\"\""""


def _domain(m):
    return {
        1: "unit interval",
        2: "unit triangle",
        3: "unit tetrahedron"
    }.get(m, "unit simplex")


def _multivariate(m):
    return {
        2: "bivariate",
        3: "trivariate",
    }.get(m, "multivariate")
