"""
Utility functions for generating math-related strings.
"""

import fractions


def str_number(a, latex_fraction=False, prettify_fractions=True, n=10):
    r"""
    Convert a number to string.

    :param a: Number we want to convert to string.
    :param bool latex_fraction: If the number is a fraction, whether or not to encode the fractions using the Latex
        "\frac" syntax. Default is to just use '/' for the fraction.
    :param bool prettify_fractions: Whether or not to convert a number which is a 'nice' fraction (see
        :func:`is_nice_fraction`) to its fraction representation.
    :param int n: How large the denominator can be in a fraction for the fraction to be considered nice. Only used
        when `prettify_fractions` is True.
    :return: String representation of the number.
    :rtype: str

    .. rubric:: Examples

    >>> str_number(1.0)
    '1'
    >>> str_number(0.0)
    '0'
    >>> str_number(0.5)
    '1 / 2'
    >>> str_number(0.5, prettify_fractions=False)
    '0.5'
    """
    if _is_integer(a):
        return str(int(a))
    if prettify_fractions:
        if is_nice_fraction(a, n):
            return convert_float_to_fraction(a, latex_fraction)
    return str(a)


def str_number_array(a, latex=False, latex_column_vector=True, latex_array_style="pmatrix"):
    r"""
    Convert an array of numbers to string.

    :param a: Array of numbers we want to convert to string.
    :param bool latex: Whether or not to encode the array using Latex syntax.
    :param bool latex_column_vector: Whether the array should be encoded as a row or column vector. Only applicable
        if `latex` is true.
    :param latex_array_style: Latex array style to use. Only applicable if `latex` is true.
    :return: String representation of the array of numbers.
    :rtype: str

    .. rubric:: Examples

    >>> str_number_array([1.0, 2.0])
    '[1, 2]'
    >>> str_number_array([0.0, 0.5], latex=True)
    '\\begin{pmatrix}0 \\\\ \\frac{1}{2}\\end{pmatrix}'
    >>> str_number_array([3, 4], latex=True, latex_column_vector=False, latex_array_style="bmatrix")
    '\\begin{bmatrix}3 & 4\\end{bmatrix}'
    """
    if latex:
        res = "\\begin{" + latex_array_style + "}"
        if latex_column_vector:
            res += " \\\\ ".join(str_number(n, latex_fraction=True) for n in a)
        else:
            res += " & ".join(str_number(n, latex_fraction=True) for n in a)
        res += "\\end{" + latex_array_style + "}"
        return res
    else:
        return "[" + ", ".join(str_number(n) for n in a) + "]"


def is_nice_fraction(f, n=10):
    """
    Check if a scalar can be expressed as a 'nice' rational number. Nice in this sense means that it can be
    written as a fraction with a denominator less than a given number n.

    :param float f: Scalar value we want to check.
    :param int n: How large the denominator can be in the fraction for the fraction to be considered nice.
    :return: Whether or not the scalar is a 'nice' fraction.
    :rtype: bool

    .. rubric:: Examples

    >>> is_nice_fraction(0.1)
    True
    >>> is_nice_fraction(0.01)
    False
    >>> is_nice_fraction(0.01, 100)
    True
    """
    r = fractions.Fraction(f).limit_denominator(n)
    return abs(r - f) < 1e-10


def convert_float_to_fraction(s, latex_fraction=False):
    r"""
    Convert a float in string format to a fraction.

    :param str s: Float value as a string.
    :param bool latex_fraction: Whether or not to encode fractions using the Latex "\frac" syntax. Default is to
        just use '/' for the fraction.
    :return: Fraction string.

    .. rubric:: Examples

    >>> convert_float_to_fraction('0.5')
    '1 / 2'
    >>> convert_float_to_fraction('0')
    '0'
    >>> convert_float_to_fraction('0.5', latex_fraction=True)
    '\\frac{1}{2}'
    >>> convert_float_to_fraction('0', latex_fraction=True)
    '0'
    >>> convert_float_to_fraction('-0.5', latex_fraction=True)
    '-\\frac{1}{2}'
    >>> convert_float_to_fraction('2', latex_fraction=True)
    '2'
    """
    f = float(s)
    r = fractions.Fraction(f).limit_denominator(1000)
    if r.numerator == 0:
        return '0'
    if r.denominator == 1:
        return str(r.numerator)
    if latex_fraction:
        if r.numerator < 0:
            return r"-\frac{" + str(-r.numerator) + "}{" + str(r.denominator) + "}"
        else:
            return r"\frac{" + str(r.numerator) + "}{" + str(r.denominator) + "}"
    else:
        return str(r.numerator) + " / " + str(r.denominator)


def str_sum(a, b):
    r"""
    Generate the string for the sum of two values, given by strings.

    :param str a: First value in the sum.
    :param str b: Second value in the sum.
    :return: String for the sum of the two values.
    :rtype: str

    .. rubric:: Examples

    >>> str_sum("a1", "a2")
    'a1 + a2'
    >>> str_sum("a1", "-a2")
    'a1 - a2'
    >>> str_sum("a1", "0")
    'a1'
    >>> str_sum("0", "-a2")
    '-a2'
    >>> str_sum("-a1", "1")
    '-a1 + 1'
    """
    if a == "0":
        return b
    if b == "0":
        return a
    if b.startswith("-"):
        return a + " - " + b[1:]
    else:
        return a + " + " + b


def str_multi_sum(a):
    r"""
    Generate the string for the sum of an array of values, given by strings.

    :param List[str] a: Values in the sum.
    :return: String for the sum of the values in the array.
    :rtype: str

    .. rubric:: Examples

    >>> str_multi_sum(["a1", "a2"])
    'a1 + a2'
    >>> str_multi_sum(["a1", "-a2"])
    'a1 - a2'
    >>> str_multi_sum(["a1", "0", "a3"])
    'a1 + a3'
    >>> str_multi_sum(["0", "-a2", "a3"])
    '-a2 + a3'
    """
    res = "0"
    for value in a:
        res = str_sum(res, value)
    return res


def str_product(a, b, multiplication_character=""):
    r"""
    Generate the string for the product of two values, given by strings.

    :param str a: First value in the product.
    :param str b: Second value in the product.
    :param str multiplication_character: Character(s) used to represent the product operator.
    :return: String for the product of the two values.
    :rtype: str

    .. rubric:: Examples

    >>> str_product("a1", "a2")
    'a1 a2'
    >>> str_product("a1", "-a2", multiplication_character="\\times")
    '-a1 \\times a2'
    >>> str_product("a1", "0")
    '0'
    >>> str_product("a1", "1")
    'a1'
    """
    if a == "0" or b == "0":
        return "0"
    if a == "1":
        return b
    if b == "1":
        return a
    if a.startswith("-"):
        if b.startswith("-"):
            return str_product(a[1:], b[1:], multiplication_character)
        else:
            return "-" + str_product(a[1:], b, multiplication_character)
    else:
        if b.startswith("-"):
            return "-" + str_product(a, b[1:], multiplication_character)
        else:
            if multiplication_character != "":
                return a + " " + multiplication_character + " " + b
            else:
                return a + " " + b


def str_multi_product(a, multiplication_character=""):
    r"""
    Generate the string for the product of an array of values, given by strings.

    :param List[str] a: Values in the product.
    :param str multiplication_character: Character(s) used to represent the product operator.
    :return: String for the product of the array of values.
    :rtype: str

    .. rubric:: Examples

    >>> str_multi_product(["a1", "a2", "a3"])
    'a1 a2 a3'
    >>> str_multi_product(["a1", "-a2"], multiplication_character="\\times")
    '-a1 \\times a2'
    >>> str_multi_product(["a1", "0", "a3"])
    '0'
    >>> str_multi_product(["a1", "1", "a3"])
    'a1 a3'
    """
    res = "1"
    for value in a:
        res = str_product(res, value, multiplication_character)
    return res


def str_exponent(base, exp, str_type="python"):
    r"""
    Generate the string for the power of one value raised to an exponent, given by strings.

    :param str base: Base value.
    :param str exp: Exponent value.
    :param str str_type: Type of string, 'python' or 'latex'. Determines the exponentiation character used, '**' or '^'.
        Also for the 'latex' string type the exponent is wrapped in braces ({}) if it's longer than one character.
    :return: String for the base raised to the power of the exponent.
    :rtype: str

    >>> str_exponent("e", "x")
    'e**x'
    >>> str_exponent("e", "x", str_type="latex")
    'e^x'
    >>> str_exponent("e", "x_2", str_type="latex")
    'e^{x_2}'
    >>> str_exponent("e", "0")
    '1'
    >>> str_exponent("e", "1")
    'e'
    >>> str_exponent("0", "0")
    '1'
    """
    if base == "0":
        if exp == "0":
            return "1"
        else:
            return "0"
    if exp == "0":
        return "1"
    if exp == "1":
        return base
    if str_type == "python":
        return base + "**" + exp
    else:
        if len(exp) == 1:
            return base + "^" + exp
        else:
            return base + "^{" + exp + "}"


def str_dot_product(a, b, multiplication_character=""):
    r"""
    Generate the string for a dot product of two arrays, with array elements given by strings.

    :param List[str] a: First array of elements in the dot product.
    :param List[str] b: Second array of elements in the dot product.
    :param str multiplication_character: Character(s) used to represent the product operator.
    :return: String for the dot product of the two arrays.
    :rtype: str

    .. rubric:: Examples

    >>> str_dot_product(["a1", "a2"], ["b1", "b2"])
    'a1 b1 + a2 b2'
    >>> str_dot_product(["a1", "a2"], ["b1", "b2"], multiplication_character="\\cdot")
    'a1 \\cdot b1 + a2 \\cdot b2'
    """
    try:
        len(a)
        assert len(a) == len(b)
    except TypeError:
        return str_product(a, b, multiplication_character)

    res = "0"
    for i in range(len(a)):
        p = str_product(a[i], b[i], multiplication_character)
        res = str_sum(res, p)
    return res


def split_long_str(string, sep=" ", long_str_threshold=-1, sep_replacement_before="", sep_replacement_after=""):
    """
    Split a long string into an array of parts.

    The string will be split at occurrences of the given separator in a way that makes sure that each part is
    shorter than the given threshold length.

    :param str string: String we want to split.
    :param str sep: The delimiter according which to split the string.
    :param int long_str_threshold: Don't split at a delimiter if the resulting parts will be shorter than this
        threshold. The default value -1 indicates that we should always split at a delimiter.
    :param str sep_replacement_before: When splitting at a separator, the separator will be replaced with this string
        in the part before the split.
    :param str sep_replacement_after: When splitting at a separator, the separator will be replaced with this string
        in the part after the split.
    :return: Parts of the original string after the split.
    :rtype: List[str]

    .. rubric Examples

    >>> split_long_str("Hello world", long_str_threshold=1)
    ['Hello', 'world']
    >>> split_long_str("Hello world", long_str_threshold=15)
    ['Hello world']
    >>> split_long_str("My quite long sentence example", long_str_threshold=15)
    ['My quite long', 'sentence', 'example']
    >>> split_long_str("Hello world", sep_replacement_before="_")
    ['Hello_', 'world']
    >>> split_long_str("Hello world", sep_replacement_after="_")
    ['Hello', '_world']
    """
    potential_parts = string.split(sep)
    if long_str_threshold == -1:
        if sep_replacement_before != "":
            for i in range(0, len(potential_parts) - 1):
                potential_parts[i] += sep_replacement_before
        if sep_replacement_after != "":
            for i in range(1, len(potential_parts)):
                potential_parts[i] = sep_replacement_after + potential_parts[i]
        return potential_parts
    parts = []
    part = potential_parts[0]
    for i in range(1, len(potential_parts)):
        if len(part + sep + potential_parts[i] + sep_replacement_before) < long_str_threshold:
            part += sep + potential_parts[i]
        else:
            parts.append(part + sep_replacement_before)
            part = sep_replacement_after + potential_parts[i]
    parts.append(part)
    return parts


def _latex_braces_wrapped_number(a):
    if len(a) == 1:
        return a
    return "{" + a + "}"


def str_sequence(symbol, n, indexing="fortran", index_style="inline"):
    """
    Generate a sequence of enumerated strings, e.g. (a1, a2, a3).

    :param str symbol: Base symbol(s) to use for each string.
    :param int n: Number of strings to generate.
    :param indexing: Which indexing to use. Can be "c" for 0-based indexing (0, 1, 2, ...), or "fortran" for
        1-based indexing (1, 2, 3, ...).
    :param index_style: How indices should be added to the base symbol(s). Can be any of "inline", "subscript",
        "superscript", "latex_subscript", "latex_superscript" or "list". See examples below. The "latex_subscript"
        differ from the "subscript" variant in that the index is wrapped in braces ({}) if it's longer than one
        character, and similarly for the "latex_superscript" variant.
    :return: List of strings.

    .. rubric:: Examples

    >>> str_sequence('a', 3)
    ['a1', 'a2', 'a3']
    >>> str_sequence('a', 3, indexing="c")
    ['a0', 'a1', 'a2']
    >>> str_sequence('a', 3, index_style="subscript")
    ['a_1', 'a_2', 'a_3']
    >>> str_sequence('a', 3, index_style="superscript")
    ['a^1', 'a^2', 'a^3']
    >>> str_sequence('a', 10, index_style="latex_subscript")
    ['a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7', 'a_8', 'a_9', 'a_{10}']
    >>> str_sequence('a', 3, index_style="list")
    ['a[1]', 'a[2]', 'a[3]']
    """
    assert n >= 0, "Sequence length cannot be negative"
    if n == 0:
        return []
    if indexing == "fortran":
        r = range(1, n + 1)
    else:
        r = range(n)
    if index_style == "inline":
        return [str(symbol) + str(i) for i in r]
    if index_style == "subscript":
        return [str(symbol) + "_" + str(i) for i in r]
    if index_style == "superscript":
        return [str(symbol) + "^" + str(i) for i in r]
    if index_style == "latex_subscript":
        return [str(symbol) + "_" + _latex_braces_wrapped_number(str(i)) for i in r]
    if index_style == "latex_superscript":
        return [str(symbol) + "^" + _latex_braces_wrapped_number(str(i)) for i in r]
    if index_style == "list":
        return [str(symbol) + "[" + str(i) + "]" for i in r]


def _is_integer(f):
    """
    Check if a scalar is an integer.

    :param f: Scalar that we want to check.
    :return: Whether or not the scalar is an integer.
    :rtype: bool
    """
    return f == int(f)
