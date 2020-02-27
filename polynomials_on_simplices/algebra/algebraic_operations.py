"""General algebraic operations."""

from inspect import Parameter, signature


def composition(f, g):
    r"""
    Create the composition :math:`f \circ g` of two functions :math:`f, g`.

    .. math::

        g : X \to Y,

        f : Y \to Z,

        f \circ g : X \to Z,

        (f \circ g)(x) = f(g(x)).

    :param f: Function applied last.
    :type f: Callable
    :param g: Function applied first.
    :type g: Callable
    :return: The composition :math:`f \circ g`.
    :rtype: Callable
    """
    f_univariate = _is_univariate_function(f)
    g_univariate = _is_univariate_function(g)

    if g_univariate:
        if f_univariate:
            def fog(x):
                return f(g(x))
        else:
            def fog(x):
                return f(*g(x))
    else:
        if f_univariate:
            def fog(*x):
                return f(g(*x))
        else:
            def fog(*x):
                return f(*g(*x))
    return fog


def _is_univariate_function(f):
    """
    Check whether or not a function is univariate.

    :param f: Function to check.
    :type f: Callable
    :return: Whether or not the function is univariate.
    :rtype: bool
    """
    try:
        sig = signature(f)
        if len(sig.parameters) > 1:
            return False
        else:
            assert len(sig.parameters) == 1
            parameter = list(dict(sig.parameters).items())[0][1]
            if parameter.kind == Parameter.VAR_POSITIONAL:
                return False
        return True
    except ValueError:
        # Signature not available e.g. for some built-in functions.
        # In that case we assume they take one argument
        return True
