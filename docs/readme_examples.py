

def basic_example_1d():
    from polynomials_on_simplices.polynomial.polynomials_monomial_basis import Polynomial
    p = Polynomial([1, 2, 3])
    print(p)
    print(p(1))
    print(2 * p)


def basic_example_2d():
    from polynomials_on_simplices.polynomial.polynomials_monomial_basis import Polynomial
    p = Polynomial([1, 2, 3], r=1, m=2)
    print(p)
    print(p((1, 2)))
    print(p**2)


def basic_example_1d2d():
    from polynomials_on_simplices.polynomial.polynomials_monomial_basis import Polynomial
    p = Polynomial([[1, 0], [2, 1], [3, 2]])
    print(p)
    print(p(1))


def lagrange_example_1d():
    import matplotlib.pyplot as plt
    from polynomials_on_simplices.calculus.plot_function import plot_function
    from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import lagrange_basis
    fig = plt.figure()
    for l in lagrange_basis(2, 1):
        plot_function(l, 0.0, 1.0, fig=fig)
    plt.show()


def lagrange_example_2d():
    import matplotlib.pyplot as plt
    from polynomials_on_simplices.calculus.plot_function import plot_bivariate_function
    from polynomials_on_simplices.geometry.primitives.simplex import unit
    from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import lagrange_basis_fn
    vertices = unit(2)
    l = lagrange_basis_fn((0, 1), 2)
    plot_bivariate_function(lambda x1, x2: l((x1, x2)), vertices)
    plt.show()


def lagrange_example_1d_arbitrary():
    import matplotlib.pyplot as plt
    from polynomials_on_simplices.calculus.plot_function import plot_function
    from polynomials_on_simplices.polynomial.polynomials_simplex_lagrange_basis import lagrange_basis_simplex
    fig = plt.figure()
    for l in lagrange_basis_simplex(2, [[1], [3]]):
        plot_function(l, 1.0, 3.0, fig=fig)
    plt.show()


def lagrange_example_2d_arbitrary():
    import matplotlib.pyplot as plt
    from polynomials_on_simplices.calculus.plot_function import plot_bivariate_function
    from polynomials_on_simplices.polynomial.polynomials_simplex_lagrange_basis import lagrange_basis_fn_simplex
    vertices = [
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ]
    l = lagrange_basis_fn_simplex((0, 1), 2, vertices)
    plot_bivariate_function(lambda x1, x2: l((x1, x2)), vertices)
    plt.show()


def lagrange_example_4d():
    from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import PolynomialLagrange
    p = PolynomialLagrange([1, 2, 3, 4, 5], r=1, m=4)
    print(p)
    print(p((0.1, 0.2, 0.3, 0.4)))


def lagrange_example_1d_dual_basis():
    from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import dual_lagrange_basis_fn, \
        lagrange_basis_fn
    l0 = lagrange_basis_fn(0, 2)
    l1 = lagrange_basis_fn(1, 2)
    q0 = dual_lagrange_basis_fn(0, 2)
    q1 = dual_lagrange_basis_fn(1, 2)
    print(q0(l0))
    print(q0(l1))
    print(q1(l0))
    print(q1(l1))


def differentiation_example():
    from polynomials_on_simplices.polynomial.polynomials_unit_simplex_bernstein_basis import PolynomialBernstein
    b = PolynomialBernstein([1, 0, 0, 1, 0, 0], r=2, m=2)
    print(b)
    print(b.latex_str_expanded())
    print(b.partial_derivative(0).latex_str_expanded())
    print(b.partial_derivative(1).latex_str_expanded())
    from polynomials_on_simplices.calculus.polynomial.polynomials_calculus import derivative
    print(derivative(b, (1, 1)))


def integration_example():
    from polynomials_on_simplices.polynomial.polynomials_unit_simplex_bernstein_basis import bernstein_basis_fn
    from polynomials_on_simplices.calculus.polynomial.polynomials_calculus import integrate_unit_simplex
    b = bernstein_basis_fn((1, 0), 1)
    print(b.latex_str_expanded())
    print(integrate_unit_simplex(b))


def piecewise_polynomial_example_1d():
    import matplotlib.pyplot as plt
    from polynomials_on_simplices.piecewise_polynomial.plot_piecewise_polynomial import \
        plot_univariate_piecewise_polynomial
    from polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial_bernstein_basis import \
        piecewise_polynomial_bernstein_basis
    lines = [[0, 1], [1, 2]]
    vertices = [[1.0], [2.0], [3.0]]
    fig = plt.figure()
    for b in piecewise_polynomial_bernstein_basis(lines, vertices, r=2):
        plot_univariate_piecewise_polynomial(b, fig=fig)
    plt.show()


def piecewise_polynomial_example_2d():
    from polynomials_on_simplices.piecewise_polynomial.plot_piecewise_polynomial import \
        plot_bivariate_piecewise_polynomial
    from polynomials_on_simplices.piecewise_polynomial.continuous_piecewise_polynomial_bernstein_basis import \
        continuous_piecewise_polynomial_bernstein_basis

    triangles = [
        [0, 1, 2],
        [1, 3, 2]
    ]
    vertices = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ]
    for b in continuous_piecewise_polynomial_bernstein_basis(triangles, vertices, r=1):
        plot_bivariate_piecewise_polynomial(b, edge_resolution=2)


if __name__ == "__main__":
    basic_example_1d()
    basic_example_2d()
    basic_example_1d2d()

    lagrange_example_1d()
    lagrange_example_2d()
    lagrange_example_1d_arbitrary()
    lagrange_example_2d_arbitrary()
    lagrange_example_4d()
    lagrange_example_1d_dual_basis()

    differentiation_example()
    integration_example()

    piecewise_polynomial_example_1d()
    piecewise_polynomial_example_2d()
