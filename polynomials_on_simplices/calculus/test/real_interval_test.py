import sys

import pytest

from polynomials_on_simplices.calculus.real_interval import equivalent_periodic_element


def test_equivalent_element():
    b = equivalent_periodic_element(1.0, 2.0)
    assert 1.0 == b
    b = equivalent_periodic_element(3.0, 2.0)
    assert 1.0 == b
    b = equivalent_periodic_element(-3.0, 2.0)
    assert 1.0 == b
    b = equivalent_periodic_element(4.5, 2.5)
    assert 2.0 == b


if __name__ == "__main__":
    pytest.main(sys.argv)
