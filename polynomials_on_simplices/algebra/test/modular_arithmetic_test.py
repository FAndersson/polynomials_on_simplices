import unittest

from polynomials_on_simplices.algebra.modular_arithmetic import IntegerModuloN


class TestIntegerModuloN(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(IntegerModuloN(-1, 3), IntegerModuloN(2, 3))

    def test_add(self):
        a = IntegerModuloN(1, 4)
        b = IntegerModuloN(2, 4)
        c = IntegerModuloN(3, 4)
        d = IntegerModuloN(0, 4)
        e = IntegerModuloN(-1, 4)

        self.assertTrue(a + b == 3)
        self.assertTrue(a + c == 0)
        self.assertTrue(a + d == 1)
        self.assertTrue(a + e == 0)
        self.assertTrue(a + 1 == 2)
        self.assertTrue(1 + a == 2)

    def test_sub(self):
        a = IntegerModuloN(1, 4)
        b = IntegerModuloN(2, 4)
        c = IntegerModuloN(3, 4)

        self.assertTrue(a - b == 3)
        self.assertTrue(a - c == 2)
        self.assertTrue(a - 1 == 0)
        self.assertTrue(2 - a == 1)

    def test_mul(self):
        a = IntegerModuloN(1, 4)
        b = IntegerModuloN(2, 4)
        c = IntegerModuloN(3, 4)

        self.assertTrue(a * b == 2)
        self.assertTrue(b * c == 2)
        self.assertTrue(a * 4 == 0)
        self.assertTrue(4 * a == 0)

    def test_compare(self):
        a = IntegerModuloN(1, 4)
        b = IntegerModuloN(2, 4)
        c = IntegerModuloN(-1, 4)
        d = IntegerModuloN(0, 4)
        e = IntegerModuloN(1, 4)

        self.assertTrue(a < b)
        self.assertTrue(a < c)
        self.assertTrue(a > d)
        self.assertTrue(a >= e)
        self.assertTrue(a <= e)


if __name__ == "__main__":
    unittest.main()
