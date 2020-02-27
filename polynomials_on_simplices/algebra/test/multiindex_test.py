import math
import unittest

import numpy as np
from scipy.special import binom

import polynomials_on_simplices.algebra.multiindex as multiindex


class MultiIndexTest(unittest.TestCase):
    def test_init(self):
        a = multiindex.MultiIndex(1, 2, 3)
        b = multiindex.MultiIndex([1, 2, 3])
        c = multiindex.MultiIndex((1, 2, 3))
        d = multiindex.MultiIndex(np.array([1, 2, 3]))
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a, d)
        e = multiindex.MultiIndex(1)
        self.assertEqual(e, multiindex.MultiIndex([1]))
        with self.assertRaises(ValueError):
            multiindex.MultiIndex(-1, 1)
        with self.assertRaises(ValueError):
            multiindex.MultiIndex(1, -1)

    def test_add(self):
        a = multiindex.MultiIndex(4, 5, 6)
        b = multiindex.MultiIndex(1, 2, 3)
        self.assertEqual(multiindex.MultiIndex(5, 7, 9), a + b)
        self.assertEqual(multiindex.MultiIndex(5, 7, 9), a + (1, 2, 3))
        self.assertEqual(multiindex.MultiIndex(5, 7, 9), a + [1, 2, 3])
        self.assertEqual(multiindex.MultiIndex(5, 7, 9), (1, 2, 3) + a)
        self.assertEqual(multiindex.MultiIndex(5, 7, 9), [1, 2, 3] + a)

    def test_sub(self):
        a = multiindex.MultiIndex(4, 5, 6)
        b = multiindex.MultiIndex(1, 2, 3)
        c = multiindex.MultiIndex(4, 5, 7)
        self.assertEqual(multiindex.MultiIndex(3, 3, 3), a - b)
        self.assertEqual(multiindex.MultiIndex(3, 3, 3), a - (1, 2, 3))
        self.assertEqual(multiindex.MultiIndex(3, 3, 3), a - [1, 2, 3])
        self.assertEqual(multiindex.MultiIndex(3, 3, 3), (4, 5, 6) - b)
        self.assertEqual(multiindex.MultiIndex(3, 3, 3), [4, 5, 6] - b)
        with self.assertRaises(ValueError):
            a - c

    def test_iadd(self):
        a = multiindex.MultiIndex(4, 5, 6)
        b = multiindex.MultiIndex(1, 2, 3)
        a += b
        self.assertEqual(multiindex.MultiIndex(5, 7, 9), a)

    def test_comparison(self):
        a = multiindex.MultiIndex(1, 2, 3)
        b = multiindex.MultiIndex(1, 2, 3)
        c = multiindex.MultiIndex(1, 2, 4)
        d = multiindex.MultiIndex(2, 3, 4, 5)
        self.assertTrue(a == b)
        self.assertTrue(a != c)
        with self.assertRaises(ValueError):
            a == d

    def test_comparison_tuple(self):
        a = multiindex.MultiIndex(1, 2, 3)
        b = (1, 2, 3)
        c = (1, 2, 4)
        d = (2, 3, 4, 5)
        self.assertTrue(a == b)
        self.assertTrue(a != c)
        with self.assertRaises(ValueError):
            a == d

    def test_comparison_list(self):
        a = multiindex.MultiIndex(1, 2, 3)
        b = [1, 2, 3]
        c = [1, 2, 4]
        d = [2, 3, 4, 5]
        self.assertTrue(a == b)
        self.assertTrue(a != c)
        with self.assertRaises(ValueError):
            a == d


class MultiIndexOperationsTest(unittest.TestCase):
    def test_norm(self):
        a = multiindex.MultiIndex(1, 2, 3)
        self.assertEqual(multiindex.norm(a), 6)

    def test_factorial(self):
        a = multiindex.MultiIndex(4, 5, 6)
        self.assertEqual(multiindex.factorial(a), 2073600)

    def test_power(self):
        a = multiindex.MultiIndex(4, 5, 6)
        x = [0.1, 0.2, 0.3]
        p = multiindex.power(x, a)
        self.assertEqual(p, 0.1**4 * 0.2**5 * 0.3**6)
        self.assertEqual(p, x**a)

    def test_binom(self):
        a = multiindex.MultiIndex(4, 5, 6)
        b = multiindex.MultiIndex(1, 2, 3)
        self.assertEqual(multiindex.binom(a, b), binom(4, 1) * binom(5, 2) * binom(6, 3))

    def test_multinom(self):
        a = multiindex.MultiIndex(4, 5, 6)
        self.assertEqual(multiindex.multinom(a),
                         math.factorial(15) / (math.factorial(4) * math.factorial(5) * math.factorial(6)))


class MultiIndexGenerateAllTest(unittest.TestCase):
    def test_1by3(self):
        mis = multiindex.generate_all(1, 3)
        mis_ref = [
            multiindex.MultiIndex(0),
            multiindex.MultiIndex(1),
            multiindex.MultiIndex(2),
            multiindex.MultiIndex(3)
        ]
        self.assertEqual(mis, mis_ref)

    def test_2by2(self):
        mis = multiindex.generate_all(2, 2)
        mis_ref = [
            multiindex.MultiIndex(0, 0),
            multiindex.MultiIndex(1, 0),
            multiindex.MultiIndex(2, 0),
            multiindex.MultiIndex(0, 1),
            multiindex.MultiIndex(1, 1),
            multiindex.MultiIndex(0, 2),
        ]
        self.assertEqual(mis, mis_ref)

    def test_3by3(self):
        mis = multiindex.generate_all(3, 3)
        mis_ref = [
            multiindex.MultiIndex(0, 0, 0),
            multiindex.MultiIndex(1, 0, 0),
            multiindex.MultiIndex(2, 0, 0),
            multiindex.MultiIndex(3, 0, 0),
            multiindex.MultiIndex(0, 1, 0),
            multiindex.MultiIndex(1, 1, 0),
            multiindex.MultiIndex(2, 1, 0),
            multiindex.MultiIndex(0, 2, 0),
            multiindex.MultiIndex(1, 2, 0),
            multiindex.MultiIndex(0, 3, 0),
            multiindex.MultiIndex(0, 0, 1),
            multiindex.MultiIndex(1, 0, 1),
            multiindex.MultiIndex(2, 0, 1),
            multiindex.MultiIndex(0, 1, 1),
            multiindex.MultiIndex(1, 1, 1),
            multiindex.MultiIndex(0, 2, 1),
            multiindex.MultiIndex(0, 0, 2),
            multiindex.MultiIndex(1, 0, 2),
            multiindex.MultiIndex(0, 1, 2),
            multiindex.MultiIndex(0, 0, 3)
        ]
        self.assertEqual(mis, mis_ref)

    def test_exact_norm_1by3(self):
        mis = multiindex.generate_all_exact_norm(1, 3)
        mis_ref = [
            multiindex.MultiIndex(3)
        ]
        self.assertEqual(mis, mis_ref)

    def test_exact_norm_2by2(self):
        mis = multiindex.generate_all_exact_norm(2, 2)
        mis_ref = [
            multiindex.MultiIndex(2, 0),
            multiindex.MultiIndex(1, 1),
            multiindex.MultiIndex(0, 2)
        ]
        self.assertEqual(mis, mis_ref)

    def test_exact_norm_3by2(self):
        mis = multiindex.generate_all_exact_norm(3, 2)
        mis_ref = [
            multiindex.MultiIndex(2, 0, 0),
            multiindex.MultiIndex(1, 1, 0),
            multiindex.MultiIndex(0, 2, 0),
            multiindex.MultiIndex(1, 0, 1),
            multiindex.MultiIndex(0, 1, 1),
            multiindex.MultiIndex(0, 0, 2)
        ]
        self.assertEqual(mis, mis_ref)

    def test_exact_norm_3by3(self):
        mis = multiindex.generate_all_exact_norm(3, 3)
        mis_ref = [
            multiindex.MultiIndex(3, 0, 0),
            multiindex.MultiIndex(2, 1, 0),
            multiindex.MultiIndex(1, 2, 0),
            multiindex.MultiIndex(0, 3, 0),
            multiindex.MultiIndex(2, 0, 1),
            multiindex.MultiIndex(1, 1, 1),
            multiindex.MultiIndex(0, 2, 1),
            multiindex.MultiIndex(1, 0, 2),
            multiindex.MultiIndex(0, 1, 2),
            multiindex.MultiIndex(0, 0, 3)
        ]
        self.assertEqual(mis, mis_ref)

    def test_general_to_exact_conversion(self):
        mis1 = multiindex.generate_all(2, 3)
        mis2 = [multiindex.exact_norm_to_general(mi) for mi in multiindex.generate_all_exact_norm(3, 3)]
        self.assertEqual(mis1, mis2)

        mis1 = [multiindex.general_to_exact_norm(mi, 2) for mi in multiindex.generate_all(2, 2)]
        mis2 = multiindex.generate_all_exact_norm(3, 2)
        self.assertEqual(mis1, mis2)


class MultiIndexGenerateTest(unittest.TestCase):
    def test_num_multiindices(self):
        self.assertEqual(6, multiindex.num_multiindices(2, 2))
        self.assertEqual(10, multiindex.num_multiindices(3, 2))
        self.assertEqual(20, multiindex.num_multiindices(3, 3))

    def test_generate(self):
        self.assertEqual(multiindex.MultiIndex([0, 0, 0]), multiindex.generate(3, 2, 0))
        self.assertEqual(multiindex.MultiIndex([1, 1, 0]), multiindex.generate(3, 2, 4))
        self.assertEqual(multiindex.MultiIndex([0, 0, 2]), multiindex.generate(3, 2, 9))


class MultiIndexGetIndexTest(unittest.TestCase):
    def test_2d(self):
        n = 2
        r = 3
        idx_ref = 0
        for mi in multiindex.MultiIndexIterator(n, r):
            idx = multiindex.get_index(mi, r)
            self.assertEqual(idx_ref, idx)
            idx_ref += 1

    def test_3d(self):
        n = 3
        r = 3
        idx_ref = 0
        for mi in multiindex.MultiIndexIterator(n, r):
            idx = multiindex.get_index(mi, r)
            self.assertEqual(idx_ref, idx)
            idx_ref += 1


class MultiIndexGenerateAllMultiCapTest(unittest.TestCase):
    def test_1by3(self):
        mis = multiindex.generate_all_multi_cap((3,))
        mis_ref = [
            multiindex.MultiIndex(0),
            multiindex.MultiIndex(1),
            multiindex.MultiIndex(2),
            multiindex.MultiIndex(3)
        ]
        self.assertEqual(mis, mis_ref)

    def test_2by2(self):
        mis = multiindex.generate_all_multi_cap((2, 2))
        mis_ref = [
            multiindex.MultiIndex(0, 0),
            multiindex.MultiIndex(1, 0),
            multiindex.MultiIndex(2, 0),
            multiindex.MultiIndex(0, 1),
            multiindex.MultiIndex(1, 1),
            multiindex.MultiIndex(2, 1),
            multiindex.MultiIndex(0, 2),
            multiindex.MultiIndex(1, 2),
            multiindex.MultiIndex(2, 2)
        ]
        self.assertEqual(mis, mis_ref)

    def test_3by3(self):
        mis = multiindex.generate_all_multi_cap((3, 1, 2))
        mis_ref = [
            multiindex.MultiIndex(0, 0, 0),
            multiindex.MultiIndex(1, 0, 0),
            multiindex.MultiIndex(2, 0, 0),
            multiindex.MultiIndex(3, 0, 0),
            multiindex.MultiIndex(0, 1, 0),
            multiindex.MultiIndex(1, 1, 0),
            multiindex.MultiIndex(2, 1, 0),
            multiindex.MultiIndex(3, 1, 0),
            multiindex.MultiIndex(0, 0, 1),
            multiindex.MultiIndex(1, 0, 1),
            multiindex.MultiIndex(2, 0, 1),
            multiindex.MultiIndex(3, 0, 1),
            multiindex.MultiIndex(0, 1, 1),
            multiindex.MultiIndex(1, 1, 1),
            multiindex.MultiIndex(2, 1, 1),
            multiindex.MultiIndex(3, 1, 1),
            multiindex.MultiIndex(0, 0, 2),
            multiindex.MultiIndex(1, 0, 2),
            multiindex.MultiIndex(2, 0, 2),
            multiindex.MultiIndex(3, 0, 2),
            multiindex.MultiIndex(0, 1, 2),
            multiindex.MultiIndex(1, 1, 2),
            multiindex.MultiIndex(2, 1, 2),
            multiindex.MultiIndex(3, 1, 2)
        ]
        self.assertEqual(mis, mis_ref)

    def test_different_iterables(self):
        # Should work with different iterable input for r
        mis = multiindex.generate_all_multi_cap([2, 2])
        mis_ref = [
            multiindex.MultiIndex(0, 0),
            multiindex.MultiIndex(1, 0),
            multiindex.MultiIndex(2, 0),
            multiindex.MultiIndex(0, 1),
            multiindex.MultiIndex(1, 1),
            multiindex.MultiIndex(2, 1),
            multiindex.MultiIndex(0, 2),
            multiindex.MultiIndex(1, 2),
            multiindex.MultiIndex(2, 2)
        ]
        self.assertEqual(mis, mis_ref)

        mis = multiindex.generate_all_multi_cap(multiindex.MultiIndex((2, 2)))
        mis_ref = [
            multiindex.MultiIndex(0, 0),
            multiindex.MultiIndex(1, 0),
            multiindex.MultiIndex(2, 0),
            multiindex.MultiIndex(0, 1),
            multiindex.MultiIndex(1, 1),
            multiindex.MultiIndex(2, 1),
            multiindex.MultiIndex(0, 2),
            multiindex.MultiIndex(1, 2),
            multiindex.MultiIndex(2, 2)
        ]
        self.assertEqual(mis, mis_ref)


class MultiIndexGenerateAllIncreasingTest(unittest.TestCase):
    def test_dimension_1(self):
        n = 1
        r = 0
        mis = multiindex.generate_all_increasing(n, r)
        mis_ref = [
            multiindex.MultiIndex(0)
        ]
        self.assertEqual(mis, mis_ref)
        r = 1
        mis = multiindex.generate_all_increasing(n, r)
        mis_ref = [
            multiindex.MultiIndex(0),
            multiindex.MultiIndex(1)
        ]
        self.assertEqual(mis, mis_ref)
        r = 2
        mis = multiindex.generate_all_increasing(n, r)
        mis_ref = [
            multiindex.MultiIndex(0),
            multiindex.MultiIndex(1),
            multiindex.MultiIndex(2)
        ]
        self.assertEqual(mis, mis_ref)

    def test_dimension_2(self):
        n = 2
        r = 0
        mis = multiindex.generate_all_increasing(n, r)
        mis_ref = []
        self.assertEqual(mis, mis_ref)
        r = 1
        mis = multiindex.generate_all_increasing(n, r)
        mis_ref = [
            multiindex.MultiIndex((0, 1))
        ]
        self.assertEqual(mis, mis_ref)
        r = 2
        mis = multiindex.generate_all_increasing(n, r)
        mis_ref = [
            multiindex.MultiIndex((0, 1)),
            multiindex.MultiIndex((0, 2)),
            multiindex.MultiIndex((1, 2))
        ]
        self.assertEqual(mis, mis_ref)

    def test_dimension_3(self):
        n = 3
        r = 0
        mis = multiindex.generate_all_increasing(n, r)
        mis_ref = []
        self.assertEqual(mis, mis_ref)
        r = 1
        mis = multiindex.generate_all_increasing(n, r)
        mis_ref = []
        self.assertEqual(mis, mis_ref)
        r = 2
        mis = multiindex.generate_all_increasing(n, r)
        mis_ref = [
            multiindex.MultiIndex((0, 1, 2))
        ]
        self.assertEqual(mis, mis_ref)
        r = 3
        mis = multiindex.generate_all_increasing(n, r)
        mis_ref = [
            multiindex.MultiIndex((0, 1, 2)),
            multiindex.MultiIndex((0, 1, 3)),
            multiindex.MultiIndex((0, 2, 3)),
            multiindex.MultiIndex((1, 2, 3))
        ]
        self.assertEqual(mis, mis_ref)


class MultiIndexGenerateAllNonDecreasingTest(unittest.TestCase):
    def test_dimension_1(self):
        n = 1
        r = 0
        mis = multiindex.generate_all_non_decreasing(n, r)
        mis_ref = [
            multiindex.MultiIndex(0)
        ]
        self.assertEqual(mis, mis_ref)
        r = 1
        mis = multiindex.generate_all_non_decreasing(n, r)
        mis_ref = [
            multiindex.MultiIndex(0),
            multiindex.MultiIndex(1)
        ]
        self.assertEqual(mis, mis_ref)
        r = 2
        mis = multiindex.generate_all_non_decreasing(n, r)
        mis_ref = [
            multiindex.MultiIndex(0),
            multiindex.MultiIndex(1),
            multiindex.MultiIndex(2)
        ]
        self.assertEqual(mis, mis_ref)

    def test_dimension_2(self):
        n = 2
        r = 0
        mis = multiindex.generate_all_non_decreasing(n, r)
        mis_ref = [
            multiindex.MultiIndex((0, 0))
        ]
        self.assertEqual(mis, mis_ref)
        r = 1
        mis = multiindex.generate_all_non_decreasing(n, r)
        mis_ref = [
            multiindex.MultiIndex((0, 0)),
            multiindex.MultiIndex((0, 1)),
            multiindex.MultiIndex((1, 1))
        ]
        self.assertEqual(mis, mis_ref)
        r = 2
        mis = multiindex.generate_all_non_decreasing(n, r)
        mis_ref = [
            multiindex.MultiIndex((0, 0)),
            multiindex.MultiIndex((0, 1)),
            multiindex.MultiIndex((1, 1)),
            multiindex.MultiIndex((0, 2)),
            multiindex.MultiIndex((1, 2)),
            multiindex.MultiIndex((2, 2))
        ]
        self.assertEqual(mis, mis_ref)

    def test_dimension_3(self):
        n = 3
        r = 0
        mis = multiindex.generate_all_non_decreasing(n, r)
        mis_ref = [
            multiindex.MultiIndex((0, 0, 0)),
        ]
        self.assertEqual(mis, mis_ref)
        r = 1
        mis = multiindex.generate_all_non_decreasing(n, r)
        mis_ref = [
            multiindex.MultiIndex((0, 0, 0)),
            multiindex.MultiIndex((0, 0, 1)),
            multiindex.MultiIndex((0, 1, 1)),
            multiindex.MultiIndex((1, 1, 1))
        ]
        self.assertEqual(mis, mis_ref)
        r = 2
        mis = multiindex.generate_all_non_decreasing(n, r)
        mis_ref = [
            multiindex.MultiIndex((0, 0, 0)),
            multiindex.MultiIndex((0, 0, 1)),
            multiindex.MultiIndex((0, 1, 1)),
            multiindex.MultiIndex((1, 1, 1)),
            multiindex.MultiIndex((0, 0, 2)),
            multiindex.MultiIndex((0, 1, 2)),
            multiindex.MultiIndex((1, 1, 2)),
            multiindex.MultiIndex((0, 2, 2)),
            multiindex.MultiIndex((1, 2, 2)),
            multiindex.MultiIndex((2, 2, 2))
        ]
        self.assertEqual(mis, mis_ref)


class MultiIndexGenerateMultiCapTest(unittest.TestCase):
    def test_generate(self):
        r = (2, 3, 4)
        mis = multiindex.generate_all_multi_cap(r)
        for i in range(3 * 4 * 5):
            mi = multiindex.generate_multi_cap(r, i)
            self.assertEqual(mi, mis[i])


class EnumerateMultiCapMultiIndices(unittest.TestCase):
    def test1(self):
        r = 1, 2
        expected_result = [
            multiindex.MultiIndex((0, 0)),
            multiindex.MultiIndex((1, 0)),
            multiindex.MultiIndex((0, 1)),
            multiindex.MultiIndex((1, 1)),
            multiindex.MultiIndex((0, 2)),
            multiindex.MultiIndex((1, 2))
        ]
        for i in range(len(expected_result)):
            result = multiindex.generate_multi_cap(r, i)
            self.assertTrue(result == expected_result[i])

    def test2(self):
        r = 2, 3
        expected_result = [
            multiindex.MultiIndex((0, 0)),
            multiindex.MultiIndex((1, 0)),
            multiindex.MultiIndex((2, 0)),
            multiindex.MultiIndex((0, 1)),
            multiindex.MultiIndex((1, 1)),
            multiindex.MultiIndex((2, 1)),
            multiindex.MultiIndex((0, 2)),
            multiindex.MultiIndex((1, 2)),
            multiindex.MultiIndex((2, 2)),
            multiindex.MultiIndex((0, 3)),
            multiindex.MultiIndex((1, 3)),
            multiindex.MultiIndex((2, 3))
        ]
        for i in range(len(expected_result)):
            result = multiindex.generate_multi_cap(r, i)
            self.assertTrue(result == expected_result[i])

    def test3(self):
        r = 2, 3, 2
        expected_result = [
            multiindex.MultiIndex((0, 0, 0)),
            multiindex.MultiIndex((1, 0, 0)),
            multiindex.MultiIndex((2, 0, 0)),
            multiindex.MultiIndex((0, 1, 0)),
            multiindex.MultiIndex((1, 1, 0)),
            multiindex.MultiIndex((2, 1, 0)),
            multiindex.MultiIndex((0, 2, 0)),
            multiindex.MultiIndex((1, 2, 0)),
            multiindex.MultiIndex((2, 2, 0)),
            multiindex.MultiIndex((0, 3, 0)),
            multiindex.MultiIndex((1, 3, 0)),
            multiindex.MultiIndex((2, 3, 0)),
            multiindex.MultiIndex((0, 0, 1)),
            multiindex.MultiIndex((1, 0, 1)),
            multiindex.MultiIndex((2, 0, 1)),
            multiindex.MultiIndex((0, 1, 1)),
            multiindex.MultiIndex((1, 1, 1)),
            multiindex.MultiIndex((2, 1, 1)),
            multiindex.MultiIndex((0, 2, 1)),
            multiindex.MultiIndex((1, 2, 1)),
            multiindex.MultiIndex((2, 2, 1)),
            multiindex.MultiIndex((0, 3, 1)),
            multiindex.MultiIndex((1, 3, 1)),
            multiindex.MultiIndex((2, 3, 1)),
            multiindex.MultiIndex((0, 0, 2)),
            multiindex.MultiIndex((1, 0, 2)),
            multiindex.MultiIndex((2, 0, 2)),
            multiindex.MultiIndex((0, 1, 2)),
            multiindex.MultiIndex((1, 1, 2)),
            multiindex.MultiIndex((2, 1, 2)),
            multiindex.MultiIndex((0, 2, 2)),
            multiindex.MultiIndex((1, 2, 2)),
            multiindex.MultiIndex((2, 2, 2)),
            multiindex.MultiIndex((0, 3, 2)),
            multiindex.MultiIndex((1, 3, 2)),
            multiindex.MultiIndex((2, 3, 2))
        ]
        for i in range(len(expected_result)):
            result = multiindex.generate_multi_cap(r, i)
            self.assertTrue(result == expected_result[i])

    def test4(self):
        r = 1, 1, 1, 1
        expected_result = [
            multiindex.MultiIndex((0, 0, 0, 0)),
            multiindex.MultiIndex((1, 0, 0, 0)),
            multiindex.MultiIndex((0, 1, 0, 0)),
            multiindex.MultiIndex((1, 1, 0, 0)),
            multiindex.MultiIndex((0, 0, 1, 0)),
            multiindex.MultiIndex((1, 0, 1, 0)),
            multiindex.MultiIndex((0, 1, 1, 0)),
            multiindex.MultiIndex((1, 1, 1, 0)),
            multiindex.MultiIndex((0, 0, 0, 1)),
            multiindex.MultiIndex((1, 0, 0, 1)),
            multiindex.MultiIndex((0, 1, 0, 1)),
            multiindex.MultiIndex((1, 1, 0, 1)),
            multiindex.MultiIndex((0, 0, 1, 1)),
            multiindex.MultiIndex((1, 0, 1, 1)),
            multiindex.MultiIndex((0, 1, 1, 1)),
            multiindex.MultiIndex((1, 1, 1, 1))
        ]
        for i in range(len(expected_result)):
            result = multiindex.generate_multi_cap(r, i)
            self.assertTrue(result == expected_result[i])


if __name__ == "__main__":
    unittest.main()
