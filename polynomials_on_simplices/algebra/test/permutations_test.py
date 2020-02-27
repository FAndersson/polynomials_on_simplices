import unittest

import numpy as np

from polynomials_on_simplices.algebra.permutations import (
    Permutation, circularly_equivalent, composition, construct_permutation, from_one_based, from_transpositions,
    generate_increasing_subset_permutations, generate_random_permutation, generate_subset_permutations,
    increasing_subset_permutations, inverse, num_fixed_points, num_increasing_subset_permutations, num_permutations,
    num_subset_permutations, permutation_matrix_positions, permutation_matrix_values, permutations, permute_positions,
    permute_values, sign, subset_permutations, swap, to_transpositions)


class TestTranspositions(unittest.TestCase):
    def test_to_transpositions(self):
        permutation = (2, 4, 3, 0, 1)
        transpositions = to_transpositions(permutation)
        expected_transpositions = [(0, 2), (0, 3), (1, 4)]
        self.assertEqual(expected_transpositions, transpositions)

        permutation = (1, 2, 3, 4, 0)
        transpositions = to_transpositions(permutation)
        expected_transpositions = [(0, 1), (0, 2), (0, 3), (0, 4)]
        self.assertEqual(expected_transpositions, transpositions)

    def test_from_transpositions(self):
        transpositions = [(0, 2), (0, 3), (1, 4)]
        permutation = from_transpositions(transpositions, 5)
        expected_permutation = (2, 4, 3, 0, 1)
        self.assertEqual(expected_permutation, permutation)

        transpositions = [(3, 4), (2, 3), (1, 2), (0, 1)]
        permutation = from_transpositions(transpositions, 5)
        expected_permutation = (1, 2, 3, 4, 0)
        self.assertEqual(expected_permutation, permutation)


class TestGroupOperations(unittest.TestCase):
    def test_composition(self):
        # Example from Wikipedia
        p = from_one_based((2, 4, 1, 3, 5))
        q = from_one_based((5, 4, 3, 2, 1))
        qp = composition(q, p)
        expected_qp = from_one_based((4, 2, 5, 3, 1))
        self.assertEqual(expected_qp, qp)
        # Same test using the Permutation class
        p = Permutation(from_one_based((2, 4, 1, 3, 5)))
        q = Permutation(from_one_based((5, 4, 3, 2, 1)))
        qp = q * p
        expected_qp = Permutation(from_one_based((4, 2, 5, 3, 1)))
        self.assertEqual(expected_qp, qp)

        sigma = from_one_based((5, 3, 1, 2, 4))
        pi = from_one_based((5, 4, 3, 2, 1))
        sigma_pi = composition(sigma, pi)
        pi_sigma = composition(pi, sigma)
        expected_sigma_pi = from_one_based((4, 2, 1, 3, 5))
        expected_pi_sigma = from_one_based((1, 3, 5, 4, 2))
        self.assertEqual(expected_sigma_pi, sigma_pi)
        self.assertEqual(expected_pi_sigma, pi_sigma)
        # Same tests using the Permutation class
        sigma = Permutation(from_one_based((5, 3, 1, 2, 4)))
        pi = Permutation(from_one_based((5, 4, 3, 2, 1)))
        sigma_pi = sigma * pi
        pi_sigma = pi * sigma
        expected_sigma_pi = Permutation(from_one_based((4, 2, 1, 3, 5)))
        expected_pi_sigma = Permutation(from_one_based((1, 3, 5, 4, 2)))
        self.assertEqual(expected_sigma_pi, sigma_pi)
        self.assertEqual(expected_pi_sigma, pi_sigma)

        # Test composition
        a = (3, 0, 2, 1)
        b = (2, 0, 1, 3)
        c = (1, 2, 0, 3)
        abc = composition(a, composition(b, c))
        expected_abc = (3, 0, 2, 1)
        self.assertEqual(expected_abc, abc)
        # Same test using the Permutation class
        a = Permutation((3, 0, 2, 1))
        b = Permutation((2, 0, 1, 3))
        c = Permutation((1, 2, 0, 3))
        abc = a * b * c
        expected_abc = Permutation((3, 0, 2, 1))
        self.assertEqual(expected_abc, abc)

        # Test evaluating a permutation
        s = Permutation(generate_random_permutation(4))
        p = Permutation(generate_random_permutation(4))
        sp = s * p
        for i in range(4):
            self.assertEqual(s(p(i)), sp(i))

    def test_inverse(self):
        # Example from Wikipedia
        p = from_one_based((2, 5, 4, 3, 1))
        p_inv = inverse(p)
        p_inv_expected = from_one_based((5, 1, 4, 3, 2))
        self.assertEqual(p_inv, p_inv_expected)
        # Same test using the Permutation class
        p = Permutation(from_one_based((2, 5, 4, 3, 1)))
        p_inv = p**-1
        p_inv_expected = Permutation(from_one_based((5, 1, 4, 3, 2)))
        self.assertEqual(p_inv, p_inv_expected)


class TestSign(unittest.TestCase):
    def test_sign(self):
        permutation = (0, 1, 2)
        s = sign(permutation)
        self.assertEqual(1, s)

        permutation = (1, 0, 2)
        s = sign(permutation)
        self.assertEqual(-1, s)

        permutation = (1, 2, 0)
        s = sign(permutation)
        self.assertEqual(1, s)

        permutation = (4, 2, 3, 1, 0)
        s = sign(permutation)
        self.assertEqual(-1, s)

        permutation = from_one_based((5, 3, 1, 2, 4))
        s = sign(permutation)
        self.assertEqual(1, s)

    def test_sign_corner_case(self):
        # Handle trivial permutation
        permutation = tuple([0])
        s = sign(permutation)
        self.assertEqual(1, s)


class TestFixedPoints(unittest.TestCase):
    def test_num_fixed_points(self):
        permutation = (0, 1, 2)
        self.assertEqual(3, num_fixed_points(permutation))
        permutation = (2, 1, 0)
        self.assertEqual(1, num_fixed_points(permutation))
        permutation = (2, 0, 1)
        self.assertEqual(0, num_fixed_points(permutation))


class TestSwap(unittest.TestCase):
    def test_swap(self):
        permutation = (0, 1, 2, 3, 4)
        permutation = swap(permutation, (0, 1))
        permutation = swap(permutation, (1, 4))
        permutation = swap(permutation, (2, 3))
        expected_permutation = (1, 4, 3, 2, 0)
        self.assertEqual(expected_permutation, permutation)

        permutation = (0, 1, 2, 3, 4)
        for i in range(4):
            transposition = (i, i + 1)
            permutation = swap(permutation, transposition)
        expected_permutation = (1, 2, 3, 4, 0)
        self.assertEqual(expected_permutation, permutation)

        for i in range(3, 0, -1):
            transposition = (i, i - 1)
            permutation = swap(permutation, transposition)
        expected_permutation = (4, 1, 2, 3, 0)
        self.assertEqual(expected_permutation, permutation)


class TestPermute(unittest.TestCase):
    def test_permute(self):
        permutation = (2, 0, 1, 3)
        sequence = [0, 1, 2, 3]
        permuted_sequence = permute_values(permutation, sequence)
        expected_sequence = [2, 0, 1, 3]
        self.assertEqual(expected_sequence, permuted_sequence)

        permuted_sequence = permute_positions(permutation, sequence)
        expected_sequence = [1, 2, 0, 3]
        self.assertEqual(expected_sequence, permuted_sequence)

        # Permutation of a generic sequence
        sequence = ['a', 'b', 'c', 'd']
        permuted_sequence = permute_values(permutation, sequence)
        expected_sequence = ['c', 'a', 'b', 'd']
        self.assertEqual(expected_sequence, permuted_sequence)

        permuted_sequence = permute_positions(permutation, sequence)
        expected_sequence = ['b', 'c', 'a', 'd']
        self.assertEqual(expected_sequence, permuted_sequence)

        # Test composition
        s = Permutation(0, 3, 1, 2)
        p = Permutation(2, 3, 1, 0)
        permuted_sequence_1 = permute_values(p, permute_values(s, sequence))
        sp = composition(s, p)
        permuted_sequence_2 = permute_values(sp, sequence)
        self.assertEqual(permuted_sequence_1, permuted_sequence_2)

        # Test composition
        s = Permutation(0, 3, 1, 2)
        p = Permutation(2, 3, 1, 0)
        permuted_sequence_1 = permute_positions(s, permute_positions(p, sequence))
        sp = composition(s, p)
        permuted_sequence_2 = permute_positions(sp, sequence)
        self.assertEqual(permuted_sequence_1, permuted_sequence_2)

    def test_random(self):
        sequence = list(range(5))
        permutation = generate_random_permutation(5)
        permuted_sequence = permute_values(permutation, sequence)
        self.assertEqual(list(permutation), permuted_sequence)

        permuted_sequence = permute_positions(permutation, sequence)
        for i in range(len(sequence)):
            self.assertEqual(sequence[i], permuted_sequence[permutation[i]])

    def test_matrix(self):
        permutation = (2, 0, 1, 3)
        sequence = [0, 1, 2, 3]
        m1 = permutation_matrix_values(permutation)
        permuted_sequence = np.dot(m1, sequence)
        expected_sequence = np.array([2, 0, 1, 3])
        self.assertTrue(np.array_equal(expected_sequence, permuted_sequence))

        m2 = permutation_matrix_positions(permutation)
        permuted_sequence = np.dot(m2, sequence)
        expected_sequence = np.array([1, 2, 0, 3])
        self.assertTrue(np.array_equal(expected_sequence, permuted_sequence))

        # Test composition
        s = Permutation(0, 3, 1, 2)
        sm = permutation_matrix_values(s)
        p = Permutation(2, 3, 1, 0)
        pm = permutation_matrix_values(p)
        sp = s * p
        spm = permutation_matrix_values(sp)
        self.assertTrue(np.array_equal(spm, np.dot(pm, sm)))

        sm = permutation_matrix_positions(s)
        pm = permutation_matrix_positions(p)
        sp = s * p
        spm = permutation_matrix_positions(sp)
        self.assertTrue(np.array_equal(spm, np.dot(sm, pm)))


class TestGenerate(unittest.TestCase):
    def test_permutations(self):
        array = (1, 2, 3)
        perm = list(generate_subset_permutations(array, 3))
        expected_perm = [
            (1, 2, 3),
            (1, 3, 2),
            (2, 1, 3),
            (2, 3, 1),
            (3, 1, 2),
            (3, 2, 1)
        ]
        self.assertEqual(expected_perm, perm)

    def test_subset_permutations(self):
        array = (0, 1, 2, 3)
        perm = list(generate_subset_permutations(array, 3))
        expected_perm = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 1),
            (0, 2, 3),
            (0, 3, 1),
            (0, 3, 2),
            (1, 0, 2),
            (1, 0, 3),
            (1, 2, 0),
            (1, 2, 3),
            (1, 3, 0),
            (1, 3, 2),
            (2, 0, 1),
            (2, 0, 3),
            (2, 1, 0),
            (2, 1, 3),
            (2, 3, 0),
            (2, 3, 1),
            (3, 0, 1),
            (3, 0, 2),
            (3, 1, 0),
            (3, 1, 2),
            (3, 2, 0),
            (3, 2, 1)
        ]
        self.assertEqual(perm, expected_perm)

    def test_increasing_subset_permutations(self):
        array = (0, 1, 2, 3)
        perm = list(generate_increasing_subset_permutations(array, 3))
        expected_perm = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3)
        ]
        self.assertEqual(expected_perm, perm)

    def test_corner_cases(self):
        self.assertEqual([], subset_permutations(4, 5))
        self.assertEqual([], subset_permutations(4, 0))

        self.assertEqual([], increasing_subset_permutations(4, 5))
        self.assertEqual([], increasing_subset_permutations(4, 0))


class TestNumPermutations(unittest.TestCase):
    def test_num_permutations(self):
        for n in range(1, 5):
            self.assertEqual(len(permutations(n)), num_permutations(n))

    def test_num_subset_permutations(self):
        for n in range(1, 5):
            for k in range(1, n + 1):
                self.assertEqual(len(subset_permutations(n, k)), num_subset_permutations(n, k))

    def test_num_increasing_subset_permutations(self):
        for n in range(1, 5):
            for k in range(1, n + 1):
                self.assertEqual(len(increasing_subset_permutations(n, k)), num_increasing_subset_permutations(n, k))


class TestCircularEquivalence(unittest.TestCase):
    def test_circularly_equivalent(self):
        self.assertTrue(circularly_equivalent((0, 1, 2), (2, 0, 1)))
        self.assertFalse(circularly_equivalent((0, 1, 2), (0, 2, 1)))

        self.assertTrue(circularly_equivalent((0, 1, 2, 3), (2, 3, 0, 1)))
        self.assertFalse(circularly_equivalent((0, 1, 2, 3), (2, 3, 1, 0)))


class TestConstructPermutation(unittest.TestCase):
    def test_len4(self):
        permutation = construct_permutation([2, 3], [0, 1], 4)
        expected_permutation = (2, 3, 0, 1)
        self.assertEqual(expected_permutation, permutation)

        permutation = construct_permutation([2, 3], [1, 0], 4)
        expected_permutation = (3, 2, 1, 0)
        self.assertEqual(expected_permutation, permutation)

    def test_len5(self):
        permutation = construct_permutation([0, 1, 2, 3, 4], [1, 4, 3, 2, 0], 6)
        expected_permutation = (1, 4, 3, 2, 0, 5)
        self.assertEqual(expected_permutation, permutation)


if __name__ == '__main__':
    unittest.main()
