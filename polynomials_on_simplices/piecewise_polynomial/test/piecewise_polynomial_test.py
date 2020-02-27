import sys

import pytest

from polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial import (
    generate_inverse_local_to_global_map, generate_inverse_vector_valued_local_to_global_map,
    generate_local_to_global_map, generate_vector_valued_local_to_global_map)


class TestLocalToGlobalMapDp:
    @staticmethod
    def test_1d():
        # Test L^2 conforming
        lines = [
            [0, 1],
            [0, 2],
            [0, 3]
        ]
        r = 0
        expected_local_to_global_map = [
            {
                (0,): 0
            },
            {
                (0,): 1
            },
            {
                (0,): 2
            }
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(lines, r)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 3

        # Test L^2 conforming
        lines = [
            [0, 1],
            [0, 2],
            [0, 3]
        ]
        r = 2
        expected_local_to_global_map = [
            {
                (0,): 0,
                (1,): 1,
                (2,): 2
            },
            {
                (0,): 3,
                (1,): 4,
                (2,): 5
            },
            {
                (0,): 6,
                (1,): 7,
                (2,): 8
            }
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(lines, r)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 9

        # Test L^2_0 conforming
        expected_local_to_global_map = [
            {
                (0,): -1,
                (1,): -1,
                (2,): -1
            },
            {
                (0,): -1,
                (1,): 0,
                (2,): 1
            },
            {
                (0,): -1,
                (1,): 2,
                (2,): 3
            }
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(lines, r, [[0, 1]])
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 4

        # Test L^2 conforming with boundary DOFs enumerated last
        r = 2
        expected_local_to_global_map = [
            {
                (0,): 6,
                (1,): 0,
                (2,): 1
            },
            {
                (0,): 7,
                (1,): 2,
                (2,): 3
            },
            {
                (0,): 8,
                (1,): 4,
                (2,): 5
            }
        ]
        local_to_global_map, num_dofs, num_interior_dofs = generate_local_to_global_map(lines, r, [[0]],
                                                                                        keep_boundary_dofs_last=True)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 9
        assert num_interior_dofs == 6

    def test_2d(self):
        # Test L^2 conforming
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        expected_local_to_global_map = [
            {
                (0, 0): 0,
                (1, 0): 1,
                (2, 0): 2,
                (3, 0): 3,
                (0, 1): 4,
                (1, 1): 5,
                (2, 1): 6,
                (0, 2): 7,
                (1, 2): 8,
                (0, 3): 9
            },
            {
                (0, 0): 10,
                (1, 0): 11,
                (2, 0): 12,
                (3, 0): 13,
                (0, 1): 14,
                (1, 1): 15,
                (2, 1): 16,
                (0, 2): 17,
                (1, 2): 18,
                (0, 3): 19
            }
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 20

        # Test L^2_0 conforming
        expected_local_to_global_map = [
            {
                (0, 0): 0,
                (1, 0): 1,
                (2, 0): 2,
                (3, 0): -1,
                (0, 1): 3,
                (1, 1): 4,
                (2, 1): -1,
                (0, 2): 5,
                (1, 2): -1,
                (0, 3): -1
            },
            {
                (0, 0): 6,
                (1, 0): 7,
                (2, 0): 8,
                (3, 0): 9,
                (0, 1): 10,
                (1, 1): 11,
                (2, 1): 12,
                (0, 2): 13,
                (1, 2): 14,
                (0, 3): -1
            }
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r, [[1, 2]])
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 15

        # Test L^2 conforming with boundary DOFs enumerated last
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        expected_local_to_global_map = [
            {
                (0, 0): 0,
                (1, 0): 1,
                (2, 0): 2,
                (3, 0): 15,
                (0, 1): 3,
                (1, 1): 4,
                (2, 1): 16,
                (0, 2): 5,
                (1, 2): 17,
                (0, 3): 18
            },
            {
                (0, 0): 6,
                (1, 0): 7,
                (2, 0): 8,
                (3, 0): 9,
                (0, 1): 10,
                (1, 1): 11,
                (2, 1): 12,
                (0, 2): 13,
                (1, 2): 14,
                (0, 3): 19
            }
        ]
        local_to_global_map, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, [[1, 2]],
                                                                                        keep_boundary_dofs_last=True)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 20
        assert num_interior_dofs == 15

        # Test L^2 conforming with boundary DOFs enumerated last, with an internal "boundary" edge
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        expected_local_to_global_map = [
            {
                (0, 0): 12,
                (1, 0): 13,
                (2, 0): 14,
                (3, 0): 15,
                (0, 1): 0,
                (1, 1): 1,
                (2, 1): 2,
                (0, 2): 3,
                (1, 2): 4,
                (0, 3): 5
            },
            {
                (0, 0): 16,
                (1, 0): 6,
                (2, 0): 7,
                (3, 0): 8,
                (0, 1): 17,
                (1, 1): 9,
                (2, 1): 10,
                (0, 2): 18,
                (1, 2): 11,
                (0, 3): 19
            }
        ]
        local_to_global_map, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, [[0, 1]],
                                                                                        keep_boundary_dofs_last=True)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 20
        assert num_interior_dofs == 12

    def test_3d(self):
        triangles = [
            [0, 1, 2, 3],
            [3, 1, 2, 4]
        ]
        r = 4
        expected_local_to_global_map = [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (2, 0, 0): 2,
                (3, 0, 0): 3,
                (4, 0, 0): 4,
                (0, 1, 0): 5,
                (1, 1, 0): 6,
                (2, 1, 0): 7,
                (3, 1, 0): 8,
                (0, 2, 0): 9,
                (1, 2, 0): 10,
                (2, 2, 0): 11,
                (0, 3, 0): 12,
                (1, 3, 0): 13,
                (0, 4, 0): 14,
                (0, 0, 1): 15,
                (1, 0, 1): 16,
                (2, 0, 1): 17,
                (3, 0, 1): 18,
                (0, 1, 1): 19,
                (1, 1, 1): 20,
                (2, 1, 1): 21,
                (0, 2, 1): 22,
                (1, 2, 1): 23,
                (0, 3, 1): 24,
                (0, 0, 2): 25,
                (1, 0, 2): 26,
                (2, 0, 2): 27,
                (0, 1, 2): 28,
                (1, 1, 2): 29,
                (0, 2, 2): 30,
                (0, 0, 3): 31,
                (1, 0, 3): 32,
                (0, 1, 3): 33,
                (0, 0, 4): 34
            },
            {
                (0, 0, 0): 35,
                (1, 0, 0): 36,
                (2, 0, 0): 37,
                (3, 0, 0): 38,
                (4, 0, 0): 39,
                (0, 1, 0): 40,
                (1, 1, 0): 41,
                (2, 1, 0): 42,
                (3, 1, 0): 43,
                (0, 2, 0): 44,
                (1, 2, 0): 45,
                (2, 2, 0): 46,
                (0, 3, 0): 47,
                (1, 3, 0): 48,
                (0, 4, 0): 49,
                (0, 0, 1): 50,
                (1, 0, 1): 51,
                (2, 0, 1): 52,
                (3, 0, 1): 53,
                (0, 1, 1): 54,
                (1, 1, 1): 55,
                (2, 1, 1): 56,
                (0, 2, 1): 57,
                (1, 2, 1): 58,
                (0, 3, 1): 59,
                (0, 0, 2): 60,
                (1, 0, 2): 61,
                (2, 0, 2): 62,
                (0, 1, 2): 63,
                (1, 1, 2): 64,
                (0, 2, 2): 65,
                (0, 0, 3): 66,
                (1, 0, 3): 67,
                (0, 1, 3): 68,
                (0, 0, 4): 69
            },
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 70

        expected_local_to_global_map = [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (2, 0, 0): 2,
                (3, 0, 0): 3,
                (4, 0, 0): -1,
                (0, 1, 0): 4,
                (1, 1, 0): 5,
                (2, 1, 0): 6,
                (3, 1, 0): 7,
                (0, 2, 0): 8,
                (1, 2, 0): 9,
                (2, 2, 0): 10,
                (0, 3, 0): 11,
                (1, 3, 0): 12,
                (0, 4, 0): 13,
                (0, 0, 1): 14,
                (1, 0, 1): 15,
                (2, 0, 1): 16,
                (3, 0, 1): -1,
                (0, 1, 1): 17,
                (1, 1, 1): 18,
                (2, 1, 1): 19,
                (0, 2, 1): 20,
                (1, 2, 1): 21,
                (0, 3, 1): 22,
                (0, 0, 2): 23,
                (1, 0, 2): 24,
                (2, 0, 2): -1,
                (0, 1, 2): 25,
                (1, 1, 2): 26,
                (0, 2, 2): 27,
                (0, 0, 3): 28,
                (1, 0, 3): -1,
                (0, 1, 3): 29,
                (0, 0, 4): -1
            },
            {
                (0, 0, 0): -1,
                (1, 0, 0): -1,
                (2, 0, 0): -1,
                (3, 0, 0): -1,
                (4, 0, 0): -1,
                (0, 1, 0): 30,
                (1, 1, 0): 31,
                (2, 1, 0): 32,
                (3, 1, 0): 33,
                (0, 2, 0): 34,
                (1, 2, 0): 35,
                (2, 2, 0): 36,
                (0, 3, 0): 37,
                (1, 3, 0): 38,
                (0, 4, 0): 39,
                (0, 1, 1): 40,
                (1, 1, 1): 41,
                (2, 1, 1): 42,
                (0, 2, 1): 43,
                (1, 2, 1): 44,
                (0, 3, 1): 45,
                (0, 0, 2): -1,
                (1, 0, 2): -1,
                (2, 0, 2): -1,
                (0, 1, 2): 46,
                (1, 1, 2): 47,
                (0, 2, 2): 48,
                (0, 1, 3): 49,
                (0, 0, 4): -1
            },
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r, [[3, 4, 1]])
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 50

    @staticmethod
    def test_1d2d():
        # 1D domain with values in R^2

        triangles = [
            [0, 1],
            [0, 2],
            [0, 3]
        ]
        n = 2
        r = 2
        expected_local_to_global_map = [
            {
                ((0,), 0): 0,
                ((0,), 1): 1,
                ((1,), 0): 2,
                ((1,), 1): 3,
                ((2,), 0): 4,
                ((2,), 1): 5
            },
            {
                ((0,), 0): 6,
                ((0,), 1): 7,
                ((1,), 0): 8,
                ((1,), 1): 9,
                ((2,), 0): 10,
                ((2,), 1): 11
            },
            {
                ((0,), 0): 12,
                ((0,), 1): 13,
                ((1,), 0): 14,
                ((1,), 1): 15,
                ((2,), 0): 16,
                ((2,), 1): 17
            }
        ]
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n)
        for j in range(len(triangles)):
            for mik, expected_value in expected_local_to_global_map[j].items():
                mi, k = mik
                assert local_to_global_map(j, mi, k) == expected_value
        assert num_dofs == 18

    @staticmethod
    def test_1d2d_sequential():
        # 1D domain with values in R^2, with sequential ordering of vector valued basis functions

        triangles = [
            [0, 1],
            [0, 2],
            [0, 3]
        ]
        n = 2
        r = 2
        expected_local_to_global_map = [
            {
                ((0,), 0): 0,
                ((0,), 1): 9,
                ((1,), 0): 1,
                ((1,), 1): 10,
                ((2,), 0): 2,
                ((2,), 1): 11
            },
            {
                ((0,), 0): 3,
                ((0,), 1): 12,
                ((1,), 0): 4,
                ((1,), 1): 13,
                ((2,), 0): 5,
                ((2,), 1): 14
            },
            {
                ((0,), 0): 6,
                ((0,), 1): 15,
                ((1,), 0): 7,
                ((1,), 1): 16,
                ((2,), 0): 8,
                ((2,), 1): 17
            }
        ]
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n,
                                                                                   ordering="sequential")
        for j in range(len(triangles)):
            for mik, expected_value in expected_local_to_global_map[j].items():
                mi, k = mik
                assert local_to_global_map(j, mi, k) == expected_value
        assert num_dofs == 18

    @staticmethod
    def test_2d2d():
        # 2D domain with values in R^2

        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        n = 2
        r = 3
        expected_local_to_global_map = [
            {
                ((0, 0), 0): 0,
                ((0, 0), 1): 1,
                ((1, 0), 0): 2,
                ((1, 0), 1): 3,
                ((2, 0), 0): 4,
                ((2, 0), 1): 5,
                ((3, 0), 0): 6,
                ((3, 0), 1): 7,
                ((0, 1), 0): 8,
                ((0, 1), 1): 9,
                ((1, 1), 0): 10,
                ((1, 1), 1): 11,
                ((2, 1), 0): 12,
                ((2, 1), 1): 13,
                ((0, 2), 0): 14,
                ((0, 2), 1): 15,
                ((1, 2), 0): 16,
                ((1, 2), 1): 17,
                ((0, 3), 0): 18,
                ((0, 3), 1): 19
            },
            {
                ((0, 0), 0): 20,
                ((0, 0), 1): 21,
                ((1, 0), 0): 22,
                ((1, 0), 1): 23,
                ((2, 0), 0): 24,
                ((2, 0), 1): 25,
                ((3, 0), 0): 26,
                ((3, 0), 1): 27,
                ((0, 1), 0): 28,
                ((0, 1), 1): 29,
                ((1, 1), 0): 30,
                ((1, 1), 1): 31,
                ((2, 1), 0): 32,
                ((2, 1), 1): 33,
                ((0, 2), 0): 34,
                ((0, 2), 1): 35,
                ((1, 2), 0): 36,
                ((1, 2), 1): 37,
                ((0, 3), 0): 38,
                ((0, 3), 1): 39
            }
        ]
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n)
        for j in range(len(triangles)):
            for mik, expected_value in expected_local_to_global_map[j].items():
                mi, k = mik
                assert local_to_global_map(j, mi, k) == expected_value
        assert num_dofs == 40

        expected_local_to_global_map = [
            {
                ((0, 0), 0): 0,
                ((0, 0), 1): 1,
                ((1, 0), 0): 2,
                ((1, 0), 1): 3,
                ((2, 0), 0): 4,
                ((2, 0), 1): 5,
                ((3, 0), 0): -1,
                ((3, 0), 1): -1,
                ((0, 1), 0): 6,
                ((0, 1), 1): 7,
                ((1, 1), 0): 8,
                ((1, 1), 1): 9,
                ((2, 1), 0): -1,
                ((2, 1), 1): -1,
                ((0, 2), 0): 10,
                ((0, 2), 1): 11,
                ((1, 2), 0): -1,
                ((1, 2), 1): -1,
                ((0, 3), 0): -1,
                ((0, 3), 1): -1
            },
            {
                ((0, 0), 0): 12,
                ((0, 0), 1): 13,
                ((1, 0), 0): 14,
                ((1, 0), 1): 15,
                ((2, 0), 0): 16,
                ((2, 0), 1): 17,
                ((3, 0), 0): 18,
                ((3, 0), 1): 19,
                ((0, 1), 0): 20,
                ((0, 1), 1): 21,
                ((1, 1), 0): 22,
                ((1, 1), 1): 23,
                ((2, 1), 0): 24,
                ((2, 1), 1): 25,
                ((0, 2), 0): 26,
                ((0, 2), 1): 27,
                ((1, 2), 0): 28,
                ((1, 2), 1): 29,
            }
        ]
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n, [[1, 2]])
        for j in range(len(triangles)):
            for mik, expected_value in expected_local_to_global_map[j].items():
                mi, k = mik
                assert local_to_global_map(j, mi, k) == expected_value
        assert num_dofs == 30

    @staticmethod
    def test_2d2d_sequential():
        # 2D domain with values in R^2, with sequential ordering of vector valued basis functions

        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        n = 2
        r = 3
        expected_local_to_global_map = [
            {
                ((0, 0), 0): 0,
                ((0, 0), 1): 20,
                ((1, 0), 0): 1,
                ((1, 0), 1): 21,
                ((2, 0), 0): 2,
                ((2, 0), 1): 22,
                ((3, 0), 0): 3,
                ((3, 0), 1): 23,
                ((0, 1), 0): 4,
                ((0, 1), 1): 24,
                ((1, 1), 0): 5,
                ((1, 1), 1): 25,
                ((2, 1), 0): 6,
                ((2, 1), 1): 26,
                ((0, 2), 0): 7,
                ((0, 2), 1): 27,
                ((1, 2), 0): 8,
                ((1, 2), 1): 28,
                ((0, 3), 0): 9,
                ((0, 3), 1): 29
            },
            {
                ((0, 0), 0): 10,
                ((0, 0), 1): 30,
                ((1, 0), 0): 11,
                ((1, 0), 1): 31,
                ((2, 0), 0): 12,
                ((2, 0), 1): 32,
                ((3, 0), 0): 13,
                ((3, 0), 1): 33,
                ((0, 1), 0): 14,
                ((0, 1), 1): 34,
                ((1, 1), 0): 15,
                ((1, 1), 1): 35,
                ((2, 1), 0): 16,
                ((2, 1), 1): 36,
                ((0, 2), 0): 17,
                ((0, 2), 1): 37,
                ((1, 2), 0): 18,
                ((1, 2), 1): 38,
                ((0, 3), 0): 19,
                ((0, 3), 1): 39
            }
        ]
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n,
                                                                                   ordering="sequential")
        for j in range(len(triangles)):
            for mik, expected_value in expected_local_to_global_map[j].items():
                mi, k = mik
                assert local_to_global_map(j, mi, k) == expected_value
        assert num_dofs == 40

        expected_local_to_global_map = [
            {
                ((0, 0), 0): 0,
                ((0, 0), 1): 15,
                ((1, 0), 0): 1,
                ((1, 0), 1): 16,
                ((2, 0), 0): 2,
                ((2, 0), 1): 17,
                ((3, 0), 0): -1,
                ((3, 0), 1): -1,
                ((0, 1), 0): 3,
                ((0, 1), 1): 18,
                ((1, 1), 0): 4,
                ((1, 1), 1): 19,
                ((2, 1), 0): -1,
                ((2, 1), 1): -1,
                ((0, 2), 0): 5,
                ((0, 2), 1): 20,
                ((1, 2), 0): -1,
                ((1, 2), 1): -1,
                ((0, 3), 0): -1,
                ((0, 3), 1): -1
            },
            {
                ((0, 0), 0): 6,
                ((0, 0), 1): 21,
                ((1, 0), 0): 7,
                ((1, 0), 1): 22,
                ((2, 0), 0): 8,
                ((2, 0), 1): 23,
                ((3, 0), 0): 9,
                ((3, 0), 1): 24,
                ((0, 1), 0): 10,
                ((0, 1), 1): 25,
                ((1, 1), 0): 11,
                ((1, 1), 1): 26,
                ((2, 1), 0): 12,
                ((2, 1), 1): 27,
                ((0, 2), 0): 13,
                ((0, 2), 1): 28,
                ((1, 2), 0): 14,
                ((1, 2), 1): 29,
            }
        ]
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n, [[1, 2]],
                                                                                   ordering="sequential")
        for j in range(len(triangles)):
            for mik, expected_value in expected_local_to_global_map[j].items():
                mi, k = mik
                assert local_to_global_map(j, mi, k) == expected_value
        assert num_dofs == 30


class TestInverseLocalToGlobalMapDp:
    @staticmethod
    def test_1d():
        # Test L^2 conforming
        lines = [
            [0, 1],
            [0, 2],
            [0, 3]
        ]
        r = 2
        local_to_global_map, num_dofs = generate_local_to_global_map(lines, r)
        inverse_local_to_global_map = generate_inverse_local_to_global_map(local_to_global_map, len(lines), num_dofs, r,
                                                                           1)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2_0 conforming
        local_to_global_map, num_dofs = generate_local_to_global_map(lines, r, [[0, 1]])
        inverse_local_to_global_map = generate_inverse_local_to_global_map(local_to_global_map, len(lines), num_dofs, r,
                                                                           1)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2 conforming with boundary DOFs enumerated last
        r = 2
        local_to_global_map, num_dofs, num_interior_dofs = generate_local_to_global_map(lines, r, [[0]],
                                                                                        keep_boundary_dofs_last=True)
        inverse_local_to_global_map = generate_inverse_local_to_global_map(local_to_global_map, len(lines), num_dofs, r,
                                                                           1)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

    def test_2d(self):
        # Test L^2 conforming
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r)
        inverse_local_to_global_map = generate_inverse_local_to_global_map(local_to_global_map, len(triangles),
                                                                           num_dofs, r, 2)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2_0 conforming
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r, [[1, 2]])
        inverse_local_to_global_map = generate_inverse_local_to_global_map(local_to_global_map, len(triangles),
                                                                           num_dofs, r, 2)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2 conforming with boundary DOFs enumerated last
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        local_to_global_map, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, [[1, 2]],
                                                                                        keep_boundary_dofs_last=True)
        inverse_local_to_global_map = generate_inverse_local_to_global_map(local_to_global_map, len(triangles),
                                                                           num_dofs, r, 2)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2 conforming with boundary DOFs enumerated last, with an internal "boundary" edge
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        local_to_global_map, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, [[0, 1]],
                                                                                        keep_boundary_dofs_last=True)
        inverse_local_to_global_map = generate_inverse_local_to_global_map(local_to_global_map, len(triangles),
                                                                           num_dofs, r, 2)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

    def test_3d(self):
        triangles = [
            [0, 1, 2, 3],
            [3, 1, 2, 4]
        ]
        r = 4
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r)
        inverse_local_to_global_map = generate_inverse_local_to_global_map(local_to_global_map, len(triangles),
                                                                           num_dofs, r, 3)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r, [[3, 4, 1]])
        inverse_local_to_global_map = generate_inverse_local_to_global_map(local_to_global_map, len(triangles),
                                                                           num_dofs, r, 3)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

    @staticmethod
    def test_1d2d():
        # Vector valued basis functions

        # Test L^2 conforming
        lines = [
            [0, 1],
            [0, 2],
            [0, 3]
        ]
        r = 2
        n = 2
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(lines, r, n)
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(lines), num_dofs, r, 1, n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2_0 conforming
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(lines, r, n, [[0, 1]])
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(lines), num_dofs, r, 1, n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2 conforming with boundary DOFs enumerated last
        r = 2
        local_to_global_map, num_dofs, num_interior_dofs =\
            generate_vector_valued_local_to_global_map(lines, r, n, [[0]], keep_boundary_dofs_last=True)
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(lines), num_dofs, r, 1, n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

    @staticmethod
    def test_1d2d_sequential():
        # Vector valued basis functions with sequential ordering

        # Test L^2 conforming
        lines = [
            [0, 1],
            [0, 2],
            [0, 3]
        ]
        r = 2
        n = 2
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(lines, r, n, ordering="sequential")
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(lines), num_dofs, r, 1, n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2_0 conforming
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(lines, r, n, [[0, 1]],
                                                                                   ordering="sequential")
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(lines), num_dofs, r, 1, n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2 conforming with boundary DOFs enumerated last
        r = 2
        local_to_global_map, num_dofs, num_interior_dofs = \
            generate_vector_valued_local_to_global_map(lines, r, n, [[0]], keep_boundary_dofs_last=True,
                                                       ordering="sequential")
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(lines), num_dofs, r, 1, n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

    @staticmethod
    def test_2d2d():
        # Vector valued basis functions

        # Test L^2 conforming
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        n = 2
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n)
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(triangles), num_dofs, r, 2,
                                                                                         n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2_0 conforming
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n, [[1, 2]])
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(triangles), num_dofs, r, 2,
                                                                                         n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2 conforming with boundary DOFs enumerated last
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        local_to_global_map, num_dofs, num_interior_dofs =\
            generate_vector_valued_local_to_global_map(triangles, r, n, [[1, 2]], keep_boundary_dofs_last=True)
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(triangles), num_dofs, r, 2,
                                                                                         n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2 conforming with boundary DOFs enumerated last, with an internal "boundary" edge
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        local_to_global_map, num_dofs, num_interior_dofs =\
            generate_vector_valued_local_to_global_map(triangles, r, n, [[0, 1]], keep_boundary_dofs_last=True)
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(triangles), num_dofs, r, 2,
                                                                                         n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

    @staticmethod
    def test_2d2d_sequential():
        # Vector valued basis functions with sequential ordering

        # Test L^2 conforming
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        n = 2
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n,
                                                                                   ordering="sequential")
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(triangles), num_dofs, r, 2,
                                                                                         n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2_0 conforming
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n, [[1, 2]],
                                                                                   ordering="sequential")
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(triangles), num_dofs, r, 2,
                                                                                         n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2 conforming with boundary DOFs enumerated last
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        local_to_global_map, num_dofs, num_interior_dofs =\
            generate_vector_valued_local_to_global_map(triangles, r, n, [[1, 2]], keep_boundary_dofs_last=True,
                                                       ordering="sequential")
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(triangles), num_dofs, r, 2,
                                                                                         n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i

        # Test L^2 conforming with boundary DOFs enumerated last, with an internal "boundary" edge
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        local_to_global_map, num_dofs, num_interior_dofs =\
            generate_vector_valued_local_to_global_map(triangles, r, n, [[0, 1]], keep_boundary_dofs_last=True,
                                                       ordering="sequential")
        inverse_local_to_global_map = generate_inverse_vector_valued_local_to_global_map(local_to_global_map,
                                                                                         len(triangles), num_dofs, r, 2,
                                                                                         n)
        for i in range(num_dofs):
            assert local_to_global_map(*inverse_local_to_global_map(i)) == i


if __name__ == '__main__':
    pytest.main(sys.argv)
