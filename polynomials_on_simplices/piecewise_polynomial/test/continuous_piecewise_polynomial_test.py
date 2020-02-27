import sys

import pytest

from polynomials_on_simplices.piecewise_polynomial.continuous_piecewise_polynomial import (
    generate_local_to_global_map, generate_local_to_global_preimage_map, generate_vector_valued_local_to_global_map,
    generate_vector_valued_local_to_global_preimage_map)


class TestLocalToGlobalMap:
    @staticmethod
    def test_1d():
        # Test H^1 conforming
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
                (0,): 0,
                (1,): 3,
                (2,): 4
            },
            {
                (0,): 0,
                (1,): 5,
                (2,): 6
            }
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(lines, r)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 7

        # Test H^1_0 conforming
        expected_local_to_global_map = [
            {
            },
            {
                (1,): 0,
                (2,): 1
            },
            {
                (1,): 2,
                (2,): 3
            }
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(lines, r, [[0, 1]])
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 4

        # Test H^1 conforming with boundary DOFs enumerated last
        lines = [
            [0, 1],
            [0, 2],
            [0, 3]
        ]
        r = 2
        expected_local_to_global_map = [
            {
                (0,): 6,
                (1,): 0,
                (2,): 1
            },
            {
                (0,): 6,
                (1,): 2,
                (2,): 3
            },
            {
                (0,): 6,
                (1,): 4,
                (2,): 5
            }
        ]
        local_to_global_map, num_dofs, num_interior_dofs = generate_local_to_global_map(lines, r, [[0]],
                                                                                        keep_boundary_dofs_last=True)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 7

    @staticmethod
    def test_2d():
        # Test H^1 conforming
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
                (0, 0): 0,
                (1, 0): 10,
                (2, 0): 11,
                (3, 0): 12,
                (0, 1): 1,
                (1, 1): 13,
                (2, 1): 14,
                (0, 2): 2,
                (1, 2): 15,
                (0, 3): 3
            }
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 16

        # Test H^1_0 conforming
        expected_local_to_global_map = [
            {
                (0, 0): 0,
                (1, 0): 1,
                (2, 0): 2,
                (0, 1): 3,
                (1, 1): 4,
                (0, 2): 5
            },
            {
                (0, 0): 0,
                (1, 0): 6,
                (2, 0): 7,
                (3, 0): 8,
                (0, 1): 1,
                (1, 1): 9,
                (2, 1): 10,
                (0, 2): 2,
                (1, 2): 11
            }
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r, [[1, 2]])
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 12

        # Test H^1 conforming with boundary DOFs enumerated last
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
                (3, 0): 12,
                (0, 1): 3,
                (1, 1): 4,
                (2, 1): 13,
                (0, 2): 5,
                (1, 2): 14,
                (0, 3): 15
            },
            {
                (0, 0): 0,
                (1, 0): 6,
                (2, 0): 7,
                (3, 0): 8,
                (0, 1): 1,
                (1, 1): 9,
                (2, 1): 10,
                (0, 2): 2,
                (1, 2): 11,
                (0, 3): 12
            }
        ]
        local_to_global_map, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, [[1, 2]],
                                                                                        keep_boundary_dofs_last=True)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 16

        # Test H^1 conforming with boundary DOFs enumerated last, with an internal "boundary" edge
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
                (0, 0): 12,
                (1, 0): 6,
                (2, 0): 7,
                (3, 0): 8,
                (0, 1): 13,
                (1, 1): 9,
                (2, 1): 10,
                (0, 2): 14,
                (1, 2): 11,
                (0, 3): 15
            }
        ]
        local_to_global_map, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, [[0, 1]],
                                                                                        keep_boundary_dofs_last=True)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 16

    @staticmethod
    def test_3d():
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
                (0, 0, 0): 34,
                (1, 0, 0): 32,
                (2, 0, 0): 27,
                (3, 0, 0): 18,
                (4, 0, 0): 4,
                (0, 1, 0): 33,
                (1, 1, 0): 29,
                (2, 1, 0): 21,
                (3, 1, 0): 8,
                (0, 2, 0): 30,
                (1, 2, 0): 23,
                (2, 2, 0): 11,
                (0, 3, 0): 24,
                (1, 3, 0): 13,
                (0, 4, 0): 14,
                (0, 0, 1): 35,
                (1, 0, 1): 36,
                (2, 0, 1): 37,
                (3, 0, 1): 38,
                (0, 1, 1): 39,
                (1, 1, 1): 40,
                (2, 1, 1): 41,
                (0, 2, 1): 42,
                (1, 2, 1): 43,
                (0, 3, 1): 44,
                (0, 0, 2): 45,
                (1, 0, 2): 46,
                (2, 0, 2): 47,
                (0, 1, 2): 48,
                (1, 1, 2): 49,
                (0, 2, 2): 50,
                (0, 0, 3): 51,
                (1, 0, 3): 52,
                (0, 1, 3): 53,
                (0, 0, 4): 54
            },
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r)
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 55

        expected_local_to_global_map = [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (2, 0, 0): 2,
                (3, 0, 0): 3,
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
                (0, 1, 1): 17,
                (1, 1, 1): 18,
                (2, 1, 1): 19,
                (0, 2, 1): 20,
                (1, 2, 1): 21,
                (0, 3, 1): 22,
                (0, 0, 2): 23,
                (1, 0, 2): 24,
                (0, 1, 2): 25,
                (1, 1, 2): 26,
                (0, 2, 2): 27,
                (0, 0, 3): 28,
                (0, 1, 3): 29,
            },
            {
                (0, 1, 0): 29,
                (1, 1, 0): 26,
                (2, 1, 0): 19,
                (3, 1, 0): 7,
                (0, 2, 0): 27,
                (1, 2, 0): 21,
                (2, 2, 0): 10,
                (0, 3, 0): 22,
                (1, 3, 0): 12,
                (0, 4, 0): 13,
                (0, 1, 1): 30,
                (1, 1, 1): 31,
                (2, 1, 1): 32,
                (0, 2, 1): 33,
                (1, 2, 1): 34,
                (0, 3, 1): 35,
                (0, 1, 2): 36,
                (1, 1, 2): 37,
                (0, 2, 2): 38,
                (0, 1, 3): 39
            },
        ]
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r, [[3, 4, 1]])
        for i in range(len(expected_local_to_global_map)):
            for mi, expected_value in expected_local_to_global_map[i].items():
                assert local_to_global_map(i, mi) == expected_value
        assert num_dofs == 40

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
                ((0,), 0): 0,
                ((0,), 1): 1,
                ((1,), 0): 6,
                ((1,), 1): 7,
                ((2,), 0): 8,
                ((2,), 1): 9
            },
            {
                ((0,), 0): 0,
                ((0,), 1): 1,
                ((1,), 0): 10,
                ((1,), 1): 11,
                ((2,), 0): 12,
                ((2,), 1): 13
            }
        ]
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n)
        for j in range(len(triangles)):
            for mik, expected_value in expected_local_to_global_map[j].items():
                mi, k = mik
                assert local_to_global_map(j, mi, k) == expected_value
        assert num_dofs == 14

    @staticmethod
    def test_1d2d_sequential():
        # 1D domain with values in R^2 with sequential ordering of basis functions

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
                ((0,), 1): 7,
                ((1,), 0): 1,
                ((1,), 1): 8,
                ((2,), 0): 2,
                ((2,), 1): 9
            },
            {
                ((0,), 0): 0,
                ((0,), 1): 7,
                ((1,), 0): 3,
                ((1,), 1): 10,
                ((2,), 0): 4,
                ((2,), 1): 11
            },
            {
                ((0,), 0): 0,
                ((0,), 1): 7,
                ((1,), 0): 5,
                ((1,), 1): 12,
                ((2,), 0): 6,
                ((2,), 1): 13
            }
        ]
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n,
                                                                                   ordering="sequential")
        for j in range(len(triangles)):
            for mik, expected_value in expected_local_to_global_map[j].items():
                mi, k = mik
                assert local_to_global_map(j, mi, k) == expected_value
        assert num_dofs == 14

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
                ((0, 0), 0): 0,
                ((0, 0), 1): 1,
                ((1, 0), 0): 20,
                ((1, 0), 1): 21,
                ((2, 0), 0): 22,
                ((2, 0), 1): 23,
                ((3, 0), 0): 24,
                ((3, 0), 1): 25,
                ((0, 1), 0): 2,
                ((0, 1), 1): 3,
                ((1, 1), 0): 26,
                ((1, 1), 1): 27,
                ((2, 1), 0): 28,
                ((2, 1), 1): 29,
                ((0, 2), 0): 4,
                ((0, 2), 1): 5,
                ((1, 2), 0): 30,
                ((1, 2), 1): 31,
                ((0, 3), 0): 6,
                ((0, 3), 1): 7
            }
        ]
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n)
        for j in range(len(triangles)):
            for mik, expected_value in expected_local_to_global_map[j].items():
                mi, k = mik
                assert local_to_global_map(j, mi, k) == expected_value
        assert num_dofs == 32

        expected_local_to_global_map = [
            {
                ((0, 0), 0): 0,
                ((0, 0), 1): 1,
                ((1, 0), 0): 2,
                ((1, 0), 1): 3,
                ((2, 0), 0): 4,
                ((2, 0), 1): 5,
                ((0, 1), 0): 6,
                ((0, 1), 1): 7,
                ((1, 1), 0): 8,
                ((1, 1), 1): 9,
                ((0, 2), 0): 10,
                ((0, 2), 1): 11,
            },
            {
                ((0, 0), 0): 0,
                ((0, 0), 1): 1,
                ((1, 0), 0): 12,
                ((1, 0), 1): 13,
                ((2, 0), 0): 14,
                ((2, 0), 1): 15,
                ((3, 0), 0): 16,
                ((3, 0), 1): 17,
                ((0, 1), 0): 2,
                ((0, 1), 1): 3,
                ((1, 1), 0): 18,
                ((1, 1), 1): 19,
                ((2, 1), 0): 20,
                ((2, 1), 1): 21,
                ((0, 2), 0): 4,
                ((0, 2), 1): 5,
                ((1, 2), 0): 22,
                ((1, 2), 1): 23,
            }
        ]
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n, [[1, 2]])
        for j in range(len(triangles)):
            for mik, expected_value in expected_local_to_global_map[j].items():
                mi, k = mik
                assert local_to_global_map(j, mi, k) == expected_value
        assert num_dofs == 24

    @staticmethod
    def test_2d2d_sequential():
        # 2D domain with values in R^2, with sequential ordering of basis functions

        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        n = 2
        r = 3
        expected_local_to_global_map = [
            {
                ((0, 0), 0): 0,
                ((0, 0), 1): 16,
                ((1, 0), 0): 1,
                ((1, 0), 1): 17,
                ((2, 0), 0): 2,
                ((2, 0), 1): 18,
                ((3, 0), 0): 3,
                ((3, 0), 1): 19,
                ((0, 1), 0): 4,
                ((0, 1), 1): 20,
                ((1, 1), 0): 5,
                ((1, 1), 1): 21,
                ((2, 1), 0): 6,
                ((2, 1), 1): 22,
                ((0, 2), 0): 7,
                ((0, 2), 1): 23,
                ((1, 2), 0): 8,
                ((1, 2), 1): 24,
                ((0, 3), 0): 9,
                ((0, 3), 1): 25
            },
            {
                ((0, 0), 0): 0,
                ((0, 0), 1): 16,
                ((1, 0), 0): 10,
                ((1, 0), 1): 26,
                ((2, 0), 0): 11,
                ((2, 0), 1): 27,
                ((3, 0), 0): 12,
                ((3, 0), 1): 28,
                ((0, 1), 0): 1,
                ((0, 1), 1): 17,
                ((1, 1), 0): 13,
                ((1, 1), 1): 29,
                ((2, 1), 0): 14,
                ((2, 1), 1): 30,
                ((0, 2), 0): 2,
                ((0, 2), 1): 18,
                ((1, 2), 0): 15,
                ((1, 2), 1): 31,
                ((0, 3), 0): 3,
                ((0, 3), 1): 19
            }
        ]
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n,
                                                                                   ordering="sequential")
        for j in range(len(triangles)):
            for mik, expected_value in expected_local_to_global_map[j].items():
                mi, k = mik
                assert local_to_global_map(j, mi, k) == expected_value
        assert num_dofs == 32

        expected_local_to_global_map = [
            {
                ((0, 0), 0): 0,
                ((0, 0), 1): 12,
                ((1, 0), 0): 1,
                ((1, 0), 1): 13,
                ((2, 0), 0): 2,
                ((2, 0), 1): 14,
                ((0, 1), 0): 3,
                ((0, 1), 1): 15,
                ((1, 1), 0): 4,
                ((1, 1), 1): 16,
                ((0, 2), 0): 5,
                ((0, 2), 1): 17,
            },
            {
                ((0, 0), 0): 0,
                ((0, 0), 1): 12,
                ((1, 0), 0): 6,
                ((1, 0), 1): 18,
                ((2, 0), 0): 7,
                ((2, 0), 1): 19,
                ((3, 0), 0): 8,
                ((3, 0), 1): 20,
                ((0, 1), 0): 1,
                ((0, 1), 1): 13,
                ((1, 1), 0): 9,
                ((1, 1), 1): 21,
                ((2, 1), 0): 10,
                ((2, 1), 1): 22,
                ((0, 2), 0): 2,
                ((0, 2), 1): 14,
                ((1, 2), 0): 11,
                ((1, 2), 1): 23,
            }
        ]
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n, [[1, 2]],
                                                                                   ordering="sequential")
        for j in range(len(triangles)):
            for mik, expected_value in expected_local_to_global_map[j].items():
                mi, k = mik
                assert local_to_global_map(j, mi, k) == expected_value
        assert num_dofs == 24


class TestLocalToGlobalPreimageMap:
    @staticmethod
    def test_1d():
        # Test H^1 conforming
        lines = [
            [0, 1],
            [0, 2],
            [0, 3]
        ]
        r = 2
        local_to_global_map, num_dofs = generate_local_to_global_map(lines, r)
        preimage_map = generate_local_to_global_preimage_map(local_to_global_map, len(lines), num_dofs, r, 1)
        for i in range(num_dofs):
            for j, nu in preimage_map({i}):
                assert local_to_global_map(j, nu) == i

        # Test H^1_0 conforming
        local_to_global_map, num_dofs = generate_local_to_global_map(lines, r, [[0, 1]])
        preimage_map = generate_local_to_global_preimage_map(local_to_global_map, len(lines), num_dofs, r, 1)
        for i in range(num_dofs):
            for j, nu in preimage_map({i}):
                assert local_to_global_map(j, nu) == i

        # Test H^1 conforming with boundary DOFs enumerated last
        r = 2
        local_to_global_map, num_dofs, num_interior_dofs = generate_local_to_global_map(lines, r, [[0]],
                                                                                        keep_boundary_dofs_last=True)
        preimage_map = generate_local_to_global_preimage_map(local_to_global_map, len(lines), num_dofs, r, 1)
        for i in range(num_dofs):
            for j, nu in preimage_map({i}):
                assert local_to_global_map(j, nu) == i

    @staticmethod
    def test_2d():
        # Test H^1 conforming
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r)
        preimage_map = generate_local_to_global_preimage_map(local_to_global_map, len(triangles), num_dofs, r, 2)
        for i in range(num_dofs):
            for j, nu in preimage_map({i}):
                assert local_to_global_map(j, nu) == i

        # Test H^1_0 conforming
        local_to_global_map, num_dofs = generate_local_to_global_map(triangles, r, [[1, 2]])
        preimage_map = generate_local_to_global_preimage_map(local_to_global_map, len(triangles), num_dofs, r, 2)
        for i in range(num_dofs):
            for j, nu in preimage_map({i}):
                assert local_to_global_map(j, nu) == i

        # Test H^1 conforming with boundary DOFs enumerated last
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        local_to_global_map, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, [[1, 2]],
                                                                                        keep_boundary_dofs_last=True)
        preimage_map = generate_local_to_global_preimage_map(local_to_global_map, len(triangles), num_dofs, r, 2)
        for i in range(num_dofs):
            for j, nu in preimage_map({i}):
                assert local_to_global_map(j, nu) == i

        # Test H^1 conforming with boundary DOFs enumerated last, with an internal "boundary" edge
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        local_to_global_map, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, [[0, 1]],
                                                                                        keep_boundary_dofs_last=True)
        preimage_map = generate_local_to_global_preimage_map(local_to_global_map, len(triangles), num_dofs, r, 2)
        for i in range(num_dofs):
            for j, nu in preimage_map({i}):
                assert local_to_global_map(j, nu) == i

    @staticmethod
    def test_3d():
        tets = [
            [0, 1, 2, 3],
            [3, 1, 2, 4]
        ]
        r = 4
        local_to_global_map, num_dofs = generate_local_to_global_map(tets, r)
        preimage_map = generate_local_to_global_preimage_map(local_to_global_map, len(tets), num_dofs, r, 3)
        for i in range(num_dofs):
            for j, nu in preimage_map({i}):
                assert local_to_global_map(j, nu) == i

        local_to_global_map, num_dofs = generate_local_to_global_map(tets, r, [[3, 4, 1]])
        preimage_map = generate_local_to_global_preimage_map(local_to_global_map, len(tets), num_dofs, r, 3)
        for i in range(num_dofs):
            for j, nu in preimage_map({i}):
                assert local_to_global_map(j, nu) == i

    @staticmethod
    def test_1d2d():
        # Vector valued basis functions

        # Test H^1 conforming
        lines = [
            [0, 1],
            [0, 2],
            [0, 3]
        ]
        r = 2
        n = 2
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(lines, r, n)
        preimage_map = generate_vector_valued_local_to_global_preimage_map(local_to_global_map, len(lines), num_dofs, r,
                                                                           1, n)
        for i in range(num_dofs):
            for j, nu, k in preimage_map({i}):
                assert local_to_global_map(j, nu, k) == i

        # Test H^1_0 conforming
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(lines, r, n, [[0, 1]])
        preimage_map = generate_vector_valued_local_to_global_preimage_map(local_to_global_map, len(lines), num_dofs, r,
                                                                           1, n)
        for i in range(num_dofs):
            for j, nu, k in preimage_map({i}):
                assert local_to_global_map(j, nu, k) == i

        # Test H^1 conforming with boundary DOFs enumerated last
        r = 2
        local_to_global_map, num_dofs, num_interior_dofs =\
            generate_vector_valued_local_to_global_map(lines, r, n, [[0]], keep_boundary_dofs_last=True)
        preimage_map = generate_vector_valued_local_to_global_preimage_map(local_to_global_map, len(lines), num_dofs, r,
                                                                           1, n)
        for i in range(num_dofs):
            for j, nu, k in preimage_map({i}):
                assert local_to_global_map(j, nu, k) == i

    @staticmethod
    def test_2d2d():
        # Vector valued basis functions

        # Test H^1 conforming
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        n = 2
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n)
        preimage_map = generate_vector_valued_local_to_global_preimage_map(local_to_global_map, len(triangles),
                                                                           num_dofs, r, 2, n)
        for i in range(num_dofs):
            for j, nu, k in preimage_map({i}):
                assert local_to_global_map(j, nu, k) == i

        # Test H^1_0 conforming
        local_to_global_map, num_dofs = generate_vector_valued_local_to_global_map(triangles, r, n, [[1, 2]])
        preimage_map = generate_vector_valued_local_to_global_preimage_map(local_to_global_map, len(triangles),
                                                                           num_dofs, r, 2, n)
        for i in range(num_dofs):
            for j, nu, k in preimage_map({i}):
                assert local_to_global_map(j, nu, k) == i

        # Test H^1 conforming with boundary DOFs enumerated last
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        local_to_global_map, num_dofs, num_interior_dofs =\
            generate_vector_valued_local_to_global_map(triangles, r, n, [[1, 2]], keep_boundary_dofs_last=True)
        preimage_map = generate_vector_valued_local_to_global_preimage_map(local_to_global_map, len(triangles),
                                                                           num_dofs, r, 2, n)
        for i in range(num_dofs):
            for j, nu, k in preimage_map({i}):
                assert local_to_global_map(j, nu, k) == i

        # Test H^1 conforming with boundary DOFs enumerated last, with an internal "boundary" edge
        triangles = [
            [0, 1, 2],
            [0, 3, 1]
        ]
        r = 3
        local_to_global_map, num_dofs, num_interior_dofs =\
            generate_vector_valued_local_to_global_map(triangles, r, n, [[0, 1]], keep_boundary_dofs_last=True)
        preimage_map = generate_vector_valued_local_to_global_preimage_map(local_to_global_map, len(triangles),
                                                                           num_dofs, r, 2, n)
        for i in range(num_dofs):
            for j, nu, k in preimage_map({i}):
                assert local_to_global_map(j, nu, k) == i


if __name__ == '__main__':
    pytest.main(sys.argv)
