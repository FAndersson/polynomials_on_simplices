import sys

import pytest

from polynomials_on_simplices.generic_tools.str_utils import split_long_str, str_dot_product


def test_str_dot_product():
    a = ["0", "a_2"]
    b = ["b_1", "b_2"]
    res = str_dot_product(a, b)
    assert res == "a_2 b_2"

    a = ["0", "0"]
    b = ["b_1", "b_2"]
    res = str_dot_product(a, b)
    assert res == "0"

    a = ["a_1", "a_2"]
    b = ["b_1", "b_2"]
    res = str_dot_product(a, b)
    assert res == "a_1 b_1 + a_2 b_2"

    a = ["a_1", "a_2"]
    b = ["b_1", "-b_2"]
    res = str_dot_product(a, b)
    assert res == "a_1 b_1 - a_2 b_2"

    a = ["a_1", "a_2"]
    b = ["-b_1", "b_2"]
    res = str_dot_product(a, b)
    assert res == "-a_1 b_1 + a_2 b_2"

    a = ["-a_1", "a_2"]
    b = ["-b_1", "b_2"]
    res = str_dot_product(a, b)
    assert res == "a_1 b_1 + a_2 b_2"

    a = ["a_1", "-a_2"]
    b = ["b_1", "-b_2"]
    res = str_dot_product(a, b)
    assert res == "a_1 b_1 + a_2 b_2"

    a = ["a_1", "a_2"]
    b = ["b_1", "b_2"]
    res = str_dot_product(a, b, multiplication_character="*")
    assert res == "a_1 * b_1 + a_2 * b_2"


def test_split_long_str():
    str_to_split = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
    parts = split_long_str(str_to_split, sep=", ", long_str_threshold=6, sep_replacement_before=",")
    expected_parts = [
        "1, 2,",
        "3, 4,",
        "5, 6,",
        "7, 8,",
        "9,",
        "10"
    ]
    assert parts == expected_parts

    str_to_split = "1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10"
    parts = split_long_str(str_to_split, sep=" + ", long_str_threshold=8, sep_replacement_after="+ ")
    expected_parts = [
        "1 + 2",
        "+ 3 + 4",
        "+ 5 + 6",
        "+ 7 + 8",
        "+ 9",
        "+ 10"
    ]
    assert parts == expected_parts

    str_to_split = '[1, -25 / 3, 70 / 3, -80 / 3, 32 / 3, -25 / 3, 140 / 3, -80, 128 / 3, 70 / 3, -80, 64, -80 / 3,'\
                   + ' 128 / 3, 32 / 3]'
    parts = split_long_str(str_to_split, ", ", long_str_threshold=61, sep_replacement_before=",")
    expected_parts = [
        '[1, -25 / 3, 70 / 3, -80 / 3, 32 / 3, -25 / 3, 140 / 3, -80,',
        '128 / 3, 70 / 3, -80, 64, -80 / 3, 128 / 3, 32 / 3]'
    ]
    assert parts == expected_parts


if __name__ == '__main__':
    pytest.main(sys.argv)
