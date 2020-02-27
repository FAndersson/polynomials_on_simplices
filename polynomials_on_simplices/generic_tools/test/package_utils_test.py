import os
import sys

import pytest

from polynomials_on_simplices.generic_tools.package_utils import (
    source_file_paths, submodule_paths, submodules, subpackage_paths)


def test_source_files():
    current_dir = os.path.dirname(__file__)
    test_dir = os.path.join(current_dir, "test_package")
    paths = set(source_file_paths(test_dir))
    path_basenames = {os.path.basename(path) for path in paths}
    expected_path_basenames = {'__init__.py', 'a.py', 'b.py', 'c.py', 'd.py', 'e.py', 'f.py', 'a_test.py',
                               'not_a_test.py'}
    assert path_basenames == expected_path_basenames


def test_source_files_relative_path():
    current_dir = os.path.dirname(__file__)
    test_dir = os.path.join(current_dir, "test_package")
    paths = set(source_file_paths(test_dir, absolute_path=False))
    expected_paths = {
        os.path.join('test_subdir', 'd.py'),
        os.path.join('test_subdir', 'not_a_test.py'),
        os.path.join('test_subpackage', 'test', 'a_test.py'),
        os.path.join('test_subpackage', 'test_nested_subdir', 'f.py'),
        os.path.join('test_subpackage', 'test_nested_subpackage', '__init__.py'),
        os.path.join('test_subpackage', 'test_nested_subpackage', 'e.py'),
        os.path.join('test_subpackage', '__init__.py'),
        os.path.join('test_subpackage', 'c.py'),
        '__init__.py',
        'a.py',
        'b.py'
    }
    assert expected_paths == paths


def test_source_file_paths_exclude_test_files():
    current_dir = os.path.dirname(__file__)
    test_dir = os.path.join(current_dir, "test_package")
    paths = set(source_file_paths(test_dir, exclude_test_files=True))
    path_basenames = {os.path.basename(path) for path in paths}
    expected_path_basenames = {'__init__.py', 'a.py', 'b.py', 'c.py', 'd.py', 'e.py', 'f.py', 'not_a_test.py'}
    assert path_basenames == expected_path_basenames


def test_source_file_paths_exclude_test_files_include_hidden_files():
    current_dir = os.path.dirname(__file__)
    test_dir = os.path.join(current_dir, "test_package")
    paths = set(source_file_paths(test_dir, exclude_test_files=True, include_hidden_files=True))
    path_basenames = {os.path.basename(path) for path in paths}
    expected_path_basenames = {'.w.py', '__init__.py', 'a.py', 'b.py', 'c.py', 'd.py', 'e.py', 'f.py', 'not_a_test.py'}
    assert path_basenames == expected_path_basenames


def test_source_files_relative_path_exclude_test_files():
    current_dir = os.path.dirname(__file__)
    test_dir = os.path.join(current_dir, "test_package")
    paths = set(source_file_paths(test_dir, absolute_path=False, exclude_test_files=True))
    expected_paths = {
        os.path.join('test_subdir', 'd.py'),
        os.path.join('test_subdir', 'not_a_test.py'),
        os.path.join('test_subpackage', 'test_nested_subdir', 'f.py'),
        os.path.join('test_subpackage', 'test_nested_subpackage', '__init__.py'),
        os.path.join('test_subpackage', 'test_nested_subpackage', 'e.py'),
        os.path.join('test_subpackage', '__init__.py'),
        os.path.join('test_subpackage', 'c.py'),
        '__init__.py',
        'a.py',
        'b.py'
    }
    assert expected_paths == paths


def test_submodules():
    current_dir = os.path.dirname(__file__)
    test_dir = os.path.join(current_dir, "test_package")

    sms = set(submodules(test_dir))
    expected_sms = {
        'a',
        'b',
        'test_subpackage.c',
        'test_subpackage.test_nested_subpackage.e'
    }
    assert sms == expected_sms


def test_submodule_paths():
    current_dir = os.path.dirname(__file__)
    test_dir = os.path.join(current_dir, "test_package")

    paths = set(submodule_paths(test_dir))
    path_basenames = {os.path.basename(path) for path in paths}
    expected_path_basenames = {'a.py', 'b.py', 'c.py', 'e.py'}
    assert path_basenames == expected_path_basenames


def test_submodule_paths_relative_path():
    current_dir = os.path.dirname(__file__)
    test_dir = os.path.join(current_dir, "test_package")

    paths = set(submodule_paths(test_dir, absolute_path=False))
    expected_paths = {
        os.path.join('test_subpackage', 'test_nested_subpackage', 'e.py'),
        os.path.join('test_subpackage', 'c.py'),
        'a.py',
        'b.py'
    }
    assert expected_paths == paths


def test_subpackage_paths():
    current_dir = os.path.dirname(__file__)
    test_dir = os.path.join(current_dir, "test_package")

    paths = set(subpackage_paths(test_dir))
    path_basenames = {os.path.basename(path) for path in paths}
    expected_path_basenames = {'test_subpackage', 'test_nested_subpackage'}
    assert path_basenames == expected_path_basenames


def test_subpackage_paths_relative_path():
    current_dir = os.path.dirname(__file__)
    test_dir = os.path.join(current_dir, "test_package")

    paths = set(subpackage_paths(test_dir, absolute_path=False))
    expected_paths = {
        os.path.join('test_subpackage', 'test_nested_subpackage'),
        'test_subpackage'
    }
    assert expected_paths == paths


if __name__ == "__main__":
    pytest.main(sys.argv)
