"""Sort imports in all library modules using isort."""

import os
import sys

from isort import SortImports
import pytest

from polynomials_on_simplices.generic_tools.package_utils import source_file_paths


def tests_imports_sorted():
    file_path = os.path.dirname(os.path.abspath(__file__))
    library_path = os.path.normpath(os.path.join(file_path, os.pardir))
    files_to_check = source_file_paths(library_path)
    for file in files_to_check:
        assert not SortImports(file, check=True).incorrectly_sorted


if __name__ == '__main__':
    pytest.main(sys.argv)
