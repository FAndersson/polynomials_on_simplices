"""Sort imports in all library modules using isort."""

import os

from isort import SortImports

from polynomials_on_simplices.generic_tools.package_utils import source_file_paths


def sort_imports():
    """
    Sort imports in all library modules using isort. Sorting is controlled by the settings in the .isort.cfg file.
    """
    library_path = os.path.dirname(os.path.abspath(__file__))
    files_to_sort_in = source_file_paths(library_path)
    for file in files_to_sort_in:
        SortImports(file)


if __name__ == '__main__':
    sort_imports()
