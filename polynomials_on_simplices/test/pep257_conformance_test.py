import logging
import os
import unittest

import pydocstyle
from pydocstyle.utils import log

from polynomials_on_simplices.generic_tools.package_utils import submodule_paths


class TestDocStyle(unittest.TestCase):
    excluded_source_files = [
        'conf.py',
        'strain.py',
    ]

    def test_conformance(self):
        """Test that we conform to PEP-257."""
        file_path = os.path.dirname(os.path.abspath(__file__))
        library_path = os.path.normpath(os.path.join(file_path, os.pardir))
        files_to_check = submodule_paths(library_path)
        files_to_check = filter(lambda path: not any(path.endswith(ex) for ex in TestDocStyle.excluded_source_files),
                                files_to_check)
        # Ignores:
        # D104: Missing docstring in public package (ignore missing docstrings in __init__.py files)
        # D105: Missing docstring in magic method (most magic methods are self-explanatory, and don't need any
        # documentation)
        # D107: Missing docstring in __init__ (no need to document the __init__method if the class already has a
        # docstring and the __init__ method takes no arguments)
        # D200: One-line docstring should fit on one line with quotes (I find this less readable, compared to having
        # the start and end quotation marks on separate lines)
        # D203: 1 blank line required before class docstring (not part of PEP-257)
        # D205: 1 blank line required between summary line and description (many functions have a description which is
        # just slightly longer than one line, and splitting that into a summary line and description part just seems
        # a bit overkill)
        # D212: Multi-line docstring summary should start at the first line (not part of PEP-257)
        # D213: Multi-line docstring summary should start at the second line (not part of PEP-257)
        # D400: First line should end with a period (for functions with a description just slightly longer than one
        # line this doesn't make sense)
        # D415: First line should end with a period, question mark, or exclamation point (see D400)
        ignore = ['D104', 'D105', 'D107', 'D200', 'D203', 'D205', 'D212', 'D213', 'D400', 'D415']
        errors = []
        log.setLevel(logging.WARNING)
        errors.extend(pydocstyle.check(files_to_check, ignore=ignore))
        for error in errors:
            if hasattr(error, 'code'):
                print(error)
        self.assertTrue(len(errors) == 0, "Found doc style errors or warnings.")


if __name__ == '__main__':
    unittest.main()
