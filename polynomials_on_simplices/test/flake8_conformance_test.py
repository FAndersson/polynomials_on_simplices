import os
import subprocess
import sys

import pytest


def test_conformance():
    """Test that all conformance checks from Flake8 passes for all source files in the repository."""
    file_path = os.path.dirname(os.path.abspath(__file__))
    library_path = os.path.abspath(os.path.join(file_path, ".."))
    ret = subprocess.call(["flake8", library_path])
    assert ret == 0


if __name__ == '__main__':
    pytest.main(sys.argv)
