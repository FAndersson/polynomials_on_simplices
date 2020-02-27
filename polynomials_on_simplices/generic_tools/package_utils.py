"""Extensions to the standard library pkgutil module."""

import os
from pathlib import Path
import pkgutil
import sys


def subpackage_paths(package_path, absolute_path=True):
    """
    Yield the path of each subpackage in a package.

    :param str package_path: Path to the package we want to investigate.
    :param bool absolute_path: Whether or not yielded paths should be absolute or relative to the package path.
    :return: Path to each subpackage in the package.
    :rtype: Iterator[str]
    """
    sys.path.append(package_path)
    for finder, name, is_pkg in pkgutil.walk_packages([package_path]):
        if is_pkg:
            path = os.path.join(package_path, name.replace('.', os.path.sep))
            if absolute_path:
                yield path
            else:
                yield os.path.relpath(path, package_path)


def submodules(package_path):
    """
    Yield each submodule in a package.

    :param str package_path: Path to the package we want to investigate.
    :return: Relative import path to each submodule in the package.
    :rtype: Iterator[str]
    """
    for submodule_path in submodule_paths(package_path, absolute_path=False):
        yield submodule_path.replace(os.path.sep, '.').replace('.py', '')


def submodule_paths(package_path, absolute_path=True):
    """
    Yield the path of each submodule in a package.

    :param str package_path: Path to the package we want to investigate.
    :param bool absolute_path: Whether or not yielded paths should be absolute or relative to the package path.
    :return: Path to each submodule in the package.
    :rtype: Iterator[str]
    """
    sys.path.append(package_path)
    for finder, name, is_pkg in pkgutil.walk_packages([package_path]):
        if not is_pkg:
            path = os.path.join(package_path, name.replace('.', os.path.sep) + ".py")
            if absolute_path:
                yield path
            else:
                yield os.path.relpath(path, package_path)


def source_file_paths(package_path, absolute_path=True, exclude_test_files=False, include_hidden_files=False,
                      include_private_files=False):
    """
    Yield the path of each Python source file (files ending with '.py') in a package.

    :param package_path: Path to the package we want to investigate.
    :param str package_path: Path to the package we want to investigate.
    :param bool absolute_path: Whether or not yielded paths should be absolute or relative to the package path.
    :param bool exclude_test_files: Whether or not to exclude unit test files. A file is considered a test file if
        it ends with '_test.py' or '_test_interactive.py' and resides in a folder called 'test'.
    :param bool include_hidden_files: Whether or not to include hidden source files in the package (files starting
        with '.' or residing in a directory starting with '.').
    :param bool include_private_files: Whether or not to include private source files in the package (files starting
        with '_').
    :return: Path to each source file in the package.
    :rtype: Iterator[str]
    """
    for dirpath, dirnames, filesnames in os.walk(package_path):
        if not include_hidden_files:
            filesnames = [f for f in filesnames if not f[0] == '.']
            dirnames[:] = [d for d in dirnames if not d[0] == '.']
        if not include_private_files:
            filesnames = [f for f in filesnames if not f[0] == '_' or f[0:2] == '__']
        for filename in filesnames:
            if os.path.splitext(filename)[1] == ".py":
                source_file = os.path.join(dirpath, filename)
                if exclude_test_files and _is_test_file(source_file):
                    continue
                if absolute_path:
                    yield source_file
                else:
                    yield os.path.relpath(source_file, package_path)


def _is_test_file(filename):
    if (filename.endswith("_test.py")
            and Path(filename).parent.parts[-1] == "test"):
        return True
    if (filename.endswith("_test_interactive.py")
            and Path(filename).parent.parts[-1] == "test"):
        return True
    return False
