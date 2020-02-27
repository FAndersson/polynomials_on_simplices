"""
The purpose of this script is to make package and module names in generated documentation as short as possible.

By default, Sphinx uses the full path for module/package names, e.g.
polynomials_on_simplices.geometry.mesh.basic_meshes.triangle_meshes. This
script would replace that name with simply triangle_meshes. This makes the generated documentation easier to
read, without really losing any information, since the modules/packages are anyhow organized hierarchically.
E.g. the triangle_meshes module is found beneath the basic_meshes package which is found beneath the mesh package which
is found beneath geometry package.

See https://stackoverflow.com/questions/25276164/sphinx-apidoc-dont-print-full-path-to-packages-and-modules.
"""

import os


def shorten_module_name(module_filename):
    """Shorten the name of a module in a .rst file.

    E.g. replace something like 'a.b.c.xxx module' on the first line with 'xxx module'.
    """
    with open(module_filename) as file:
        file_content = file.read()
    first_line_end = file_content.find("\n")
    if first_line_end == -1:
        return
    module_name_end = file_content.find(" module", 0, first_line_end)
    if module_name_end == -1:
        return
    module_name_last_part = file_content[0:module_name_end].rfind('.')
    file_content = file_content[module_name_last_part + 1:]
    with open(module_filename, 'w') as file:
        file.write(file_content)


def shorten_package_name(module_filename):
    """Shorten the name of a package in a .rst file.

    E.g. replace something like 'a.b.c.xxx package' on the first line with 'xxx package'.
    """
    with open(module_filename) as file:
        file_content = file.read()
    first_line_end = file_content.find("\n")
    if first_line_end == -1:
        return
    module_name_end = file_content.find(" package", 0, first_line_end)
    if module_name_end == -1:
        return
    module_name_last_part = file_content[0:module_name_end].rfind('.')
    file_content = file_content[module_name_last_part + 1:]
    with open(module_filename, 'w') as file:
        file.write(file_content)


def shorten_all_rst_files(directory):
    """Run :func:`shorten_module_name` and :func:`shorten_package_name` on all .rst files in the given directory.
    """
    for file in os.listdir(directory):
        if file.endswith(".rst"):
            shorten_module_name(file)
            shorten_package_name(file)


if __name__ == "__main__":
    shorten_all_rst_files(os.path.dirname(os.path.realpath(__file__)))
