#!/bin/bash

# Generate library documentation from docstrings using Sphinx.
# Dependencies: 
# * Sphinx
# * Git

# Abort on any error
set -e
# Echo commands
set -x

# Update version in conf.py
python3 set_version_from_git.py

# Install requirements
pip3 install -r requirements.txt

# Run script which shortens module and package names in generated API files
python3 shorten_module_and_package_names.py

# Generate documentation
sphinx-build -W -d _build/doctrees . -b html _build/html

# Copy generated documentation to 'generated_documentation' folder
mkdir -p generated_documentation
cd generated_documentation
VERSION=$(git describe --always)
OUTPUT_DIR="polynomials_on_simplices_documentation_"$VERSION
mkdir "$OUTPUT_DIR"
cp -r ../_build/html "$OUTPUT_DIR"/html
cd ..
