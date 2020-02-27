#!/bin/bash

# Zip all folders in the generated_documentation directory

# Abort on any error
set -e
# Echo commands
set -x

cd generated_documentation
for dir in *; do zip -r "$dir".zip "$dir"; done
cd ..
