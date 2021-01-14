#!/usr/bin/env bash

set -euo pipefail

pip install numpy

NUMPY_INCLUDE="$(python -c 'import numpy; print(numpy.get_include())')"
echo "Numpy header files location: $NUMPY_INCLUDE"

ln -sfv "$NUMPY_INCLUDE/numpy" "$PETSC_DIR/$PETSC_ARCH/include"
