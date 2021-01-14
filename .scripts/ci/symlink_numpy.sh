#!/usr/bin/env bash

set -euo pipefail

pip install numpy

NUMPY_INCLUDE="$(python -c 'import numpy; print(numpy.get_include())')"
LINKNAME="$PETSC_DIR/$PETSC_ARCH/include"

echo "Numpy header files location: $NUMPY_INCLUDE"

if [[ -L "$LINKNAME" ]]; then
    unlink "$LINKNAME"
fi

ln -s "$NUMPY_INCLUDE/numpy" "$LINKNAME"
