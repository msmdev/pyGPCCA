#!/usr/bin/env bash

set -euo pipefail

function install_petsc_macos {
    curl -O "https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-$PC_VERSION.tar.gz"
    tar -xzf "petsc-lite-$PC_VERSION.tar.gz"
    pushd "petsc-$PC_VERSION"

    ./configure --with-cc=mpicc --with-cxx=mpicxx --with-debugging=0 --with-mpi=1
    make all
    make check
    # make install  # only to move the files into the appropriate location

    popd
}

function install_slepc_macos {
    curl -O "https://slepc.upv.es/download/distrib/slepc-$SC_VERSION.tar.gz"
    tar -xzf "slepc-$SC_VERSION.tar.gz"
    pushd "slepc-$SC_VERSION"

    ./configure --with-arpack-dir=/usr/local/Cellar/arpack
    make all
    make check
    # make install  # only to move into the appropriate location

    popd
}


if [[ "$RUNNER_OS" == "Linux" ]]; then
    echo "Installing PETSc/SLEPc dependencies for Linux"
    sudo apt-get update -y
    sudo apt-get install gcc gfortran libopenmpi-dev libblas-dev liblapack-dev petsc-dev slepc-dev -y
elif [[ "$RUNNER_OS" == "macOS" ]]; then
    echo "Installing PETSc/SLEPc with SLEPc dir: '$SLEPC_DIR', PETSc dir: '$PETSC_DIR', PETSc arch: '$PETSC_ARCH'"
    pushd "$HOME"

    brew install gcc open-mpi openblas lapack arpack
    install_petsc_macos
    install_slepc_macos

    popd

    # this seems to be only necessary on the CI
    echo "Symlinking numpy"
    python -m pip install --upgrade pip
    pip install numpy

    NUMPY_INCLUDE="$(python -c 'import numpy; print(numpy.get_include())')"
    ln -sfv "$NUMPY_INCLUDE/numpy" "$PETSC_DIR/$PETSC_ARCH/include"
else
    echo "Invalid OS for PETSc/SLEPc dependencies: $OS"
    exit 42
fi
