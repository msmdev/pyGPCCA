#!/usr/bin/env bash

set -euo pipefail

function install_petsc_macos {
    curl -O "https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-$PC_VERSION.tar.gz"
    tar -xzf "petsc-lite-$PC_VERSION.tar.gz"
    pushd "petsc-$PC_VERSION"

    ./configure --with-cc=mpicc --with-cxx=mpicxx --with-debugging=0 --with-mpi=1
    make all
    make check

    popd
}

function install_slepc_macos {
    curl -O "https://slepc.upv.es/download/distrib/slepc-$SC_VERSION.tar.gz"
    tar -xzf "slepc-$SC_VERSION.tar.gz"
    pushd "slepc-$SC_VERSION"

    ./configure --with-arpack-dir=/usr/local/Cellar/arpack
    make all
    make check

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
else
    echo "Invalid OS for PETSc/SLEPc dependencies: $OS"
    exit 42
fi
