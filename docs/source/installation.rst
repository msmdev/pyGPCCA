Installation
============
:mod:`pygpcca` requires Python version >= 3.6 to run.

PyPI
----

You can install :mod:`pygpcca` by running::

    pip install pygpcca

Conda-forge
-----------
TODO.

Development version
-------------------
To install the development version of :mod:`pygpcca` from GitHub, run::

    pip install git+https://github.com/msmdev/pygpcca

.. _Installing PETSc and SLEPc:

Installing PETSc and SLEPc
--------------------------

TODO. On a Debian-like system, run::

    # update the package manager
    sudo apt-get update
    sudo apt-get upgrade

    # install a message passing interface
    sudo apt-get install libopenmpi-dev  # alt.: conda install -c conda-forge openmpi
    pip install --user mpi4py  # alt.: conda install -c anaconda mpi4py

    # install petsc and and petsc4py
    pip install --user petsc  # alt.: conda install -c conda-forge petsc
    pip install --user petsc4py  # alt.: conda install -c conda-forge petsc4py

    # install slepc and slepsc4py
    pip install --user slepc  # alt.: conda install -c conda-forge slepc
    pip install --user slepc4py  # alt.: conda install -c conda-forge slepc4py
