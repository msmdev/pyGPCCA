Installation
============
*pyGPCCA* requires Python >= 3.6 to run. If any problems arise, please consult the
`Troubleshooting`_ section.

Methods
-------

Conda
+++++
*pyGPCCA* is available as a `conda package <https://anaconda.org/conda-forge/pygpcca>`_ and can be installed as::

    conda install -c conda-forge pygpcca

This is the recommended way of installing, since this package also includes `PETSc`_/`SLEPc`_ libraries.
We use `PETSc`_/`SLEPc`_ internally to speed up the computation of the leading Schur vectors. These are optional
dependencies - if they're not present, we compute a full Schur decomposition instead and sort it using the method
introduced by `Brandts (2002)`_. Note that this scales cubically in sample number, making it essential to use
`PETSc`_/`SLEPc`_ for large sample numbers. `PETSc`_/`SLEPc`_ implement iterative methods to only compute
the leading Schur vectors, which is computationally much less expensive.

PyPI
++++
In order to install *pyGPCCA* from `The Python Package Index <https://pypi.org/project/pygpcca/>`_, run::

    pip install pygpcca
    # or with libraries utilizing PETSc/SLEPc
    pip install pygpcca[slepc]

Development version
+++++++++++++++++++
If you want to use the development version of *pyGPCCA* from `GitHub <https://github.com/msmdev/pygpcca>`_, run::

    pip install git+https://github.com/msmdev/pygpcca

Troubleshooting
---------------
During the installation of ``petsc``, ``petsc4py``, ``slepc``, and ``slepc4py``, the following error(s) might appear::

    ERROR: Failed building wheel for <package name>

However, this should be fine if in the end, it also outputs::

    Successfully installed <package name>

To quickly verify that the packages have been installed, you can run::

    python3 -c "import petsc4py; import slepc4py; print(petsc4py.__version__, slepc4py.__version__)"

Debian-based systems
++++++++++++++++++++
Below are an alternative steps for installing `PETSc`_/`SLEPc`_, in case any problems arise, especially when installing
from `PyPI`_::

    # install dependencies
    sudo apt-get update -y
    sudo apt-get install gcc gfortran libopenmpi-dev libblas-dev liblapack-dev petsc-dev slepc-dev -y

    # install a message passing interface for Python
    pip install --user mpi4py

    # install petsc and and petsc4py
    pip install --user petsc
    pip install --user petsc4py

    # install slepc and slepc4py
    pip install --user slepc
    pip install --user slepc4py

macOS
+++++
The most robust way is to follow the `PETSc installation guide`_ and the `SLEPc installation guide`_ or to take a look
at our continuous integration `steps <https://github.com/msmdev/pyGPCCA/blob/main/.scripts/ci/install_dependencies.sh>`_
for macOS.

The installation steps can be roughly outlined as::

    # install dependencies
    brew install gcc open-mpi openblas lapack arpack

    # follow the PETSc installation steps
    # follow the SLEPc installation steps

    # install petsc4py
    pip install --user petsc4py
    # install slepc4py
    pip install --user petsc4py

.. _`Brandts (2002)`: https://doi.org/10.1002/nla.274
.. _`PETSc`: https://www.mcs.anl.gov/petsc/
.. _`SLEPc`: https://slepc.upv.es/
.. _`PETSc installation guide`: https://www.mcs.anl.gov/petsc/documentation/installation.html
.. _`SLEPc installation guide`: https://slepc.upv.es/documentation/instal.htm
