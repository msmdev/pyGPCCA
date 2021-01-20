|PyPI| |Conda| |CI| |Coverage|

pyGPCCA - Generalized Perron Cluster Cluster Analysis
+++++++++++++++++++++++++++++++++++++++++++++++++++++
Generalized Perron Cluster Cluster Analysis program to coarse-grain reversible and non-reversible Markov State Models.

Markov State Models (MSM) enable the identification and analysis of metastable states and related kinetics in a
very instructive manner. They are widely used, e.g. to model molecular or cellular kinetics.
Common state-of-the-art Markov state modeling methods and tools are very well suited to model reversible processes in
closed equilibrium systems. However, most are not well suited to deal with non-reversible or even non-autonomous
processes of non-equilibrium systems.
To overcome this limitation, the Generalized Robust Perron Cluster Cluster Analysis (G-PCCA) was developed.
The G-PCCA method implemented in the *pyGPCCA* program readily handles equilibrium as well as non-equilibrium data by
utilizing real Schur vectors instead of eigenvectors.
*pyGPCCA* enables the semiautomatic coarse-graining of transition matrices representing the dynamics of the system
under study. Utilizing *pyGPCCA*, metastable states as well as cyclic kinetics can be identified and modeled.

Installation
------------
We support multiple ways of installing *pyGPCCA*. If any problems arise during the installation,
please refer to `Installation troubleshooting`_.

Conda
=====
*pyGPCCA* is available as a `conda package <https://anaconda.org/conda-forge/pygpcca>`_ and can be installed as::

    conda install -c conda-forge pygpcca

This is the recommended way of installing, since this package also includes `PETSc`_/`SLEPc`_ libraries.
We use `PETSc`_/`SLEPc`_ internally to speed up the computation of the leading Schur vectors. These are optional
dependencies - if they're not present, we compute a full Schur decomposition instead and sort it using the method
introduced by `Brandts (2002)`_. Note that this scales cubically in sample number, making it essential to use
`PETSc`_/`SLEPc`_ for large sample numbers. `PETSc`_/`SLEPc`_ implement iterative methods to only compute
the leading Schur vectors, which is computationally much less expensive.

PyPI
====
In order to install *pyGPCCA* from `The Python Package Index <https://pypi.org/project/pygpcca/>`_, run::

    pip install pygpcca
    # or with libraries utilizing PETSc/SLEPc
    pip install pygpcca[slepc]

Development version
===================
If you want to use the development version of *pyGPCCA* from `GitHub <https://github.com/msmdev/pygpcca>`_, run::

    pip install git+https://github.com/msmdev/pygpcca

Usage
-----
Afterwards *pyGPCCA* can be imported in Python as::

  import pygpcca as gp

*pyGPCCA* can be used as outlined in the following:

- Initialize a GPCCA object with a transition matrix ``P``::

    gpcca = gp.GPCCA(P)

- Get a list of minChi values for numbers of macrostates ``m`` in an interval ``[2, 30]`` to determine an interval
  ``[m_min, m_max]`` of (nearly) optimal numbers of macrostates for clustering::

    minChi_list = gpcca.minChi(2, 30)

- Optimize the clustering for numbers of macrostates ``m`` in the previously determined interval ``[m_min, m_max]`` and
  find the optimal number of macrostates ``n_metastable`` in the given interval::

    gpcca.optimize({'m_min':2, 'm_max':10})

- Afterwards, the optimal number of macrostates ``n_metastable`` can be accessed via::

    gpcca.n_metastable

- The optimal coarse-grained matrix can be accessed via::

    gpcca.coarse_grained_transition_matrix

- The memberships are available via::

    gpcca.memberships

Installation troubleshooting
----------------------------
During the installation of ``petsc``, ``petsc4py``, ``slepc``, and ``slepc4py``, the following error(s) might appear::

    ERROR: Failed building wheel for <package name>

However, this should be fine if in the end, it also outputs::

    Successfully installed <package name>

To quickly verify that the packages have been installed, you can run::

    python3 -c "import petsc4py; import slepc4py; print(petsc4py.__version__, slepc4py.__version__)"

Debian-based systems
====================
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
=====
The most robust way is to follow the `PETSc installation guide`_ and the `SLEPc installation guide`_ or to take a look
at our continuous integration `steps <./.scripts/ci/install_dependencies.sh>`_ for macOS.

The installation steps can be roughly outlined as::

    # install dependencies
    brew install gcc open-mpi openblas lapack arpack

    # follow the PETSc installation steps
    # follow the SLEPc installation steps

    # install petsc4py
    pip install --user petsc4py
    # install slepc4py
    pip install --user petsc4py

.. |PyPI| image:: https://img.shields.io/pypi/v/pygpcca
    :target: https://pypi.org/project/pygpcca
    :alt: PyPI

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/pygpcca
    :target: https://anaconda.org/conda-forge/pygpcca
    :alt: Conda

.. |CI| image:: https://img.shields.io/github/workflow/status/msmdev/pygpcca/CI/main
    :target: https://github.com/msmdev/pygpcca/actions
    :alt: CI

.. |Coverage| image:: https://img.shields.io/codecov/c/github/msmdev/pygpcca/main
    :target: https://codecov.io/gh/msmdev/pygpcca
    :alt: Coverage

.. _`PETSc`: https://www.mcs.anl.gov/petsc/
.. _`SLEPc`: https://slepc.upv.es/
.. _`Brandts (2002)`: https://doi.org/10.1002/nla.274
.. _`PETSc installation guide`: https://www.mcs.anl.gov/petsc/documentation/installation.html
.. _`SLEPc installation guide`: https://slepc.upv.es/documentation/instal.htm
