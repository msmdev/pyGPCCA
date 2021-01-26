|PyPI| |Conda| |Cite| |CI| |Docs| |Coverage| |License| |GitHubDownloads| |PyPIdownloads| |CondaDownloads|

.. |PyPI| image:: https://img.shields.io/pypi/v/pygpcca
    :target: https://pypi.org/project/pygpcca/
    :alt: PyPI

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/pygpcca
    :target: https://anaconda.org/conda-forge/pygpcca
    :alt: Conda

.. |Cite| image:: https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.8b00079-blue
    :target: https://doi.org/10.1021/acs.jctc.8b00079
    :alt: Cite

.. |CI| image:: https://img.shields.io/github/workflow/status/msmdev/pygpcca/CI/main
    :target: https://github.com/msmdev/pygpcca/actions
    :alt: CI

.. |Docs|  image:: https://img.shields.io/readthedocs/pygpcca
    :target: https://pygpcca.readthedocs.io/en/latest
    :alt: Documentation

.. |Coverage| image:: https://img.shields.io/codecov/c/github/msmdev/pygpcca/main
    :target: https://codecov.io/gh/msmdev/pygpcca
    :alt: Coverage

.. |License| image:: https://img.shields.io/github/license/msmdev/pyGPCCA?color=green
    :target: https://github.com/msmdev/pyGPCCA/blob/main/LICENSE.txt
    :alt: GitHub

.. |GitHubDownloads| image:: https://img.shields.io/github/downloads/msmdev/pyGPCCA/total?label=github%20downloads
    :target: https://github.com/msmdev/pyGPCCA/releases/
    :alt: GitHub all releases

.. |PyPIdownloads| image:: https://img.shields.io/pypi/dm/gpcca?label=pypi%20downloads
    :target: https://pypi.org/project/pygpcca/
    :alt: PyPI - Downloads

.. |CondaDownloads| image:: https://img.shields.io/conda/dn/conda-forge/pygpcca?label=conda%20downloads
    :target: https://anaconda.org/conda-forge/pygpcca
    :alt: Conda

pyGPCCA - Generalized Perron Cluster Cluster Analysis
=====================================================
Generalized Perron Cluster Cluster Analysis program to coarse-grain reversible and non-reversible Markov State Models.

Markov State Models (MSM) enable the identification and analysis of metastable states and related kinetics in a
very instructive manner. They are widely used, e.g. to model molecular or cellular kinetics. |br|
Common state-of-the-art Markov state modeling methods and tools are very well suited to model reversible processes in
closed equilibrium systems. However, most are not well suited to deal with non-reversible or even non-autonomous
processes of non-equilibrium systems. |br|
To overcome this limitation, the Generalized Robust Perron Cluster Cluster Analysis (G-PCCA) was developed.
The G-PCCA method implemented in the *pyGPCCA* program readily handles equilibrium as well as non-equilibrium data by
utilizing real Schur vectors instead of eigenvectors. |br|
*pyGPCCA* enables the semiautomatic coarse-graining of transition matrices representing the dynamics of the system
under study. Utilizing *pyGPCCA*, metastable states as well as cyclic kinetics can be identified and modeled.

If you use *pyGPCCA* or parts of it, please cite `JCTC (2018)`_.

.. _JCTC (2018): https://pubs.acs.org/doi/abs/10.1021/acs.jctc.8b00079

Installation
------------
We support multiple ways of installing *pyGPCCA*. If any problems arise, please consult the
`troubleshooting <https://pygpcca.readthedocs.io/en/latest/installation.html#troubleshooting>`_
section in the documentation.

Conda
+++++
*pyGPCCA* is available as a `conda package <https://anaconda.org/conda-forge/pygpcca>`_ and can be installed as::

    conda install -c conda-forge pygpcca

This is the recommended way of installing, since this package also includes `PETSc`_/`SLEPc`_ libraries.
We use `PETSc`_/`SLEPc`_ internally to speed up the computation of leading Schur vectors (both are optional)

.. _`PETSc`: https://www.mcs.anl.gov/petsc/
.. _`SLEPc`: https://slepc.upv.es/

PyPI
++++
In order to install *pyGPCCA* from `The Python Package Index <https://pypi.org/project/pygpcca/>`_, run::

    pip install pygpcca
    # or with libraries utilizing PETSc/SLEPc
    pip install pygpcca[slepc]

Example
-------
Please refer to our `example usage <https://pygpcca.readthedocs.io/en/latest/example.html>`_ in the documentation.

.. include:: docs/source/acknowledgements.rst

.. |br| raw:: html

  <br/>
