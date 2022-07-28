|PyPI| |Conda| |Cite| |Zenodo| |CI| |Docs| |Coverage| |License| |PyPIdownloads|

.. |PyPI| image:: https://img.shields.io/pypi/v/pygpcca
    :target: https://pypi.org/project/pygpcca/
    :alt: PyPI

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/pygpcca
    :target: https://anaconda.org/conda-forge/pygpcca
    :alt: Conda

.. |Cite| image:: https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.8b00079-blue
    :target: https://doi.org/10.1021/acs.jctc.8b00079
    :alt: Cite
    
.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6914001.svg
   :target: https://doi.org/10.5281/zenodo.6914001
   :alt: Zenodo

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
    :alt: License

.. |PyPIdownloads| image:: https://static.pepy.tech/personalized-badge/pygpcca?period=total&units=international_system&left_color=grey&right_color=blue&left_text=pypi%20downloads
    :target: https://pepy.tech/project/pygpcca
    :alt: PyPI - Downloads

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

How to cite pyGPCCA
-------------------
If you use *pyGPCCA* or parts of it, cite `JCTC (2018)`_ and `pyGPCCA`_.

.. _JCTC (2018): https://pubs.acs.org/doi/abs/10.1021/acs.jctc.8b00079
.. _pyGPCCA: https://doi.org/10.5281/zenodo.6913970

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
We use `PETSc`_/`SLEPc`_ internally to speed up the computation of leading Schur vectors (both are optional).

.. _`PETSc`: https://www.mcs.anl.gov/petsc/

PyPI
++++
In order to install *pyGPCCA* from `The Python Package Index <https://pypi.org/project/pygpcca/>`_, run::

    pip install pygpcca
    # or with libraries utilizing PETSc/SLEPc
    pip install pygpcca[slepc]

Example
-------
Please refer to our `example usage <https://pygpcca.readthedocs.io/en/latest/example.html>`_ in the documentation.

Key Contributors
----------------
* `Bernhard Reuter`_: lead developer, maintainer.
* `Michal Klein`_: developer, diverse contributions.
* `Marius Lange`_: developer, diverse contributions.

.. _Bernhard Reuter: https://github.com/msmdev
.. _Michal Klein: https://github.com/michalk8
.. _Marius Lange: https://github.com/Marius1311

Acknowledgements
----------------
We thank `Marcus Weber`_ and the Computational Molecular Design (`CMD`_) group at the Zuse Institute Berlin (`ZIB`_)
for the longstanding and productive collaboration in the field of Markov modeling of non-reversible molecular dynamics.
M. Weber, together with K. Fackeldey, had the original idea to employ Schur vectors instead of eigenvectors in the
coarse-graining of non-reversible transition matrices. |br|
Further, we would like to thank `Fabian Paul`_ for valuable discussions regarding the sorting of Schur vectors and his
effort to translate the original Sorting routine for real Schur forms `SRSchur`_ published by `Jan Brandts`_ from MATLAB
into `Python code`_,
M. Weber and `Alexander Sikorski`_ for pointing us to `SLEPc`_ for sorted partial Schur decompositions,
and A. Sikorski for supplying us with an `code example`_ and guidance how to interface SLEPc in Python.
The development of *pyGPCCA* started - based on the original `GPCCA`_ program written in MATLAB - at the beginning of
2020 in a fork of `MSMTools`_, since it was planned to integrate GPCCA into MSMTools at this time.
Due to this, some similarities in structure and code (indicated were evident) can be found.
Futher the utility functions found in `pygpcca/utils/_utils.py`_ originate from MSMTools.

.. _`Marcus Weber`: https://www.zib.de/members/weber
.. _`CMD`: https://www.zib.de/numeric/cmd
.. _`ZIB`: https://www.zib.de/
.. _`Fabian Paul`: https://github.com/fabian-paul
.. _`SRSchur`: http://m2matlabdb.ma.tum.de/SRSchur.m?MP_ID=119
.. _`Jan Brandts`: https://doi.org/10.1002/nla.274
.. _`Python code`: https://gist.github.com/fabian-paul/14679b43ed27aa25fdb8a2e8f021bad5
.. _`Alexander Sikorski`: https://www.zib.de/members/sikorski
.. _`SLEPc`: https://slepc.upv.es/
.. _`code example`: https://github.com/zib-cmd/cmdtools/blob/1c6b6d8e1c35bb487fcf247c5c1c622b4b665b0a/src/cmdtools/analysis/pcca.py#L64
.. _`GPCCA`: https://github.com/msmdev/gpcca
.. _`MSMTools`: https://github.com/markovmodel/msmtools
.. _`pygpcca/utils/_utils.py`: https://github.com/msmdev/pyGPCCA/blob/main/pygpcca/utils/_utils.py

.. |br| raw:: html

  <br/>
