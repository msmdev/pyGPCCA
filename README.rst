|PyPI| |Conda| |CI| |Coverage|

pyGPCCA - Generalized Perron Cluster Cluster Analysis
=====================================================
Generalized Perron Cluster Cluster Analysis program to coarse-grain reversible and non-reversible Markov State Models.

Markov State Models (MSM) enable the identification and analysis of metastable states and related kinetics in a very instructive manner. They are widely used, e.g. to model molecular or cellular kinetics. 
Common state-of-the-art Markov state modeling methods and tools are very well suited to model reversible processes in closed equilibrium systems. However, most are not well suited to deal with non-reversible or even non-autonomous processes of non-equilibrium systems. 
To get over this shortcoming, the Generalized Robust Perron Cluster Cluster Analysis (G-PCCA) was developed. The G-PCCA method implemented in the pyGPCCA program readily handles equilibrium as well as non-equilibrium data by utilizing real Schur vectors instead of eigenvectors. pyGPCCA enables the semiautomatic coarse graining of transition matrices representing the dynamics of the system under study. Utilizing pyGPCCA, metastable states as well as cyclic kinetics can be identified and modeled.

Installation
------------
To install the development version of *pyGPCCA* from GitHub, run::

    pip install git+https://github.com/msmdev/pygpcca

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
    
Usage
-----

Afterwards pyGPCCA can be imported in Python

``import pygpcca as gp``

pyGPCCA can be used as outlined in the following:

    - Initialize a GPCCA object with a transition matrix `P`:

    ``gpcca = gp.GPCCA(P)``
    
    - Get a list of minChi values for numbers of macrostates `m` in an interval ``[2,30]`` to determine an interval ``[m_min, m_max]`` of (nearly) optimal numbers of macrostates for clustering: 

    ``minChi_list = gpcca.minChi(2, 30)``

    - Optimize the clustering for numbers of macrostates `m` in the previously determined interval ``[m_min, m_max]`` and find the optimal number of macrostates `n_metastable` in the given interval:

    ``gpcca.optimize({'m_min':2, 'm_max':10})``
    
    - Afterwards, the optimal number of macrostates `n_metastable` can be accessed via:
    
    ``gpcca.n_metastable``
    
    - The optimal coarse-grained matrix can be accessed via:

    ``gpcca.coarse_grained_transition_matrix``

    - The memberships are available via:

    ``gpcca.memberships``
