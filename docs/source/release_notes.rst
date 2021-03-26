Release Notes
=============

.. role:: small

Version 1.0
-----------

1.0.2 :small:`2021-03-26`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Bugfixes

- Fix not catching ``ArpackError`` when computing stationary distribution and mypy-related linting issues
  `PR 21 <https://github.com/msmdev/pyGPCCA/pull/21>`_.

.. rubric:: Improvements

- Use PETSc/SLEPc, if installed, to speed up the computation of the stationary distribution
  `PR 22 <https://github.com/msmdev/pyGPCCA/pull/22>`_.

1.0.1 :small:`2021-02-01`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. rubric:: General

- Minor improvements/fixes in README and acknowledgements.

1.0.0 :small:`2021-01-29`
~~~~~~~~~~~~~~~~~~~~~~~~~

Initial release.
