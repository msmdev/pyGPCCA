Release Notes
=============

.. role:: small

Version 1.0
-----------
1.0.3 :small:`2022-02-13`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Fixes

- Fix CI, unpin some requirements, pin docs, enable doc linting
  `#25 <https://github.com/msmdev/pyGPCCA/pull/25>`_,
  `#26 <https://github.com/msmdev/pyGPCCA/pull/26>`_.
- Patch release preparation
  `#35 <https://github.com/msmdev/pyGPCCA/pull/35>`_.

.. rubric:: Improvements

- Print deviations, if a test is failing since a threshold is exceeded
  `#29 <https://github.com/msmdev/pyGPCCA/pull/29>`_.
- Adjust too tight thresholds in some tests
  `#30 <https://github.com/msmdev/pyGPCCA/pull/30>`_,
  `#34 <https://github.com/msmdev/pyGPCCA/pull/34>`_.

1.0.2 :small:`2021-03-26`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Bugfixes

- Fix not catching ``ArpackError`` when computing stationary distribution and mypy-related linting issues
  `#21 <https://github.com/msmdev/pyGPCCA/pull/21>`_.

.. rubric:: Improvements

- Use PETSc/SLEPc, if installed, to speed up the computation of the stationary distribution
  `#22 <https://github.com/msmdev/pyGPCCA/pull/22>`_.

1.0.1 :small:`2021-02-01`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. rubric:: General

- Minor improvements/fixes in README and acknowledgments.

1.0.0 :small:`2021-01-29`
~~~~~~~~~~~~~~~~~~~~~~~~~

Initial release.
