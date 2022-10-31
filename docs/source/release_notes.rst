Release Notes
=============

.. role:: small

Version 1.0
-----------
1.0.4 :small:`2022-10-31`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Fixes

- Fix 'Operation done in wrong order' error when calling SLEPc
  `#42 <https://github.com/msmdev/pyGPCCA/pull/42>`_.
- Minor pre-commit/linting fixes
  `#39 <https://github.com/msmdev/pyGPCCA/pull/39>`_,
  `#40 <https://github.com/msmdev/pyGPCCA/pull/40>`_,
  `#41 <https://github.com/msmdev/pyGPCCA/pull/41>`_,
  `#44 <https://github.com/msmdev/pyGPCCA/pull/44>`_,
  `#45 <https://github.com/msmdev/pyGPCCA/pull/45>`_,
  `#46 <https://github.com/msmdev/pyGPCCA/pull/46>`_.
- Fix intersphinx numpy/scipy
  `#37 <https://github.com/msmdev/pyGPCCA/pull/37>`_.

.. rubric:: Improvements

- Update and improve documentation and README
  `#47 <https://github.com/msmdev/pyGPCCA/pull/47>`_.

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
