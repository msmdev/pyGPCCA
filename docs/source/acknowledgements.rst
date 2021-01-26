Acknowledgements
================
We thank `Marcus Weber`_ and the Computational Molecular Design (`CMD`_) group at the Zuse Institute Berlin (`ZIB`_)
for the longstanding productive collaboration on the field of Markov modeling of nonreversible molecular dynamics.
M. Weber, together with K. Fackeldey, had the original idea to employ Schur vectors instead of eigenvectors in the
coarse-graining of non-reversible transition matrices years ago.
Further, we would like to thank `Fabian Paul`_ for valueable discussions regarding the sorting of Schur vectors and his
effort to translate the original Sorting routine for real Schur forms of `Brandts`_ from MATLAB into `Python code`_,
M. Weber and `Alexander Sikorski`_ for pointing us to SLEPc for sorted partial Schur decompositions,
and A. Sikorski for supplying us with an `code example`_ and guidance how to interface SLEPc in Python.

.. _`Marcus Weber`: https://www.zib.de/members/weber
.. _`CMD`: https://www.zib.de/numeric/cmd
.. _`ZIB`: https://www.zib.de/
.. _`Fabian Paul`: https://github.com/fabian-paul
.. _`Brandts`: https://onlinelibrary.wiley.com/doi/abs/10.1002/nla.274
.. _`Python code`: https://gist.github.com/fabian-paul/14679b43ed27aa25fdb8a2e8f021bad5
.. _`Alexander Sikorski`: https://www.zib.de/members/sikorski
.. _`code example`: https://github.com/zib-cmd/cmdtools/blob/1c6b6d8e1c35bb487fcf247c5c1c622b4b665b0a/src/cmdtools/analysis/pcca.py#L64
