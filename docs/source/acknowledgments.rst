Acknowledgments
----------------
We thank `Marcus Weber`_ and the Computational Molecular Design (`CMD`_) group at the Zuse Institute Berlin (`ZIB`_)
for the longstanding and productive collaboration in the field of Markov modeling of non-reversible molecular dynamics.
M. Weber, together with Susanna Röblitz and K. Fackeldey, had the original idea to employ Schur
vectors instead of eigenvectors in the coarse-graining of non-reversible transition matrices.
Further, we would like to thank `Fabian Paul`_ for valuable discussions regarding the sorting of Schur vectors and his
effort to translate the original sorting routine for real Schur forms, SRSchur published by `Jan Brandts`_,
from MATLAB into `Python code`_,
M. Weber and `Alexander Sikorski`_ for pointing us to `SLEPc`_ for sorted partial Schur decompositions,
and A. Sikorski for supplying us with an `code example`_ and guidance how to interface SLEPc in Python.
The development of *pyGPCCA* started - based on the original `GPCCA`_ program written in MATLAB - at the beginning of
2020 in a fork of `MSMTools`_, since it was planned to integrate GPCCA into MSMTools at this time.
Due to this, some similarities in structure and code (indicated were evident) can be found.
Further, utility functions found in `pygpcca/utils/_utils.py`_ originate from MSMTools.

.. _`Marcus Weber`: https://www.zib.de/members/weber
.. _`CMD`: https://www.zib.de/numeric/cmd
.. _`ZIB`: https://www.zib.de/
.. _`Fabian Paul`: https://github.com/fabian-paul
.. _`Jan Brandts`: https://doi.org/10.1002/nla.274
.. _`Python code`: https://gist.github.com/fabian-paul/14679b43ed27aa25fdb8a2e8f021bad5
.. _`Alexander Sikorski`: https://www.zib.de/members/sikorski
.. _`SLEPc`: https://slepc.upv.es/
.. _`code example`: https://github.com/zib-cmd/cmdtools/blob/1c6b6d8e1c35bb487fcf247c5c1c622b4b665b0a/src/cmdtools/analysis/pcca.py#L64
.. _`GPCCA`: https://github.com/msmdev/gpcca
.. _`MSMTools`: https://github.com/markovmodel/msmtools
.. _`pygpcca/utils/_utils.py`: https://github.com/msmdev/pyGPCCA/blob/main/pygpcca/utils/_utils.py
