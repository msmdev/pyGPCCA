# This file is part of pyGPCCA.
#
# Copyright (c) 2020 Bernhard Reuter.
# With contributions of Marius Lange, Michal Klein and Alexander Sikorski.
# Based on the original MATLAB GPCCA code authored by Bernhard Reuter, Susanna Roeblitz and Marcus Weber,
# Zuse Institute Berlin, Takustrasse 7, 14195 Berlin
# We like to thank A. Sikorski and M. Weber for pointing us to SLEPc for partial Schur decompositions of
# sparse matrices.
# Further parts of sorted_krylov_schur were developed based on the function krylov_schur
# https://github.com/zib-cmd/cmdtools/blob/1c6b6d8e1c35bb487fcf247c5c1c622b4b665b0a/src/cmdtools/analysis/pcca.py#L64,
# written by Alexander Sikorski.
# --------------------------------------------------------------------------------------------------------------------
# If you use this code or parts of it, cite the following reference:
# --------------------------------------------------------------------------------------------------------------------
# Reuter, B., Weber, M., Fackeldey, K., Röblitz, S., & Garcia, M. E. (2018).
# Generalized Markov State Modeling Method for Nonequilibrium Biomolecular Dynamics:
# Exemplified on Amyloid β Conformational Dynamics Driven by an Oscillating Electric Field.
# Journal of Chemical Theory and Computation, 14(7), 3579–3594. https://doi.org/10.1021/acs.jctc.8b00079
# --------------------------------------------------------------------------------------------------------------------
# pyGPCCA is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser
# General Public License as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------------------------------------------------
from typing import Tuple, Union
import sys

if not sys.warnoptions:
    import os
    import warnings

    warnings.simplefilter("always", category=UserWarning)  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "always::UserWarning"  # Also affect subprocesses

from scipy.linalg import schur, rsf2csf, subspace_angles
from scipy.sparse import issparse, spmatrix, csr_matrix, isspmatrix_csr
import numpy as np

from pygpcca.utils._docs import d
from pygpcca._sort_real_schur import sort_real_schur
from pygpcca.utils._constants import EPS, DEFAULT_SCHUR_METHOD, NO_PETSC_SLEPC_FOUND_MSG

try:
    import petsc4py
    import slepc4py
except ImportError:
    petsc4py = None
    slepc4py = None

__all__ = ["sorted_schur"]


def _initialize_matrix(M: "petsc4py.PETSc.Mat", P: Union[np.ndarray, spmatrix]) -> None:
    """
    Initialize PETSc matrix.

    Parameters
    ----------
    M
        :mod:`petsc4py` matrix to initialize.
    P
        :mod:`numpy` array or :mod:`scipy` sparse matrix from which we take the data.

    Returns
    -------
    Nothing, just initializes `M`. If `P` is an :class:`numpy.ndarray`,
    `M` will also be dense. If `P` is a :class:`scipy.sparse.spmatrix`,
    `M` will become a CSR matrix regardless of `P`'s sparse format.
    """
    if issparse(P):
        if not isspmatrix_csr(P):
            warnings.warn("Only CSR sparse matrices are supported, converting.")
            P = csr_matrix(P)
        M.createAIJ(size=P.shape, csr=(P.indptr, P.indices, P.data))  # type: ignore[union-attr]
    else:
        M.createDense(list(np.shape(P)), array=P)


@d.dedent
def _check_conj_split(eigenvalues: np.ndarray) -> bool:
    """
    Check whether using m eigenvalues cuts through a block of complex conjugates.

    If the last (`m`th) eigenvalue is not real, check whether it
    forms a complex conjugate pair with the second-last eigenvalue.
    If that is not the case, then choosing `m` clusters would cut through a
    block of complex conjugates.

    Parameters
    ----------
    eigenvalues
        %(eigenvalues_m)s

    Returns
    -------
    ``True`` if a block of complex conjugate eigenvalues is split, ``False`` otherwise.
    """
    last_eigenvalue, second_last_eigenvalue = eigenvalues[-1], eigenvalues[-2]
    splits_block = False
    if last_eigenvalue.imag > EPS:
        splits_block = not np.isclose(last_eigenvalue, np.conj(second_last_eigenvalue))

    return splits_block


@d.dedent
def _check_schur(P: np.ndarray, Q: np.ndarray, R: np.ndarray, eigenvalues: np.ndarray, method: str) -> None:
    """
    Run a number of checks on the sorted Schur decomposition.

    Parameters
    ----------
    %(P)s
    Q
        %(Q_sort)s
    R
        %(R_sort)s
    eigenvalues
        %(eigenvalues_m)s
    %(method)s

    Returns
    -------
    Nothing.
    """
    m = len(eigenvalues)

    # check the dimensions
    if Q.shape[1] != len(eigenvalues):
        raise ValueError(f"Number of Schur vectors does not match number of eigenvalues for `method={method!r}`.")
    if R.shape[0] != R.shape[1]:
        raise ValueError(f"R is not rectangular for `method={method!r}`.")
    if P.shape[0] != Q.shape[0]:
        raise ValueError(f"First dimension in P does not match first dimension in Q for `method={method!r}`.")
    if R.shape[0] != Q.shape[1]:
        raise ValueError(f"First dimension in R does not match second dimension in Q for `method={method!r}`.")

    # check whether things are real
    if not np.all(np.isreal(Q)):
        raise TypeError(
            f"The orthonormal basis of the subspace returned by `method={method!r}` is not real. "
            "G-PCCA needs real basis vectors to work."
        )

    dummy = np.dot(P, csr_matrix(Q) if issparse(P) else Q)
    if issparse(dummy):
        dummy = dummy.toarray()

    dummy1 = np.dot(Q, np.diag(eigenvalues))
    # dummy2 = np.concatenate((dummy, dummy1), axis=1)
    dummy3 = subspace_angles(dummy, dummy1)
    # test1 = ( ( matrix_rank(dummy2) - matrix_rank(dummy) ) == 0 )
    test2 = np.allclose(dummy3, 0.0, atol=1e-8, rtol=1e-5)
    test3 = dummy3.shape[0] == m
    dummy4 = subspace_angles(dummy, Q)
    test4 = np.allclose(dummy4, 0.0, atol=1e-6, rtol=1e-5)
    if not test4:
        raise ValueError(
            f"According to `scipy.linalg.subspace_angles()`, `{method}` didn't "
            f"return an invariant subspace of P. The subspace angles are: `{dummy4}`."
        )

    if not test2:
        warnings.warn(
            f"According to `scipy.linalg.subspace_angles()`, `{method}` didn't "
            f"return the invariant subspace associated with the top k eigenvalues, "
            f"since the subspace angles between the column spaces of P*Q and Q*L "
            f"aren't near zero (L is a diagonal matrix with the "
            f"sorted top eigenvalues on the diagonal). The subspace angles are: `{dummy3}`."
        )

    if not test3:
        warnings.warn(
            f"According to `scipy.linalg.subspace_angles()`, the dimension of the "
            f"column space of P*Q and/or Q*L is not equal to m (L is a diagonal "
            f"matrix with the sorted top eigenvalues on the diagonal), method=`{method}`."
        )


@d.dedent
def sorted_krylov_schur(
    P: Union[spmatrix, np.ndarray], k: int, z: str = "LM", tol: float = 1e-16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Calculate an orthonormal basis of the subspace associated with the `k`
    dominant eigenvalues of `P` using the Krylov-Schur method as implemented in SLEPc.

    This functions requires :mod:`petsc4py` and :mod:`slepc4py`.

    Parameters
    ----------
    %(P)s
    %(k)s
    %(z)s
    tol
        Convergence criterion used by SLEPc internally. If you are dealing
        with ill-conditioned matrices, consider decreasing this value to
        get accurate results.

    Returns
    -------
    Tuple of the following:

    R
        %(R_sort)s
    Q
        %(Q_sort)s
    eigenvalues
        %(eigenvalues_k)s
    eigenvalues_error
        Array of shape `(k,)` containing the error, based on the residual
        norm, of the `i`th eigenpair at index `i`.
    """  # noqa: D205, D400
    # We like to thank A. Sikorski and M. Weber for pointing us to SLEPc for partial Schur decompositions of
    # sparse matrices.
    # Further parts of sorted_krylov_schur were developed based on the function krylov_schur
    # https://github.com/zib-cmd/cmdtools/blob/1c6b6d8e1c35bb487fcf247c5c1c622b4b665b0a/src/cmdtools/analysis/pcca.py#L64,
    # written by Alexander Sikorski.
    from petsc4py import PETSc
    from slepc4py import SLEPc

    M = PETSc.Mat().create()
    _initialize_matrix(M, P)
    # Creates EPS object.
    E = SLEPc.EPS()
    E.create()
    # Set the matrix associated with the eigenvalue problem.
    E.setOperators(M)
    # Select the particular solver to be used in the EPS object: Krylov-Schur
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    # Set the number of eigenvalues to compute and the dimension of the subspace.
    E.setDimensions(nev=k)
    # set the tolerance used in the convergence criterion
    E.setTolerances(tol=tol)

    # Specify which portion of the spectrum is to be sought.
    # LARGEST_MAGNITUDE: Largest magnitude (default).
    # LARGEST_REAL: Largest real parts.
    # All possible Options can be found here:
    # (see: https://slepc.upv.es/slepc4py-current/docs/apiref/slepc4py.SLEPc.EPS.Which-class.html)
    if z == "LM":
        E.setWhichEigenpairs(E.Which.LARGEST_MAGNITUDE)
    elif z == "LR":
        E.setWhichEigenpairs(E.Which.LARGEST_REAL)
    else:
        raise ValueError(f"Invalid spectrum sorting options `{z}`. Valid options are: 'LM', 'LR'.")

    # Solve the eigensystem.
    E.solve()

    # getInvariantSubspace() gets an orthonormal basis of the computed invariant subspace.
    # It returns a list of vectors.
    # The returned real vectors span an invariant subspace associated with the computed eigenvalues.
    # We take the sequence of 1-D arrays and stack them as columns to make a single 2-D array.
    Q = np.column_stack([x.array for x in E.getInvariantSubspace()])

    try:
        # otherwise, R would be of shape `(k + 1, k)`
        E.getDS().setExtraRow(False)
    except AttributeError:
        pass
    # Get the schur form
    R = E.getDS().getMat(SLEPc.DS.MatType.A)
    R.view()
    R = R.getDenseArray().astype(np.float64)

    # Gets the number of converged eigenpairs.
    nconv = E.getConverged()

    # Warn, if nconv smaller than k.
    if nconv < k:
        warnings.warn(f"The number of converged eigenpairs is `{nconv}`, but `{k}` were requested.")

    # Collect the k dominant eigenvalues.
    eigenvalues = []
    eigenvalues_error = []
    for i in range(nconv):
        # Get the i-th eigenvalue as computed by solve().
        eigenval = E.getEigenvalue(i)
        eigenvalues.append(eigenval)
        # Computes the error (based on the residual norm) associated with the i-th computed eigenpair.
        eigenval_error = E.computeError(i)
        eigenvalues_error.append(eigenval_error)

    # convert lists with eigenvalues and errors to arrays (while keeping excess eigenvalues and errors)
    eigenvalues = np.asarray(eigenvalues)  # type: ignore[assignment]
    eigenvalues_error = np.asarray(eigenvalues_error)  # type: ignore[assignment]

    return R, Q, eigenvalues, eigenvalues_error  # type: ignore[return-value]


@d.dedent
def sorted_brandts_schur(P: np.ndarray, k: int, z: str = "LM") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a sorted Schur decomposition.

    This function uses :mod:`scipy` for the decomposition and Brandts'
    method (see [Brandts02]_) for the sorting.

    Parameters
    ----------
    %(P)s
    %(k)s
    %(z)s

    Returns
    -------
    Tuple of the following:

    R
        %(R_sort)s
    Q
        %(Q_sort)s
    eigenvalues
        %(eigenvalues_k)s
    """
    # Make a Schur decomposition of P.
    R, Q = schur(P, output="real")

    # Sort the Schur matrix and vectors.
    Q, R, ap = sort_real_schur(Q, R, z=z, b=k)

    # Warnings
    if np.any(np.array(ap) > 1.0):
        warnings.warn("Reordering of Schur matrix was inaccurate.")

    # compute eigenvalues
    T, _ = rsf2csf(R, Q)
    eigenvalues = np.diag(T)[:k]

    return R, Q, eigenvalues


@d.dedent
def sorted_schur(
    P: Union[np.ndarray, spmatrix], m: int, z: str = "LM", method: str = DEFAULT_SCHUR_METHOD, tol_krylov: float = 1e-16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``m`` dominant real Schur vectors or an orthonormal basis spanning the same invariant subspace.

    Parameters
    ----------
    %(P)s
    %(m)s
    %(z)s
    %(method)s
    %(tol_krylov)s

    Returns
    -------
    Tuple of the following:

    R
        %(R_sort)s
    Q
        %(Q_sort)s
    eigenvalues
        %(eigenvalues_m)s
    """
    if method == "krylov":
        if petsc4py is None or slepc4py is None:
            method = DEFAULT_SCHUR_METHOD
            warnings.warn(NO_PETSC_SLEPC_FOUND_MSG)

    if method != "krylov" and issparse(P):
        raise ValueError("Sparse implementation is only available for `method='krylov'`.")

    # make sure we have enough eigenvalues to check for block splitting
    n = P.shape[0]
    if m > n:
        raise ValueError(f"Requested more groups than states: {m} > {n}.")

    # compute the sorted schur decomposition
    if method == "brandts":
        R, Q, eigenvalues = sorted_brandts_schur(P=P, k=m, z=z)
    elif method == "krylov":
        R, Q, eigenvalues, _ = sorted_krylov_schur(P=P, k=m, z=z, tol=tol_krylov)
    else:
        raise ValueError(f"Unknown method `{method!r}`.")

    # check for splitting pairs of complex conjugates
    if m < n:
        if _check_conj_split(eigenvalues[:m]):
            raise ValueError(
                f"Clustering into {m} clusters will split complex conjugate eigenvalues. "
                "Request one cluster more or less."
            )
        Q, R, eigenvalues = Q[:, :m], R[:m, :m], eigenvalues[:m]

    # check the returned schur decomposition
    _check_schur(P=P, Q=Q, R=R, eigenvalues=eigenvalues, method=method)

    return R, Q, eigenvalues
