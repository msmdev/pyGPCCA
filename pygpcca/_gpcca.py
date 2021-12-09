# This file is part of pyGPCCA.
#
# Copyright (c) 2020 Bernhard Reuter.
# With contributions of Marius Lange and Michal Klein.
# Based on the original MATLAB GPCCA code authored by Bernhard Reuter, Susanna Roeblitz and Marcus Weber,
# Zuse Institute Berlin, Takustrasse 7, 14195 Berlin
# ---------------------------------------------------------------------------------------------------------------------
# The development of pyGPCCA started at the beginning of 2020 in a fork of MSMTools
# (Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER);
# provided under LGPL-3.0 License), since at this time it was planned to integrate GPCCA into it.
# Due to this, some similarities in structure/AST and code (indicated were evident) between pcca.py
# https://github.com/markovmodel/msmtools/blob/93126608c6fa9c3197f4fae2f6da93140762b047/msmtools/analysis/dense/pcca.py
# and _gpcca.py can be found.
# ---------------------------------------------------------------------------------------------------------------------
# If you use this code or parts of it, cite the following reference:
# ---------------------------------------------------------------------------------------------------------------------
# Reuter, B., Weber, M., Fackeldey, K., Röblitz, S., & Garcia, M. E. (2018).
# Generalized Markov State Modeling Method for Nonequilibrium Biomolecular Dynamics:
# Exemplified on Amyloid β Conformational Dynamics Driven by an Oscillating Electric Field.
# Journal of Chemical Theory and Computation, 14(7), 3579–3594. https://doi.org/10.1021/acs.jctc.8b00079
# ---------------------------------------------------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------------------------------------------------

__author__ = __maintainer__ = "Bernhard Reuter"
__email__ = "bernhard-reuter@gmx.de"
__copyright__ = "Copyright 2020, Bernhard Reuter"
__credits__ = [
    "Bernhard Reuter",
    "Marcus Weber",
    "Susanna Roeblitz",
    "Marius Lange",
    "Michal Klein",
    "Fabian Paul",
    "Alexander Sikorski",
]


from typing import Dict, List, Tuple, Union, Callable, Optional, TYPE_CHECKING

try:
    from functools import cached_property  # type: ignore[attr-defined]
except ImportError:
    from functools import lru_cache

    def cached_property(fn: Callable) -> property:  # type: ignore[no-redef,type-arg]
        """Cached property backport."""  # noqa: D401
        # mypy complains about overriding the same name
        return property(lru_cache(maxsize=1)(fn))


import sys
import logging

if not sys.warnoptions:
    import os
    import warnings

    warnings.simplefilter("always", category=UserWarning)  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "always::UserWarning"  # Also affect subprocesses

from scipy.linalg import subspace_angles
from scipy.sparse import issparse, spmatrix
from scipy.optimize import fmin
import numpy as np
import scipy.sparse as sp

from pygpcca.utils._docs import d
from pygpcca.utils._utils import (
    connected_sets,
    is_transition_matrix,
    stationary_distribution,
)
from pygpcca._sorted_schur import sorted_schur, _check_conj_split
from pygpcca.utils._constants import EPS, DEFAULT_SCHUR_METHOD

__all__ = ["gpcca_coarsegrain", "GPCCA"]
OArray = Optional[np.ndarray]


@d.dedent
def _gram_schmidt_mod(X: np.ndarray, eta: np.ndarray) -> np.ndarray:
    r"""
    :math:`\eta`-orthonormalize Schur vectors.

    This uses a modified, numerically stable version of Gram-Schmidt
    Orthonormalization.

    Parameters
    ----------
    X
        Array of shape `(n, m)` consisting columnwise of the `m` dominant
        Schur vectors of :math:`\tilde{P} = \mathtt{diag}(\sqrt{\eta}) P \mathtt{diag}(1.0. / \sqrt{eta})`.
    %(eta)s

    Returns
    -------
    Array of shape `(n, m)` with the orthonormalized `m` dominant Schur
    vectors of :math:`\tilde{P}` in columns.
    The elements of the first column are constantly equal :math:`\sqrt{eta}`.
    """
    # Keep copy of the original (Schur) vectors for later sanity check.
    Xc = np.copy(X)

    # Initialize matrices.
    n, m = X.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    # Search for the constant (Schur) vector, if explicitly present.
    max_i = 0
    for i in range(m):
        vsum = np.sum(X[:, i])
        dummy = np.ones(X[:, i].shape) * (vsum / n)
        if np.allclose(X[:, i], dummy, rtol=1e-6, atol=1e-5):
            max_i = i  # TODO: check, if more than one vec fulfills this

    # Shift non-constant first (Schur) vector to the right.
    X[:, max_i] = X[:, 0]
    # Set first (Schur) vector equal sqrt(eta) (In _do_schur() the Q-matrix, orthogonalized by
    # _gram_schmidt_mod(), will be multiplied with 1.0./sqrt(eta) - so the first (Schur) vector will
    # become the unit vector 1!).
    X[:, 0] = np.sqrt(eta)
    # Raise, if the subspace changed!
    dummy = subspace_angles(X, Xc)
    if not np.allclose(dummy, 0.0, atol=1e-7, rtol=1e-5):
        logging.error(Xc)
        logging.error(X)
        raise ValueError(
            "The subspace of Q derived by shifting a non-constant first (Schur)vector "
            "to the right and setting the first (Schur) vector equal sqrt(eta) doesn't "
            f"match the subspace of the original Q! The subspace angles are: {dummy}. "
            f"Number of clusters: {m}."
        )

    # eta-orthonormalization
    for j in range(m):
        v = X[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i].conj(), v)
            v = v - np.dot(R[i, j], Q[:, i])
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = np.true_divide(v, R[j, j])

    # Raise, if the subspace changed!
    dummy = subspace_angles(Q, Xc)
    if not np.allclose(dummy, 0.0, atol=1e-7, rtol=1e-5):
        raise ValueError(
            "The subspace of Q derived by eta-orthogonalization doesn't match the "
            f"subspace of the original Q! The subspace angles are: {dummy}. "
            f"Number of clusters: {m}."
        )
    # Raise, if the (Schur)vectors aren't orthogonal!
    if not np.allclose(Q.conj().T.dot(Q), np.eye(Q.shape[1]), atol=1e-8, rtol=1e-5):
        dev = np.max(np.abs(Q.conj().T.dot(Q) - np.eye(Q.shape[1])))
        raise ValueError(
            f"(Schur)vectors do not appear to be orthogonal. Largest absolute element-wise deviation in "
            f"(Q^*TQ - I) is {dev}"
        )

    return Q


@d.dedent
def _do_schur(
    P: Union[np.ndarray, spmatrix],
    eta: np.ndarray,
    m: int,
    z: str = "LM",
    method: str = DEFAULT_SCHUR_METHOD,
    tol_krylov: float = 1e-16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Firstly, a Schur decomposition of the `(n, n)` transition matrix `P`
    is performed, with due regard to the input distribution of states `eta`.

    In theory `eta` can be an arbitrary distribution as long as it is
    a valid probability distribution (i.e., sums up to 1).
    A neutral and valid choice would be the uniform distribution (default).
    In case of a reversible transition matrix, the stationary distribution
    :math:`\pi` can (but don't has to) be used here.
    In case of a non-reversible `P`, some initial or average distribution of
    the states might be chosen instead of the uniform distribution.

    Afterwards the Schur form and Schur vector matrix are sorted by
    sorting the `m` dominant (default: with the largest magnitude)
    eigenvalues to the top left of the Schur form in descending order
    and correspondingly sorting the associated Schur vectors
    to the left of the Schur vector matrix.

    Finally, nly the top left `(m, m)` part of the sorted Schur form and the
    associated left `(n, m)` part of the correspondingly sorted Schur
    vector matrix are returned.

    Parameters
    ----------
    %(P)s
    %(eta)s
    %(m)s
        These correspond to the `m` dominant (default: with the largest
        magnitude) eigenvalues.
    %(z)s
    %(method)s
    %(tol_krylov)s

    Returns
    -------
    Triple of the following:

    X
        %(Q_sort)s
    R
        %(R_sort)s
    eigenvalues
        %(eigenvalues_m)s
    """  # noqa: D205, D400
    # Exceptions
    N1 = P.shape[0]
    N2 = P.shape[1]
    if m < 0:
        raise ValueError("The number of clusters/states is not supposed to be negative.")
    if N1 != N2:
        raise ValueError("P matrix isn't quadratic.")
    if eta.shape[0] != N1:
        raise ValueError("eta vector length doesn't match with the shape of P.")
    if not np.allclose(np.sum(P, 1), 1.0, rtol=1e-6, atol=1e-6):  # previously eps
        dev = np.max(np.abs(np.sum(P, 1) - 1.0))
        raise ValueError(
            f"Not all rows of P sum up to one. P must be a row-stochastic matrix Largest deviation from row-sums to 1 "
            f"is {dev}."
        )
    if not np.all(eta > EPS):
        smallest_eta = np.min(eta)
        raise ValueError(f"Not all elements of eta are > 0. The smallest element is {smallest_eta}")

    # Weight the stochastic matrix P by the input (initial) distribution eta.
    if issparse(P):
        A = sp.dia_matrix(([np.sqrt(eta)], [0]), shape=P.shape)
        B = sp.dia_matrix(([1.0 / np.sqrt(eta)], [0]), shape=P.shape)
        P_bar = A.dot(P).dot(B)
    else:
        P_bar = np.diag(np.sqrt(eta)).dot(P).dot(np.diag(1.0 / np.sqrt(eta)))

    # Make a Schur decomposition of P_bar and sort the Schur vectors (and form).
    R, Q, eigenvalues = sorted_schur(P_bar, m, z, method, tol_krylov=tol_krylov)  # Pbar!!!

    # Orthonormalize the sorted Schur vectors Q via modified Gram-Schmidt-orthonormalization,
    # if the (Schur)vectors aren't orthogonal!
    if not np.allclose(Q.T.dot(Q * eta[:, None]), np.eye(Q.shape[1]), rtol=1e6 * EPS, atol=1e6 * EPS):
        logging.debug("The Schur vectors aren't D-orthogonal so they are D-orthogonalized.")
        Q = _gram_schmidt_mod(Q, eta)
        # Transform the orthonormalized Schur vectors of P_bar back
        # to orthonormalized Schur vectors X of P.
        X = np.true_divide(Q, np.sqrt(eta)[:, None])
    else:
        # Search for the constant (Schur) vector, if explicitly present.
        n, m = Q.shape
        max_i = 0
        for i in range(m):
            vsum = np.sum(Q[:, i])
            dummy = np.ones(Q[:, i].shape) * (vsum / n)
            if np.allclose(Q[:, i], dummy, rtol=1e-6, atol=1e-5):
                max_i = i  # TODO: check, if more than one vec fulfills this

        # Shift non-constant first (Schur) vector to the right.
        Q[:, max_i] = Q[:, 0]
        # Transform the orthonormalized Schur vectors of P_bar back
        # to orthonormalized Schur vectors X of P.
        X = np.true_divide(Q, np.sqrt(eta)[:, None])
        # Set first (Schur) vector equal 1.
        X[:, 0] = 1.0

    if not X.shape[0] == N1:
        raise ValueError(
            f"The number of rows `n={X.shape[0]}` of the Schur vector matrix X doesn't match "
            f"those `n={P.shape[0]}` of P."
        )
    # Raise, if the first column X[:,0] of the Schur vector matrix isn't constantly equal 1!
    if not np.allclose(X[:, 0], 1.0, atol=1e-8, rtol=1e-5):
        dev = np.max(np.abs(X[:, 0] - 1.0))
        raise ValueError(
            f"The first column X[:, 0] of the Schur vector matrix isn't constantly equal 1. The largest "
            f"deviation from one is {dev}."
        )

    # Raise, if the (Schur)vectors aren't D-orthogonal (don't fullfill the orthogonality condition)!
    if not np.allclose(X.T.dot(X * eta[:, None]), np.eye(X.shape[1]), atol=1e-6, rtol=1e-5):
        dev = np.max(np.abs(X.T.dot(X * eta[:, None]) - np.eye(X.shape[1])))
        raise ValueError(
            f"Schur vectors appear to not be D-orthogonal. The largets deviation of X^T D X from the "
            f"identity matrix is {dev}"
        )

    # Raise, if X doesn't fullfill the invariant subspace condition!
    dp = np.dot(P, sp.csr_matrix(X) if issparse(P) else X)
    dummy = subspace_angles(dp.toarray() if issparse(dp) else dp, np.dot(X, R))

    test = np.allclose(dummy, 0.0, atol=1e-6, rtol=1e-5)
    test1 = dummy.shape[0] == m
    if not test:
        raise ValueError(
            f"According to `scipy.linalg.subspace_angles()`, X isn't an invariant "
            f"subspace of P, since the subspace angles between the column spaces "
            f"of P*X and X*R aren't near zero. The subspace angles are: `{dummy}`."
        )

    if not test1:
        warnings.warn(
            "According to `scipy.linalg.subspace_angles()` the dimension of the "
            f"column spaces of P*X and/or X*R is not equal to {m}."
        )

    return X, R, eigenvalues


@d.dedent
def _initialize_rot_matrix(X: np.ndarray) -> np.ndarray:
    """
    Initialize the rotation matrix.

    Parameters
    ----------
    X
        %(Q_sort)s

    Returns
    -------
    Initial (non-optimized) rotation matrix of shape `(m, m)`.
    """
    # Search start simplex vertices ('inner simplex algorithm').
    index = _indexsearch(X)

    # Local copy of the Schur vectors.
    # Xc = np.copy(X)

    # Raise or warn if condition number is (too) high.
    condition = np.linalg.cond(X[index, :])
    if condition >= (1.0 / EPS):
        raise ValueError(
            f"The condition number {condition} of the matrix of start simplex vertices "
            "X[index, :] is too high for safe inversion (to build the initial rotation matrix)."
        )
    if condition > 1e4:
        warnings.warn(
            f"The condition number {condition} of the matrix of start simplex vertices "
            "X[index, :] is quite high for safe inversion (to build the initial rotation matrix)."
        )

    # Compute transformation matrix rot_matrix as initial guess for local optimization (maybe not feasible!).
    return np.linalg.pinv(X[index, :])


@d.dedent
def _indexsearch(X: np.ndarray) -> np.ndarray:
    """
    Find a simplex structure in the data.

    Parameters
    ----------
    X
        %(Q_sort)s

    Returns
    -------
    Vector of shape `(m,)` with indices of data points that constitute the
    vertices of a simplex.
    """
    n, m = X.shape

    # Sanity check.
    if n < m:
        raise ValueError(
            f"The Schur vector matrix of shape {X.shape} has more columns than rows. "
            f"You can't get a {m}-dimensional simplex from {n} data vectors."
        )
    # Check if the first, and only the first eigenvector is constant.
    diffs = np.abs(np.max(X, axis=0) - np.min(X, axis=0))
    if not np.isclose(1.0 + diffs[0], 1.0, rtol=1e-6):
        raise ValueError(
            f"First Schur vector is not constant 1. This indicates that the Schur vectors "
            f"are incorrectly sorted. Cannot search for a simplex structure in the data. The largest deviation from 1 "
            f"is {diffs[0]}."
        )
    if not np.all(diffs[1:] > 1e-6):
        which = np.sum(diffs[1:] <= 1e-6)
        raise ValueError(
            f"{which} Schur vector(s) after the first one are constant. Probably the Schur vectors "
            "are incorrectly sorted. Cannot search for a simplex structure in the data."
        )

    # local copy of the eigenvectors
    ortho_sys = np.copy(X)

    index = np.zeros(m, dtype=np.int64)
    max_dist = 0.0

    # First vertex: row with largest norm.
    for i in range(n):
        dist = np.linalg.norm(ortho_sys[i, :])
        if dist > max_dist:
            max_dist = dist
            index[0] = i

    # Translate coordinates to make the first vertex the origin.
    ortho_sys -= np.ones((n, 1)).dot(ortho_sys[index[0], np.newaxis])
    # Would be shorter, but less readable: ortho_sys -= X[index[0], np.newaxis]

    # All further vertices as rows with maximum distance to existing subspace.
    for j in range(1, m):
        max_dist = 0.0
        temp = np.copy(ortho_sys[index[j - 1], :])
        for i in range(n):
            sclprod = ortho_sys[i, :].dot(temp)
            ortho_sys[i, :] -= sclprod * temp
            distt = np.linalg.norm(ortho_sys[i, :])
            if distt > max_dist:  # and i not in index[0:j]: #in _pcca_connected_isa() of pcca.py
                max_dist = distt
                index[j] = i
        ortho_sys /= max_dist

    return index


@d.dedent
def _objective(alpha: np.ndarray, X: np.ndarray) -> float:
    """
    Compute objective function value.

    Parameters
    ----------
    alpha
        Vector of shape `((m - 1) ^ 2,)` containing the flattened and
        cropped rotation matrix ``rot_matrix[1:, 1:]``.
    X
        %(Q_sort)s

    Returns
    -------
    Current value of the objective function :math:`f = m - trace(S)`
    (Eq. 16 from [Roeblitz13]_).
    """
    # Dimensions.
    n, m = X.shape
    k = m - 1

    # Initialize rotation matrix.
    rot_mat = np.zeros((m, m), dtype=np.float64)

    # Sanity checks.
    if alpha.shape[0] != k ** 2:
        raise ValueError(
            "The shape of alpha doesn't match with the shape of X: "
            f"It is not a ({k}^2,)-vector, but of dimension {alpha.shape}. X is of shape `{X.shape}`."
        )

    # Now reshape alpha into a (k,k)-matrix.
    rot_crop_matrix = np.reshape(alpha, (k, k))

    # Complete rot_mat to meet constraints (positivity, partition of unity).
    rot_mat[1:, 1:] = rot_crop_matrix
    rot_mat = _fill_matrix(rot_mat, X)

    # Compute value of the objective function.
    # from Matlab: optval = m - trace( diag(1 ./ A(1,:)) * (A' * A) )
    return m - np.trace(np.diag(1.0 / rot_mat[0, :]).dot(rot_mat.conj().T.dot(rot_mat)))  # type: ignore[no-any-return]


@d.dedent
def _opt_soft(X: np.ndarray, rot_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""
    Optimize the G-PCCA rotation matrix such that the memberships are
    exclusively non-negative and compute the membership matrix.

    Parameters
    ----------
    X
        %(Q_sort)s
    rot_matrix
        Initial (non-optimized) rotation matrix of shape `(m, m)`.

    Returns
    -------
    Triple of the following:

    rot_matrix
        %(rot_matrix_ret)s
    chi
        %(chi_ret)s
    fopt
        Optimal value of the objective function :math:`f_{opt} = m - \\mathtt{trace}(S)`
        (Eq. 16 from [Roeblitz13]_).
    """  # noqa: D205, D400
    n, m = X.shape

    # Sanity checks.
    if not (rot_matrix.shape[0] == rot_matrix.shape[1]):
        raise ValueError("Rotation matrix isn't quadratic.")
    if not (rot_matrix.shape[0] == m):
        raise ValueError("The dimensions of the rotation matrix don't match with the number of Schur vectors.")
    if rot_matrix.shape[0] < 2:
        raise ValueError(f"Expected the rotation matrix to be at least of shape (2, 2), found {rot_matrix.shape}.")

    # Reduce optimization problem to size (m-1)^2 by cropping the first row and first column from rot_matrix
    rot_crop_matrix = rot_matrix[1:, 1:]

    # Now reshape rot_crop_matrix into a linear vector alpha.
    k = m - 1
    alpha = np.reshape(rot_crop_matrix, k ** 2)
    # TODO: Implement Gauss Newton Optimization to speed things up esp. for m > 10
    alpha, fopt, _, _, _ = fmin(_objective, alpha, args=(X,), full_output=True, disp=False)

    # Now reshape alpha into a (k,k)-matrix.
    rot_crop_matrix = np.reshape(alpha, (k, k))

    # Complete rot_mat to meet constraints (positivity, partition of unity).
    rot_matrix[1:, 1:] = rot_crop_matrix
    rot_matrix = _fill_matrix(rot_matrix, X)

    # Compute the membership matrix.
    chi = np.dot(X, rot_matrix)

    # Check for negative elements in chi and handle them.
    if np.any(chi < 0.0):
        if np.any(chi < -1e4 * EPS):
            min_el = np.min(chi)
            raise ValueError(f"Some elements of chi are significantly negative. The minimal element in chi is {min_el}")
        else:
            chi[chi < 0.0] = 0.0
            chi = np.true_divide(1.0, np.sum(chi, axis=1))[:, np.newaxis] * chi
            if not np.allclose(np.sum(chi, axis=1), 1.0, atol=1e-8, rtol=1e-5):
                dev = np.max(np.abs(np.sum(chi, axis=1) - 1.0))
                raise ValueError(
                    f"The rows of chi don't sum up to 1.0 after rescaling. Maximum deviation from 1 is {dev}"
                )

    return rot_matrix, chi, fopt


@d.dedent
def _fill_matrix(rot_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Make the rotation matrix feasible.

    Parameters
    ----------
    rot_matrix
        (Infeasible) rotation matrix of shape `(m, m)`.
    X
        %(Q_sort)s

    Returns
    -------
    Feasible rotation matrix of shape `(m, m)`.
    """
    n, m = X.shape

    # Sanity checks.
    if not (rot_matrix.shape[0] == rot_matrix.shape[1]):
        raise ValueError("Rotation matrix isn't quadratic.")
    if not (rot_matrix.shape[0] == m):
        raise ValueError("The dimensions of the rotation matrix don't match with the number of Schur vectors.")

    # Compute first column of rot_mat by row sum condition.
    rot_matrix[1:, 0] = -np.sum(rot_matrix[1:, 1:], axis=1)

    # Compute first row of A by maximum condition.
    dummy = -np.dot(X[:, 1:], rot_matrix[1:, :])
    rot_matrix[0, :] = np.max(dummy, axis=0)

    # Reskale rot_mat to be in the feasible set.
    rot_matrix = rot_matrix / np.sum(rot_matrix[0, :])

    # Make sure, that there are no zero or negative elements in the first row of A.
    if np.any(rot_matrix[0, :] == 0):
        raise ValueError("First row of rotation matrix has elements = 0.")
    if np.min(rot_matrix[0, :]) < 0:
        raise ValueError("First row of rotation matrix has elements < 0.")

    return rot_matrix


@d.dedent
def _cluster_by_isa(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Classification of dynamical data based on `m` orthonormal Schur vectors of the (row-stochastic) transition matrix.

    Hereby `m` determines the number of clusters to cluster the data into.
    The applied method is the Inner Simplex Algorithm (ISA).
    Constraint: The Schur vector matrix `X` matrix needs to contain at
    least `m` Schur vectors.

    This function assumes that the state space is fully connected.

    Parameters
    ----------
    X
        %(Q_sort)s

    Returns
    -------
    Tuple of the following:

    chi
        %(chi_ret)s
    minChi
        minChi indicator, see [Roeblitz13]_ and [Reuter18]_.
    """
    # compute rotation matrix
    rot_matrix = _initialize_rot_matrix(X)

    # Compute the membership matrix.
    chi = np.dot(X, rot_matrix)

    # compute the minChi indicator
    minChi = np.amin(chi)

    return chi, minChi


@d.dedent
def _gpcca_core(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""
    Core of the G-PCCA [Reuter18]_ spectral clustering method with optimized memberships.

    Clusters the dominant `m` Schur vectors of a transition matrix.
    This algorithm generates a fuzzy clustering such that the resulting
    membership functions are as crisp (characteristic) as possible.

    Parameters
    ----------
    X
        %(Q_sort)s

    Returns
    -------
    Triple of the following:

    chi
        %(chi_ret)s
    rot_matrix
        %(rot_matrix_ret)s
    crispness
        %(crispness_ret)s
    """
    m = np.shape(X)[1]

    rot_matrix = _initialize_rot_matrix(X)

    rot_matrix, chi, fopt = _opt_soft(X, rot_matrix)

    # calculate crispness of the decomposition of the state space into m clusters
    crispness = (m - fopt) / m

    return chi, rot_matrix, crispness


@d.dedent
def _coarsegrain(P: Union[np.ndarray, spmatrix], eta: np.ndarray, chi: np.ndarray) -> np.ndarray:
    r"""
    Coarse-grain `P` such that the (dominant) Perron eigenvalues are preserved.

    Uses:

    ..math: P_c = (\chi^T D \chi)^{-1} (\chi^T D P \chi)

    with :math:`D` being a diagonal matrix with `eta` on its diagonal [Reuter18]_.

    Parameters
    ----------
    %(P)s
    %(eta)s
    chi
        %(chi_ret)s

    Returns
    -------
    The coarse-grained row-stochastic transition matrix.
    """
    # Matlab: Pc = pinv(chi'*diag(eta)*chi)*(chi'*diag(eta)*P*chi)

    # need to make sure here that memory does not explode, and P is never densified
    W = np.linalg.pinv(chi.T.dot(chi * eta[:, None]))
    V = chi.T * eta
    if issparse(P):
        V = sp.csr_matrix(V)
    A = V.dot(P).dot(chi)

    return W.dot(A)


@d.dedent
def gpcca_coarsegrain(
    P: Union[np.ndarray, spmatrix],
    m: Union[int, Tuple[int, int], List[int], Dict[str, int]],
    eta: Optional[np.ndarray] = None,
    z: str = "LM",
    method: str = DEFAULT_SCHUR_METHOD,
) -> np.ndarray:
    r"""
    Coarse-grain the transition matrix `P` into `m` sets using G-PCCA [Reuter18]_.

    Performs optimized spectral clustering via G-PCCA and coarse-grains `P`
    such that the dominant Perron eigenvalues are preserved using:

    .. math:: P_c = (\chi^T D \chi)^{-1} (\chi^T D P \chi)

    with :math:`D` being a diagonal matrix with `eta` on its diagonal [Reuter18]_.

    Parameters
    ----------
    %(P)s
    %(m_optimize)s
    %(eta)s
        If `None` (default), uniform distribution is used.
    %(z_P)s
    %(method)s
        See the `installation <https://pygpcca.readthedocs.io/en/latest/installation.html>`_ instructions
        for more information.

    Returns
    -------
    The coarse-grained row-stochastic transition matrix.

    References
    ----------
    If you use this code or parts of it, please cite [Reuter18]_.
    """
    # Matlab: Pc = pinv(chi'*diag(eta)*chi)*(chi'*diag(eta)*P*chi)
    chi = GPCCA(P, eta=eta, z=z, method=method).optimize(m).memberships

    return _coarsegrain(P, eta=eta, chi=chi)


@d.dedent
class GPCCA:
    """
    G-PCCA [Reuter18]_ spectral clustering method with optimized memberships.

    Clusters the dominant `m` Schur vectors of a transition matrix.

    This algorithm generates a fuzzy clustering such that the resulting
    membership functions are as crisp (characteristic) as possible.

    Parameters
    ----------
    %(P)s
    %(eta)s
        If `None`, uniform distribution is used.
    %(z_P)s
    %(method)s
        See the `installation <https://pygpcca.readthedocs.io/en/latest/installation.html>`_ instructions
        for more information.

    References
    ----------
    If you use this code or parts of it, please cite [Reuter18]_.
    """

    def __init__(
        self,
        P: Union[np.ndarray, spmatrix],
        eta: Optional[np.ndarray] = None,
        z: str = "LM",
        method: str = DEFAULT_SCHUR_METHOD,
    ):
        if not is_transition_matrix(P):
            raise ValueError("Input matrix P is not a transition matrix.")
        if z not in ["LM", "LR"]:
            raise ValueError("You didn't give a valid sorting criterion z. Valid options are: 'LM', 'LR'.")
        if method not in ["brandts", "krylov"]:
            raise ValueError(
                "You didn't give a valid method to determine the invariant subspace. "
                "Valid options are: 'brandts', 'krylov'."
            )

        n = np.shape(P)[0]
        if eta is None:
            eta = np.true_divide(np.ones(P.shape[0]), P.shape[0])
        if len(eta) != n:
            raise ValueError(f"eta vector length ({len(eta)}) doesn't match with the shape of " f"P[{n}, {n}].")

        self._P = P.astype(np.float64)
        self._eta: np.ndarray = eta.astype(np.float64)
        self._z: str = z
        self._method: str = method

        # _p stands for precomputed
        self._p_X: OArray = None
        self._p_R: OArray = None
        self._p_eigenvalues: OArray = None
        # these are the actual values, accessed by the properties
        self._X: OArray = None
        self._R: OArray = None
        self._eigenvalues: OArray = None
        self._top_eigenvalues: OArray = None

        self._m_opt: Optional[int] = None
        self._chi: OArray = None
        self._rot_matrix: OArray = None
        self._crispness_opt: Optional[float] = None
        self._crispness: OArray = None

        self._pi: OArray = None
        self._pi_coarse: OArray = None
        self._eta_coarse: OArray = None
        self._P_coarse: OArray = None

    def _do_schur_helper(self, m: int) -> None:
        n = np.shape(self._P)[0]
        if self._p_X is not None and self._p_R is not None and self._p_eigenvalues is not None:
            Xdim1, Xdim2 = self._p_X.shape
            Rdim1, Rdim2 = self._p_R.shape
            if Xdim1 != n:
                raise ValueError(
                    f"The first dimension of X is `{Xdim1}`. This doesn't match with the dimension of P[{n}, {n}]."
                )
            if Rdim1 != Rdim2:
                raise ValueError("The Schur form R is not quadratic.")
            if Xdim2 != Rdim1:
                raise ValueError(
                    f"The first dimension of X is `{Xdim1}`. "
                    f"This doesn't match with the dimension of R[{Rdim1}, {Rdim2}]."
                )
            if Rdim2 < m:
                self._p_X, self._p_R, self._p_eigenvalues = _do_schur(
                    self._P, eta=self._eta, m=m, z=self._z, method=self._method
                )
            else:
                # if we are using pre-computed decomposition, check splitting
                if m < n:
                    if len(self._p_eigenvalues) < m:
                        raise ValueError(
                            f"Can't check complex conjugate block splitting for {m} clusters with only "
                            f"{len(self._p_eigenvalues)} eigenvalues."
                        )
                    else:
                        if _check_conj_split(self._p_eigenvalues[:m]):
                            raise ValueError(
                                f"Clustering into {m} clusters will split complex conjugate eigenvalues. "
                                f"Request one cluster more or less."
                            )
                        logging.info("Using pre-computed Schur decomposition")
        else:
            self._p_X, self._p_R, self._p_eigenvalues = _do_schur(
                self._P, eta=self._eta, m=m, z=self._z, method=self._method
            )

    def minChi(self, m_min: int, m_max: int) -> List[float]:
        r"""
        Calculate the minChi indicator (see [Reuter18]_) for every :math:`m \in [m_{min},m_{max}]`.

        The minChi indicator can be used to determine an interval
        :math:`I \subset [m_{min},m_{max}]` of good (potentially optimal)
        numbers of clusters.

        Afterwards either one :math:`m \in I`(with maximal `minChi`) or the
        whole interval :math:`I` is chosen as input to :meth:`optimize`
        for further optimization.

        Parameters
        ----------
        m_min
            Minimal number of clusters to group into.
        m_max
            Maximal number of clusters to group into.

        Returns
        -------
        List of minChi indicators for cluster numbers :math:`m \in [m_{min},m_{max}]`,
        see [Roeblitz13]_, [Reuter18]_.
        """
        # Validate Input.
        if m_min >= m_max:
            raise ValueError(f"m_min ({m_min}) must be smaller than m_max ({m_max}).")
        if m_min in [0, 1]:
            raise ValueError(f"There is no point in clustering into `{m_min}` clusters.")

        # Calculate Schur matrix R and Schur vector matrix X, if not adequately given.
        self._do_schur_helper(m_max)
        if TYPE_CHECKING:
            assert isinstance(self._p_X, np.ndarray)

        minChi_list: List[float] = []
        for m in range(m_min, m_max + 1):
            # Xm = np.copy(X[:, :m])
            _, minChi = _cluster_by_isa(self._p_X[:, :m])
            minChi_list.append(minChi)

        return minChi_list

    # G-PCCA coarse-graining
    @d.dedent
    def optimize(
        self,
        m: Union[int, Tuple[int, int], List[int], Dict[str, int]],
    ) -> "GPCCA":
        r"""
        Full G-PCCA [Reuter18]_ spectral clustering method with optimized memberships.

        It also has the option to optimize the number of clusters
        (macrostates) `m` as well.

        If a single integer `m` is given, the method clusters the dominant
        `m` Schur vectors of the :attr:`transition_matrix`.
        The algorithm generates a fuzzy clustering such that the resulting
        membership functions `chi` are as crisp (characteristic) as
        possible, given `m`.

        Instead of a single number of clusters `m`, a :class:`tuple`
        or a :class:`dict` ``{'m_min': int, 'm_max': int}``
        containing a minimum and a maximum number of clusters can be given.
        This results in repeated execution of the G-PCCA core algorithm
        for :math:`m \in [m_{min},m_{max}]`. Among the resulting clusterings,
        the sharpest/crispest one (with maximal `crispness`) will be selected.

        Parameters
        ----------
        %(m_optimize)s

            See :meth:`minChi` for selection of good (potentially optimal)
            number of clusters.

        Returns
        -------
        Returns self and updates the following attributes:




            - :attr:`coarse_grained_input_distribution`
            - :attr:`coarse_grained_stationary_distribution`
            - :attr:`coarse_grained_transition_matrix`
            - :attr:`crispness_values`
            - :attr:`dominant_eigenvalues`
            - :attr:`input_distribution`
            - :attr:`macrostate_assignment`
            - :attr:`macrostate_sets`
            - :attr:`memberships`
            - :attr:`n_m`
            - :attr:`optimal_crispness`
            - :attr:`rotation_matrix`
            - :attr:`schur_matrix`
            - :attr:`schur_vectors`
            - :attr:`stationary_probability`
            - :attr:`top_eigenvalues`
            - :attr:`transition_matrix`
        """
        n = self._P.shape[0]

        # extract m_min, m_max, if given, else take single m
        if isinstance(m, (tuple, list)):
            if len(m) != 2:
                raise ValueError(f"Expected range to be of size 2, found `{len(m)}`.")
            m_list = m
            if m[0] >= m[1]:
                raise ValueError(f"m_min ({m[0]}) must be smaller than m_max ({m[1]}).")
        elif isinstance(m, dict):
            m_min = m["m_min"]
            m_max = m["m_max"]
            if m_min >= m_max:
                raise ValueError(f"m_min ({m_min}) must be smaller than m_max ({m_max}).")
            m_list = [m_min, m_max]
        elif isinstance(m, int):
            m_list = [m]
        else:
            raise TypeError(f"Invalid type `{type(m).__name__}`.")

        # validate input
        if max(m_list) > n:
            raise ValueError(
                f"Number of macrostates `({max(m_list)})` exceeds number "
                f"of states of the transition matrix `({n})`."
            )
        if min(m_list) in [0, 1]:
            raise ValueError(f"There is no point in clustering into `{m}` clusters.")

        # The following code enclosed by >>>... ...<<< originates (with some adjustments) from MSMTools
        # Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER).
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # test connectivity
        components = connected_sets(self._P)
        n_components = len(components)
        # Store components as closed (with positive equilibrium distribution)
        # or as transition states (with vanishing equilibrium distribution).
        closed_components = []
        for i in range(n_components):
            component = components[i]
            rest = list(set(range(n)) - set(component))
            # is component closed?
            if np.sum(self._P[component, :][:, rest]) == 0:
                closed_components.append(component)
        n_closed_components = len(closed_components)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Calculate Schur matrix R and Schur vector matrix X, if not adequately given.
        self._do_schur_helper(max(m_list))

        if TYPE_CHECKING:
            assert isinstance(self._p_X, np.ndarray)
            assert isinstance(self._p_R, np.ndarray)
            assert isinstance(self._p_eigenvalues, np.ndarray)

        # Initialize lists to collect results.
        chi_list: List[np.ndarray] = []
        rot_matrix_list: List[np.ndarray] = []
        crispness_list: List[float] = []
        # Iterate over m
        for m in range(min(m_list), max(m_list) + 1):
            if len(self._p_eigenvalues) < m:
                raise ValueError(
                    f"Can't check complex conjugate block splitting for {m} clusters with only "
                    f"{len(self._p_eigenvalues)} eigenvalues."
                )
            if _check_conj_split(self._p_eigenvalues[:m]):
                warnings.warn(
                    f"Clustering into {m} clusters will split complex conjugate eigenvalues. "
                    f"Skipping clustering into {m} clusters."
                )
                crispness_list.append(0.0)
                chi_list.append(np.zeros((n, m)))
                rot_matrix_list.append(np.zeros((m, m)))
                continue

            # Reduce X according to m and make a work copy.
            # Xm = np.copy(X[:, :m])
            chi, rot_matrix, crispness = _gpcca_core(self._p_X[:, :m])
            # check if we have at least m dominant sets. If less than m, we warn.
            nmeta = np.count_nonzero(chi.sum(axis=0))
            if m > nmeta:
                crispness_list.append(-crispness)
                warnings.warn(
                    f"`{m}` macrostates requested, but transition matrix only has "
                    f"`{nmeta}` macrostates. Request less macrostates."
                )
            # Check, if we have enough clusters to support the disconnected sets.
            elif m < n_closed_components:
                crispness_list.append(-crispness)
                warnings.warn(
                    f"Number of macrostates `({m})` is too small. "
                    f"Transition matrix has `{n_closed_components}` disconnected components."
                )
            else:
                crispness_list.append(crispness)
            chi_list.append(chi)
            rot_matrix_list.append(rot_matrix)

        if np.any(np.array(crispness_list) > 0.0):
            if len(m_list) > 1 and max(m_list) == n:
                warnings.warn(
                    f"Clustering {n} data points into {max(m_list)} clusters is always perfectly crisp. "
                    f"Thus m={max(m_list)} won't be included in the search for the optimal cluster number."
                )
                opt_idx = int(np.argmax(crispness_list[:-1]))
            else:
                opt_idx = int(np.argmax(crispness_list))
        else:
            raise ValueError("Clustering wasn't successful. Try different cluster numbers.")
        self._m_opt = min(m_list) + opt_idx
        self._chi = chi_list[opt_idx]
        self._rot_matrix = rot_matrix_list[opt_idx]
        self._crispness = np.array(crispness_list)
        self._crispness_opt = crispness_list[opt_idx]
        self._X = self._p_X[:, : self._m_opt]
        self._R = self._p_R[: self._m_opt, : self._m_opt]
        self._top_eigenvalues = self._p_eigenvalues[: self._m_opt]
        self._eigenvalues = self._p_eigenvalues[: max(m_list)]

        if TYPE_CHECKING:
            assert isinstance(self.memberships, np.ndarray)

        # coarse-grained stationary distribution
        self._pi_coarse = (
            None if self.stationary_probability is None else np.dot(self.memberships.T, self.stationary_probability)
        )
        # coarse-grained input (initial) distribution of states
        self._eta_coarse = np.dot(self.memberships.T, self.input_distribution)
        # coarse-grain transition matrix
        self._P_coarse = _coarsegrain(self.transition_matrix, eta=self.input_distribution, chi=self.memberships)

        return self

    @property
    def coarse_grained_input_distribution(self) -> OArray:
        r"""
        Coarse grained input distribution of shape `(n_m,)`.

        .. math:: \eta_c = \chi^T \eta
        """
        return self._eta_coarse

    @property
    def coarse_grained_stationary_probability(self) -> OArray:
        r"""
        Coarse grained stationary distribution of shape `(n_m,)`.

        .. math:: \pi_c = \chi^T \pi
        """
        return self._pi_coarse

    @property
    def coarse_grained_transition_matrix(self) -> OArray:
        r"""
        Coarse grained transition matrix of shape `(n_m, n_m)`.

        .. math:: P_c = (\chi^T D \chi)^{-1} (\chi^T D P \chi)

        with :math:`D` being a diagonal matrix with :math:`\eta` on its diagonal.
        """
        return self._P_coarse

    @property  # type: ignore[misc]
    @d.dedent
    def crispness_values(self) -> OArray:
        """
        Vector of crispness values for clustering into the requested cluster numbers.

        %(crispness_ret)s
        """
        return self._crispness

    @property  # type: ignore[misc]
    @d.dedent
    def dominant_eigenvalues(self) -> OArray:
        """
        Dominant :attr:`n_m` eigenvalues of `P`.

        Vector of shape `(n_m,)` containing the `n_m` dominant eigenvalues of `P`.
        """
        return self._top_eigenvalues

    @property
    def input_distribution(self) -> np.ndarray:
        r"""
        Input probability distribution of the (micro)states.

        In theory :math:`\eta` can be an arbitrary distribution as long as it is
        a valid probability distribution (i.e., sums up to 1).
        A neutral and valid choice would be the uniform distribution (default).

        In case of a reversible transition matrix, the stationary distribution
        :math:`\pi` can (but don't has to) be used here.
        In case of a non-reversible `P`, some initial or average distribution of
        the states might be chosen instead of the uniform distribution.

        Vector of shape `(n,)` which sums to 1.
        """
        return self._eta

    @property
    def macrostate_assignment(self) -> OArray:
        """
        Crisp clustering using G-PCCA.

        This is recommended only for visualization purposes.
        You *cannot* compute any actual quantity of the coarse-grained
        kinetics without employing the fuzzy memberships!

        Returns
        -------
        Integer vector of shape `(n,)` containing the macrostate
        each microstate is located in.

        Credits
        -------
        The code and docstring of this property origins (with some adjustments) from MSMTools,
        Copyright (c) 2015, 2014 Computational Molecular Biology Group,
        Freie Universitaet Berlin (GER).
        """
        return None if self.memberships is None else np.argmax(self.memberships, axis=1)  # type: ignore[return-value]

    @property
    def macrostate_sets(self) -> Optional[List[np.ndarray]]:
        """
        Crisp clustering using G-PCCA.

        This is recommended only for visualization purposes.
        You *cannot* compute any actual quantity of the coarse-grained
        kinetics without employing the fuzzy memberships!

        Returns
        -------
        A list of length equal to :attr:`n_m`.

        Each element is an array with microstate indexes contained in it.

        Credits
        -------
        The code and docstring of this property origins (with some adjustments) from MSMTools,
        Copyright (c) 2015, 2014 Computational Molecular Biology Group,
        Freie Universitaet Berlin (GER).
        """
        return (
            None
            if self.macrostate_assignment is None or self.n_m is None
            else [np.where(self.macrostate_assignment == i)[0] for i in range(self.n_m)]
        )

    @property
    def memberships(self) -> OArray:
        r"""
        Array of shape `(n, n_m)` containing the membership :math:`\chi_{ij}` (or probability)
        of each microstate :math:`i` (to be assigned) to each macrostate or cluster :math:`j`.

        The rows sum to 1.
        """  # noqa: D205, D400
        return self._chi

    @property
    def n_m(self) -> Optional[int]:
        """Optimal number of clusters or macrostates to group the `n` microstates into."""
        return self._m_opt

    @property  # type: ignore[misc]
    @d.dedent
    def optimal_crispness(self) -> Optional[float]:
        """
        Crispness for clustering into :attr:`n_m` clusters.

        %(crispness_ret)s
        """
        return self._crispness_opt

    @property
    def rotation_matrix(self) -> OArray:
        r"""
        Optimized rotation matrix :math:`A`.

        Array of shape `(n_m, n_m)` which rotates the dominant Schur vectors
        to yield the G-PCCA :attr:`memberships`, i.e. :math:`\chi = X A`.
        """
        return self._rot_matrix

    @property
    def schur_matrix(self) -> OArray:
        r"""
        Ordered top left part of shape `(n_m, n_m)` of the real Schur matrix of :math:`P`.

        The ordered real partial Schur matrix :math:`R` of :math:`P` fulfills

        .. math:: P Q = Q R

        with the ordered matrix of dominant Schur vectors :math:`Q`.
        """
        return self._R

    @property
    def schur_vectors(self) -> OArray:
        r"""
        Array :math:`Q` of shape `(n, n_m)` with `n_m` sorted Schur vectors in the columns.

        The constant Schur vector is in the first column.
        """
        return self._X

    @cached_property
    def stationary_probability(self) -> OArray:
        r"""
        Stationary probability distribution :math:`\pi` of the microstates.

        Vector of shape `(n,)` which sums to 1.
        """
        try:
            return stationary_distribution(self._P)
        except Exception as e:  # noqa: B902
            warnings.warn(f"Stationary distribution couldn't be calculated. Reason: {e}.")
        return None

    @property
    def top_eigenvalues(self) -> OArray:
        """
        Top `m` respective `m_max` eigenvalues of `P`.

        If a single integer `m` was given, the upper `m` eigenvalues are returned.

        If a :class:`tuple` or :class:`dict` containing a minimum `m_min` and maximum number
        `m_max` of clusters was given, the upper `m_max` eigenvalues are returned.
        """
        return self._eigenvalues

    @property
    def transition_matrix(self) -> Union[np.ndarray, spmatrix]:
        """Row-stochastic transition matrix `P`."""
        return self._P

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[n={self.transition_matrix.shape[0]}, n_macrostates={self.n_m}]"

    def __str__(self) -> str:
        return repr(self)
