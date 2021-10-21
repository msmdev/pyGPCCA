# This file is part of pyGPCCA.
#
# The code and documentation of the functions below origins (with some adjustments) from MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from typing import List, Tuple, Union
from functools import singledispatch

from scipy.linalg import eig, lu_solve, lu_factor
from scipy.sparse import csgraph, spmatrix, csr_matrix, isspmatrix_csr
from scipy.sparse.linalg import eigs
import numpy as np

from pygpcca.utils._docs import d
from pygpcca.utils._checks import ensure_ndarray_or_sparse
from pygpcca.utils._constants import EPS

__all__ = [
    "connected_sets",
    "is_transition_matrix",
    "stationary_distribution",
]


@singledispatch
def connected_sets(C: Union[np.ndarray, spmatrix], directed: bool = True) -> List[np.ndarray]:
    """
    Compute connected sets of microstates.

    Connected components for a directed graph with edge-weights
    given by the count matrix.

    Parameters
    ----------
    C
        Count matrix specifying edge weights.
    directed
        Whether to compute connected components for a directed or undirected graph.

    Returns
    -------
    Each entry is an array containing all vertices (states) in the corresponding connected component. The list is sorted
    according to the size of the individual components. The largest connected set is the first entry in the list.

    Notes
    -----
    Viewing the count matrix as the adjacency matrix of a (directed) graph the connected components are given by the
    connected components of that graph. Connected components of a graph can be efficiently computed using
    Tarjan's algorithm [1]_.

    References
    ----------
    .. [1] Tarjan, R E. 1972. Depth-first search and linear graph
        algorithms. SIAM Journal on Computing 1 (2): 146-160.

    Credits
    -------
    The code and docstring of this function origins (with some adjustments) from MSMTools,
    Copyright (c) 2015, 2014 Computational Molecular Biology Group,
    Freie Universitaet Berlin (GER).
    """
    raise NotImplementedError(type(C))


@connected_sets.register(np.ndarray)
def _csd(C: np.ndarray, directed: bool = True) -> List[np.ndarray]:
    return connected_sets(csr_matrix(C), directed=directed)


@connected_sets.register(spmatrix)
def _css(C: spmatrix, directed: bool = True) -> List[np.ndarray]:
    if not isspmatrix_csr(C):
        C = csr_matrix(C)

    M = C.shape[0]
    # compute connected components of C. nc is the number of components,
    # indices contain the component labels of the states
    nc, indices = csgraph.connected_components(C, directed=directed, connection="strong")

    # discrete states
    states = np.arange(M)

    # order indices
    ind = np.argsort(indices)
    indices = indices[ind]

    # order states
    states = states[ind]
    # the state index tuple is now of the following form (states, indices)=
    # ([s_23, s_17,...,s_3, s_2, ...], [0, 0, ..., 1, 1, ...])

    # find number of states per component
    count = np.bincount(indices)

    # cumulative sum of count gives start and end indices of components
    csum = np.zeros(len(count) + 1, dtype=np.uint32)
    csum[1:] = np.cumsum(count)

    # generate list containing components, sort each component by increasing state label
    cc = [np.sort(states[csum[i] : csum[i + 1]]) for i in range(nc)]

    # sort by size of component - largest component first
    return sorted(cc, key=lambda x: -len(x))


@singledispatch
def is_transition_matrix(T: Union[np.ndarray, spmatrix], tol: float = 1e-12) -> bool:
    r"""
    Check if the given matrix is a transition matrix.

    Parameters
    ----------
    T
        Matrix to check.
    tol
        Floating point tolerance to check with.

    Returns
    -------
    True, if ``T`` is a valid transition matrix, false otherwise.

    Notes
    -----
    A valid transition matrix :math:`P=(p_{ij})` has non-negative elements, :math:`p_{ij} \geq 0`, and elements of each
    row sum up to one, :math:`\sum_j p_{ij} = 1`. Matrices wit this property are also called stochastic matrices.

    Credits
    -------
    The code and docstring of this function origins (with some adjustments) from MSMTools,
    Copyright (c) 2015, 2014 Computational Molecular Biology Group,
    Freie Universitaet Berlin (GER).
    """
    raise NotImplementedError(type(T))


@is_transition_matrix.register(spmatrix)
def _itmd(T: spmatrix, tol: float = 1e-12) -> bool:
    T = ensure_ndarray_or_sparse(T, ndim=2, uniform=True, kind="numeric")

    if not isspmatrix_csr(T):
        T = csr_matrix(T)  # compressed sparse row for fast row slicing
    values = T.data  # non-zero entries of T

    # check entry-wise positivity
    is_positive: bool = np.allclose(values, np.abs(values), rtol=tol)

    # check row normalization
    is_normed: bool = np.allclose(T.sum(axis=1), 1.0, rtol=tol)

    return is_positive and is_normed


@is_transition_matrix.register(np.ndarray)
def _itms(T: np.ndarray, tol: float = 1e-12) -> bool:
    T = ensure_ndarray_or_sparse(T, ndim=2, uniform=True, kind="numeric")

    dim = T.shape[0]
    X = np.abs(T) - T
    x = np.sum(T, axis=1)

    return X.max() < 2.0 * tol and np.abs(x - np.ones(dim)).max() < dim * tol  # type: ignore[no-any-return]


@singledispatch
@d.dedent
def stationary_distribution(P: Union[np.ndarray, spmatrix]) -> np.ndarray:
    r"""
    Compute stationary distribution of stochastic matrix `P`.

    Parameters
    ----------
    %(P)s

    Returns
    -------
    Vector of stationary probabilities.

    Notes
    -----
    The stationary distribution :math:`\pi` is the left eigenvector corresponding to the non-degenerate eigenvalue
    :math:`\lambda=1` of a reversible transition matrix :math:`P`,

    .. math:: \pi^T P =\pi^T.

    Credits
    -------
    The code and docstring of this function origins (with some adjustments) from MSMTools,
    Copyright (c) 2015, 2014 Computational Molecular Biology Group,
    Freie Universitaet Berlin (GER).
    """
    raise NotImplementedError(type(P))


@stationary_distribution.register(np.ndarray)
def _sdd(P: np.ndarray) -> np.ndarray:
    try:
        mu = stationary_distribution_from_backward_iteration(P)
        if np.any(mu < 0):  # numerical problem, fall back to more robust algorithm.
            raise RuntimeError("Encountered negative value.")
    except RuntimeError:
        mu = stationary_distribution_from_eigenvector(P)
        if np.any(mu < 0):  # still? Then set to 0 and renormalize
            mu = np.maximum(mu, 0.0)
            mu /= mu.sum()

    # check whether this really is a stationary distribution
    _is_stationary_distribution(P, mu)

    return mu


def _eigs_slepc(P: spmatrix, k: int, which: "str" = "LR", tol: float = EPS) -> Tuple[np.ndarray, np.ndarray]:
    from petsc4py import PETSc
    from slepc4py import SLEPc

    M = PETSc.Mat().create()
    if not isspmatrix_csr(P):
        P = csr_matrix(P)
    M.createAIJ(size=P.shape, csr=(P.indptr, P.indices, P.data))

    E = SLEPc.EPS()
    E.create()
    E.setOperators(M)
    E.setDimensions(k)
    E.setTolerances(tol=tol)
    if which == "LR":
        E.setWhichEigenpairs(E.Which.LARGEST_REAL)
    elif which == "LM":
        E.setWhichEigenpairs(E.Which.LARGEST_MAGNITUDE)
    else:
        raise NotImplementedError(f"`which={which}` is not implemented.")
    E.solve()

    nconv = E.getConverged()
    if nconv < k:
        raise ValueError(f"Requested `{k}` eigenvalues/vectors, but only `{nconv}` converged.")

    xr, _ = M.getVecs()
    xi, _ = M.getVecs()

    eigenvalues, eigenvectors = [], []
    for i in range(k):
        # Get the i-th eigenvalue as computed by solve().
        eigenvalues.append(E.getEigenpair(i, xr, xi))
        if eigenvalues[-1].imag != 0.0:
            eigenvectors.append([complex(r, i) for r, i in zip(xr.getArray(), xi.getArray())])
        else:
            eigenvectors.append(list(xr.getArray()))

    return np.asarray(eigenvalues), np.asarray(eigenvectors).T


@stationary_distribution.register(spmatrix)
def _sds(P: spmatrix) -> np.ndarray:
    # get the top two eigenvalues and vecs so we can check for irreducibility
    try:
        vals, vecs = _eigs_slepc(P.T, k=2, which="LR")
    except ImportError:
        vals, vecs = eigs(P.T, k=2, which="LR", ncv=None)

    # check for irreducibility
    if np.allclose(vals, 1, rtol=1e2 * EPS, atol=1e2 * EPS):
        second_largest = np.min(vals)
        raise ValueError(f"This matrix is reducible. The second largest eigenvalue is {second_largest}.")

    # sort by real part and take the top one
    p = np.argsort(vals.real)[::-1]
    vecs = vecs[:, p]
    top_vec = vecs[:, 0]

    # check for imaginary component
    imaginary_component = top_vec.imag
    if not np.allclose(imaginary_component, 0, rtol=EPS, atol=EPS):
        raise ValueError("Top eigenvector has imaginary component.")
    top_vec = top_vec.real

    # check the sign structure
    if not (top_vec > -1e4 * EPS).all() and not (top_vec < 1e4 * EPS).all():
        el_min, el_max = np.min(top_vec), np.max(top_vec)
        raise ValueError(f"Top eigenvector has both positive and negative entries. It has range = [{el_min}, {el_max}]")
    top_vec = np.abs(top_vec)
    pi = top_vec / np.sum(top_vec)

    # check whether this really is a stationary distribution
    _is_stationary_distribution(P, pi)

    # normalize to 1 and return
    return pi


def backward_iteration(A: np.ndarray, mu: float, x0: np.ndarray, tol: float = 1e-14, maxiter: int = 100) -> np.ndarray:
    """
    Find eigenvector to approximate eigenvalue via backward iteration.

    Parameters
    ----------
    A
        Matrix for which eigenvector is desired.
    mu
        Approximate eigenvalue for desired eigenvector.
    x0
        Initial guess for eigenvector.
    tol
        Tolerance parameter for termination of iteration.

    Returns
    -------
    Eigenvector to approximate eigenvalue ``mu``.

    Credits
    -------
    The code and docstring of this function origins (with some adjustments) from MSMTools,
    Copyright (c) 2015, 2014 Computational Molecular Biology Group,
    Freie Universitaet Berlin (GER).
    """
    T = A - mu * np.eye(A.shape[0])
    # LU-factor of T
    lupiv = lu_factor(T)
    # starting iterate with ||y_0||=1
    r0 = 1.0 / np.linalg.norm(x0)
    y0 = x0 * r0
    # local variables for inverse iteration
    y = 1.0 * y0
    r = 1.0 * r0
    for _ in range(maxiter):
        x = lu_solve(lupiv, y)
        r = 1.0 / np.linalg.norm(x)
        y = x * r
        if r <= tol:
            return y

    raise RuntimeError(f"Failed to converge after `{maxiter}` iterations, residuum is `{r}`.")


@d.dedent
def stationary_distribution_from_backward_iteration(P: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Fast computation of the stationary vector using backward iteration.

    Parameters
    ----------
    %(P)s
    eps
        Perturbation parameter for the true eigenvalue.

    Returns
    -------
    Stationary vector.

    Credits
    -------
    The code and docstring of this function origins (with some adjustments) from MSMTools,
    Copyright (c) 2015, 2014 Computational Molecular Biology Group,
    Freie Universitaet Berlin (GER).
    """
    A = np.transpose(P)
    mu = 1.0 - eps
    x0 = np.ones(P.shape[0])
    y = backward_iteration(A, mu, x0)

    return y / np.sum(y)


@d.dedent
def stationary_distribution_from_eigenvector(P: np.ndarray) -> np.ndarray:
    r"""
    Compute stationary distribution of stochastic matrix `P`.

    The stationary distribution is the left eigenvector corresponding to the
    non-degenerate eigenvalue :math:`\lambda=1`.

    Parameters
    ----------
    %(P)s

    Returns
    -------
    Vector of stationary probabilities.

    Credits
    -------
    The code and docstring of this function origins (with some adjustments) from MSMTools,
    Copyright (c) 2015, 2014 Computational Molecular Biology Group,
    Freie Universitaet Berlin (GER).
    """
    val, L = eig(P, left=True, right=False)

    # sorted eigenvalues and left and right eigenvectors
    perm = np.argsort(val)[::-1]

    L = L[:, perm]
    # make sure that stationary distribution is non-negative and l1-normalized
    nu = np.abs(L[:, 0])

    return nu / np.sum(nu)


def _is_stationary_distribution(T: Union[np.ndarray, spmatrix], pi: np.ndarray) -> bool:

    # check the shapes
    if not T.shape[0] == T.shape[1] or not T.shape[0] == pi.shape[0]:
        raise ValueError("Shape mismatch.")

    # check for invariance
    if not np.allclose(T.T.dot(pi), pi, rtol=1e6 * EPS, atol=1e6 * EPS):
        dev = np.max(np.abs(T.T.dot(pi) - pi))
        raise ValueError(
            f"Stationary distribution is not invariant under the transition matrix. Maximal deviation = " f"{dev}"
        )

    # check for positivity
    if not (pi > -1e4 * EPS).all():
        dev = np.min(pi)
        raise ValueError(f"Stationary distribution has negative elements. Minimal element = {dev}")

    # check whether it sums to one
    if not np.allclose(pi.sum(), 1, rtol=1e4 * EPS, atol=1e4 * EPS):
        dev = np.abs(pi.sum() - 1)
        raise ValueError(f"Stationary distribution doe not sum to one. Deviation = {dev}.")

    return True
