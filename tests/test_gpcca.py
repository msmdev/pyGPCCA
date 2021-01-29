# This file is part of pyGPCCA.
#
# Copyright (c) 2020 Bernhard Reuter.
# With contributions of Marius Lange and Michal Klein.
# Based on the original MATLAB GPCCA code authored by Bernhard Reuter, Susanna Roeblitz and Marcus Weber,
# Zuse Institute Berlin, Takustrasse 7, 14195 Berlin
# --------------------------------------------------
# If you use this code or parts of it, cite the following reference:
# ------------------------------------------------------------------
# Reuter, B., Weber, M., Fackeldey, K., Röblitz, S., & Garcia, M. E. (2018).
# Generalized Markov State Modeling Method for Nonequilibrium Biomolecular Dynamics:
# Exemplified on Amyloid β Conformational Dynamics Driven by an Oscillating Electric Field.
# Journal of Chemical Theory and Computation, 14(7), 3579–3594. https://doi.org/10.1021/acs.jctc.8b00079
# ----------------------------------------------------------------
# pyGPCCA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from typing import Optional
from operator import itemgetter
from itertools import combinations

import pytest

from scipy.linalg import lu, pinv, eigvals, hilbert, subspace_angles
from scipy.sparse import issparse, csr_matrix
import numpy as np

from pygpcca._gpcca import (
    GPCCA,
    _do_schur,
    _opt_soft,
    _objective,
    _gpcca_core,
    _fill_matrix,
    _indexsearch,
    _cluster_by_isa,
    _gram_schmidt_mod,
    gpcca_coarsegrain,
    _initialize_rot_matrix,
)
from tests.conftest import mu, assert_allclose, get_known_input, skip_if_no_petsc_slepc
from pygpcca._sort_real_schur import sort_real_schur

eps = np.finfo(np.float64).eps * 1e10


def _assert_schur(
    P: np.ndarray,
    X: np.ndarray,
    RR: np.ndarray,
    N: Optional[int] = None,
    subspace: bool = False,
):
    if N is not None:
        np.testing.assert_array_equal(P.shape, [N, N])
        np.testing.assert_array_equal(X.shape, [N, N])
        np.testing.assert_array_equal(RR.shape, [N, N])

    if subspace:
        assert_allclose(subspace_angles(P @ X, X @ RR), 0.0, atol=1e-6, rtol=1e-5)
    else:
        assert np.all(np.abs(X @ RR - P @ X) < eps), np.abs(X @ RR - P @ X).max()
    assert np.all(np.abs(X[:, 0] - 1) < eps), np.abs(X[:, 0]).max()


def _find_permutation(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    expected
        Array of shape ``(N, M).``
    actual
        Array of shape ``(N, M).``

    Returns
    -------
    :class:`numpy.ndarray`
        Array of shape ``(M,)``.
    """
    assert expected.shape == actual.shape
    perm = []
    temp = {i: expected[:, i] for i in range(expected.shape[1])}

    for a in actual.T:
        perm.append(
            min(
                ((ix, np.linalg.norm(a - e)) for ix, e in temp.items()),
                key=itemgetter(1),
            )[0]
        )
        temp.pop(perm[-1])

    return np.array(perm)


class TestGPCCAMatlabRegression:
    def test_empty_P(self):
        with pytest.raises(AssertionError, match=r"Expected shape 2 but given array has shape \d+"):
            GPCCA(np.array([]))

    def test_non_square_P(self):
        with pytest.raises(AssertionError, match=r"Given array is not uniform: \[\d+ \d+\]"):
            GPCCA(np.random.normal(size=(4, 3)))

    def test_empty_sd(self, P: np.ndarray):
        with pytest.raises(ValueError, match=r"eta vector length"):
            GPCCA(P, eta=[])

    def test_too_small_kkmin(self, P: np.ndarray, sd: np.ndarray):
        g = GPCCA(P, eta=sd)
        with pytest.raises(ValueError, match=r"There is no point in clustering into `0` clusters."):
            g.minChi(m_min=0, m_max=10)

    def test_k_input(self, P: np.ndarray, sd: np.ndarray):
        g = GPCCA(P, eta=sd)
        with pytest.raises(ValueError, match=r"m_min \(5\) must be smaller than m_max \(3\)."):
            g.minChi(m_min=5, m_max=3)

    def test_normal_case(
        self,
        P: np.ndarray,
        sd: np.ndarray,
        count_sd: np.ndarray,
        count_Pc: np.ndarray,
        count_chi: np.ndarray,
    ):
        assert_allclose(sd, count_sd)

        g = GPCCA(P, eta=sd)
        g.optimize((2, 10))

        Pc = g.coarse_grained_transition_matrix
        assert_allclose(Pc, count_Pc, atol=eps)

        assert_allclose(Pc.sum(1), 1.0)
        assert_allclose(g.coarse_grained_transition_matrix.sum(1), 1.0)
        assert_allclose(g.memberships.sum(1), 1.0)

        assert np.max(subspace_angles(g.memberships, count_chi)) < eps

    def test_init_final_rot_matrix_brandts(
        self,
        svecs_mu0: np.ndarray,
        A_mu0_init: np.ndarray,
        A_mu0: np.ndarray,
    ):
        init_rot = _initialize_rot_matrix(svecs_mu0)
        _, final_rot, _ = _gpcca_core(svecs_mu0)

        assert_allclose(init_rot, A_mu0_init)
        assert_allclose(final_rot, A_mu0)


class TestGPCCAMatlabUnit:
    def test_do_schur(self, example_matrix_mu: np.ndarray):
        N = 9
        P, sd = get_known_input(example_matrix_mu)
        X, RR, _ = _do_schur(P, eta=sd, m=N)

        _assert_schur(P, X, RR, N)

    def test_schur_b_pos(self):
        N = 9
        mu0 = mu(0)
        P, sd = get_known_input(mu0)
        X, RR, _ = _do_schur(P, eta=sd, m=3)

        np.testing.assert_array_equal(P.shape, [N, N])
        np.testing.assert_array_equal(X.shape, [9, 3])
        np.testing.assert_array_equal(RR.shape, [3, 3])

        _assert_schur(P, X, RR, N=None)

    def test_schur_b_neg(self):
        mu0 = mu(0)
        P, sd = get_known_input(mu0)
        with pytest.raises(
            ValueError,
            match="The number of clusters/states is not supposed to be negative",
        ):
            _do_schur(P, eta=sd, m=-3)

    def test_fill_matrix_not_square(self):
        with pytest.raises(ValueError, match="Rotation matrix isn't quadratic."):
            _fill_matrix(np.zeros((3, 4)), np.empty((3, 4)))

    def test_fill_matrix_shape_error(self):
        with pytest.raises(
            ValueError,
            match="The dimensions of the rotation matrix don't match with the number of Schur vectors",
        ):
            _fill_matrix(np.zeros((3, 3)), np.empty((3, 4)))

    def test_gram_schmidt_shape_error_1(self):
        with pytest.raises(ValueError, match=r"not enough values to unpack"):
            _gram_schmidt_mod(np.array([3, 1]), np.array([1]))

    def test_gram_schmidt_shape_error_2(self):
        with pytest.raises(ValueError, match=r"not enough values to unpack"):
            _gram_schmidt_mod(
                np.array([3, 1]),
                np.array([np.true_divide(9, np.sqrt(10)), np.true_divide(1, np.sqrt(10))]),
            )

    def test_gram_schmidt_mod_R2(self):
        Q = _gram_schmidt_mod(np.array([[3, 1], [2, 2]], dtype=np.float64), np.array([0.5, 0.5]))
        s = np.sqrt(0.5)

        orthosys = np.array([[s, -s], [s, s]])

        assert_allclose(Q, orthosys)

    def test_gram_schmidt_mod_R4(self):
        Q = _gram_schmidt_mod(
            np.array([[1, 1, 1, 1], [-1, 4, 4, 1], [4, -2, 2, 0]], dtype=np.float64).T,
            np.array([0.25, 0.25, 0.25, 0.25]),
        )
        d = np.true_divide
        s2 = np.sqrt(2)
        s3 = np.sqrt(3)

        u1 = np.array([0.5] * 4)
        u2 = np.array([d(-1, s2), d(s2, 3), d(s2, 3), d(-1, 3 * s2)])
        u3 = np.array([d(1, 2 * s3), d(-5, 6 * s3), d(7, 6 * s3), d(-5, 6 * s3)])
        orthosys = np.array([u1, u2, u3]).T

        assert_allclose(Q, orthosys)

    def test_indexshape_shape_error(self):
        with pytest.raises(ValueError, match=r"The Schur vector matrix of shape \(3, 4\) has more columns than rows"):
            _indexsearch(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

    def test_indexsearch_1(self):
        v = np.eye(6)
        v = np.r_[np.zeros((1, 6)), v]

        sys = np.c_[v[0], v[1], v[0], v[2], v[0], v[3], v[0], v[4], v[0], v[5], v[0], v[6]].T

        with pytest.raises(
            ValueError,
            match=r"First Schur vector is not constant 1. "
            r"This indicates that the Schur vectors are incorrectly sorted. "
            r"Cannot search for a simplex structure in the data.",
        ):
            _ = _indexsearch(sys)

    def test_indexsearch_2(self):
        v3 = np.array([0, 0, 3])
        p1 = np.array([0.75, 1, 0])
        v1 = np.array([1.5, 0, 0])
        v0 = np.array([0, 0, 0])
        p3 = np.array([0.375, 0.5, 0.75])
        v2 = np.array([0, 2, 0])
        p2 = np.array([0, 1.2, 1.2])
        p4 = np.array([0.15, 0.2, 0.6])
        p5 = np.array([0, 0.6, 0.3])

        sys = np.c_[v3, p1, v1, v0, p3, v2, p2, p4, p5].T

        with pytest.raises(
            ValueError,
            match=r"First Schur vector is not constant 1. "
            r"This indicates that the Schur vectors are incorrectly sorted. "
            r"Cannot search for a simplex structure in the data.",
        ):
            _ = _indexsearch(sys)

    def test_initialize_A_shape_error_1(self):
        X = np.zeros((3, 4))
        X[:, 0] = 1.0
        with pytest.raises(
            ValueError,
            match=r"The Schur vector matrix of shape \(\d+, \d+\) has more columns than rows. "
            r"You can't get a \d+-dimensional simplex from \d+ data vectors.",
        ):
            _initialize_rot_matrix(X)

    def test_initialize_A_first_is_not_constant(self):
        X = np.zeros((4, 4))
        X[0, 0] = 1.0
        with pytest.raises(
            ValueError,
            match="First Schur vector is not constant 1. This indicates that the Schur vectors are incorrectly sorted. "
            "Cannot search for a simplex structure in the data.",
        ):
            _initialize_rot_matrix(X)

    def test_initialize_A_second_and_rest_are_constant(self):
        X = np.zeros((3, 3))
        X[:, 0] = 1.0
        X[:, 2] = 2
        with pytest.raises(
            ValueError,
            match=r"2 Schur vector\(s\) after the first one are constant. Probably the Schur vectors are incorrectly "
            "sorted. Cannot search for a simplex structure in the data.",
        ):
            _initialize_rot_matrix(X)

    def test_initialize_A_condition(self):
        dummy = hilbert(14)
        dummy = dummy[:, :-1]
        dummy[:, 0] = 1.0

        with pytest.raises(ValueError, match="The condition number .*"):
            _initialize_rot_matrix(dummy)

    def test_initialize_A(self):
        mu0 = mu(0)
        P, sd = get_known_input(mu0)
        X, _, _ = _do_schur(P, sd, m=4)
        evs = X[:, :4]

        A = _initialize_rot_matrix(evs)
        index = _indexsearch(evs)
        A_exp = pinv(X[index, :4])

        assert_allclose(A, A_exp)

    def test_initialize_A_condition_warning(self):
        dummy = hilbert(6)
        dummy = dummy[:, :-1]
        dummy[:, 0] = 1.0

        with pytest.warns(UserWarning):
            _ = _initialize_rot_matrix(dummy)

    def test_objective_shape_error_1(self):
        svecs = np.zeros((4, 3))
        svecs[:, 0] = 1.0
        alpha = np.zeros((9,))

        with pytest.raises(ValueError, match="The shape of alpha doesn't match with the shape of X: .+"):
            _objective(alpha, svecs)

    def test_objective_shape_error_2(self):
        svecs = np.zeros((3, 4))
        svecs[:, 0] = 1.0
        alpha = np.zeros((4,))

        with pytest.raises(ValueError, match="The shape of alpha doesn't match with the shape of X: .+"):
            _objective(alpha, svecs)

    def test_objective_1st_col(self, mocker):
        # check_in_matlab: _objective
        P, _ = get_known_input(mu(0))
        N, M = P.shape[0], 4

        _, L, _ = lu(P[:, :M])
        mocker.patch(
            "pygpcca._sorted_schur",
            return_value=(np.eye(M), L, np.array([np.nan] * M)),
        )
        mocker.patch("pygpcca._gpcca._gram_schmidt_mod", return_value=L)

        with pytest.raises(
            ValueError,
            match=r"The first column X\[:, 0\] of the Schur " r"vector matrix isn't constantly equal 1.",
        ):
            _do_schur(P, eta=np.true_divide(np.ones((N,), dtype=np.float64), N), m=M)

    def test_objective_1(self, svecs_mu0: np.ndarray, A_mu0_init: np.ndarray, A_mu0: np.ndarray):
        k = 3
        alpha = np.zeros((k - 1) ** 2)
        for i in range(k - 1):
            for j in range(k - 1):
                alpha[j + i * (k - 1)] = A_mu0_init[i + 1, j + 1]

        act_val = _objective(alpha, svecs_mu0)
        exp_val = k - np.sum(np.true_divide(np.sum(A_mu0 ** 2, axis=0), A_mu0[0, :]))

        assert_allclose(act_val, exp_val)

    def test_objective_2(self, svecs_mu1000: np.ndarray, A_mu1000_init: np.ndarray, A_mu1000: np.ndarray):
        k = 5
        alpha = np.zeros((k - 1) ** 2)
        for i in range(k - 1):
            for j in range(k - 1):
                alpha[j + i * (k - 1)] = A_mu1000_init[i + 1, j + 1]

        act_val = _objective(alpha, svecs_mu1000)
        exp_val = k - np.sum(np.true_divide(np.sum(A_mu1000 ** 2, axis=0), A_mu1000[0, :]))

        assert_allclose(act_val, exp_val)

    def test_opt_soft_shape_error_1(self):
        A = np.zeros((2, 3), dtype=np.float64)
        scvecs = np.zeros((3, 4))
        scvecs[:, 0] = 1.0

        with pytest.raises(ValueError, match="Rotation matrix isn't quadratic."):
            _opt_soft(scvecs, A)

    def test_opt_soft_shape_error_2(self):
        A = np.zeros((3, 3), dtype=np.float64)
        scvecs = np.zeros((2, 4))
        scvecs[:, 0] = 1.0

        with pytest.raises(
            ValueError,
            match="The dimensions of the rotation matrix don't match with the number of Schur vectors.",
        ):
            _opt_soft(scvecs, A)

    def test_opt_soft_shape_error_3(self):
        A = np.zeros((1, 1), dtype=np.float64)
        scvecs = np.zeros((1, 1))
        scvecs[:, 0] = 1.0

        with pytest.raises(
            ValueError,
            match=r"Expected the rotation matrix to be at least of shape \(2, 2\)",
        ):
            _opt_soft(scvecs, A)

    def test_opt_soft_shape_error_4(self):
        # test assertion for schur vector (N,k)-matrix  with k>N
        # the check is done only in `_initialize_rot_matrix`
        # check in matlab: _opt_soft
        scvecs = np.zeros((3, 4))
        scvecs[:, 0] = 1.0

        with pytest.raises(
            ValueError,
            match=r"The Schur vector matrix of shape .* has more columns than rows",
        ):
            _indexsearch(scvecs)

    def test_opt_soft_nelder_mead_mu0(self, svecs_mu0: np.ndarray, A_mu0: np.ndarray):
        A, chi, fopt = _opt_soft(svecs_mu0, A_mu0)

        crispness = np.true_divide(3 - fopt, 3)

        assert_allclose(crispness, 0.973, atol=1e-3)

    def test_opt_soft_nelder_mead_mu1000(self, svecs_mu1000: np.ndarray, A_mu1000: np.ndarray):
        A, chi, fopt = _opt_soft(svecs_mu1000, A_mu1000)

        crispness = np.true_divide(5 - fopt, 5)

        assert_allclose(crispness, 0.804, atol=0.0025)

    def test_opt_soft_nelder_mead_more(self):
        kmin, kmax = 2, 8
        kopt = []
        ks = np.arange(kmin, kmax)

        for mu_ in [10, 50, 100, 200, 500, 1000]:
            mu_ = mu(mu_)
            P, sd = get_known_input(mu_)
            X, _, _ = _do_schur(P, eta=sd, m=kmax)

            crisp = [-np.inf] * (kmax - kmin)
            for j, k in enumerate(range(kmin, kmax)):
                svecs = X[:, :k]
                A = _initialize_rot_matrix(svecs)

                _, _, fopt = _opt_soft(svecs, A)
                crisp[j] = (k - fopt) / k

            kopt.append(ks[np.argmax(crisp)])

        np.testing.assert_array_equal(kopt, [3, 3, 3, 2, 2, 7])

    def test_cluster_by_first_col_not_1(self):
        svecs = np.zeros((4, 3))
        svecs[0, 0] = 1

        with pytest.raises(
            ValueError,
            match="First Schur vector is not constant 1. This indicates that the Schur vectors are incorrectly sorted. "
            "Cannot search for a simplex structure in the data",
        ):
            _cluster_by_isa(svecs)

    def test_cluster_by_isa_shape_error(self):
        svecs = np.zeros((3, 4))
        svecs[:, 1] = 1.0

        with pytest.raises(
            ValueError,
            match=r"The Schur vector matrix of shape \(\d+, \d+\) has more columns than rows. You can't get a "
            r"\d+-dimensional simplex from \d+ data vectors.",
        ):
            _cluster_by_isa(svecs)

    def test_cluster_by_isa(self, chi_isa_mu0_n3: np.ndarray, chi_isa_mu100_n3: np.ndarray):
        # chi_sa_mu0_n3 has permuted 2nd and 3d columns when compared to the matlab version
        for mu_, chi_exp in zip([0, 100], [chi_isa_mu0_n3, chi_isa_mu100_n3]):
            P, sd = get_known_input(mu(mu_))
            X, _, _ = _do_schur(P, sd, m=3)
            chi, _ = _cluster_by_isa(X[:, :3])

            chi = chi[:, _find_permutation(chi_exp, chi)]

            assert_allclose(chi.T @ chi, chi_exp.T @ chi_exp)
            assert_allclose(chi, chi_exp)

    def test_use_minChi(self):
        kmin, kmax = 2, 9
        kopt = []

        for mu_ in [10, 50, 100, 200, 500, 1000]:
            P, sd = get_known_input(mu(mu_))
            g = GPCCA(P, eta=sd)
            minChi = g.minChi(kmin, kmax)

            kopt.append(kmax - 1 - np.argmax(np.flipud(minChi[1:-1])))

        np.testing.assert_array_equal(kopt, [3] * 5 + [7])

    def test_gpcca_brandts_sparse_is_not_densified(self, P: np.ndarray, sd: np.ndarray):
        with pytest.raises(ValueError, match=r"Sparse implementation is only available for `method='krylov'`."):
            GPCCA(csr_matrix(P), eta=sd, method="brandts").optimize(3)

    def test_sort_real_schur(self, R_i: np.ndarray):
        def sort_evals(e: np.ndarray, take: int = 4) -> np.ndarray:
            return e[np.argsort(np.linalg.norm(np.c_[e.real, e.imag], axis=1))][:take]

        # test_SRSchur_num_t
        Q = np.eye(4)
        QQ, RR, ap = sort_real_schur(Q, R_i, z="LM", b=0)

        assert np.all(np.array(ap) <= 1), ap

        EQ = np.true_divide(np.linalg.norm(Q - QQ.T @ QQ, ord=1), eps)
        assert_allclose(EQ, 1.0, atol=5)

        EA = np.true_divide(
            np.linalg.norm(R_i - QQ @ RR @ QQ.T, ord=1),
            eps * np.linalg.norm(R_i, ord=1),
        )
        assert_allclose(EA, 1.0, atol=5)

        l1 = sort_evals(eigvals(R_i))
        l2 = sort_evals(eigvals(RR))

        EL = np.true_divide(np.abs(l1 - l2), eps * np.abs(l1))
        assert_allclose(EL, 1.0, atol=5)


@skip_if_no_petsc_slepc
class TestPETScSLEPc:
    def test_do_schur_krylov(self, example_matrix_mu: np.ndarray):
        N = 9
        P, sd = get_known_input(example_matrix_mu)

        X_k, RR_k, _ = _do_schur(P, eta=sd, m=N, method="krylov")

        _assert_schur(P, X_k, RR_k, N)

    def test_do_schur_krylov_eq_brandts(self, example_matrix_mu: np.ndarray):
        P, sd = get_known_input(example_matrix_mu)

        X_b, RR_b, _ = _do_schur(P, eta=sd, m=3, method="brandts")
        X_k, RR_k, _ = _do_schur(P, eta=sd, m=3, method="krylov")

        # check if it's a correct Schur form
        _assert_schur(P, X_b, RR_b, N=None)
        _assert_schur(P, X_k, RR_k, N=None)
        # check if they span the same subspace
        assert np.max(subspace_angles(X_b, X_k)) < eps

    def test_do_schur_sparse(self, example_matrix_mu: np.ndarray):
        N = 9
        P, sd = get_known_input(example_matrix_mu)

        X_k, RR_k, _ = _do_schur(csr_matrix(P), eta=sd, m=N, method="krylov")

        _assert_schur(P, X_k, RR_k, N)

    def test_normal_case_sparse(
        self,
        P: np.ndarray,
        sd: np.ndarray,
        count_sd: np.ndarray,
        count_Pc: np.ndarray,
        count_chi: np.ndarray,
        count_chi_sparse: np.ndarray,
    ):
        assert_allclose(sd, count_sd)

        g = GPCCA(csr_matrix(P), eta=sd, method="krylov")
        g.optimize((2, 10))

        Pc = g.coarse_grained_transition_matrix
        assert_allclose(Pc, count_Pc, atol=eps)

        assert_allclose(Pc.sum(1), 1.0)
        assert_allclose(g.coarse_grained_transition_matrix.sum(1), 1.0)
        assert_allclose(g.memberships.sum(1), 1.0)

        # regenerated ground truth memberships
        chi = g.memberships
        chi = chi[:, _find_permutation(count_chi_sparse, chi)]
        assert_allclose(chi, count_chi_sparse, atol=eps)

        # ground truth memberships from matlab
        chi = chi[:, _find_permutation(count_chi, chi)]
        assert np.max(np.abs(chi - count_chi)) < 1e-4

    def test_coarse_grain_sparse(self, P: np.ndarray, sd: np.ndarray, count_Pc: np.ndarray):
        Pc = gpcca_coarsegrain(csr_matrix(P), m=(2, 10), eta=sd, method="krylov")

        assert_allclose(Pc.sum(1), 1.0)
        assert_allclose(Pc, count_Pc, atol=eps)

    def test_coarse_grain_sparse_eq_dense(self, example_matrix_mu: np.ndarray):
        P, sd = get_known_input(example_matrix_mu)

        Pc_b = gpcca_coarsegrain(P, m=3, eta=sd, method="brandts")
        Pc_k = gpcca_coarsegrain(csr_matrix(P), m=3, eta=sd, method="krylov")

        assert_allclose(Pc_k, Pc_b)

    def test_memberships_normal_case_sparse_vs_dense(
        self,
        P: np.ndarray,
        sd: np.ndarray,
        count_sd: np.ndarray,
    ):
        assert_allclose(sd, count_sd)  # sanity check

        g_d = GPCCA(P, eta=sd)
        g_d.optimize((2, 10))

        g_s = GPCCA(csr_matrix(P), eta=sd, method="krylov")
        g_s.optimize((2, 10))

        # also passes without this
        ms, md = g_s.memberships, g_d.memberships
        cs, cd = (
            g_s.coarse_grained_transition_matrix,
            g_d.coarse_grained_transition_matrix,
        )
        perm = _find_permutation(md, ms)

        ms = ms[:, perm]
        assert_allclose(ms, md)

        cs = cs[perm, :][:, perm]
        assert_allclose(cs, cd)

    def test_gpcca_krylov_sparse_eq_dense_mu(self, example_matrix_mu: np.ndarray):
        mu = int(example_matrix_mu[2, 4])
        if mu == 1000:
            pytest.skip("rtol=0.03359514, atol=3.73976903e+14")
        opt_clust = {0: 3, 10: 3, 50: 3, 100: 3, 200: 2, 500: 2, 1000: 5}[mu]

        P, sd = get_known_input(example_matrix_mu)

        g_s = GPCCA(csr_matrix(P), eta=sd, method="krylov").optimize(opt_clust)
        g_d = GPCCA(P, eta=sd, method="krylov").optimize(opt_clust)
        g_b = GPCCA(P, eta=sd, method="brandts").optimize(opt_clust)

        assert issparse(g_s.transition_matrix)
        assert not issparse(g_d.transition_matrix)
        assert not issparse(g_b.transition_matrix)

        assert_allclose(g_s.memberships.sum(1), 1.0)
        assert_allclose(g_d.memberships.sum(1), 1.0)
        assert_allclose(g_b.memberships.sum(1), 1.0)

        X_k, X_kd, X_b = g_s.schur_vectors, g_d.schur_vectors, g_b.schur_vectors
        RR_k, RR_kd, RR_b = g_s.schur_matrix, g_d.schur_matrix, g_b.schur_matrix

        # check if it's a correct Schur form
        _assert_schur(P, X_k, RR_k, N=None)
        _assert_schur(P, X_kd, RR_kd, N=None)
        _assert_schur(P, X_b, RR_b, N=None)
        # check if they span the same subspace
        assert np.max(subspace_angles(X_k, X_kd)) < eps
        assert np.max(subspace_angles(X_kd, X_b)) < eps

        ms, md, mb = g_s.memberships, g_d.memberships, g_b.memberships
        cs, cd, cb = (
            g_s.coarse_grained_transition_matrix,
            g_d.coarse_grained_transition_matrix,
            g_b.coarse_grained_transition_matrix,
        )

        for left, right in combinations(["brandts", "dense_krylov", "sparse_krylov"], r=2):
            ml, cl = locals()[f"m{left[0]}"], locals()[f"c{left[0]}"]
            mr, cr = locals()[f"m{right[0]}"], locals()[f"c{right[0]}"]

            perm = _find_permutation(ml, mr)

            mr = mr[:, perm]
            assert_allclose(mr, ml, atol=1e-4)

            cr = cr[perm, :][:, perm]
            try:
                assert_allclose(cr, cl, atol=1e-4)
            except AssertionError as e:
                raise RuntimeError(f"Comparing: {left} and {right}.") from e

    def test_gpcca_krylov_sparse_eq_dense_count(self, P: np.ndarray, sd: np.ndarray):
        # all of them cluster optimally into 3 clusters
        g_s = GPCCA(csr_matrix(P), eta=sd, method="krylov").optimize([2, 5])
        g_d = GPCCA(P, eta=sd, method="krylov").optimize([2, 5])
        g_b = GPCCA(P, eta=sd, method="brandts").optimize([2, 5])

        assert issparse(g_s.transition_matrix)
        assert not issparse(g_d.transition_matrix)
        assert not issparse(g_b.transition_matrix)

        assert_allclose(g_s.memberships.sum(1), 1.0)
        assert_allclose(g_d.memberships.sum(1), 1.0)
        assert_allclose(g_b.memberships.sum(1), 1.0)

        X_k, X_kd, X_b = g_s.schur_vectors, g_d.schur_vectors, g_b.schur_vectors
        RR_k, RR_kd, RR_b = g_s.schur_matrix, g_d.schur_matrix, g_b.schur_matrix

        # check if it's a correct Schur form
        _assert_schur(P, X_k, RR_k, N=None, subspace=True)
        _assert_schur(P, X_kd, RR_kd, N=None, subspace=True)
        _assert_schur(P, X_b, RR_b, N=None, subspace=True)
        # check if they span the same subspace
        assert np.max(subspace_angles(X_k, X_kd)) < eps
        assert np.max(subspace_angles(X_kd, X_b)) < eps

        ms, md, mb = g_s.memberships, g_d.memberships, g_b.memberships
        cs, cd, cb = (
            g_s.coarse_grained_transition_matrix,
            g_d.coarse_grained_transition_matrix,
            g_b.coarse_grained_transition_matrix,
        )

        for left, right in combinations(["brandts", "dense_krylov", "sparse_krylov"], r=2):
            ml, cl = locals()[f"m{left[0]}"], locals()[f"c{left[0]}"]
            mr, cr = locals()[f"m{right[0]}"], locals()[f"c{right[0]}"]

            perm = _find_permutation(ml, mr)

            mr = mr[:, perm]
            assert_allclose(mr, ml)

            cr = cr[perm, :][:, perm]
            try:
                assert_allclose(cr, cl)
            except AssertionError as e:
                raise RuntimeError(f"Comparing: {left} and {right}.") from e

    def _generate_ground_truth_rot_matrices(self):
        # this function generates the data for "test_init_final_rotation_matrix"
        P, sd = get_known_input(mu(0))
        g_ks = GPCCA(csr_matrix(P), method="krylov").optimize(3)
        g_kd = GPCCA(P, method="krylov").optimize(3)

        for g in [g_ks, g_kd]:
            g.schur_vectors
            _initialize_rot_matrix(sd)
            g.rotation_matrix

    def test_init_final_rot_matrix_krylov_sparse(
        self,
        svecs_mu0_krylov_sparse: np.ndarray,
        A_mu0_krylov_sparse_init: np.ndarray,
        A_mu0_krylov_sparse: np.ndarray,
    ):
        init_rot = _initialize_rot_matrix(svecs_mu0_krylov_sparse)
        _, final_rot, _ = _gpcca_core(svecs_mu0_krylov_sparse)

        assert_allclose(init_rot, A_mu0_krylov_sparse_init)
        assert_allclose(final_rot, A_mu0_krylov_sparse)

    def test_init_final_rot_matrix_krylov_dense(
        self,
        svecs_mu0_krylov_dense: np.ndarray,
        A_mu0_krylov_dense_init: np.ndarray,
        A_mu0_krylov_dense: np.ndarray,
    ):
        init_rot = _initialize_rot_matrix(svecs_mu0_krylov_dense)
        _, final_rot, _ = _gpcca_core(svecs_mu0_krylov_dense)

        assert_allclose(init_rot, A_mu0_krylov_dense_init)
        assert_allclose(final_rot, A_mu0_krylov_dense)


class TestCustom:
    @pytest.mark.parametrize("method", ["krylov", "brandts"])
    def test_P_i(self, P_i: np.ndarray, method: str):
        if method == "krylov":
            pytest.importorskip("mpi4py")
            pytest.importorskip("petsc4py")
            pytest.importorskip("slepc4py")

        g = GPCCA(P_i, eta=None, method=method)

        for m in range(2, 8):
            try:
                g.optimize(m)
            except ValueError:
                continue

            X, RR = g.schur_vectors, g.schur_matrix

            assert_allclose(g.memberships.sum(1), 1.0)
            assert_allclose(g.coarse_grained_transition_matrix.sum(1), 1.0)
            assert_allclose(g.coarse_grained_input_distribution.sum(), 1.0)
            if g.coarse_grained_stationary_probability is not None:
                assert_allclose(g.coarse_grained_stationary_probability.sum(), 1.0)
            np.testing.assert_allclose(X[:, 0], 1.0)

            assert np.max(subspace_angles(P_i @ X, X @ RR)) < eps

    @pytest.mark.parametrize("method", ["krylov", "brandts"])
    def test_P_2_LM(
        self,
        P_2: np.ndarray,
        minChi_P_2_LM: np.ndarray,
        crispness_values_P_2_LM: np.ndarray,
        optimal_crispness_P_2_LM: np.float64,
        n_m_P_2_LM: np.int64,
        top_eigenvalues_P_2_LM: np.ndarray,
        method: str,
    ):
        if method == "krylov":
            pytest.importorskip("mpi4py")
            pytest.importorskip("petsc4py")
            pytest.importorskip("slepc4py")

        g = GPCCA(P_2, eta=None, z="LM", method=method)

        # The following very crude minChi testing is necessary,
        # since the initial guess for the rotation matrix and thus minChi can vary.
        minChi = g.minChi(2, 12)
        assert len(minChi) == len(minChi_P_2_LM)
        assert minChi[0] > -1e-08
        assert minChi[1] > -1e-08
        assert minChi[10] > -1e-08

        g.optimize({"m_min": 2, "m_max": 12})
        n_m = g.n_m

        assert_allclose(g.crispness_values, crispness_values_P_2_LM)
        assert_allclose(g.optimal_crispness, optimal_crispness_P_2_LM)
        assert_allclose(n_m, n_m_P_2_LM)
        assert_allclose(g.top_eigenvalues, top_eigenvalues_P_2_LM)
        assert_allclose(g.dominant_eigenvalues, top_eigenvalues_P_2_LM[:n_m])

    def test_split_warning_LM(self, P_2: np.ndarray):

        g = GPCCA(P_2, eta=None, z="LM")

        with pytest.warns(
            UserWarning,
            match="Clustering into 4 clusters will split complex conjugate eigenvalues. "
            "Skipping clustering into 4 clusters.",
        ):
            g.optimize({"m_min": 2, "m_max": 5})
        with pytest.warns(
            UserWarning,
            match="Clustering into 6 clusters will split complex conjugate eigenvalues. "
            "Skipping clustering into 6 clusters.",
        ):
            g.optimize({"m_min": 5, "m_max": 7})
        with pytest.warns(
            UserWarning,
            match="Clustering into 9 clusters will split complex conjugate eigenvalues. "
            "Skipping clustering into 9 clusters.",
        ):
            g.optimize({"m_min": 8, "m_max": 11})
        with pytest.warns(
            UserWarning,
            match="Clustering 12 data points into 12 clusters is always perfectly crisp. "
            "Thus m=12 won't be included in the search for the optimal cluster number.",
        ):
            g.optimize({"m_min": 11, "m_max": 12})

    def test_split_raise_LM(self, P_2: np.ndarray):

        g = GPCCA(P_2, eta=None, z="LM")

        with pytest.raises(
            ValueError,
            match="Clustering into 4 clusters will split complex conjugate eigenvalues. "
            "Request one cluster more or less.",
        ):
            g.optimize(4)
        with pytest.raises(
            ValueError,
            match="Clustering into 6 clusters will split complex conjugate eigenvalues. "
            "Request one cluster more or less.",
        ):
            g.optimize(6)
        with pytest.raises(
            ValueError,
            match="Clustering into 9 clusters will split complex conjugate eigenvalues. "
            "Request one cluster more or less.",
        ):
            g.optimize(9)

    @pytest.mark.parametrize("method", ["krylov", "brandts"])
    def test_P_2_LR(
        self,
        P_2: np.ndarray,
        minChi_P_2_LR: np.ndarray,
        crispness_values_P_2_LR: np.ndarray,
        optimal_crispness_P_2_LR: np.float64,
        n_m_P_2_LR: np.int64,
        top_eigenvalues_P_2_LR: np.ndarray,
        method: str,
    ):
        if method == "krylov":
            pytest.importorskip("mpi4py")
            pytest.importorskip("petsc4py")
            pytest.importorskip("slepc4py")

        g = GPCCA(P_2, eta=None, z="LR", method=method)

        # The following very crude minChi testing is necessary,
        # since the initial guess for the rotation matrix and thus minChi can vary.
        minChi = g.minChi(2, 12)
        assert len(minChi) == len(minChi_P_2_LR)
        assert minChi[0] > -1e-08
        assert minChi[1] > -1e-08
        assert minChi[3] > -1e-01
        assert minChi[10] > -1e-08

        g.optimize({"m_min": 2, "m_max": 12})
        n_m = g.n_m

        assert_allclose(g.crispness_values, crispness_values_P_2_LR)
        assert_allclose(g.optimal_crispness, optimal_crispness_P_2_LR)
        assert_allclose(n_m, n_m_P_2_LR)
        assert_allclose(g.top_eigenvalues, top_eigenvalues_P_2_LR)
        assert_allclose(g.dominant_eigenvalues, top_eigenvalues_P_2_LR[:n_m])

    def test_split_warning_LR(self, P_2: np.ndarray):

        g = GPCCA(P_2, eta=None, z="LR")

        with pytest.warns(
            UserWarning,
            match="Clustering into 7 clusters will split complex conjugate eigenvalues. "
            "Skipping clustering into 7 clusters.",
        ):
            g.optimize({"m_min": 2, "m_max": 8})
        with pytest.warns(
            UserWarning,
            match="Clustering into 9 clusters will split complex conjugate eigenvalues. "
            "Skipping clustering into 9 clusters.",
        ):
            g.optimize({"m_min": 8, "m_max": 10})
        with pytest.warns(
            UserWarning,
            match="Clustering into 11 clusters will split complex conjugate eigenvalues. "
            "Skipping clustering into 11 clusters.",
        ):
            g.optimize({"m_min": 10, "m_max": 12})

    def test_split_raise_LR(self, P_2: np.ndarray):

        g = GPCCA(P_2, eta=None, z="LR")

        with pytest.raises(
            ValueError,
            match="Clustering into 7 clusters will split complex conjugate eigenvalues. "
            "Request one cluster more or less.",
        ):
            g.optimize(7)
        with pytest.raises(
            ValueError,
            match="Clustering into 9 clusters will split complex conjugate eigenvalues. "
            "Request one cluster more or less.",
        ):
            g.optimize(9)
        with pytest.raises(
            ValueError,
            match="Clustering into 11 clusters will split complex conjugate eigenvalues. "
            "Request one cluster more or less.",
        ):
            g.optimize(11)

    def test_optimize_range_all_invalid(self, P_2: np.ndarray, mocker):

        g = GPCCA(P_2, eta=None, z="LR")

        mocker.patch(
            "pygpcca._gpcca._gpcca_core",
            # chi, rot. mat., crispness
            return_value=(np.empty((P_2.shape[0], 3)), np.empty_like((3, 3)), 0),
        )
        with pytest.raises(ValueError, match=r"Clustering wasn't successful. Try different cluster numbers."):
            g.optimize([3, P_2.shape[0]])


class TestUtils:
    def test_transition_matrix_dtype(self, P_2: np.ndarray):
        g = GPCCA(P_2, eta=None, z="LR")

        assert g.transition_matrix.dtype == np.float64

    @pytest.mark.parametrize("eta", [None, np.ones((12,), dtype=np.float16) / 12.0])
    def test_input_distribution_dtype(self, P_2: np.ndarray, eta: Optional[np.ndarray]):
        g = GPCCA(P_2, eta=eta, z="LR")

        assert g.input_distribution.dtype == np.float64
