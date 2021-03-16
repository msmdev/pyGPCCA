from typing import Tuple

import pytest

from scipy.sparse import issparse, csr_matrix
import numpy as np

from tests.conftest import bdc, assert_allclose
from pygpcca.utils._utils import (
    stationary_distribution,
    stationary_distribution_from_eigenvector,
    stationary_distribution_from_backward_iteration,
)


def _create_qp(dim: int) -> Tuple[np.array, np.array]:
    p = np.zeros(dim)
    p[0:-1] = 0.5

    q = np.zeros(dim)
    q[1:] = 0.5

    p[dim // 2 - 1] = 0.001
    q[dim // 2 + 1] = 0.001

    return q, p


class TestStationaryDistribution:
    @pytest.mark.parametrize("dim", [10, 100, 1000])
    def test_stat_dist_decomposition(self, dim: int):
        q, p = _create_qp(dim)
        P, mu_expected = bdc(q=q, p=p, sparse=False)
        mu_actual = stationary_distribution_from_eigenvector(P)

        assert_allclose(mu_expected, mu_actual)

    @pytest.mark.parametrize("dim", [10, 100, 1000])
    def test_stat_dist_iteration(self, dim: int):
        q, p = _create_qp(dim)
        P, mu_expected = bdc(q=q, p=p, sparse=False)
        mu_actual = stationary_distribution_from_backward_iteration(P)

        assert_allclose(mu_expected, mu_actual)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_stat_dist_sparse_dense(self, sparse: bool):
        q, p = _create_qp(100)
        P, mu_expected = bdc(q=q, p=p, sparse=sparse)
        mu_actual = stationary_distribution(P)

        assert_allclose(mu_expected, mu_actual)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_stat_dist_regression(self, P: np.ndarray, P_mu: np.ndarray, sparse: bool):
        if sparse:
            P = csr_matrix(P)
        else:
            assert not issparse(P)

        mu_actual = stationary_distribution(P)
        assert_allclose(P_mu, mu_actual)

    def test_stat_dist_regression_test_matrices(
        self,
        test_matrix_1: np.ndarray,
        test_matrix_1_stationary_distribution,
        test_matrix_2: np.ndarray,
        test_matrix_2_stationary_distribution,
        test_matrix_3: np.ndarray,
    ):
        # For the first two matrices, the stationary distribution exists and is unique
        assert_allclose(
            test_matrix_1_stationary_distribution,
            stationary_distribution(test_matrix_1),
        )
        assert_allclose(test_matrix_2_stationary_distribution, stationary_distribution(test_matrix_2))

        # For the third matrix, the stationary distribution is not uniquely defined
        with pytest.raises(ValueError, match=r"This matrix is reducible."):
            stationary_distribution(test_matrix_3)
