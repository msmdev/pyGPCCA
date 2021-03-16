from scipy.linalg import subspace_angles
from scipy.sparse import csr_matrix
import numpy as np

from tests.conftest import assert_allclose, skip_if_no_petsc_slepc
from pygpcca.utils._utils import _eigs_slepc


@skip_if_no_petsc_slepc
class TestEigendecompositionSLEPSc:
    def test_eigendecomposition_slepsc_regression(
        self,
        test_matrix_1: np.ndarray,
        test_matrix_1_eigenvalues: np.ndarray,
        test_matrix_1_eigenvectors: np.ndarray,
        test_matrix_2: np.ndarray,
        test_matrix_2_eigenvalues: np.ndarray,
        test_matrix_2_eigenvectors: np.ndarray,
    ):

        eigs_1, vecs_1 = _eigs_slepc(csr_matrix(test_matrix_1), k=11)
        assert_allclose(
            test_matrix_1_eigenvalues,
            eigs_1,
        )
        assert_allclose(subspace_angles(test_matrix_1_eigenvectors, vecs_1), 0.0, atol=1e-6, rtol=1e-5)

        eigs_2, vecs_2 = _eigs_slepc(csr_matrix(test_matrix_2), k=13)
        assert_allclose(test_matrix_2_eigenvalues, eigs_2)
        assert_allclose(subspace_angles(test_matrix_2_eigenvectors, vecs_2), 0.0, atol=1e-6, rtol=1e-5)
