from docrep import DocstringProcessor

P = """\
P
    The transition matrix (row-stochastic)."""

m = """\
m
    The number of clusters to group into."""

k = """\
k
    The number of eigenvalues and Schur vectors to sort."""

m_optimize = """\
m
    The number of clusters or a range where a search for potentially optimal
    cluster numbers is performed. Valid options are:

        - :class:`int`: number of clusters to group into.
        - :class:`tuple`: minimal and maximal number of clusters.
        - :class:`dict`: minimal and maximal number of clusters given as ``{'m_min': int, 'm_max': int}``."""

z = """\
z
    Specifies which portion of the spectrum is to be sought.
    The subspace returned will be associated with this part of the spectrum.
    Valid options are:

        - 'LM': largest magnitude (default).
        - 'LR': largest real part."""

z_P = """\
z
    Specifies which portion of the eigenvalue spectrum of `P` is to be sought.
    The returned invariant subspace of `P` will be associated with this part
    of the spectrum. Valid options are:

        - 'LM': largest magnitude (default).
        - 'LR': largest real part."""

method = """\
method
    Which method to use to determine the invariant subspace. Valid options are:

        - 'brandts': perform a full Schur decomposition of `P` utilizing
          :func:`scipy.linalg.schur` (without the intrinsic sorting option,
          since it is flawed) and sort the returned Schur form `R` and Schur vector
          matrix `Q` afterwards using a routine published by Brandts
          [Brandts02]_.
          This is well tested and thus the default method,
          although it is also the slowest choice.
        - 'krylov': calculate an orthonormal basis of the subspace
          associated with the `m` dominant eigenvalues of `P` using the
          Krylov-Schur method as implemented in ``SLEPc``.
          This is the fastest choice and especially suitable for very
          large `P`, but it is still experimental.
"""

tol_krylov = """\
tol_krylov
    The convergence criterion used by ``SLEPc`` internally.
    This is only relevant if you use ``method='krylov'``.
    If you are dealing with ill-conditioned matrices,
    consider decreasing this value to get accurate results."""

eta = """\
eta
    The input probability distribution of the (micro)states.
    In theory `eta` can be an arbitrary distribution as long as it is
    a valid probability distribution (i.e., sums up to 1).
    A neutral and valid choice would be the uniform distribution (default).

    In case of a reversible transition matrix, the stationary distribution
    can (but don't has to) be used here.
    In case of a non-reversible `P`, some initial or average distribution of
    the states might be chosen instead of the uniform distribution.

    Vector of shape `(n,)` which sums to 1."""

chi_ret = r"""An array of shape `(n, m)` containing the membership :math:`\chi_{ij}` (or probability)
of each state :math:`i` (to be assigned) to each cluster :math:`j`. The rows sum up to 1."""

rot_matrix_ret = r"""The optimized rotation matrix :math:`A` of shape `(m, m)` that rotates the dominant
Schur vectors to yield the G-PCCA memberships, i.e., :math:`\chi = X A`."""

crispness_ret = r"""The crispness :math:`\xi \in [0,1]` quantifies the optimality of the
solution (higher is better). It characterizes how crisp (sharp) the
decomposition of the state space into `m` clusters is.
It is given via (Eq. 17 from [Roeblitz13]_):

.. math:: \xi = (m - f_{opt}) / m = \mathtt{trace}(S) / m = \mathtt{trace}(\tilde{D} \chi^T D \chi) / m

with :math:`D` being a diagonal matrix with :math:`\eta` on its diagonal.
"""

Q_sort = """A matrix of shape `(n, m)` with ordered `m` dominant Schur vectors in the columns.
The constant Schur vector (being constantly 1) is in the first column."""

R_sort = r"""The ordered top left part of shape `(m, m)` of the real Schur matrix of `P`.
The ordered real partial Schur matrix `R` of `P` fulfills

.. math:: \tilde{P} Q = Q R

with the ordered real matrix of dominant Schur vectors `Q`."""

eigenvalues_m = """An array of shape `(m,)` containing the `m` dominant eigenvalues of `P`."""

eigenvalues_k = """An array of shape `(k,)` containing the `k` dominant eigenvalues of `P`."""

d = DocstringProcessor(
    P=P,
    m=m,
    k=k,
    m_optimize=m_optimize,
    z=z,
    z_P=z_P,
    method=method,
    tol_krylov=tol_krylov,
    eta=eta,
    chi_ret=chi_ret,
    rot_matrix_ret=rot_matrix_ret,
    crispness_ret=crispness_ret,
    Q_sort=Q_sort,
    R_sort=R_sort,
    eigenvalues_m=eigenvalues_m,
    eigenvalues_k=eigenvalues_k,
)
