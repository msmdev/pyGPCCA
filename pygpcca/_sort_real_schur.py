# Python version (translated by Fabian Paul; revised by Bernhard Reuter, Michal Klein) of the following original work:
# --------------------------------------------------------------------------------------------------------------------
# Title:	Sorting Real Schur Forms
# Author:	Jan Brandts
# E-Mail:	brandts-AT-science.uva.nl
# http://m2matlabdb.ma.tum.de/download.jsp?MC_ID=3&MP_ID=119
# http://dx.doi.org/10.1002/nla.274
# Institution:	University of Amsterdam
# Description:	In Matlab 6, there exists a command to generate a real Schur form,
# wheras another transforms a real Schur form into a complex one.
# There do not exist commands to prescribe the order in which the eigenvalues appear on the diagonal of the upper
# (quasi-) triangular factor T. For the complex case, a routine is sketched in Golub and Van Loan (1996),
# that orders the diagonal of T according to their distance to a target value.
# In the reference below, we give a Matlab routine to sort real Schur forms in Matlab.
# It is based on a block-swapping procedure by Bai and Demmel (1993).
# Sorting real Schur forms, both partially and completely,
# has important applications in the computation of real invariant subspaces.
# Reference:    J.H. Brandts. Matlab code for sorted real Schur forms.
# Numerical Linear Algebra with Applications 9(3):249-261 (2002)
# Keywords:	    Real Schur Form, sorting, Bai-Demmel algorithm, swapping
# Based on the original Matlab File Version: 1.0
# --------------------------------------------------------------------------------------------------------------------
# All references to equations or pages made in the comments are referencing
# Jan Brandts. Matlab Code for Sorted Real Schur Forms. Preprint No. 1180,
# January, 2001, Universiteit Utrecht,
# https://www.math.uu.nl/publications/preprints/1180.pdf
# --------------------------------------------------------------------------------------------------------------------

from typing import List, Tuple, Union

import numpy as np

from pygpcca.utils._docs import d

__all__ = ["sort_real_schur"]
expensive_asserts = False


@d.dedent
def sort_real_schur(
    Q: np.ndarray, R: np.ndarray, z: str, b: float, inplace: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    r"""
    Partially or completely sort the real Schur form `R` and  Schur vectors `Q` of a square matrix `A`.

    The diagonal blocks of `R` will be ordered with respect to a target `z`.

    The blocks on the diagonal are associated with either real eigenvalues,
    in case of 1x1 blocks, or pairs of complex eigenvalues, in case of 2x2 blocks.

    The number of ordered blocks is determined by a parameter `b`.
    A vector `ap` warns for inaccuracy of the solution, if an entry of `ap` exceeds one.
    This function is based on MATLAB code originally published by Brandts [Brandts02]_.

    Parameters
    ----------
    Q
        Orthogonal real matrix `Q` of Schur vectors such that :math:`AQ = QR`.
    R
        Quasi-triangular real Schur form `R` such that :math:`AQ = QR`.
    %(z)s
    b
        Determines the length of the ordering with respect to `z`.
        Valid options are:

            - ``b < 0``: ``-b`` blocks will be sorted.
            - ``b > 0``: b or ``b + 1`` eigenvalues will be sorted,
              depending on the sizes of the blocks.
            - ``b = 0``: the whole Schur form will be sorted.

    inplace
        Determines, if the supplied `Q` and `R` matrices are sorted in place (``ìnplace = True``) or
        if copies are made and sorted (``inplace = False``; default).

    Returns
    -------
    Tuple of the following:

        - Q : orthogonal real `(n, n)` Schur vector matrix `Q` such that :math:`AQ = QR`
          with the diagonal blocks ordered with respect to the target `z`.
        - R : quasi-triangular real `(n, n)` Schur matrix `R` such that :math:`AQ = QR`
          with the diagonal blocks ordered with respect to the target `z`.
        - ap : A list `ap` warns for inaccuracy of the solution, if an entry of `ap` exceeds one.
    """  # noqa: D401
    eps = np.finfo(R.dtype).eps
    if not np.all(np.abs(np.tril(R, -2)) <= 100 * eps):
        raise ValueError("R is not block-triangular.")
    if not inplace:
        Q = Q.copy()
        R = R.copy()

    r = np.where(np.abs(np.diag(R, -1)) > 100 * eps)[0]  # detect sub-diagonal nonzero entries
    s = [
        i for i in range(R.shape[0] + 1) if i not in r + 1
    ]  # construct from them a vector s with the-top left positions of each block

    p = np.empty((len(s) - 1,), dtype=np.complex128)

    for k in range(1, len(s) - 1):  # debug
        assert R[s[k], s[k] - 1] <= 100 * eps  # debug

    for k in range(len(s) - 1):  # ranging over all blocks
        sk = s[k]
        if s[k + 1] - sk == 2:  # if the block is 2x2
            Q, R = normalize(Q, R, slice(sk, s[k + 1]), inplace=True)  # normalize it
            # store the eigenvalues
            p[k] = R[sk, sk] + np.lib.scimath.sqrt(R[sk + 1, sk] * R[sk, sk + 1])  # type: ignore[attr-defined]
        else:  # (the one with the positive imaginary part is sufficient)
            assert s[k + 1] - sk == 1  # debug
            p[k] = R[s[k], s[k]]  # if the block is 1x1, only store the eigenvalue

    ap = []

    for k in swaplist(p, s, z, b):  # For k ranging over all neighbor-swaps
        assert k + 2 < len(s)  # debug
        v = list(range(s[k], s[k + 1]))  # collect the coordinates of the blocks
        w = list(range(s[k + 1], s[k + 2]))
        assert v[0] != w[0]  # debug
        if len(v) == 2:
            assert v[0] < v[1]  # debug
        if len(w) == 2:
            assert w[0] < w[1]  # debug
        if (
            __debug__ and expensive_asserts
        ):  # debug: check that we are moving the larger eigenvalues to the left (expensive test)
            if v[0] < w[0]:  # debug
                arr = [p[k], p[k + 1]]  # debug
                _, which = select(arr, z)  # debug
                assert which == 1  # debug
            else:  # debug
                arr = [p[k + 1], p[k]]  # debug
                _, which = select(arr, z)  # debug
                assert which == 1  # debug
        vw = v + w
        nrA = np.linalg.norm(R[vw, :][:, vw], ord=np.inf)  # compute norm of the matrix A from eq. (6)
        Q, R = swap(Q, R, v, w, inplace=True)  # swap the blocks
        p[k], p[k + 1] = p[k + 1], p[k]  # debug
        s[k + 1] = s[k] + s[k + 2] - s[k + 1]  # update positions of blocks
        v = list(range(s[k], s[k + 1]))  # update block-coordinates
        w = list(range(s[k + 1], s[k + 2]))
        if len(v) == 2:  # if the first block is 2 x 2
            Q, R = normalize(Q, R, v, inplace=True)  # normalize it
        if len(w) == 2:  # if the second block is 2 x 2
            Q, R = normalize(Q, R, w, inplace=True)  # normalize it
        ap.append(
            np.linalg.norm(R[w, :][:, v], ord=np.inf) / (10 * eps * nrA)
        )  # measure size of bottom-left block (see p.6, Sect. 2.3)

    R = R - np.tril(R, -2)  # Zero the below-block entries
    for k in range(1, len(s) - 1):  # to get a quasi-triangle again
        R[s[k], s[k] - 1] = 0

    return Q, R, ap


# Based on the original MATLAB code bellow:
# -------------------------------------------------------------------------
# r = find(abs(diag(R,-1)) > 100*eps);
# s = 1:size(R,1)+1;
# s(r+1) = [];
#
# for k=1:length(s)-1;
#  sk = s(k);
#  if s(k+1)-sk == 2
#    [Q,R] = normalize(Q,R,sk:s(k+1)-1);
#    p(k)  = R(sk,sk)+sqrt(R(sk+1,sk)*R(sk,sk+1));
#  else
#    p(k)  = R(s(k),s(k));
# end
# end
#
# for k = swaplist(p,s,z,b);
#  v      = s(k):s(k+1)-1;
#  w      = s(k+1):s(k+2)-1;
#  nrA    = norm(R([v,w],[v,w]),inf);
#  [Q,R]  = swap(Q,R,v,w);
#  s(k+1) = s(k)+s(k+2)-s(k+1);
#  v      = s(k):s(k+1)-1;
#  w      = s(k+1):s(k+2)-1;
#  if length(v)==2
#    [Q,R] = normalize(Q,R,v);
#  end
#  if length(w)==2
#    [Q,R] = normalize(Q,R,w);
#  end
#  ap(k)  = norm(R(w,v),inf)/(10*eps*nrA);
# end
#
# R = R - tril(R,-2);
# for j=2:length(s)-1; R(s(j),s(j)-1)=0; end
# -------------------------------------------------------------------------


def normalize(
    U: np.ndarray, S: np.ndarray, v: Union[slice, List[int]], inplace: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a Givens rotation such that the two-by-two diagonal block of `S` situated at diagonal positions \
    ``v[0]``, ``v[1]`` is in standardized form.

    I.e., the diagonal entries are equal, and the off-diagonal elements are of opposite sign.

    Parameters
    ----------
    U
        Orthogonal real matrix.
    S
        Quasi-triangular real matrix.
    v
        List of diagonal positions of the considered block of `S`.
    inplace
        Determines, if the supplied `U` and `S` matrices are used in place (``ìnplace = True``) or
        if copies are made and manipulated (``inplace = False``; default).

    Returns
    -------
    Tuple of the following:

        - U : Orthogonal real matrix.
        - S : Quasi-triangular real matrix with the two-by-two diagonal block of `S` situated at
          diagonal positions ``v[0]``, ``v[1]`` in standardized form.
    """
    Q = rot(S[v, :][:, v])  # Determine the Givens rotation needed for standardization -
    if not inplace:
        S = S.copy()
        U = U.copy()
    S[:, v] = np.dot(S[:, v], Q)  # and apply it left and right to S, and right to U.
    S[v, :] = np.dot(Q.T, S[v, :])  # Only rows and columns with indices in the vector v can be affected by this.
    U[:, v] = np.dot(U[:, v], Q)
    return U, S


# Based on the original MATLAB code bellow:
# -------------------------------------------------------------------------
#  function [U,S] = normalize(U,S,v);
#  n  = size(S,1);
#  Q  = rot(S(v,v));
#  S(:,v) = S(:,v)*Q;
#  S(v,:) = Q'*S(v,:);
#  U(:,v) = U(:,v)*Q;
# -------------------------------------------------------------------------


def rot(X: np.ndarray) -> np.ndarray:
    r"""
    Compute a Givens rotation needed in the :func:`normalize`.

    Parameters
    ----------
    X
        Two-by-two block of the quasi-triangular real Schur matrix.

    Returns
    -------
    Two-by-two block in standardized from.
    """
    c = 1.0  # Start with the identity transformation, and if needed, change it into ...
    s = 0.0
    if X[0, 0] != X[1, 1]:
        tau = (X[0, 1] + X[1, 0]) / (X[0, 0] - X[1, 1])
        off = (tau ** 2 + 1) ** 0.5
        v = [tau - off, tau + off]
        w = int(np.argmin(np.abs(v)))
        c = 1.0 / (1.0 + v[w] ** 2) ** 0.5  # ... the cosine and sine as given in Section 2.3.1
        s = v[w] * c

    return np.array([[c, -s], [s, c]], dtype=X.dtype)


# Based on the original MATLAB code bellow:
# -------------------------------------------------------------------------
#  function Q = rot(X);
#  c = 1; s = 0;
#  if X(1,1)~=X(2,2);
#    tau   = (X(1,2)+X(2,1))/(X(1,1)-X(2,2));
#    off   = sqrt(tau^2+1);
#    v     = [tau - off, tau + off];
#    [d,w] = min(abs(v));
#    c     = 1/sqrt(1+v(w)^2);
#    s     = v(w)*c;
#  end
#  Q = [c -s;s c];
# -------------------------------------------------------------------------


@d.dedent
def swaplist(p: Union[np.ndarray, List[float]], s: List[int], z: str, b: float) -> List[int]:
    """
    Produce a list `v` of swaps of neighboring blocks needed to order the eigenvalues assembled in the vector `p` \
    from closest to `z` to farthest away from `z`, taking into account the parameter `b`.

    To do so, Python's :func:`sorted`, producing a stable sort, is used to realize
    the objective ordering of the diagonal blocks. This objective ordering
    can easily be defined, since all eigenvalues can be extracted from the
    given real Schur form. This, in turn, results in an objective
    permutation of the given ordering, which can be realized by `n` swaps
    of neighboring pairs, to be represented by a swaplist `v`.

    p
        List of eigenvalues (only one copy for each complex-conjugate pair).
    s
        List of the the-top left positions of each block.
    %(z)s
    b
        Determines the length of the ordering with respect to `z`.
        Valid options are:

            - ``b < 0``: ``-b`` blocks will be sorted.
            - ``b > 0``: b or ``b+1`` eigenvalues will be sorted, depending on the sizes of the blocks.
            - ``b = 0``: the whole Schur form will be sorted.

    Returns
    -------
    Swaplist `v`, where ``v[j] = k`` means that in the `j`-th swap, the `k`-th and `k+1`-th block should be swapped.
    """
    p_orig = p  # debug
    n = len(p)
    p = list(p)
    k = 0
    v: List[int] = []
    srtd = 0  # Number of sorted eigenvalues.
    q = list(np.diff(s))  # Compute block sizes.
    q_orig = list(q)  # debug
    fini = False
    while not fini:
        _, j = select(p[k:n], z)  # Determine which block will go to position k
        p_j = p[k + j]  # debug
        p[k : n + 1] = [p[j + k]] + p[k:n]  # insert this block at position k,
        assert p[k] == p_j  # debug
        del p[j + k + 1]  # and remove it from where it was taken.
        if expensive_asserts and __debug__:
            assert np.all(sorted(p) == sorted(p_orig))  # debug
        q_j = q[k + j]  # debug
        q[k : n + 1] = [q[j + k]] + q[k:n]  # Similar for the block-sizes
        assert q[k] == q_j  # debug
        del q[j + k + 1]
        if expensive_asserts and __debug__:
            assert np.all(sorted(q) == sorted(q_orig))  # debug
        v = v + list(range(k, j + k))[::-1]  # Update the list of swaps for this block
        srtd = srtd + q[k]  # Update the number of sorted eigenvalues
        k += 1
        fini = k >= n - 1 or k == -b or srtd == b or (srtd == b + 1 and b != 0)
    return v


# Based on the original MATLAB code bellow:
# -------------------------------------------------------------------------
#  function v = swaplist(p,s,z,b);
#  n = length(p);
#  k = 0; v = [];
#  srtd = 0;
#  q = diff(s);
#  fini = 0;
#  while ~fini
#    k        = k+1;
#    [dum,j]  = select(p(k:n),z);
#    p(k:n+1) = [p(j+k-1) p(k:n)];
#    p(j+k)   = [];
#    q(k:n+1) = [q(j+k-1) q(k:n)];
#    q(j+k)   = [];
#    v        = [v,j+k-2:-1:k];
#    srtd     = srtd + q(k);
#    fini     = (k==n-1)|(k==-b)|(srtd==b)|((srtd==b+1)&(b~=0));
#  end
# -------------------------------------------------------------------------


@d.dedent
def select(p: Union[List[str], np.ndarray], z: str) -> Tuple[float, int]:
    """
    Determine which block is next in the ordering (needed in :func:`normalize`).

    Parameters
    ----------
    p
        List of eigenvalues.
    %(z)s

    Returns
    -------
    Block that is next in the ordering.
    """
    if z == "LM":
        pos = int(np.argmax(np.abs(p)))
        return np.abs(p[pos]), pos
    elif z == "LR":
        pos = int(np.argmax(np.real(p)))
        return np.real(p[pos]), pos
    else:
        raise NotImplementedError(z)
        # possible further sorting critera, if needed...
        # y = np.real(z) + np.abs(np.imag(z))*1j  # Move target to the upper half plane.
        # delta = np.abs(np.array(p) - y)
        # pos = np.argmin(delta)  # Find block closest to the target.
        # return delta[pos], pos


# Based on the original MATLAB code bellow:
# -------------------------------------------------------------------------
#  function [val,pos] = select(p,z);
#  y = real(z)+abs(imag(z))*i;
#  [val pos] = min(abs(p-y));
# -------------------------------------------------------------------------


def swap(
    U: np.ndarray, S: np.ndarray, v: List[int], w: List[int], inplace: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Swap the two blocks on the diagonal of `S` at positions symbolized by the entries of `v` and `w`.

    U
        Orthogonal real matrix.
    S
        Quasi-triangular real matrix.
    v
        List of integers (either one integer, if one has a one-by-one block
        or two integers, if one has a two-by-two block) indicating the block
        to swap against the block indicated by `w`.
    w
        List of integers (either one integer, if one has a one-by-one block
        or two integers, if one has a two-by-two block) indicating the block
        to swap against the block indicated by `v`.
    inplace
        Determines, if the supplied `U` and `S` matrices are used
        in place (``ìnplace = True``) or if copies are made and manipulated
        (``inplace = False``; default).

    Returns
    -------
    Tuple of the following:

        - U : Orthogonal real matrix.
        - S : Quasi-triangular real matrix with the two blocks on the diagonal of `S`, at positions symbolized
          by the entries of `v` and `w`, swapped.
    """
    p, q = S[v, :][:, w].shape  # p and q are block sizes
    Ip = np.eye(p)
    Iq = np.eye(q)
    r = np.concatenate(
        [S[v, w[j]] for j in range(q)]
    )  # Vectorize right-hand side for Kronecker product formulation of the Sylvester equations (7).
    K = np.kron(Iq, S[v, :][:, v]) - np.kron(S[w, :][:, w].T, Ip)  # Kronecker product system matrix.
    L, H, P, Q = lu_complpiv(K, inplace=True)  # LU-decomposition of this matrix.
    e = np.min(np.abs(np.diag(H)))  # Scaling factor to prevent overflow.
    sigp = np.arange(p * q)
    for k in range(p * q - 1):  # Implement permutation P of the LU-decomposition PAQ=LU ...
        sigp[[k, P[k]]] = sigp[[P[k], k]].copy()
    r = e * r[sigp]  # ... scale and permute the right-hand side.
    try:
        x = np.linalg.solve(H, np.linalg.solve(L, r))  # and solve the two triangular systems.
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Condition number of H is {np.linalg.cond(H)}.") from e
    sigq = np.arange(p * q)
    for k in range(p * q - 1):  # Implement permutation Q of the LU-decomposition PAQ=LU ...
        sigq[[k, Q[k]]] = sigq[[Q[k], k]].copy()
    x[sigq] = x.copy()  # ... and permute the solution.
    X = np.vstack(
        [x[j * p : (j + 1) * p] for j in range(q)]
    ).T  # De-vectorize the solution back to a block, or, quit Kronecker formulation.
    Q, R = np.linalg.qr(np.vstack((-X, e * Iq)), mode="complete")  # Householder QR-decomposition of X.
    vw = list(v) + list(w)
    if not inplace:
        S = S.copy()
        U = U.copy()
    S[:, vw] = np.dot(S[:, vw], Q)  # Perform the actual swap by left- and right-multiplication of S by Q,
    S[vw, :] = np.dot(Q.T, S[vw, :])
    U[:, vw] = np.dot(U[:, vw], Q)  # and, right-multiplication of U by Q
    return U, S


# Based on the original MATLAB code bellow:
# -------------------------------------------------------------------------
#  function [U,S] = swap(U,S,v,w);
#  [p,q] = size(S(v,w)); Ip = eye(p); Iq = eye(q);
#  r = [];
#  for j=1:q
#    r = [r;S(v,w(j))];
#  end
#  K = kron(Iq,S(v,v))-kron(S(w,w)',Ip);
#  [L,H,P,Q] = lu_complpiv(K);
#  e = min(abs(diag(H)));
#  sigp = 1:p*q;
#  for k = 1:p*q-1;
#    sigp([k,P(k)]) = sigp([P(k),k]);
#  end
#  r = e*r(sigp);
#  x = (H\(L\r));
#  sigq = 1:p*q;
#  for k = 1:p*q-1;
#    sigq([k,Q(k)]) = sigq([Q(k),k]);
#  end
#  x(sigq) = x;
#  X = [];
#  for j=1:q
#    X = [X,x((j-1)*p+1:j*p)];
#  end
#  [Q,R]      = qr([-X;e*Iq]);
#  S(:,[v,w]) = S(:,[v,w])*Q;
#  S([v,w],:) = Q'*S([v,w],:);
#  U(:,[v,w]) = U(:,[v,w])*Q;
# -------------------------------------------------------------------------


def lu_complpiv(A: np.ndarray, inplace: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute the LU-decomposition of a matrix `A` with complete pivoting.

    I. e., :math:`PAQ = LU` with permutations `P`, `Q` symbolized by vectors.

    Parameters
    ----------
    A
        Square matrix.
    inplace
        Determines, if the supplied `A` matrix is used in place (``ìnplace = True``) or
        if a copy is made and used (``inplace = False``; default).

    Returns
    -------
    Tuple of the following:

        - L : Lower triangular matrix.
        - U : Upper triangular matrix.
        - P : Permutation matrix, which, when left-multiplied to `A`, reorders the rows of `A`.
        - Q : Permutation matrix, which, when right-multiplied to `A`, reorders the columns of `A`.
    """
    if not inplace or (__debug__ and expensive_asserts):
        A_inp = A  # debug
        A = A.copy()
    n = A.shape[0]
    P = np.zeros(n - 1, dtype=int)
    Q = np.zeros(n - 1, dtype=int)
    for k in range(
        n - 1
    ):  # See Golub and Van Loan, p. 118 for comments on this LU-decomposition with complete pivoting.
        Ak = A[k:n, :][:, k:n]
        rw, cl = np.unravel_index(np.argmax(np.abs(Ak), axis=None), Ak.shape)
        rw += k
        cl += k
        A[[k, rw], :] = A[[rw, k], :].copy()
        A[:, [k, cl]] = A[:, [cl, k]].copy()
        P[k] = rw
        Q[k] = cl
        if A[k, k] != 0:
            rs = slice(k + 1, n)
            A[rs, k] = A[rs, k] / A[k, k]
            A[rs, :][:, rs] = A[rs, :][:, rs] - A[rs, k][:, np.newaxis] * A[k, rs]
    U = np.tril(A.T).T
    L = np.tril(A, -1) + np.eye(n)
    if __debug__ and expensive_asserts:
        perm_p = np.arange(n)  # debug
        for k in range(n - 1):  # debug
            perm_p[[k, P[k]]] = perm_p[[P[k], k]].copy()  # debug
        perm_q = np.arange(n)  # debug
        for k in range(n - 1):  # debug
            perm_q[[k, Q[k]]] = perm_q[[Q[k], k]].copy()  # debug
        assert np.allclose(A_inp[perm_p, :][:, perm_q], np.dot(L, U))  # debug
    return L, U, P, Q


# Based on the original MATLAB code bellow:
# -------------------------------------------------------------------------
#    function [L,U,P,Q] = lu_complpiv(A);
#    P = []; Q = []; n = size(A,1);
#    for k=1:n-1;
#      [a,r] = max(abs(A(k:n,k:n)));
#      [dummy,c] = max(abs(a));
#      cl  = c+k-1;
#      rw  = r(c)+k-1;
#      A([k,rw],:) = A([rw,k],:);
#      A(:,[k,cl]) = A(:,[cl,k]);
#      P(k) = rw; Q(k) = cl;
#      if A(k,k) ~= 0;
#        rs = k+1:n;
#        A(rs,k)  = A(rs,k)/A(k,k);
#        A(rs,rs) = A(rs,rs)-A(rs,k)*A(k,rs);
#      end
#    end
#    U = tril(A')'; L = tril(A,-1) + eye(n);
# -------------------------------------------------------------------------


if __name__ == "__main__":
    import scipy
    import scipy.linalg

    expensive_asserts = True
    for _ in range(100):
        n = np.random.randint(2, 50)
        A = np.random.randn(n, n)
        if n % 10 == 0:
            z = np.inf
        else:
            z = float(
                np.random.randn(1)
            )  # + abs(float(np.random.randn(1)))*1j # TODO: rewrite the whole test to cover complex
        R, Q = scipy.linalg.schur(A, output="real")
        T, Z = scipy.linalg.rsf2csf(R, Q)
        ev_orig = np.diag(T)
        delta_orig = np.abs(ev_orig - z)
        eps = np.finfo(R.dtype).eps
        assert np.allclose(np.dot(A, Q), np.dot(Q, R))
        r = np.count_nonzero(np.abs(np.diag(R, -1)) > 100 * eps)

        Q, R, ap = sort_real_schur(Q, R, z, 0, inplace=(n % 2 == 0))

        # TODO: move the assersion comments to messages
        assert np.allclose(np.dot(A, Q), np.dot(Q, R))  # check that still a decomposition of the original matrix

        # test that Q and R have the correct structure
        assert np.allclose(np.dot(Q, Q.T), np.eye(A.shape[0]))  # Q orthonormal
        assert np.all(np.tril(R, -2) == 0)  # R triangular
        assert r == np.count_nonzero(np.abs(np.diag(R, -1)) > 100 * eps)  # number of blocks in R is preserved

        # check that eigenvalues are sorted
        T, Z = scipy.linalg.rsf2csf(R, Q)
        ev = np.diag(T)
        np.allclose(sorted(ev), sorted(ev_orig))  # check that eigenvalues were preserved
        if np.isinf(z):
            delta = -np.abs(ev)
        else:
            delta = np.abs(ev - z)
        assert np.all(delta[0:-1] <= delta[1:] + 100 * eps), (np.max(delta[0:-1] - delta[1:]), delta)
