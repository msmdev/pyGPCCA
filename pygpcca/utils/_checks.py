from typing import Tuple, Union, Optional

from scipy.sparse import issparse, spmatrix
import numpy as np

from pygpcca.utils._docs import d

__all__ = ["ensure_ndarray_or_sparse"]


@d.get_sections(base="assert_array", sections=["Parameters"])
def assert_array(
    A: Union[np.ndarray, spmatrix],
    shape: Optional[Tuple[int, ...]] = None,
    uniform: Optional[bool] = None,
    ndim: Optional[int] = None,
    size: Optional[int] = None,
    dtype: Optional[Union[type, np.dtype]] = None,  # type: ignore[type-arg]
    kind: Optional[str] = None,
) -> None:
    """
    Assert whether the given array or sparse matrix has the given properties.

    Parameters
    ----------
    A
        The array-like object under investigation.
    shape
        Assert if the array has the requested shape. Be careful with vectors because this will distinguish between row
        vectors `(1, n)`, column vectors `(n, 1)` and arrays `(n,)`. If you want to be less specific,
        consider using ``size`` option.
    uniform
        If not `None`, asserts whether the array dimensions are uniform (e.g. square for a ``ndim=2`` array) or not.
    ndim
        Assert if the array has the requested dimension.
    size
        Assert if the array has the requested number of elements.
    dtype
        Assert if the array data has the requested data type. This check is strong, e.g. int and int64 are not equal.
        If you want a weaker check, consider the ``kind`` option.
    kind
        Check if the array is of the specified kind. Options include 'i' for integer types, 'f' for  float types.
        Check :attr:`numpy.dtype.kind` for possible options. An additional option is 'numeric' for either
        :class:`integer` or :class`float`.

    Returns
    -------
    Nothing, just performs aforementioned the checks.

    Raises
    ------
    AssertionError
        If assertions have failed.
    """
    try:
        if shape is not None:
            if not np.array_equal(np.shape(A), shape):
                raise AssertionError(f"Expected shape {shape}, but given array has shape {np.shape(A)}.")
        if uniform is not None:
            shapearr = np.array(np.shape(A))
            is_uniform = np.count_nonzero(shapearr - shapearr[0]) == 0
            if uniform and not is_uniform:
                raise AssertionError(f"Given array is not uniform: {shapearr}.")
            elif not uniform and is_uniform:
                raise AssertionError(f"Given array is not nonuniform: {shapearr}.")
        if size is not None:
            if not np.size(A) == size:
                raise AssertionError(f"Expected size {size}, but given array has size {np.size(A)}.")
        if ndim is not None:
            if not ndim == np.ndim(A):
                raise AssertionError(f"Expected shape {ndim} but given array has shape {np.ndim(A)}.")
        if dtype is not None:
            # now we must create an array if we don't have one yet
            if not isinstance(A, np.ndarray) and not issparse(A):
                A = np.array(A)
            if not np.dtype(dtype) == A.dtype:
                raise AssertionError(f"Expected data type {dtype} but given array has data type {A.dtype}.")
        if kind is not None:
            # now we must create an array if we don't have one yet
            if not isinstance(A, np.ndarray) and not issparse(A):
                A = np.array(A)
            if kind == "numeric":
                if not (A.dtype.kind == "i" or A.dtype.kind == "f"):
                    raise AssertionError(f"Expected numerical data, but given array has data kind {A.dtype.kind}.")
            elif not A.dtype.kind == kind:
                raise AssertionError(f"Expected data kind {kind} but given array has data kind {A.dtype.kind}.")
    except AssertionError:
        raise
    except Exception as e:
        raise AssertionError(
            f"Given argument is not an array of the expected shape or type: {A}, type={type(A).__name__}."
        ) from e


@d.dedent
def ensure_ndarray_or_sparse(
    A: Union[np.ndarray, spmatrix],
    shape: Optional[Tuple[int, ...]] = None,
    uniform: Optional[bool] = None,
    ndim: Optional[int] = None,
    size: Optional[int] = None,
    dtype: Optional[Union[type, np.dtype]] = None,  # type: ignore[type-arg]
    kind: Optional[str] = None,
) -> Union[np.ndarray, spmatrix]:
    """
    Ensure ``A`` is an array or a sparse matrix and assert that the given parameters match.

    Parameters
    ----------
    %(assert_array.parameters)s

    Returns
    -------
    If ``A`` is an already valid :class:`numpy.ndarray` or :class:`scipy.sparse.spmatrix`, then it is simply returned.
    Otherwise returns a :class:`numpy.ndarray` copy of the array-like object ``A``.
    """
    if not isinstance(A, np.ndarray) and not issparse(A):
        try:
            A = np.array(A)
        except Exception as e:
            raise AssertionError(f"Given argument cannot be converted to an numpy.ndarray: {A}.") from e
    assert_array(A, shape=shape, uniform=uniform, ndim=ndim, size=size, dtype=dtype, kind=kind)

    return A
