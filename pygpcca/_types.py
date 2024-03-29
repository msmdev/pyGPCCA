from typing import Any, Optional

import numpy as np

__all__ = ["ArrayLike"]

try:
    from numpy.typing import NDArray

    ArrayLike = NDArray[Any]
except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]


OArray = Optional[ArrayLike]
