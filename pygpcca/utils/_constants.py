import numpy as np

EPS = np.finfo(np.float64).eps

DEFAULT_SCHUR_METHOD = "brandts"
NO_PETSC_SLEPC_FOUND_MSG = (
    "Unable to import PETSc or SLEPc.\n"
    "You can install it from: https://slepc4py.readthedocs.io/en/stable/install.html\n"
    f"Defaulting to `method={DEFAULT_SCHUR_METHOD!r}`."
)
