"""Shared quadratic-program data types used across the CVI modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class ConstraintSet:
    """Linear constraint block l <= A x <= u (paper's canonical QP form,
    Section "Canonical QP Problem")."""

    A: sp.csc_matrix
    l: np.ndarray
    u: np.ndarray
    names: list[str]
