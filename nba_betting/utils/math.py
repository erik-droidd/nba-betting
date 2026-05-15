"""Shared numeric utilities."""
from __future__ import annotations

import math

import numpy as np


def logit(p):
    """Numerically stable logit. Accepts scalars or numpy arrays."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    """Inverse of logit. Accepts scalars or numpy arrays."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def logit_scalar(p: float) -> float:
    """Scalar logit for use in non-numpy contexts."""
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return math.log(p / (1.0 - p))


def sigmoid_scalar(x: float) -> float:
    """Scalar sigmoid for use in non-numpy contexts."""
    return 1.0 / (1.0 + math.exp(-float(x)))
