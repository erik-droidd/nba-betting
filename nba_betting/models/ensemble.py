"""Ensemble combining Elo and gradient-boosted predictions.

Uses **log-odds (logit) averaging** rather than naive probability averaging.
Averaging in probability space pulls extreme predictions toward 50% (e.g.
mean(0.95, 0.85) = 0.90, but the corresponding log-odds 2.94 + 1.73 → mean
2.34 → 0.91). On near-extreme NBA games this small difference compounds
across the full slate and is one source of probability compression.

The ensemble weight is no longer hardcoded — `learn_ensemble_weight()`
selects it via grid search on a held-out walk-forward fold by minimizing
log-loss. The chosen weight is persisted to disk and loaded at predict
time, with a sensible default if the artifact is missing.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from nba_betting.config import MODELS_DIR

ENSEMBLE_WEIGHT_PATH = MODELS_DIR / "ensemble_weight.joblib"

# Fallback if no learned weight has been saved yet. 0.3 weight on Elo
# matches the prior hardcoded value, but the goal is to replace it via
# learn_ensemble_weight().
DEFAULT_ELO_WEIGHT = 0.3


def _logit(p):
    """Numerically stable logit. Accepts scalars or numpy arrays."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    """Inverse of _logit."""
    return 1.0 / (1.0 + np.exp(-x))


def ensemble_predict(
    elo_prob: float,
    xgb_prob: float,
    elo_weight: float | None = None,
) -> float:
    """Combine Elo and XGBoost predictions via log-odds weighted average.

    Args:
        elo_prob: P(home win) from Elo model.
        xgb_prob: P(home win) from calibrated XGBoost.
        elo_weight: Weight for Elo's log-odds (0-1). XGBoost gets the
            complement. If None, loads the learned weight from disk.

    Returns:
        Combined P(home win).
    """
    if elo_weight is None:
        elo_weight = load_ensemble_weight()
    xgb_weight = 1.0 - elo_weight
    z = elo_weight * _logit(elo_prob) + xgb_weight * _logit(xgb_prob)
    return float(_sigmoid(z))


def ensemble_predict_batch(
    elo_probs: np.ndarray,
    xgb_probs: np.ndarray,
    elo_weight: float | None = None,
) -> np.ndarray:
    """Batch version of ensemble_predict (vectorized log-odds blend)."""
    if elo_weight is None:
        elo_weight = load_ensemble_weight()
    xgb_weight = 1.0 - elo_weight
    z = elo_weight * _logit(elo_probs) + xgb_weight * _logit(xgb_probs)
    return _sigmoid(z)


def learn_ensemble_weight(
    elo_probs: np.ndarray,
    xgb_probs: np.ndarray,
    y_true: np.ndarray,
    grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> tuple[float, dict[float, float]]:
    """Grid-search the optimal Elo weight by minimizing log-loss.

    Pure-Python: avoids fitting a sklearn model on top of two probability
    columns. Returns the best weight and the full {weight: log_loss} table
    so callers can log it.
    """
    from sklearn.metrics import log_loss

    elo_probs = np.asarray(elo_probs, dtype=float)
    xgb_probs = np.asarray(xgb_probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    table: dict[float, float] = {}
    for w in grid:
        blended = ensemble_predict_batch(elo_probs, xgb_probs, elo_weight=w)
        table[float(w)] = float(log_loss(y_true, blended, labels=[0, 1]))
    best = min(table.items(), key=lambda kv: kv[1])
    return best[0], table


def save_ensemble_weight(weight: float) -> Path:
    """Persist the learned ensemble weight."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(float(weight), ENSEMBLE_WEIGHT_PATH)
    return ENSEMBLE_WEIGHT_PATH


def load_ensemble_weight() -> float:
    """Load the learned ensemble weight, falling back to DEFAULT_ELO_WEIGHT."""
    if not ENSEMBLE_WEIGHT_PATH.exists():
        return DEFAULT_ELO_WEIGHT
    try:
        return float(joblib.load(ENSEMBLE_WEIGHT_PATH))
    except Exception:
        return DEFAULT_ELO_WEIGHT
