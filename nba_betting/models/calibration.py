"""Probability calibration via isotonic regression or Platt scaling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from nba_betting.config import MODELS_DIR


CALIBRATED_MODEL_PATH = MODELS_DIR / "gbm_calibrated.joblib"


def calibrate_model(
    model,
    X_cal,
    y_cal: np.ndarray,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """Calibrate a trained model using isotonic regression or Platt scaling.

    Default is isotonic. Platt's sigmoid forces an S-curve through the
    residuals, which systematically squeezes both tails toward 50% on a
    near-balanced binary problem like NBA home wins (~57% home win rate).
    That compression generates phantom underdog edges. Isotonic regression
    is non-parametric — it learns whatever monotonic mapping minimizes
    log-loss on the calibration set without imposing a shape, so extreme
    probabilities are preserved.

    Args:
        model: Trained classifier.
        X_cal: Calibration feature matrix (DataFrame preferred to preserve feature names).
        y_cal: Calibration targets.
        method: "isotonic" (default, recommended) or "sigmoid" (Platt).

    Returns:
        CalibratedClassifierCV wrapping the original model.
    """
    import warnings as _warnings
    # sklearn 1.8+ removed cv="prefit", use FrozenEstimator instead
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(model),
            method=method,
        )
    except ImportError:
        # Older sklearn: use cv="prefit"
        calibrated = CalibratedClassifierCV(
            model,
            method=method,
            cv="prefit",
        )
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
        calibrated.fit(X_cal, y_cal)
    return calibrated


def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Compute calibration metrics.

    Returns:
        Dict with brier_score, expected_calibration_error, and per-bin data.
    """
    brier = brier_score_loss(y_true, y_prob)

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins_data = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        avg_predicted = y_prob[mask].mean()
        avg_actual = y_true[mask].mean()
        gap = abs(avg_predicted - avg_actual)
        ece += (n_in_bin / len(y_true)) * gap

        bins_data.append({
            "bin_start": round(bin_edges[i], 2),
            "bin_end": round(bin_edges[i + 1], 2),
            "count": int(n_in_bin),
            "avg_predicted": round(avg_predicted, 4),
            "avg_actual": round(avg_actual, 4),
            "gap": round(gap, 4),
        })

    return {
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "bins": bins_data,
    }


def save_calibrated_model(model: CalibratedClassifierCV) -> Path:
    """Save calibrated model."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, CALIBRATED_MODEL_PATH)
    return CALIBRATED_MODEL_PATH


def load_calibrated_model() -> CalibratedClassifierCV | None:
    """Load calibrated model. Returns None if not found."""
    if not CALIBRATED_MODEL_PATH.exists():
        return None
    return joblib.load(CALIBRATED_MODEL_PATH)
