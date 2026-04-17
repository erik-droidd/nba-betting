"""Stacked meta-learner (Tier 2.2) — logistic-regression blender on top
of base Elo + GBM log-odds.

Why a meta-learner: the current ensemble uses a single grid-searched
Elo weight that's constant across games. But the Elo model's skill
varies (e.g., fantastic on mismatched teams, weak on close games with
injury noise) and so does the GBM's. A small logistic regression taking
the **log-odds** of both predictions as features — optionally with a
couple of light context features — learns a game-dependent blend.

Keeping the blender tiny (≤4 coefficients) preserves interpretability
and avoids overfitting: we are training on the out-of-fold predictions
from walk-forward validation, which is naturally a small calibration
set (one season's worth of games).

Falls back gracefully to the existing log-odds grid-search ensemble if
the meta-model hasn't been trained yet — callers check for the artifact
via `load_meta_model()` and use the simpler blender when it's absent.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from nba_betting.config import MODELS_DIR
from nba_betting.models.ensemble import _logit, _sigmoid

META_MODEL_PATH = MODELS_DIR / "ensemble_meta.joblib"


def _build_meta_X(
    elo_probs: np.ndarray,
    xgb_probs: np.ndarray,
    context: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Assemble the meta-learner feature matrix.

    Base features: ``logit(elo_prob)``, ``logit(xgb_prob)``, and their
    disagreement ``|logit(elo) - logit(xgb)|`` — when the two base models
    disagree strongly, the meta-learner tends to weight one direction or
    the other rather than blending.

    Optional context features (rest differential, injury impact diff, etc.)
    can be passed in the ``context`` dict; they're appended as additional
    columns. Must be the same length as elo_probs/xgb_probs.
    """
    elo_probs = np.asarray(elo_probs, dtype=float)
    xgb_probs = np.asarray(xgb_probs, dtype=float)

    elo_logit = _logit(elo_probs)
    xgb_logit = _logit(xgb_probs)
    disagreement = np.abs(elo_logit - xgb_logit)

    cols = [elo_logit, xgb_logit, disagreement]
    if context:
        for name in sorted(context.keys()):
            col = np.asarray(context[name], dtype=float)
            if len(col) != len(elo_probs):
                raise ValueError(
                    f"context feature '{name}' length {len(col)} != "
                    f"base-prob length {len(elo_probs)}"
                )
            cols.append(col)
    return np.column_stack(cols)


def fit_meta_model(
    elo_probs: np.ndarray,
    xgb_probs: np.ndarray,
    y_true: np.ndarray,
    context: dict[str, np.ndarray] | None = None,
    C: float = 1.0,
) -> dict:
    """Fit a logistic regression meta-learner.

    Args:
        elo_probs: P(home win) from Elo, one per out-of-fold game.
        xgb_probs: P(home win) from calibrated GBM, aligned.
        y_true: 0/1 home-win outcomes.
        context: optional dict of named context features (each an array).
        C: LR regularization strength (smaller = stronger regularization).

    Returns:
        {"model": fitted LogisticRegression, "context_keys": [...],
         "n_train": int, "train_log_loss": float}

    The saved dict can be round-tripped through save/load helpers.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    X = _build_meta_X(elo_probs, xgb_probs, context)
    y = np.asarray(y_true, dtype=int)

    # Very light L2, no intercept bias because the base log-odds already
    # carry a centered signal. Liblinear is fine on a ~1000-row meta-fit.
    model = LogisticRegression(
        C=C, solver="liblinear", fit_intercept=True, max_iter=200,
    )
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]

    return {
        "model": model,
        "context_keys": sorted(context.keys()) if context else [],
        "n_train": int(len(y)),
        "train_log_loss": float(log_loss(y, proba, labels=[0, 1])),
    }


def predict_meta(
    elo_probs,
    xgb_probs,
    context: dict | None = None,
    artifact: dict | None = None,
) -> np.ndarray:
    """Apply a fitted meta-learner to produce blended P(home win).

    Accepts either scalars or arrays. If ``artifact`` is None, loads
    from disk — raises FileNotFoundError if nothing is saved.
    """
    if artifact is None:
        artifact = load_meta_model()
        if artifact is None:
            raise FileNotFoundError(
                "No meta-model saved. Call fit_meta_model() + save_meta_model() "
                "or fall back to ensemble_predict()."
            )

    elo_arr = np.atleast_1d(np.asarray(elo_probs, dtype=float))
    xgb_arr = np.atleast_1d(np.asarray(xgb_probs, dtype=float))

    # If the fitted model used context, the caller must supply matching keys.
    ctx_keys = artifact.get("context_keys") or []
    ctx_for_build: dict[str, np.ndarray] | None = None
    if ctx_keys:
        if context is None:
            raise ValueError(
                f"meta-model requires context keys {ctx_keys} but none supplied"
            )
        missing = [k for k in ctx_keys if k not in context]
        if missing:
            raise ValueError(f"missing context keys for meta-model: {missing}")
        ctx_for_build = {
            k: np.atleast_1d(np.asarray(context[k], dtype=float)) for k in ctx_keys
        }

    X = _build_meta_X(elo_arr, xgb_arr, ctx_for_build)
    proba = artifact["model"].predict_proba(X)[:, 1]
    return proba


def save_meta_model(artifact: dict) -> Path:
    """Persist the fitted meta-model artifact dict."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, META_MODEL_PATH)
    return META_MODEL_PATH


def load_meta_model() -> dict | None:
    """Load the fitted meta-model, or None if not yet trained."""
    if not META_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(META_MODEL_PATH)
    except Exception:
        return None
