"""Gradient boosting model for NBA game prediction with walk-forward validation.

Uses sklearn's HistGradientBoostingClassifier (LightGBM-like, no external deps).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)

from nba_betting.config import MODELS_DIR


MODEL_PATH = MODELS_DIR / "gbm_latest.joblib"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.joblib"
FEATURE_MEANS_PATH = MODELS_DIR / "feature_means.joblib"

# Default hyperparameters
DEFAULT_PARAMS = {
    "max_iter": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 42,
    "verbose": 0,
    "early_stopping": True,
    "n_iter_no_change": 20,
    "validation_fraction": 0.15,
}


def _get_feature_cols(X: pd.DataFrame) -> list[str]:
    """Get feature columns (excluding metadata columns)."""
    return [c for c in X.columns if not c.startswith("_")]


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict = None,
) -> HistGradientBoostingClassifier:
    """Train a gradient boosting model on the full provided dataset."""
    if params is None:
        params = DEFAULT_PARAMS.copy()

    feature_cols = _get_feature_cols(X)
    # Fit on DataFrame so feature names are preserved (consistent with prediction)
    X_train = X[feature_cols]

    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y.values)

    return model


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 3,
    params: dict = None,
) -> dict:
    """Walk-forward validation across seasons.

    Splits data chronologically: trains on earlier data, tests on later.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
        # Disable early stopping for walk-forward (we have explicit train/test)
        wf_params = {**params, "early_stopping": False}
    else:
        wf_params = params

    if "_date" not in X.columns:
        raise ValueError("X must contain _date column for temporal splitting")

    feature_cols = _get_feature_cols(X)
    dates = pd.to_datetime(X["_date"])

    X_sorted = X.copy()
    X_sorted["_date_parsed"] = dates
    X_sorted = X_sorted.sort_values("_date_parsed")
    y_sorted = y.loc[X_sorted.index]

    # Split at July 1 of each year (NBA season boundary)
    years = X_sorted["_date_parsed"].dt.year.unique()
    split_dates = []
    for yr in sorted(years):
        july = pd.Timestamp(f"{yr}-07-01")
        if X_sorted["_date_parsed"].min() < july < X_sorted["_date_parsed"].max():
            split_dates.append(july)

    if len(split_dates) < 1:
        return {"folds": [], "aggregate": {}}

    split_dates = split_dates[-n_splits:]

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_calibrated = []

    for i, split_date in enumerate(split_dates):
        train_mask = X_sorted["_date_parsed"] < split_date

        if i + 1 < len(split_dates):
            test_mask = (X_sorted["_date_parsed"] >= split_date) & (
                X_sorted["_date_parsed"] < split_dates[i + 1]
            )
        else:
            test_mask = X_sorted["_date_parsed"] >= split_date

        if train_mask.sum() < 100 or test_mask.sum() < 50:
            continue

        # Per-fold isotonic calibration (#6 improvement): hold out the
        # last 20% of the training fold as a calibration set. The model
        # is fit on the first 80%, isotonic is fit on the held-out 20%,
        # and the test fold is scored with the calibrated probabilities.
        # This mirrors what happens at train time and gives honest
        # per-fold ECE numbers.
        train_idx = X_sorted.index[train_mask]
        cal_cutoff = int(len(train_idx) * 0.8)
        train_fit_idx = train_idx[:cal_cutoff]
        cal_idx = train_idx[cal_cutoff:]

        X_tr = X_sorted.loc[train_fit_idx, feature_cols]
        y_tr = y_sorted.loc[train_fit_idx].values
        X_cal = X_sorted.loc[cal_idx, feature_cols]
        y_cal = y_sorted.loc[cal_idx].values
        X_te = X_sorted.loc[test_mask, feature_cols]
        y_te = y_sorted.loc[test_mask].values

        if len(X_tr) < 80 or len(X_cal) < 20:
            # Fall back to full training without calibration split
            X_tr = X_sorted.loc[train_mask, feature_cols]
            y_tr = y_sorted.loc[train_mask].values
            X_cal = None
            y_cal = None

        model = HistGradientBoostingClassifier(**wf_params)
        model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Per-fold calibration when we have a held-out calibration set
        y_prob_cal = y_prob
        cal_ece = None
        if X_cal is not None and y_cal is not None and len(y_cal) >= 20:
            try:
                from nba_betting.models.calibration import calibrate_model, evaluate_calibration
                cal_model = calibrate_model(model, X_cal, y_cal, method="isotonic")
                y_prob_cal = cal_model.predict_proba(X_te)[:, 1]
                cal_eval = evaluate_calibration(y_te, y_prob_cal)
                cal_ece = cal_eval["ece"]
            except Exception:
                y_prob_cal = y_prob  # Fall back to uncalibrated

        acc = accuracy_score(y_te, y_pred)
        brier = brier_score_loss(y_te, y_prob)
        ll = log_loss(y_te, y_prob)

        # Also report calibrated metrics when available
        brier_cal = brier_score_loss(y_te, y_prob_cal) if y_prob_cal is not y_prob else None
        ll_cal = log_loss(y_te, y_prob_cal) if y_prob_cal is not y_prob else None

        fold_result = {
            "fold": i + 1,
            "split_date": str(split_date.date()),
            "train_size": int(train_mask.sum()),
            "test_size": int(test_mask.sum()),
            "accuracy": round(acc, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
        }
        if brier_cal is not None:
            fold_result["brier_calibrated"] = round(brier_cal, 4)
        if ll_cal is not None:
            fold_result["log_loss_calibrated"] = round(ll_cal, 4)
        if cal_ece is not None:
            fold_result["ece_calibrated"] = round(cal_ece, 4)

        fold_results.append(fold_result)

        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_prob.tolist())
        all_y_pred_calibrated.extend(y_prob_cal.tolist())

    aggregate = {}
    if all_y_true:
        all_y_true_arr = np.array(all_y_true)
        all_y_pred_arr = np.array(all_y_pred)
        all_y_cal_arr = np.array(all_y_pred_calibrated)
        aggregate = {
            "accuracy": round(accuracy_score(all_y_true_arr, (all_y_pred_arr >= 0.5).astype(int)), 4),
            "brier_score": round(brier_score_loss(all_y_true_arr, all_y_pred_arr), 4),
            "log_loss": round(log_loss(all_y_true_arr, all_y_pred_arr), 4),
            "total_games": len(all_y_true),
        }
        # Add calibrated aggregate when per-fold calibration ran
        if len(all_y_cal_arr) == len(all_y_true_arr):
            try:
                aggregate["brier_calibrated"] = round(
                    brier_score_loss(all_y_true_arr, all_y_cal_arr), 4,
                )
                aggregate["log_loss_calibrated"] = round(
                    log_loss(all_y_true_arr, all_y_cal_arr), 4,
                )
            except Exception:
                pass

    return {"folds": fold_results, "aggregate": aggregate}


def search_hyperparams(
    X: pd.DataFrame,
    y: pd.Series,
    grid: list[dict] | None = None,
    n_splits: int = 3,
    metric: str = "log_loss_calibrated",
) -> dict:
    """Tier 2.1 — grid-search hyperparameters via walk-forward validation.

    Each grid point is a full override dict (merged on top of DEFAULT_PARAMS),
    evaluated end-to-end with ``walk_forward_validate``. The best config is
    selected by minimizing ``metric`` on the aggregate across folds — by
    default the calibrated log-loss (falls back to raw log_loss if per-fold
    calibration was skipped). Grid is intentionally small (8 combos) to
    keep runtime bounded on a ~6k-row training set.

    Returns:
        {"best_params": {...}, "best_score": float, "all_results": [...]}.
        The caller writes ``best_params`` into config / persists them so
        subsequent ``train_model`` calls pick them up.
    """
    if grid is None:
        # Compact grid spanning depth / LR / iter / regularization. All
        # combos produce models in ~15 seconds each on the full history.
        grid = [
            {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300, "l2_regularization": 1.0},
            {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300, "l2_regularization": 1.0},
            {"max_depth": 5, "learning_rate": 0.05, "max_iter": 300, "l2_regularization": 1.0},
            {"max_depth": 4, "learning_rate": 0.03, "max_iter": 500, "l2_regularization": 1.0},
            {"max_depth": 4, "learning_rate": 0.08, "max_iter": 200, "l2_regularization": 1.0},
            {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300, "l2_regularization": 0.5},
            {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300, "l2_regularization": 2.0},
            {"max_depth": 3, "learning_rate": 0.08, "max_iter": 200, "l2_regularization": 0.5},
        ]

    results = []
    best_score = float("inf")
    best_params: dict | None = None

    for i, overrides in enumerate(grid):
        params = {**DEFAULT_PARAMS, **overrides}
        try:
            wf = walk_forward_validate(X, y, n_splits=n_splits, params=params)
        except Exception as e:
            results.append({"overrides": overrides, "error": str(e)})
            continue

        agg = wf.get("aggregate") or {}
        # Prefer calibrated log-loss; fall back to raw if calibration was
        # skipped (small fold → not enough data for isotonic).
        score = agg.get(metric)
        if score is None:
            score = agg.get("log_loss", float("inf"))

        entry = {
            "overrides": overrides,
            "score": round(float(score), 4),
            "aggregate": agg,
        }
        results.append(entry)

        if score < best_score:
            best_score = float(score)
            best_params = params

    # Fallback if every grid point errored
    if best_params is None:
        best_params = DEFAULT_PARAMS.copy()

    return {
        "best_params": best_params,
        "best_score": round(best_score, 4) if best_score != float("inf") else None,
        "all_results": results,
    }


BEST_PARAMS_PATH = MODELS_DIR / "best_hyperparams.joblib"


def save_best_hyperparams(params: dict) -> Path:
    """Persist grid-search-winning hyperparams so ``train`` can pick them up."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(params, BEST_PARAMS_PATH)
    return BEST_PARAMS_PATH


def load_best_hyperparams() -> dict | None:
    """Load grid-search-winning hyperparams. Returns None if not saved yet."""
    if not BEST_PARAMS_PATH.exists():
        return None
    try:
        return joblib.load(BEST_PARAMS_PATH)
    except Exception:
        return None


def save_model(
    model: HistGradientBoostingClassifier,
    feature_cols: list[str],
    feature_means: dict[str, float] | None = None,
) -> Path:
    """Save trained model, feature column list, and feature means for imputation."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    if feature_means is not None:
        joblib.dump(feature_means, FEATURE_MEANS_PATH)
    return MODEL_PATH


def _file_mtime(path: Path) -> float:
    """Return file mtime, or 0 if missing — used as a cache invalidation key."""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


# Tier 3.1 — in-memory cache keyed by file mtime. The API route previously
# re-joblib-loaded the GBM (and feature means) on every request; that's a
# ~200–500ms disk hit per prediction. We cache them as module-level
# globals and invalidate automatically when the artifact is retrained
# (mtime changes).
_MODEL_CACHE: dict[str, object] = {}
_MEANS_CACHE: dict[str, object] = {}


def load_model(force_reload: bool = False) -> tuple[HistGradientBoostingClassifier, list[str]] | None:
    """Load trained model and feature columns. Returns None if not found.

    Cached in-process: the same Python process will only deserialize the
    model once per mtime. Pass ``force_reload=True`` to bypass the cache
    (useful from CLI train-followed-by-predict flows).
    """
    if not MODEL_PATH.exists() or not FEATURE_COLS_PATH.exists():
        return None

    model_mtime = _file_mtime(MODEL_PATH)
    cols_mtime = _file_mtime(FEATURE_COLS_PATH)
    cache_key = (model_mtime, cols_mtime)

    cached = _MODEL_CACHE.get("value")
    if not force_reload and cached is not None and _MODEL_CACHE.get("key") == cache_key:
        return cached  # type: ignore[return-value]

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    payload = (model, feature_cols)
    _MODEL_CACHE["value"] = payload
    _MODEL_CACHE["key"] = cache_key
    return payload


def load_feature_means(force_reload: bool = False) -> dict[str, float] | None:
    """Load saved feature means for prediction imputation. Returns None if not found.

    Cached in-process like ``load_model``.
    """
    if not FEATURE_MEANS_PATH.exists():
        return None
    mtime = _file_mtime(FEATURE_MEANS_PATH)
    cached = _MEANS_CACHE.get("value")
    if not force_reload and cached is not None and _MEANS_CACHE.get("key") == mtime:
        return cached  # type: ignore[return-value]
    means = joblib.load(FEATURE_MEANS_PATH)
    _MEANS_CACHE["value"] = means
    _MEANS_CACHE["key"] = mtime
    return means


def get_feature_importance(
    model: HistGradientBoostingClassifier,
    feature_cols: list[str],
    X: pd.DataFrame | np.ndarray | None = None,
    y: pd.Series | np.ndarray | None = None,
    top_n: int = 15,
) -> list[tuple[str, float]]:
    """Compute permutation feature importance.

    HistGradientBoostingClassifier has no native feature_importances_, so we
    use sklearn's permutation_importance on a sample of the training data.
    """
    if X is None or y is None or len(feature_cols) == 0:
        return [(c, 0.0) for c in feature_cols[:top_n]]

    from sklearn.inspection import permutation_importance

    # Use DataFrame to preserve feature names (model was trained with names)
    if isinstance(X, pd.DataFrame):
        X_df = X[feature_cols]
    else:
        X_df = pd.DataFrame(X, columns=feature_cols)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)

    # Sample for speed if dataset is large
    n = len(X_df)
    if n > 1000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=1000, replace=False)
        X_sample = X_df.iloc[idx]
        y_sample = y_arr[idx]
    else:
        X_sample = X_df
        y_sample = y_arr

    try:
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
            result = permutation_importance(
                model, X_sample, y_sample,
                n_repeats=5, random_state=42, n_jobs=1, scoring="neg_log_loss",
            )
        importances = result.importances_mean
    except Exception:
        importances = np.zeros(len(feature_cols))

    pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    return pairs[:top_n]
