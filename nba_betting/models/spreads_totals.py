"""Spread and total (over/under) regression heads.

The main classifier predicts P(home win). This module adds two regressors
that predict:

- **spread** — expected (home_score - away_score). Negative = away favored.
- **total** — expected (home_score + away_score).

We use the exact same feature matrix as the win-probability model
(minus the injury score-leak targets) and train a separate
`HistGradientBoostingRegressor` for each target. At prediction time we
compare the predicted value against the market line and emit a pick
when the gap exceeds a minimum edge threshold.

Why two regression heads instead of one joint model? Spreads and totals
are empirically near-independent — a team can be favored by 10 in a
pace-down game or a pace-up game — so fitting them separately is both
simpler and marginally more accurate. It also means we can reuse the
GBM + walk-forward + NaN-imputation machinery we already trust for the
classifier.

Artifact layout:
    trained_models/
      spread_regressor.joblib   # HistGradientBoostingRegressor
      total_regressor.joblib    # HistGradientBoostingRegressor
      regressor_feature_cols.joblib  # list[str] (same cols for both)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from nba_betting.config import MODELS_DIR

SPREAD_PATH = Path(MODELS_DIR) / "spread_regressor.joblib"
TOTAL_PATH = Path(MODELS_DIR) / "total_regressor.joblib"
REG_COLS_PATH = Path(MODELS_DIR) / "regressor_feature_cols.joblib"


# Empirical NBA defaults — used as safe priors when the regressors
# haven't been trained yet.
_DEFAULT_TOTAL = 225.0
_DEFAULT_SPREAD = 0.0

# Minimum gap (points) between model and market before we emit a pick.
# NBA point-spread markets are efficient; 1.5 pts is a reasonable floor
# that still lets through cases where the model strongly disagrees with
# the book. For totals the threshold is a bit looser because book
# disagreement on pace tends to be noisier.
SPREAD_EDGE_PTS = 1.5
TOTAL_EDGE_PTS = 2.5


def _get_feature_cols(X: pd.DataFrame) -> list[str]:
    """Same rule as xgboost_model._get_feature_cols — drop _-prefixed cols."""
    return [c for c in X.columns if not c.startswith("_")]


def train_spread_total_regressors(
    X: pd.DataFrame,
) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor, dict]:
    """Train both regressors on the full historical matrix.

    Requires the caller passed an X that carries `_home_score` and
    `_away_score` metadata columns (populated by `build_feature_matrix`).
    Returns (spread_model, total_model, metrics_dict).
    """
    if "_home_score" not in X.columns or "_away_score" not in X.columns:
        raise ValueError(
            "Feature matrix missing _home_score/_away_score metadata. "
            "Rebuild with build_feature_matrix()."
        )

    feature_cols = _get_feature_cols(X)
    X_features = X[feature_cols]

    home_scores = pd.to_numeric(X["_home_score"], errors="coerce")
    away_scores = pd.to_numeric(X["_away_score"], errors="coerce")

    y_spread = (home_scores - away_scores).astype(float)
    y_total = (home_scores + away_scores).astype(float)

    # Drop rows with no score (shouldn't happen post-build_feature_matrix
    # but we defend anyway)
    mask = y_spread.notna() & y_total.notna()
    X_features = X_features[mask]
    y_spread = y_spread[mask]
    y_total = y_total[mask]

    # Simple 80/20 temporal holdout for validation metrics, using the
    # `_date` metadata column if present on the original X.
    n = len(X_features)
    split = int(n * 0.8)

    params = dict(
        max_iter=400,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42,
    )

    spread_model = HistGradientBoostingRegressor(**params)
    spread_model.fit(X_features.iloc[:split], y_spread.iloc[:split])
    spread_pred = spread_model.predict(X_features.iloc[split:])
    spread_mae = float(np.mean(np.abs(spread_pred - y_spread.iloc[split:].values)))

    total_model = HistGradientBoostingRegressor(**params)
    total_model.fit(X_features.iloc[:split], y_total.iloc[:split])
    total_pred = total_model.predict(X_features.iloc[split:])
    total_mae = float(np.mean(np.abs(total_pred - y_total.iloc[split:].values)))

    # Refit on the full history before saving — the model ships to
    # production, so we want it trained on every observed game, not just
    # 80% of them. The held-out MAE above is the honest quality estimate.
    spread_model.fit(X_features, y_spread)
    total_model.fit(X_features, y_total)

    return spread_model, total_model, {
        "spread_mae": spread_mae,
        "total_mae": total_mae,
        "n_train": int(n),
    }


def save_regressors(
    spread_model: HistGradientBoostingRegressor,
    total_model: HistGradientBoostingRegressor,
    feature_cols: list[str],
) -> None:
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(spread_model, SPREAD_PATH)
    joblib.dump(total_model, TOTAL_PATH)
    joblib.dump(list(feature_cols), REG_COLS_PATH)


def load_regressors() -> tuple | None:
    """Return (spread_model, total_model, feature_cols) or None if not trained."""
    if not (SPREAD_PATH.exists() and TOTAL_PATH.exists() and REG_COLS_PATH.exists()):
        return None
    try:
        spread_model = joblib.load(SPREAD_PATH)
        total_model = joblib.load(TOTAL_PATH)
        feature_cols = joblib.load(REG_COLS_PATH)
        return spread_model, total_model, feature_cols
    except Exception:
        return None


def predict_spread_total(
    feat_row: pd.DataFrame,
    regressors: tuple | None = None,
) -> tuple[float, float]:
    """Return (predicted_spread, predicted_total) for a single-row feature frame.

    Falls back to empirical NBA priors (0-point spread, 225 total) if
    the regressors aren't trained yet. `feat_row` must have all columns
    the regressor expects — caller is responsible for alignment and NaN
    imputation (same rules as the classifier path).
    """
    if regressors is None:
        regressors = load_regressors()
    if regressors is None:
        return _DEFAULT_SPREAD, _DEFAULT_TOTAL

    spread_model, total_model, feature_cols = regressors

    # Align columns; missing cols become 0 (prediction-time NaN impute
    # should have happened upstream).
    aligned = feat_row.copy()
    for c in feature_cols:
        if c not in aligned.columns:
            aligned[c] = 0.0
    aligned = aligned[feature_cols]

    spread_pred = float(spread_model.predict(aligned)[0])
    total_pred = float(total_model.predict(aligned)[0])
    return spread_pred, total_pred


def generate_spread_total_picks(
    predicted_spread: float,
    predicted_total: float,
    market_spread: float | None,
    market_total: float | None,
    home_team: str,
    away_team: str,
) -> dict:
    """Compare model to market and emit spread/total picks.

    Returns a dict with keys `spread_pick`, `spread_edge`, `total_pick`,
    `total_edge`. `*_pick` is one of:
      - spread: "HOME_COVER" | "AWAY_COVER" | "NO BET"
      - total: "OVER" | "UNDER" | "NO BET"

    Edge values are signed point differences (model - market) so the UI
    can say "model sees 3.5 fewer points than the book".
    """
    out = {
        "spread_pick": "NO BET",
        "spread_edge": 0.0,
        "total_pick": "NO BET",
        "total_edge": 0.0,
    }

    # Market spread convention in this codebase (ESPN): negative =
    # home favored (home_spread). If market spread is -5.5, the home
    # team is "laying 5.5". Our predicted_spread = home_score -
    # away_score, so matching conventions means we compare
    # predicted_spread to -market_spread. We normalize both sides to
    # "expected home margin" for the comparison.
    if market_spread is not None:
        market_home_margin = -float(market_spread)
        gap = predicted_spread - market_home_margin
        out["spread_edge"] = round(gap, 2)
        if gap >= SPREAD_EDGE_PTS:
            # Model thinks home wins by more than the book — take home.
            out["spread_pick"] = f"{home_team} {market_spread:+.1f}"
            out["spread_side"] = "HOME_COVER"
        elif gap <= -SPREAD_EDGE_PTS:
            out["spread_pick"] = f"{away_team} {-market_spread:+.1f}"
            out["spread_side"] = "AWAY_COVER"

    if market_total is not None:
        gap = predicted_total - float(market_total)
        out["total_edge"] = round(gap, 2)
        if gap >= TOTAL_EDGE_PTS:
            out["total_pick"] = f"OVER {market_total:.1f}"
            out["total_side"] = "OVER"
        elif gap <= -TOTAL_EDGE_PTS:
            out["total_pick"] = f"UNDER {market_total:.1f}"
            out["total_side"] = "UNDER"

    return out
