"""Score walk-forward predictions against REAL closing lines.

`walk_forward_validate` scores the model against outcomes; `run_backtest`
against an Elo proxy (or real odds where they exist, but only on the
games it bets). Neither answers the questions that actually set the
betting knobs:

* **Moneyline:** how much better (or worse) than the closing line is the
  model on the SAME games, and which market-shrinkage weight λ
  (``MARKET_SHRINKAGE_LAMBDA``) minimises log-loss? λ = 0.6 is a
  judgment call; this is the harness that can replace it with a fitted
  value once the snapshot cron has a season of lines.
* **Spread / total:** are the regression heads' picks (model vs market
  gap beyond ``SPREAD_EDGE_PTS`` / ``TOTAL_EDGE_PTS``) covering at better
  than the 52.4% break-even, and is the model's margin/total more
  accurate than the market's?

Everything here is out-of-fold: per July-1 fold the classifier is fit on
the first 80% of the training slice and isotonic-calibrated on the last
20% (exactly the backtest's recipe), the Elo probability is the
feature-matrix column, the blend uses the learned ensemble weight, and
the spread/total regressors use ``spreads_totals.REG_PARAMS``. Closing
lines come from ``batch_closing_lines`` (latest snapshot per source):
Polymarket for the moneyline (ESPN fallback), ESPN for spread/total.

Caveats: Polymarket prices are not de-vigged (only the home price is
stored), which flatters the market slightly on log-loss; and the verdict
is withheld below ``--min-games`` because a few hundred games cannot
separate λ values that differ by 0.1.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nba_betting.utils.math import logit, sigmoid


def paired_t(loss_a, loss_b) -> float:
    """t-stat of mean(loss_a − loss_b); positive → b has lower loss."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    if len(d) < 2:
        return 0.0
    sd = d.std(ddof=1)
    return float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def brier(p, y):
    return (_clip(p) - np.asarray(y, dtype=float)) ** 2


def logloss(p, y):
    p = _clip(p)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def shrink(model_p, market_p, lam: float):
    """Vectorized ``shrink_to_market``: σ((1−λ)·logit(model) + λ·logit(market))."""
    return sigmoid((1.0 - lam) * logit(_clip(model_p)) + lam * logit(_clip(market_p)))


def score_lambda_grid(model_p, market_p, y, grid=None) -> dict[float, float]:
    """{λ: mean log-loss of the shrunk probability} over ``grid``."""
    grid = grid if grid is not None else [round(0.1 * i, 1) for i in range(11)]
    return {float(lam): float(logloss(shrink(model_p, market_p, lam), y).mean()) for lam in grid}


def cover_stats(model_val, market_val, actual, threshold: float) -> dict:
    """Pick/hit accounting for a points market (spread or total).

    A pick is made when |model − market| >= threshold, on the model's
    side; it hits when the actual lands on that side of the market line;
    pushes (actual == line) are excluded from both counts.
    """
    m = np.asarray(model_val, dtype=float)
    k = np.asarray(market_val, dtype=float)
    a = np.asarray(actual, dtype=float)
    gap = m - k
    pick = np.abs(gap) >= threshold
    push = a == k
    scored = pick & ~push
    hit = scored & (np.sign(a - k) == np.sign(gap))
    n_picks = int(scored.sum())
    n_hits = int(hit.sum())
    return {
        "picks": n_picks,
        "hits": n_hits,
        "hit_rate": (n_hits / n_picks) if n_picks else 0.0,
        "pushes": int((pick & push).sum()),
    }


def _points_section(model_val, market_val, actual, threshold) -> dict:
    n = len(actual)
    if n == 0:
        return {"n": 0}
    m = np.asarray(model_val, dtype=float)
    k = np.asarray(market_val, dtype=float)
    a = np.asarray(actual, dtype=float)
    e_model = np.abs(m - a)
    e_market = np.abs(k - a)
    e_avg = np.abs((m + k) / 2 - a)
    return {
        "n": n,
        "model_mae": float(e_model.mean()),
        "market_mae": float(e_market.mean()),
        "avg_mae": float(e_avg.mean()),
        "t_market_vs_model": paired_t(e_model, e_market),
        **cover_stats(m, k, a, threshold),
    }


def _fold_predictions(X: pd.DataFrame, y: pd.Series, n_splits: int) -> pd.DataFrame:
    """Out-of-fold blend probability, predicted margin and total per test
    game, using the backtest's per-fold recipe."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    from nba_betting.models.calibration import calibrate_model
    from nba_betting.models.ensemble import ensemble_predict_batch
    from nba_betting.models.spreads_totals import REG_PARAMS
    from nba_betting.models.xgboost_model import DEFAULT_PARAMS, _get_feature_cols

    fc = _get_feature_cols(X)
    X = X.copy()
    X["_d"] = pd.to_datetime(X["_date"])
    X = X.sort_values("_d")
    y = y.loc[X.index]
    years = sorted(X["_d"].dt.year.unique())
    splits = [pd.Timestamp(f"{yr}-07-01") for yr in years
              if X["_d"].min() < pd.Timestamp(f"{yr}-07-01") < X["_d"].max()][-n_splits:]
    params = {**DEFAULT_PARAMS, "early_stopping": False}
    parts = []
    for i, sp in enumerate(splits):
        tr = X["_d"] < sp
        te = (X["_d"] >= sp) & ((X["_d"] < splits[i + 1]) if i + 1 < len(splits) else True)
        if tr.sum() < 100 or te.sum() < 50:
            continue
        Xtr, ytr, Xte = X[tr], y[tr], X[te]
        cut = int(len(Xtr) * 0.8)
        gbm = HistGradientBoostingClassifier(**params).fit(Xtr.iloc[:cut][fc], ytr.iloc[:cut])
        model = gbm
        if len(Xtr) - cut >= 20:
            try:
                model = calibrate_model(gbm, Xtr.iloc[cut:][fc], ytr.iloc[cut:].values, method="isotonic")
            except Exception:
                model = gbm
        gbm_p = model.predict_proba(Xte[fc])[:, 1]
        elo_p = Xte["elo_home_prob"].astype(float).values
        blend = ensemble_predict_batch(elo_p, gbm_p)
        hs = pd.to_numeric(Xtr["_home_score"]); as_ = pd.to_numeric(Xtr["_away_score"])
        spread_m = HistGradientBoostingRegressor(**REG_PARAMS).fit(Xtr[fc], (hs - as_).values)
        total_m = HistGradientBoostingRegressor(**REG_PARAMS).fit(Xtr[fc], (hs + as_).values)
        parts.append(pd.DataFrame({
            "game_id": Xte["_game_id"].values,
            "date": Xte["_d"].dt.date.values,
            "home_team_id": Xte["_home_team_id"].astype(int).values,
            "away_team_id": Xte["_away_team_id"].astype(int).values,
            "y": y[te].astype(float).values,
            "actual_margin": (pd.to_numeric(Xte["_home_score"]) - pd.to_numeric(Xte["_away_score"])).values,
            "actual_total": (pd.to_numeric(Xte["_home_score"]) + pd.to_numeric(Xte["_away_score"])).values,
            "model_p": blend,
            "pred_margin": spread_m.predict(Xte[fc]),
            "pred_total": total_m.predict(Xte[fc]),
        }))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def evaluate_against_market(n_splits: int = 3) -> dict:
    """Run the harness. Returns ``{"moneyline": {...}, "spread": {...},
    "total": {...}}`` — each with ``n`` (0 when no closing lines join)."""
    from nba_betting.config import MARKET_SHRINKAGE_LAMBDA
    from nba_betting.data.odds_tracker import batch_closing_lines
    from nba_betting.features.builder import build_feature_matrix
    from nba_betting.models.spreads_totals import SPREAD_EDGE_PTS, TOTAL_EDGE_PTS

    empty = {"moneyline": {"n": 0}, "spread": {"n": 0}, "total": {"n": 0}}
    X, y = build_feature_matrix(recompute_elos=False)
    if X.empty:
        return empty
    preds = _fold_predictions(X, y, n_splits)
    if preds.empty:
        return empty

    lines = batch_closing_lines()
    ml_p, ml_y, ml_model = [], [], []
    sp_model, sp_market, sp_actual = [], [], []
    tot_model, tot_market, tot_actual = [], [], []
    for r in preds.itertuples(index=False):
        key = (r.date, r.home_team_id, r.away_team_id)
        poly = lines.get((*key, "polymarket"))
        espn = lines.get((*key, "espn"))
        mkt = None
        for src in (poly, espn):
            if src and src.get("home_prob") is not None and 0 < src["home_prob"] < 1:
                mkt = float(src["home_prob"])
                break
        if mkt is not None:
            ml_p.append(mkt); ml_y.append(r.y); ml_model.append(r.model_p)
        if espn and espn.get("spread") is not None:
            # ESPN spread is the home line (negative = home favoured);
            # market home margin = -spread (spreads_totals convention).
            sp_model.append(r.pred_margin); sp_market.append(-float(espn["spread"])); sp_actual.append(r.actual_margin)
        if espn and espn.get("over_under") is not None:
            tot_model.append(r.pred_total); tot_market.append(float(espn["over_under"])); tot_actual.append(r.actual_total)

    out = dict(empty)
    if ml_y:
        ml_model = np.asarray(ml_model); ml_p = np.asarray(ml_p); ml_y = np.asarray(ml_y)
        table = score_lambda_grid(ml_model, ml_p, ml_y)
        best_lambda = min(table, key=table.get)
        live = shrink(ml_model, ml_p, MARKET_SHRINKAGE_LAMBDA)
        best = shrink(ml_model, ml_p, best_lambda)
        out["moneyline"] = {
            "n": int(len(ml_y)),
            "model_brier": float(brier(ml_model, ml_y).mean()),
            "model_ll": float(logloss(ml_model, ml_y).mean()),
            "market_brier": float(brier(ml_p, ml_y).mean()),
            "market_ll": float(logloss(ml_p, ml_y).mean()),
            "live_brier": float(brier(live, ml_y).mean()),
            "live_ll": float(logloss(live, ml_y).mean()),
            "best_lambda": float(best_lambda),
            "best_brier": float(brier(best, ml_y).mean()),
            "best_ll": float(logloss(best, ml_y).mean()),
            "lambda_table": table,
            "t_market_vs_model": paired_t(brier(ml_model, ml_y), brier(ml_p, ml_y)),
            "t_best_vs_live": paired_t(logloss(live, ml_y), logloss(best, ml_y)),
        }
    out["spread"] = _points_section(sp_model, sp_market, sp_actual, SPREAD_EDGE_PTS)
    out["total"] = _points_section(tot_model, tot_market, tot_actual, TOTAL_EDGE_PTS)
    return out
