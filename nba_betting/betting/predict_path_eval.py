"""Predict-path backtest harness.

`walk_forward_validate` scores the *training* feature path (`build_feature_matrix`,
which uses a correct per-game `shift(1)`). It does NOT exercise the *prediction*
path (`build_prediction_features`, which reads each team's latest stored row).
Those differ — most importantly, the live predict path's rolling features lag by
one completed game (it reads the team's last game's `shift(1)` window, which
excludes that game). This harness replays historical games through the REAL
`build_prediction_features` to measure what the live model actually does, and to
A/B two variants of the rolling lookup:

- **lagged**  — what live `predict` does today: use the team's row for its game
  immediately BEFORE the target (its rolling window excludes that game).
- **correct** — include the team's most recent completed game: use the target
  game's own row (its `shift(1)` window includes everything up to the prior
  game). This is what a "fixed" predict path would feed.

Both are already present in `rolling_df`: `correct` = `rolling_df[(team, G)]`,
`lagged` = `rolling_df[(team, prev_game_before_G)]`. Elo features are held
identical between the two arms (real pre-game Elos), so the only difference is
the rolling lag — isolating its effect.

Verdict (2025-26 holdout, ~1.3k games): the lag is **within noise and
slightly favorable** (correct − lagged: acc −0.7pp, Brier +0.001). The most
recent single game is noisy and excluding it de-noises the estimate, so the
lag is NOT a bug to fix. Re-run this before changing any predict-path feature
logic; it's the only thing that scores the predict path end to end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _build_rolling_like_engine() -> pd.DataFrame:
    """Reproduce the rolling frame the PredictionEngine builds (incl. the
    vectorized Four-Factors reroll)."""
    from nba_betting.features.rolling import compute_rolling_features
    from nba_betting.features.four_factors import (
        add_four_factors, add_opponent_rebound_data, add_rolling_four_factors,
    )
    from nba_betting.features.rest_days import add_rest_features

    rdf = compute_rolling_features()
    if rdf.empty:
        return rdf
    rdf = add_four_factors(rdf)
    rdf = add_opponent_rebound_data(rdf)
    rdf = add_rest_features(rdf)
    return add_rolling_four_factors(rdf)


def evaluate_predict_path(split: str = "2025-07-01") -> dict:
    """A/B the live (lagged) vs include-latest (correct) rolling lookup on a
    walk-forward holdout, through the real `build_prediction_features`.

    Returns ``{"n", "correct", "lagged", "delta", "mean_abs_prob_change"}``
    where each metric block is ``{"accuracy", "brier", "log_loss"}``.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
    from sqlalchemy import select

    from nba_betting.features.builder import build_feature_matrix, build_prediction_features
    from nba_betting.models.xgboost_model import DEFAULT_PARAMS, _get_feature_cols
    from nba_betting.models.calibration import calibrate_model
    from nba_betting.db.session import get_session
    from nba_betting.db.models import EloRating

    rdf = _build_rolling_like_engine()
    if rdf.empty:
        return {"n": 0}
    team_frames = {
        tid: g.sort_values(["date", "game_id"]).reset_index(drop=True)
        for tid, g in rdf.groupby("team_id")
    }
    team_pos = {
        tid: {gid: i for i, gid in enumerate(tf["game_id"])}
        for tid, tf in team_frames.items()
    }

    session = get_session()
    try:
        elo_map = {
            (r.game_id, r.team_id): (r.elo_before, r.elo_off_before, r.elo_def_before)
            for r in session.execute(select(EloRating)).scalars().all()
        }
    finally:
        session.close()

    X, y = build_feature_matrix(recompute_elos=False)
    if X.empty:
        return {"n": 0}
    fc = _get_feature_cols(X)
    fm = X.attrs.get("feature_means", {})
    X = X.copy()
    X["_d"] = pd.to_datetime(X["_date"])
    split_ts = pd.Timestamp(split)

    tr = X.index[X["_d"] < split_ts]
    if len(tr) < 200:
        return {"n": 0}
    cut = int(len(tr) * 0.8)
    base = HistGradientBoostingClassifier(**{**DEFAULT_PARAMS, "early_stopping": False})
    base.fit(X.loc[tr[:cut], fc], y.loc[tr[:cut]])
    cal = calibrate_model(base, X.loc[tr[cut:], fc], y.loc[tr[cut:]].values, method="isotonic")

    def _predict(hid, aid, gid, latest):
        he = elo_map.get((gid, hid), (1500.0, 1500.0, 1500.0))
        ae = elo_map.get((gid, aid), (1500.0, 1500.0, 1500.0))
        fr = build_prediction_features(
            hid, aid, rdf, he[0], ae[0], feature_means=fm,
            home_elo_off=he[1], home_elo_def=he[2],
            away_elo_off=ae[1], away_elo_def=ae[2],
            latest_stats_by_team=latest, injury_impacts={},
        )
        if fr is None:
            return None
        for c in fc:
            if c not in fr.columns:
                fr[c] = fm.get(c, 0)
        return float(cal.predict_proba(fr[fc])[0, 1])

    cor, lag, ys = [], [], []
    for idx, gr in X[X["_d"] >= split_ts].iterrows():
        gid = gr["_game_id"]
        hid, aid = int(gr["_home_team_id"]), int(gr["_away_team_id"])
        if hid not in team_pos or aid not in team_pos:
            continue
        if gid not in team_pos[hid] or gid not in team_pos[aid]:
            continue
        ph, pa = team_pos[hid][gid], team_pos[aid][gid]
        if ph < 1 or pa < 1:
            continue  # need a prior game for the lagged arm
        cor_latest = {hid: team_frames[hid].iloc[ph].to_dict(), aid: team_frames[aid].iloc[pa].to_dict()}
        lag_latest = {hid: team_frames[hid].iloc[ph - 1].to_dict(), aid: team_frames[aid].iloc[pa - 1].to_dict()}
        pc, pl = _predict(hid, aid, gid, cor_latest), _predict(hid, aid, gid, lag_latest)
        if pc is None or pl is None:
            continue
        cor.append(pc)
        lag.append(pl)
        ys.append(float(y.loc[idx]))

    if not ys:
        return {"n": 0}
    cor = np.clip(np.array(cor), 1e-6, 1 - 1e-6)
    lag = np.clip(np.array(lag), 1e-6, 1 - 1e-6)
    ys = np.array(ys)

    def _m(p):
        return {
            "accuracy": round(float(accuracy_score(ys, (p >= 0.5).astype(int))), 4),
            "brier": round(float(brier_score_loss(ys, p)), 4),
            "log_loss": round(float(log_loss(ys, p)), 4),
        }

    cm, lm = _m(cor), _m(lag)
    return {
        "n": int(len(ys)),
        "correct": cm,
        "lagged": lm,
        "delta": {k: round(cm[k] - lm[k], 4) for k in cm},
        "mean_abs_prob_change": round(float(np.abs(cor - lag).mean()), 4),
    }
