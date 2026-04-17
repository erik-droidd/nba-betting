"""API routes for the NBA betting system."""
from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/predictions/today")
def get_predictions(bankroll: float = Query(1000.0)):
    """Get today's game predictions with model probabilities and market odds."""
    from nba_betting.data.nba_stats import fetch_todays_games, fetch_upcoming_games
    from nba_betting.data.polymarket import get_nba_odds
    from nba_betting.models.elo import (
        get_current_elos,
        get_current_off_def_elos,
        predict_home_win_prob,
    )
    from nba_betting.models.xgboost_model import load_model
    from nba_betting.models.calibration import load_calibrated_model
    from nba_betting.models.ensemble import ensemble_predict
    from nba_betting.betting.recommendations import generate_recommendations
    from nba_betting.config import INITIAL_ELO

    games = fetch_todays_games()
    showing_date = None
    if not games:
        games = fetch_upcoming_games(days_ahead=7)
        if games:
            showing_date = (games[0].get("game_time_utc") or "")[:10] or None
        else:
            return {"games": [], "model": "none", "message": "No games scheduled in the next 7 days"}

    elos = get_current_elos()
    if not elos:
        return {"error": "No Elo ratings. Run sync first."}
    # Tier 1.3 — pull split off/def Elo alongside the aggregate. Falls
    # back silently (to INITIAL_ELO) if the migration hasn't run yet.
    try:
        off_def_elos = get_current_off_def_elos()
    except Exception:
        off_def_elos = {}

    # Try to load XGBoost (or its calibrated wrapper) for the ensemble.
    # IMPORTANT: load_model() deserializes joblib from disk, so we cache
    # it in a single variable instead of calling it multiple times.
    model_name = "elo"
    predict_fn = None
    rolling_df = None

    calibrated = load_calibrated_model()
    xgb_result = load_model()  # (estimator, feature_cols) or None — call ONCE

    if calibrated or xgb_result:
        model_name = "ensemble"
        from nba_betting.features.rolling import compute_rolling_features
        from nba_betting.features.four_factors import add_four_factors, add_opponent_rebound_data
        from nba_betting.features.rest_days import add_rest_features
        from nba_betting.features.builder import build_prediction_features
        from nba_betting.models.xgboost_model import load_feature_means

        rolling_df = compute_rolling_features()
        if not rolling_df.empty:
            rolling_df = add_four_factors(rolling_df)
            rolling_df = add_opponent_rebound_data(rolling_df)
            rolling_df = add_rest_features(rolling_df)

            rolling_df = rolling_df.sort_values(["team_id", "date", "game_id"])
            four_factor_cols = ["efg_pct", "tov_pct", "orb_pct", "ft_rate"]
            for team_id, team_df in rolling_df.groupby("team_id"):
                idx = team_df.index
                for col in four_factor_cols:
                    for w in (5, 10, 20):
                        roll_col = f"{col}_roll_{w}"
                        rolling_df.loc[idx, roll_col] = (
                            team_df[col].shift(1)
                            .rolling(window=w, min_periods=max(1, w // 2))
                            .mean().values
                        )

        # When the calibrated wrapper is present we still need feature_cols
        # from the underlying base estimator (the joblib payload of load_model).
        actual_model = calibrated if calibrated else xgb_result[0]
        feature_cols = xgb_result[1] if xgb_result else []
        feat_means = load_feature_means()
        # Prefer the base (uncalibrated) GBM for feature attribution: isotonic
        # calibration distorts LOO-delta magnitudes even though it preserves
        # ranking. See recommendations.generate_recommendations docstring.
        driver_model = xgb_result[0] if xgb_result else actual_model

        driver_contexts: dict = {}
        spread_total_predictions: dict = {}
        from nba_betting.models.spreads_totals import (
            load_regressors as _load_regs,
            predict_spread_total as _predict_st,
        )
        _regressors = _load_regs()

        def _predict(home_elo, away_elo, home_id=None, away_id=None):
            if rolling_df is None or rolling_df.empty or not feature_cols:
                return predict_home_win_prob(home_elo, away_elo)

            # Inject live line-movement features at prediction time.
            extra = {}
            if home_id and away_id:
                _game = next(
                    (g for g in games
                     if g["home_team_id"] == home_id and g["away_team_id"] == away_id),
                    None,
                )
                if _game:
                    lm = line_movements.get(
                        (_game["home_team_abbr"], _game["away_team_abbr"]), {},
                    )
                    extra["spread_movement"] = lm.get("spread_movement", 0.0)
                    extra["prob_movement"] = lm.get("prob_movement", 0.0)
                    extra["odds_disagreement"] = lm.get("odds_disagreement", 0.0)

            # Tier 1.3 — inject split off/def Elo when available.
            h_off_def = off_def_elos.get(home_id) if home_id else None
            a_off_def = off_def_elos.get(away_id) if away_id else None
            feat_row = build_prediction_features(
                home_id, away_id, rolling_df, home_elo, away_elo,
                feature_means=feat_means,
                extra_features=extra or None,
                home_elo_off=h_off_def[0] if h_off_def else None,
                home_elo_def=h_off_def[1] if h_off_def else None,
                away_elo_off=a_off_def[0] if a_off_def else None,
                away_elo_def=a_off_def[1] if a_off_def else None,
            )
            if feat_row is None:
                return predict_home_win_prob(home_elo, away_elo)
            for c in feature_cols:
                if c not in feat_row.columns:
                    feat_row[c] = feat_means.get(c, 0) if feat_means else 0
            feat_row = feat_row[feature_cols]
            xgb_prob = actual_model.predict_proba(feat_row)[0, 1]
            # Stash the aligned row for lazy driver attribution in
            # generate_recommendations (computed only for bets we'd actually
            # take).
            if home_id is not None and away_id is not None:
                driver_contexts[(home_id, away_id)] = feat_row
            if _regressors is not None and home_id is not None and away_id is not None:
                try:
                    spread_total_predictions[(home_id, away_id)] = _predict_st(
                        feat_row, _regressors,
                    )
                except Exception:
                    pass
            elo_prob = predict_home_win_prob(home_elo, away_elo)
            return ensemble_predict(elo_prob, xgb_prob)

        predict_fn = _predict

    try:
        market_odds = get_nba_odds()
    except Exception:
        market_odds = []

    # ESPN odds as fallback
    try:
        from nba_betting.data.espn_odds import get_espn_odds
        espn_odds = get_espn_odds()
    except Exception:
        espn_odds = []

    # Injuries
    from nba_betting.data.injuries import load_injuries
    injuries = load_injuries()

    # Line movement data
    line_movements = {}
    try:
        from nba_betting.data.odds_tracker import get_line_movement
        from nba_betting.db.models import Team
        from nba_betting.db.session import get_session
        from sqlalchemy import select as sa_select
        from datetime import date as date_type
        _s = get_session()
        _team_lkp = {t.abbreviation: t.id for t in _s.execute(sa_select(Team)).scalars().all()}
        _s.close()
        for g in games:
            h_id = _team_lkp.get(g["home_team_abbr"])
            a_id = _team_lkp.get(g["away_team_abbr"])
            if h_id and a_id:
                lm = get_line_movement(date_type.today(), h_id, a_id)
                if lm.get("n_snapshots", 0) > 0:
                    line_movements[(g["home_team_abbr"], g["away_team_abbr"])] = lm
    except Exception:
        pass

    # Rolling context for explanations. rolling_df is initialized to None
    # at the top of the function, so this check is safe whether or not the
    # ensemble branch ran.
    rolling_context = {}
    if rolling_df is not None and not rolling_df.empty:
        for tid, tdf in rolling_df.groupby("team_id"):
            if not tdf.empty:
                rolling_context[tid] = tdf.sort_values("date").iloc[-1].to_dict()

    recommendations = generate_recommendations(
        games, elos, market_odds, bankroll, predict_fn,
        injuries=injuries,
        rolling_context=rolling_context,
        line_movements=line_movements,
        espn_odds=espn_odds,
        spread_total_predictions=locals().get("spread_total_predictions"),
        driver_contexts=locals().get("driver_contexts"),
        driver_model=locals().get("driver_model"),
        driver_feature_means=locals().get("feat_means"),
    )

    # Serialize
    results = []
    for rec in recommendations:
        results.append({
            "home_team": rec.home_team,
            "away_team": rec.away_team,
            "model_home_prob": round(rec.model_home_prob, 4),
            "market_home_prob": round(rec.market_home_prob, 4),
            "bet_side": rec.bet_side,
            "edge": round(rec.edge, 4),
            "ev_per_dollar": round(rec.ev_per_dollar, 4),
            "kelly_pct": round(rec.kelly_pct, 4),
            "bet_size": round(rec.bet_size, 2),
            "badge": rec.badge,
            "explanation": rec.explanation,
            "spread": rec.spread,
            "over_under": rec.over_under,
            "home_injury_adj": round(rec.home_injury_adj, 4),
            "away_injury_adj": round(rec.away_injury_adj, 4),
            "shrunken_home_prob": (
                round(rec.shrunken_home_prob, 4)
                if rec.shrunken_home_prob is not None else None
            ),
            "drivers": [
                {"feature": d[0], "delta": round(d[1], 4), "value": round(d[2], 4)}
                for d in (rec.drivers or [])[:3]
            ] if rec.drivers else None,
            "predicted_spread": rec.predicted_spread,
            "predicted_total": rec.predicted_total,
            "spread_pick": rec.spread_pick,
            "spread_edge": rec.spread_edge,
            "total_pick": rec.total_pick,
            "total_edge": rec.total_edge,
        })

    return {
        "games": results,
        "model": model_name,
        "market_count": len(market_odds),
        "showing_date": showing_date,
    }


@router.get("/elo")
def get_elo_ratings():
    """Get current Elo ratings for all teams."""
    from nba_betting.models.elo import get_current_elos
    from nba_betting.db.models import Team
    from nba_betting.db.session import get_session
    from sqlalchemy import select

    elos = get_current_elos()
    session = get_session()
    try:
        teams = session.execute(select(Team)).scalars().all()
        ratings = [
            {"team": t.abbreviation, "name": t.name, "elo": round(t.current_elo or 1500, 1)}
            for t in teams
        ]
    finally:
        session.close()

    ratings.sort(key=lambda x: x["elo"], reverse=True)
    return {"ratings": ratings}


@router.get("/performance")
def get_performance():
    """Get historical prediction performance metrics."""
    from nba_betting.betting.tracker import compute_performance, update_results

    update_results()
    return compute_performance()


@router.get("/injuries")
def get_injuries():
    """Get current injury list."""
    from nba_betting.data.injuries import load_injuries
    injuries = load_injuries()
    return {"injuries": [
        {
            "player": i.player_name,
            "team": i.team_abbr,
            "status": i.status,
            "reason": i.reason,
            "impact": i.impact_rating,
        }
        for i in injuries
    ]}
