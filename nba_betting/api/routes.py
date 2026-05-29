"""API routes for the NBA betting system."""
from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/predictions/today")
def get_predictions(bankroll: float = Query(1000.0)):
    """Get today's game predictions with model probabilities and market odds."""
    from nba_betting.data.nba_stats import fetch_todays_games, fetch_upcoming_games
    from nba_betting.data.polymarket import get_nba_odds
    from nba_betting.models.elo import get_current_elos, get_current_off_def_elos
    from nba_betting.betting.recommendations import generate_recommendations

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

    # Build the shared prediction engine — the same module cli.predict uses,
    # so the two paths can't diverge (they did once: the off/def-Elo bug #29).
    from nba_betting.prediction_service import PredictionEngine
    engine = PredictionEngine(games, off_def_elos, blend=True)
    model_name = engine.model_name
    predict_fn = engine.predict if engine.available else None

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

    # Line movement data. Mirrors the cli.predict path: query snapshots
    # using each game's UTC date prefix (matching how snapshot_current_odds
    # files them), with today_et() as the fallback when the timestamp is
    # missing or unparseable.
    line_movements = {}
    try:
        from nba_betting.data.odds_tracker import get_line_movement
        from nba_betting.data.nba_stats import today_et
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
            if not (h_id and a_id):
                continue
            gtu = (g.get("game_time_utc") or "")[:10]
            game_date = today_et()
            if len(gtu) == 10 and gtu[4] == "-" and gtu[7] == "-":
                try:
                    game_date = date_type.fromisoformat(gtu)
                except ValueError:
                    pass
            lm = get_line_movement(game_date, h_id, a_id)
            if lm.get("n_snapshots", 0) > 0:
                line_movements[(g["home_team_abbr"], g["away_team_abbr"])] = lm
    except Exception:
        pass

    # Hand the engine the live context it predicts against, then recommend.
    engine.injuries = injuries
    engine.line_movements = line_movements
    rolling_context = engine.rolling_context()

    recommendations = generate_recommendations(
        games, elos, market_odds, bankroll, predict_fn,
        injuries=injuries,
        rolling_context=rolling_context,
        line_movements=line_movements,
        espn_odds=espn_odds,
        spread_total_predictions=engine.spread_total_predictions,
        driver_contexts=engine.driver_contexts,
        driver_model=engine.driver_model,
        driver_feature_means=engine.feat_means,
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
