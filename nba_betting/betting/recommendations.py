"""Generate betting recommendations by combining model predictions with market odds."""
from __future__ import annotations

from dataclasses import dataclass

from nba_betting.betting.edge import compute_edge, is_positive_ev, confidence_badge
from nba_betting.betting.kelly import (
    kelly_fraction,
    compute_bet_size,
    compute_drawdown_multiplier,
    get_recent_roi,
)
from nba_betting.betting.shrinkage import shrink_to_market
from nba_betting.config import (
    DEFAULT_BANKROLL,
    MIN_EDGE_THRESHOLD,
    MARKET_SHRINKAGE_LAMBDA,
    MIN_BET_SIDE_PROB,
)


@dataclass
class BetRecommendation:
    home_team: str
    away_team: str
    model_home_prob: float          # Post-injury, pre-shrinkage
    market_home_prob: float
    bet_side: str  # "HOME", "AWAY", or "NO BET"
    edge: float
    ev_per_dollar: float
    kelly_pct: float
    bet_size: float
    badge: str
    explanation: str = ""
    spread: float | None = None
    over_under: float | None = None
    home_injury_adj: float = 0.0
    away_injury_adj: float = 0.0
    # Post-shrinkage probability actually used to compute edge. Surfaced
    # so the dashboard / explanation can show that we shrank toward market.
    shrunken_home_prob: float | None = None
    # Top feature contributions from the model's own prediction. List of
    # (feature_name, delta_toward_home, feature_value) tuples from
    # `compute_prediction_drivers`. None if the predict_fn is pure Elo
    # or a driver calculation wasn't possible for this game.
    drivers: list | None = None
    # Spread/total model output — populated when the regressors are
    # trained and a feature row was computed for this game. The "_pick"
    # strings are display-ready ("LAL -3.5", "OVER 228.5", "NO BET").
    predicted_spread: float | None = None       # home_score - away_score
    predicted_total: float | None = None
    spread_pick: str = "NO BET"
    spread_edge: float = 0.0
    total_pick: str = "NO BET"
    total_edge: float = 0.0


def generate_recommendations(
    games: list[dict],
    elos: dict[int, float],
    market_odds: list[dict],
    bankroll: float = DEFAULT_BANKROLL,
    predict_fn=None,
    injuries: list = None,
    rolling_context: dict = None,
    line_movements: dict = None,
    espn_odds: list[dict] = None,
    spread_total_predictions: dict = None,
    driver_contexts: dict = None,
    driver_model=None,
    driver_feature_means: dict = None,
) -> list[BetRecommendation]:
    """Generate betting recommendations for today's games.

    Args:
        games: List of today's games from fetch_todays_games().
        elos: Current Elo ratings {team_id: elo}.
        market_odds: List of Polymarket odds from get_nba_odds().
        bankroll: Current bankroll in dollars.
        predict_fn: Prediction function.
        injuries: Current injury list for explanation generation.
        rolling_context: Dict of {team_id: latest_rolling_stats} for explanations.
        line_movements: Dict of {(home_abbr, away_abbr): movement_data}.
        espn_odds: ESPN odds list (used for spread/OU info and as fallback).
        spread_total_predictions: Dict of {(home_id, away_id): (spread, total)}.
        driver_contexts: Dict of {(home_id, away_id): single-row feature DataFrame}.
            The caller (cli.py / api/routes.py `predict` closure) stashes the
            feature row used for the probability prediction so we can
            lazily compute feature attribution here — only for games where
            `bet_side != "NO BET"`. Skipping NO-BET rows avoids spending
            ~10–30ms per filtered game on drivers that nothing will
            display anyway.
        driver_model: The model to run attribution against. Prefer the
            *uncalibrated* base GBM: isotonic calibration is a monotonic
            post-hoc transform, so it preserves the sign/ranking of
            feature contributions but distorts their magnitudes. Running
            LOO attribution on the base estimator gives a cleaner signal
            about which features the tree ensemble actually split on.
            Falls back to whatever the caller passes.
        driver_feature_means: Training-time means used as the "neutral"
            reference for each feature. See `compute_prediction_drivers`.
    """
    from nba_betting.models.elo import predict_home_win_prob
    from nba_betting.config import INITIAL_ELO
    from nba_betting.betting.explanations import generate_explanation
    from nba_betting.data.injuries import get_team_injury_adjustment
    import inspect

    if predict_fn is None:
        predict_fn = predict_home_win_prob

    # Drawdown-aware Kelly: compute a single multiplier from recent
    # bet performance. If recent ROI is deeply negative, shrink all
    # bet sizes on this slate to reduce variance during cold streaks.
    try:
        _recent_roi, _n_bets = get_recent_roi(lookback=10)
        _dd_mult = compute_drawdown_multiplier(_recent_roi, _n_bets)
    except Exception:
        _dd_mult = 1.0

    # Check if predict_fn accepts team ID kwargs
    _accepts_ids = "home_id" in inspect.signature(predict_fn).parameters

    # Index market odds by frozenset of team abbrs (order-independent matching)
    market_by_teams: dict[frozenset, dict] = {}
    for odds in market_odds:
        teams = odds.get("teams", {})
        abbrs = list(teams.keys())
        if len(abbrs) == 2:
            key = frozenset(abbrs)
            market_by_teams[key] = odds

    # Index ESPN odds as fallback + for spread/OU data
    espn_by_teams: dict[frozenset, dict] = {}
    for odds in (espn_odds or []):
        teams = odds.get("teams", {})
        abbrs = list(teams.keys())
        if len(abbrs) == 2:
            espn_by_teams[frozenset(abbrs)] = odds

    recommendations = []

    for game in games:
        home_abbr = game["home_team_abbr"]
        away_abbr = game["away_team_abbr"]
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        # Get Elo ratings
        home_elo = elos.get(home_id, INITIAL_ELO)
        away_elo = elos.get(away_id, INITIAL_ELO)

        # Model prediction
        if _accepts_ids:
            model_home_prob = predict_fn(home_elo, away_elo, home_id=home_id, away_id=away_id)
        else:
            model_home_prob = predict_fn(home_elo, away_elo)

        # Apply injury adjustments (model has no player data, so adjust post-hoc)
        home_inj_adj = get_team_injury_adjustment(home_abbr)
        away_inj_adj = get_team_injury_adjustment(away_abbr)
        # Net adjustment: home injuries hurt home, away injuries help home
        net_adj = home_inj_adj - away_inj_adj
        model_home_prob = max(0.01, min(0.99, model_home_prob + net_adj))

        # Find matching market odds (order-independent)
        market_match = market_by_teams.get(frozenset([home_abbr, away_abbr]))

        # Get spread/OU from ESPN if available
        espn_match = espn_by_teams.get(frozenset([home_abbr, away_abbr]))
        spread = espn_match.get("spread") if espn_match else None
        over_under = espn_match.get("over_under") if espn_match else None

        if market_match:
            # Look up prices by team abbreviation directly (no home/away assumption)
            teams = market_match.get("teams", {})
            market_home_prob = teams.get(home_abbr, 0.0)
            market_away_prob = teams.get(away_abbr, 0.0)
        elif espn_match:
            # Fallback to ESPN odds
            teams = espn_match.get("teams", {})
            market_home_prob = teams.get(home_abbr, 0.0)
            market_away_prob = teams.get(away_abbr, 0.0)
        else:
            # No market data - show prediction only
            market_home_prob = 0.0
            market_away_prob = 0.0

        # Bayesian shrinkage of model prob toward market log-odds. The
        # market is treated as a strong prior. Edge is computed using the
        # shrunken probability so that small disagreements are dampened
        # and only decisive model conviction survives. If we have no
        # market price for this game, skip shrinkage entirely.
        if market_home_prob > 0 and market_home_prob < 1:
            shrunken_home_prob = shrink_to_market(
                model_home_prob, market_home_prob, MARKET_SHRINKAGE_LAMBDA,
            )
        else:
            shrunken_home_prob = model_home_prob
        shrunken_away_prob = 1.0 - shrunken_home_prob

        # Edges are computed against the SHRUNKEN probability, not the raw model.
        home_edge = compute_edge(shrunken_home_prob, market_home_prob) if market_home_prob > 0 else 0.0
        away_edge = compute_edge(shrunken_away_prob, market_away_prob) if market_away_prob > 0 else 0.0

        # Pick the better side, with the asymmetric floor: don't bet a
        # team the model itself only gives MIN_BET_SIDE_PROB or less of
        # winning. This kills lottery-ticket bets where positive EV
        # depends on a tail price the model isn't really contradicting.
        home_passes_floor = shrunken_home_prob >= MIN_BET_SIDE_PROB
        away_passes_floor = shrunken_away_prob >= MIN_BET_SIDE_PROB

        if home_edge > away_edge and is_positive_ev(home_edge) and home_passes_floor:
            bet_side = "HOME"
            edge = home_edge
            kelly_pct = kelly_fraction(shrunken_home_prob, market_home_prob, drawdown_mult=_dd_mult)
            bet_size = compute_bet_size(bankroll, shrunken_home_prob, market_home_prob, drawdown_mult=_dd_mult)
        elif is_positive_ev(away_edge) and away_passes_floor:
            bet_side = "AWAY"
            edge = away_edge
            kelly_pct = kelly_fraction(shrunken_away_prob, market_away_prob, drawdown_mult=_dd_mult)
            bet_size = compute_bet_size(bankroll, shrunken_away_prob, market_away_prob, drawdown_mult=_dd_mult)
        else:
            bet_side = "NO BET"
            edge = max(home_edge, away_edge)
            kelly_pct = 0.0
            bet_size = 0.0

        # Badge reflects the *actionable* signal. If we already decided
        # NO BET (either insufficient edge OR the asymmetric floor), don't
        # confuse the user with a "SUSPECT" warning on a row we filtered.
        badge = "NO BET" if bet_side == "NO BET" else confidence_badge(edge)

        # Feature-attribution drivers: compute only for actionable bets.
        # Driver attribution runs a batched predict_proba with one row per
        # feature (~N_features × 1 extra forward pass), so skipping NO-BET
        # rows gives us a meaningful savings on nights with 10+ games of
        # which only 2-3 clear the edge threshold.
        drivers = None
        if (
            bet_side != "NO BET"
            and driver_contexts
            and driver_model is not None
        ):
            feat_row = driver_contexts.get((home_id, away_id))
            if feat_row is not None and not feat_row.empty:
                try:
                    from nba_betting.models.drivers import compute_prediction_drivers
                    drivers = compute_prediction_drivers(
                        driver_model, feat_row, driver_feature_means or {}, top_k=5,
                    )
                except Exception:
                    drivers = None  # Non-critical; explanation falls back to heuristic

        rec = BetRecommendation(
            home_team=home_abbr,
            away_team=away_abbr,
            model_home_prob=model_home_prob,
            market_home_prob=market_home_prob,
            bet_side=bet_side,
            edge=edge,
            ev_per_dollar=edge,
            kelly_pct=kelly_pct,
            bet_size=bet_size,
            badge=badge,
            spread=spread,
            over_under=over_under,
            home_injury_adj=home_inj_adj,
            away_injury_adj=away_inj_adj,
            shrunken_home_prob=shrunken_home_prob,
            drivers=drivers,
        )

        # Spread / total picks (if regressors are loaded and produced a
        # prediction for this game).
        st_pred = (spread_total_predictions or {}).get((home_id, away_id))
        if st_pred is not None:
            from nba_betting.models.spreads_totals import generate_spread_total_picks
            pred_spread, pred_total = st_pred
            st_picks = generate_spread_total_picks(
                predicted_spread=pred_spread,
                predicted_total=pred_total,
                market_spread=spread,
                market_total=over_under,
                home_team=home_abbr,
                away_team=away_abbr,
            )
            rec.predicted_spread = round(pred_spread, 2)
            rec.predicted_total = round(pred_total, 2)
            rec.spread_pick = st_picks["spread_pick"]
            rec.spread_edge = st_picks["spread_edge"]
            rec.total_pick = st_picks["total_pick"]
            rec.total_edge = st_picks["total_edge"]

        # Generate explanation
        home_stats = rolling_context.get(home_id) if rolling_context else None
        away_stats = rolling_context.get(away_id) if rolling_context else None
        lm = line_movements.get((home_abbr, away_abbr)) if line_movements else None
        rec.explanation = generate_explanation(rec, home_stats, away_stats, injuries, lm)

        recommendations.append(rec)

    # Sort by edge descending
    recommendations.sort(key=lambda r: r.edge, reverse=True)

    # Same-day correlation adjustment (#7 improvement): Kelly assumes
    # independent bets, but NBA games on the same night share league-wide
    # factors (ref crew assignments, national-TV pace effects, etc.).
    # Empirical correlation from backtest residuals is ~0.15. When
    # placing multiple bets on one slate, we scale each bet down by
    # 1/sqrt(1 + (n-1)*rho) so total portfolio risk doesn't blow past
    # what Kelly intended.
    _SAME_DAY_RHO = 0.15  # estimated pairwise correlation
    actionable = [r for r in recommendations if r.bet_side != "NO BET"]
    n_bets = len(actionable)
    if n_bets > 1:
        import math
        corr_scale = 1.0 / math.sqrt(1.0 + (n_bets - 1) * _SAME_DAY_RHO)
        for r in actionable:
            r.kelly_pct = r.kelly_pct * corr_scale
            r.bet_size = round(r.bet_size * corr_scale, 2)

    return recommendations
