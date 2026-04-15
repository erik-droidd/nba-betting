"""Historical backtesting: simulate the full betting pipeline on past games."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from nba_betting.betting.edge import compute_edge, confidence_badge
from nba_betting.betting.kelly import kelly_fraction
from nba_betting.betting.shrinkage import shrink_to_market
from nba_betting.config import (
    DEFAULT_BANKROLL,
    MIN_EDGE_THRESHOLD,
    MAX_BET_PCT,
    MARKET_SHRINKAGE_LAMBDA,
    MIN_BET_SIDE_PROB,
    ELO_HOME_ADVANTAGE,
)
from nba_betting.models.elo import expected_score
from nba_betting.models.ensemble import ensemble_predict


def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    bankroll: float = DEFAULT_BANKROLL,
    n_splits: int = 3,
    use_ensemble: bool = True,
    apply_live_strategy: bool | None = None,
    use_real_odds: bool = False,
) -> dict:
    """Run a historical backtest of the betting strategy.

    Trains models using walk-forward splits, then simulates betting on
    each test-fold game using Kelly sizing and edge thresholds.

    Args:
        X: Feature matrix from build_feature_matrix() (with _date, _game_id metadata).
        y: Target series (1 = home win).
        bankroll: Starting bankroll.
        n_splits: Number of walk-forward splits.
        use_ensemble: If True, blend Elo + GBM. Otherwise GBM-only.
        apply_live_strategy: If True, apply the same Bayesian shrinkage
            (`shrink_to_market`) and asymmetric bet-side floor
            (`MIN_BET_SIDE_PROB`) that the live `predict` pipeline uses,
            so backtest ROI is a closer proxy for what the live strategy
            would actually have booked. If False, use the raw model vs
            Elo-proxy edge (legacy behavior — useful for pure model eval).
            If None (default), follows `use_real_odds`: on when real odds
            are available (live-equivalent simulation), off otherwise
            (pure model-quality benchmark against the Elo proxy, since
            shrinking the model toward Elo is a degenerate null-op).

            IMPORTANT CAVEAT: if `use_real_odds=False`, the "market" is
            an Elo proxy — we do not have historical Polymarket/ESPN
            closing lines for the full span of the training data. Real
            efficient-market prices would be *harder* than Elo, so live
            ROI will almost certainly come in lower than Elo-proxy
            backtest ROI. Treat the number as an upper bound on live
            performance, not a forecast.
        use_real_odds: If True, look up a real closing-line snapshot from
            the `odds_snapshots` table for each test game (via
            `get_closing_line`) and use that as the market price instead
            of the Elo proxy. Falls back to Elo when no snapshot exists
            (early dates, games we didn't cover). Requires that the
            snapshot collector cron has been running long enough for the
            evaluation window to have coverage.

    Returns:
        Dict with per-bet results, aggregate metrics, and bankroll curve.
    """
    # Resolve the live-strategy default against use_real_odds. Shrinking
    # model -> Elo-proxy is a no-op in log-odds space when they agree and
    # just dampens model conviction when they diverge, so the Elo-proxy
    # backtest is informative only when the raw model is evaluated
    # directly. With real closing lines the strategy matches live.
    if apply_live_strategy is None:
        apply_live_strategy = use_real_odds
    from nba_betting.models.xgboost_model import DEFAULT_PARAMS

    if "_date" not in X.columns:
        raise ValueError("X must contain _date column for temporal splitting")

    feature_cols = [c for c in X.columns if not c.startswith("_")]
    dates = pd.to_datetime(X["_date"])

    X_sorted = X.copy()
    X_sorted["_date_parsed"] = dates
    X_sorted = X_sorted.sort_values("_date_parsed")
    y_sorted = y.loc[X_sorted.index]

    # Split at July 1 boundaries
    years = X_sorted["_date_parsed"].dt.year.unique()
    split_dates = []
    for yr in sorted(years):
        july = pd.Timestamp(f"{yr}-07-01")
        if X_sorted["_date_parsed"].min() < july < X_sorted["_date_parsed"].max():
            split_dates.append(july)

    if len(split_dates) < 1:
        return {"bets": [], "summary": {}, "bankroll_curve": [bankroll]}

    split_dates = split_dates[-n_splits:]

    all_bets = []
    current_bankroll = bankroll
    bankroll_curve = [current_bankroll]
    wf_params = {**DEFAULT_PARAMS, "early_stopping": False}

    # Lazy import only if real-odds mode is requested; keeps the
    # Elo-proxy path free of DB work.
    _get_closing_line = None
    _real_odds_hits = 0
    _real_odds_misses = 0
    if use_real_odds:
        from nba_betting.data.odds_tracker import get_closing_line
        _get_closing_line = get_closing_line
        if "_home_team_id" not in X_sorted.columns or "_away_team_id" not in X_sorted.columns:
            raise ValueError(
                "use_real_odds=True requires _home_team_id / _away_team_id metadata "
                "in the feature matrix. Rebuild with build_feature_matrix()."
            )

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

        X_tr = X_sorted.loc[train_mask, feature_cols].values
        y_tr = y_sorted.loc[train_mask].values

        model = HistGradientBoostingClassifier(**wf_params)
        model.fit(X_tr, y_tr)

        # Simulate betting on each test game
        test_indices = X_sorted.index[test_mask]
        for idx in test_indices:
            row = X_sorted.loc[idx]
            actual = y_sorted.loc[idx]

            feat_vals = row[feature_cols].values.reshape(1, -1)
            gbm_prob = model.predict_proba(feat_vals)[0, 1]

            # Get Elo probability from features
            elo_prob = row.get("elo_home_prob", 0.5)

            if use_ensemble:
                raw_model_home = ensemble_predict(elo_prob, gbm_prob)
            else:
                raw_model_home = gbm_prob

            # Use Elo as "market" proxy (since we don't generally have
            # historical Polymarket / ESPN closing lines for older games).
            market_home_prob = elo_prob

            if use_real_odds and _get_closing_line is not None:
                row_date = row["_date_parsed"]
                game_day = row_date.date() if hasattr(row_date, "date") else row_date
                snap = _get_closing_line(
                    game_day,
                    int(row["_home_team_id"]),
                    int(row["_away_team_id"]),
                )
                if snap and snap.get("home_prob") and 0 < snap["home_prob"] < 1:
                    market_home_prob = float(snap["home_prob"])
                    _real_odds_hits += 1
                else:
                    _real_odds_misses += 1

            market_away_prob = 1.0 - market_home_prob

            # Mirror the live `predict` pipeline: shrink the model prob
            # toward the market log-odds, then enforce the asymmetric
            # floor on the side we're about to bet. The caller can opt out
            # (`apply_live_strategy=False`) to measure the raw model.
            if apply_live_strategy and 0 < market_home_prob < 1:
                model_home_prob = shrink_to_market(
                    raw_model_home, market_home_prob, MARKET_SHRINKAGE_LAMBDA,
                )
            else:
                model_home_prob = raw_model_home

            model_away_prob = 1.0 - model_home_prob

            # Compute edge for both sides
            home_edge = compute_edge(model_home_prob, market_home_prob) if market_home_prob > 0 else 0.0
            away_edge = compute_edge(model_away_prob, market_away_prob) if market_away_prob > 0 else 0.0

            # Asymmetric bet-side floor: don't bet a team the (shrunken)
            # model itself only gives MIN_BET_SIDE_PROB or less of winning.
            if apply_live_strategy:
                home_passes_floor = model_home_prob >= MIN_BET_SIDE_PROB
                away_passes_floor = model_away_prob >= MIN_BET_SIDE_PROB
            else:
                home_passes_floor = True
                away_passes_floor = True

            # Pick best side
            if (
                home_edge > away_edge
                and home_edge >= MIN_EDGE_THRESHOLD
                and home_passes_floor
            ):
                bet_side = "HOME"
                edge = home_edge
                bet_prob = model_home_prob
                market_prob = market_home_prob
                won = bool(actual == 1)
            elif away_edge >= MIN_EDGE_THRESHOLD and away_passes_floor:
                bet_side = "AWAY"
                edge = away_edge
                bet_prob = model_away_prob
                market_prob = market_away_prob
                won = bool(actual == 0)
            else:
                # No bet
                continue

            kelly_pct = kelly_fraction(bet_prob, market_prob)
            bet_size = min(current_bankroll * kelly_pct, current_bankroll * MAX_BET_PCT)
            bet_size = max(0, bet_size)

            if bet_size <= 0:
                continue

            # Calculate profit
            decimal_odds = 1.0 / market_prob if market_prob > 0 else 0
            if won:
                profit = bet_size * (decimal_odds - 1)
            else:
                profit = -bet_size

            current_bankroll += profit
            bankroll_curve.append(current_bankroll)

            all_bets.append({
                "date": str(row["_date_parsed"].date()) if hasattr(row["_date_parsed"], "date") else str(row["_date_parsed"]),
                "bet_side": bet_side,
                "model_prob": round(bet_prob, 4),
                "market_prob": round(market_prob, 4),
                "edge": round(edge, 4),
                "kelly_pct": round(kelly_pct, 4),
                "bet_size": round(bet_size, 2),
                "won": won,
                "profit": round(profit, 2),
                "bankroll": round(current_bankroll, 2),
                "badge": confidence_badge(edge),
            })

            # Stop if bankrupt
            if current_bankroll <= 0:
                break

        if current_bankroll <= 0:
            break

    # Compute summary
    summary = _compute_summary(all_bets, bankroll, bankroll_curve)
    if use_real_odds:
        summary["real_odds_hits"] = _real_odds_hits
        summary["real_odds_misses"] = _real_odds_misses
        total = _real_odds_hits + _real_odds_misses
        summary["real_odds_coverage"] = (
            round(_real_odds_hits / total, 4) if total else 0.0
        )

    return {
        "bets": all_bets,
        "summary": summary,
        "bankroll_curve": bankroll_curve,
    }


def _compute_summary(bets: list[dict], initial_bankroll: float, bankroll_curve: list[float]) -> dict:
    """Compute aggregate backtest metrics."""
    if not bets:
        return {"total_bets": 0}

    wins = sum(1 for b in bets if b["won"])
    losses = len(bets) - wins
    total_wagered = sum(b["bet_size"] for b in bets)
    total_profit = sum(b["profit"] for b in bets)
    roi = total_profit / total_wagered if total_wagered > 0 else 0

    # Sharpe ratio (annualized, assuming ~5 bets/day over ~200 days)
    profits = [b["profit"] for b in bets]
    if len(profits) > 1:
        mean_profit = np.mean(profits)
        std_profit = np.std(profits, ddof=1)
        sharpe = (mean_profit / std_profit) * np.sqrt(len(profits)) if std_profit > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    peak = bankroll_curve[0]
    max_drawdown = 0.0
    for val in bankroll_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    # Edge distribution
    edges = [b["edge"] for b in bets]

    return {
        "total_bets": len(bets),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(bets), 4),
        "total_wagered": round(total_wagered, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 4),
        "sharpe_ratio": round(sharpe, 4),
        "final_bankroll": round(bankroll_curve[-1], 2),
        "max_drawdown": round(max_drawdown, 4),
        "avg_edge": round(np.mean(edges), 4),
        "avg_bet_size": round(total_wagered / len(bets), 2),
    }
