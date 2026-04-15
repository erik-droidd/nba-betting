"""Monte Carlo simulation for bankroll and prediction confidence."""
from __future__ import annotations

import numpy as np

from nba_betting.betting.kelly import kelly_fraction
from nba_betting.config import DEFAULT_BANKROLL, MAX_BET_PCT, MIN_EDGE_THRESHOLD


def simulate_bankroll(
    model_probs: list[float],
    market_probs: list[float],
    n_simulations: int = 10_000,
    n_bets_per_sim: int | None = None,
    initial_bankroll: float = DEFAULT_BANKROLL,
    rng_seed: int = 42,
) -> dict:
    """Monte Carlo simulation of bankroll evolution.

    Takes historical model and market probabilities from backtest results,
    then simulates many seasons of betting by resampling.

    Args:
        model_probs: List of model-estimated probabilities for bet outcomes.
        market_probs: List of market (implied) probabilities for same outcomes.
        n_simulations: Number of Monte Carlo paths.
        n_bets_per_sim: Bets per simulation (defaults to len(model_probs)).
        initial_bankroll: Starting bankroll.
        rng_seed: Random seed for reproducibility.

    Returns:
        Dict with percentile outcomes, ruin probability, and distribution stats.
    """
    rng = np.random.default_rng(rng_seed)

    if not model_probs or not market_probs:
        return {"error": "No bet data provided"}

    model_arr = np.array(model_probs)
    market_arr = np.array(market_probs)
    n_bets = n_bets_per_sim or len(model_probs)
    n_source = len(model_probs)

    final_bankrolls = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)
    ruin_count = 0

    for sim in range(n_simulations):
        bankroll = initial_bankroll
        peak = bankroll
        max_dd = 0.0

        # Resample bets from the historical distribution
        indices = rng.integers(0, n_source, size=n_bets)

        for idx in indices:
            p_model = model_arr[idx]
            p_market = market_arr[idx]

            kelly_pct = kelly_fraction(p_model, p_market)
            bet_size = min(bankroll * kelly_pct, bankroll * MAX_BET_PCT)
            bet_size = max(0, bet_size)

            if bet_size <= 0:
                continue

            # Simulate outcome based on model probability
            won = rng.random() < p_model
            decimal_odds = 1.0 / p_market if p_market > 0 else 0

            if won:
                bankroll += bet_size * (decimal_odds - 1)
            else:
                bankroll -= bet_size

            # Track drawdown
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

            if bankroll <= 0:
                ruin_count += 1
                bankroll = 0
                break

        final_bankrolls[sim] = bankroll
        max_drawdowns[sim] = max_dd

    # Compute statistics
    profit_arr = final_bankrolls - initial_bankroll
    roi_arr = profit_arr / initial_bankroll

    return {
        "n_simulations": n_simulations,
        "n_bets_per_sim": n_bets,
        "initial_bankroll": initial_bankroll,
        "median_final_bankroll": round(float(np.median(final_bankrolls)), 2),
        "mean_final_bankroll": round(float(np.mean(final_bankrolls)), 2),
        "pct_5": round(float(np.percentile(final_bankrolls, 5)), 2),
        "pct_25": round(float(np.percentile(final_bankrolls, 25)), 2),
        "pct_75": round(float(np.percentile(final_bankrolls, 75)), 2),
        "pct_95": round(float(np.percentile(final_bankrolls, 95)), 2),
        "probability_of_profit": round(float((final_bankrolls > initial_bankroll).mean()), 4),
        "probability_of_ruin": round(ruin_count / n_simulations, 4),
        "median_roi": round(float(np.median(roi_arr)), 4),
        "mean_roi": round(float(np.mean(roi_arr)), 4),
        "median_max_drawdown": round(float(np.median(max_drawdowns)), 4),
        "worst_max_drawdown": round(float(np.max(max_drawdowns)), 4),
    }


def simulate_prediction_confidence(
    model_prob: float,
    market_prob: float,
    noise_std: float = 0.03,
    n_simulations: int = 10_000,
    rng_seed: int = 42,
) -> dict:
    """Estimate confidence interval for a single bet's profitability.

    Adds Gaussian noise to model probability to simulate estimation uncertainty,
    then computes what fraction of simulations remain profitable.

    Args:
        model_prob: Model's estimated probability.
        market_prob: Market implied probability.
        noise_std: Standard deviation of probability noise (default 3%).
        n_simulations: Number of simulations.
        rng_seed: Random seed.

    Returns:
        Dict with P(profitable), edge distribution stats.
    """
    rng = np.random.default_rng(rng_seed)

    noisy_probs = rng.normal(model_prob, noise_std, n_simulations)
    noisy_probs = np.clip(noisy_probs, 0.01, 0.99)

    edges = noisy_probs * (1.0 / market_prob) - 1.0

    return {
        "model_prob": model_prob,
        "market_prob": market_prob,
        "noise_std": noise_std,
        "edge_mean": round(float(edges.mean()), 4),
        "edge_median": round(float(np.median(edges)), 4),
        "edge_5th": round(float(np.percentile(edges, 5)), 4),
        "edge_95th": round(float(np.percentile(edges, 95)), 4),
        "prob_positive_edge": round(float((edges > 0).mean()), 4),
        "prob_above_threshold": round(float((edges >= MIN_EDGE_THRESHOLD).mean()), 4),
    }
