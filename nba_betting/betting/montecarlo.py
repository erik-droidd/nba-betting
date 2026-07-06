"""Monte Carlo simulation for bankroll and prediction confidence.

**Important — why this module was rewritten.**

The original implementation simulated bet outcomes by drawing
``won ~ Bernoulli(p_model)`` — i.e. the model's own predicted
probability was used to flip the coin. That is a tautology: the
simulator assumed the model is perfectly calibrated AND always right
in expectation, which, combined with fractional-Kelly compounding
over ~2000 resampled bets, produced absurd results (median final
bankroll in the trillions, ``P(Profit) = 100%``). It was a bug, not a
feature — it answered the question "what happens if the model is
always correct?" which is uninteresting.

The honest replacement uses **empirical bootstrap resampling**:
we draw index tuples from the backtest's actual ``(p_model, p_market,
won)`` records. If the model truly has edge, that edge is preserved
in the resampled win rate; if it doesn't, the bootstrap will expose
the variance. A second mode ("market_right") assumes the efficient-
market null (``won ~ Bernoulli(p_market)``) to provide a pessimistic
bound.

The "model_right" mode is intentionally not exposed — it always
prints unrealistic numbers and has no legitimate use case.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from nba_betting.betting.kelly import kelly_fraction
from nba_betting.config import DEFAULT_BANKROLL, MAX_BET_PCT, MIN_EDGE_THRESHOLD


SimulationMode = Literal["empirical", "market_right"]


def simulate_bankroll(
    model_probs: Sequence[float],
    market_probs: Sequence[float],
    won_outcomes: Sequence[bool] | None = None,
    mode: SimulationMode = "empirical",
    n_simulations: int = 10_000,
    n_bets_per_sim: int | None = None,
    initial_bankroll: float = DEFAULT_BANKROLL,
    rng_seed: int = 42,
) -> dict:
    """Monte Carlo bankroll simulation with honest outcome generation.

    Takes historical bet records from a backtest and resamples them
    (with replacement) to simulate alternative bankroll trajectories.

    Args:
        model_probs: Historical model-estimated probabilities, one per
            resolved bet.
        market_probs: Historical market (implied) probabilities, aligned
            with ``model_probs``.
        won_outcomes: Historical realized outcomes (True if the bet won)
            aligned with ``model_probs``. **Required when**
            ``mode="empirical"`` — this is how we preserve the actual
            win rate the model achieved, not the win rate it claimed.
        mode: Outcome-generation strategy:

            - ``"empirical"`` (default): on each bet we pick a tuple
              ``(p_model_i, p_market_i, won_i)`` uniformly at random
              from the historical records. The realized win flag is
              used directly. This is the honest bootstrap — it
              preserves the observed edge (or lack thereof) and
              answers "what would the bankroll curve look like on
              another 2000 bets with the same skill?".
            - ``"market_right"``: for each bet, flip a coin with
              probability ``p_market``. This represents the pessimistic
              efficient-market null (the model has zero skill and the
              market price is the truth). Use as a sanity floor —
              the bankroll should drift down under this mode.

        n_simulations: Number of Monte Carlo paths.
        n_bets_per_sim: Bets per simulation. Defaults to
            ``len(model_probs)``. Longer horizons amplify both skill
            and variance.
        initial_bankroll: Starting bankroll.
        rng_seed: Random seed for reproducibility.

    Returns:
        Dict with percentile outcomes, ruin probability, and other
        distribution stats. Adds ``mode`` to the output for clarity.

    Raises:
        ValueError: if ``mode="empirical"`` and ``won_outcomes`` is
            missing, or if input arrays have mismatched lengths.
    """
    # len() instead of truthiness: numpy arrays raise on ambiguous bool().
    if model_probs is None or market_probs is None or len(model_probs) == 0 or len(market_probs) == 0:
        return {"error": "No bet data provided", "mode": mode}

    model_arr = np.asarray(model_probs, dtype=float)
    market_arr = np.asarray(market_probs, dtype=float)

    if model_arr.shape != market_arr.shape:
        raise ValueError(
            "model_probs and market_probs must have the same length"
        )

    if mode == "empirical":
        if won_outcomes is None:
            raise ValueError(
                "mode='empirical' requires won_outcomes (the realized bet "
                "outcomes from backtest). Without it we cannot honestly "
                "resample. Use mode='market_right' if you only have probs."
            )
        won_arr = np.asarray(won_outcomes, dtype=bool)
        if won_arr.shape != model_arr.shape:
            raise ValueError(
                "won_outcomes must align with model_probs/market_probs"
            )
    elif mode == "market_right":
        # won_outcomes is ignored in this mode (we draw from Bernoulli(p_market))
        # but if provided, validate length so a caller bug surfaces early.
        if won_outcomes is not None:
            won_arr_check = np.asarray(won_outcomes, dtype=bool)
            if won_arr_check.shape != model_arr.shape:
                raise ValueError(
                    "won_outcomes must align with model_probs/market_probs"
                )
        won_arr = None  # outcomes drawn from p_market during sim
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. Use 'empirical' or 'market_right'."
        )

    rng = np.random.default_rng(rng_seed)
    n_source = len(model_arr)
    n_bets = n_bets_per_sim or n_source

    # Vectorized simulation: pre-sample all indices (S × B), pre-compute
    # Kelly fractions and decimal odds for the source pool, then loop over
    # bets (n_bets ≈ 200) while operating on all simulations in parallel.
    # Reduces ~12M Python iterations to ~200 NumPy array operations.

    # Pre-sample bootstrap indices: shape (n_simulations, n_bets)
    all_indices = rng.integers(0, n_source, size=(n_simulations, n_bets))

    # Pre-compute Kelly fractions for each source record (fraction of bankroll).
    # kelly_fraction already caps at MAX_BET_PCT, so no second min() needed.
    kelly_fracs = np.array(
        [kelly_fraction(float(model_arr[j]), float(market_arr[j])) for j in range(n_source)],
        dtype=float,
    )

    # Pre-compute decimal odds for each source record.
    dec_odds = np.where(market_arr > 0, 1.0 / market_arr, 0.0)

    # Pre-generate all outcomes: shape (n_simulations, n_bets).
    if mode == "empirical":
        outcomes = won_arr[all_indices]  # boolean array, no random draws needed
    else:  # market_right
        outcomes = rng.random(size=(n_simulations, n_bets)) < market_arr[all_indices]

    # Path-dependent bankroll evolution: loop over bet steps, vectorize over sims.
    # Once a simulation's bankroll reaches 0 it stays at 0 (bet_size = 0 × frac = 0).
    bankrolls = np.full(n_simulations, float(initial_bankroll))
    peaks = bankrolls.copy()
    max_dds = np.zeros(n_simulations)

    for t in range(n_bets):
        idx_t = all_indices[:, t]           # (S,) source indices for this bet step
        kf_t = kelly_fracs[idx_t]           # (S,) Kelly fractions
        bet_sizes = bankrolls * kf_t        # (S,) zero when bankroll=0 or kf=0
        won_t = outcomes[:, t]              # (S,) boolean outcomes
        odds_t = dec_odds[idx_t]            # (S,) decimal odds

        profit = np.where(won_t, bet_sizes * (odds_t - 1.0), -bet_sizes)
        bankrolls = np.maximum(bankrolls + profit, 0.0)  # floor at 0 on ruin

        peaks = np.maximum(peaks, bankrolls)
        cur_dd = np.where(peaks > 0, (peaks - bankrolls) / peaks, 0.0)
        max_dds = np.maximum(max_dds, cur_dd)

    final_bankrolls = bankrolls
    max_drawdowns = max_dds
    ruin_count = int((final_bankrolls == 0).sum())

    profit_arr = final_bankrolls - initial_bankroll
    roi_arr = profit_arr / initial_bankroll

    # Horizon-invariant geometric mean log-return per bet.
    # This is the honest measure of skill because it doesn't explode
    # with the number of bets. Compounded bankrolls can look absurd
    # (e.g. $1B after 2000 bets with 0.5% edge per bet) but
    # per-bet log growth is bounded and directly comparable
    # across empirical vs. market_right. The gap between the two is
    # the cleanest statement of skill.
    safe_finals = np.maximum(final_bankrolls, 1e-9)
    log_growth_per_bet = np.log(safe_finals / initial_bankroll) / max(n_bets, 1)

    return {
        "mode": mode,
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
        # Horizon-invariant metrics — these are the honest diagnostic.
        "median_log_growth_per_bet": round(float(np.median(log_growth_per_bet)), 6),
        "mean_log_growth_per_bet": round(float(np.mean(log_growth_per_bet)), 6),
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
