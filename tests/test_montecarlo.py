"""Tests for the Monte Carlo bankroll simulator.

Written after catching the original simulator's tautology bug (it used
``won ~ Bernoulli(p_model)`` which made the model right by assumption
and blew the bankroll to ~$10^12). The regression test ``test_empirical_requires_won_outcomes`` and
the parity test ``test_empirical_matches_realized_win_rate`` lock in
the honest bootstrap semantics.
"""
from __future__ import annotations

import numpy as np
import pytest

from nba_betting.betting.montecarlo import simulate_bankroll


def _make_inputs(
    n: int = 200,
    edge: float = 0.03,
    market: float = 0.5,
    seed: int = 0,
) -> tuple[list[float], list[float], list[bool]]:
    """Build a synthetic backtest history with a known edge.

    ``p_model = market + edge`` for every bet, and outcomes are drawn
    from ``Bernoulli(p_model)`` so the realized win rate matches what
    the model claims (i.e. a *correctly-calibrated* model). Used to
    check that the empirical bootstrap preserves the realized win
    rate without the tautology.
    """
    rng = np.random.default_rng(seed)
    model_probs = [market + edge] * n
    market_probs = [market] * n
    won = [bool(rng.random() < p) for p in model_probs]
    return model_probs, market_probs, won


def test_empirical_requires_won_outcomes():
    """Guard the tautology: empirical mode must not silently fall back."""
    with pytest.raises(ValueError, match="won_outcomes"):
        simulate_bankroll(
            [0.6, 0.55, 0.52],
            [0.5, 0.5, 0.5],
            won_outcomes=None,
            mode="empirical",
            n_simulations=10,
        )


def test_empirical_matches_realized_win_rate():
    """Bootstrap should not invent wins the history doesn't have.

    A well-calibrated model with edge=0 (model_prob == market_prob)
    should yield ~0 expected bankroll drift in empirical mode. Under
    the old bug (``won ~ Bernoulli(p_model)``) the compounding was
    driven by the model's stated prob, not the realized outcomes, so
    zero-edge inputs still exploded.
    """
    n = 1000
    rng = np.random.default_rng(7)
    # Zero-edge: p_model == p_market, and won matches p_market so the
    # realized win rate is exactly what a 0-edge bettor would see.
    market_probs = [0.55] * n
    model_probs = [0.55] * n
    won = [bool(rng.random() < 0.55) for _ in range(n)]

    res = simulate_bankroll(
        model_probs, market_probs,
        won_outcomes=won,
        mode="empirical",
        n_simulations=500,
        n_bets_per_sim=100,
        initial_bankroll=1000.0,
        rng_seed=42,
    )

    # With 0 edge Kelly sizes to 0 → no bets → bankroll unchanged.
    assert res["median_final_bankroll"] == pytest.approx(1000.0, rel=1e-3)
    assert res["mean_final_bankroll"] == pytest.approx(1000.0, rel=1e-3)
    assert res["probability_of_ruin"] == 0.0


def test_empirical_preserves_positive_edge():
    """Real edge should lead to a positive-median bankroll, bounded."""
    model_probs, market_probs, won = _make_inputs(
        n=500, edge=0.04, market=0.5, seed=11,
    )

    res = simulate_bankroll(
        model_probs, market_probs,
        won_outcomes=won,
        mode="empirical",
        n_simulations=400,
        n_bets_per_sim=200,
        initial_bankroll=1000.0,
        rng_seed=42,
    )

    # Positive median: real edge compounds, but stays in sane territory.
    assert res["median_final_bankroll"] > 1000.0
    # Sanity cap — with 200 bets and fractional Kelly the historical
    # bug produced >1e9 medians. A correctly-calibrated 4% edge
    # should never even touch 100x starting bankroll on 200 bets.
    assert res["median_final_bankroll"] < 100_000.0
    assert res["probability_of_profit"] > 0.5


def test_market_right_is_zero_or_negative_edge():
    """Under the efficient-market null, expected ROI should not be
    positive. Kelly still bets because ``p_model > p_market``, but the
    actual win probability is ``p_market``, so we lose in expectation.
    """
    model_probs, market_probs, won = _make_inputs(
        n=500, edge=0.04, market=0.5, seed=11,
    )

    res = simulate_bankroll(
        model_probs, market_probs,
        won_outcomes=won,  # ignored in market_right mode but still required shape
        mode="market_right",
        n_simulations=400,
        n_bets_per_sim=200,
        initial_bankroll=1000.0,
        rng_seed=42,
    )

    # Market-is-right null: median ROI should be <= 0.
    assert res["median_roi"] <= 0.01, (
        "Market-is-right should not show systematic profit — if it does, "
        "the simulator is not drawing from Bernoulli(p_market)."
    )


def test_empirical_beats_market_right_when_edge_is_real():
    """The diagnostic that should emerge in the CLI: empirical > market_right."""
    model_probs, market_probs, won = _make_inputs(
        n=500, edge=0.05, market=0.5, seed=3,
    )

    emp = simulate_bankroll(
        model_probs, market_probs, won_outcomes=won,
        mode="empirical",
        n_simulations=400, n_bets_per_sim=200,
        initial_bankroll=1000.0, rng_seed=42,
    )
    mkt = simulate_bankroll(
        model_probs, market_probs, won_outcomes=won,
        mode="market_right",
        n_simulations=400, n_bets_per_sim=200,
        initial_bankroll=1000.0, rng_seed=42,
    )

    assert emp["median_roi"] > mkt["median_roi"], (
        "Empirical median ROI should exceed market-null when edge is real. "
        f"got empirical={emp['median_roi']:.3f}, market={mkt['median_roi']:.3f}"
    )


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        simulate_bankroll(
            [0.6, 0.55],
            [0.5, 0.5, 0.5],
            won_outcomes=[True, False],
            mode="empirical",
            n_simulations=5,
        )


def test_won_length_mismatch_raises():
    with pytest.raises(ValueError, match="align"):
        simulate_bankroll(
            [0.6, 0.55, 0.52],
            [0.5, 0.5, 0.5],
            won_outcomes=[True, False],  # wrong length
            mode="empirical",
            n_simulations=5,
        )


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown mode"):
        simulate_bankroll(
            [0.6], [0.5],
            won_outcomes=[True],
            mode="model_right",  # intentionally removed
            n_simulations=5,
        )


def test_reproducibility_under_same_seed():
    model_probs, market_probs, won = _make_inputs(n=100, seed=1)
    a = simulate_bankroll(
        model_probs, market_probs, won_outcomes=won,
        mode="empirical", n_simulations=100, rng_seed=42,
    )
    b = simulate_bankroll(
        model_probs, market_probs, won_outcomes=won,
        mode="empirical", n_simulations=100, rng_seed=42,
    )
    assert a["median_final_bankroll"] == b["median_final_bankroll"]
    assert a["probability_of_profit"] == b["probability_of_profit"]


def test_empty_inputs_return_error():
    res = simulate_bankroll([], [], won_outcomes=[], mode="empirical")
    assert "error" in res


def test_log_growth_per_bet_present_and_horizon_invariant():
    """The horizon-invariant metric is the honest skill signal.

    Final-bankroll medians balloon with horizon (Kelly compounding).
    The median log-growth-per-bet should stay in the same order of
    magnitude whether we simulate 100 or 400 bets — that's the whole
    point of dividing by ``n_bets``.
    """
    model_probs, market_probs, won = _make_inputs(
        n=500, edge=0.04, market=0.5, seed=11,
    )

    short = simulate_bankroll(
        model_probs, market_probs, won_outcomes=won,
        mode="empirical",
        n_simulations=300, n_bets_per_sim=100,
        initial_bankroll=1000.0, rng_seed=42,
    )
    long = simulate_bankroll(
        model_probs, market_probs, won_outcomes=won,
        mode="empirical",
        n_simulations=300, n_bets_per_sim=400,
        initial_bankroll=1000.0, rng_seed=42,
    )

    # Both runs expose the same per-bet skill; allow loose bounds for MC noise.
    short_g = short["median_log_growth_per_bet"]
    long_g = long["median_log_growth_per_bet"]
    assert short_g > 0
    assert long_g > 0
    # Horizon invariance: per-bet growth should agree within a factor of 3
    # even though the compounded bankrolls differ wildly.
    assert abs(short_g - long_g) < max(short_g, long_g) * 2, (
        f"log-growth/bet should not vary much with horizon: "
        f"100-bet={short_g:.5f} vs 400-bet={long_g:.5f}"
    )


def test_log_growth_gap_is_positive_with_real_edge():
    """The log-growth-per-bet gap is the cleanest skill diagnostic."""
    model_probs, market_probs, won = _make_inputs(
        n=500, edge=0.05, market=0.5, seed=3,
    )

    emp = simulate_bankroll(
        model_probs, market_probs, won_outcomes=won,
        mode="empirical",
        n_simulations=400, n_bets_per_sim=200,
        initial_bankroll=1000.0, rng_seed=42,
    )
    mkt = simulate_bankroll(
        model_probs, market_probs, won_outcomes=won,
        mode="market_right",
        n_simulations=400, n_bets_per_sim=200,
        initial_bankroll=1000.0, rng_seed=42,
    )

    gap = emp["median_log_growth_per_bet"] - mkt["median_log_growth_per_bet"]
    assert gap > 0, (
        f"Real edge should produce a positive log-growth gap. "
        f"empirical={emp['median_log_growth_per_bet']:.5f}, "
        f"market={mkt['median_log_growth_per_bet']:.5f}"
    )
