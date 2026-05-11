"""Tests for the `simulate` CLI's horizon-by-default + live-strategy-by-default.

Context: with the Elo-proxy backtest's positive per-bet log-growth, the prior
default of `n_bets_per_sim = len(backtest_bets)` (~2000-3000 bets across 3
seasons) compounded P(Profit) toward 100% by arithmetic — mathematically
correct but operationally useless. The fix:

1. ``simulate`` CLI now applies live strategy (Bayesian shrinkage + bet-side
   floor) in the underlying backtest by default. This matches what `predict`
   actually does and produces a better-calibrated bet pool.
2. ``simulate`` defaults ``n_bets_per_sim`` to ``min(200, len(bets))`` —
   one season's worth of high-conviction bets. The full-pool behavior is
   still available via ``--horizon <full>``.

These tests pin both decisions so a refactor doesn't silently restore the
2000+-bet default that saturates P(Profit).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CLI surface: option defaults wired correctly
# ---------------------------------------------------------------------------


def _cli_default(name: str):
    """Return the underlying default value for a typer Option.

    Typer wraps option defaults in ``OptionInfo``; the actual user-facing
    default lives on ``OptionInfo.default``. For positional/non-typer
    params we just return the inspect default directly.
    """
    from nba_betting.cli import simulate
    sig = inspect.signature(simulate)
    if name not in sig.parameters:
        raise AssertionError(f"`simulate` has no parameter `{name}`")
    raw = sig.parameters[name].default
    return getattr(raw, "default", raw)


def test_simulate_has_horizon_option_defaulting_to_none():
    """--horizon must exist and default to None (CLI resolves at call time).

    Pinning ``None`` (not the integer default) because the resolution rule
    ``min(200, len(bets))`` requires knowing the bet pool size and can't
    live in the signature default.
    """
    assert _cli_default("horizon") is None, (
        "horizon default should be None (resolved at runtime to "
        "min(200, len(bets)))"
    )


def test_simulate_has_live_strategy_option_defaulting_to_none():
    """--live-strategy/--no-live-strategy with None default.

    Resolution rule: ``True if live_strategy is None else live_strategy`` —
    we want the simulator to project the live system by default, but the
    signature default is None so users can pass --no-live-strategy
    explicitly.
    """
    assert _cli_default("live_strategy") is None, (
        "live_strategy default should be None (resolved at runtime to True "
        "so the simulator projects the live system)"
    )


# ---------------------------------------------------------------------------
# Underlying math: shorter horizon must expose downside variance that the
# full-pool default hides.
# ---------------------------------------------------------------------------


def _fake_positive_edge_pool(n: int = 200, seed: int = 0):
    """Build a synthetic bet pool with real ~+0.5%/bet log-growth.

    Model claims p=0.60, market p=0.50, true win rate 0.58. This gives the
    same flavor of inflated edge that the Elo-proxy backtest produces, so
    the regression test mirrors the production failure mode.
    """
    rng = np.random.default_rng(seed)
    model_probs = [0.60] * n
    market_probs = [0.50] * n
    # Real win rate 0.58 — the model is overconfident by 0.02 but the
    # edge is still positive.
    wins = (rng.random(n) < 0.58).tolist()
    return model_probs, market_probs, wins


def test_short_horizon_exposes_downside_that_full_pool_hides():
    """At horizon=2000, P(Profit) saturates ≈ 100%; at horizon=50,
    P(Profit) is materially below 100% so the user can see the downside
    distribution. This is the core reason the CLI defaults the horizon to
    a one-season-ish number rather than the full backtest pool.
    """
    from nba_betting.betting.montecarlo import simulate_bankroll

    mp, mk, w = _fake_positive_edge_pool(n=300, seed=1)

    long_horizon = simulate_bankroll(
        mp, mk, won_outcomes=w, mode="empirical",
        n_simulations=1500, n_bets_per_sim=2000, rng_seed=1,
    )
    short_horizon = simulate_bankroll(
        mp, mk, won_outcomes=w, mode="empirical",
        n_simulations=1500, n_bets_per_sim=50, rng_seed=1,
    )

    # Long horizon saturates — this is the failure mode.
    assert long_horizon["probability_of_profit"] >= 0.98, (
        "expected long horizon to saturate P(Profit) ≥ 98% (the bug "
        "we're protecting against), got "
        f"{long_horizon['probability_of_profit']:.3f}"
    )
    # Short horizon must leave meaningful downside visible.
    assert short_horizon["probability_of_profit"] < 0.97, (
        "expected short horizon to expose downside (P(Profit) < 97% so "
        "the percentile distribution actually informs the user), got "
        f"{short_horizon['probability_of_profit']:.3f}"
    )
    # And the log-growth/bet must be approximately unchanged between
    # horizons — that's the horizon-invariant skill metric.
    assert abs(
        long_horizon["median_log_growth_per_bet"]
        - short_horizon["median_log_growth_per_bet"]
    ) < 0.002, "log-growth/bet should be roughly horizon-invariant"


def test_default_horizon_constant_is_one_season_ish():
    """The default-horizon constant in cli.simulate must be in the
    100-300 range. Below 100 loses statistical signal; above 300 starts
    re-saturating P(Profit) on an Elo-proxy edge.
    """
    import re
    from nba_betting import cli
    source = inspect.getsource(cli.simulate)
    m = re.search(r"_DEFAULT_HORIZON\s*=\s*(\d+)", source)
    assert m is not None, "couldn't find _DEFAULT_HORIZON constant in cli.simulate"
    value = int(m.group(1))
    assert 100 <= value <= 300, (
        f"_DEFAULT_HORIZON should be in the 100-300 range "
        f"(~one season of high-conviction bets), got {value}"
    )
