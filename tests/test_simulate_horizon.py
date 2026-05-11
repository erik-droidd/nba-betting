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


# ---------------------------------------------------------------------------
# Default horizon is now date-density derived — replaces the prior
# `_DEFAULT_HORIZON = 200` constant. The cap was arbitrary: a heavy bettor
# saw a horizon ~half a season's worth, a selective bettor saw three
# seasons in a single horizon. The data-driven default fixes both edges.
# ---------------------------------------------------------------------------


def _synthetic_bets(start_date: str, span_days: int, bets_per_day: float) -> list[dict]:
    """Build a synthetic backtest bet list with controlled date density.

    Each returned dict has a ``"date"`` field (the only field
    ``_estimate_one_season_bets`` reads). Bets are spread linearly across
    the span so ``len(out) / span_days == bets_per_day`` to within rounding.
    """
    from datetime import date, timedelta
    start = date.fromisoformat(start_date)
    n_total = round(span_days * bets_per_day)
    return [
        {"date": (start + timedelta(days=round(i * span_days / max(n_total, 1)))).isoformat()}
        for i in range(n_total)
    ]


def test_estimate_one_season_projects_density_over_240_days():
    """A 3-season backtest at 2 bets/day → ~480 bets/season default."""
    from nba_betting.cli import _estimate_one_season_bets

    bets = _synthetic_bets("2023-01-01", span_days=900, bets_per_day=2.0)
    horizon, reason = _estimate_one_season_bets(bets)
    # 240 days × 2 bets/day = 480, allow rounding slop.
    assert 450 <= horizon <= 510, (
        f"3-season × 2 bets/day backtest should project to ~480 bets/season, "
        f"got {horizon} ({reason})"
    )
    assert "season" in reason.lower()


def test_estimate_one_season_scales_with_bet_rate():
    """Selective bettor (0.5 bets/day) → smaller horizon than heavy
    bettor (5 bets/day), both for the same backtest span.

    This is the property the old fixed 200 cap couldn't deliver: it gave
    every user the same horizon regardless of how often they bet.
    """
    from nba_betting.cli import _estimate_one_season_bets

    selective = _synthetic_bets("2023-01-01", span_days=900, bets_per_day=0.5)
    heavy = _synthetic_bets("2023-01-01", span_days=900, bets_per_day=5.0)

    h_selective, _ = _estimate_one_season_bets(selective)
    h_heavy, _ = _estimate_one_season_bets(heavy)

    assert h_heavy > h_selective * 5, (
        f"heavy bettor should have ≥5× the horizon of a selective one, "
        f"got selective={h_selective}, heavy={h_heavy}"
    )


def test_estimate_one_season_falls_back_to_pool_for_short_backtest():
    """A 2-week backtest's density would extrapolate absurdly — fall back
    to the full pool and surface the reason so the user knows.
    """
    from nba_betting.cli import _estimate_one_season_bets

    bets = _synthetic_bets("2023-01-01", span_days=14, bets_per_day=3.0)
    horizon, reason = _estimate_one_season_bets(bets)
    assert horizon == len(bets), (
        f"short backtests should fall back to full pool, got horizon={horizon} "
        f"vs pool={len(bets)}"
    )
    assert "full pool" in reason.lower()


def test_estimate_one_season_caps_at_pool_size():
    """If the projection exceeds the available pool (rare but possible
    with a short, dense backtest like 60 days at 4 bets/day), cap at the
    pool size rather than oversample.
    """
    from nba_betting.cli import _estimate_one_season_bets

    # 60 days × 4 bets/day = 240 bets, projection wants 240×4 = 960.
    bets = _synthetic_bets("2023-01-01", span_days=60, bets_per_day=4.0)
    horizon, _ = _estimate_one_season_bets(bets)
    assert horizon <= len(bets), (
        f"horizon must never exceed pool size, got horizon={horizon} "
        f"vs pool={len(bets)}"
    )


def test_estimate_one_season_handles_missing_or_bad_dates():
    """Unparseable date field shouldn't crash — fall back to pool size."""
    from nba_betting.cli import _estimate_one_season_bets

    bets = [{"date": "not-a-date"} for _ in range(50)]
    horizon, reason = _estimate_one_season_bets(bets)
    assert horizon == len(bets)
    assert "full pool" in reason.lower()
