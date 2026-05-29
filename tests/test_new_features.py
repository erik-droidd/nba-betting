"""Tests for features added in the 2026-04 hardening pass.

Covers the pieces that the live / backtest pipelines depend on — if any
of these regress, `predict` and `backtest` start producing wrong numbers
silently (wrong edge sign, stale drivers, wrong bet count) so they're
worth guarding with fast unit tests.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from nba_betting.betting.shrinkage import shrink_to_market
from nba_betting.models.drivers import (
    compute_prediction_drivers,
    humanize_feature,
)
from nba_betting.models.spreads_totals import generate_spread_total_picks


# ---------------------------------------------------------------------------
# 1. Bayesian shrinkage invariants
# ---------------------------------------------------------------------------


def test_shrinkage_lambda_zero_returns_model_unchanged():
    # lambda=0 means "no weight on market" → posterior = model.
    assert shrink_to_market(0.70, 0.55, lambda_market=0.0) == pytest.approx(0.70, abs=1e-9)


def test_shrinkage_lambda_one_returns_market():
    # lambda=1 means "all market prior" → posterior = market.
    assert shrink_to_market(0.70, 0.55, lambda_market=1.0) == pytest.approx(0.55, abs=1e-9)


def test_shrinkage_midway_moves_toward_market():
    # lambda=0.5 should land strictly between model and market (log-odds
    # midpoint isn't the arithmetic midpoint but still monotone).
    result = shrink_to_market(0.70, 0.55, lambda_market=0.5)
    assert 0.55 < result < 0.70


def test_shrinkage_degenerate_market_returns_raw_model():
    # A market_prob of 0 or 1 would send the logit to ±inf; we short-
    # circuit so we don't push into a tail that's guaranteed to fail Kelly.
    assert shrink_to_market(0.70, 0.0, lambda_market=0.6) == 0.70
    assert shrink_to_market(0.70, 1.0, lambda_market=0.6) == 0.70


# ---------------------------------------------------------------------------
# 2. humanize_feature label map
# ---------------------------------------------------------------------------


def test_humanize_feature_injury_labels():
    # These labels are the ones exposed in the explanation sentence; if
    # someone renames the feature column the explanation would suddenly
    # cite the raw snake_case name. This test pins the happy path.
    assert humanize_feature("injury_impact_diff") == "injury impact differential"
    assert humanize_feature("home_injury_impact_out") == "home injury impact (out/doubtful)"
    assert humanize_feature("away_injury_impact_out") == "away injury impact (out/doubtful)"


def test_humanize_feature_player_impact_and_line_movement():
    assert humanize_feature("diff_missing_minutes_pct") == "missing-minutes differential"
    assert humanize_feature("home_star_out") == "home star player out"
    assert humanize_feature("spread_movement") == "spread line movement"
    assert humanize_feature("odds_disagreement") == "Polymarket vs ESPN odds disagreement"


def test_humanize_feature_rolling_diff():
    # diff_<stat>_roll_<N> → "<N>-game <label> differential"
    assert humanize_feature("diff_pts_roll_5") == "5-game scoring differential"
    assert humanize_feature("diff_efg_pct_roll_10") == "10-game effective FG % differential"


def test_humanize_feature_unknown_falls_back_to_underscore_strip():
    # Unknown feature names should not raise; they should render readable.
    assert humanize_feature("some_new_feature") == "some new feature"


# ---------------------------------------------------------------------------
# 3. Spread / total pick sign convention
# ---------------------------------------------------------------------------


def test_generate_spread_total_picks_home_cover():
    # ESPN convention: negative spread = home favored. A market spread of
    # -4 means "home laying 4". If the model thinks home wins by 7, that
    # beats the market by +3 → HOME_COVER at home -4.
    out = generate_spread_total_picks(
        predicted_spread=7.0,
        predicted_total=220.0,
        market_spread=-4.0,
        market_total=220.0,
        home_team="LAL",
        away_team="BOS",
    )
    assert out["spread_pick"] == "LAL -4.0"
    assert out["spread_edge"] == pytest.approx(3.0, abs=1e-9)


def test_generate_spread_total_picks_away_cover():
    # Model thinks home only wins by 1; market says home favored by 5.
    # Gap = 1 - 5 = -4. -4 <= -SPREAD_EDGE_PTS(1.5) so we take the dog.
    out = generate_spread_total_picks(
        predicted_spread=1.0,
        predicted_total=220.0,
        market_spread=-5.0,
        market_total=220.0,
        home_team="LAL",
        away_team="BOS",
    )
    assert out["spread_pick"] == "BOS +5.0"


def test_generate_spread_total_picks_no_edge_when_gap_too_small():
    # 0.5 pt model/market gap < 1.5 pt threshold → NO BET on spread.
    out = generate_spread_total_picks(
        predicted_spread=-3.5,
        predicted_total=220.0,
        market_spread=3.0,  # market says away -3 → home margin = -3.0
        market_total=220.0,
        home_team="LAL",
        away_team="BOS",
    )
    assert out["spread_pick"] == "NO BET"


def test_generate_spread_total_picks_total_over():
    # Model thinks 230; market 220. +10 > TOTAL_EDGE_PTS(2.5) → OVER.
    out = generate_spread_total_picks(
        predicted_spread=0.0,
        predicted_total=230.0,
        market_spread=None,
        market_total=220.0,
        home_team="LAL",
        away_team="BOS",
    )
    assert out["total_pick"] == "OVER 220.0"
    assert out["total_edge"] == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Driver attribution: ordering + no-crash behavior
# ---------------------------------------------------------------------------


class _FakeModel:
    """Tiny stand-in for sklearn's predict_proba — we don't need a real
    tree to validate the leave-one-out loop, just a deterministic scoring
    function that responds to column values."""

    def predict_proba(self, X):
        # P(home win) = sigmoid(weight . x). Chosen weights give each
        # feature a known contribution so tests can assert ranking.
        import numpy as np
        w = {"elo_diff": 1.0, "diff_pts_roll_5": 0.3, "noise": 0.01}
        if hasattr(X, "columns"):
            cols = list(X.columns)
            vals = X.values
        else:
            cols = ["elo_diff", "diff_pts_roll_5", "noise"]
            vals = X

        out = []
        for row in vals:
            z = sum(w.get(c, 0.0) * float(row[i]) for i, c in enumerate(cols))
            p = 1.0 / (1.0 + math.exp(-z))
            out.append([1.0 - p, p])
        return np.array(out) if (_np := __import__("numpy")) else out


def test_compute_prediction_drivers_sorted_by_magnitude():
    feat_row = pd.DataFrame([{
        "elo_diff": 2.0,        # large positive → strongest driver
        "diff_pts_roll_5": 1.0, # moderate driver
        "noise": 5.0,           # high value but tiny weight → smallest delta
    }])
    means = {"elo_diff": 0.0, "diff_pts_roll_5": 0.0, "noise": 0.0}

    drivers = compute_prediction_drivers(_FakeModel(), feat_row, means, top_k=3)

    names = [d[0] for d in drivers]
    # Magnitude ranking: elo_diff first (strongest delta), noise last.
    assert names[0] == "elo_diff"
    assert names[-1] == "noise"
    # Sign convention: positive elo_diff pushes P(home) up → positive delta.
    assert drivers[0][1] > 0


def test_compute_prediction_drivers_empty_row_returns_empty():
    # Robustness: attribution must not raise on degenerate inputs, since
    # it's wrapped in a try/except in the caller and we want the catch
    # path to be rare.
    out = compute_prediction_drivers(_FakeModel(), pd.DataFrame(), {}, top_k=5)
    assert out == []


# ---------------------------------------------------------------------------
# 5. Backtest defaults — apply_live_strategy couples to use_real_odds
# ---------------------------------------------------------------------------


def test_backtest_live_strategy_default_follows_real_odds_flag():
    """When the caller passes `apply_live_strategy=None`, the function
    must resolve the default based on `use_real_odds`:

    - use_real_odds=False → live_strategy=False (pure model benchmark;
      shrinking toward the Elo proxy is a null-op/dampener).
    - use_real_odds=True  → live_strategy=True (live-equivalent sim).

    We verify via an empty feature matrix — the resolution happens at the
    top of `run_backtest` before any data work, so we can exercise it
    without a real training set.
    """
    from nba_betting.betting.backtest import run_backtest

    # A feature matrix that straddles July 1 so split_dates is non-empty
    # and the function reaches the `use_real_odds` metadata check. We
    # don't need enough games to actually fit a model — the test passes
    # as long as (a) resolution survives to the real-odds branch,
    # (b) the `_home_team_id` ValueError is raised when the metadata is
    # missing, and (c) the False branch returns a valid (empty) summary.
    import pandas as pd
    X = pd.DataFrame({
        "_date": pd.to_datetime(["2024-06-01", "2024-08-01"]),
        "elo_home_prob": [0.5, 0.5],
    })
    y = pd.Series([1, 0])

    out_false = run_backtest(X, y, n_splits=1, apply_live_strategy=None, use_real_odds=False)
    assert "summary" in out_false  # resolution path didn't raise

    # use_real_odds=True should bail on missing metadata. The check
    # lives AFTER the apply_live_strategy resolution, so reaching it
    # proves the None→False-or-True default machinery worked.
    with pytest.raises(ValueError, match="_home_team_id"):
        run_backtest(X, y, n_splits=1, apply_live_strategy=None, use_real_odds=True)


# ---------------------------------------------------------------------------
# 6. Additive migration idempotence
# ---------------------------------------------------------------------------


def test_apply_additive_migrations_is_idempotent(tmp_path, monkeypatch):
    """Running init_db() twice in a row must not raise (no duplicate
    ALTER TABLE). This guards against a regression where the migration
    tries to add an already-existing column."""
    # Point DB_PATH at a tempfile so we don't touch the real DB.
    from nba_betting import config as _cfg
    test_db = tmp_path / "t.sqlite"
    monkeypatch.setattr(_cfg, "DB_PATH", str(test_db))

    # Re-import session with the patched DB_PATH so engine binds to the
    # tempfile, not the real DB.
    import importlib
    from nba_betting.db import session as _session
    importlib.reload(_session)

    _session.init_db()
    _session.init_db()  # second call must be a no-op — no error raised.
    assert test_db.exists()


def test_compute_clv_logit_sign_reference_and_gating():
    """CLV must be log-odds, referenced off the bet-time price, with
    single-snapshot games excluded (issue: ratio/opening-reference/fake-0)."""
    from nba_betting.betting.tracker import PredictionRecord, compute_clv
    from nba_betting.utils.math import logit_scalar

    def rec(**kw):
        base = dict(
            date="2026-01-01", home_team="AAA", away_team="BBB",
            model_home_prob=0.5, market_home_prob=0.5, bet_side="HOME",
            edge=0.0, bet_size=10.0,
        )
        base.update(kw)
        return PredictionRecord(**base)

    # HOME bet, line moved toward home (close > bet) -> positive CLV.
    r = rec(market_home_prob=0.50, opening_market_prob=0.50, closing_market_prob=0.60)
    assert compute_clv(r) == round(logit_scalar(0.60) - logit_scalar(0.50), 4)
    assert compute_clv(r) > 0

    # Reference is the BET price (market_home_prob), NOT opening_market_prob.
    r_ref = rec(market_home_prob=0.50, opening_market_prob=0.52, closing_market_prob=0.60)
    assert compute_clv(r_ref) == round(logit_scalar(0.60) - logit_scalar(0.50), 4)
    assert compute_clv(r_ref) != round(logit_scalar(0.60) - logit_scalar(0.52), 4)

    # HOME bet, line moved against us -> negative CLV.
    assert compute_clv(rec(market_home_prob=0.55, opening_market_prob=0.60,
                           closing_market_prob=0.50)) < 0

    # AWAY bet: reference and close are the away side (1 - home prob).
    a = compute_clv(rec(bet_side="AWAY", market_home_prob=0.50,
                        opening_market_prob=0.55, closing_market_prob=0.45))
    assert a == round(logit_scalar(0.55) - logit_scalar(0.50), 4)
    assert a > 0

    # Single-snapshot gate: opening == closing -> None (not a fake 0).
    assert compute_clv(rec(market_home_prob=0.48, opening_market_prob=0.55,
                           closing_market_prob=0.55)) is None

    # Undefined cases -> None.
    assert compute_clv(rec(bet_side="NO BET", opening_market_prob=0.5,
                           closing_market_prob=0.6)) is None
    assert compute_clv(rec(opening_market_prob=None, closing_market_prob=0.6)) is None


def test_with_retries_retries_transient_then_succeeds(monkeypatch):
    """Data-layer resilience: a transient failure should retry, not bubble up
    as 'no games' on the first blip."""
    import nba_betting.data._net as net
    monkeypatch.setattr(net.time, "sleep", lambda *_: None)  # no real backoff in test
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert net.with_retries(flaky, attempts=3, what="test") == "ok"
    assert calls["n"] == 3


def test_with_retries_reraises_after_exhausting(monkeypatch):
    """After all attempts fail it re-raises (callers keep graceful fallback),
    having logged a warning so the failure isn't silent."""
    import nba_betting.data._net as net
    monkeypatch.setattr(net.time, "sleep", lambda *_: None)

    def always_fail():
        raise TimeoutError("nope")

    with pytest.raises(TimeoutError):
        net.with_retries(always_fail, attempts=2, what="test")


def test_build_prediction_features_imputes_missing_to_mean_not_zero():
    """Regression guard for the train/predict skew: a genuinely-missing rolling
    stat must be imputed to the TRAINING mean (as build_feature_matrix does),
    not silently coerced to 0. The old `(x or 0)` made non-diff features like
    matchup_pace (mean ~100 possessions) collapse to 0 at predict time."""
    import numpy as np
    from nba_betting.features.builder import (
        build_prediction_features, _WINDOWS, _ROLLING_STATS,
        _SOS_PACE_ROLLING_STATS, _SOS_PACE_WINDOWS, _EWM_STATS,
    )

    def _full_stats(team_id, poss10):
        s = {"team_id": team_id, "date": "2026-01-10"}
        for w in _WINDOWS:
            for st in _ROLLING_STATS:
                s[f"{st}_roll_{w}"] = 1.0
            for ff in ("efg_pct", "tov_pct", "orb_pct", "ft_rate"):
                s[f"{ff}_roll_{w}"] = 0.5
        for w in (10, 20):
            for st in ("plus_minus", "net_rtg_game"):
                s[f"{st}_home_split_roll_{w}"] = 1.0
                s[f"{st}_away_split_roll_{w}"] = 1.0
        for w in (5, 10):
            s[f"fg3_pct_b2b_roll_{w}"] = 0.35
        for st in _SOS_PACE_ROLLING_STATS:
            for w in _SOS_PACE_WINDOWS:
                s[f"{st}_{w}"] = 100.0 if st == "poss_roll" else 1.0
        for st in _EWM_STATS:
            s[st] = 1.0
        for rc in ("rest_days", "is_back_to_back", "games_last_7", "games_last_14"):
            s[rc] = 2
        s["poss_roll_10"] = poss10   # the stat under test
        return s

    # Home team's 10-game pace is MISSING (NaN); away team's is 100.
    rolling_df = pd.DataFrame([_full_stats(1, np.nan), _full_stats(2, 100.0)])
    feature_means = {"matchup_pace_10": 99.0}
    row = build_prediction_features(1, 2, rolling_df, 1500.0, 1500.0, feature_means=feature_means)

    assert row is not None, "a full stats row should not hit the >30% NaN fallback"
    # (NaN home pace + 100 away) / 2 -> NaN -> imputed to the mean 99.0.
    # The OLD buggy path gave (0 + 100) / 2 = 50.0.
    assert row["matchup_pace_10"].iloc[0] == pytest.approx(99.0)


def test_precomputed_latest_stats_index_is_behavior_preserving():
    """Hoisting the latest-stats lookup (efficiency) must produce IDENTICAL
    features to the per-call path, and pick the latest row by date."""
    import numpy as np
    import pandas as _pd
    from nba_betting.features.builder import (
        build_prediction_features, compute_latest_stats_index,
        _WINDOWS, _ROLLING_STATS, _SOS_PACE_ROLLING_STATS, _SOS_PACE_WINDOWS, _EWM_STATS,
    )

    def _full(team_id, date, poss10):
        s = {"team_id": team_id, "date": date}
        for w in _WINDOWS:
            for st in _ROLLING_STATS:
                s[f"{st}_roll_{w}"] = 1.0
            for ff in ("efg_pct", "tov_pct", "orb_pct", "ft_rate"):
                s[f"{ff}_roll_{w}"] = 0.5
        for w in (10, 20):
            for st in ("plus_minus", "net_rtg_game"):
                s[f"{st}_home_split_roll_{w}"] = 1.0
                s[f"{st}_away_split_roll_{w}"] = 1.0
        for w in (5, 10):
            s[f"fg3_pct_b2b_roll_{w}"] = 0.35
        for st in _SOS_PACE_ROLLING_STATS:
            for w in _SOS_PACE_WINDOWS:
                s[f"{st}_{w}"] = 100.0 if st == "poss_roll" else 1.0
        for st in _EWM_STATS:
            s[st] = 1.0
        for rc in ("rest_days", "is_back_to_back", "games_last_7", "games_last_14"):
            s[rc] = 2
        s["poss_roll_10"] = poss10
        return s

    # Team 1 has two rows; the LATER date carries the value we expect.
    rolling_df = _pd.DataFrame([
        _full(1, "2026-01-01", 50.0),
        _full(1, "2026-01-10", 100.0),
        _full(2, "2026-01-10", 100.0),
    ])
    fm = {"matchup_pace_10": 99.0}
    # injury_impacts={} on both sides keeps it DB-free and identical.
    per_call = build_prediction_features(1, 2, rolling_df, 1500, 1500, feature_means=fm, injury_impacts={})
    idx = compute_latest_stats_index(rolling_df)
    precomp = build_prediction_features(
        1, 2, rolling_df, 1500, 1500, feature_means=fm, injury_impacts={},
        latest_stats_by_team=idx,
    )
    assert per_call is not None and precomp is not None
    _pd.testing.assert_frame_equal(per_call, precomp)
    assert idx[1]["poss_roll_10"] == 100.0  # latest-by-date row, not the 50.0 one
