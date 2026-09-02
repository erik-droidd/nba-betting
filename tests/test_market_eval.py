"""market-eval harness: pure scoring helpers (2026-09)."""
from __future__ import annotations

import numpy as np
import pytest

from nba_betting.betting.market_eval import (
    cover_stats, paired_t, score_lambda_grid, shrink, _points_section,
)
from nba_betting.betting.shrinkage import shrink_to_market


def test_shrink_matches_scalar_shrinkage():
    model = np.array([0.7, 0.4, 0.55])
    market = np.array([0.6, 0.5, 0.5])
    out = shrink(model, market, 0.6)
    for i in range(3):
        assert out[i] == pytest.approx(shrink_to_market(model[i], market[i], 0.6), abs=1e-9)
    assert shrink(model, market, 0.0).tolist() == pytest.approx(model.tolist(), abs=1e-9)
    assert shrink(model, market, 1.0).tolist() == pytest.approx(market.tolist(), abs=1e-9)


def test_lambda_grid_prefers_the_better_source():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.2, 0.8, 2000)
    y = (rng.uniform(size=2000) < truth).astype(float)
    market = np.clip(truth + rng.normal(0, 0.02, 2000), 0.01, 0.99)   # sharp
    model = np.clip(truth + rng.normal(0, 0.12, 2000), 0.01, 0.99)    # noisy
    table = score_lambda_grid(model, market, y)
    assert set(table) == {round(0.1 * i, 1) for i in range(11)}
    assert min(table, key=table.get) >= 0.7            # leans on the sharp market
    table2 = score_lambda_grid(market, model, y)        # roles swapped
    assert min(table2, key=table2.get) <= 0.3


def test_cover_stats_sides_and_pushes():
    model = np.array([5.0, -3.0, 2.0, 4.0, 1.0])
    market = np.array([2.0, 0.0, 1.0, 2.0, 1.0])
    actual = np.array([7.0, 1.0, 5.0, 2.0, 9.0])
    # picks at 1.5: g0 (home, actual 7 > 2 -> hit), g1 (away, actual 1 > 0 -> miss),
    # g3 (home, push at 2 -> excluded); g2 and g4 below threshold.
    s = cover_stats(model, market, actual, 1.5)
    assert (s["picks"], s["hits"], s["pushes"]) == (2, 1, 1)
    assert s["hit_rate"] == pytest.approx(0.5)


def test_points_section_and_paired_t():
    sec = _points_section([10.0, 0.0], [8.0, 0.0], [9.0, 0.0], 1.5)
    assert sec["n"] == 2 and sec["model_mae"] == 0.5 and sec["market_mae"] == 0.5
    assert sec["avg_mae"] == pytest.approx(0.0)
    assert _points_section([], [], [], 1.5) == {"n": 0}
    assert paired_t([1, 1, 1], [1, 1, 1]) == 0.0
    assert paired_t([2, 3, 4, 3], [1, 1, 1, 1]) > 2     # b clearly lower loss
    assert paired_t([1.0], [0.0]) == 0.0                 # n < 2 guard
