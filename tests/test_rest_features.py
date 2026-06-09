"""Equivalence tests for the vectorized add_rest_features.

The original implementation looped over every (team, game) pair and rescanned
all past dates per game — O(n²) per team and ~5.7s on the full 3-season
history. The vectorized version must produce byte-identical columns. A frozen
copy of the original loop lives here as the reference oracle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nba_betting.features.rest_days import add_rest_features


def _reference_add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Frozen copy of the original O(n²) loop implementation."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team_id", "date", "game_id"])

    rest_days_list = []
    b2b_list = []
    games_7_list = []
    games_14_list = []

    for team_id, team_df in df.groupby("team_id"):
        dates = team_df["date"].values
        for i in range(len(dates)):
            if i == 0:
                rest_days_list.append(3)
                b2b_list.append(0)
            else:
                delta = (dates[i] - dates[i - 1]) / pd.Timedelta(days=1)
                rest = min(int(delta), 7)
                rest_days_list.append(rest)
                b2b_list.append(1 if rest <= 1 else 0)

            current_date = dates[i]
            past_dates = dates[:i]
            g7 = sum(1 for d in past_dates if (current_date - d) / pd.Timedelta(days=1) <= 7)
            g14 = sum(1 for d in past_dates if (current_date - d) / pd.Timedelta(days=1) <= 14)
            games_7_list.append(g7)
            games_14_list.append(g14)

    df["rest_days"] = rest_days_list
    df["is_back_to_back"] = b2b_list
    df["games_last_7"] = games_7_list
    df["games_last_14"] = games_14_list
    return df


_REST_COLS = ["rest_days", "is_back_to_back", "games_last_7", "games_last_14"]


def _schedule_df(rows: list[tuple[int, str]]) -> pd.DataFrame:
    """Build a minimal (team_id, date, game_id) frame from (team, date) tuples."""
    return pd.DataFrame({
        "team_id": [t for t, _ in rows],
        "date": [d for _, d in rows],
        "game_id": list(range(1, len(rows) + 1)),
    })


def _assert_matches_reference(df: pd.DataFrame) -> None:
    got = add_rest_features(df)
    want = _reference_add_rest_features(df)
    for col in _REST_COLS:
        np.testing.assert_array_equal(
            got[col].to_numpy(), want[col].to_numpy(), err_msg=col,
        )


def test_matches_reference_on_random_schedule():
    """Dense random multi-team schedule with b2bs and long gaps."""
    rng = np.random.default_rng(7)
    rows = []
    for team in (1610612737, 1610612738, 1610612739):
        day = pd.Timestamp("2025-10-20")
        for _ in range(60):
            day = day + pd.Timedelta(days=int(rng.integers(1, 6)))
            rows.append((team, str(day.date())))
    _assert_matches_reference(_schedule_df(rows))


def test_window_boundaries_exactly_7_and_14_days():
    """Games exactly 7/14 days back are INCLUDED in the trailing counts."""
    df = _schedule_df([
        (1, "2025-01-01"),
        (1, "2025-01-08"),   # exactly 7 days after game 1
        (1, "2025-01-15"),   # exactly 14 days after game 1, 7 after game 2
        (1, "2025-01-23"),   # 8 days after game 3 — outside the 7-day window
    ])
    got = add_rest_features(df)
    assert got["games_last_7"].tolist() == [0, 1, 1, 0]
    assert got["games_last_14"].tolist() == [0, 1, 2, 1]
    _assert_matches_reference(df)


def test_first_game_defaults_and_b2b():
    """First game of a team: rest_days=3, no b2b. Next-day game: b2b=1.
    Rest is capped at 7."""
    df = _schedule_df([
        (1, "2025-01-01"),
        (1, "2025-01-02"),   # back-to-back
        (1, "2025-01-20"),   # 18-day gap, capped at 7
        (2, "2025-01-05"),   # other team's first game
    ])
    got = add_rest_features(df)
    team1 = got[got["team_id"] == 1]
    assert team1["rest_days"].tolist() == [3, 1, 7]
    assert team1["is_back_to_back"].tolist() == [1 if r == 1 else 0 for r in [3, 1, 7]]
    team2 = got[got["team_id"] == 2]
    assert team2["rest_days"].tolist() == [3]
    assert team2["is_back_to_back"].tolist() == [0]
    _assert_matches_reference(df)


def test_unsorted_input_and_extra_columns_preserved():
    """Input order must not matter, and non-schedule columns pass through."""
    df = _schedule_df([
        (2, "2025-02-03"),
        (1, "2025-02-01"),
        (1, "2025-02-02"),
        (2, "2025-02-01"),
    ])
    df["pts"] = [100.0, 110.0, 120.0, 130.0]
    got = add_rest_features(df)
    assert "pts" in got.columns
    assert len(got) == 4
    _assert_matches_reference(df)
