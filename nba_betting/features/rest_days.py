"""Rest days and schedule density features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest-related features per team-game.

    Features:
    - rest_days: days since last game (capped at 7; 3 for a team's first game)
    - is_back_to_back: 1 if rest_days <= 1
    - games_last_7: number of games in trailing 7-day window (inclusive)
    - games_last_14: number of games in trailing 14-day window (inclusive)

    Vectorized: rest/b2b come from a per-team date diff; the trailing-window
    counts use searchsorted on each team's sorted dates (O(n log n) per team).
    The previous per-game rescan of all past dates was O(n²) per team and
    dominated the whole feature build (~5.7s of ~6.3s on a 3-season history).
    tests/test_rest_features.py pins equivalence against the original loop.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team_id", "date", "game_id"])

    delta_days = df.groupby("team_id", sort=False)["date"].diff().dt.days
    rest = delta_days.clip(upper=7)
    df["rest_days"] = rest.fillna(3).astype(int)
    df["is_back_to_back"] = ((rest <= 1) & delta_days.notna()).astype(int)

    games_7 = np.empty(len(df), dtype=np.int64)
    games_14 = np.empty(len(df), dtype=np.int64)
    pos = 0
    # sort=False keeps groups in row order, so positional writes line up with df.
    for _, team_df in df.groupby("team_id", sort=False):
        dates = team_df["date"].to_numpy()
        n = len(dates)
        idx = np.arange(n)
        # First index whose date falls inside the trailing window; everything
        # from there up to (but excluding) the current game is in-window.
        games_7[pos:pos + n] = idx - np.searchsorted(
            dates, dates - np.timedelta64(7, "D"), side="left",
        )
        games_14[pos:pos + n] = idx - np.searchsorted(
            dates, dates - np.timedelta64(14, "D"), side="left",
        )
        pos += n

    df["games_last_7"] = games_7
    df["games_last_14"] = games_14

    return df
