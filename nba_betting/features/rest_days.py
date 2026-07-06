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


def rest_features_for_upcoming(team_dates, game_date) -> dict[str, int]:
    """Schedule-correct rest features for an UPCOMING game.

    ``add_rest_features`` computes each historical game's rest profile from
    the actual schedule — so a training row's ``rest_days`` describes the
    game being predicted. The live predict path used to reuse the team's
    *last completed game's* rest values instead (a one-game-stale profile:
    a team that played last night showed the rest it had *before* that
    game). Unlike the rolling-stat lag — evaluated and pinned as noise in
    ``predict_path_eval`` — rest for the upcoming game is deterministic
    and knowable exactly, so there is no de-noising argument for the lag.

    Mirrors ``add_rest_features`` semantics exactly (pinned by
    tests/test_rest_features.py::test_upcoming_matches_add_rest_features):
    rest capped at 7, default 3 with no history, b2b = rest <= 1, trailing
    windows inclusive of the exact 7/14-day boundary.

    Args:
        team_dates: The team's completed-game dates (any datetime-like);
            need not be sorted.
        game_date: The upcoming game's (ET) calendar date.

    Returns:
        ``{"rest_days", "is_back_to_back", "games_last_7", "games_last_14"}``.
    """
    import pandas as pd

    game_ts = pd.Timestamp(game_date).normalize()
    dates = pd.to_datetime(pd.Series(list(team_dates))).dt.normalize()
    dates = dates[dates < game_ts].sort_values().to_numpy()

    if len(dates) == 0:
        return {"rest_days": 3, "is_back_to_back": 0,
                "games_last_7": 0, "games_last_14": 0}

    delta = int((game_ts.to_datetime64() - dates[-1]) / np.timedelta64(1, "D"))
    rest = min(delta, 7)
    g7 = int(len(dates) - np.searchsorted(
        dates, game_ts.to_datetime64() - np.timedelta64(7, "D"), side="left"))
    g14 = int(len(dates) - np.searchsorted(
        dates, game_ts.to_datetime64() - np.timedelta64(14, "D"), side="left"))
    return {
        "rest_days": rest,
        "is_back_to_back": int(rest <= 1),
        "games_last_7": g7,
        "games_last_14": g14,
    }
