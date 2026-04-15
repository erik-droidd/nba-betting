"""Rest days and schedule density features."""
from __future__ import annotations

import pandas as pd


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest-related features per team-game.

    Features:
    - rest_days: days since last game (capped at 7)
    - is_back_to_back: 1 if rest_days == 1
    - games_last_7: number of games in trailing 7-day window
    - games_last_14: number of games in trailing 14-day window
    """
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
                rest_days_list.append(3)  # Default for first game
                b2b_list.append(0)
            else:
                delta = (dates[i] - dates[i - 1]) / pd.Timedelta(days=1)
                rest = min(int(delta), 7)
                rest_days_list.append(rest)
                b2b_list.append(1 if rest <= 1 else 0)

            # Schedule density: games in trailing windows
            current_date = dates[i]
            past_dates = dates[:i]  # Exclude current game
            g7 = sum(1 for d in past_dates if (current_date - d) / pd.Timedelta(days=1) <= 7)
            g14 = sum(1 for d in past_dates if (current_date - d) / pd.Timedelta(days=1) <= 14)
            games_7_list.append(g7)
            games_14_list.append(g14)

    df["rest_days"] = rest_days_list
    df["is_back_to_back"] = b2b_list
    df["games_last_7"] = games_7_list
    df["games_last_14"] = games_14_list

    return df
