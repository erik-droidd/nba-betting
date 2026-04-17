"""Oliver's Four Factors feature computation."""
from __future__ import annotations

import pandas as pd


def add_four_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add Oliver's Four Factors columns to a game stats DataFrame.

    The Four Factors are:
    1. eFG% = (FGM + 0.5 * FG3M) / FGA  (shooting efficiency)
    2. TOV% = TOV / (FGA + 0.44 * FTA + TOV)  (turnover rate)
    3. ORB% = OREB / (OREB + OPP_DREB)  (offensive rebounding)
    4. FT_RATE = FTM / FGA  (free throw rate)
    """
    df = df.copy()

    # eFG%
    df["efg_pct"] = (df["fgm"] + 0.5 * df["fg3m"]) / df["fga"].replace(0, 1)

    # Turnover rate
    possessions_approx = df["fga"] + 0.44 * df["fta"] + df["tov"]
    df["tov_pct"] = df["tov"] / possessions_approx.replace(0, 1)

    # Free throw rate
    df["ft_rate"] = df["ftm"] / df["fga"].replace(0, 1)

    return df


def add_opponent_rebound_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add opponent DREB to compute ORB%.

    Requires DataFrame to have game_id, team_id, home_team_id, away_team_id, dreb.

    Tier 3.2 — vectorized. The opponent of each team-game is simply the
    *other* row within the same game_id group, so we can derive opp_dreb
    by grouping on game_id and subtracting this row's dreb from the
    two-row sum. That's a pure pandas broadcast; the old groupby+.apply
    was O(N_games * 2) Python calls, this is a single vectorized op.
    """
    df = df.copy()

    # Sum dreb within each game (one number per game_id, broadcast to rows).
    # In a well-formed game stats table each game_id has exactly two rows —
    # the opponent's dreb = game_total_dreb - this_row_dreb.
    df["_game_dreb_total"] = df.groupby("game_id")["dreb"].transform("sum")
    df["opp_dreb"] = df["_game_dreb_total"] - df["dreb"]
    df = df.drop(columns=["_game_dreb_total"])

    # Safety fallback: rows where a game unexpectedly has a single team
    # (malformed data) would get opp_dreb == 0 from the math above, which
    # is the same behavior as the previous .apply fallback — so no
    # explicit handling is needed.

    # ORB% = OREB / (OREB + OPP_DREB)
    total_reb_chances = df["oreb"] + df["opp_dreb"]
    df["orb_pct"] = df["oreb"] / total_reb_chances.replace(0, 1)

    return df
