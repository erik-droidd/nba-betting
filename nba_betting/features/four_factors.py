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
    """
    df = df.copy()

    # For each team-game row, find the opponent's DREB
    # The opponent is the other team in the same game
    opp_dreb = {}
    for game_id, group in df.groupby("game_id"):
        if len(group) != 2:
            continue
        rows = group.to_dict("records")
        for i in range(2):
            j = 1 - i
            key = (game_id, rows[i]["team_id"])
            opp_dreb[key] = rows[j]["dreb"]

    df["opp_dreb"] = df.apply(
        lambda r: opp_dreb.get((r["game_id"], r["team_id"]), 0), axis=1
    )

    # ORB% = OREB / (OREB + OPP_DREB)
    total_reb_chances = df["oreb"] + df["opp_dreb"]
    df["orb_pct"] = df["oreb"] / total_reb_chances.replace(0, 1)

    return df
