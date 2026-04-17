"""Rolling window statistics for team performance features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select

from nba_betting.config import INITIAL_ELO
from nba_betting.db.models import EloRating, Game, GameStats, Team
from nba_betting.db.session import get_session


def rolling_ewm(series: pd.Series, halflife: float = 10.0) -> pd.Series:
    """Exponential-decay rolling mean (Tier 1.5).

    Uses ``shift(1)`` to exclude the current observation (no-leakage), then
    an exponentially weighted mean with the given halflife. The default
    halflife of 10 games means a game from a month ago contributes ~25%
    of the weight of last night's game — useful for catching mid-season
    form shifts (trades, lineup changes) that uniform 5/10/20 windows are
    slow to register.

    Returned series is aligned to the input index. NaN where insufficient
    history.
    """
    return series.shift(1).ewm(halflife=halflife, min_periods=2).mean()


def _load_game_stats_df() -> pd.DataFrame:
    """Load all game stats joined with game info into a DataFrame."""
    session = get_session()
    try:
        rows = session.execute(
            select(
                Game.id.label("game_id"),
                Game.date,
                Game.season,
                Game.home_team_id,
                Game.away_team_id,
                Game.home_score,
                Game.away_score,
                Game.home_win,
                GameStats.team_id,
                GameStats.fgm,
                GameStats.fga,
                GameStats.fg_pct,
                GameStats.fg3m,
                GameStats.fg3a,
                GameStats.fg3_pct,
                GameStats.ftm,
                GameStats.fta,
                GameStats.ft_pct,
                GameStats.oreb,
                GameStats.dreb,
                GameStats.reb,
                GameStats.ast,
                GameStats.stl,
                GameStats.blk,
                GameStats.tov,
                GameStats.pts,
                GameStats.plus_minus,
            )
            .join(GameStats, Game.id == GameStats.game_id)
            .where(Game.home_score.isnot(None))
            .order_by(Game.date, Game.id)
        ).all()

        columns = [
            "game_id", "date", "season", "home_team_id", "away_team_id",
            "home_score", "away_score", "home_win", "team_id",
            "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
            "ftm", "fta", "ft_pct", "oreb", "dreb", "reb",
            "ast", "stl", "blk", "tov", "pts", "plus_minus",
        ]
        return pd.DataFrame(rows, columns=columns)
    finally:
        session.close()


def _load_pre_game_elo_df() -> pd.DataFrame:
    """Per (game_id, team_id) → opponent's pre-game Elo (Tier 1.1 SOS).

    Used to compute rolling mean opponent Elo per team, which then anchors
    SOS-adjusted net rating. Returns empty DataFrame if Elo hasn't been
    computed yet — the caller falls back to INITIAL_ELO.
    """
    session = get_session()
    try:
        rows = session.execute(
            select(EloRating.game_id, EloRating.team_id, EloRating.elo_before)
        ).all()
    finally:
        session.close()
    if not rows:
        return pd.DataFrame(columns=["game_id", "team_id", "elo_before"])
    return pd.DataFrame(rows, columns=["game_id", "team_id", "elo_before"])


def compute_rolling_features(windows: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    """Compute rolling averages of key stats per team.

    For each team-game, computes rolling mean over the last N games
    using .shift(1) to exclude the current game (prevents leakage).

    Returns a DataFrame indexed by (game_id, team_id) with rolling columns.
    """
    df = _load_game_stats_df()
    if df.empty:
        return pd.DataFrame()

    # Derive points-against per team-game from the home/away scores already
    # loaded with the row. This is the foundation for net rating and
    # Pythagorean expectation features. Vectorized with np.where — the
    # equivalent .apply(axis=1) is ~100x slower on the full game history.
    df["pts_against"] = np.where(
        df["team_id"] == df["home_team_id"],
        df["away_score"],
        df["home_score"],
    )
    df["pts_for"] = df["pts"]
    df["net_rtg_game"] = df["pts_for"] - df["pts_against"]

    # Tier 1.1 — attach opponent's pre-game Elo per row. We compute this
    # by self-joining the (game_id → team→elo_before) mapping. If Elo
    # hasn't been backfilled yet (cold start) we default to INITIAL_ELO so
    # downstream SOS features are neutral rather than NaN.
    elo_df = _load_pre_game_elo_df()
    if not elo_df.empty:
        # opponent_id per row
        df["opponent_id"] = np.where(
            df["team_id"] == df["home_team_id"],
            df["away_team_id"],
            df["home_team_id"],
        )
        df = df.merge(
            elo_df.rename(columns={"team_id": "opponent_id", "elo_before": "opp_elo"}),
            on=["game_id", "opponent_id"],
            how="left",
        )
        df["opp_elo"] = df["opp_elo"].fillna(INITIAL_ELO)
        df = df.drop(columns=["opponent_id"], errors="ignore")
    else:
        df["opp_elo"] = INITIAL_ELO

    # Tier 1.2 — pace per team-game (possessions). We need opponent's
    # rebounding context to estimate possessions accurately, but a clean
    # standalone approximation is fine for a rolling-window predictor:
    # poss ≈ FGA + 0.44*FTA + TOV - OREB. Average team plays ~100/game.
    df["poss"] = (
        df["fga"].astype(float)
        + 0.44 * df["fta"].astype(float)
        + df["tov"].astype(float)
        - df["oreb"].astype(float)
    ).clip(lower=60.0)  # floor sanity-checks against bad scrape data

    # Stats to compute rolling averages for. Tier 1.1 adds ``opp_elo`` so
    # each team gets a rolling mean strength-of-schedule signal, and Tier
    # 1.2 adds ``poss`` so the model sees recent possessions-per-game
    # (pace). Both are used downstream to build SOS-adjusted net rating
    # and a pace-differential feature.
    stat_cols = [
        "pts", "pts_against", "net_rtg_game",
        "fg_pct", "fg3_pct", "ft_pct",
        "oreb", "dreb", "reb", "ast", "stl", "blk", "tov",
        "plus_minus", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
        "opp_elo", "poss",
    ]

    # Sort by team and date for proper rolling
    df = df.sort_values(["team_id", "date", "game_id"])

    # Stats for home/away-specific rolling splits. Only the strongest
    # predictors — keeps feature bloat in check while capturing venue
    # effects missed by the all-games averages.
    _HA_SPLIT_STATS = ["plus_minus", "net_rtg_game"]
    _HA_SPLIT_WINDOWS = (10, 20)

    all_rolling = []

    for team_id, team_df in df.groupby("team_id"):
        team_df = team_df.copy()

        # --- All-game rolling (original) ---
        for window in windows:
            for col in stat_cols:
                # shift(1) ensures we only use past games
                rolled = (
                    team_df[col]
                    .shift(1)
                    .rolling(window=window, min_periods=max(1, window // 2))
                    .mean()
                )
                team_df[f"{col}_roll_{window}"] = rolled

        # --- Tier 1.1: SOS-adjusted net rating ---
        # The raw rolling net_rtg treats a +10 stretch vs tanking teams the
        # same as +10 vs contenders. We divide by a strength-of-schedule
        # multiplier derived from the team's recent mean opponent Elo:
        #   adj_net = net_rtg / (1 + (opp_elo - 1500) / 500)
        # A team whose recent opponents averaged Elo=1600 gets its net
        # rating *boosted* (denominator < 1), and vice versa for weak
        # opponents. The 500-point scale roughly spans the NBA's realistic
        # Elo range around the mean.
        for window in windows:
            net_col = f"net_rtg_game_roll_{window}"
            opp_col = f"opp_elo_roll_{window}"
            if net_col in team_df.columns and opp_col in team_df.columns:
                sos_denom = 1.0 + (team_df[opp_col] - 1500.0) / 500.0
                # Guard against zero/negative denominators at extreme
                # schedule edges (shouldn't happen in practice with NBA
                # Elo ranges, but defensive code wins).
                sos_denom = sos_denom.clip(lower=0.5, upper=1.5)
                team_df[f"adj_net_rtg_roll_{window}"] = team_df[net_col] / sos_denom

        # --- Tier 1.5: Exponentially-weighted "recent form" features ---
        # Halflife 10 games puts most weight on the last 2 weeks of play.
        # Complements the uniform 5/10/20-window rolls by reacting faster
        # to lineup changes and hot/cold streaks without throwing away
        # older context entirely.
        team_df["net_rtg_game_ewm_10"] = rolling_ewm(team_df["net_rtg_game"], halflife=10.0)
        team_df["plus_minus_ewm_10"] = rolling_ewm(team_df["plus_minus"], halflife=10.0)

        # --- Back-to-back 3PT cold streak (#8 improvement) ---
        # Teams shoot ~1.5% worse from 3 on the second night of a B2B.
        # This feature captures the average fg3_pct on recent B2B second-
        # nights. `is_back_to_back` isn't available yet (rest_days are
        # added later in builder.py), so we derive it from the date diff.
        dates = pd.to_datetime(team_df["date"])
        day_diff = dates.diff().dt.days
        b2b_mask = day_diff <= 1  # played yesterday = back-to-back
        fg3_b2b = team_df["fg3_pct"].where(b2b_mask, other=np.nan)
        for window in (5, 10):
            rolled = (
                fg3_b2b
                .shift(1)
                .rolling(window=window, min_periods=2)
                .mean()
            )
            team_df[f"fg3_pct_b2b_roll_{window}"] = rolled.ffill()

        # --- Home / away split rolling (#2 improvement) ---
        # A team's performance at home can differ substantially from on
        # the road. For each split stat, mask to home-only or away-only
        # games, shift(1) to prevent leakage, and rolling-mean. The
        # resulting columns are *conditional* averages: e.g.
        # `plus_minus_home_split_roll_10` = avg plus/minus in last 10
        # HOME games. We then fill forward so away game rows still carry
        # the latest home-split value — the model can compare "how does
        # this team perform at home vs how the opponent performs away".
        is_home = team_df["team_id"] == team_df["home_team_id"]
        is_away = ~is_home

        for window in _HA_SPLIT_WINDOWS:
            min_p = max(1, window // 3)  # relaxed min_periods for sparser splits
            for col in _HA_SPLIT_STATS:
                for tag, mask in (("home_split", is_home), ("away_split", is_away)):
                    split_vals = team_df[col].where(mask, other=np.nan)
                    rolled = (
                        split_vals
                        .shift(1)
                        .rolling(window=window, min_periods=min_p)
                        .mean()
                    )
                    # Forward-fill so every row carries the latest split
                    # value (an away game row still gets the most recent
                    # home-split average).
                    team_df[f"{col}_{tag}_roll_{window}"] = rolled.ffill()

        all_rolling.append(team_df)

    result = pd.concat(all_rolling, ignore_index=True)
    return result


def get_team_rolling_stats(team_id: int, df: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 20)) -> dict:
    """Get the most recent rolling stats for a specific team.

    Returns a flat dict of {feature_name: value}.
    """
    team_rows = df[df["team_id"] == team_id].sort_values("date")
    if team_rows.empty:
        return {}

    latest = team_rows.iloc[-1]
    stats = {}

    stat_cols = [
        "pts", "fg_pct", "fg3_pct", "ft_pct",
        "oreb", "dreb", "reb", "ast", "stl", "blk", "tov",
        "plus_minus", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    ]

    for window in windows:
        for col in stat_cols:
            key = f"{col}_roll_{window}"
            if key in latest.index:
                stats[key] = latest[key]

    return stats
