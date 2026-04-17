"""Assemble the full feature matrix for model training and inference."""
from __future__ import annotations

import pandas as pd
import numpy as np
from sqlalchemy import select

from nba_betting.db.models import Game, Team
from nba_betting.db.session import get_session
from nba_betting.features.rolling import compute_rolling_features
from nba_betting.features.four_factors import add_four_factors, add_opponent_rebound_data
from nba_betting.features.rest_days import add_rest_features
from nba_betting.models.elo import compute_all_elos, expected_score
from nba_betting.config import ELO_HOME_ADVANTAGE, INITIAL_ELO


# Features used from rolling stats (for each window size). Includes
# pts_against and net_rtg_game so that the model has direct access to
# defensive performance and net rating differential — the canonical
# strongest predictors in NBA modeling.
_ROLLING_STATS = [
    "pts", "pts_against", "net_rtg_game",
    "fg_pct", "fg3_pct", "oreb", "dreb", "ast", "tov", "plus_minus",
]

# Tier 1.1 + 1.2 — new rolling columns that deserve diff features in the
# model. ``adj_net_rtg_roll_{w}`` is the SOS-normalized net rating;
# ``poss_roll_{w}`` is the team's recent pace; ``opp_elo_roll_{w}`` is
# the recent strength of schedule itself. The pace-diff tells the model
# the game will be fast or slow; the SOS-adjusted net rating tells the
# model how good each team actually is adjusted for who they've played.
_SOS_PACE_ROLLING_STATS = ["adj_net_rtg_roll", "poss_roll", "opp_elo_roll"]
_SOS_PACE_WINDOWS = (5, 10, 20)

# Tier 1.5 — exponentially weighted recent-form features. Single halflife
# (no window suffix) because the EWM averages all of history with
# exponential decay; one column per stat is enough.
_EWM_STATS = ["net_rtg_game_ewm_10", "plus_minus_ewm_10"]

# Window sizes
_WINDOWS = (5, 10, 20)

# Pythagorean expectation exponent for basketball. Daryl Morey's empirical
# fit. The Pythagorean win % is one of the strongest single-feature
# predictors of team strength — better than raw W-L because it captures
# the magnitude of wins and losses, not just the count.
_PYTH_EXPONENT = 14.0


def _pythagorean_expectation(pts_for: float, pts_against: float) -> float:
    """Daryl Morey's basketball Pythagorean win expectation (scalar).

    pyth = PF^14 / (PF^14 + PA^14)

    Returns 0.5 on degenerate input. Clipped to [0.01, 0.99] for stability.
    Used by build_prediction_features() (single row at inference time).
    For batch computation, use _pythagorean_expectation_vec().
    """
    import math
    if pts_for is None or pts_against is None:
        return 0.5
    try:
        pf = float(pts_for)
        pa = float(pts_against)
    except (TypeError, ValueError):
        return 0.5
    # Explicit NaN check — `nan <= 0` is False, so without this NaN values
    # would skip every degenerate-handling branch and propagate through
    # log/max/min in undefined ways. 0.5 means "no information".
    if math.isnan(pf) or math.isnan(pa):
        return 0.5
    if pf <= 0 and pa <= 0:
        return 0.5
    if pa <= 0:
        return 0.99
    if pf <= 0:
        return 0.01
    # Use logs for numerical stability with the high exponent
    log_pf = _PYTH_EXPONENT * math.log(pf)
    log_pa = _PYTH_EXPONENT * math.log(pa)
    m = max(log_pf, log_pa)
    pyth = math.exp(log_pf - m) / (math.exp(log_pf - m) + math.exp(log_pa - m))
    return max(0.01, min(0.99, pyth))


def _pythagorean_expectation_vec(pts_for, pts_against):
    """Vectorized Pythagorean expectation over numpy/pandas arrays.

    Same math as the scalar version but operates on whole columns. The
    training matrix has ~6k rows so this gives a significant speedup vs
    `.apply(lambda r: ..., axis=1)` (which is row-iteration in Python).
    """
    pf = pd.to_numeric(pts_for, errors="coerce").astype(float)
    pa = pd.to_numeric(pts_against, errors="coerce").astype(float)

    # Default everything to 0.5; we will overwrite the valid rows.
    out = np.full(len(pf), 0.5, dtype=float)
    nan_mask = pf.isna() | pa.isna()
    valid = ~nan_mask
    pf_v = pf.where(valid, 1.0).to_numpy()
    pa_v = pa.where(valid, 1.0).to_numpy()

    # Numerically stable softmax-style computation in log space.
    # log(PF^k) = k*log(PF), then subtract the max for stability.
    safe_pf = np.where(pf_v > 0, pf_v, 1e-9)
    safe_pa = np.where(pa_v > 0, pa_v, 1e-9)
    log_pf = _PYTH_EXPONENT * np.log(safe_pf)
    log_pa = _PYTH_EXPONENT * np.log(safe_pa)
    m = np.maximum(log_pf, log_pa)
    num = np.exp(log_pf - m)
    den = num + np.exp(log_pa - m)
    pyth = num / den

    # Degenerate score handling matches the scalar version exactly.
    both_zero = (pf_v <= 0) & (pa_v <= 0)
    pa_zero = (pf_v > 0) & (pa_v <= 0)
    pf_zero = (pf_v <= 0) & (pa_v > 0)
    pyth = np.where(both_zero, 0.5, pyth)
    pyth = np.where(pa_zero, 0.99, pyth)
    pyth = np.where(pf_zero, 0.01, pyth)
    pyth = np.clip(pyth, 0.01, 0.99)

    out[valid.to_numpy()] = pyth[valid.to_numpy()]
    return out


def build_feature_matrix() -> tuple[pd.DataFrame, pd.Series]:
    """Build the complete feature matrix for all historical games.

    Returns:
        X: DataFrame of features (one row per game)
        y: Series of targets (1 = home win, 0 = away win)
    """
    # Step 1: Compute rolling stats for all team-game rows
    rolling_df = compute_rolling_features(_WINDOWS)
    if rolling_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Step 2: Add Four Factors
    rolling_df = add_four_factors(rolling_df)
    rolling_df = add_opponent_rebound_data(rolling_df)

    # Step 3: Add rest features
    rolling_df = add_rest_features(rolling_df)

    # Step 4: Compute rolling Four Factors (need to redo rolling after adding them)
    rolling_df = rolling_df.sort_values(["team_id", "date", "game_id"])
    four_factor_cols = ["efg_pct", "tov_pct", "orb_pct", "ft_rate"]
    for team_id, team_df in rolling_df.groupby("team_id"):
        idx = team_df.index
        for col in four_factor_cols:
            for window in _WINDOWS:
                roll_col = f"{col}_roll_{window}"
                rolling_df.loc[idx, roll_col] = (
                    team_df[col]
                    .shift(1)
                    .rolling(window=window, min_periods=max(1, window // 2))
                    .mean()
                    .values
                )

    # Step 5: Compute Elo ratings and get per-game snapshots. Tier 1.3 —
    # load the off/def Elo columns too so we can build split-Elo features
    # in addition to the aggregate.
    elo_ratings = compute_all_elos()

    session = get_session()
    try:
        from nba_betting.db.models import EloRating
        elo_rows = session.execute(
            select(
                EloRating.game_id,
                EloRating.team_id,
                EloRating.elo_before,
                EloRating.elo_off_before,
                EloRating.elo_def_before,
            )
        ).all()
        elo_df = pd.DataFrame(
            elo_rows,
            columns=["game_id", "team_id", "elo_before", "elo_off_before", "elo_def_before"],
        )
        # Back-fill rows from before the off/def migration with the
        # aggregate Elo so downstream arithmetic stays well-defined.
        if not elo_df.empty:
            elo_df["elo_off_before"] = elo_df["elo_off_before"].fillna(elo_df["elo_before"])
            elo_df["elo_def_before"] = elo_df["elo_def_before"].fillna(elo_df["elo_before"])
    finally:
        session.close()

    # Step 6: Pivot to one row per game (home vs away features). Carry
    # raw scores through so spreads/totals regressors can pull their
    # targets from the same pivoted frame.
    games = rolling_df[
        ["game_id", "date", "season", "home_team_id", "away_team_id",
         "home_win", "home_score", "away_score"]
    ].drop_duplicates("game_id")

    # Split into home and away team rows
    home_stats = rolling_df[rolling_df["team_id"] == rolling_df["home_team_id"]].copy()
    away_stats = rolling_df[rolling_df["team_id"] == rolling_df["away_team_id"]].copy()

    # Build feature columns for home and away
    feature_cols = []
    for window in _WINDOWS:
        for stat in _ROLLING_STATS:
            feature_cols.append(f"{stat}_roll_{window}")
        for ff in four_factor_cols:
            feature_cols.append(f"{ff}_roll_{window}")

    # Tier 1.1 + 1.2 — SOS-adjusted net rating, pace, opp-Elo rolling.
    # These are named `{stat}_roll_{w}` where {stat} already includes the
    # trailing `_roll` (e.g. `adj_net_rtg_roll`), so the column names in
    # rolling.py are `adj_net_rtg_roll_{w}` etc. We pass them through as
    # additional feature_cols — per-window diff features are built below.
    sos_pace_feature_cols = [
        f"{stat}_{w}"
        for stat in _SOS_PACE_ROLLING_STATS
        for w in _SOS_PACE_WINDOWS
    ]
    # Tier 1.5 — EWM columns (no window suffix; the halflife is baked in).
    ewm_feature_cols = list(_EWM_STATS)

    rest_cols = ["rest_days", "is_back_to_back", "games_last_7", "games_last_14"]

    # Home/away split columns — per-venue rolling averages of plus_minus
    # and net_rtg_game that capture a team's *home-specific* and *away-
    # specific* performance. These are only available for the larger
    # rolling windows (10, 20) to keep the feature set tight.
    _HA_SPLIT_STATS = ["plus_minus", "net_rtg_game"]
    _HA_SPLIT_WINDOWS = (10, 20)
    ha_split_cols = []
    for w in _HA_SPLIT_WINDOWS:
        for s in _HA_SPLIT_STATS:
            ha_split_cols.append(f"{s}_home_split_roll_{w}")
            ha_split_cols.append(f"{s}_away_split_roll_{w}")

    # Back-to-back 3PT cold streak columns
    b2b_cols = ["fg3_pct_b2b_roll_5", "fg3_pct_b2b_roll_10"]

    # Rename columns with home_/away_ prefix
    home_rename = {col: f"home_{col}" for col in feature_cols + rest_cols + ha_split_cols}
    away_rename = {col: f"away_{col}" for col in feature_cols + rest_cols + ha_split_cols}

    # Filter split columns to those that actually exist in the data (if
    # rolling hasn't produced them — e.g. in a minimal test DB — we skip
    # them gracefully rather than crashing).
    actual_ha = [c for c in ha_split_cols if c in home_stats.columns]
    actual_b2b = [c for c in b2b_cols if c in home_stats.columns]
    # Tier 1.1/1.2/1.5 — only carry through the new columns that the
    # rolling stage actually produced (cold-start DBs without Elo backfill
    # won't have the SOS columns, for instance).
    actual_sos_pace = [c for c in sos_pace_feature_cols if c in home_stats.columns]
    actual_ewm = [c for c in ewm_feature_cols if c in home_stats.columns]
    extra_rolling = actual_ha + actual_b2b + actual_sos_pace + actual_ewm

    # Extend rename maps for new columns
    for col in extra_rolling:
        home_rename[col] = f"home_{col}"
        away_rename[col] = f"away_{col}"

    home_features = home_stats[["game_id"] + feature_cols + rest_cols + extra_rolling].rename(columns=home_rename)
    away_features = away_stats[["game_id"] + feature_cols + rest_cols + extra_rolling].rename(columns=away_rename)

    # Merge on game_id
    game_features = games.merge(home_features, on="game_id", how="left")
    game_features = game_features.merge(away_features, on="game_id", how="left")

    # Add Elo features. Tier 1.3 — carry off/def Elo through the same
    # merge so the model sees offense and defense as separate signals,
    # not just an aggregate.
    if not elo_df.empty:
        home_elo = elo_df.rename(columns={
            "elo_before": "home_elo",
            "elo_off_before": "home_elo_off",
            "elo_def_before": "home_elo_def",
            "team_id": "h_tid",
        })
        away_elo = elo_df.rename(columns={
            "elo_before": "away_elo",
            "elo_off_before": "away_elo_off",
            "elo_def_before": "away_elo_def",
            "team_id": "a_tid",
        })

        game_features = game_features.merge(
            home_elo[["game_id", "h_tid", "home_elo", "home_elo_off", "home_elo_def"]],
            left_on=["game_id", "home_team_id"],
            right_on=["game_id", "h_tid"],
            how="left",
        ).drop(columns=["h_tid"], errors="ignore")

        game_features = game_features.merge(
            away_elo[["game_id", "a_tid", "away_elo", "away_elo_off", "away_elo_def"]],
            left_on=["game_id", "away_team_id"],
            right_on=["game_id", "a_tid"],
            how="left",
        ).drop(columns=["a_tid"], errors="ignore")
    else:
        game_features["home_elo"] = INITIAL_ELO
        game_features["away_elo"] = INITIAL_ELO
        game_features["home_elo_off"] = INITIAL_ELO
        game_features["home_elo_def"] = INITIAL_ELO
        game_features["away_elo_off"] = INITIAL_ELO
        game_features["away_elo_def"] = INITIAL_ELO

    for c in ("home_elo", "away_elo", "home_elo_off", "home_elo_def",
              "away_elo_off", "away_elo_def"):
        game_features[c] = game_features[c].fillna(INITIAL_ELO)

    # Step 7: Add differential features
    game_features["elo_diff"] = game_features["home_elo"] - game_features["away_elo"]
    # Tier 1.3 — split Elo differentials. The model already sees the
    # aggregate `elo_diff`; these let it learn that, e.g., a great offense
    # vs a bad defense matters more than the aggregate alone suggests.
    game_features["elo_off_diff"] = (
        game_features["home_elo_off"] - game_features["away_elo_off"]
    )
    game_features["elo_def_diff"] = (
        game_features["home_elo_def"] - game_features["away_elo_def"]
    )
    # Matchup asymmetries: home offense vs away defense, and vice versa.
    # These are the canonical "net advantage" splits in basketball
    # analytics — a good offense facing a bad defense is a stronger
    # signal than either rating alone.
    game_features["home_off_vs_away_def"] = (
        game_features["home_elo_off"] - game_features["away_elo_def"]
    )
    game_features["away_off_vs_home_def"] = (
        game_features["away_elo_off"] - game_features["home_elo_def"]
    )
    # Vectorized Elo expected-score formula. Equivalent to applying
    # expected_score row-by-row, but ~50x faster on the full matrix.
    _elo_diff_for_prob = (
        game_features["away_elo"] - (game_features["home_elo"] + ELO_HOME_ADVANTAGE)
    )
    game_features["elo_home_prob"] = 1.0 / (1.0 + np.power(10.0, _elo_diff_for_prob / 400.0))

    for window in _WINDOWS:
        for stat in _ROLLING_STATS:
            h = f"home_{stat}_roll_{window}"
            a = f"away_{stat}_roll_{window}"
            game_features[f"diff_{stat}_roll_{window}"] = (
                game_features[h] - game_features[a]
            )
        for ff in four_factor_cols:
            h = f"home_{ff}_roll_{window}"
            a = f"away_{ff}_roll_{window}"
            game_features[f"diff_{ff}_roll_{window}"] = (
                game_features[h] - game_features[a]
            )

    game_features["rest_diff"] = (
        game_features["home_rest_days"] - game_features["away_rest_days"]
    )

    # Home/away split differentials — compare the home team's *home*
    # performance against the away team's *away* performance. This is the
    # most natural matchup-specific comparison and what the model should
    # learn to exploit for venue-dependent teams.
    for w in _HA_SPLIT_WINDOWS:
        for s in _HA_SPLIT_STATS:
            h_col = f"home_{s}_home_split_roll_{w}"
            a_col = f"away_{s}_away_split_roll_{w}"
            diff_col = f"diff_{s}_venue_roll_{w}"
            if h_col in game_features.columns and a_col in game_features.columns:
                game_features[diff_col] = (
                    game_features[h_col] - game_features[a_col]
                )

    # Back-to-back 3PT cold differential: how much worse does each team
    # shoot on B2B nights? The diff captures relative fatigue effects.
    for w in (5, 10):
        h_col = f"home_fg3_pct_b2b_roll_{w}"
        a_col = f"away_fg3_pct_b2b_roll_{w}"
        diff_col = f"diff_fg3_pct_b2b_roll_{w}"
        if h_col in game_features.columns and a_col in game_features.columns:
            game_features[diff_col] = (
                game_features[h_col] - game_features[a_col]
            )

    # Tier 1.1 + 1.2 — SOS-adjusted net rating diff, pace diff (= expected
    # pace of the matchup), and SOS (opp-Elo) diff as a standalone
    # "strength of recent schedule" signal the model can use to discount
    # rolling stats.
    for stat in _SOS_PACE_ROLLING_STATS:
        for w in _SOS_PACE_WINDOWS:
            h_col = f"home_{stat}_{w}"
            a_col = f"away_{stat}_{w}"
            diff_col = f"diff_{stat}_{w}"
            if h_col in game_features.columns and a_col in game_features.columns:
                if stat == "poss_roll":
                    # Pace-diff is traditionally a *sum* (expected game
                    # pace is the average of both teams' possessions),
                    # but the differential matters too — a fast team
                    # facing a slow team creates stylistic mismatch.
                    # Keep the diff form for consistency with other
                    # features; the model can sum them internally.
                    game_features[diff_col] = game_features[h_col] - game_features[a_col]
                    # Also emit expected matchup pace (mean of the two).
                    game_features[f"matchup_pace_{w}"] = (
                        (game_features[h_col] + game_features[a_col]) / 2.0
                    )
                else:
                    game_features[diff_col] = game_features[h_col] - game_features[a_col]

    # Tier 1.5 — EWM diffs. One column per stat (no window suffix).
    for stat in _EWM_STATS:
        h_col = f"home_{stat}"
        a_col = f"away_{stat}"
        diff_col = f"diff_{stat}"
        if h_col in game_features.columns and a_col in game_features.columns:
            game_features[diff_col] = game_features[h_col] - game_features[a_col]

    # Pythagorean expectation per team (window 20 = ~quarter of season).
    # The diff feature gives the model a direct measure of which team has
    # been "actually winning at the level their scoring suggests".
    for window in _WINDOWS:
        h_pf_col = f"home_pts_roll_{window}"
        h_pa_col = f"home_pts_against_roll_{window}"
        a_pf_col = f"away_pts_roll_{window}"
        a_pa_col = f"away_pts_against_roll_{window}"
        if h_pf_col in game_features.columns and h_pa_col in game_features.columns:
            game_features[f"home_pyth_roll_{window}"] = _pythagorean_expectation_vec(
                game_features[h_pf_col], game_features[h_pa_col]
            )
            game_features[f"away_pyth_roll_{window}"] = _pythagorean_expectation_vec(
                game_features[a_pf_col], game_features[a_pa_col]
            )
            game_features[f"diff_pyth_roll_{window}"] = (
                game_features[f"home_pyth_roll_{window}"]
                - game_features[f"away_pyth_roll_{window}"]
            )

    # Step 7b: Injury features from the historical_injuries snapshot
    # table. For games predating the daily snapshot collector these will
    # all be 0 — the model learns injuries are "unknown, treat as
    # average" for those rows and starts using the feature as soon as
    # coverage kicks in. See `nba_betting/data/injuries.py` for schema.
    _attach_injury_features(game_features)

    # Step 7c: Line-movement features from the odds_snapshots table.
    # Same forward-accumulating convention as injuries: zero for games
    # before we started snapshotting odds, non-zero once the cron has
    # accumulated at least two snapshots per game. The model will learn
    # to ignore the zero rows (they carry no signal on their own) and
    # weight the non-zero rows accordingly.
    _attach_line_movement_features(game_features)

    # Step 8: Select final feature columns
    model_features = (
        ["home_elo", "away_elo", "elo_diff", "elo_home_prob",
         # Tier 1.3 — split Elo features and matchup asymmetries.
         "home_elo_off", "away_elo_off", "home_elo_def", "away_elo_def",
         "elo_off_diff", "elo_def_diff",
         "home_off_vs_away_def", "away_off_vs_home_def"]
        + [f"home_{col}" for col in rest_cols]
        + [f"away_{col}" for col in rest_cols]
        + ["rest_diff",
           "home_injury_impact_out", "away_injury_impact_out",
           "injury_impact_diff",
           # Line-movement features (zero for pre-snapshot games).
           "spread_movement", "prob_movement", "odds_disagreement"]
    )
    for window in _WINDOWS:
        for stat in _ROLLING_STATS:
            model_features.append(f"diff_{stat}_roll_{window}")
        for ff in four_factor_cols:
            model_features.append(f"diff_{ff}_roll_{window}")
        # Pythagorean differential for this window
        if f"diff_pyth_roll_{window}" in game_features.columns:
            model_features.append(f"diff_pyth_roll_{window}")

    # Home/away split differentials (only for windows that exist)
    for w in _HA_SPLIT_WINDOWS:
        for s in _HA_SPLIT_STATS:
            diff_col = f"diff_{s}_venue_roll_{w}"
            if diff_col in game_features.columns:
                model_features.append(diff_col)

    # Back-to-back 3PT cold differentials
    for w in (5, 10):
        diff_col = f"diff_fg3_pct_b2b_roll_{w}"
        if diff_col in game_features.columns:
            model_features.append(diff_col)

    # Tier 1.1 + 1.2 — SOS-adjusted net rating, pace-diff, opp-Elo roll.
    for stat in _SOS_PACE_ROLLING_STATS:
        for w in _SOS_PACE_WINDOWS:
            diff_col = f"diff_{stat}_{w}"
            if diff_col in game_features.columns:
                model_features.append(diff_col)
            if stat == "poss_roll":
                mp_col = f"matchup_pace_{w}"
                if mp_col in game_features.columns:
                    model_features.append(mp_col)

    # Tier 1.5 — EWM recent-form diffs.
    for stat in _EWM_STATS:
        diff_col = f"diff_{stat}"
        if diff_col in game_features.columns:
            model_features.append(diff_col)

    # Drop rows with too many NaNs (early-season games without enough history)
    game_features = game_features.sort_values("date")
    X = game_features[model_features].copy()
    y = game_features["home_win"].astype(float).copy()
    game_ids = game_features["game_id"].copy()
    dates = game_features["date"].copy()
    home_team_ids = game_features["home_team_id"].copy()
    away_team_ids = game_features["away_team_id"].copy()
    home_scores = game_features["home_score"].copy()
    away_scores = game_features["away_score"].copy()

    # Drop rows where more than 30% of features are NaN
    valid_mask = X.notna().mean(axis=1) >= 0.7
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    game_ids = game_ids[valid_mask].reset_index(drop=True)
    dates = dates[valid_mask].reset_index(drop=True)
    home_team_ids = home_team_ids[valid_mask].reset_index(drop=True)
    away_team_ids = away_team_ids[valid_mask].reset_index(drop=True)
    home_scores = home_scores[valid_mask].reset_index(drop=True)
    away_scores = away_scores[valid_mask].reset_index(drop=True)

    # Compute feature means BEFORE filling NaN (for use in prediction imputation)
    feature_means = X.mean().to_dict()

    # Fill remaining NaN with feature means (consistent with prediction imputation)
    X = X.fillna(X.mean()).fillna(0)

    # Attach metadata for walk-forward splitting and for joining historical
    # odds snapshots back to games during real-odds backtests.
    X["_game_id"] = game_ids.values
    X["_date"] = dates.values
    X["_home_team_id"] = home_team_ids.values
    X["_away_team_id"] = away_team_ids.values
    # Raw scores for spreads/totals regression. Kept in metadata columns
    # (underscore-prefixed) so they never leak into the classification
    # model's feature set.
    X["_home_score"] = home_scores.values
    X["_away_score"] = away_scores.values

    # Attach feature means as attribute for saving with the model
    X.attrs["feature_means"] = feature_means

    return X, y


def _attach_injury_features(game_features: pd.DataFrame) -> None:
    """Add `home_injury_impact_out`, `away_injury_impact_out`, and their
    differential to the per-game DataFrame (in-place).

    Reads from the `historical_injuries` table keyed on (snapshot_date,
    team_abbr). Games from before the snapshot collector started running
    will get 0 for both sides — which is fine because we still have the
    Elo + rolling stats, and the model will simply learn "when the
    injury feature is 0 treat it as average signal."

    This function is best-effort: if the table is empty or a lookup
    fails, the features default to 0 rather than raising.
    """
    try:
        from sqlalchemy import select

        from nba_betting.db.models import HistoricalInjury, Team
        from nba_betting.db.session import get_session
        from nba_betting.data.injuries import _status_multiplier

        session = get_session()
        try:
            injuries = session.execute(select(HistoricalInjury)).scalars().all()
            team_abbr_by_id = {
                t.id: t.abbreviation
                for t in session.execute(select(Team)).scalars().all()
            }
        finally:
            session.close()
    except Exception:
        game_features["home_injury_impact_out"] = 0.0
        game_features["away_injury_impact_out"] = 0.0
        game_features["injury_impact_diff"] = 0.0
        return

    # Build a (date, team_abbr) -> total weighted impact lookup.
    lookup: dict[tuple, float] = {}
    for inj in injuries:
        if inj.snapshot_date is None or not inj.team_abbr:
            continue
        key = (inj.snapshot_date, inj.team_abbr.upper())
        lookup[key] = lookup.get(key, 0.0) + (
            (inj.impact_rating or 0.0) * _status_multiplier(inj.status or "")
        )

    # Tier 3.2 — vectorize the injury-impact lookup. Previously this was
    # two Python list-comprehensions (~5k iterations each), now it's a
    # single DataFrame merge keyed on (game_date, team_abbr). Big-O
    # unchanged, constant factor ~30-50x on typical training sets.
    if not lookup:
        game_features["home_injury_impact_out"] = 0.0
        game_features["away_injury_impact_out"] = 0.0
    else:
        # Build a flat (team_abbr, snapshot_date) -> impact DataFrame.
        lookup_df = pd.DataFrame(
            [(d, abbr, impact) for (d, abbr), impact in lookup.items()],
            columns=["_inj_date", "_inj_abbr", "_inj_impact"],
        )

        # Normalize the game date to plain `date` to match lookup key type.
        gf_dates = pd.to_datetime(game_features["date"]).dt.date

        # Home team merge
        home_abbrs = game_features["home_team_id"].map(team_abbr_by_id)
        home_join = pd.DataFrame({
            "_inj_date": gf_dates.values,
            "_inj_abbr": home_abbrs.fillna("").str.upper().values,
        })
        home_merged = home_join.merge(
            lookup_df.assign(_inj_abbr=lookup_df["_inj_abbr"].str.upper()),
            on=["_inj_date", "_inj_abbr"], how="left",
        )
        game_features["home_injury_impact_out"] = (
            home_merged["_inj_impact"].fillna(0.0).values
        )

        # Away team merge — same pattern
        away_abbrs = game_features["away_team_id"].map(team_abbr_by_id)
        away_join = pd.DataFrame({
            "_inj_date": gf_dates.values,
            "_inj_abbr": away_abbrs.fillna("").str.upper().values,
        })
        away_merged = away_join.merge(
            lookup_df.assign(_inj_abbr=lookup_df["_inj_abbr"].str.upper()),
            on=["_inj_date", "_inj_abbr"], how="left",
        )
        game_features["away_injury_impact_out"] = (
            away_merged["_inj_impact"].fillna(0.0).values
        )

    game_features["injury_impact_diff"] = (
        game_features["home_injury_impact_out"]
        - game_features["away_injury_impact_out"]
    )


def _attach_line_movement_features(game_features: pd.DataFrame) -> None:
    """Add `spread_movement`, `prob_movement`, `odds_disagreement` columns
    to each game row (in-place) by joining against `odds_snapshots`.

    Games without snapshots (the entire pre-snapshot-era backfill) get
    0.0 for all three. Best-effort: any exception falls through to zeros
    rather than breaking the whole feature build.
    """
    try:
        from nba_betting.data.odds_tracker import batch_line_movements_by_game

        movements = batch_line_movements_by_game()
    except Exception:
        game_features["spread_movement"] = 0.0
        game_features["prob_movement"] = 0.0
        game_features["odds_disagreement"] = 0.0
        return

    if not movements:
        game_features["spread_movement"] = 0.0
        game_features["prob_movement"] = 0.0
        game_features["odds_disagreement"] = 0.0
        return

    def _lookup(d, h, a) -> dict:
        # OddsSnapshot.game_date is a `date`; Game.date can be a Timestamp.
        # Normalize to `date` so the tuple key matches.
        gd = d.date() if hasattr(d, "date") else d
        return movements.get((gd, int(h), int(a))) or {}

    dates = pd.to_datetime(game_features["date"]).tolist()
    homes = game_features["home_team_id"].tolist()
    aways = game_features["away_team_id"].tolist()

    spread_mv, prob_mv, disagree = [], [], []
    for d, h, a in zip(dates, homes, aways):
        m = _lookup(d, h, a)
        spread_mv.append(float(m.get("spread_movement", 0.0)))
        prob_mv.append(float(m.get("prob_movement", 0.0)))
        disagree.append(float(m.get("odds_disagreement", 0.0)))

    game_features["spread_movement"] = spread_mv
    game_features["prob_movement"] = prob_mv
    game_features["odds_disagreement"] = disagree


def build_prediction_features(
    home_team_id: int,
    away_team_id: int,
    rolling_df: pd.DataFrame,
    home_elo: float,
    away_elo: float,
    feature_means: dict[str, float] | None = None,
    extra_features: dict[str, float] | None = None,
    home_elo_off: float | None = None,
    home_elo_def: float | None = None,
    away_elo_off: float | None = None,
    away_elo_def: float | None = None,
) -> pd.DataFrame | None:
    """Build a single-row feature vector for a prediction.

    Args:
        home_team_id: NBA.com team ID for home team
        away_team_id: NBA.com team ID for away team
        rolling_df: Pre-computed rolling features DataFrame
        home_elo: Current Elo rating for home team
        away_elo: Current Elo rating for away team
        feature_means: Training-set feature means for NaN imputation.
        extra_features: Additional prediction-time features (player impact,
            line movement, etc.) to merge into the feature row.

    Returns:
        Single-row DataFrame matching the training feature columns,
        or None if too many features are missing (>30% NaN).
    """
    four_factor_cols = ["efg_pct", "tov_pct", "orb_pct", "ft_rate"]
    rest_cols_list = ["rest_days", "is_back_to_back", "games_last_7", "games_last_14"]

    def _get_latest_stats(team_id: int) -> dict:
        team_rows = rolling_df[rolling_df["team_id"] == team_id].sort_values("date")
        if team_rows.empty:
            return {}
        return team_rows.iloc[-1].to_dict()

    home_stats = _get_latest_stats(home_team_id)
    away_stats = _get_latest_stats(away_team_id)

    # Off/def Elo default to the aggregate Elo if caller didn't supply
    # them (keeps backwards compatibility with older callers that only
    # have a single Elo per team).
    h_off = home_elo_off if home_elo_off is not None else home_elo
    h_def = home_elo_def if home_elo_def is not None else home_elo
    a_off = away_elo_off if away_elo_off is not None else away_elo
    a_def = away_elo_def if away_elo_def is not None else away_elo

    row = {
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "elo_home_prob": expected_score(home_elo + ELO_HOME_ADVANTAGE, away_elo),
        # Tier 1.3 — split Elo and matchup asymmetries.
        "home_elo_off": h_off,
        "away_elo_off": a_off,
        "home_elo_def": h_def,
        "away_elo_def": a_def,
        "elo_off_diff": h_off - a_off,
        "elo_def_diff": h_def - a_def,
        "home_off_vs_away_def": h_off - a_def,
        "away_off_vs_home_def": a_off - h_def,
    }

    # Rest features
    for col in rest_cols_list:
        row[f"home_{col}"] = home_stats.get(col, 3 if col == "rest_days" else 0)
        row[f"away_{col}"] = away_stats.get(col, 3 if col == "rest_days" else 0)
    row["rest_diff"] = row["home_rest_days"] - row["away_rest_days"]

    # Rolling stat differentials
    for window in _WINDOWS:
        for stat in _ROLLING_STATS:
            h_val = home_stats.get(f"{stat}_roll_{window}", 0)
            a_val = away_stats.get(f"{stat}_roll_{window}", 0)
            row[f"diff_{stat}_roll_{window}"] = (h_val or 0) - (a_val or 0)
        for ff in four_factor_cols:
            h_val = home_stats.get(f"{ff}_roll_{window}", 0)
            a_val = away_stats.get(f"{ff}_roll_{window}", 0)
            row[f"diff_{ff}_roll_{window}"] = (h_val or 0) - (a_val or 0)
        # Pythagorean expectation differential — must mirror the training matrix
        h_pf = home_stats.get(f"pts_roll_{window}")
        h_pa = home_stats.get(f"pts_against_roll_{window}")
        a_pf = away_stats.get(f"pts_roll_{window}")
        a_pa = away_stats.get(f"pts_against_roll_{window}")
        row[f"diff_pyth_roll_{window}"] = (
            _pythagorean_expectation(h_pf, h_pa)
            - _pythagorean_expectation(a_pf, a_pa)
        )

    # Home/away split differentials — compare home team's home performance
    # vs away team's away performance, mirroring the training matrix.
    _HA_SPLIT_STATS_P = ["plus_minus", "net_rtg_game"]
    _HA_SPLIT_WINDOWS_P = (10, 20)
    for w in _HA_SPLIT_WINDOWS_P:
        for s in _HA_SPLIT_STATS_P:
            h_val = home_stats.get(f"{s}_home_split_roll_{w}")
            a_val = away_stats.get(f"{s}_away_split_roll_{w}")
            row[f"diff_{s}_venue_roll_{w}"] = (
                (h_val or 0) - (a_val or 0)
            )

    # Back-to-back 3PT cold differentials
    for w in (5, 10):
        h_val = home_stats.get(f"fg3_pct_b2b_roll_{w}")
        a_val = away_stats.get(f"fg3_pct_b2b_roll_{w}")
        row[f"diff_fg3_pct_b2b_roll_{w}"] = (h_val or 0) - (a_val or 0)

    # Tier 1.1 + 1.2 — SOS-adjusted net rating, pace, opp-Elo diffs.
    for stat in _SOS_PACE_ROLLING_STATS:
        for w in _SOS_PACE_WINDOWS:
            h_val = home_stats.get(f"{stat}_{w}")
            a_val = away_stats.get(f"{stat}_{w}")
            row[f"diff_{stat}_{w}"] = (h_val or 0) - (a_val or 0)
            if stat == "poss_roll":
                row[f"matchup_pace_{w}"] = ((h_val or 0) + (a_val or 0)) / 2.0

    # Tier 1.5 — EWM recent-form diffs.
    for stat in _EWM_STATS:
        h_val = home_stats.get(stat)
        a_val = away_stats.get(stat)
        row[f"diff_{stat}"] = (h_val or 0) - (a_val or 0)

    # Injury features — mirror training. We use the current injuries
    # (today's live list) for inference, which matches how
    # `persist_historical_injuries` stores them for backtest. If the
    # lookup fails we fall back to 0 (treated as "unknown, average").
    try:
        from nba_betting.data.injuries import load_injuries, _status_multiplier
        from nba_betting.db.models import Team
        from nba_betting.db.session import get_session
        from sqlalchemy import select

        cur_injuries = load_injuries()
        session = get_session()
        try:
            abbr_by_id = {
                t.id: t.abbreviation
                for t in session.execute(select(Team)).scalars().all()
            }
        finally:
            session.close()

        def _impact_for(team_id: int) -> float:
            abbr = abbr_by_id.get(team_id, "")
            total = 0.0
            for inj in cur_injuries:
                if inj.team_abbr.upper() == abbr.upper():
                    total += (inj.impact_rating or 0.0) * _status_multiplier(inj.status or "")
            return total

        row["home_injury_impact_out"] = _impact_for(home_team_id)
        row["away_injury_impact_out"] = _impact_for(away_team_id)
    except Exception:
        row["home_injury_impact_out"] = 0.0
        row["away_injury_impact_out"] = 0.0
    row["injury_impact_diff"] = (
        row["home_injury_impact_out"] - row["away_injury_impact_out"]
    )

    # Merge extra prediction-time features (player impact, line movement, etc.)
    if extra_features:
        row.update(extra_features)

    df = pd.DataFrame([row])

    # Check NaN percentage — if >30%, signal caller to fall back to Elo-only
    nan_pct = df.isna().mean(axis=1).iloc[0]
    if nan_pct > 0.3:
        return None

    # Impute NaN with training-set feature means (consistent with training)
    if feature_means:
        df = df.fillna(feature_means)
    df = df.fillna(0)

    return df
