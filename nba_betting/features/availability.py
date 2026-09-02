"""Player-availability features from the player game logs (2026-09).

Until this module existed, the six player-availability features
(``home/away_missing_minutes_pct``, ``home/away_star_out``,
``diff_missing_minutes_pct``, ``diff_available_talent``) were **constant
0.0 for every training row** — the model had never seen them carry
information, so the live values injected at predict time could not
matter. ``player_game_stats`` (one row per player per game appeared in)
lets us reconstruct, for every historical team-game, which rotation
regulars did NOT play — the best available proxy for "who was out".

Definitions (identical for training and the live path):

* **Typical minutes / talent** of a player for a team: mean over the
  player's last ``ROLL_GAMES`` appearances for that team *within the
  season*, using only games strictly before the target (shift(1) — no
  leakage). Talent = pts + ast + reb.
* **Regular**: appeared in at least 60% of the team's last
  ``WINDOW_GAMES`` games this season (at least one), with typical
  minutes >= ``REGULAR_MIN_MINUTES``. Season-scoped so off-season
  departures don't read as absences in October.
* **Departed**: a regular who has since appeared for a different team
  (mid-season trade) is not counted as absent.
* **missing_minutes_pct** = Σ typical minutes of absent regulars / Σ
  typical minutes of all regulars (0 when the team has no regulars yet,
  e.g. the first game of a season — "unknown, average", like before).
* **star_out** = any absent regular with typical minutes >= ``STAR_MINUTES``.
* **available_talent** = Σ typical talent of PRESENT regulars / Σ typical
  talent of all regulars — the fraction of the usual production that
  dressed. (``diff_available_talent`` is the home − away difference of
  this fraction, replacing the old raw-sum definition, which was inert.)

Training uses actual participation for "absent"; the live path uses the
injury list (regular matched by normalized name × miss probability from
status) for *expected* absence. That train/predict gap is the usual one
for injury features and is documented in ARCHITECTURE §4.10.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from sqlalchemy import select

from nba_betting.db.models import Game, PlayerGameStat
from nba_betting.db.session import get_session

ROLL_GAMES = 10          # appearances used for "typical" minutes / talent
WINDOW_GAMES = 5         # team games used for regular status
REGULAR_SHARE = 0.6      # appeared in >= 60% of the window
REGULAR_MIN_MINUTES = 12.0
STAR_MINUTES = 30.0

AVAILABILITY_COLS = ("missing_minutes_pct", "star_out", "available_talent")

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_name(name: str) -> str:
    """Accent/punctuation/suffix-insensitive key so ESPN and NBA.com
    spellings of the same player match ("Luka Dončić" == "Luka Doncic",
    "Gary Trent Jr." == "Gary Trent")."""
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    n = n.lower().replace(".", "").replace("'", "").replace("-", " ")
    n = _SUFFIX_RE.sub("", n)
    return " ".join(n.split())


def load_player_game_df() -> pd.DataFrame:
    """``player_game_stats`` joined with the game's date/season. Empty
    DataFrame (with the expected columns) when the table is empty."""
    session = get_session()
    try:
        rows = session.execute(
            select(
                PlayerGameStat.game_id, Game.date, Game.season,
                PlayerGameStat.team_id, PlayerGameStat.player_id,
                PlayerGameStat.player_name, PlayerGameStat.minutes,
                PlayerGameStat.pts, PlayerGameStat.ast, PlayerGameStat.reb,
            )
            .join(Game, Game.id == PlayerGameStat.game_id)
            .where(Game.home_score.isnot(None))
        ).all()
    finally:
        session.close()
    cols = ["game_id", "date", "season", "team_id", "player_id", "player_name",
            "minutes", "pts", "ast", "reb"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _team_matrices(team_df: pd.DataFrame, team_games: pd.DataFrame):
    """Pivot one team-season into (minutes, talent) matrices indexed by the
    team's games in order, columns = players. NaN = did not appear."""
    played = team_df[team_df["minutes"] > 0]
    minutes = played.pivot_table(index="game_id", columns="player_id", values="minutes", aggfunc="max")
    talent_src = played.assign(_t=played["pts"] + played["ast"] + played["reb"])
    talent = talent_src.pivot_table(index="game_id", columns="player_id", values="_t", aggfunc="max")
    order = team_games["game_id"].tolist()
    minutes = minutes.reindex(index=order)
    talent = talent.reindex(index=order, columns=minutes.columns)
    return minutes, talent


def _rolling_state(minutes: pd.DataFrame, talent: pd.DataFrame):
    """Per (game, player) pre-game state: typical minutes, typical talent,
    appearances in the trailing window, games elapsed in the window.
    All shifted by one game so the target game is excluded."""
    present = minutes.notna()
    typ_min = minutes.rolling(ROLL_GAMES, min_periods=1).mean().shift(1)
    typ_tal = talent.rolling(ROLL_GAMES, min_periods=1).mean().shift(1)
    apps = present.astype(float).rolling(WINDOW_GAMES, min_periods=1).sum().shift(1)
    n_games = pd.Series(np.arange(1, len(minutes) + 1), index=minutes.index)
    window_len = n_games.clip(upper=WINDOW_GAMES).shift(1).fillna(0)
    return present, typ_min, typ_tal, apps, window_len


def _regular_mask(typ_min: pd.DataFrame, apps: pd.DataFrame, window_len: pd.Series) -> pd.DataFrame:
    need = np.maximum(1.0, np.ceil(REGULAR_SHARE * window_len.to_numpy()))[:, None]
    return (apps.to_numpy() >= need) & (typ_min.to_numpy() >= REGULAR_MIN_MINUTES)


def compute_availability_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team_id): ``missing_minutes_pct``, ``star_out``,
    ``available_talent`` from actual participation. Leak-free: regular
    status and typical minutes come from games strictly before each game.
    """
    empty = pd.DataFrame(columns=["game_id", "team_id", *AVAILABILITY_COLS])
    if player_df is None or player_df.empty:
        return empty

    df = player_df.sort_values(["date", "game_id"])
    # First appearance date per (player, team) — used to detect departures:
    # a regular who has appeared for ANOTHER team after his last game with
    # this team is traded, not injured.
    first_seen = df.groupby(["player_id", "team_id"])["date"].min()

    out = []
    for (season, team_id), tdf in df.groupby(["season", "team_id"], sort=False):
        team_games = tdf[["game_id", "date"]].drop_duplicates("game_id").sort_values(["date", "game_id"])
        minutes, talent = _team_matrices(tdf, team_games)
        if minutes.shape[1] == 0:
            continue
        present, typ_min, typ_tal, apps, window_len = _rolling_state(minutes, talent)
        regular = _regular_mask(typ_min, apps, window_len)

        # Departure mask: player p is "departed" at game g if he first
        # appeared for some other team on/after his last appearance here
        # and before g. Approximated per player from first_seen elsewhere.
        departed = np.zeros_like(regular, dtype=bool)
        dates = team_games["date"].to_numpy()
        for j, pid in enumerate(minutes.columns):
            others = first_seen.loc[pid] if pid in first_seen.index.get_level_values(0) else None
            if others is None or not isinstance(others, pd.Series):
                continue
            others = others.drop(index=team_id, errors="ignore")
            if others.empty:
                continue
            # last appearance for this team before each game (ffill of dates)
            app_dates = pd.Series(np.where(present.iloc[:, j], dates, np.datetime64("NaT")),
                                  index=minutes.index).ffill().shift(1)
            for other_first in others.to_numpy():
                gone = (app_dates.to_numpy() < other_first) & (dates >= other_first)
                departed[:, j] |= np.nan_to_num(gone, nan=False)

        regular = regular & ~departed
        absent = regular & ~present.to_numpy()
        tm = np.nan_to_num(typ_min.to_numpy(), nan=0.0)
        tt = np.nan_to_num(typ_tal.to_numpy(), nan=0.0)
        reg_min = (tm * regular).sum(axis=1)
        miss_min = (tm * absent).sum(axis=1)
        reg_tal = (tt * regular).sum(axis=1)
        pres_tal = (tt * (regular & present.to_numpy())).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            missing_pct = np.where(reg_min > 0, miss_min / reg_min, 0.0)
            avail_tal = np.where(reg_tal > 0, pres_tal / reg_tal, 0.0)
        star_out = ((tm >= STAR_MINUTES) & absent).any(axis=1).astype(float)
        out.append(pd.DataFrame({
            "game_id": minutes.index.to_numpy(),
            "team_id": team_id,
            "missing_minutes_pct": np.round(missing_pct, 4),
            "star_out": star_out,
            "available_talent": np.round(avail_tal, 4),
        }))
    if not out:
        return empty
    return pd.concat(out, ignore_index=True)


def latest_regulars(player_df: pd.DataFrame) -> dict[int, list[dict]]:
    """``{team_id: [{player_id, name_key, typical_minutes, typical_talent}]}``
    — each team's current regulars as of its last completed game (the
    trailing window INCLUDING that game, since the next game is upcoming).
    Used by the live path to weight injury-list entries."""
    if player_df is None or player_df.empty:
        return {}
    df = player_df.sort_values(["date", "game_id"])
    out: dict[int, list[dict]] = {}
    for team_id, tdf in df.groupby("team_id", sort=False):
        # Regular status is season-scoped: use the team's latest season.
        last_season = tdf.iloc[-1]["season"]
        sdf = tdf[tdf["season"] == last_season]
        team_games = sdf[["game_id", "date"]].drop_duplicates("game_id").sort_values(["date", "game_id"])
        minutes, talent = _team_matrices(sdf, team_games)
        if minutes.shape[1] == 0:
            continue
        present = minutes.notna()
        typ_min = minutes.rolling(ROLL_GAMES, min_periods=1).mean().iloc[-1]
        typ_tal = talent.rolling(ROLL_GAMES, min_periods=1).mean().iloc[-1]
        apps = present.astype(float).rolling(WINDOW_GAMES, min_periods=1).sum().iloc[-1]
        window_len = min(WINDOW_GAMES, len(minutes))
        need = max(1.0, np.ceil(REGULAR_SHARE * window_len))
        names = sdf.groupby("player_id")["player_name"].last()
        regs = []
        for pid in minutes.columns:
            if apps[pid] >= need and typ_min[pid] >= REGULAR_MIN_MINUTES:
                regs.append({
                    "player_id": int(pid),
                    "name_key": normalize_name(names.get(pid, "")),
                    "typical_minutes": float(typ_min[pid]),
                    "typical_talent": float(typ_tal[pid]),
                })
        out[int(team_id)] = regs
    return out


def expected_availability(regulars: list[dict], injuries: list, team_abbr: str) -> dict[str, float]:
    """Live counterpart of ``compute_availability_features`` for one team:
    expected missing-minutes share, star-out flag, and available-talent
    share given the current injury list (``PlayerInjury`` objects) for
    ``team_abbr``. Regulars not on the injury list count as available."""
    from nba_betting.data.injuries import _status_multiplier

    if not regulars:
        return {"missing_minutes_pct": 0.0, "star_out": 0.0, "available_talent": 0.0}
    miss_by_key: dict[str, float] = {}
    for inj in injuries:
        if (inj.team_abbr or "").upper() != (team_abbr or "").upper():
            continue
        key = normalize_name(inj.player_name)
        miss_by_key[key] = max(miss_by_key.get(key, 0.0), _status_multiplier(inj.status))
    reg_min = sum(r["typical_minutes"] for r in regulars)
    reg_tal = sum(r["typical_talent"] for r in regulars)
    miss_min = 0.0
    pres_tal = 0.0
    star_out = 0.0
    for r in regulars:
        p_miss = miss_by_key.get(r["name_key"], 0.0)
        miss_min += r["typical_minutes"] * p_miss
        pres_tal += r["typical_talent"] * (1.0 - p_miss)
        if r["typical_minutes"] >= STAR_MINUTES and p_miss >= 0.5:
            star_out = 1.0
    return {
        "missing_minutes_pct": round(miss_min / reg_min, 4) if reg_min > 0 else 0.0,
        "star_out": star_out,
        "available_talent": round(pres_tal / reg_tal, 4) if reg_tal > 0 else 0.0,
    }
