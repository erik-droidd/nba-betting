"""Player-availability features from player game logs (2026-09)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nba_betting.features.availability import (
    compute_availability_features,
    expected_availability,
    latest_regulars,
    normalize_name,
)
from nba_betting.data.injuries import PlayerInjury


def _season(team_id, n_games, roster, season="2025-26", start=date(2025, 10, 20),
            absences=None, game_prefix="g"):
    """Rows for one team's season: ``roster`` = {player_id: (name, minutes)};
    ``absences`` = {game_index: [player_id, ...]} who did not play."""
    absences = absences or {}
    rows = []
    for i in range(n_games):
        gid = f"{game_prefix}{team_id}_{i:03d}"
        d = start + timedelta(days=2 * i)
        for pid, (name, mins) in roster.items():
            if pid in absences.get(i, []):
                continue
            rows.append({"game_id": gid, "date": pd.Timestamp(d), "season": season,
                         "team_id": team_id, "player_id": pid, "player_name": name,
                         "minutes": mins, "pts": mins / 2, "ast": 2, "reb": 3})
    return rows


ROSTER = {
    1: ("Star One", 36), 2: ("Second Guy", 32), 3: ("Third Man", 28),
    4: ("Fourth", 24), 5: ("Fifth", 20), 6: ("Bench A", 14),
    7: ("Garbage Time", 4),   # never a regular (< 12 min)
}


def test_normalize_name():
    assert normalize_name("Luka Dončić") == "luka doncic"
    assert normalize_name("Gary Trent Jr.") == "gary trent"
    assert normalize_name("Jaren Jackson Jr") == normalize_name("Jaren  Jackson Jr.")
    assert normalize_name("Karl-Anthony Towns") == "karl anthony towns"
    assert normalize_name("") == ""


def test_star_absence_is_measured_without_leakage():
    df = pd.DataFrame(_season(10, 12, ROSTER, absences={6: [1], 7: [1], 9: [6]}))
    av = compute_availability_features(df).sort_values("game_id").reset_index(drop=True)
    assert len(av) == 12
    # First game of the season: no history -> no regulars -> neutral zeros.
    assert av.loc[0, ["missing_minutes_pct", "star_out", "available_talent"]].tolist() == [0.0, 0.0, 0.0]
    # Game 2: everyone who played game 1 with >= 12 min is a regular; all present.
    assert av.loc[1, "missing_minutes_pct"] == 0.0 and av.loc[1, "available_talent"] == 1.0
    # Game 6: the 36-mpg star sits -> 36 / (36+32+28+24+20+14) of regular minutes missing, star_out.
    reg_min = 36 + 32 + 28 + 24 + 20 + 14
    assert av.loc[6, "missing_minutes_pct"] == pytest.approx(36 / reg_min, abs=1e-3)
    assert av.loc[6, "star_out"] == 1.0
    assert av.loc[6, "available_talent"] == pytest.approx(1 - (36 / 2 + 5) / sum(m / 2 + 5 for _, m in list(ROSTER.values())[:6]), abs=1e-3)
    # Game 7: star still out; his typical minutes come from games BEFORE 7
    # (he did not play game 6, which must not drag his average to 0).
    assert av.loc[7, "missing_minutes_pct"] == pytest.approx(36 / reg_min, abs=1e-3)
    # Game 8: star back -> nothing missing. Game 9: 14-min bench regular sits, not a star.
    assert av.loc[8, "missing_minutes_pct"] == 0.0
    assert av.loc[9, "star_out"] == 0.0 and 0 < av.loc[9, "missing_minutes_pct"] < 0.1


def test_regular_status_is_season_scoped_and_traded_players_are_not_absent():
    # Season A: player 1 is a star. Season B: he is gone (signed elsewhere).
    rows = _season(10, 6, ROSTER, season="2024-25", start=date(2025, 3, 1), game_prefix="a")
    roster_b = {k: v for k, v in ROSTER.items() if k != 1}
    rows += _season(10, 6, roster_b, season="2025-26", start=date(2025, 10, 20), game_prefix="b")
    # Mid-season B trade: player 2 appears for team 20 from game 3 on.
    rows += _season(20, 6, {2: ("Second Guy", 30), 8: ("Other", 30), 9: ("Other2", 30)},
                    season="2025-26", start=date(2025, 10, 26), game_prefix="c")
    rows = [r for r in rows if not (r["team_id"] == 10 and r["player_id"] == 2
                                    and r["season"] == "2025-26" and r["game_id"] >= "b10_003")]
    df = pd.DataFrame(rows)
    av = compute_availability_features(df)
    b = av[av["game_id"].str.startswith("b")].sort_values("game_id").reset_index(drop=True)
    # Off-season departure of player 1 never reads as an absence in season B.
    assert b.loc[1, "missing_minutes_pct"] == 0.0 and b.loc[2, "star_out"] == 0.0
    # Player 2 traded before game b_003: not counted as absent from game 3 on
    # (he appeared for team 20 on 2025-10-26, before b_003 on 2025-10-26... use dates).
    assert (b.loc[3:, "missing_minutes_pct"] == 0.0).all()


def test_latest_regulars_and_expected_availability():
    df = pd.DataFrame(_season(10, 8, ROSTER))
    regs = latest_regulars(df)[10]
    names = {r["name_key"]: r for r in regs}
    assert set(names) == {"star one", "second guy", "third man", "fourth", "fifth", "bench a"}
    assert names["star one"]["typical_minutes"] == pytest.approx(36.0)

    injuries = [
        PlayerInjury("Star One", "BOS", "Out", ""),
        PlayerInjury("Bench A", "BOS", "Questionable", ""),
        PlayerInjury("Fourth", "LAL", "Out", ""),      # other team: ignored
    ]
    e = expected_availability(regs, injuries, "BOS")
    reg_min = 36 + 32 + 28 + 24 + 20 + 14
    assert e["missing_minutes_pct"] == pytest.approx((36 * 1.0 + 14 * 0.5) / reg_min, abs=1e-3)
    assert e["star_out"] == 1.0
    assert 0.5 < e["available_talent"] < 1.0
    # No injuries -> fully available.
    assert expected_availability(regs, [], "BOS") == {
        "missing_minutes_pct": 0.0, "star_out": 0.0, "available_talent": 1.0}
    # No regulars known -> neutral zeros (matches the training cold-start rows).
    assert expected_availability([], injuries, "BOS")["available_talent"] == 0.0


def test_empty_input():
    assert compute_availability_features(pd.DataFrame()).empty
    assert latest_regulars(pd.DataFrame()) == {}
