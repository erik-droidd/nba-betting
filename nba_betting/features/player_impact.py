"""Live player-availability features for an upcoming game.

Mirrors ``features/availability.py`` (the training-side definitions) using
each team's current regulars — from the player game logs — and the current
ESPN injury list: expected missing share of regular-rotation minutes, a
star-out flag, and the share of usual production expected to be available.

Before 2026-09 this module weighted injuries by ``PlayerStat`` minutes,
which no sync ever populated (always 0), so the six features were inert
both here and in training. Now both sides use the same source and the
same definitions (ARCHITECTURE §4.10).
"""
from __future__ import annotations

from nba_betting.data.injuries import PlayerInjury

_ZERO = {"missing_minutes_pct": 0.0, "star_out": 0.0, "available_talent": 0.0}


def compute_player_impact_features(
    home_team_id: int,
    away_team_id: int,
    injuries: list[PlayerInjury],
    home_abbr: str = "",
    away_abbr: str = "",
    regulars: dict[int, list[dict]] | None = None,
) -> dict[str, float]:
    """Six features for a matchup:
    ``home/away_missing_minutes_pct``, ``home/away_star_out``,
    ``diff_missing_minutes_pct``, ``diff_available_talent``.

    ``regulars`` is ``{team_id: [...]}`` from
    ``availability.latest_regulars`` (the engine builds it once per slate).
    When it is None it is computed from the DB; a team with no known
    regulars gets neutral zeros — the same cold-start convention as the
    training rows for the first games of a season.
    """
    from nba_betting.features.availability import expected_availability

    if regulars is None:
        try:
            from nba_betting.features.availability import latest_regulars, load_player_game_df
            regulars = latest_regulars(load_player_game_df())
        except Exception:
            regulars = {}

    home = expected_availability(regulars.get(home_team_id, []), injuries, home_abbr) \
        if regulars.get(home_team_id) else dict(_ZERO)
    away = expected_availability(regulars.get(away_team_id, []), injuries, away_abbr) \
        if regulars.get(away_team_id) else dict(_ZERO)

    return {
        "home_missing_minutes_pct": home["missing_minutes_pct"],
        "away_missing_minutes_pct": away["missing_minutes_pct"],
        "home_star_out": float(home["star_out"]),
        "away_star_out": float(away["star_out"]),
        "diff_missing_minutes_pct": round(home["missing_minutes_pct"] - away["missing_minutes_pct"], 4),
        "diff_available_talent": round(home["available_talent"] - away["available_talent"], 4),
    }
