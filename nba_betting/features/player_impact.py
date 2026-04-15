"""Player-level impact features based on injuries and depth charts.

Computes how much each team is weakened by missing players, using a
Weighted Available Talent (WAT) approach.
"""
from __future__ import annotations

from nba_betting.data.injuries import PlayerInjury, _status_multiplier
from nba_betting.data.player_stats import get_team_players


def compute_player_impact_features(
    home_team_id: int,
    away_team_id: int,
    injuries: list[PlayerInjury],
    home_abbr: str = "",
    away_abbr: str = "",
) -> dict[str, float]:
    """Compute player impact features for a matchup.

    Uses injury list + player database to estimate how much each team
    is weakened by missing/injured players.

    Returns dict with 6 features:
    - home_missing_minutes_pct, away_missing_minutes_pct
    - home_star_out, away_star_out (binary)
    - diff_missing_minutes_pct, diff_available_talent
    """
    home_players = get_team_players(home_team_id)
    away_players = get_team_players(away_team_id)

    # Index injuries by team
    home_injuries = {i.player_name.lower(): i for i in injuries if i.team_abbr == home_abbr}
    away_injuries = {i.player_name.lower(): i for i in injuries if i.team_abbr == away_abbr}

    home_missing, home_star, home_wat = _compute_team_impact(home_players, home_injuries)
    away_missing, away_star, away_wat = _compute_team_impact(away_players, away_injuries)

    return {
        "home_missing_minutes_pct": home_missing,
        "away_missing_minutes_pct": away_missing,
        "home_star_out": float(home_star),
        "away_star_out": float(away_star),
        "diff_missing_minutes_pct": home_missing - away_missing,
        "diff_available_talent": home_wat - away_wat,
    }


def _compute_team_impact(
    players: list[dict],
    injuries: dict[str, PlayerInjury],
) -> tuple[float, bool, float]:
    """Compute impact metrics for a single team.

    Returns:
        missing_minutes_pct: fraction of team's typical minutes unavailable
        star_out: True if any player with MPG >= 30 is likely out
        wat_score: Weighted Available Talent score
    """
    if not players:
        return 0.0, False, 0.0

    total_minutes = sum(p["minutes_per_game"] for p in players)
    if total_minutes <= 0:
        # No minutes data — fall back to depth chart rank
        total_minutes = 240.0  # 48 min * 5 starters
        for p in players:
            if p["depth_chart_rank"] <= 5:
                p["minutes_per_game"] = 30.0
            elif p["depth_chart_rank"] <= 10:
                p["minutes_per_game"] = 15.0
            else:
                p["minutes_per_game"] = 5.0
        total_minutes = sum(p["minutes_per_game"] for p in players)

    missing_minutes = 0.0
    star_out = False
    wat_score = 0.0

    for p in players:
        name_lower = p["player_name"].lower()
        mpg = p["minutes_per_game"]
        talent = p["points_per_game"] + p["assists_per_game"] + p["rebounds_per_game"]

        inj = injuries.get(name_lower)
        if inj:
            miss_prob = _status_multiplier(inj.status)
            missing_minutes += mpg * miss_prob

            if mpg >= 30 and miss_prob >= 0.5:
                star_out = True

            # WAT: add only the expected available portion
            wat_score += talent * (1.0 - miss_prob)
        else:
            wat_score += talent

    missing_pct = missing_minutes / total_minutes if total_minutes > 0 else 0.0
    return round(missing_pct, 4), star_out, round(wat_score, 2)
