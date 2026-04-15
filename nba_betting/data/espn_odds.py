"""Extract betting odds from ESPN scoreboard API."""
from __future__ import annotations

from nba_betting.data.espn import fetch_scoreboard


def _moneyline_to_prob(ml: float | None) -> float | None:
    """Convert American moneyline to implied probability.

    Positive ML (underdog): prob = 100 / (ml + 100)
    Negative ML (favorite): prob = -ml / (-ml + 100)
    """
    if ml is None:
        return None
    if ml > 0:
        return 100.0 / (ml + 100.0)
    elif ml < 0:
        return -ml / (-ml + 100.0)
    else:
        return 0.5


def get_espn_odds(date_str: str | None = None) -> list[dict]:
    """Get current NBA game odds from ESPN.

    Returns list of dicts matching the Polymarket odds format:
    - teams: dict mapping team abbreviation -> implied probability
    - spread: home team spread (negative = favored)
    - over_under: total points line
    - event_title: matchup description
    - source: "espn"
    """
    games = fetch_scoreboard(date_str)
    odds_list = []

    for game in games:
        odds_data = game.get("odds", {})
        home_abbr = game["home_team"]["abbr"]
        away_abbr = game["away_team"]["abbr"]

        home_ml = odds_data.get("home_moneyline")
        away_ml = odds_data.get("away_moneyline")

        home_prob = _moneyline_to_prob(home_ml)
        away_prob = _moneyline_to_prob(away_ml)

        # If we got both moneylines, normalize to sum to 1 (remove vig)
        if home_prob is not None and away_prob is not None:
            total = home_prob + away_prob
            if total > 0:
                home_prob /= total
                away_prob /= total
        elif home_prob is not None:
            away_prob = 1.0 - home_prob
        elif away_prob is not None:
            home_prob = 1.0 - away_prob
        else:
            # No moneyline available — try spread as a rough proxy
            spread = odds_data.get("spread")
            if spread is not None:
                # Rough approximation: each point of spread ≈ 2.5% probability
                home_prob = 0.5 + (spread * -0.025)  # negative spread = favorite
                home_prob = max(0.05, min(0.95, home_prob))
                away_prob = 1.0 - home_prob
            else:
                continue  # No odds available at all

        odds_list.append({
            "teams": {home_abbr: home_prob, away_abbr: away_prob},
            "spread": odds_data.get("spread"),
            "over_under": odds_data.get("over_under"),
            "event_title": f"{away_abbr} @ {home_abbr}",
            "source": "espn",
            "provider": odds_data.get("provider", ""),
        })

    return odds_list
