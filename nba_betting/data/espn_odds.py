"""Extract betting odds from ESPN scoreboard API."""
from __future__ import annotations

from nba_betting.data.espn import fetch_scoreboard

# Rough spread -> win-probability proxy used only when ESPN has a spread but
# no moneyline: each point of spread ≈ 2.5% probability.
_SPREAD_PROB_PER_POINT = 0.025


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


def spread_to_prob_heuristic(spread: float) -> float:
    """Home win probability implied by the home spread under the 2.5%/pt
    proxy (negative spread = home favored), clamped to [0.05, 0.95]."""
    return max(0.05, min(0.95, 0.5 - spread * _SPREAD_PROB_PER_POINT))


def is_spread_heuristic(home_prob: float | None, spread: float | None) -> bool:
    """True when ``home_prob`` is exactly the spread proxy for ``spread``.

    Every ESPN ``home_prob`` stored before the 2026-10 moneyline fix is this
    proxy (ESPN had moved its moneylines, so none were ever parsed). Pick'em
    spreads are excluded: there the proxy (0.5) and a real -110/-110 market
    coincide, and 0.5 is the right answer either way.
    """
    if home_prob is None or spread is None or spread == 0:
        return False
    return abs(home_prob - spread_to_prob_heuristic(spread)) < 1e-9


def market_home_prob(odds: dict, home_abbr: str) -> float | None:
    """Home win probability from an ESPN odds entry, but only when it is a
    real market price (moneyline-derived). Snapshot writers store this, not
    ``odds["teams"]``, so the spread proxy never lands in ``odds_snapshots``
    disguised as a closing line."""
    if odds.get("prob_source", "moneyline") != "moneyline":
        return None
    return (odds.get("teams") or {}).get(home_abbr)


def get_espn_odds(date_str: str | None = None) -> list[dict]:
    """Get current NBA game odds from ESPN.

    Returns list of dicts matching the Polymarket odds format:
    - teams: dict mapping team abbreviation -> implied probability
    - prob_source: "moneyline" (de-vigged market price) or "spread"
      (the 2.5%/pt proxy, used only when ESPN lists no moneyline)
    - home_moneyline / away_moneyline: raw American odds (None if absent)
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
        prob_source = "moneyline"

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
                home_prob = spread_to_prob_heuristic(spread)
                away_prob = 1.0 - home_prob
                prob_source = "spread"
            else:
                continue  # No odds available at all

        odds_list.append({
            "teams": {home_abbr: home_prob, away_abbr: away_prob},
            "prob_source": prob_source,
            "home_moneyline": home_ml,
            "away_moneyline": away_ml,
            "spread": odds_data.get("spread"),
            "over_under": odds_data.get("over_under"),
            "event_title": f"{away_abbr} @ {home_abbr}",
            "source": "espn",
            "provider": odds_data.get("provider", ""),
        })

    return odds_list
