"""Polymarket API integration for NBA odds."""
from __future__ import annotations

import json
import re
from typing import Optional

import requests

from nba_betting.config import GAMMA_API_BASE, CLOB_API_BASE

# Short team name (as used in Polymarket titles) -> NBA abbreviation
# Order matters: longer names must come first to avoid substring collisions
# (e.g., "hornets" before "nets", "trail blazers" before "blazers")
_SHORT_TO_ABBR_ORDERED = [
    ("timberwolves", "MIN"),
    ("trail blazers", "POR"),
    ("cavaliers", "CLE"),
    ("mavericks", "DAL"),
    ("grizzlies", "MEM"),
    ("warriors", "GSW"),
    ("clippers", "LAC"),
    ("pelicans", "NOP"),
    ("hornets", "CHA"),
    ("celtics", "BOS"),
    ("rockets", "HOU"),
    ("nuggets", "DEN"),
    ("pistons", "DET"),
    ("pacers", "IND"),
    ("lakers", "LAL"),
    ("knicks", "NYK"),
    ("thunder", "OKC"),
    ("raptors", "TOR"),
    ("blazers", "POR"),
    ("wizards", "WAS"),
    ("sixers", "PHI"),
    ("wolves", "MIN"),
    ("magic", "ORL"),
    ("bucks", "MIL"),
    ("bulls", "CHI"),
    ("hawks", "ATL"),
    ("kings", "SAC"),
    ("spurs", "SAS"),
    ("heat", "MIA"),
    ("suns", "PHX"),
    ("nets", "BKN"),
    ("cavs", "CLE"),
    ("mavs", "DAL"),
    ("jazz", "UTA"),
    ("76ers", "PHI"),
]


def _name_to_abbr(name: str) -> Optional[str]:
    """Convert a single team name/nickname to abbreviation."""
    lower = name.lower().strip()
    for team_name, abbr in _SHORT_TO_ABBR_ORDERED:
        if team_name == lower:
            return abbr
    return None


def _extract_teams_from_title(title: str) -> Optional[tuple[str, str]]:
    """Extract two team abbreviations from a title like 'Bulls vs. Wizards'.

    Returns (first_team, second_team) in order of appearance.
    """
    # Split on "vs." or "vs"
    parts = re.split(r'\s+vs\.?\s+', title, maxsplit=1)
    if len(parts) != 2:
        return None

    first = _name_to_abbr(parts[0])
    second = _name_to_abbr(parts[1])

    if first and second and first != second:
        return (first, second)
    return None


def _is_game_event(title: str) -> bool:
    """Check if an event title represents a single game matchup."""
    lower = title.lower()
    if "vs" not in lower:
        return False
    exclude = ["champion", "mvp", "rookie", "playoffs", "division", "record",
               "season", "draft", "will ", "retire", "over or under"]
    return not any(kw in lower for kw in exclude)


def fetch_nba_game_events() -> list[dict]:
    """Fetch active NBA game-level events from Polymarket."""
    all_events = []
    offset = 0

    while True:
        resp = requests.get(
            f"{GAMMA_API_BASE}/events",
            params={
                "tag_slug": "nba",
                "active": "true",
                "closed": "false",
                "limit": 100,
                "offset": offset,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        for event in data:
            title = event.get("title", "")
            if _is_game_event(title):
                all_events.append(event)

        if len(data) < 100:
            break
        offset += 100

    return all_events


def get_nba_odds() -> list[dict]:
    """Get current NBA game odds from Polymarket.

    Polymarket structures NBA game moneylines with outcomes named after teams
    (e.g., outcomes=["Bulls", "Wizards"], prices=["0.665", "0.335"]).
    The price for a team outcome = implied win probability.

    Returns list of dicts with:
    - teams: dict mapping team abbreviation -> implied probability
    - event_title
    """
    events = fetch_nba_game_events()
    odds = []

    for event in events:
        title = event.get("title", "")
        teams = _extract_teams_from_title(title)
        if not teams:
            continue

        first_abbr, second_abbr = teams
        markets = event.get("markets", [])

        # Skip events where ALL markets are closed (game resolved/expired)
        if markets and all(m.get("closed") for m in markets):
            continue

        team_prices = {}

        for market in markets:
            # Skip individual closed markets — they hold stale post-game prices
            if market.get("closed"):
                continue

            outcomes_raw = market.get("outcomes", [])
            prices_raw = market.get("outcomePrices", [])
            question = market.get("question", "")

            # Polymarket returns these as JSON strings, not lists
            if isinstance(outcomes_raw, str):
                try:
                    outcomes = json.loads(outcomes_raw)
                except (json.JSONDecodeError, TypeError):
                    outcomes = []
            else:
                outcomes = outcomes_raw

            if isinstance(prices_raw, str):
                try:
                    outcome_prices = json.loads(prices_raw)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []
            else:
                outcome_prices = prices_raw

            if not outcomes or not outcome_prices:
                continue

            # Case 1: Moneyline market with team names as outcomes
            # e.g., outcomes=["Bulls", "Wizards"], prices=["0.665", "0.335"]
            if len(outcomes) == 2 and question.lower() == title.lower():
                for outcome, price_str in zip(outcomes, outcome_prices):
                    abbr = _name_to_abbr(outcome)
                    if abbr and abbr in (first_abbr, second_abbr):
                        try:
                            team_prices[abbr] = float(price_str)
                        except (ValueError, TypeError):
                            pass
                if team_prices:
                    break  # Found the moneyline, no need to check more markets

            # Case 2: Yes/No markets per team
            # e.g., question="Will the Bulls win?", outcomes=["Yes","No"]
            if len(outcomes) == 2:
                yes_idx = next(
                    (i for i, o in enumerate(outcomes) if o.lower() == "yes"),
                    None,
                )
                if yes_idx is not None:
                    # Check which team this question is about
                    q_lower = question.lower()
                    for team_name, abbr in _SHORT_TO_ABBR_ORDERED:
                        if team_name in q_lower and abbr in (first_abbr, second_abbr):
                            try:
                                team_prices[abbr] = float(outcome_prices[yes_idx])
                            except (ValueError, TypeError):
                                pass
                            break

        # Get prices for our two teams
        price1 = team_prices.get(first_abbr)
        price2 = team_prices.get(second_abbr)

        # If we have one, infer the other
        if price1 is not None and price2 is None:
            price2 = 1.0 - price1
        elif price2 is not None and price1 is None:
            price1 = 1.0 - price2

        if price1 is not None and price2 is not None:
            # Reject extreme/degenerate prices: post-game (0/1), arbitrage gaps, NaN
            if (
                price1 <= 0.01 or price1 >= 0.99
                or price2 <= 0.01 or price2 >= 0.99
                or abs((price1 + price2) - 1.0) > 0.05
            ):
                continue
            odds.append({
                "teams": {first_abbr: price1, second_abbr: price2},
                "event_title": title,
            })

    return odds
