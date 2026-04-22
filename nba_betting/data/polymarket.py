"""Polymarket API integration for NBA odds."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import requests

from nba_betting.config import GAMMA_API_BASE, CLOB_API_BASE

NBA_TZ = ZoneInfo("America/New_York")

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
    """Convert a single team name/nickname to abbreviation.

    Tier 3.5 — fuzzy fallback. Exact match is tried first (the fast
    path that already handles 99% of titles). When that fails we fall
    back to substring containment against the ordered dictionary — the
    order is already longest-first for collision-safety, so a substring
    match like "trail blazers" inside "portland trail blazers tonight"
    still picks POR correctly. Only called for titles where the strict
    parser couldn't produce a pair, so it doesn't regress the fast path.
    """
    lower = name.lower().strip()
    for team_name, abbr in _SHORT_TO_ABBR_ORDERED:
        if team_name == lower:
            return abbr
    # Fuzzy fallback: substring containment with longest-first ordering.
    for team_name, abbr in _SHORT_TO_ABBR_ORDERED:
        if team_name in lower:
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


_SLUG_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})$")


def _slug_date(slug: str) -> Optional[str]:
    """ET game date from a Polymarket event slug like ``nba-orl-det-2026-04-22``."""
    m = _SLUG_DATE_RE.search(slug or "")
    return m.group(1) if m else None


def game_date_et(game: dict) -> Optional[str]:
    """ET calendar date (``YYYY-MM-DD``) of a scheduled game's tipoff."""
    gtu = game.get("game_time_utc") or ""
    if not gtu:
        return None
    try:
        if gtu.endswith("Z") and "+" not in gtu:
            gtu = gtu[:-1] + "+00:00"
        dt = datetime.fromisoformat(gtu)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(NBA_TZ).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def index_odds_by_pair(odds: list[dict]) -> dict[frozenset, list[dict]]:
    """Group odds dicts by team-pair, keeping every candidate.

    Polymarket publishes a separate event per matchup date, so one pair
    can have several active events at once. Callers must disambiguate
    with :func:`match_odds_for_game`.
    """
    idx: dict[frozenset, list[dict]] = {}
    for o in odds:
        teams = o.get("teams", {})
        if len(teams) != 2:
            continue
        idx.setdefault(frozenset(teams.keys()), []).append(o)
    return idx


def match_odds_for_game(
    index: dict[frozenset, list[dict]],
    pair: frozenset,
    et_date: Optional[str],
) -> Optional[dict]:
    """Return the odds dict for ``pair`` on ``et_date``, or ``None``.

    Prefers an exact (pair, game_date) match. Falls back to the lone
    candidate only if there is exactly one and it has no parseable
    date — preserves legacy behavior without resurrecting the
    multi-event collision bug (2026-04-22: stored 0.645 for DET when
    the true moneyline was 0.785, because a future rematch overwrote
    tonight's event under a pair-only index).
    """
    cands = index.get(pair, ())
    if et_date:
        for c in cands:
            if c.get("game_date") == et_date:
                return c
    if len(cands) == 1 and not cands[0].get("game_date"):
        return cands[0]
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
        slug = event.get("slug", "")
        game_date = _slug_date(slug)
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
                "event_slug": slug,
                "game_date": game_date,
            })

    return odds
