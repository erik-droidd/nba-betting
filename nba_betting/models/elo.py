"""FiveThirtyEight-style Elo rating system for NBA teams."""
from __future__ import annotations

from datetime import date

from sqlalchemy import select, func

from nba_betting.config import (
    INITIAL_ELO,
    ELO_K_FACTOR,
    ELO_HOME_ADVANTAGE,
    ELO_CARRYOVER,
)
from nba_betting.db.models import Game, Team, EloRating
from nba_betting.db.session import get_session


def expected_score(elo_a: float, elo_b: float) -> float:
    """Probability that team A beats team B given their Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 400.0))


def mov_multiplier(mov: int, elo_winner: float, elo_loser: float) -> float:
    """Margin-of-victory multiplier with autocorrelation correction."""
    return ((abs(mov) + 3) ** 0.8) / (7.5 + 0.006 * (elo_winner - elo_loser))


def update_elo(
    home_elo: float,
    away_elo: float,
    home_score: int,
    away_score: int,
    k: float = ELO_K_FACTOR,
) -> tuple[float, float]:
    """Update Elo ratings after a game. Returns (new_home_elo, new_away_elo)."""
    # Home team gets advantage in prediction but not in rating update
    home_expected = expected_score(home_elo + ELO_HOME_ADVANTAGE, away_elo)
    home_win = 1.0 if home_score > away_score else 0.0
    mov = abs(home_score - away_score)

    if home_score > away_score:
        mult = mov_multiplier(mov, home_elo, away_elo)
    else:
        mult = mov_multiplier(mov, away_elo, home_elo)

    change = k * mult * (home_win - home_expected)
    return home_elo + change, away_elo - change


def season_carryover(elo: float) -> float:
    """Regress Elo toward the mean between seasons."""
    return INITIAL_ELO + ELO_CARRYOVER * (elo - INITIAL_ELO)


def predict_home_win_prob(home_elo: float, away_elo: float) -> float:
    """Predict P(home win) including home-court advantage."""
    return expected_score(home_elo + ELO_HOME_ADVANTAGE, away_elo)


def compute_all_elos() -> dict[int, float]:
    """Process all historical games and compute current Elo ratings.

    Returns a dict of {team_id: current_elo}.
    """
    session = get_session()
    try:
        # Get all games ordered by date
        games = (
            session.execute(
                select(Game)
                .where(Game.home_score.isnot(None))
                .order_by(Game.date, Game.id)
            )
            .scalars()
            .all()
        )

        if not games:
            return {}

        # Clear existing Elo records
        session.query(EloRating).delete()

        # Initialize all teams
        teams = session.execute(select(Team)).scalars().all()
        elos: dict[int, float] = {t.id: INITIAL_ELO for t in teams}

        # Track season transitions for carryover
        current_season = None

        for game in games:
            # Apply carryover at season boundaries
            if game.season != current_season:
                if current_season is not None:
                    for team_id in elos:
                        elos[team_id] = season_carryover(elos[team_id])
                current_season = game.season

            home_id = game.home_team_id
            away_id = game.away_team_id

            # Ensure teams have ratings
            if home_id not in elos:
                elos[home_id] = INITIAL_ELO
            if away_id not in elos:
                elos[away_id] = INITIAL_ELO

            home_before = elos[home_id]
            away_before = elos[away_id]

            home_after, away_after = update_elo(
                home_before, away_before, game.home_score, game.away_score
            )

            # Store snapshots
            session.add(EloRating(
                team_id=home_id,
                date=game.date,
                game_id=game.id,
                elo_before=home_before,
                elo_after=home_after,
            ))
            session.add(EloRating(
                team_id=away_id,
                date=game.date,
                game_id=game.id,
                elo_before=away_before,
                elo_after=away_after,
            ))

            elos[home_id] = home_after
            elos[away_id] = away_after

        # Update current_elo on team records
        for team in teams:
            if team.id in elos:
                team.current_elo = elos[team.id]

        session.commit()
        return elos

    finally:
        session.close()


def get_current_elos() -> dict[int, float]:
    """Get current Elo ratings from the database."""
    session = get_session()
    try:
        teams = session.execute(select(Team)).scalars().all()
        return {t.id: (t.current_elo or INITIAL_ELO) for t in teams}
    finally:
        session.close()
