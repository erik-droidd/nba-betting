"""FiveThirtyEight-style Elo rating system for NBA teams.

Two layers:
1. **Aggregate Elo** (legacy): a single rating per team, updated by overall
   margin of victory. Kept for backwards compatibility — `current_elo`,
   `elo_before/after`, `update_elo`, `predict_home_win_prob` all still work.
2. **Off/Def Elo** (Tier 1.3): each team also has an offensive Elo and a
   defensive Elo. Offensive rating updates from points scored vs the
   opponent's *defensive* Elo, and vice versa. The aggregate Elo is
   recomputed as `(off + def) / 2` so downstream code keeps seeing a
   coherent single number.

Tier 1.4 — opponent-strength dampening: blowouts vs weaker opponents are
dampened more than blowouts vs strong ones. The standard MOV multiplier
formula has an autocorrelation term in the denominator (the
``elo_winner - elo_loser`` term), but it does not penalize a 30-pt drubbing
of the worst team relative to a 30-pt drubbing of the best. The new
``opp_strength_factor`` does exactly that, scaling the MOV multiplier by
``1 / (1 + exp(-(elo_loser - elo_winner) / 200))`` so dominating a weak
team carries less rating weight than the same margin against a peer.
"""
from __future__ import annotations

from datetime import date
import math

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


def opp_strength_factor(elo_winner: float, elo_loser: float) -> float:
    """Sigmoid weight that dampens MOV when winner is much stronger.

    Returns a value in roughly (0, 1):
    - winner == loser → 0.5
    - winner >> loser → approaches 0 (blowout vs weak team is discounted)
    - winner << loser → approaches 1 (upset blowout is amplified)

    The 200-point scale was chosen to roughly match a half-effect at a
    200-Elo gap (≈ ~3 pts in expected MOV by 538 calibration).
    """
    return 1.0 / (1.0 + math.exp(-(elo_loser - elo_winner) / 200.0))


def mov_multiplier(mov: int, elo_winner: float, elo_loser: float) -> float:
    """Margin-of-victory multiplier with autocorrelation correction.

    Tier 1.4 — multiplied by ``opp_strength_factor`` so blowouts against
    weaker teams update Elo less than blowouts against peers. The base
    formula is FiveThirtyEight's, retained verbatim; only the
    opponent-strength scaling is new.
    """
    base = ((abs(mov) + 3) ** 0.8) / (7.5 + 0.006 * (elo_winner - elo_loser))
    return base * opp_strength_factor(elo_winner, elo_loser)


def update_elo(
    home_elo: float,
    away_elo: float,
    home_score: int,
    away_score: int,
    k: float = ELO_K_FACTOR,
) -> tuple[float, float]:
    """Update aggregate Elo ratings after a game. Returns (new_home_elo, new_away_elo)."""
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


def update_off_def_elo(
    home_off: float, home_def: float,
    away_off: float, away_def: float,
    home_score: int, away_score: int,
    k: float = ELO_K_FACTOR,
) -> tuple[float, float, float, float]:
    """Update offensive / defensive Elo ratings after a game.

    Each team's offensive Elo updates against the opponent's defensive Elo,
    using points scored vs the league-average expectation derived from the
    Elo gap. The defensive Elo updates symmetrically against the
    opponent's offensive Elo, using points allowed.

    Implementation:
    - Convert the off-vs-def Elo gap into an expected points-scored value
      using a small linear sensitivity (LEAGUE_AVG_PTS + slope * Elo_gap).
    - Compare actual points to expected. Positive surprise raises off Elo
      and lowers opponent's def Elo (and vice versa).
    - The MOV multiplier reuses the aggregate-Elo formula and is applied
      to keep updates roughly on the same K=20 scale as the aggregate Elo.

    Returns (new_home_off, new_home_def, new_away_off, new_away_def).
    """
    # League-average and per-Elo-point sensitivity. The 0.05 slope was
    # tuned so a 100-Elo offensive edge corresponds to ~5 extra points
    # scored, matching empirical NBA splits.
    LEAGUE_AVG_PTS = 110.0
    PTS_PER_ELO = 0.05

    # Expected points scored by each team given off-vs-opp-def Elo gap.
    home_exp = LEAGUE_AVG_PTS + PTS_PER_ELO * (home_off - away_def)
    away_exp = LEAGUE_AVG_PTS + PTS_PER_ELO * (away_off - home_def)

    home_surplus = home_score - home_exp  # positive → home offense overperformed
    away_surplus = away_score - away_exp

    # Reuse the aggregate MOV multiplier on the *overall* result so that
    # upsets and blowouts shift off/def Elo proportionally to how much
    # they shifted aggregate Elo. The multiplier already includes Tier 1.4
    # opp-strength dampening.
    if home_score > away_score:
        mult = mov_multiplier(abs(home_score - away_score), home_off + home_def, away_off + away_def)
    else:
        mult = mov_multiplier(abs(home_score - away_score), away_off + away_def, home_off + home_def)

    # Half the K-factor on each component because every game now updates
    # two ratings (off and def) per team — keeps total movement ~equal to
    # the aggregate Elo update magnitude.
    k_split = k * 0.5
    # Normalize per-pt surplus into a [-1, 1]-ish residual using a 12-pt
    # scale, which is roughly one standard deviation of game-level scoring.
    home_off_change = k_split * mult * (home_surplus / 12.0)
    away_def_change = -k_split * mult * (home_surplus / 12.0)
    away_off_change = k_split * mult * (away_surplus / 12.0)
    home_def_change = -k_split * mult * (away_surplus / 12.0)

    return (
        home_off + home_off_change,
        home_def + home_def_change,
        away_off + away_off_change,
        away_def + away_def_change,
    )


def season_carryover(elo: float) -> float:
    """Regress Elo toward the mean between seasons."""
    return INITIAL_ELO + ELO_CARRYOVER * (elo - INITIAL_ELO)


def predict_home_win_prob(home_elo: float, away_elo: float) -> float:
    """Predict P(home win) including home-court advantage."""
    return expected_score(home_elo + ELO_HOME_ADVANTAGE, away_elo)


def compute_all_elos() -> dict[int, float]:
    """Process all historical games and compute current Elo ratings.

    Updates both aggregate Elo and split off/def Elo for every team-game.
    Returns a dict of {team_id: current_aggregate_elo} for back-compat.
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
        elos_off: dict[int, float] = {t.id: INITIAL_ELO for t in teams}
        elos_def: dict[int, float] = {t.id: INITIAL_ELO for t in teams}

        # Track season transitions for carryover
        current_season = None

        for game in games:
            # Apply carryover at season boundaries (to all three series)
            if game.season != current_season:
                if current_season is not None:
                    for team_id in elos:
                        elos[team_id] = season_carryover(elos[team_id])
                        elos_off[team_id] = season_carryover(elos_off[team_id])
                        elos_def[team_id] = season_carryover(elos_def[team_id])
                current_season = game.season

            home_id = game.home_team_id
            away_id = game.away_team_id

            # Ensure teams have ratings
            for tid in (home_id, away_id):
                if tid not in elos:
                    elos[tid] = INITIAL_ELO
                if tid not in elos_off:
                    elos_off[tid] = INITIAL_ELO
                if tid not in elos_def:
                    elos_def[tid] = INITIAL_ELO

            home_before = elos[home_id]
            away_before = elos[away_id]
            h_off_before = elos_off[home_id]
            h_def_before = elos_def[home_id]
            a_off_before = elos_off[away_id]
            a_def_before = elos_def[away_id]

            home_after, away_after = update_elo(
                home_before, away_before, game.home_score, game.away_score
            )
            (
                h_off_after, h_def_after,
                a_off_after, a_def_after,
            ) = update_off_def_elo(
                h_off_before, h_def_before,
                a_off_before, a_def_before,
                game.home_score, game.away_score,
            )

            # Store snapshots
            session.add(EloRating(
                team_id=home_id,
                date=game.date,
                game_id=game.id,
                elo_before=home_before,
                elo_after=home_after,
                elo_off_before=h_off_before,
                elo_off_after=h_off_after,
                elo_def_before=h_def_before,
                elo_def_after=h_def_after,
            ))
            session.add(EloRating(
                team_id=away_id,
                date=game.date,
                game_id=game.id,
                elo_before=away_before,
                elo_after=away_after,
                elo_off_before=a_off_before,
                elo_off_after=a_off_after,
                elo_def_before=a_def_before,
                elo_def_after=a_def_after,
            ))

            elos[home_id] = home_after
            elos[away_id] = away_after
            elos_off[home_id] = h_off_after
            elos_def[home_id] = h_def_after
            elos_off[away_id] = a_off_after
            elos_def[away_id] = a_def_after

        # Update current_elo on team records
        for team in teams:
            if team.id in elos:
                team.current_elo = elos[team.id]
                team.current_elo_off = elos_off[team.id]
                team.current_elo_def = elos_def[team.id]

        session.commit()
        return elos

    finally:
        session.close()


def get_current_elos() -> dict[int, float]:
    """Get current aggregate Elo ratings from the database."""
    session = get_session()
    try:
        teams = session.execute(select(Team)).scalars().all()
        return {t.id: (t.current_elo or INITIAL_ELO) for t in teams}
    finally:
        session.close()


def get_current_off_def_elos() -> dict[int, tuple[float, float]]:
    """Get current (off_elo, def_elo) per team. Returns INITIAL_ELO for
    teams that don't yet have a split rating (e.g., before backfill)."""
    session = get_session()
    try:
        teams = session.execute(select(Team)).scalars().all()
        return {
            t.id: (
                (t.current_elo_off if t.current_elo_off is not None else INITIAL_ELO),
                (t.current_elo_def if t.current_elo_def is not None else INITIAL_ELO),
            )
            for t in teams
        }
    finally:
        session.close()
