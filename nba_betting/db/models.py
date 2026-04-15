from sqlalchemy import Column, Integer, Float, String, Boolean, Date, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)  # NBA.com team ID
    abbreviation = Column(String(3), unique=True, nullable=False)
    name = Column(String(50), nullable=False)
    conference = Column(String(4))  # East / West
    current_elo = Column(Float, default=1500.0)


class Game(Base):
    __tablename__ = "games"

    id = Column(String(20), primary_key=True)  # NBA.com game ID
    date = Column(Date, nullable=False, index=True)
    season = Column(String(7), nullable=False)  # e.g. "2025-26"
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_win = Column(Boolean)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])


class GameStats(Base):
    __tablename__ = "game_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String(20), ForeignKey("games.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    fgm = Column(Integer)
    fga = Column(Integer)
    fg_pct = Column(Float)
    fg3m = Column(Integer)
    fg3a = Column(Integer)
    fg3_pct = Column(Float)
    ftm = Column(Integer)
    fta = Column(Integer)
    ft_pct = Column(Float)
    oreb = Column(Integer)
    dreb = Column(Integer)
    reb = Column(Integer)
    ast = Column(Integer)
    stl = Column(Integer)
    blk = Column(Integer)
    tov = Column(Integer)
    pts = Column(Integer)
    plus_minus = Column(Float)


class EloRating(Base):
    __tablename__ = "elo_ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    game_id = Column(String(20), ForeignKey("games.id"))
    elo_before = Column(Float, nullable=False)
    elo_after = Column(Float, nullable=False)


class PlayerStat(Base):
    __tablename__ = "player_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    espn_player_id = Column(String(20), index=True)
    player_name = Column(String(100))
    team_id = Column(Integer, ForeignKey("teams.id"))
    season = Column(String(7))
    position = Column(String(5))
    depth_chart_rank = Column(Integer)  # 1=starter
    minutes_per_game = Column(Float, default=0.0)
    points_per_game = Column(Float, default=0.0)
    assists_per_game = Column(Float, default=0.0)
    rebounds_per_game = Column(Float, default=0.0)
    plus_minus_per_game = Column(Float, default=0.0)
    last_updated = Column(DateTime)

    team = relationship("Team", foreign_keys=[team_id])


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Foreign key to the concrete Game row once one exists. Nullable
    # because a snapshot can be taken hours before tipoff, when we only
    # know team pair + date.
    game_id = Column(String(20), ForeignKey("games.id"), nullable=True, index=True)
    game_date = Column(Date, nullable=False, index=True)
    home_team_id = Column(Integer, ForeignKey("teams.id"))
    away_team_id = Column(Integer, ForeignKey("teams.id"))
    source = Column(String(20))  # "polymarket", "espn"
    timestamp = Column(DateTime, nullable=False)
    home_prob = Column(Float)
    spread = Column(Float)
    over_under = Column(Float)


class HistoricalInjury(Base):
    """Daily snapshot of who was out on a given date.

    Populated by `injury sync` (or the cron job) — each run upserts one
    row per (date, player_id) so we can reconstruct the injury list as of
    any historical prediction time. Used by
    `build_feature_matrix` to retroactively add player-impact features.
    """
    __tablename__ = "historical_injuries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(Date, nullable=False, index=True)
    player_id = Column(String(20), index=True)
    player_name = Column(String(100))
    team_abbr = Column(String(5), index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True, index=True)
    status = Column(String(20))  # Out, Day-to-Day, GTD, etc.
    reason = Column(String(200))
    impact_rating = Column(Float)  # 0–10, auto from depth-chart rank
