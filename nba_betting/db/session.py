from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from nba_betting.config import DB_PATH
from nba_betting.db.models import Base

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


# Lightweight additive migrations. SQLAlchemy's create_all() only creates
# tables that don't exist — it never ALTERs columns onto an existing one.
# For a zero-ceremony project like this we don't want Alembic, so we just
# detect missing columns on init and ALTER TABLE them in. This is safe
# because every entry here is strictly additive (new nullable columns).
_ADDITIVE_COLUMNS: list[tuple[str, str, str]] = [
    # (table_name, column_name, column_ddl_type)
    ("odds_snapshots", "game_id", "VARCHAR(20)"),
    # Tier 1.3 — split off/def Elo on Team and EloRating.
    ("teams", "current_elo_off", "FLOAT"),
    ("teams", "current_elo_def", "FLOAT"),
    ("elo_ratings", "elo_off_before", "FLOAT"),
    ("elo_ratings", "elo_off_after", "FLOAT"),
    ("elo_ratings", "elo_def_before", "FLOAT"),
    ("elo_ratings", "elo_def_after", "FLOAT"),
]


def _apply_additive_migrations() -> None:
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for table, col, ddl in _ADDITIVE_COLUMNS:
            if table not in existing_tables:
                continue
            cols = {c["name"] for c in insp.get_columns(table)}
            if col not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))


def init_db():
    """Create all tables if they don't exist and apply additive migrations."""
    Base.metadata.create_all(engine)
    _apply_additive_migrations()


def get_session():
    """Get a new database session."""
    init_db()
    return SessionLocal()
