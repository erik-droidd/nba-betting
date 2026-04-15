from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "trained_models"
DB_PATH = DATA_DIR / "nba_betting.db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Bankroll management
DEFAULT_BANKROLL = 1000.0
MIN_EDGE_THRESHOLD = 0.02  # 2% minimum edge to bet
SUSPICIOUS_EDGE_THRESHOLD = 0.15  # 15% edge — flag as potentially unreliable
KELLY_FRACTION = 0.25  # Quarter-Kelly
MAX_BET_PCT = 0.05  # 5% of bankroll per bet
MAX_EXPOSURE_PCT = 0.25  # 25% total simultaneous exposure

# Bayesian shrinkage of model probability toward market log-odds.
# 0.0 = pure model. 1.0 = pure market (never bets). 0.6 is market-leaning,
# treats market as a strong prior. See nba_betting/betting/shrinkage.py.
MARKET_SHRINKAGE_LAMBDA = 0.6

# Minimum confidence the model must hold on the side it wants to bet on.
# Stops the system from betting an underdog the model itself only gives a
# 15% chance of winning, regardless of edge math. Standard quant practice
# (avoid "lottery ticket" bets where the EV is positive only because of
# extreme prices the model isn't actually contradicting).
MIN_BET_SIDE_PROB = 0.30

# Elo parameters (FiveThirtyEight)
INITIAL_ELO = 1500.0
ELO_K_FACTOR = 20.0
ELO_HOME_ADVANTAGE = 100.0
ELO_CARRYOVER = 0.75

# NBA API rate limiting
NBA_API_DELAY_SECONDS = 2.5

# Polymarket API
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

# ESPN API
ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_API_DELAY_SECONDS = 1.5

# Current NBA season
CURRENT_SEASON = "2025-26"

# NBA team abbreviation mapping (Polymarket name -> NBA.com abbreviation)
TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# Reverse mapping
ABBR_TO_TEAM_NAME = {v: k for k, v in TEAM_NAME_TO_ABBR.items() if len(k) > 5}
