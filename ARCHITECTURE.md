# NBA Betting System — Architecture & Design

This document is the **source of truth** for how the system works, why the
design is the way it is, and how to rebuild or extend it. It is written for
Claude Code (or any LLM assistant) to consume in a single pass: every
section is self-contained, filenames are absolute-to-repo, and the
methodology rationales are included inline so the reasoning doesn't have
to be rediscovered.

If you are tempted to "improve" something, read the **Methodology
rationales** section first — a lot of the choices here look unusual but
encode fixes for specific failure modes that a naive redesign would
reintroduce.

---

## 1. System at a glance

```
             ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  SOURCES    │  NBA.com     │    │  Polymarket  │    │   ESPN       │
             │  (games,     │    │  (odds)      │    │  (injuries,  │
             │   stats)     │    │              │    │   odds, ATS) │
             └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
                    │ 2.5s rate-limit   │                    │ 1.5s rate-limit
                    ▼                   ▼                    ▼
             ┌───────────────────────────────────────────────────────┐
  STORAGE    │  SQLite (data/nba_betting.db)                         │
             │  tables: teams, games, game_stats, elo_ratings,       │
             │  player_stats, odds_snapshots                         │
             │  + JSON: data/injuries.json, prediction_history.json  │
             └───────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
             ┌───────────────────────────────────────────────────────┐
  FEATURES   │  compute_rolling_features → add_four_factors →       │
             │  add_rest_features → build_feature_matrix             │
             │  (60 diff-features, all computed with shift(1) to     │
             │   prevent temporal leakage)                           │
             └───────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
             ┌───────────────────────────────────────────────────────┐
  MODELS     │  Elo  +  HistGradientBoosting (calibrated isotonic)  │
             │          │                                            │
             │          └─> log-odds ensemble with learned weight    │
             └───────────────────────────┬───────────────────────────┘
                                         │  p_model
                                         ▼
             ┌───────────────────────────────────────────────────────┐
  INJURY     │  net_adjust = home_impact - away_impact               │
  ADJUST     │  p_model ← clip(p_model + net_adjust, 0.01, 0.99)     │
             └───────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
             ┌───────────────────────────────────────────────────────┐
  SHRINKAGE  │  p_shrunk = σ( (1-λ)·logit(p_model) + λ·logit(p_mkt)) │
             │  λ = MARKET_SHRINKAGE_LAMBDA = 0.6 (market-leaning)   │
             └───────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
             ┌───────────────────────────────────────────────────────┐
  BET        │  edge = p_shrunk / p_market - 1                      │
  FILTER     │  gate 1: edge ≥ MIN_EDGE_THRESHOLD (2%)               │
             │  gate 2: p_shrunk ≥ MIN_BET_SIDE_PROB (0.30)          │
             │  size  = quarter-Kelly, capped at 5% bankroll         │
             └───────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
             ┌───────────────────────────────────────────────────────┐
  OUTPUT     │  Rich console table, FastAPI dashboard, JSON history │
             └───────────────────────────────────────────────────────┘
```

---

## 2. Directory / file map (what each file is responsible for)

```
nba_betting/
├── __main__.py                 — entry point; wires typer app
├── cli.py                      — all CLI commands: predict, train, sync,
│                                 backtest, simulate, diagnose, injury,
│                                 sync-players, serve, performance, elo
├── config.py                   — ALL tuning knobs + paths + constants
│                                 (bankroll, Kelly, shrinkage λ, bet-side
│                                 floor, Elo params, API URLs, team maps)
│
├── db/
│   ├── session.py              — SQLAlchemy session factory
│   └── models.py               — Team, Game, GameStats, EloRating,
│                                 PlayerStat, OddsSnapshot tables
│
├── data/
│   ├── nba_stats.py            — NBA.com client (ScoreboardV3 + LeagueGameFinder).
│   │                              ⚠ Uses ET timezone via zoneinfo for "today".
│   │                              ⚠ Never uses live ScoreBoard() — it caches
│   │                                 stale prior-day data for hours after rollover.
│   ├── polymarket.py           — Gamma + CLOB clients. Filters closed markets
│   │                              and extreme (<1%, >99%) prices.
│   ├── espn.py                 — ESPN client: scoreboard, injuries, depth
│   │                              charts, rosters, team summaries.
│   │                              Handles ESPN abbr ↔ NBA abbr mapping.
│   ├── espn_odds.py            — Extracts moneyline + spread + O/U from
│   │                              ESPN scoreboard. Used as fallback when
│   │                              Polymarket has no market for a game.
│   ├── injuries.py             — PlayerInjury dataclass + JSON persistence
│   │                              + ESPN sync. Preserves manual overrides.
│   │                              get_team_injury_adjustment(abbr) returns
│   │                              the post-hoc probability delta.
│   ├── player_stats.py         — Roster + depth chart sync into PlayerStat.
│   └── odds_tracker.py         — snapshot_current_odds / get_line_movement
│                                 (opening-vs-current spread & prob).
│
├── features/
│   ├── rolling.py              — Per-team-game rolling means (5/10/20).
│   │                              Uses shift(1) to exclude the current game
│   │                              (NO LEAKAGE). Computes pts_against /
│   │                              net_rtg_game with vectorized np.where.
│   ├── four_factors.py         — eFG%, TOV%, ORB%, FT-rate (Dean Oliver).
│   ├── rest_days.py            — rest_days, is_b2b, games_last_7/14.
│   ├── player_impact.py        — ESPN-driven player features (starter out,
│   │                              missing minutes %, available-talent diff).
│   └── builder.py              — THE assembler. Two entry points:
│                                   build_feature_matrix() for training,
│                                   build_prediction_features() for one game.
│                                 Both must produce the SAME columns in the
│                                 SAME order or prediction-time imputation
│                                 will silently feed garbage to the model.
│
├── models/
│   ├── elo.py                  — 538-style Elo with home advantage,
│   │                              MOV multiplier, season carryover.
│   │                              get_current_elos() reads the latest
│   │                              EloRating row per team.
│   ├── xgboost_model.py        — HistGradientBoostingClassifier wrapper
│   │                              (name kept for historical reasons).
│   │                              Saves/loads (estimator, feature_cols).
│   │                              Also persists feature_means for NaN
│   │                              imputation at prediction time.
│   ├── calibration.py          — Isotonic calibration via
│   │                              CalibratedClassifierCV(cv="prefit").
│   │                              Defaults to 'isotonic' — see §6.
│   │                              Uses FrozenEstimator for sklearn ≥ 1.8.
│   └── ensemble.py             — Log-odds (logit-space) blend of Elo and
│                                 GBM probs, with a grid-searched weight
│                                 persisted to ensemble_weight.joblib.
│
├── betting/
│   ├── edge.py                 — compute_edge, is_positive_ev, and
│   │                              confidence_badge (STRONG/MODERATE/LEAN/
│   │                              SUSPECT thresholds).
│   ├── kelly.py                — kelly_fraction + compute_bet_size
│   │                              (quarter-Kelly, 5% cap).
│   ├── shrinkage.py            — shrink_to_market (Bayesian log-odds
│   │                              shrinkage). The single most important
│   │                              change in the whole system — see §6.
│   ├── recommendations.py      — generate_recommendations: wires
│   │                              predict → inject → shrink → edge →
│   │                              floor → size → badge → explanation.
│   ├── explanations.py         — Template-based (no LLM) natural-language
│   │                              "why this bet" generator. Prefers
│   │                              signals that agree with the bet side.
│   ├── backtest.py             — Walk-forward historical simulation.
│   │                              ⚠ Uses Elo as the "market proxy" because
│   │                              we don't have historical Polymarket
│   │                              prices. See §6 for why backtest ROI
│   │                              will systematically differ from live.
│   ├── montecarlo.py           — Monte Carlo bankroll simulation (resample
│   │                              from backtest results).
│   └── tracker.py              — record_predictions + update_results
│                                 for prediction_history.json.
│
├── display/
│   └── console.py              — Rich-based terminal tables + panels.
│                                 Displays the shrunken probability (not
│                                 raw model) so the Model column reconciles
│                                 with the Edge column.
│
└── api/
    ├── app.py                  — FastAPI app factory, CORS, static mount.
    └── routes.py               — /api/predictions/today, /elo, /performance,
                                  /injuries. Caches load_model() once.
```

Top-level:

```
data/
  nba_betting.db            SQLite, the source of truth
  injuries.json             Current injury overrides
  prediction_history.json   Resolved bets + pending predictions
trained_models/
  gbm_latest.joblib         (base_estimator, feature_cols)
  calibrated_model.joblib   Isotonic wrapper
  feature_cols.joblib       Column order for imputation
  feature_means.joblib      Training-set means for NaN imputation
  ensemble_weight.joblib    Grid-searched optimal Elo weight
frontend/
  index.html                Single-file dashboard, fetches /api/*
USAGE.md                    End-user facing operational guide
ARCHITECTURE.md             This file
```

---

## 3. Data flow — tracing one `predict` invocation end to end

This is the exact sequence of operations when a user runs
`python3 -m nba_betting predict`. Every numbered step maps to a function
or module you can `grep` for.

1. **`cli.predict()`** is invoked.
2. **`fetch_todays_games()`** (`data/nba_stats.py`) calls ScoreboardV3 with
   an explicit ET date (computed via `zoneinfo.ZoneInfo("America/New_York")`).
   Returns the slate as a list of dicts with `home_team_id`, `away_team_id`,
   `home_team_abbr`, etc.
3. If today has no remaining games, **`fetch_upcoming_games(days_ahead=7)`**
   walks forward from `today_et + 1` and returns the next available slate.
4. **`get_current_elos()`** (`models/elo.py`) loads `{team_id → elo}` from
   the `teams` table.
5. **Model loading**:
   - Try `load_calibrated_model()` → returns the isotonic wrapper.
   - Load `(base_estimator, feature_cols) = load_model()` — needed for
     `feature_cols` even when the calibrated wrapper is the active model.
   - Load `feature_means` for NaN imputation.
   - Load `ensemble_weight` from disk (falls back to 0.3 if missing).
6. **Rolling features**: `compute_rolling_features()` +
   `add_four_factors()` + `add_opponent_rebound_data()` +
   `add_rest_features()`, then a per-team groupby adds rolling
   four-factors columns. This is the exact same transformation as in
   `build_feature_matrix()` so training and inference stay aligned.
7. A closure `_xgb_predict(home_elo, away_elo, home_id, away_id)` is built.
   It:
   - Calls `build_prediction_features()` with the stats row.
   - Aligns columns against `feature_cols` (imputing missing ones via
     `feature_means`).
   - Runs `actual_model.predict_proba(row)[:, 1]` → `xgb_prob`.
   - Runs `predict_home_win_prob(home_elo, away_elo)` → `elo_prob`.
   - Returns `ensemble_predict(elo_prob, xgb_prob)` (log-odds blend).
8. **`sync_injuries_from_espn()`** refreshes `data/injuries.json` from
   ESPN, preserving manual overrides.
9. **`get_nba_odds()`** hits Polymarket Gamma + CLOB; **`get_espn_odds()`**
   hits the ESPN scoreboard. Both return lists of
   `{"teams": {ABBR: prob}, "spread": ..., "over_under": ...}`.
10. **`snapshot_current_odds()`** persists current odds for later line
    movement analysis.
11. **`generate_recommendations()`** (`betting/recommendations.py`) is
    called with all the above. Per game, it:
    - Runs `predict_fn` → `model_home_prob`.
    - Applies injury adjustment: `model_home_prob += home_impact - away_impact`,
      clipped to `[0.01, 0.99]`.
    - Looks up `market_home_prob` (Polymarket first, ESPN fallback).
    - **Shrinks**: `shrunken_home_prob = shrink_to_market(model, market, 0.6)`.
    - Computes `home_edge` and `away_edge` **against the shrunken
      probability** — not the raw model.
    - Picks the best side iff it's positive-EV AND passes the
      `MIN_BET_SIDE_PROB = 0.30` floor.
    - Sizes with quarter-Kelly, capped at 5% bankroll.
    - Assigns badge: `NO BET` for filtered rows, else STRONG/MODERATE/
      LEAN/SUSPECT by edge magnitude.
    - Generates an explanation that prefers signals agreeing with the
      bet side (and flags honest disagreement otherwise).
12. **`display_recommendations()`** renders the table with the shrunken
    probability in the Model column.
13. **`record_predictions()`** appends to `data/prediction_history.json`.

---

## 4. Feature pipeline

### 4.1 Source: `GameStats` joined with `Game`

`features/rolling.py::_load_game_stats_df()` joins the per-team-game box
score with the game header so every row has both the team's own stats
AND the game's home/away team IDs. This lets us derive:

- **`pts_against`**: `np.where(team_id == home_team_id, away_score, home_score)`.
  Vectorized — the naive `.apply(axis=1)` was ~100× slower on the full
  3.5k-game history.
- **`pts_for`**: alias of `pts`.
- **`net_rtg_game`**: `pts_for - pts_against`.

### 4.2 Rolling transformation

For each `(team_id)` group, for each stat column, for each window
`w ∈ (5, 10, 20)`:

```python
team_df[col].shift(1).rolling(window=w, min_periods=max(1, w//2)).mean()
```

**The `shift(1)` is critical** — it ensures the window only contains
games strictly before the one we're predicting. Without it, the training
target leaks into its own features and you'll see a completely
uninterpretable 70%+ accuracy that collapses on live data.

### 4.3 Four Factors + rest + opponent rebound context

`features/four_factors.py` adds eFG%, TOV%, ORB%, FT-rate per team-game.
These are re-rolled after joining in `builder.py`'s step 4 (they need
the same 5/10/20 windows with shift(1)).

`features/rest_days.py` computes `rest_days`, `is_back_to_back`,
`games_last_7`, `games_last_14` from each team's game schedule.

### 4.4 Pythagorean expectation

`_pythagorean_expectation(pf, pa) = pf^14 / (pf^14 + pa^14)` — Daryl
Morey's empirical basketball exponent. Two implementations:

- **Scalar** `_pythagorean_expectation(pf, pa)` — used in
  `build_prediction_features()` for single-row inference.
- **Vectorized** `_pythagorean_expectation_vec(pf_series, pa_series)` —
  used in `build_feature_matrix()` for the full history. Numerically
  stable via log-space softmax. **Tests (in-memory, see §8) confirm the
  two implementations produce identical output for valid, degenerate,
  and NaN inputs.**

The diff feature `diff_pyth_roll_w = home_pyth - away_pyth` at each
window became a top-10 feature in walk-forward permutation importance.

### 4.5 Pivoting to one row per game

`build_feature_matrix()` splits the long-format rolling DataFrame into
`home_stats` and `away_stats`, renames columns with `home_`/`away_`
prefixes, and merges on `game_id`. Then it computes:

- **Absolute features**: `home_elo`, `away_elo`, `elo_diff`, `elo_home_prob`
  (vectorized via `np.power(10, diff/400)`, not `.apply`).
- **Diff features**: `diff_{stat}_roll_{w}` for every stat and window.
- **Pythagorean diffs** for each window via the vectorized helper.
- **Rest diff**: `home_rest_days - away_rest_days`.

The final model-feature list is assembled in `builder.py::build_feature_matrix()`
step 8 as `model_features` and is stored to `feature_cols.joblib` by the
training path so inference can round-trip it.

### 4.6 Imputation strategy

- `build_feature_matrix()` drops rows where >30% of features are NaN
  (early-season games without enough history).
- Remaining NaNs are filled with column means.
- Those means are saved to `feature_means.joblib`.
- `build_prediction_features()` then uses those EXACT same means to
  impute any missing values at prediction time. This prevents the
  "train on filled, predict on zero" silent bias.

---

## 5. Model layer

### 5.1 Elo (`models/elo.py`)

538-style Elo:
- K-factor `ELO_K_FACTOR = 20`, home bonus `ELO_HOME_ADVANTAGE = 100`.
- MOV multiplier (Elo Auto-corrects for blowouts vs. Elo difference).
- Season carryover `ELO_CARRYOVER = 0.75` applied at the season boundary
  (`compute_all_elos()` detects season changes).
- `get_current_elos()` reads the latest `EloRating` per team.

`expected_score(elo_a, elo_b) = 1 / (1 + 10^((elo_b - elo_a) / 400))`
matches the Bradley-Terry formulation. In the feature builder this is
vectorized; for the live prediction it's called directly per-team.

### 5.2 Gradient boosting (`models/xgboost_model.py`)

- `HistGradientBoostingClassifier` from sklearn — the name "xgboost" in
  the filename/artifact is historical; we swapped for hist-GBM because
  it handles NaN natively and trains faster on this scale.
- Walk-forward validation with July 1 season boundaries
  (`walk_forward_validate()`).
- Permutation feature importance with `neg_log_loss` scoring is more
  honest than sklearn's built-in `feature_importances_`, which can be
  misleading for tree models.
- Persisted as `(estimator, feature_cols)` — the feature_cols are the
  source of truth for column order everywhere else.

### 5.3 Calibration (`models/calibration.py`)

**Default = isotonic** (not Platt/sigmoid). Reason: the home-win base
rate is ~54.6%, which is near enough to 50% that Platt's sigmoid
over-compresses the tails. On walk-forward folds, isotonic gives
calibrated ECE ≈ 0.00 while Platt was 0.03-0.04 with visible
tail-compression.

Uses `CalibratedClassifierCV(cv="prefit")` wrapped in `FrozenEstimator`
(required for sklearn ≥ 1.8, which removed direct prefit support).

### 5.4 Ensemble (`models/ensemble.py`)

**Log-odds (logit-space) blending**, not naive probability averaging:

```python
z = w_elo · logit(p_elo) + (1 - w_elo) · logit(p_gbm)
p_ensemble = sigmoid(z)
```

Why log-odds: averaging probabilities compresses extremes. If Elo says
90% and GBM says 70%, the probability average is 80%, but that implies
the models are much less confident than both actually are. The log-odds
average preserves the "both models agree it's a strong favorite"
signal.

**Weight learning** (`learn_ensemble_weight()`): grid-search
`w_elo ∈ [0.0, 0.1, ..., 1.0]` on the calibration fold, pick the one
minimizing `sklearn.metrics.log_loss`. Persisted to
`trained_models/ensemble_weight.joblib` and reloaded at prediction
time. Typical learned value: ~0.5.

---

## 6. Methodology rationales (read before changing anything)

### 6.1 Bayesian shrinkage toward market (`betting/shrinkage.py`)

**Problem**: without shrinkage, the system produced 12-14 "SUSPECT"
>15% edges on a 15-game slate every night. Most of those were phantom
disagreements — the model was off by 5-8 percentage points on games
where the market was actually right, and that 5-8 pp at a steep line
looks like a 30%+ edge.

**Fix**: treat the market as a **strong Bayesian prior**, the model as
a **likelihood**, and combine in log-odds space:

```
posterior_logit = (1 - λ) · model_logit + λ · market_logit
```

- `λ = 0` → pure model (phantom edges everywhere)
- `λ = 1` → pure market (bets nothing)
- `λ = 0.6` → default. Market is dominant; only decisive model
  conviction survives.

Edge is computed against `p_shrunk`, NOT `p_model`. This is the #1
source of the prior UX bug where "model 62% vs market 70%" would show
an edge that didn't reconcile arithmetically — now they reconcile
because both numbers come from the same post-shrinkage source.

**Do NOT remove or bypass this unless you have a calibrated reason to
trust the model more than the market.**

### 6.2 Asymmetric bet-side floor (`MIN_BET_SIDE_PROB = 0.30`)

**Problem**: even with shrinkage, a model that gives a team 15% and a
market that gives them 10% produces a positive edge, but betting a 15%
team is a lottery ticket. The model isn't really contradicting the
market — it's just noise on a tail.

**Fix**: refuse to bet a side whose shrunken probability is below 0.30,
regardless of edge math. This is standard quant practice and kills
almost all residual SUSPECT rows.

Applied AFTER shrinkage — so a team the raw model favored at 45% but
shrunk to 25% won't pass the floor. That's correct: if the market
pulled us back, the market's view is that we don't have conviction.

### 6.3 Isotonic > Platt for NBA home-win calibration

See §5.3. On a ~54% base rate, Platt's sigmoid over-compresses both
tails. Isotonic is non-parametric and handles near-balanced problems
better. The ECE delta is small in absolute terms (~0.03 → ~0.005) but
matters a lot for high-confidence bets.

### 6.4 Log-odds ensemble vs probability average

See §5.4. Probability averaging loses tail information; log-odds
averaging preserves it. This matters most when Elo and GBM agree on
direction but disagree on magnitude — the log-odds average respects
their joint conviction.

### 6.5 ET timezone for "today"

**Problem**: system was in Vienna → `date.today()` was a day ahead of
the NBA scheduling day → we filtered to empty → showed "next game day"
incorrectly.

**Fix**: `_today_et()` via `zoneinfo.ZoneInfo("America/New_York")`.
Also dropped the live `ScoreBoard()` endpoint, which caches stale
prior-day results for hours after midnight ET, in favor of
`ScoreboardV3` with an explicit date string.

### 6.6 Backtest modes: Elo proxy vs real odds, raw model vs live strategy

`betting/backtest.py` runs walk-forward validation and simulates
betting. Two orthogonal flags control what's being measured:

- `apply_live_strategy` (default `None`) — if on, the bet loop applies
  the same Bayesian shrinkage (`shrink_to_market`) and asymmetric
  bet-side floor (`MIN_BET_SIDE_PROB`) that `predict` uses live. If
  off (`--raw-model` on the CLI), it uses the raw post-ensemble model
  probability against the market proxy. When `None`, the default
  **follows `use_real_odds`**: off with the Elo proxy (shrinking toward
  Elo is a null-op/dampener and would hide raw model lift), on with
  real odds (live-equivalent simulation). Pass the explicit flag to
  override either direction.
- `use_real_odds` (default `False`) — if on, look up each test game
  in the `odds_snapshots` table via `get_closing_line()` and use the
  captured Polymarket/ESPN price as the market. If no snapshot exists
  we fall through to Elo proxy and count it as a miss in
  `real_odds_coverage`. Off means Elo is the market everywhere.

**Why Elo as the market proxy** (when real odds aren't available): we
don't have historical Polymarket prices before the snapshot collector
started. Elo is the next-best benchmark because it uses zero
contemporaneous information beyond game results. The tradeoff is that
Elo is much weaker than a real efficient market, so Elo-proxy ROI is
systematically **overstated** vs. live behavior.

Combinations and what they tell you:

| `apply_live_strategy` | `use_real_odds` | What it measures |
|---|---|---|
| False | False | Raw model quality vs a weak proxy — upper bound on model skill. **(Default when `--real-odds` is off.)** |
| True  | False | Strategy simulation with Elo as market — directional but optimistic; pass `--live-strategy` explicitly. |
| False | True  | Raw model quality vs real books — pass `--raw-model --real-odds` for the cleanest "can the model beat the book" read. |
| True  | True  | True live-equivalent simulation — the number closest to forecasting live ROI. **(Default when `--real-odds` is on.)** |

Until the snapshot collector has accumulated months of data, coverage
on the real-odds path is low. The summary dict now includes
`real_odds_hits`, `real_odds_misses`, and `real_odds_coverage` so the
CLI table can show what fraction of test games actually used real
prices.

---

## 7. Config knobs (`nba_betting/config.py`)

Everything worth tuning is in one file. The most impactful knobs:

| Knob | Default | Effect |
|------|---------|--------|
| `MARKET_SHRINKAGE_LAMBDA` | `0.6` | Higher → trust market more, fewer bets |
| `MIN_BET_SIDE_PROB` | `0.30` | Higher → refuse more underdog bets |
| `MIN_EDGE_THRESHOLD` | `0.02` | Minimum edge to bet |
| `SUSPICIOUS_EDGE_THRESHOLD` | `0.15` | Above this → SUSPECT badge |
| `KELLY_FRACTION` | `0.25` | Quarter-Kelly (conservative) |
| `MAX_BET_PCT` | `0.05` | 5% of bankroll per bet |
| `MAX_EXPOSURE_PCT` | `0.25` | 25% total simultaneous exposure |
| `ELO_K_FACTOR` | `20.0` | Elo update speed |
| `ELO_HOME_ADVANTAGE` | `100.0` | Home bonus in Elo points |
| `ELO_CARRYOVER` | `0.75` | Season-to-season Elo persistence |
| `NBA_API_DELAY_SECONDS` | `2.5` | NBA.com rate limit |
| `ESPN_API_DELAY_SECONDS` | `1.5` | ESPN rate limit |

---

## 8. How to verify the system after a change

Run in order — each step gates the next:

```bash
# 1. Feature pipeline builds cleanly
.venv/bin/python3 -c "
from nba_betting.features.builder import build_feature_matrix
X, y = build_feature_matrix()
print(X.shape, 'NaN?', X.isna().any().any())
"
# Expect: ~(3500+, 60), NaN? False

# 2. Scalar and vectorized Pythagorean agree
.venv/bin/python3 -c "
from nba_betting.features.builder import _pythagorean_expectation as s, _pythagorean_expectation_vec as v
import pandas as pd, numpy as np
pf = pd.Series([110, 105, 100, 0, np.nan, 90, 100])
pa = pd.Series([100, 110, 100, 100, 110, 0, np.nan])
assert all(abs(v(pf, pa)[i] - s(pf[i], pa[i])) < 1e-9 for i in range(len(pf)))
print('OK')
"

# 3. Shrinkage math reconciles in the UI
.venv/bin/python3 -m nba_betting predict | tail -40
# Expect: Model% and Edge% are algebraically consistent with Market%.

# 4. No SUSPECT badges on NO BET rows
# (check table output — NO BET rows should show NO BET, not SUSPECT)

# 5. Walk-forward still in the expected range
.venv/bin/python3 -m nba_betting train
# Expect: WF accuracy 63-65%, Brier ~0.223, calibrated ECE < 0.02

# 6. End-to-end diagnose
.venv/bin/python3 -m nba_betting diagnose

# 7. Unit tests for the hardening pass (shrinkage, drivers, spreads,
#    backtest defaults, migration idempotence).
.venv/bin/python3 -m pytest tests/test_new_features.py -v
# Expect: 16 passed in < 2s.

# 8. Check how much historical data has accumulated for the new
#    injury/odds features. < 30 distinct days = don't bother retraining
#    with those features yet.
.venv/bin/python3 -m nba_betting readiness-status
```

---

## 9. Rebuilding from scratch — the short version

If this repo were gone and you had to rebuild it:

1. **Skeleton**: `pip install sqlalchemy pandas numpy scikit-learn
   nba-api httpx typer rich fastapi uvicorn joblib`. Create the package
   with `config.py`, `db/models.py`, `db/session.py` first.
2. **Data layer**: write `data/nba_stats.py` (`fetch_todays_games`,
   `fetch_upcoming_games`, `sync_games` with ET timezone). Use
   ScoreboardV3, never live ScoreBoard. Write `data/polymarket.py` (Gamma
   + CLOB clients). Write `data/espn.py` for injuries/odds/rosters.
3. **DB**: define Team, Game, GameStats, EloRating, PlayerStat,
   OddsSnapshot. Keep team IDs = NBA.com IDs (not auto-increment).
4. **Elo**: implement `models/elo.py` with MOV + home + carryover.
   Populate `EloRating` via `compute_all_elos()` that iterates games
   chronologically.
5. **Features**:
   - `features/rolling.py`: shift(1) + rolling windows. Derive
     `pts_against` with vectorized `np.where`.
   - `features/four_factors.py`: Dean Oliver's Four Factors.
   - `features/rest_days.py`: rest/b2b/7-day/14-day game counts.
   - `features/builder.py`: the pivot + diff assembler. Include
     Pythagorean (scalar + vectorized). Save feature_means.
6. **Model**:
   - `models/xgboost_model.py`: HistGradientBoostingClassifier wrapper.
   - `models/calibration.py`: CalibratedClassifierCV(cv="prefit") with
     FrozenEstimator and `method="isotonic"`.
   - `models/ensemble.py`: log-odds blend, grid-search weight on the
     calibration fold.
7. **Betting**:
   - `betting/edge.py`: compute_edge, badges.
   - `betting/kelly.py`: quarter-Kelly with 5% cap.
   - **`betting/shrinkage.py`: the logit-space prior update.**
   - `betting/recommendations.py`: the orchestrator. Predict → inject
     → shrink → edge → floor → size → explain.
8. **Explanations**: template-based, prefer signals agreeing with bet.
9. **Display**: Rich console + FastAPI + static `frontend/index.html`.
   In both, show the SHRUNKEN probability in the Model column.
10. **CLI**: Typer app with `predict/train/sync/backtest/diagnose/…`.

**Critical invariants to preserve**:
- `shift(1)` in every rolling computation.
- Same columns in same order between training and prediction.
- Edge computed against `p_shrunk`, not `p_model`.
- UI shows `p_shrunk` in the Model column.
- `MIN_BET_SIDE_PROB = 0.30` floor applied to `p_shrunk`, not `p_model`.
- Isotonic (not Platt) calibration.
- Log-odds (not probability-average) ensemble.
- ET timezone for "today", ScoreboardV3 not live ScoreBoard.
- `load_model()` cached in `routes.py` (not called 3 times).
- Driver attribution runs on the **base GBM**, not the calibrated
  wrapper, and is computed **lazily** (only for `bet_side != "NO BET"`).
- Backtest `apply_live_strategy` defaults to `None` and resolves from
  `use_real_odds` — never hard-code `True`, or Elo-proxy backtests
  become self-dampening.
- Snapshot `game_date` is parsed from `game_time_utc[:10]`, not set to
  `date.today()`, so upcoming-day snapshots join to the right game.

---

## 10. Known limitations / future work

### 10.1 Addressed (2026-04)

All five previously-listed limitations now have implementations. The
flags below are what landed and where to look:

- **Historical market odds** — new `snapshot-odds` CLI command
  (`nba_betting/cli.py` → `snapshot_odds`) captures a Polymarket + ESPN
  snapshot on demand; schedule it every 30 min via cron. Snapshots land
  in `odds_snapshots` (now with a `game_id` FK, added via an additive
  migration in `nba_betting/db/session.py`). Backtest has a new
  `--real-odds` mode (`run_backtest(..., use_real_odds=True)` in
  `nba_betting/betting/backtest.py`) that joins snapshots back via
  `get_closing_line()` in `nba_betting/data/odds_tracker.py`. Falls
  through to Elo proxy for dates with no coverage and reports a
  `real_odds_coverage` ratio so the user knows how much of the bet set
  actually used real prices.
- **Historical injury archive** — new `historical_injuries` table in
  `nba_betting/db/models.py`. Every `injury sync` (and every `predict`,
  which triggers a sync) now calls `persist_historical_injuries()` in
  `nba_betting/data/injuries.py`, which idempotently upserts a dated
  snapshot. `build_feature_matrix` now attaches three features
  (`home_injury_impact_out`, `away_injury_impact_out`,
  `injury_impact_diff`) via `_attach_injury_features()`. Games older
  than the snapshot collector get 0 (treated as "unknown, average") so
  training stays stable; coverage grows organically forward in time.
- **Spread / total modeling** — new `nba_betting/models/spreads_totals.py`
  trains two `HistGradientBoostingRegressor` heads (margin, total)
  from the same feature matrix, saved as `spread_regressor.joblib` and
  `total_regressor.joblib`. Predict-time path in `cli.predict` and
  `api/routes.py` loads them and calls
  `generate_spread_total_picks()` which compares the model against the
  market line with a 1.5-pt spread floor and 2.5-pt total floor.
  `BetRecommendation` carries `predicted_spread`, `predicted_total`,
  `spread_pick`, `spread_edge`, `total_pick`, `total_edge`; the CLI
  display and the `/predictions/today` JSON both surface them.
- **SHAP-style prediction drivers** — new
  `nba_betting/models/drivers.py` with
  `compute_prediction_drivers(model, feat_row, feature_means, top_k)`.
  The approach is a leave-one-out-to-mean attribution (batched into
  one `predict_proba` call, ~10 ms/game) rather than a real SHAP lib —
  deterministic, no extra dependency, and agrees with
  `shap.TreeExplainer` on the top-3 features ~85% of the time.
  Drivers are attached to `BetRecommendation.drivers` and consumed by
  `_driver_from_attribution()` in `nba_betting/betting/explanations.py`,
  which cites the strongest agreeing driver (falling back to the old
  rolling-stat heuristic if no drivers are available).
- **Backtest applies live strategy** — `run_backtest` now imports
  `shrink_to_market` and `MIN_BET_SIDE_PROB` and applies both. The
  default is `apply_live_strategy=None`, which resolves against
  `use_real_odds`: off with the Elo proxy (pure model benchmark), on
  with real odds (live-equivalent simulation). The old raw-model-vs-
  Elo-proxy mode is still available via `--raw-model`. Combined with
  `--real-odds`, this gives four backtest configurations along two
  orthogonal axes.

### 10.1a Hardening pass (2026-04)

Follow-on fixes to the 10.1 items after a post-implementation audit:

- **Backtest defaults corrected** — `apply_live_strategy` was
  originally `True` by default, which meant the Elo-proxy backtest was
  shrinking the model toward Elo (a null-op that just dampened raw
  lift). It's now `None` and resolves from `use_real_odds` as described
  above in §6.6 — preserves live-equivalent simulation when real odds
  are available while giving a clean raw-model benchmark otherwise.
- **Snapshot game_date** — `snapshot_current_odds()` now parses
  `game_time_utc[:10]` when `fetch_upcoming_games()` returns tomorrow's
  slate, so snapshots are filed under the actual game date and
  `get_closing_line()` joins cleanly at backtest time.
- **Driver attribution on base GBM** — drivers are now computed
  against the uncalibrated base GBM (`load_model()[0]`) rather than
  the calibrated wrapper. Isotonic calibration is monotonic, so it
  preserves driver ranking but distorts LOO-delta magnitudes; running
  attribution on the raw tree ensemble gives cleaner "model split on
  this" signal.
- **Lazy driver attribution** — `generate_recommendations` now
  computes drivers only when `bet_side != "NO BET"`. The closure in
  `cli.predict` / `api/routes.py` stashes `feat_row` into a
  `driver_contexts` dict keyed by `(home_id, away_id)`, and
  `generate_recommendations` calls `compute_prediction_drivers` lazily
  after the edge gate. Saves ~70% of the attribution cost on a typical
  night.
- **humanize_feature coverage extended** — a single `_DIRECT_LABELS`
  dict in `models/drivers.py` now covers injury, player-impact, and
  line-movement columns (`injury_impact_diff`, `diff_missing_minutes_pct`,
  `spread_movement`, `odds_disagreement`, etc.) so the explanation
  sentence never renders raw snake_case.
- **Driver noise floor** — `_driver_from_attribution()` in
  `betting/explanations.py` now requires `|delta| >= 0.005` before
  citing a driver, so a 0.1pp LOO jitter can't masquerade as the
  "primary driver".
- **Dashboard renders picks + drivers** — `frontend/index.html` now
  shows a secondary pick row (spread/total calls with model vs market
  gap) and a driver-chips row (top-3 LOO attributions) under each
  game, matching what the console output and JSON API already emit.
- **Test coverage** — new `tests/test_new_features.py` (16 tests, all
  pass) guards shrinkage invariants, the humanize_feature label map,
  the spread sign convention, driver ordering, the backtest default
  resolution, and additive-migration idempotence. See §8 for
  invocation.
- **Readiness-status command** — `python3 -m nba_betting readiness-
  status` (in `nba_betting/cli.py` → `readiness_status`) counts
  distinct `snapshot_date`s across `HistoricalInjury` and
  `OddsSnapshot` and tiers each stream as cold / partial / ready
  (cutoff: 30 days). Nudges the user toward retraining once both
  streams have enough variation.

### 10.2 Remaining caveats

- **Real-odds coverage is bootstrapping** — `snapshot-odds` only
  accumulates data going forward, so `--real-odds` backtests will have
  low coverage until a cron has been running for at least a few weeks
  of NBA games. Until then the Elo proxy remains the default.
- **Injury features are likewise forward-looking** — the historical
  injury archive is only populated from the day `injury sync` starts
  running. Old training games have `injury_impact_*` = 0; the model
  learns to treat that as "unknown, average" but the feature will only
  become truly predictive once it has a season or two of real data
  under it.
- **Driver attribution is not exact Shapley** — it ignores feature
  interactions, so on trees with strong split dependencies the top
  driver can be slightly off. Good enough to cite in a sentence; don't
  use these numbers for anything load-bearing.
- **Spread/total picks do not fund a Kelly bet** — the regression
  heads emit picks but there is no market-probability analog (books
  quote vig-adjusted odds, not implied point-spread probability), so
  the recommendations layer doesn't compute a Kelly stake. Treat the
  spread/total picks as information bets only.
- **Still no props / player-level modeling.** Out of scope for now.
