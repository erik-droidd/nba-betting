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
                    │ 1.5s rate-limit   │                    │ 0.8s rate-limit
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
             │  (~92 diff-features: rolling stats, SOS-adjusted      │
             │   net rating, pace/poss, EWM, off/def Elo — all       │
             │   computed with shift(1) to prevent temporal leakage) │
             └───────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
             ┌───────────────────────────────────────────────────────┐
  MODELS     │  Elo (+ off/def split) + HistGradientBoosting        │
             │          (calibrated isotonic, per-fold grid search)  │
             │          │                                            │
             │          └─> meta-learner → log-odds ensemble fallback│
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
             │  size  = signal-dependent Kelly × slate portfolio opt  │
             │          (Gaussian copula correlation, SLSQP solver,  │
             │           fallback to per-bet quarter-Kelly if 1 bet) │
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
│   ├── nba_stats.py            — NBA.com client (ScoreboardV3 + LeagueGameLog).
│   │                              ⚠ Uses ET timezone via zoneinfo for "today".
│   │                              ⚠ Never uses live ScoreBoard() — it caches
│   │                                 stale prior-day data for hours after rollover.
│   │                              ⚠ fetch_season_games() unions Regular Season
│   │                                 + Play-In + Playoffs. Pre-fix the call
│   │                                 only fetched Regular Season, so post-
│   │                                 April predictions never resolved.
│   ├── polymarket.py           — Gamma + CLOB clients. Filters closed markets
│   │                              and extreme (<1%, >99%) prices. Fuzzy
│   │                              substring fallback in _name_to_abbr()
│   │                              catches non-standard title formats.
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
│   ├── lineups.py              — Pre-game probable starters via ESPN
│   │                              scoreboard (~30–90 min before tip).
│   │                              fetch_probable_starters() returns
│   │                              {team_abbr: [player_name]}; empty if
│   │                              ESPN hasn't published yet. Called lazily
│   │                              in cli.predict — non-critical, failure
│   │                              is silently swallowed.
│   │                              apply_lineup_bumps() raises impact rating
│   │                              for starter-tier players absent from the
│   │                              published lineup (surprise DNPs).
│   ├── player_stats.py         — Roster + depth chart sync into PlayerStat.
│   └── odds_tracker.py         — snapshot_current_odds / get_line_movement
│                                 (opening-vs-current spread & prob).
│                                 Auto-deduplicates: skips snapshot if
│                                 prices moved < 0.5% within a 4h window.
│
├── features/
│   ├── rolling.py              — Per-team-game rolling means (5/10/20).
│   │                              Uses shift(1) to exclude the current game
│   │                              (NO LEAKAGE). Computes pts_against /
│   │                              net_rtg_game with vectorized np.where.
│   │                              Also attaches opp_elo per row (SOS),
│   │                              poss (possessions ≈ FGA+0.44FTA+TOV),
│   │                              adj_net_rtg per window (SOS-adjusted),
│   │                              and EWM stats (halflife=10, shift(1)).
│   ├── four_factors.py         — eFG%, TOV%, ORB%, FT-rate (Dean Oliver).
│   │                              opp_dreb now vectorized via groupby
│   │                              transform (no per-row apply).
│   ├── rest_days.py            — rest_days, is_b2b, games_last_7/14.
│   ├── player_impact.py        — ESPN-driven player features: WAT score,
│   │                              missing_minutes_pct, star_out flag.
│   │                              compute_player_impact_features() is called
│   │                              at predict time (cli.py + api/routes.py)
│   │                              and injects live values into extra_features.
│   │                              Training uses 0.0 for all historical rows
│   │                              (same forward-accumulating convention as
│   │                              injury_impact); the model learns the signal
│   │                              once live data accumulates.
│   └── builder.py              — THE assembler. Two entry points:
│                                   build_feature_matrix(recompute_elos=True)
│                                   for training (flag skips Elo rebuild when
│                                   already done upstream),
│                                   build_prediction_features() for one game.
│                                 Both must produce the SAME columns in the
│                                 SAME order or prediction-time imputation
│                                 will silently feed garbage to the model.
│                                 Four Factors rolling uses groupby.transform
│                                 (vectorized, ~2× faster than per-team loop).
│
├── models/
│   ├── elo.py                  — 538-style Elo with home advantage,
│   │                              MOV multiplier (× opponent-strength
│   │                              sigmoid), season carryover, and a
│   │                              parallel off/def Elo split.
│   │                              get_current_elos() and
│   │                              get_current_off_def_elos() read the
│   │                              latest EloRating rows per team.
│   ├── xgboost_model.py        — HistGradientBoostingClassifier wrapper
│   │                              (name kept for historical reasons).
│   │                              Saves/loads (estimator, feature_cols).
│   │                              Also persists feature_means for NaN
│   │                              imputation at prediction time.
│   │                              Per-fold hyperparameter grid search
│   │                              (max_depth/lr/iter/l2); best params
│   │                              saved to best_params.joblib.
│   │                              Mtime-keyed in-process model cache.
│   ├── calibration.py          — Isotonic calibration via
│   │                              CalibratedClassifierCV(cv="prefit").
│   │                              Defaults to 'isotonic' — see §6.
│   │                              Uses FrozenEstimator for sklearn ≥ 1.8.
│   │                              Mtime-keyed in-process cache.
│   ├── stacking.py             — Logistic regression meta-learner trained
│   │                              on out-of-fold [elo_logit, gbm_logit,
│   │                              |disagreement|]. Saved to
│   │                              ensemble_meta.joblib; absent = fallback
│   │                              to log-odds blend.
│   └── ensemble.py             — blend_predictions(): prefers meta-learner
│                                 when ensemble_meta.joblib is present,
│                                 falls back to log-odds (logit-space)
│                                 blend with a grid-searched weight
│                                 persisted to ensemble_weight.joblib.
│
├── utils/
│   └── math.py                 — Shared logit/sigmoid (array + scalar).
│                                 Single source of truth imported by
│                                 ensemble.py, shrinkage.py, stacking.py.
│
├── betting/
│   ├── edge.py                 — compute_edge, is_positive_ev, and
│   │                              confidence_badge (STRONG/MODERATE/LEAN/
│   │                              SUSPECT thresholds).
│   ├── kelly.py                — kelly_fraction + compute_bet_size.
│   │                              signal_dependent_lambda() scales the
│   │                              Kelly fraction by multiplicative edge
│   │                              (prob/market - 1), CLV, and disagreement
│   │                              factors (clamped 0.25×–1.25× base).
│   ├── portfolio.py            — optimize_slate(): slate-level Kelly via
│   │                              scipy SLSQP with a Gaussian copula
│   │                              correlation model. Wired into
│   │                              recommendations.generate_recommendations.
│   │                              Falls back to haircut per-bet Kelly for
│   │                              1 bet or optimizer failure.
│   │                              build_simple_correlation() provides
│   │                              the default same-day +0.05 ρ matrix.
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
│                                 for prediction_history.json. Stores
│                                 market_home_prob (pick-time price) for CLV;
│                                 update_results() fills closing_market_prob
│                                 and compute_clv() sets the log-odds CLV
│                                 (close - bet), gated on ≥2 snapshots.
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

> ⚠ **Known limitation — predict-time one-game lag (not yet fixed).**
> `build_prediction_features` reads each team's *latest stored row*
> (`_get_latest_stats` → `iloc[-1]`), whose rolling value is the `shift(1)`
> window — i.e. it **excludes that team's most recent completed game**. For
> the *upcoming* game the correct window is "the last `w` completed games,
> *including* the most recent." So live predictions use form that is one
> game stale (confirmed: a team's `net_rtg_game_roll_5` read 16.8 at predict
> when the last-5-including-latest mean was 21.4 — 27% off at `w=5`). A
> correct fix recomputes the rolling **including** the latest game (e.g. by
> appending a synthetic next-game row per team and re-running the `shift(1)`
> rolling, so it works uniformly across the derived stats — SOS-adj, EWM,
> venue splits). Deferred deliberately: it rewrites the predict feature path
> and changes every live feature value, so it needs a predict-path backtest
> harness to validate rather than a blind change.

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

### 4.5 SOS-adjusted rolling stats

`compute_rolling_features()` now also computes `opp_elo_roll_{5,10,20}`:
the rolling mean of the pre-game Elo of each opponent faced. This is
merged in from the `EloRating` table using the `game_id` + `opponent_id`
join so the current game's Elo is never included (shift already handles
the team's own row; the opponent join is done on the pre-game row).

`adj_net_rtg_roll_{w} = net_rtg / clip(1 + (opp_elo - 1500)/500, 0.5, 1.5)`
normalises net rating relative to opponent strength. A 15-point win over
a 1650-Elo team is worth more than the same margin over a 1350-Elo team.

`diff_adj_net_rtg_roll_{w}` and `diff_opp_elo_roll_{w}` are added as
diff features in `build_feature_matrix()`.

### 4.6 Pace / possessions features

`compute_rolling_features()` computes per-team-game possessions:
```
poss = FGA + 0.44·FTA + TOV - OREB
```
(clipped at 60 as a sanity floor). This is rolled over 5/10/20 windows
with shift(1). `matchup_pace_{w} = (home_poss_roll_{w} + away_poss_roll_{w}) / 2`
captures the expected tempo of a specific matchup.

### 4.7 EWM rolling stats

`rolling_ewm(series, halflife=10.0)` applies `shift(1).ewm(halflife=10).mean()`
so recent form is weighted exponentially more than older games. Two EWM
stats are currently added: `net_rtg_game_ewm_10` and `plus_minus_ewm_10`.
The diff feature `diff_net_rtg_game_ewm_10` ranked #2 in permutation
importance after training on the expanded feature set.

### 4.8 Off/def Elo features

`EloRating` now stores `elo_off_before` and `elo_def_before` per game
(populated by `compute_all_elos()` which runs the parallel off/def split
alongside the aggregate Elo). `build_feature_matrix()` joins these and
exposes:
- `home_elo_off`, `home_elo_def`, `away_elo_off`, `away_elo_def`
- `elo_off_diff = home_elo_off - away_elo_off`
- `elo_def_diff = home_elo_def - away_elo_def`
- `home_off_vs_away_def = home_elo_off - away_elo_def` (matchup quality)
- `away_off_vs_home_def = away_elo_off - home_elo_def`

Columns fill from the aggregate Elo when off/def rows are absent (e.g.,
before the first backfill run), so training is stable across schema versions.

### 4.9 Pivoting to one row per game

`build_feature_matrix()` splits the long-format rolling DataFrame into
`home_stats` and `away_stats`, renames columns with `home_`/`away_`
prefixes, and merges on `game_id`. Then it computes:

- **Absolute features**: `home_elo`, `away_elo`, `elo_diff`, `elo_home_prob`
  (vectorized via `np.power(10, diff/400)`, not `.apply`).
- **Diff features**: `diff_{stat}_roll_{w}` for every stat and window.
- **Pythagorean diffs** for each window via the vectorized helper.
- **Rest diff**: `home_rest_days - away_rest_days`.
- **SOS / pace / EWM diffs**: as described in §4.5–4.7.
- **Off/def Elo features**: as described in §4.8.

The final model-feature list is assembled in `builder.py::build_feature_matrix()`
step 8 as `model_features` and is stored to `feature_cols.joblib` by the
training path so inference can round-trip it.

### 4.10 Player impact features

`features/player_impact.py::compute_player_impact_features()` produces 6
features from the current ESPN injury list + `PlayerStat` roster:

- `home_missing_minutes_pct` / `away_missing_minutes_pct`: fraction of
  each team's typical minutes unavailable due to injury/rest.
- `home_star_out` / `away_star_out`: 1.0 if any player averaging ≥ 30 mpg
  is at least 50% likely to miss (Out/Doubtful).
- `diff_missing_minutes_pct`: home minus away missing-minutes fraction.
- `diff_available_talent`: home minus away Weighted Available Talent
  (WAT = Σ (pts + ast + reb) × P(available) per player).

**Training**: all 6 features are set to 0.0 for every historical game
(no per-game player availability archive exists). The model learns to
treat 0 as "unknown, average" — the same forward-accumulating convention
as `injury_impact_out`. Signal accumulates as live predictions write real
values and the model is retrained.

**Prediction time**: `cli.py::_xgb_predict` and `api/routes.py::_predict`
call `compute_player_impact_features()` and inject the results into
`extra_features` before `build_prediction_features()`. The `injuries`
list is captured from the outer closure scope and already includes
ESPN sync + lineup bump adjustments.

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
- K-factor `ELO_K_FACTOR = 20`, home bonus `ELO_HOME_ADVANTAGE = 40`.
  **Not 100** — the classic 538 value implies a 64% home-win prob at
  equal Elo, but the modern NBA home edge has collapsed to ~55% (our
  3-season sample: 54.7% / 54.6% / 55.6%). An end-to-end sweep that
  recomputes both the ratings and the predictions at each value bottoms
  out at ~40 on Brier, log-loss, **and** accuracy simultaneously, and
  makes the mean Elo prediction match the empirical base rate. See §6.10.
- MOV multiplier dampened by an **opponent-strength sigmoid**:
  `opp_factor = 1 / (1 + exp(-(elo_loser - elo_winner) / 200))`. A 30-point
  blowout over a weak opponent inflates ratings less than the same margin
  over a contender.
- Season carryover `ELO_CARRYOVER = 0.75` applied at the season boundary
  (`compute_all_elos()` detects season changes).
- `get_current_elos()` reads the latest `EloRating` per team.

**Off/def Elo split** (`update_off_def_elo()`): alongside the aggregate
Elo, each team's offensive Elo is updated from points scored vs the
opponent's defensive Elo (and vice versa). The residual is normalised by
a 12-point scale (`PTS_PER_ELO = 0.05`, `k_split = k × 0.5`). This lets
the model distinguish a high-scoring team with a porous defence from a
balanced team with the same aggregate Elo.

`get_current_off_def_elos()` returns `{team_id: (elo_off, elo_def)}` and
is called by `api/routes.py` for live predictions.

`expected_score(elo_a, elo_b) = 1 / (1 + 10^((elo_b - elo_a) / 400))`
matches the Bradley-Terry formulation. In the feature builder this is
vectorized; for the live prediction it's called directly per-team.

### 5.2 Gradient boosting (`models/xgboost_model.py`)

- `HistGradientBoostingClassifier` from sklearn — the name "xgboost" in
  the filename/artifact is historical; we swapped for hist-GBM because
  it handles NaN natively and trains faster on this scale.
- Walk-forward validation with July 1 season boundaries
  (`walk_forward_validate(n_splits, return_oof)`). When `return_oof=True`
  (used by the `train` CLI), per-fold OOF arrays
  (`oof_elo_probs`, `oof_gbm_cal_probs`, `oof_y_true`) are returned for
  meta-learner fitting.
- **Per-fold hyperparameter grid search** (`search_hyperparams()`): each
  WF fold runs an 8-combo grid over `max_depth ∈ {4,5}`,
  `learning_rate ∈ {0.03,0.05}`, `max_iter ∈ {300,500}`, and
  `l2_regularization ∈ {0,0.1}`, picking the combo with best held-out
  log-loss on a 15% calibration slice. Best params saved to
  `trained_models/best_params.joblib` and reloaded for final training.
- **Temporal early-stopping split** (`train_model()`): when `_date` is
  present in X, uses a two-stage approach — fit a temporary model on the
  first 85% chronologically to find `n_iter_`, then retrain on all data
  with `early_stopping=False, max_iter=n_iter_`. Avoids pulling future
  games into the early-stopping validation set.
- **Mtime-keyed in-process cache**: `load_model()` and `load_feature_means()`
  skip disk reads if the artifact hasn't changed since last load — saves
  ~50ms per repeated API call.
- Permutation feature importance with `neg_log_loss` scoring is more
  honest than sklearn's built-in `feature_importances_`, which can be
  misleading for tree models.
- Persisted as `(estimator, feature_cols)` — the feature_cols are the
  source of truth for column order everywhere else.

### 5.3 Calibration (`models/calibration.py`)

**Default = isotonic** (not Platt/sigmoid) for the **final** model,
calibrated on the full ~772-game tail slice: the home-win base rate is
~54.6%, near enough to 50% that Platt's sigmoid over-compresses the tails.

⚠ **Honest ECE is ~0.03, not ~0.00.** A prior version of this doc (and the
`train` output) reported "calibrated ECE ≈ 0.00" — that was an **in-sample
artifact**: isotonic regression interpolates its own fit points, so
scoring it on the same slice it was fit on always looks near-perfect.
On honest **out-of-fold** data the calibrated GBM's ECE is ~0.032 and
Elo's is ~0.058 (now what `train` prints). The fix lives in §6.11.

⚠ **Per-fold isotonic on small slices over-extremizes.** The final
full-slice isotonic is fine, but the *per-fold* calibration inside
`walk_forward_validate` fits isotonic on ~250-game slices, which pushes
some probabilities to 0/1 and actually **loses** to sigmoid out-of-fold
(per-fold OOF: isotonic acc 62.2% / ll 0.669 vs sigmoid 64.7% / 0.631).
This only affects the OOF arrays that feed the meta-learner; production's
final calibration is unaffected. Revisit if the meta-learner is ever
wired into `predict`.

Uses `CalibratedClassifierCV(cv="prefit")` wrapped in `FrozenEstimator`
(required for sklearn ≥ 1.8, which removed direct prefit support).

### 5.4 Ensemble (`models/ensemble.py` + `models/stacking.py`)

**Production path — static log-odds (logit-space) blend** via
`ensemble_predict()`:

```python
z = w_elo · logit(p_elo) + (1 - w_elo) · logit(p_gbm)
p_ensemble = sigmoid(z)
```

This is what `cli.predict`, `api/routes.py`, and `backtest.py` actually
call. Why log-odds: averaging probabilities compresses extremes. If Elo
says 90% and GBM says 70%, the probability average is 80%, but that
implies the models are much less confident than both actually are. The
log-odds average preserves the "both models agree it's a strong
favorite" signal.

**Weight learning** (`learn_ensemble_weight()`): grid-search
`w_elo ∈ [0.0, 0.1, ..., 1.0]`, pick the one minimizing
`sklearn.metrics.log_loss`, **on the walk-forward out-of-fold
predictions** (not the isotonic calibration slice — see §6.11).
Persisted to `trained_models/ensemble_weight.joblib` and reloaded at
prediction time. Typical learned value: **~0.9 (Elo-leaning)** — on
honest out-of-fold data the calibrated GBM is the *weaker* model
(acc ~62-64% vs Elo's ~67% once Elo's home edge is calibrated, §6.10),
so it earns only ~10% of the blend. It still helps marginally: the
blend's Brier/log-loss beat pure Elo's.

> **Can the GBM be made to pull more weight?** Investigated thoroughly
> (issue #23): regularization (shallower/fewer trees, higher `l2`,
> `min_samples_leaf`), feature pruning to the top-k by permutation
> importance, dropping the Elo features for orthogonality, and sigmoid
> vs isotonic per-fold calibration. **None robustly beat the current
> blend.** Every candidate landed within ~1 Brier SE (±0.006) and traded
> the small early walk-forward fold for the recent one; feature pruning's
> apparent gain was sensitive to the selection slice (sometimes +0.001,
> sometimes −0.001 log-loss at the production weight), i.e. noise. The
> honest conclusion: with the current feature set the GBM is a thin
> complement and is correctly down-weighted. The real remaining lever is
> **feature maturation** — the injury / line-movement / player-availability
> features are still ~0 for historical games (§4.10) and only start
> carrying signal once enough live-season data accumulates (see
> `readiness-status`). Re-open the GBM question after a `READY` tier, not
> before. Do not ship within-noise model-tuning changes to a live model.

**Trained-but-not-wired — stacked meta-learner** (`models/stacking.py`):
`train` also fits a logistic-regression meta-learner on the same OOF
predictions (`[logit(p_elo), logit(p_gbm), |disagreement|]`) and saves it
to `ensemble_meta.joblib` when ≥ 200 OOF games exist. It's reachable via
`blend_predictions()`, **but no production caller currently uses
`blend_predictions`** — every live path calls `ensemble_predict`. On the
current data the fitted meta-learner is anyway Elo-dominated (coef ratio
~37:1), i.e. it independently rediscovers the ~0.9 static weight, so the
two paths agree. Wiring `blend_predictions` into `predict` is a candidate
future change (game-dependent weighting), but only worthwhile once the
meta-learner is trained on cleaner GBM inputs (its OOF GBM probs use
per-fold isotonic, which over-extremizes on small slices — §6.11).

### 5.5 Bet sizing (`betting/kelly.py` + `betting/portfolio.py`)

**Signal-dependent Kelly fraction** (`signal_dependent_lambda()`): the base
Kelly fraction is scaled by three factors derived from the current bet signal:

- `edge_factor`: scales 0.6× for lean edges (2-3%), up to 1.0× for strong
  (5-15%), and tapers back to 0.5× for suspect (>15%) signals.
- `clv_factor`: scales 1.15× when CLV t-stat > 1.5 (model historically
  beats the closing line), 0.7× when CLV < -1 (model historically fades).
- `disagree_factor`: scales 0.85× when the model-market **probability gap**
  `|p_model − p_market|` is large (≥0.15), 0.65× when extreme (≥0.20) —
  extra caution when conviction rests on outlier disagreement. (The caller
  passes the probability gap, not `edge`: passing `edge` double-counted the
  `edge_factor` signal and let longshots — where `edge = gap/market` blows
  up at small prices — masquerade as extreme disagreement.)

The composite is clamped to `[0.25×, 1.25×]` of the base `KELLY_FRACTION`.

**Slate-level portfolio Kelly** (`optimize_slate()` in `betting/portfolio.py`,
wired into `recommendations.generate_recommendations`): when N > 1
positive-EV bets are present, a `scipy.optimize.minimize(SLSQP)` solver
maximises `E[log(1 + Σ fᵢ·rᵢ)]` (approximated via Gaussian copula Monte
Carlo with 2,000 samples) in full-Kelly space, then post-shrinks by
`kelly_lambda · drawdown_mult` and enforces `0 ≤ fᵢ ≤ MAX_BET_PCT` and
`Σ fᵢ ≤ MAX_EXPOSURE_PCT`. `build_simple_correlation()` provides the
default correlation matrix (same-day off-diagonal ρ = 0.05). Falls back
to proportional per-bet Kelly haircut when the solver fails. For a single
actionable bet the portfolio call is skipped — per-bet Kelly is exact.

**Closing Line Value (CLV)** is tracked per bet. `record_predictions()`
stores `market_home_prob` (the price at pick time) and `opening_market_prob`
(earliest snapshot); `update_results()` fills `closing_market_prob` from the
latest pre-game snapshot, then `compute_clv()` sets, **in log-odds space**:

```
clv = logit(close_side) - logit(bet_side)     # bet_side priced off market_home_prob
```

Positive ⇒ the line moved toward our side after we bet (we beat the close).
**Note the sign:** it's `close − bet`, not `bet − close` — beating the close
is positive. Two guards make the metric honest (see the `clv` review):
- **Reference = the bet-time price** (`market_home_prob`), not the opening
  snapshot — otherwise you measure total market drift, not your CLV.
- **CLV is `None` unless ≥2 distinct snapshots exist** (`opening != closing`).
  A single-snapshot game has `open==close` by construction; recording that as
  `0` diluted the mean and inflated the sample. `compute_clv()` runs over the
  whole history on every `update_results()`, so the definition is applied
  retroactively. The `clv` CLI reports per-bet CLV (logit) and a t-statistic.

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

### 6.7 Why running `backtest` twice produces identical numbers

`backtest` is **deterministic against the games-table snapshot plus the
model code**. Three things make this so:

- The walk-forward loop **trains its own per-fold models**. It does not
  load `gbm_latest.joblib`. Running `train` between two `backtest` runs
  changes the *saved* model but not the backtest output.
- `HistGradientBoostingClassifier` is reproducible with a fixed
  `random_state`, and the SLSQP / Monte Carlo paths use seeded RNGs.
- The test set is whatever's in the `games` table. If sync hasn't pulled
  in new games, the test set hasn't moved.

So "same backtest numbers as last week" is the *expected* signal that
nothing changed in either the data or the algorithm. To meaningfully
move the metric you need new games (run `sync`), feature/model code
changes, or different config knobs (`MARKET_SHRINKAGE_LAMBDA`,
`MIN_BET_SIDE_PROB`, etc.). Re-running `train` alone will not.

### 6.8 `update_results` ±1-day fallback

`record_predictions` now files each entry under the *game's* ET date,
taken from `BetRecommendation.game_date` (which
`generate_recommendations` derives from the game's UTC tipoff via
`game_date_et()`). This is the same convention `Game.date` uses, so
the exact-date match in `update_results` resolves predictions cleanly
— including for upcoming-day slates where `today_et()` ≠ the game's
date (predict-on-a-quiet-night → next-day games).

The ±1-day fallback is now **mostly defensive** for very old records
that pre-date the per-game-date fix — two historical drift modes are
covered:

1. **Local-date drift**: pre-PR-#13, `record_predictions` used
   `date.today()` instead of `today_et()`, so evening sessions in
   non-ET timezones filed records under tomorrow-local while the game
   was stored under today-ET.
2. **Prediction-run-date drift**: pre-this-fix, even after PR #13 the
   record was filed under `today_et()` rather than the game's ET
   date, which broke matching whenever the slate was for an upcoming
   day. The 15 stuck `2026-04-10 → 2026-04-12` records in our
   real history are this category.

The fallback resolves only when **exactly one candidate** exists in
the window. NBA matchups don't repeat on consecutive calendar days
even in playoffs (Game 1 → Game 2 is always 2+ days apart), so the
single-candidate rule is safe. New predictions written under this fix
should never need the fallback — they should resolve via the exact
match — so a sustained uptick in fallback-resolved records would be
a signal that something upstream is misfiling dates again.

### 6.9 Why the `simulate` defaults look the way they do

The Monte Carlo `simulate` CLI bootstraps `(p_model, p_market, won)`
tuples from a backtest's bet pool. Two defaults need explicit
explanation because they materially shape what users see:

**`--horizon` defaults to a date-density projection over one NBA
season (~240 days, regular + playoffs).** Implemented by
`_estimate_one_season_bets()` in `nba_betting/cli.py`: compute the
backtest's bet density `pool_size / span_days`, multiply by 240, then
clip to `[20, pool_size]`. Each invocation prints the chosen horizon
plus a one-line reason so the user can see what density was inferred.

This replaced the prior fixed cap `min(200, len(bets))`, which was
arbitrary in two failure modes:

- A **heavy bettor** (e.g. 4 bets/day on every slate) has ~960 bets
  per season — capping at 200 hid the bulk of a realistic season's
  variance.
- A **selective bettor** (e.g. 0.3 bets/day after raising the edge
  floor) has ~70 bets per season — the 200 cap then bunched three
  seasons of bets into a single horizon, re-saturating P(Profit).

The full backtest pool (typically 3 seasons of bets) is also not the
right forward-looking horizon for end users: with the model's
realized log-growth/bet ~ +0.005 against the Elo proxy, compounding
2000+ bets at any positive drift saturates `P(Profit) → ~100%` by
arithmetic alone — mathematically correct, operationally useless. The
date-density default targets a horizon where the percentile
distribution actually informs decisions.

Fallback behavior: backtests with fewer than two distinct dates, an
unparseable `"date"` field, or a span < 30 days fall back to the full
pool (extrapolating bet density from a 2-week window would inflate
the projected horizon by 10×+). The CLI surfaces the fallback reason
in its rationale line.

Override with `--horizon N` when you want a specific value (including
the full pool size for the original compounded behavior).

**`--live-strategy` defaults to ON.** The simulator's job is to
project the *live system* — what `predict` would actually do — not
to bound raw model skill. Defaulting `apply_live_strategy=True` in
the underlying backtest applies the same Bayesian shrinkage and
asymmetric bet-side floor that live betting does, so the bet pool
the bootstrap draws from matches reality. Pass `--no-live-strategy`
to recover the raw-model bounding behavior (or use
`backtest --raw-model` for the canonical version of that question;
see §6.6).

**Inflated-edge banner.** When `--real-odds` is off (Elo proxy used)
AND the empirical log-growth/bet exceeds ~0.003 (≈30bps/bet), a
warning banner prints before the table noting that the numbers are
arithmetic of an overstated per-bet edge, not a real-world forecast.
The market-is-right column is the apples-to-apples skill comparison
in that regime.

**Headline is `log-growth/bet`, not `P(Profit)`.** Compounded
metrics scale with horizon; log-growth/bet does not. The latter is
the honest, comparable measure of skill across horizons and against
the market-null. The table puts horizon-invariant metrics first and
labels every horizon-dependent metric with `@ horizon=N bets` so
the dependency is unambiguous.

### 6.10 Elo home-court advantage is calibrated, not the 538 default

**Problem**: the inherited `ELO_HOME_ADVANTAGE = 100` is the classic
FiveThirtyEight value, tuned to an older NBA where home teams won ~60%+
of games. At equal Elo it implies P(home) = `expected_score(1500+100,
1500)` = **0.640**. Our three seasons of data have a home-win rate of
**0.550** (54.7% / 54.6% / 55.6% per season), so every Elo prediction
carried a systematic **+8 to +9 pp home bias**. That bias propagated
into the `elo_home_prob`, `home_elo`, `away_elo`, and `elo_diff`
features the GBM consumes, into the log-odds ensemble (Elo gets ~30% of
the blend weight), and into the `--model elo` path directly.

**Fix**: `ELO_HOME_ADVANTAGE = 40`. Chosen by an **end-to-end sweep**
that recomputes the entire rating history *and* the predictions at each
candidate value (not just re-scoring fixed ratings — the home bonus also
enters the rating update via `home_expected`), then evaluates on the
walk-forward out-of-fold set with fixed GBM hyperparameters (no
grid-search noise):

| HA | Elo-only acc | Ensemble acc | Ensemble Brier | Ensemble log-loss |
|----|:---:|:---:|:---:|:---:|
| 100 | 62.6% | 63.9% | 0.2197 | 0.6295 |
| 60  | 65.7% | 66.0% | 0.2143 | 0.6188 |
| 50  | 66.1% | 66.6% | 0.2139 | 0.6179 |
| **40** | **66.8%** | **66.8%** | **0.2130** | **0.6153** |
| 35  | 67.1% | 66.9% | 0.2142 | 0.6182 |

40 is the joint minimum of Brier and log-loss and is tied-best on
accuracy, while making the mean Elo prediction (0.554) match the base
rate (0.550). The GBM-*only* metrics are nearly flat across HA because
its dominant feature `elo_diff` is a difference (the home bonus cancels)
— so the **single `train`-table accuracy number understates this fix**;
the gain shows up in the ensemble that `predict` actually uses
(+2.9pp accuracy, −0.0067 Brier, −0.0142 log-loss vs the old default).

**Do NOT revert to 100.** If a future NBA season's home edge rises,
re-run the sweep rather than guessing — the optimum is the value whose
mean Elo prediction equals the realized home-win rate.

### 6.11 The ensemble weight must be learned out-of-fold, not in-sample

**Problem**: `train` selected the static Elo-vs-GBM blend weight by
grid-searching log-loss on the **isotonic calibration slice** — but it
scored the *calibrated* GBM on the very slice the isotonic step was fit
on. Isotonic regression interpolates its own fit points, so the GBM
looked near-perfectly calibrated there (the tell-tale `ECE = 0.0000`),
while Elo got no such in-sample boost. The comparison was rigged toward
the GBM: it picked **w_elo ≈ 0.30** (70% weight on the GBM).

On honest **out-of-fold** data the calibrated GBM is the *weaker* model
(acc ~62%, log-loss 0.669) and Elo is stronger (acc ~67%, log-loss
0.620, once §6.10 is applied). So the old weight put the majority of the
blend on the worse model. Because **production blends via the static
`ensemble_predict` weight** (the meta-learner is not wired in — §5.4),
this directly degraded live predictions.

**Fix**: select the weight (and report calibration ECE/Brier) on the
walk-forward OOF arrays `learn_ensemble_weight(oof_elo, oof_gbm, oof_y)`
— the same honest predictions the meta-learner trains on. The optimizer
now picks **w_elo ≈ 0.90**. Measured on the OOF set:

| Blend | Accuracy | Brier | Log-loss |
|---|:---:|:---:|:---:|
| old `w_elo = 0.30` (in-sample pick) | 65.8% | 0.2179 | 0.6459 |
| new `w_elo = 0.90` (OOF pick) | **66.8%** | **0.2137** | **0.6166** |

i.e. **+1.0pp accuracy and −0.029 log-loss in production**, on top of
§6.10. The GBM isn't useless — at ~10% weight it still improves the
blend's Brier/log-loss over pure Elo — it just shouldn't dominate. Never
score a calibrated model on its own calibration set; if you need a
single honest held-out number, use the OOF predictions.

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
| `ELO_HOME_ADVANTAGE` | `40.0` | Home bonus in Elo points — calibrated to the modern ~55% home-win rate, not the 538 default of 100. See §6.10 |
| `ELO_CARRYOVER` | `0.75` | Season-to-season Elo persistence |
| `NBA_API_DELAY_SECONDS` | `1.5` | NBA.com rate limit (reduced from 2.5) |
| `ESPN_API_DELAY_SECONDS` | `0.8` | ESPN rate limit (reduced from 1.5) |

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
# Expect: ~(3500+, 92), NaN? False

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
# Expect: GBM-only WF accuracy 63-65%, Brier ~0.225, calibrated ECE < 0.02.
# NOTE: this table is the GBM alone — it understates the system because
# `elo_diff` (its top feature) is HA-invariant. The *ensemble* the live
# `predict` uses scores ~66-67% accuracy / ~0.213 Brier on the same OOF
# (see §6.10). New features (EWM, SOS-adj, pace, off/def Elo) should
# appear in top-10 permutation importance.

# 6. End-to-end diagnose
.venv/bin/python3 -m nba_betting diagnose

# 7. Full test suite (89 tests across seven files).
.venv/bin/python3 -m pytest tests/ -v
# Expect: 89 passed in < 5s.
# test_new_features.py       — 16 tests (shrinkage, drivers, spreads, migration)
# test_improvements.py       — 15 tests (rolling stats, Four Factors, Elo,
#   portfolio optimizer exposure cap)
# test_tier_improvements.py  — 14 tests (off/def Elo, EWM, meta-learner,
#   signal-dependent Kelly, portfolio Kelly, dedup, fuzzy matching, cache)
# test_montecarlo.py         — 12 tests (empirical bootstrap, market-null,
#   log-growth invariance, reproducibility, validation)
# test_simulate_horizon.py   —  8 tests (data-driven horizon, density scaling)
# test_snapshot_jsonl.py     — 14 tests (JSONL round-trip, idempotence,
#   ESPN fallback, Polymarket date disambiguation)
# test_playoff_sync_and_resolve.py — 10 tests (play-in/playoff sync,
#   update_results matching, record_predictions ET-date filing)

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
- Isotonic (not Platt) calibration for the final full-slice model.
- Log-odds (not probability-average) ensemble.
- `ELO_HOME_ADVANTAGE` calibrated to the realized home-win rate (~40,
  not the 538 default of 100) — §6.10.
- Ensemble weight + calibration ECE measured **out-of-fold**, never on
  the isotonic calibration slice (which fakes ECE≈0) — §6.11.
- Kelly `disagree_factor` is fed the probability gap `|p_model−p_market|`,
  not `edge` (avoids double-counting `edge_factor`).
- ET timezone for "today", ScoreboardV3 not live ScoreBoard.
- `load_model()` cached in `routes.py` (not called 3 times).
- Driver attribution runs on the **base GBM**, not the calibrated
  wrapper, and is computed **lazily** (only for `bet_side != "NO BET"`).
- Backtest `apply_live_strategy` defaults to `None` and resolves from
  `use_real_odds` — never hard-code `True`, or Elo-proxy backtests
  become self-dampening.
- Snapshot `game_date` must equal the game's **ET** date (= `Game.date`), or
  it won't join via `get_closing_line`. The GH runner is DB-free and can only
  guess (UTC date, or capture date when `game_time_utc` is missing), so
  **`import_snapshots_jsonl` re-resolves every record to the real DB game**
  (`_resolve_game_for_snapshot`: nearest upcoming game to the capture
  timestamp) and stores `Game.date` + `game_id`. Capture-side
  `_game_date_from_game` prefers the ET date too, but import is authoritative.
  `reresolve_existing_snapshots()` back-fills older rows. This was the
  2%-coverage bug: pre-tipoff snapshots filed under the capture/UTC date
  never joined (see §6.6 note).

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
- **Snapshot game_date** — snapshots must be filed under the game's **ET**
  date to join `get_closing_line()`. Capture can only guess (UTC/capture
  date), so `import_snapshots_jsonl` re-resolves each record to the real DB
  game (`_resolve_game_for_snapshot`) and stores `Game.date` + `game_id`;
  `reresolve_existing_snapshots()` fixes historical rows. Earlier this was
  done UTC-side at capture, which misfiled pre-tipoff snapshots by 1-2 days
  and capped real-odds coverage at ~2% (see §6.6 / invariants).
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

### 10.1b Prediction engine & efficiency improvements (2026-04)

Three tiers of improvements shipped after the hardening pass:

**Tier 1 — Prediction quality:**
- **SOS-adjusted rolling net rating** — `adj_net_rtg_roll_{5,10,20}` in
  `features/rolling.py`. Divides net rating by a strength-of-schedule
  multiplier derived from rolling mean opponent Elo. Exposed as
  `diff_adj_net_rtg_roll_{w}` in the model feature set.
- **Pace / possessions** — `poss_roll_{5,10,20}` per team and
  `matchup_pace_{w}` (home + away average). Possessions approximated as
  `FGA + 0.44·FTA + TOV - OREB`.
- **EWM rolling stats** — `rolling_ewm(halflife=10)` with shift(1) on
  `net_rtg_game` and `plus_minus`. `diff_net_rtg_game_ewm_10` became
  the #2 permutation-importance feature after retraining.
- **Off/def Elo split** — parallel offensive/defensive Elo updated via
  `update_off_def_elo()`. Schema migration adds `elo_off_before/after`
  and `elo_def_before/after` to `EloRating`, `current_elo_off/def` to
  `Team`. Exposes four matchup-aware diff features (see §4.8, §5.1).
- **Opponent-strength MOV dampening** — `opp_strength_factor()` sigmoid
  multiplied into the MOV multiplier so blowouts over weak opponents
  inflate ratings less (see §5.1).

**Tier 2 — Model architecture & bet sizing:**
- **Per-fold hyperparameter grid search** — 8-combo grid inside each WF
  fold; best params reused for final training (see §5.2).
- **Stacked meta-learner** — logistic regression on out-of-fold logits;
  present as `ensemble_meta.joblib`, falls back to log-odds blend (see §5.4).
- **CLV tracking** — `market_home_prob` (pick-time price) stored per
  prediction; `update_results()` + `compute_clv()` compute the log-odds CLV
  post-game (`close − bet`, gated on ≥2 snapshots). `clv` CLI command and
  `performance` table both surface CLV (logit) + t-stat.
- **Slate-level portfolio Kelly** — joint SLSQP optimization with Gaussian
  copula correlation for correlated same-day bets, wired into
  `recommendations.generate_recommendations` (see §5.5).
- **Signal-dependent Kelly fraction** — edge/CLV/disagreement scaling
  of the base Kelly multiplier, clamped to `[0.25×, 1.25×]` (see §5.5).

**Tier 3 — Efficiency:**
- **Mtime-keyed model cache** — `load_model()`, `load_feature_means()`,
  and `load_calibrated_model()` cache in-process and only reload on
  artifact change. Saves ~50ms per API call.
- **Vectorized Four Factors** — `opp_dreb` computed via `groupby`
  transform instead of per-row apply (3-5× speedup on full history).
- **Vectorized injury impact** — `build_feature_matrix()` merges injury
  impact via pandas merge instead of list comprehension.
- **API rate limits reduced** — `NBA_API_DELAY_SECONDS` 2.5→1.5,
  `ESPN_API_DELAY_SECONDS` 1.5→0.8 (well under observed throttle points).
- **Odds snapshot deduplication** — `_is_duplicate()` skips write if
  prices moved < 0.5% within a 4h window; safe to run on tight crons.
- **Polymarket fuzzy fallback** — `_name_to_abbr()` tries exact match
  then falls back to ordered substring containment for non-standard titles.

### 10.1c Code-review hardening pass (2026-05)

A full audit of the system's logic, calibration alignment, and hot paths:

**Bugs fixed:**

- **Off/def Elo scale** (`models/elo.py`) — `update_off_def_elo` was
  passing `home_off + home_def` (~3000) to `mov_multiplier` which
  expects aggregate Elo (~1500). The `opp_strength_factor` sigmoid uses
  a 200-point scale, so the summed value doubled the dampening. Fixed
  to pass `(off + def) / 2`.
- **Kelly edge formula** (`betting/kelly.py`) — `signal_dependent_lambda`
  received `prob - market_price` (probability difference) but its
  thresholds (`< 0.02`, `<= 0.08`, `<= 0.12`, `<= 0.20`) are calibrated
  for multiplicative edge (`prob / market_price - 1`). For a 60%/50%
  game the two values are 0.10 and 0.20 respectively — a 2× discrepancy
  that caused high-edge bets to be under-penalized. Fixed to compute
  `edge = prob * (1.0 / market_price) - 1.0`.
- **Backtest calibration** (`betting/backtest.py`) — `run_backtest` trained
  a raw `HistGradientBoostingClassifier` per fold and used its uncalibrated
  probabilities for edge/Kelly. The live predict path uses the isotonic
  calibrated model. Each backtest fold now holds out its last 20% of
  training data, re-fits on the 80%, applies isotonic calibration, and
  uses the calibrated model for test predictions — matching the live
  pipeline. Falls back to uncalibrated on small folds (< 20 samples).
- **Sharpe ratio** (`betting/backtest.py`) — the old formula
  `(mean_profit / std_profit) * sqrt(n)` is the t-statistic for "mean
  P&L ≠ 0", not Sharpe — it scaled with bankroll size and grew with
  bet count. Replaced with return-based Sharpe: `mean(profit/bet_size) /
  std(profit/bet_size) * sqrt(1000)` (1000 ≈ NBA season bet count),
  making it stationary and bankroll-size-independent.

**Portfolio optimizer wired:**

- `betting/recommendations.py` — the ad-hoc `1/sqrt(1 + (n-1)·ρ)`
  correlation haircut is replaced by `optimize_slate()` (see §5.5).
  `portfolio.py` was fully implemented but never imported until this pass.

**New shared utilities:**

- `nba_betting/utils/math.py` — single source of truth for `logit`,
  `sigmoid`, `logit_scalar`, `sigmoid_scalar`. Previously each of
  `ensemble.py`, `shrinkage.py`, and `stacking.py` defined its own copy;
  all now import from here.

**Efficiency improvements:**

- **Vectorized `simulate_bankroll`** (`betting/montecarlo.py`) — the
  `for sim in range(n_simulations): for bet in ...` double loop (~12M
  Python iterations at 60k sims × 200 bets) replaced with a NumPy
  batch approach: pre-sample all indices as `(S, B)`, pre-compute Kelly
  fractions and decimal odds for the source pool, pre-generate all
  outcomes, then loop over bet steps only (~200 iterations) with NumPy
  array ops across all simulations simultaneously. 60k × 200 bets: ~0.16s.
- **Temporal early stopping** (`models/xgboost_model.py`) — `train_model`
  previously used `validation_fraction=0.15` (random internal split),
  which could pull future-season games into the early-stopping holdout.
  When `_date` is present in `X`, it now sorts chronologically, fits a
  temporary model on the first 85% to find `optimal_n_iter`, then
  retrains on the full dataset with `early_stopping=False, max_iter=optimal_n_iter`.
  Falls back to the existing random-split behavior when `_date` is absent.
- **Four Factors rolling** (`features/builder.py`) — the nested
  `for team_id in groupby: for col: for window: rolling_df.loc[idx] = ...`
  loop replaced with `groupby().transform()` which uses pandas' internal
  C path throughout. ~2× faster on a full 3-season history.
- **`compute_prediction_drivers` matrix** (`models/drivers.py`) — the
  neutralized feature matrix was built via `pd.concat([row]*N)` (N full
  DataFrame copies) + `iloc` cell patching. Now uses `np.tile(row_vals,
  (N, 1))` + `np.fill_diagonal` — pre-fills every row with baseline
  values and sets the diagonal to neutral means in one operation. ~2×
  faster at 150 features (14ms → 7ms per attribution call).
- **`build_feature_matrix(recompute_elos=True)`** (`features/builder.py`) —
  new flag lets callers that already computed Elos (e.g. the train
  pipeline before the first `build_feature_matrix` call) skip the
  expensive DB wipe-and-rebuild. Default `True` preserves existing behavior.

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
- **Meta-learner requires WF data to train** — `ensemble_meta.joblib`
  is only produced when there are sufficient out-of-fold folds. On a
  fresh install with limited history, the system silently falls back to
  the log-odds blend. This is intentional — a meta-learner on 1 fold
  would overfit badly.
- **CLV bootstrapping** — `clv` is only populated after a bet resolves AND
  the odds snapshot collector captured **≥2 snapshots** (a genuine closing
  line distinct from the open). The CLV t-stat becomes meaningful after ~30
  such bets — far fewer games qualify than total bets until coverage grows.
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

### 10.1d Closing dead-code gaps (2026-05)

Two modules were fully implemented but never wired into the training or
prediction pipelines:

**Player impact features** (`features/player_impact.py`):

`compute_player_impact_features()` computes 6 features — WAT score,
missing-minutes %, and a star-out flag for each team — but was never
called anywhere. Now wired:

- `features/builder.py` step 7d adds all 6 columns as **0.0 for
  historical games** (same forward-accumulating convention as
  `injury_impact`). This gets them into `feature_cols.joblib` so the
  model can learn their signal once live data accumulates.
- `cli.py` `_xgb_predict` and `api/routes.py` `_predict` now call
  `compute_player_impact_features` and inject the live values into
  `extra_features` before `build_prediction_features`. The `injuries`
  list from the outer scope is available via closure.
- The Four Factors rolling loop in `cli.py`'s predict path was also
  updated to the vectorized `groupby().transform()` form (matching the
  `builder.py` training path, for consistency).

**Stacked meta-learner** (`models/stacking.py`):

`fit_meta_model` / `save_meta_model` existed in stacking.py but the
`train` CLI command never called them, so `ensemble_meta.joblib` was
never produced and `blend_predictions()` always fell back to the static
log-odds blend.

- `walk_forward_validate()` gains a `return_oof=True` flag. When set,
  it appends per-fold `elo_home_prob` (from the feature matrix),
  calibrated GBM probability, and ground-truth label into
  `"oof_elo_probs"`, `"oof_gbm_cal_probs"`, `"oof_y_true"` arrays in
  the return dict.
- The `train` command calls `walk_forward_validate(X, y, return_oof=True)`
  and, after saving the calibrated model and ensemble weight, fits a
  logistic regression meta-learner on the OOF arrays when ≥ 200 OOF
  games are available. The artifact is saved to
  `trained_models/ensemble_meta.joblib`.
- After the next `train` run, `blend_predictions()` will automatically
  use the meta-learner (game-dependent Elo/GBM weights) instead of the
  static grid-searched scalar.
