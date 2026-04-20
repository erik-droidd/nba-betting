# NBA Betting System — Usage Guide

## Overview

This system predicts NBA game outcomes using an ensemble model (Elo ratings + gradient boosting), compares predictions against market odds (Polymarket + ESPN/DraftKings), and recommends bets when it finds an edge. It uses a **Bayesian-shrunken** model probability (treating the market as a strong prior), an **asymmetric bet-side floor** that kills lottery-ticket bets, quarter-Kelly sizing, and generates plain-English explanations for each recommendation.

---

## Initial Setup (First Time Only)

### 1. Install Dependencies

```bash
cd "NBA Betting"
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Load Historical Data

Fetch 3 seasons of NBA game data from NBA.com and compute Elo ratings. This takes several minutes due to API rate limiting (1.5s per call).

```bash
python3 -m nba_betting sync --seasons 3
```

This will:
- Download team box scores for the last 3 seasons
- Store them in `data/nba_betting.db` (SQLite)
- Compute Elo ratings for all 30 teams
- Auto-resolve any pending predictions from previous runs

### 3. Train the Model

```bash
python3 -m nba_betting train
```

This will:
- Build a feature matrix (~92 features: rolling stats, Four Factors, **SOS-adjusted net rating**, **pace/possessions**, **EWM-weighted stats**, **off/def Elo split**, Pythagorean expectation, rest days, Elo)
- Run walk-forward validation (trains on older data, tests on newer) with July 1 season boundaries — **with per-fold hyperparameter grid search** (max_depth, learning_rate, max_iter, regularization); best params saved to `trained_models/best_params.joblib`
- Report accuracy, Brier score, and log loss per fold
- Train the final model on all data
- **Calibrate probabilities via isotonic regression** (replaces Platt sigmoid, which over-compressed tails on the ~54% home-win base rate)
- **Grid-search the optimal Elo-vs-GBM ensemble weight** by minimizing log-loss on the calibration fold — the learned weight is saved to `trained_models/ensemble_weight.joblib` and reloaded automatically at prediction time
- When sufficient out-of-fold data is available, fit a **stacked logistic meta-learner** on [elo_logit, gbm_logit, |disagreement|] and save it to `trained_models/ensemble_meta.joblib`; falls back to the log-odds blend otherwise
- Save all model artifacts to `trained_models/`

**Expected output**: ~63-65% walk-forward accuracy, Brier score ~0.223, calibrated ECE ~0.00-0.02.

### 4. Sync Player Rosters (Optional, Improves Predictions)

```bash
python3 -m nba_betting sync-players
```

Fetches rosters and depth charts from ESPN for all 30 teams. Used for injury impact estimation. Rate-limited, takes ~2 minutes.

---

## Daily Workflow

Run these commands on game days to get betting recommendations.

### Step 1: Sync Today's Data

```bash
python3 -m nba_betting sync
```

Fetches any new game results since your last sync. Updates Elo ratings and resolves pending predictions. Run this once per day, ideally in the morning or before games start.

### Step 2: Get Predictions

```bash
python3 -m nba_betting predict
```

This is the main command. It will:
1. Fetch today's NBA games from NBA.com (uses **US Eastern time** to determine "today" regardless of your local timezone, so users in Europe/Asia see the correct slate). If today has no remaining scheduled games, it falls back to the next available game day within the next 7 days and labels the output accordingly.
2. Load current Elo ratings
3. Sync injuries from ESPN (automatic — ~150 players tracked)
4. Fetch market odds from Polymarket (filters out closed/resolved markets) and ESPN/DraftKings as fallback
5. Snapshot odds for line movement tracking
6. Run the ensemble model (Elo + calibrated GBM, blended in **log-odds space** with a weight learned during training; stacked meta-learner used when available)
7. **Apply injury adjustments** to the raw model probability (each injured player reduces team strength based on their impact rating)
8. **Bayesian-shrink** the injury-adjusted model probability toward the market log-odds (λ = 0.6 by default — market-leaning). This is the single biggest change to how the system filters bets: small model-vs-market disagreements get pulled back to the market, and only decisive conviction survives.
9. Compute edge against the **shrunken** probability (not the raw model) — so model%, market%, and edge% reconcile exactly in the UI
10. Apply the **asymmetric bet-side floor** (`MIN_BET_SIDE_PROB = 0.30`): the system refuses to bet a team the model itself only gives a <30% chance of winning, even if the edge math looks positive. This kills "lottery-ticket" bets where positive EV depends on a tail price the model isn't really contradicting.
11. Size bets using **signal-dependent quarter-Kelly** (fraction scales with edge magnitude) and **slate-level portfolio Kelly** (accounts for correlated same-day bets via Gaussian copula; falls back to per-bet Kelly when only 1 bet)
12. Display recommendations with explanations (the explanation prefers feature signals that *agree* with the bet; if every surface stat contradicts the bet, it says so honestly rather than parroting misleading reasoning)

**Output columns**:
| Column | Meaning |
|--------|---------|
| Matchup | Away @ Home |
| Model | Model's **post-shrinkage** P(home win) — the same number the edge is computed against, so the math reconciles in the UI |
| Market | Polymarket/ESPN implied P(home win) |
| Bet | Team abbreviation to bet on (or "—" for no bet) |
| Edge | Expected value per $1 against market price: `(shrunken_prob / market_prob) - 1` |
| Kelly | Optimal bet fraction of bankroll (quarter-Kelly with 5% cap) |
| Size | Dollar amount to bet |
| Signal | Confidence: STRONG (5-15%), MODERATE (3-5%), LEAN (2-3%), SUSPECT (>15%). Rows rejected by the bet-side floor show `NO BET` — never SUSPECT — to avoid confusing warnings on rows we already filtered out. |

**Sub-rows rendered below each matchup** (console + dashboard):
- **Spread / Total picks**: when ESPN spread and O/U lines are available and the model disagrees by more than the edge threshold (1.5 pts spread, 2.5 pts total), a pick row appears, e.g. `BOS +4.5` or `OVER 221.5`. No sub-row is shown when the model's point-margin edge is below the threshold.
- **Driver chips**: top 3 features driving the model's prediction, each showing the feature label and its ±probability shift. Positive chips (green) push toward the home team; negative chips (red) push toward the away team. These come from a leave-one-out-to-mean attribution on the base GBM (not the isotonic wrapper), so the magnitudes are self-consistent. Chips are filtered by a 0.5 pp noise floor — sub-threshold shifts are suppressed.

Below the table, each recommended bet includes a plain-English explanation of why the model favors it, with the displayed model% matching the Model column exactly.

### How to Interpret the Results

**Model vs Market — what do they mean?**

- **Model** is the system's estimated probability that the **home team** wins (e.g., "41.7%" means the model thinks the home team wins ~42% of the time).
- **Market** is the implied probability from Polymarket or ESPN odds — what the betting market thinks.
- The **Bet** column shows the team abbreviation you should bet on. If it shows "—", there's no recommended bet.

**Value betting — why bet on a team with <50% win probability?**

The system finds **value**, not just winners. A bet is profitable when the payout exceeds the risk, even if the team loses more often than it wins.

**Example**: The model says WAS has a 42% chance to win, but the market prices them at 30%.
- At 30% market odds, a $1 bet pays ~$3.33 if WAS wins.
- Expected value: 42% × $3.33 = $1.40 return per $1 bet → **+$0.40 profit per dollar**.
- Over many bets like this, you profit even though WAS loses most individual games.

The **Edge** column shows this expected profit per dollar (41.4% in this example). It's calculated as:
```
Edge = (Model Probability × Decimal Odds) - 1
     = (Model Probability / Market Probability) - 1
```

**Signal levels:**

| Signal | Edge | Meaning |
|--------|------|---------|
| **STRONG** | 5-15% | High-confidence bet. Model sees significant mispricing. |
| **MODERATE** | 3-5% | Good bet. Solid edge but smaller margin. |
| **LEAN** | 2-3% | Marginal. Skip if unsure — the edge is thin. |
| **SUSPECT** | >15% | **Warning**: Edge is unrealistically large. This usually means stale/incorrect market data, thin Polymarket liquidity, or a model error. **Verify the market odds manually before betting.** |

**What to do with each signal:**
- **STRONG/MODERATE**: Bet the team shown in the "Bet" column on Polymarket (or your platform). Use the dollar amount in "Size".
- **LEAN**: Optional — only bet if you trust the matchup context.
- **SUSPECT**: Do **not** bet blindly. Open Polymarket/the sportsbook and check if the market price is accurate. If the real odds differ from what the system shows, the edge is artificial. The CLI prints a red `⚠ SUSPECT EDGE` warning above the explanation, and the dashboard shows a red badge.
- **NO BET / "—"**: The model doesn't see enough edge. Skip this game. Also shown when the market column displays "N/A" — meaning no live market is available (game already started, no Polymarket listing, etc.).

**About Bayesian shrinkage (the key filter):**

Before the edge is computed, the model's probability is pulled toward the market in log-odds space:

```
posterior_logit = (1 - λ) · model_logit + λ · market_logit
```

- `λ = 0.0` → pure model (old behavior, produced tons of phantom >15% edges)
- `λ = 1.0` → pure market (never bets)
- **`λ = 0.6`** → default. Market is treated as a strong prior; model conviction has to be large to move the needle.

This is the Bayesian-correct stance when the prior (market) is known to be highly efficient and the model is roughly competitive with it. You can tune `MARKET_SHRINKAGE_LAMBDA` in `nba_betting/config.py` if you want a more aggressive (lower λ) or more conservative (higher λ) system. If a game has no market price at all, shrinkage is skipped — but the asymmetric floor still applies.

**About the asymmetric bet-side floor:**

Even with positive-EV edge math, the system refuses to bet a side the model itself gives less than `MIN_BET_SIDE_PROB = 0.30` to win. This is standard quant practice: if your model only assigns 15% to a team winning, betting them at 10% market odds is a "lottery ticket" — the math says +EV, but the model isn't really contradicting the market, it's betting the noise on a tail.

With shrinkage + floor, the system now produces **far fewer** SUSPECT badges and much smaller daily exposure. On a recent 15-game slate the counts went from 12 SUSPECT / 14 actionable / $700 exposure → 1 SUSPECT / 4 actionable / $119 exposure.

**About injury adjustments:**

The model itself was trained only on team-level historical stats — it has no awareness of who's playing tonight. To compensate, the system applies a post-hoc adjustment based on the current ESPN injury report:
- A starter (impact 7) being **Out** lowers their team's win probability by ~4%
- A star (impact 8-10) being **Out** lowers it by ~5-6%
- A team's total injury hit is capped at -15%

This adjustment is applied **before** edge is computed, so the Model column already reflects today's injuries. If you see the explanation mention an injury, that injury has already been factored into the displayed probability.

**Options**:
```bash
python3 -m nba_betting predict --bankroll 5000    # Custom bankroll
python3 -m nba_betting predict --model elo        # Elo-only (no GBM)
python3 -m nba_betting predict --model ensemble   # Force ensemble
```

### Step 2b: Snapshot Odds (for Line Movement Tracking)

```bash
python3 -m nba_betting snapshot-odds
```

Records a point-in-time snapshot of current Polymarket + ESPN odds for today's games in the `odds_snapshots` database table. The model uses consecutive snapshots to compute three features: `spread_movement`, `prob_movement`, and `odds_disagreement` (Polymarket vs ESPN). These features have zero signal until at least two snapshots exist for a game.

Snapshots are **automatically deduplicated**: if prices haven't moved by more than 0.5% since the last snapshot within 4 hours, the row is skipped, so running this frequently is safe.

**Run this on a cron every 30–60 minutes during the season** (e.g. `*/30 * * * * cd "NBA Betting" && .venv/bin/python -m nba_betting snapshot-odds`). After ~30 days of snapshots the `odds_disagreement` feature becomes meaningful for retraining.

> **Note**: odds snapshots accumulate forward — historical games in the training set have these features set to 0.0. The model learns to use them only when they're non-zero (i.e., live season data).

#### Running Snapshots Remotely (GitHub Actions)

Based in Europe? NBA games tip off between 00:00 and 03:30 UTC — your local machine is asleep. The repo ships a GitHub Actions workflow that runs on GitHub's infrastructure instead. It writes JSONL snapshot records and commits them back to the repo; you `git pull` the next morning and import them locally.

**One-time setup:**

1. Push the repo to GitHub (if not already).
2. Go to **Repo → Settings → Actions → General → Workflow permissions** and select **"Read and write permissions"**. (The workflow already declares `permissions: contents: write`, but the repo setting is still required.)
3. Visit the **Actions** tab and confirm the workflow named **snapshot-odds** appears. The first run may need a manual "Enable workflow" click.
4. Click **Run workflow** → **Run workflow** once to smoke-test end-to-end. You should see a new commit `chore(snapshots): YYYY-MM-DD HH:MMZ [skip ci]` and a file in `data/odds_snapshots/`.

**Daily usage:**

```bash
python3 -m nba_betting import-snapshots --pull   # One-shot: git pull + load JSONL → DB
```

`--pull` runs `git pull --ff-only` in the repo root before importing, so snapshots committed overnight by the GitHub Actions runner land in your local working copy and then in your local `odds_snapshots` table in a single command. If `--pull` fails (no upstream, merge conflict), it prints a warning and still imports whatever files already exist locally — safe to run daily.

Without `--pull` you'll need to `git pull` manually first:

```bash
git pull                                     # Pull the overnight snapshot commits
python3 -m nba_betting import-snapshots      # Load JSONL → local odds_snapshots table
```

`import-snapshots` is **idempotent**: safe to rerun on any cadence, duplicates are detected on `(game_date, home_team_id, away_team_id, source, timestamp)` and skipped. You can also point it at a single file:

```bash
python3 -m nba_betting import-snapshots --path data/odds_snapshots/2026-04-18.jsonl
```

**Note on the `espn-fallback` source:** when the GH Actions runner prints `source=espn-fallback` with a `note: nba-api returned 0 games; using ESPN fallback`, that's the expected path — `stats.nba.com` silently blocks datacenter IPs, so the runner falls through to ESPN's scoreboard endpoint. The CLI reports this as `ok` with a cyan `note` line, not a yellow warn; the snapshot is still captured and committed normally.

**Cron schedule:** concentrated in the *pre-tipoff* window (where line movement actually has predictive signal), not during live games:

| Window (UTC) | ET equivalent | Cadence | Purpose |
|---|---|---|---|
| 13:00–17:00 | 9 AM – 1 PM | hourly | Late-morning injury news + early weekend tipoffs |
| 18:00–21:30 | 2 PM – 5:30 PM | every 30 min | Afternoon build-up, main Woj/Shams drop window |
| 22:00–01:45 | 6 PM – 9:45 PM | every 15 min | Dense closing-line capture through ET tipoffs |
| 02:00–02:45 | 10 PM – 10:45 PM | every 15 min | West Coast closing line |

~33 runs/day, ~1000 Actions-minutes/month (comfortably under the 2000-min free tier). No runs 03:00–13:00 UTC: all games are live/final and the code filters those out anyway. See [.github/workflows/snapshot-odds.yml](.github/workflows/snapshot-odds.yml) to tweak.

> **⚠️ GitHub's 60-day inactivity rule:** scheduled workflows are automatically disabled if the repo has no new commits for 60 days. The NBA offseason (June–October) exceeds this. **First action each October**: visit the Actions tab and re-enable the `snapshot-odds` workflow, then trigger a manual run to verify. The workflow itself commits daily during the season, so mid-season deactivation shouldn't happen.

### Step 3: Place Bets

Use the recommendations to place bets on Polymarket or your preferred platform. The system recommends quarter-Kelly sizing (conservative) with a 5% max per bet and 25% max total exposure.

**Rules of thumb**:
- Only bet on STRONG or MODERATE signals (3%+ edge)
- LEAN signals (2-3% edge) are marginal — skip if unsure
- Never exceed the recommended bet size
- The system caps total exposure at 25% of bankroll

---

## Checking Performance

### After Games Complete

Once games finish, sync the results and check how your predictions did:

```bash
python3 -m nba_betting sync          # Fetches final scores, resolves predictions
python3 -m nba_betting performance   # Shows accuracy, ROI, calibration
```

**Performance output includes**:
- **Prediction Accuracy**: % of games where model picked the correct winner
- **Bet Win Rate**: % of placed bets that won
- **Total Wagered / Profit / ROI**: Dollar amounts and return on investment
- **Max Drawdown**: Worst peak-to-trough decline
- **Closing Line Value (CLV)**: Average logit-delta between your bet price and the closing price — the gold-standard skill metric (positive = beating the closing line)
- **Calibration Check**: Predicted vs actual win rates by probability bin (should be close to diagonal)

### Closing Line Value

```bash
python3 -m nba_betting clv
```

Shows a per-bet CLV breakdown: your bet price, the closing Polymarket price, the logit delta, and whether each bet beat the line. Includes a rolling average CLV and a t-statistic (the statistical measure of whether your CLV is meaningfully positive). CLV is the fastest way to validate betting skill — 50 bets of positive CLV is statistically significant, whereas ROI needs hundreds.

### Backtesting (Historical Simulation)

To see how the strategy would have performed on past data:

```bash
python3 -m nba_betting backtest
python3 -m nba_betting backtest --bankroll 5000 --splits 3
```

Reports: win rate, ROI, Sharpe ratio, max drawdown, and per-signal breakdown.

**Backtest modes** — four combinations control whether real market odds and live shrinkage are applied:

| Command | `--real-odds` | `--live-strategy` | What it measures |
|---------|:---:|:---:|-----------------|
| `backtest` | off | off | **Pure model benchmark** — ideal for ablation. No shrinkage, no real market odds; uses Elo-proxy as the "market". |
| `backtest --real-odds` | on | **on** (auto) | **Live-equivalent simulation** — applies the same Bayesian shrinkage and asymmetric floor that `predict` uses. Best for realistic ROI estimates. |
| `backtest --real-odds --no-live-strategy` | on | off | Real odds with shrinkage disabled — useful for isolating the effect of shrinkage on ROI. |
| `backtest --raw-model` | off | off | Uses the **base GBM** (pre-isotonic calibration) — ablation to measure the value added by calibration. |

The `--live-strategy` flag defaults to `None`, which resolves to `True` when `--real-odds` is set and `False` otherwise. Pass `--live-strategy` / `--no-live-strategy` explicitly to override.

```bash
python3 -m nba_betting backtest --real-odds                      # Live-equivalent (recommended)
python3 -m nba_betting backtest --real-odds --no-live-strategy   # Real odds, no shrinkage
python3 -m nba_betting backtest --raw-model                      # Pre-calibration ablation
```

### Monte Carlo Simulation

To understand the range of possible outcomes:

```bash
python3 -m nba_betting simulate                         # runs both modes
python3 -m nba_betting simulate --mode empirical
python3 -m nba_betting simulate --mode market_right
python3 -m nba_betting simulate --real-odds --n-sims 50000
```

Runs 10,000 (default) simulated seasons by **bootstrapping from the
backtest's actual resolved bets**. Each simulated bet draws a real
historical `(p_model, p_market, won)` tuple with replacement — the
realized win flag is used directly, so the realized win rate is
preserved. Two modes are reported side-by-side:

- `empirical` (honest): resamples actual outcomes. If the model has
  edge, you'll see median ROI > 0 here; if not, you won't.
- `market_right` (pessimistic null): simulates each bet from
  `Bernoulli(p_market)` — the efficient-market assumption that our
  model has zero skill. Expected ROI should be ≤ 0. Acts as a
  sanity floor.

The diagnostic is the **gap** between the two: positive empirical ROI
combined with non-positive `market_right` ROI is evidence of real
edge (not just Kelly compounding). The tool prints this gap.

**Read the `Log-Growth / Bet` rows, not the compounded bankrolls.**
Kelly compounding makes final-bankroll medians balloon with the
number of bets — a real 0.5%-per-bet edge compounds to ~270,000×
starting bankroll over 2,000 bets, which is honest math but looks
absurd. The horizon-invariant per-bet log-return (`Median Log-Growth
/ Bet`) stays in the same order of magnitude regardless of horizon,
so it's the cleanest statement of skill. Rough guide:

- `+0.003` to `+0.010` per bet in `empirical` mode → plausible real edge.
- `0.000` or lower in `market_right` → expected under the efficient-
  market null. A positive value there would mean our simulator has a
  bug.
- The **gap** between empirical and market-null log-growth per bet
  is what you actually have. The tool prints it directly.

**Note on an earlier bug:** an earlier version of `simulate` flipped
each bet's outcome with `Bernoulli(p_model)` — a tautology that made
the model right by construction and produced nonsense medians in the
trillions with `P(Profit) = 100%`. That path has been removed; see
[`nba_betting/betting/montecarlo.py`](nba_betting/betting/montecarlo.py)
for the full rationale. If you see old outputs claiming ~100%
P(Profit) and ROI in the billions of percent, re-run after pulling
this fix.

---

## Diagnostics & Troubleshooting

### Validate the Pipeline

```bash
python3 -m nba_betting diagnose
```

Checks:
- Elo ratings exist and are reasonable (mean ~1500)
- GBM model and calibration loaded
- Feature means saved for prediction imputation
- Polymarket odds fetched and prices correct
- Today's games with Elo predictions

### Check Feature Readiness

```bash
python3 -m nba_betting readiness-status
```

Reports how many days of injury and odds-snapshot data have accumulated. The player-impact and line-movement features are forward-accumulating — they start at zero and become meaningful only after enough live-season data exists. Use this command monthly to know when it's worth retraining:

| Status | Injury days | Snapshot days | Meaning |
|--------|:-----------:|:-------------:|---------|
| **COLD** | < 5 | < 5 | Features are essentially zero — retraining now gains nothing from them |
| **PARTIAL** | 5–29 | 5–29 | Sparse signal. Retraining helps a little but wait for READY |
| **READY** | ≥ 30 | ≥ 30 | Enough data — retrain with `train` to unlock the new features |

The output also prints actionable nudges (e.g. "Run `snapshot-odds` on a cron to accumulate line-movement data").

### Run the Test Suite

```bash
cd "NBA Betting" && .venv/bin/python3 -m pytest tests/ -v
```

45 fast unit tests across three test files:
- **`test_new_features.py`** (16): shrinkage invariants, `humanize_feature` label map, spread/total pick sign convention, driver attribution ordering, backtest `apply_live_strategy` default coupling, and additive DB migration idempotence.
- **`test_improvements.py`** (15): rolling stats, Four Factors, Elo accuracy.
- **`test_tier_improvements.py`** (14): off/def Elo asymmetry, SOS-adjusted stats, EWM weighting, meta-learner round-trip, signal-dependent Kelly monotonicity, portfolio exposure cap, vectorized opponent-DREB, odds-snapshot dedup, Polymarket fuzzy name matching, model cache mtime invalidation.

Run this after any model or pipeline change to catch silent regressions before they corrupt live predictions.

### Common Issues

**"Market" column shows N/A for every game**
- Polymarket has no live (open) market for that game. The system filters out closed/resolved markets to prevent stale prices from yesterday's results bleeding into today's edges. If every game shows N/A, you're likely running `predict` after games have started — markets close at tipoff.
- Run `python3 -m nba_betting diagnose` to confirm Polymarket is reachable.

**Showing the wrong day's games (or "next game day" appears unexpectedly)**
- "Today" is determined in **US Eastern time** (the NBA scheduling timezone), not your local timezone. If you're in Europe or Asia and run `predict` early in your morning, ET may still be on the prior day — that's expected behavior, not a bug.
- The system queries `ScoreboardV3` with an explicit ET date (not the live `ScoreBoard()` endpoint, which can return a stale prior-day cache for hours after rollover).
- If today has zero scheduled games, the title becomes `Recommendations for YYYY-MM-DD (next game day)` and shows the next slate within 7 days. To force-check today, run `diagnose` to see what date the system resolved.

**Many bets show SUSPECT (>15% edge)**
- The model and market disagree wildly. Likely causes: thin Polymarket liquidity on that matchup, a bad data feed, or the model hasn't been retrained recently. Run `python3 -m nba_betting train` and verify the walk-forward accuracy is in the 62-66% range and ECE < 0.04.

**Predictions ignore obvious injuries**
- Run `python3 -m nba_betting injury sync` to refresh from ESPN, then `python3 -m nba_betting injury list` to confirm key players are flagged. If a star is missing from ESPN's report, add a manual override with `injury add`.

### Manage Injuries

```bash
python3 -m nba_betting injury sync              # Auto-sync from ESPN
python3 -m nba_betting injury list              # View current injuries
python3 -m nba_betting injury add "LeBron James" --team LAL --impact 9  # Manual override
python3 -m nba_betting injury remove "LeBron James"
python3 -m nba_betting injury clear             # Clear all
```

ESPN injuries are auto-synced every time you run `predict`. Manual overrides are preserved across ESPN syncs.

### View Elo Ratings

```bash
python3 -m nba_betting elo
```

Shows all 30 teams ranked by Elo rating with deviation from league average.

---

## Web Dashboard

```bash
python3 -m nba_betting serve
```

Opens a web dashboard at `http://localhost:8050` with three tabs:
- **Predictions**: Today's games with model/market probabilities, bet recommendations, spread, O/U, and explanations. If today (ET) has no remaining scheduled games, the header shows `Recommendations for YYYY-MM-DD (next game day)` and renders the next available slate.
- **Elo Ratings**: Team rankings
- **Performance**: Historical accuracy and ROI metrics

---

## When to Run What

| When | Command | Why |
|------|---------|-----|
| First time | `sync --seasons 3` then `train` | Load data and build model |
| Start of season | `sync --seasons 3` then `train` | Retrain with fresh data |
| Daily (morning) | `sync` | Get yesterday's results, update Elo |
| Before games | `predict` | Get today's recommendations |
| Every 30–60 min (season) | `snapshot-odds` | Capture line movement; run on a cron (or GitHub Actions — see §Step 2b) |
| Daily (morning, EU users) | `import-snapshots --pull` | Git-pull + load JSONL snapshots written overnight by GitHub Actions |
| After games | `sync` then `performance` | Check results and accuracy |
| After games | `clv` | Review Closing Line Value skill metric |
| Weekly | `backtest --real-odds` | Realistic ROI estimate with shrinkage applied |
| Monthly | `train` | Retrain model with latest data |
| Monthly | `sync-players` | Update player rosters and depth charts |
| Monthly | `readiness-status` | Check if injury/odds features have enough data to retrain |
| As needed | `diagnose` | Debug issues with predictions |
| After any code change | `pytest tests/ -v` | Guard against silent regressions (45 tests) |

---

## Data Flow

```
NBA.com (game stats) ──┐
                       ├──> SQLite DB ──> Feature Matrix ──> GBM Model ─┐
ESPN (injuries,        │                                                 │
  odds, depth charts) ─┘                                                 │
                                                                         ├──> Recommendations
Polymarket (odds) ─────────> Market Prices ──────────────────────────────┘         │
ESPN/DraftKings (odds) ────> Fallback Prices + Spread/O/U ──────────────┘         │
                                                                                   ▼
                                                                    Terminal / Dashboard
```

## File Structure

```
data/
  nba_betting.db          # SQLite database (games, stats, Elo, player stats, odds)
  prediction_history.json # Prediction tracking for performance analysis
  injuries.json           # Current injury list (ESPN + manual overrides)

trained_models/
  gbm_latest.joblib         # Trained GBM base model
  calibrated_model.joblib   # Isotonic-calibrated model (wraps the base)
  feature_cols.joblib       # Feature column order
  feature_means.joblib      # Training means for NaN imputation
  ensemble_weight.joblib    # Grid-searched optimal Elo weight for log-odds blend
  best_params.joblib        # Best hyperparameters found during per-fold grid search
  ensemble_meta.joblib      # Stacked logistic meta-learner (present after sufficient WF data)
```

---

## All Commands Reference

```bash
# Data & model
python3 -m nba_betting sync --seasons 3         # Fetch game data + compute Elo
python3 -m nba_betting train                     # Train GBM model + calibrate
python3 -m nba_betting sync-players              # Sync player rosters from ESPN

# Predictions
python3 -m nba_betting predict                   # Today's recommendations + explanations
python3 -m nba_betting predict --bankroll 5000   # Custom bankroll
python3 -m nba_betting snapshot-odds             # Snapshot current odds (run on a cron)
python3 -m nba_betting snapshot-odds --jsonl data/odds_snapshots  # DB-free, for GitHub Actions
python3 -m nba_betting import-snapshots --pull   # Git-pull + load JSONL snapshots from GH Actions (daily)
python3 -m nba_betting import-snapshots          # Same, but without the git pull step

# Backtesting (four modes)
python3 -m nba_betting backtest                              # Pure model benchmark (no market odds)
python3 -m nba_betting backtest --real-odds                  # Live-equivalent (shrinkage applied) ← recommended
python3 -m nba_betting backtest --real-odds --no-live-strategy  # Real odds, shrinkage off
python3 -m nba_betting backtest --raw-model                  # Pre-calibration ablation
python3 -m nba_betting backtest --bankroll 5000 --splits 3   # Custom bankroll / folds

# Performance & analysis
python3 -m nba_betting elo                       # Current Elo standings
python3 -m nba_betting performance               # Historical accuracy + ROI + CLV
python3 -m nba_betting clv                       # Per-bet Closing Line Value breakdown
python3 -m nba_betting simulate                  # MC bootstrap (empirical + market-null)
python3 -m nba_betting simulate --mode empirical # honest bootstrap only
python3 -m nba_betting simulate --real-odds      # live-equivalent MC
python3 -m nba_betting simulate --n-sims 50000

# Diagnostics
python3 -m nba_betting diagnose                  # Validate prediction pipeline
python3 -m nba_betting readiness-status          # Check injury/odds feature accumulation tiers
pytest tests/ -v                                 # 45 unit tests (run after any code change)

# Injuries
python3 -m nba_betting injury sync               # Auto-sync injuries from ESPN
python3 -m nba_betting injury list               # View current injury list
python3 -m nba_betting injury add 'Name' --team LAL --impact 8  # Manual override
python3 -m nba_betting injury remove 'Name'
python3 -m nba_betting injury clear

# Dashboard
python3 -m nba_betting serve                     # Launch web dashboard at localhost:8050
python3 -m nba_betting commands                  # Show this help in terminal
```
