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

Fetch 5 seasons of NBA game data (team AND player game logs — the latter power the player-availability features and the Elo availability term) from NBA.com and compute Elo ratings (a few API calls per season, ~1-5 s each plus the 1.5 s rate-limit delay). The current season is derived from today's date (rolls over July 1), so this command keeps following the league from one season to the next without edits.

```bash
python3 -m nba_betting sync --seasons 5
```

This will:
- Download team box scores AND player game logs for the last 5 seasons
- Store them in `data/nba_betting.db` (SQLite)
- Compute Elo ratings for all 30 teams
- Auto-resolve any pending predictions from previous runs

### 3. Train the Model

```bash
python3 -m nba_betting train
```

This will:
- Build a feature matrix (~98 features: rolling stats, Four Factors, **SOS-adjusted net rating**, **pace/possessions**, **EWM-weighted stats**, **off/def Elo split**, Pythagorean expectation, rest days, Elo, **player availability** from the player game logs (missing-minutes %, star-out flag, available-production share — real for every historical game since 2026-09), injury impact, line movement)
- Run walk-forward validation (trains on older data, tests on newer) with July 1 season boundaries, using deliberately **heavily regularized GBM defaults** (depth 3, learning rate 0.02, 200 rounds, 100-sample leaves, L2 10 — the earlier defaults overfit at this sample size; see ARCHITECTURE §5.2). Uses a two-stage **temporal early-stopping split**: fit a temporary model on the first 85% of training data to find the optimal iteration count, then retrain on all data with that fixed count (avoids leaking future games into the early-stopping validation set)
- Report accuracy, Brier score, log loss, and calibrated ECE per fold
- Train the final model on all data
- **Calibrate probabilities via isotonic regression** (replaces Platt sigmoid, which over-compressed tails on the ~54% home-win base rate)
- **Grid-search the optimal Elo-vs-GBM ensemble weight** by minimizing log-loss on the calibration fold — the learned weight is saved to `trained_models/ensemble_weight.joblib` and reloaded automatically at prediction time
- **Fit a stacked logistic meta-learner** on the walk-forward out-of-fold predictions [elo_logit, gbm_logit, |disagreement|] and save it to `trained_models/ensemble_meta.joblib` (requires ≥ 200 OOF games; falls back to the log-odds blend otherwise). The meta-learner learns game-dependent Elo/GBM weights instead of the static grid-searched scalar.
- Save all model artifacts to `trained_models/`

**Expected output** (5 seasons + player logs synced, 3 walk-forward folds): ~67% walk-forward accuracy on the GBM-only table, Brier ~0.212, log-loss ~0.612; the "Elo (out-of-fold)" line should read Brier ~0.208 with ECE below 0.01, and the learned Elo weight 0.8-0.9. The blended ensemble that `predict` uses is within noise of the Elo model alone — on box-score features the Elo rating (with its home-court and back-to-back adjustments) already captures the available signal; the GBM's share is mainly the conduit for the live injury / line-movement features (ARCHITECTURE §5.4).

### 4. Sync Player Rosters (Optional, not on the prediction path)

```bash
python3 -m nba_betting sync-players
```

Fetches rosters and depth charts from ESPN for all 30 teams into the `player_stats` table. Since 2026-09 predictions do **not** read it: player availability comes from the NBA player game logs that `sync` loads, and injury impact ratings come straight from ESPN depth charts during the injury sync. Keep it only if you want roster data in the DB for your own analysis. Rate-limited, takes ~2 minutes.

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
7. **Account for who is out** inside the model itself: the Elo rating carries a back-to-back term and an availability term (share of each team's regular-rotation minutes expected to be missing, from the ESPN injury list matched to the player game logs), and the GBM sees the same availability features. The old heuristic injury shift is shown for reference but no longer applied.
8. **Bayesian-shrink** the model probability toward the market log-odds (λ = 0.6 by default — market-leaning). This is the single biggest change to how the system filters bets: small model-vs-market disagreements get pulled back to the market, and only decisive conviction survives.
9. Compute edge against the **shrunken** probability (not the raw model) — so model%, market%, and edge% reconcile exactly in the UI
10. Apply the **asymmetric bet-side floor** (`MIN_BET_SIDE_PROB = 0.30`): the system refuses to bet a team the model itself only gives a <30% chance of winning, even if the edge math looks positive. This kills "lottery-ticket" bets where positive EV depends on a tail price the model isn't really contradicting.
11. Size bets using **signal-dependent quarter-Kelly** (fraction scales with edge magnitude) and **slate-level portfolio Kelly**: when multiple positive-EV bets exist, `optimize_slate()` runs a SLSQP joint optimization with a Gaussian copula correlation matrix to maximize expected log-bankroll across the full slate, enforcing per-bet (5%) and total-exposure (25%) caps; falls back to proportional per-bet Kelly when the optimizer fails or only one bet qualifies
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

**About injuries:**

Availability is modelled inside the prediction, not bolted on afterwards. Each team's current rotation (regulars and their typical minutes, from the player game logs) is matched to the ESPN injury report; the expected share of missing rotation minutes lowers that team's Elo (`ELO_AVAILABILITY_SCALE`, about 3 points of win probability for a star) and feeds the GBM's availability features, which were trained on five seasons of real "who did not play" history. The `home_injury_adj` / `away_injury_adj` values you may see in the API are the old heuristic estimate and are display-only (`APPLY_POST_HOC_INJURY_ADJUSTMENT = False`). Two things to know: a player who missed the entire prior stretch is not a "regular", so listing him changes nothing (the team's rating already reflects his absence); and for the first ~5 games of a season the regulars come from last season's final games, so the term is weak until the new rotation has played.

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

Based in Europe? NBA games tip off between 16:00 and 03:30 UTC, mostly while you're asleep. The repo ships a GitHub Actions workflow that runs on GitHub's infrastructure instead. It writes JSONL snapshot records and commits them back to `main`; you pull and import them locally the next morning.

**One-time setup:**

1. Push the repo to GitHub (if not already).
2. Go to **Repo → Settings → Actions → General → Workflow permissions** and select **"Read and write permissions"**. (The workflow declares `contents: write` and `actions: write`, but the repo setting is still required.)
3. Visit the **Actions** tab and confirm the workflow named **snapshot-odds** appears. The first run may need a manual "Enable workflow" click.
4. Click **Run workflow** → **Run workflow** once to smoke-test end-to-end. You should see a new commit `chore(snapshots): YYYY-MM-DD HH:MMZ [skip ci]` and a file in `data/odds_snapshots/`.

**Daily usage:**

```bash
python3 -m nba_betting import-snapshots --pull   # One-shot: fetch origin/main + load JSONL → DB
```

`--pull` fetches `origin/main` and fast-forwards your checkout to it before importing, so snapshots committed overnight by the GitHub Actions runner land in your local working copy and then in your local `odds_snapshots` table in a single command. If it can't fast-forward (e.g. you're on a branch with local commits), it prints a warning and still imports whatever files already exist locally — safe to run daily.

`import-snapshots` is **idempotent**: a record's key is `(home_team_id, away_team_id, source, timestamp)` — one capture of one matchup — so rerunning it on any cadence imports nothing twice. Each record is matched to its game from the record's own ET `game_date`; records whose game isn't in the DB yet (future or preseason games) are stored unlinked and linked by the next `sync`. You can also point it at a single file:

```bash
python3 -m nba_betting import-snapshots --path data/odds_snapshots/2026-04-18.jsonl
```

**How the workflow runs (a self-paced loop, not a cron grid).** Until 2026-10 the workflow used ~33 cron slots per day. GitHub's scheduler delivered about 6 of them, often hours late (no runs at all 09:00–15:00 UTC; runs at 04:00–09:00 UTC when nothing was scheduled), so the last capture before a 7 PM ET tip was a median **97 minutes** old. Now each run is one long job (`snapshot-loop`) that paces itself on the next tip-off:

| Next tip-off | Odds capture every | Injury list every |
|---|---|---|
| more than 12 h away | — (run stops: "idle") | — |
| 3–12 h | 30 min | 1 h |
| 1–3 h | 15 min | 15 min within 2 h, else 1 h |
| under 1 h | 5 min | 15 min |

A run lasts up to 5.5 h (GitHub's job limit is 6 h), pushing its new files every 30 min. If games are still ahead when its time is up, it dispatches the next run itself (`workflow_dispatch` starts within seconds), so the chain covers a game day from morning to the last West Coast tip. After the last tip, the run stops without a successor. An hourly cron (`:17`) restarts the chain the next day; cron firings that land while a run is active wait as "pending" and get cancelled when the successor is queued, so a few cancelled runs per day in the Actions tab are normal. Unchanged lines are written at most every 30 min (heartbeat), which keeps the files small without losing any line value. The slate comes from ESPN only (`--skip-nba-api`): stats.nba.com never answers GitHub's IPs, and waiting out its timeouts used to cost ~4.6 min per capture.

Logic: [`nba_betting/data/snapshot_loop.py`](nba_betting/data/snapshot_loop.py); workflow: [.github/workflows/snapshot-odds.yml](.github/workflows/snapshot-odds.yml); commit/push: [.github/scripts/commit-snapshots.sh](.github/scripts/commit-snapshots.sh) (odds files merge as a union via `.gitattributes`, so concurrent appends never conflict).

**ESPN odds** carry the real DraftKings moneyline (de-vigged) plus spread and total; the raw American odds are stored in the JSONL too. If ESPN lists only a spread, the record keeps the spread/total and leaves `home_prob` empty. Before 2026-10 ESPN had moved its moneyline field and none were parsed, so every stored ESPN `home_prob` was a 2.5%/point spread proxy. `repair-snapshots` and the importer blank those out.

> **GitHub's 60-day inactivity rule:** scheduled workflows are disabled after 60 days without repo activity. The injury list changes almost daily, so the workflow keeps committing through the offseason and should stay enabled. If it is ever disabled, re-enable it in the Actions tab and trigger a manual run.

#### Daily Injury Snapshots (same workflow)

The same loop also captures the **full ESPN injury list** for the NBA (ET) day to `data/injury_snapshots/<date>.jsonl`. **Each team's lines freeze at its tip-off**: they keep updating until the team's game starts, then stay as they were at the last pre-tip capture (tip times come from ESPN's scoreboard). Training joins this file to the same day's games, so a list refreshed after tip-off used to leak injuries suffered *in* that game into the "pre-game" features. If the scoreboard can't be read, an existing day file is left untouched. The file isn't rewritten (no commit) when nothing changed. This is what grows the `historical_injuries` table, the training-side source of the `injury_impact_*` features.

`import-snapshots` (with or without `--pull`) loads the injury files too, replacing each day's rows in `historical_injuries` (idempotent). To capture locally instead of via GitHub:

```bash
python3 -m nba_betting snapshot-injuries            # refresh injuries.json + today's historical_injuries rows
python3 -m nba_betting snapshot-injuries --jsonl data/injury_snapshots   # DB-free file mode
```

#### One-off: repair snapshot rows written before 2026-10

```bash
cp data/nba_betting.db data/nba_betting.backup.db
python3 -m nba_betting repair-snapshots --dry-run   # report only
python3 -m nba_betting repair-snapshots
```

Removes duplicate rows (from the old import key), re-files closing lines that the old resolver attached to the *next* game of a playoff series (re-matched from the JSONL files), re-dates old local `predict` captures from their game id, and blanks the fake ESPN probabilities. Idempotent; a DB built from scratch with current code doesn't need it.

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

### Scoring Against the Closing Line

```bash
python3 -m nba_betting market-eval
```

Joins walk-forward out-of-fold predictions to the **real closing lines** captured by the snapshot cron and scores model vs market on the same games: moneyline Brier / log-loss with the shrinkage weight λ that minimises log-loss (vs the live `MARKET_SHRINKAGE_LAMBDA`), spread and total MAE, and the hit rate of the model's spread/total picks against the 52.4% break-even. Every section prints `n`; the verdict is withheld until ~300 games have a closing line (`--min-games`), roughly one season of the cron. Run it each off-season — it is the harness that decides λ and whether spread/total picks earn a stake. As of 2026-09 (126 games): the model ties the closing line on the moneyline, shrinking helps, λ=0.6 is within noise of the best, and the total head is worse than the book.

### Backtesting (Historical Simulation)

To see how the strategy would have performed on past data:

```bash
python3 -m nba_betting backtest
python3 -m nba_betting backtest --bankroll 5000 --splits 3
```

Reports: win rate, ROI, Sharpe ratio (annualized return-based, normalized to ~1000 bets/year), max drawdown, and per-signal breakdown.

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
backtest's actual resolved bets**. The inner simulation loop is fully
vectorized — 60,000 simulations × 200 bets completes in ~0.2 seconds. Each simulated bet draws a real
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

Reports how many days of injury and odds-snapshot data have accumulated. The `injury_impact_*` and line-movement features are forward-accumulating — they start at zero and become meaningful only after enough live-season data exists (the player-availability features are different: they are real for every historical game since 2026-09). Use this command monthly to know when it's worth retraining:

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

183 fast unit tests (~4 s); the main files:
- **`test_new_features.py`** (16): shrinkage invariants, `humanize_feature` label map, spread/total pick sign convention, driver attribution ordering, backtest `apply_live_strategy` default coupling, and additive DB migration idempotence.
- **`test_improvements.py`** (15): rolling stats, Four Factors, Elo; portfolio optimizer exposure cap and negative-EV behaviour.
- **`test_tier_improvements.py`** (14): off/def Elo asymmetry, SOS-adjusted stats, EWM weighting, meta-learner round-trip, signal-dependent Kelly monotonicity, portfolio exposure cap, vectorized opponent-DREB, odds-snapshot dedup, Polymarket fuzzy name matching, model cache mtime invalidation.
- **`test_montecarlo.py`** (12): empirical bootstrap correctness, market-null behaviour, horizon-invariant log-growth metrics, reproducibility, input validation.
- **`test_simulate_horizon.py`** (8): data-driven horizon projection, density scaling, edge-case fallbacks.
- **`test_snapshot_jsonl.py`** (25): JSONL round-trip, idempotence on the capture key, game matching (late-tip captures stay on tonight's game; no guessing for preseason), legacy ESPN spread-proxy blanking, `repair-snapshots`, ESPN-only slate, timestamp after fetch, per-date ESPN odds, heartbeat dedupe.
- **`test_snapshot_loop.py`** (16): tip-paced cadence, idle vs budget stop (successor dispatch), fetch-failure retries, error tolerance, push throttling.
- **`test_espn_odds.py`** (15): moneyline parsing from ESPN's current and legacy shapes, de-vig, spread-proxy flagging, single-source `prob_movement`, no bets priced off the spread proxy.
- **`test_injury_jsonl.py`** (10): ET-dated injury files, per-team tip-off freeze, scoreboard tip detection, idempotent import.
- **`test_playoff_sync_and_resolve.py`** (10): play-in/playoff game union, `update_results` date matching, `record_predictions` ET-date filing.

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
| First time | `sync --seasons 5` then `train` | Load data and build model |
| Start of season | `sync --seasons 5` then `train` | Retrain with fresh data (the season string is derived from the date — nothing to edit) |
| Daily (morning) | `sync` | Get yesterday's results, update Elo |
| Before games | `predict` | Get today's recommendations |
| Every 30–60 min (season) | `snapshot-odds` | Capture line movement; run on a cron (or GitHub Actions — see §Step 2b) |
| Daily (morning, EU users) | `import-snapshots --pull` | Git-pull + load the odds AND injury JSONL snapshots written overnight by GitHub Actions |
| After games | `sync` then `performance` | Check results and accuracy |
| After games | `clv` | Review Closing Line Value skill metric |
| Weekly | `backtest --real-odds` | Realistic ROI estimate with shrinkage applied |
| Monthly | `train` | Retrain model with latest data |
| Monthly | `sync-players` | Update player rosters and depth charts |
| Monthly | `readiness-status` | Check if injury/odds features have enough data to retrain |
| As needed | `diagnose` | Debug issues with predictions |
| After any code change | `pytest tests/ -v` | Guard against silent regressions (183 tests) |

---

## Data Flow

```
NBA.com (team + player game logs) ──┐
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
  nba_betting.db          # SQLite database (games, team + player game logs, Elo, odds, injury history)
  injury_snapshots/       # Daily ESPN injury JSONL from the GitHub Actions cron (committed)
  odds_snapshots/         # Odds JSONL from the cron (committed)
  prediction_history.json # Prediction tracking for performance analysis
  injuries.json           # Current injury list (ESPN + manual overrides)

trained_models/
  gbm_latest.joblib         # Trained GBM base model
  gbm_calibrated.joblib     # Isotonic-calibrated model (wraps the base) — what predict runs
  feature_cols.joblib       # Feature column order
  feature_means.joblib      # Training means for NaN imputation
  ensemble_weight.joblib    # Out-of-fold-learned Elo weight for the log-odds blend
  ensemble_meta.joblib      # Stacked logistic meta-learner (fitted, not wired into predict)
  spread_regressor.joblib   # Margin regression head
  total_regressor.joblib    # Total regression head
  regressor_feature_cols.joblib
```

---

## All Commands Reference

```bash
# Data & model
python3 -m nba_betting sync --seasons 5         # Fetch game data + compute Elo
python3 -m nba_betting train                     # Train GBM model + calibrate
python3 -m nba_betting sync-players              # Sync player rosters from ESPN

# Predictions
python3 -m nba_betting predict                   # Today's recommendations + explanations
python3 -m nba_betting predict --bankroll 5000   # Custom bankroll
python3 -m nba_betting snapshot-odds             # Snapshot current odds (run on a cron)
python3 -m nba_betting snapshot-odds --jsonl data/odds_snapshots  # DB-free, one capture
python3 -m nba_betting snapshot-loop --skip-nba-api --commit-cmd CMD  # Self-paced capture loop (what GitHub Actions runs)
python3 -m nba_betting repair-snapshots --dry-run # One-off cleanup of pre-2026-10 snapshot rows (drop --dry-run to apply)
python3 -m nba_betting import-snapshots --pull   # Git-pull + load odds + injury JSONL snapshots from GH Actions (daily)
python3 -m nba_betting import-snapshots          # Same, but without the git pull step
python3 -m nba_betting snapshot-injuries         # Today's ESPN injury list -> historical_injuries (or --jsonl DIR for file mode)
python3 -m nba_betting market-eval               # Model vs real closing lines (moneyline λ, spread/total picks); needs ~300 lined games

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
pytest tests/ -v                                 # 183 unit tests (run after any code change)

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
