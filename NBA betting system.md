# Data-Driven NBA Prediction and Polymarket Betting System

## 1. Introduction: sports betting as quantitative finance

Sports betting, when approached rigorously, is a quantitative finance problem. The bettor acts as a market maker or trader seeking mispriced contracts—binary options on game outcomes—where the true probability diverges from the market-implied probability. Prediction markets like **Polymarket** make this analogy explicit: each outcome trades as a share priced between $0.00 and $1.00, with the price directly representing the crowd's implied probability. A share priced at $0.65 implies a 65% probability; if your model estimates 72%, you have identified an edge.

Polymarket operates a **Central Limit Order Book (CLOB)** on the Polygon blockchain, offering NBA moneylines, spreads, totals, and player props as binary yes/no contracts. Unlike traditional sportsbooks with built-in margins of 4–8%, Polymarket charges variable fees averaging **0.8–1.5%** (formula: `fee = 0.0625 × price × (1 - price)`), making it a more favorable venue for sophisticated bettors. Marquee NBA games generate **$500K–$2M in volume**, providing sufficient liquidity for meaningful bet sizes.

The central thesis of this project is that a well-calibrated probabilistic model, combining modern machine learning with domain-specific feature engineering, can identify systematic mispricings in Polymarket's NBA markets. The academic literature is sobering—prediction markets are generally efficient—but research also identifies specific pockets of inefficiency: small markets, slow injury-news incorporation, and derivative markets. A disciplined system that exploits even a **2–5% edge** with proper bankroll management can generate meaningful risk-adjusted returns.

---

## 2. Problem definition: edge identification and risk control

The system's objective decomposes into three linked sub-problems, each with precise mathematical formulation.

**Probability estimation.** For each upcoming NBA game, estimate P(home win) with higher accuracy than the Polymarket implied probability. The model must produce *calibrated* probabilities—when it says 70%, the home team should win approximately 70% of such games. Discrimination (ranking games by likelihood) alone is insufficient; calibration drives profitable bet sizing.

**Edge detection and +EV identification.** Define the edge as: `Edge = P_model × decimal_odds - 1`. A bet is positive expected value (+EV) when Edge > 0, meaning the model's probability exceeds the market's implied probability. In practice, a **minimum edge threshold of 2%** filters noise and accounts for execution costs. The system should rank all available bets by expected value, present confidence intervals around probability estimates, and flag games where the model disagrees most with the market.

**Bankroll management and risk control.** Size bets using a fractional Kelly Criterion to maximize long-term geometric growth while limiting drawdown risk. The system targets a **risk of ruin below 1%**, caps individual bets at 5% of bankroll, and limits total simultaneous exposure to 25%. The tension between growth maximization and ruin avoidance is the core risk management challenge.

---

## 3. Literature and methodology review

### ELO ratings: the foundational baseline

The Elo rating system, adapted from chess, assigns each team a numerical strength parameter and predicts outcomes via a logistic function: **P(A beats B) = 1 / (1 + 10^(-(R_A - R_B)/400))**. After each game, ratings update by `R_new = R_old + K × (actual - expected)`.

FiveThirtyEight's NBA Elo implementation, documented by Nate Silver and Reuben Fischer-Baum (2015), represents the most prominent application. Their system uses a **K-factor of 20** (surprisingly high, implying NBA results contain genuine signal about team quality), a constant **100-point home-court bonus** (~3.5 points), and a nonlinear margin-of-victory multiplier: `G = ((MOV + 3)^0.8) / (7.5 + 0.006 × (Elo_winner - Elo_loser))`. The denominator corrects autocorrelation—strong teams are expected to win by more, so blowouts carry less new information. Year-to-year carryover retains 75% of end-of-season Elo. An independent replication (nicidob.github.io) found pure Elo correctly picks NBA winners approximately **67–68% of the time**, with remarkable robustness across parameter variations. The conversion to point spread is: `Spread = (Elo_diff + 100) / 28`, where every 28 Elo points equals ~1 NBA point.

### Bradley-Terry models and their ELO equivalence

The Bradley-Terry (BT) model (Bradley & Terry, 1952) is mathematically equivalent to Elo: each team receives a log-strength parameter β_i, and `P(i beats j) = 1 / (1 + exp(-(β_i - β_j)))`. With a scale factor of 400 and base 10, BT *is* Elo. The model's power lies in its extensions. **Cattelan, Varin & Firth (2013, JRSS-C)** introduced dynamic Bradley-Terry models with time-varying abilities for NBA data, modeling strength evolution via random walks. **Király & Qian (2017)** unified BT's explanatory power with Elo's online learning, enabling feature incorporation without losing parsimony. The BT framework also supports formal hypothesis testing—for instance, testing whether home-court advantage α significantly differs from zero.

### Bayesian updating for game predictions

Bayesian approaches model team strength as latent variables with prior distributions, updating posteriors as games are observed. **Glickman & Stern (1998)** developed the seminal Bayesian state-space model where team strengths follow a first-order autoregressive process, capturing both within-season fluctuation and between-season personnel changes. **Manner (2016)** applied this directly to NBA data and found that forecast combinations of model predictions and betting odds yielded slight improvements over either alone—but confirmed that **beating the market remains extremely difficult**.

Bayesian networks offer a complementary approach. **Alameda-Basora (2019, Iowa State)** built an expert Bayesian network for in-game NBA prediction using directed acyclic graphs with conditional probability tables. The model updated probabilities sequentially (after Q1, Q2, Q3) and was the **only strategy yielding positive profit** in live Over/Under betting during the 2018–19 season. **Arbel et al. (2022)** proposed beta-conjugate priors for in-game win probability, with time-dependent prior specification that outperformed both MLE and standard Bayesian estimators on Brier score.

### Monte Carlo simulation for season and game modeling

FiveThirtyEight's projections simulate the remaining NBA schedule **50,000 times**, running "hot"—after each simulated game, team ratings adjust based on the simulated result, which informs subsequent simulated games. This captures cascading uncertainty: a key injury doesn't just affect one game but ripples through the schedule. **Garnica-Caparrós et al. (2023, Winter Simulation Conference)** extended this with agent-based logic incorporating tanking incentives and rest management, evaluated across 10 NBA seasons. **Štrumbelj & Vračar (2012, International Journal of Forecasting)** developed a homogeneous Markov model for basketball match simulation as an alternative stochastic framework.

### Machine learning approaches: what the papers report

The ML literature on NBA prediction is extensive. **Loeffelholz, Bednar & Bauer (2009, JQAS)** pioneered neural network application, achieving **74.33% accuracy** via Bayesian belief network fusion of multiple neural nets—outperforming human experts at 68.67%. More recently, a **PLOS ONE study (2024)** found XGBoost outperformed KNN, LightGBM, SVM, Random Forest, Logistic Regression, and Decision Trees across five metrics, with SHAP analysis identifying **field goal percentage, defensive rebounds, and turnovers** as consistently dominant features. A **Scientific Reports/Nature study (2025)** evaluated stacking ensembles across seven classifiers on 2021–2024 NBA data.

The literature converges on consistent accuracy ranges for pre-game prediction:

- Simple baselines (always pick home team): ~60%
- Elo rating systems: 65–68%
- Standard ML (logistic regression, SVM, random forest): 62–68%
- Advanced ML (XGBoost, neural nets, stacking): 65–74%
- Betting markets (implied by lines): 69–73%

**Pre-game prediction accuracy typically plateaus around 65–70%** using team-level aggregate statistics, reflecting irreducible uncertainty in NBA outcomes. Oliver's "Four Factors"—effective field goal percentage, turnover percentage, offensive rebound percentage, and free throw rate—appear consistently as foundational predictive features.

### Point spread and win probability modeling

**Stern (1994, JASA)** established that NBA score differences follow Brownian motion with drift. The conversion from point spread to win probability uses a logistic function: `Win% = 1 / (1 + exp(-k × PointSpread))` where **k ≈ 0.14–0.16** for the NBA. Empirically, a 1-point favorite wins ~54% of the time; a 7-point favorite wins ~72%. **Lopez, Matthews & Baumer (2018, Annals of Applied Statistics)** found the NBA has the **largest talent gap** among major North American sports (σ ≈ 0.274), meaning the best team wins most often in basketball.

### Decision theory: Kelly Criterion and expected value

The Kelly Criterion (Kelly, 1956) maximizes long-run geometric growth. For binary bets: **f* = (bp - q) / b**, where b = net decimal odds, p = win probability, q = 1 - p. **Fractional Kelly is universally preferred in practice**: half-Kelly retains ~75% of expected growth with only ~25% of variance, while quarter-Kelly achieves ~51% of growth with ~1/11th of variance. Thorp (2006) demonstrated the criterion's application across blackjack, sports betting, and financial markets. **Baker & McHale (2013, Decision Analysis)** showed how to optimally bet under parameter uncertainty, formally adjusting Kelly for estimation error. For simultaneous bets, the multi-bet Kelly problem requires numerical optimization—individual Kelly fractions cannot be applied independently.

---

## 4. Data sources and feature engineering

### The nba_api library: 253+ endpoints, zero cost

The `nba_api` Python library (v1.11.4, MIT license) wraps NBA.com's undocumented stats API, providing the most comprehensive free NBA data source available. No API key is required. Key endpoint categories for prediction modeling include:

- **TeamEstimatedMetrics**: Returns E_OFF_RATING, E_DEF_RATING, E_NET_RATING, E_PACE—the single most valuable endpoint for team-level prediction features
- **LeagueGameLog**: All games with full box score lines (FGM, FGA, FG%, 3P%, FT%, REB, AST, STL, BLK, TOV, PTS, PLUS_MINUS), filterable by team or player
- **BoxScoreAdvancedV3**: Per-game advanced stats including ORtg, DRtg, TS%, eFG%, AST%, USG_PCT
- **BoxScoreFourFactorsV3**: Oliver's Four Factors for both teams per game
- **TeamDashboardByGeneralSplits**: Home/away splits, win/loss splits, monthly breakdowns
- **PlayByPlayV3**: Full play-by-play data for possession-level analysis
- **ScoreboardV3** and live endpoints: Real-time game data during play

**Critical operational constraints**: NBA.com blocks requests from AWS, Heroku, Google Cloud, and most cloud provider IPs. All data collection must run locally or through residential proxies. Rate limiting requires ~2–3 second delays between requests. The recommended pattern is batch-downloading to a local database, never making real-time calls from production servers. NBA.com also frequently deprecates endpoints without notice (PlayByPlayV2, ScoreboardV2 both recently removed), making fallback sources essential.

### Basketball Reference: the gold standard for historical data

Basketball Reference provides the most accurate, historically deep basketball statistics available, accessible via web scraping (20 requests/minute limit). The `basketball_reference_scraper` Python library handles rate limiting automatically and provides access to team ratings (ORtg, DRtg, SRS, MOV), advanced player metrics (PER, TS%, Usage Rate, WS, BPM, VORP), the Four Factors, and critically, **injury reports**—the only free source for this data. Historical coverage extends to the 1946–47 season.

### Supplementary free APIs

**BallDontLie** (balldontlie.io) offers a free tier with 5 requests/minute covering basic game scores, player stats, standings, and season averages. Advanced stats require a paid tier ($9.99/month). **API-Sports** provides 100 requests/day across all NBA endpoints but lacks advanced analytics. **SportRadar** offers a 30-day trial with 1,000 requests but production pricing starts at **$500+/month**—not viable as a free source. **SportsData.io**'s free tier returns scrambled (fake) data, rendering it useless for model building.

### Polymarket integration: programmatic odds access

Polymarket exposes three APIs, all with public read access requiring no authentication:

| API | Base URL | Purpose |
|-----|----------|---------|
| **Gamma API** | `gamma-api.polymarket.com` | Market discovery, NBA events, tags |
| **CLOB API** | `clob.polymarket.com` | Live prices, orderbooks, midpoints |
| **Data API** | `data-api.polymarket.com` | Volume, trades, open interest |

The integration workflow is: (1) query Gamma API `/sports` for NBA series/tag IDs, (2) fetch active NBA events via `/events?series_id=<NBA_ID>&active=true`, (3) extract `clobTokenIds` from each market, (4) query CLOB API `/midpoint?token_id=<ID>` for the best probability estimate. The official Python SDK (`py-clob-client`, pip-installable) provides convenient wrappers. WebSocket connections at `wss://ws-subscriptions-clob.polymarket.com/ws/market` enable real-time price streaming. Rate limits are generous: 4,000 requests/10 seconds for Gamma, 100/10 seconds for individual price queries.

### Feature engineering taxonomy

**Team-level features** (primary predictive power): rolling offensive/defensive ratings (5, 10, 20-game windows), net rating, pace, effective FG%, turnover rate, offensive rebound rate, free throw rate (the Four Factors), home/away performance splits, and recent form (win streak, last-N performance). All rolling stats must **exclude the current game** to prevent data leakage.

**Contextual features**: rest days between games (computed from schedule data), back-to-back indicator, schedule density (games in trailing 7/14-day windows), travel distance (computed from arena coordinates via great-circle distance—no API provides this directly), and altitude effects. Home-court advantage has declined over time from ~5.8 points in 1987–88 to **~2.4 points** in recent seasons.

**Player-level features**: Usage rate, PER, plus-minus, and injury status. Injury impact is best modeled by estimating the win shares or VORP of unavailable players and adjusting team-level projections accordingly. Basketball Reference's injury report module is the best free source for this data.

---

## 5. Modeling approach: a concrete pipeline

### Feature selection methodology

The recommended approach uses **SHAP values** as the primary feature importance method, providing both global importance rankings and local explanations for individual predictions. Permutation importance serves as a model-agnostic cross-check. Recursive Feature Elimination (RFE) with XGBoost handles dimensionality reduction. Academic studies consistently identify the top predictive features as: field goal percentage, defensive rebounds, turnovers, three-point percentage, net rating, and rest days.

### Model architecture: baseline through advanced

**Baseline (Elo + logistic regression).** An Elo rating system with FiveThirtyEight's parameters provides a strong, interpretable starting point at 67–68% accuracy. Logistic regression on team-level features (Four Factors, recent form, home/away) serves as the ML baseline at ~62–65% accuracy. Both models produce well-calibrated probabilities by design.

**Primary model (XGBoost/LightGBM).** Gradient boosting on the full feature set represents the production model. XGBoost is preferred based on the PLOS ONE benchmarking study showing it outperforms all alternatives on NBA data. LightGBM serves as a secondary model for ensemble diversity. Hyperparameter tuning via Optuna with Bayesian optimization targets log-loss on validation data. Handle home-team class imbalance (~58–60% baseline) via `scale_pos_weight`.

**Ensemble.** Stack XGBoost and LightGBM predictions using a logistic regression meta-learner, or simply average their calibrated probabilities. The cmunch1/nba-prediction project demonstrates this approach with proper calibration integration.

### Probability calibration

Raw gradient boosting outputs require post-hoc calibration. Use scikit-learn's `CalibratedClassifierCV` with `method='sigmoid'` (Platt scaling) for the typical NBA dataset size (~1,200 games/season). Platt scaling fits two parameters to a logistic function on held-out calibration data, providing robust calibration with limited samples. Isotonic regression (`method='isotonic'`) offers more flexibility but requires **>1,000 calibration samples** to outperform Platt scaling (Niculescu-Mizil & Caruana, 2005, ICML). Evaluate with reliability diagrams, Brier score, and Expected Calibration Error (ECE).

### Edge detection: model vs. market

For each game, compute: `Edge = P_model × decimal_odds_polymarket - 1`. If Edge > 0.02 (2% threshold), flag as a betting candidate. The Kelly fraction is: `f* = Edge / (decimal_odds - 1)`. Apply fractional Kelly (λ = 0.25 to 0.5) for the actual bet size. Present the user with: model probability, Polymarket implied probability, edge magnitude, expected value per dollar, and recommended Kelly fraction.

### Assumptions and limitations by model type

| Model | Key Assumptions | Primary Limitation |
|-------|----------------|-------------------|
| Elo | Constant K-factor, logistic error distribution | Cannot incorporate injuries, roster changes, matchup-specific effects |
| Logistic Regression | Linear decision boundary in feature space | Misses nonlinear interactions between features |
| XGBoost/LightGBM | Features are informative, no severe concept drift | Requires careful tuning; can overfit to historical patterns that don't persist |
| Ensemble | Component models have diverse error patterns | Added complexity with diminishing accuracy gains |

---

## 6. Backtesting framework: rigorous validation

### Walk-forward validation is mandatory

Standard k-fold cross-validation violates temporal ordering and introduces lookahead bias. The only proper validation method for sports prediction is **walk-forward validation** (also called forward chaining or time-series cross-validation).

**Expanding window**: Train on seasons [1, ..., k], test on season k+1, then train on [1, ..., k+1], test on k+2. The training set grows monotonically. Best when more data improves the model and statistical relationships are stable. Implemented via scikit-learn's `TimeSeriesSplit`.

**Rolling window**: Train on the most recent N games (e.g., 2 seasons), test on the next block. The window slides forward with fixed size. Best when concept drift exists—recent data is more representative than historical data. Implemented via the `tscv.GapRollForward` package. A **gap** between train and test sets prevents leakage from lagged features.

### Evaluation metrics and benchmarks

| Metric | Formula | NBA Benchmark |
|--------|---------|--------------|
| **Accuracy** | Correct picks / Total games | >52.4% to break even at -110; 65–70% for good models |
| **Brier Score** | Mean of (predicted - actual)² | ~0.20 for competent NBA models; 0.25 for random |
| **Log Loss** | -Mean of [y·log(p) + (1-y)·log(1-p)] | ~0.58 for good NBA models; 0.693 for random |
| **Calibration (ECE)** | Weighted mean |predicted - observed| across bins | <0.05 is well-calibrated |
| **ROI** | Total profit / Total wagered | >0% after fees; 2–5% is excellent |
| **Sharpe Ratio** | Mean return / Std(returns) | >0.5 solid; >1.0 excellent |
| **Max Drawdown** | Largest peak-to-trough decline | Full Kelly: 50%+; Half-Kelly: ~25% |

### Betting simulation methodology

Simulate the complete betting pipeline on historical data: (1) generate model probabilities using only data available before each game, (2) compare against historical Polymarket odds (or proxy odds from closing lines), (3) apply edge threshold filtering (>2%), (4) size bets using fractional Kelly, (5) track bankroll evolution, drawdowns, and cumulative ROI. The `BacktestBuddy` and `sports-betting` Python packages provide ready-made frameworks for this simulation.

### Avoiding lookahead bias and overfitting

Five critical rules: **never use future data in features** (all rolling stats must use only past games); **use odds available at bet placement time**, not closing odds; **respect data availability timestamps** for injuries and lineups; **select hyperparameters only within training windows**, never on the full dataset; and **test across multiple seasons** to confirm the strategy isn't data-mined to a specific period. Adversarial validation using SHAP values—comparing feature importance between train and test distributions—provides an additional check against distribution shift.

---

## 7. System architecture: from model to web application

### Backend: FastAPI + scheduled pipeline

The backend uses **Python with FastAPI** for model serving, APScheduler for daily data pipeline execution, and PostgreSQL (Railway-hosted) for persistent storage. The core API endpoints include:

- `GET /predictions/today` — Returns all games with model probabilities, market odds, edge, EV, and Kelly fraction
- `GET /odds/compare/{game_id}` — Detailed model-vs-market comparison for a specific game
- `GET /recommendations/today` — Filtered +EV bets with sizing recommendations
- `GET /performance/history` — Historical accuracy, ROI, and calibration metrics

ML models load once at application startup via `joblib.load()`. Prediction endpoints use synchronous functions (not `async def`) since ML inference is CPU-bound. The daily pipeline—triggered by APScheduler at a fixed time or by GitHub Actions cron—fetches new game data from `nba_api` (running locally to avoid cloud IP blocks), scrapes injury reports from Basketball Reference, pulls current Polymarket odds via the Gamma/CLOB APIs, generates predictions, and stores results in PostgreSQL.

```
Data Flow:
nba_api (local) ──→ Feature Engineering ──→ Model Inference ──→ PostgreSQL
Basketball Ref ──↗         ↑                      ↓
Polymarket API ──→ Odds Comparison ←── Model Probabilities ──→ API Response
```

### Frontend: Next.js dashboard

A **Next.js application deployed on Vercel** provides the user interface. Key components include: game prediction cards (team logos, model probability gauge, market odds comparison), a sortable recommendations table (matchup, model prob, market prob, edge, EV, Kelly fraction, recommendation badge), a calibration chart (reliability diagram updated weekly), and a performance tracker (cumulative ROI, P/L curve, accuracy over time). The frontend fetches data from the FastAPI backend and refreshes on a configurable interval (e.g., every 30 seconds for live odds).

### Deployment and DevOps

| Component | Platform | Rationale |
|-----------|----------|-----------|
| Frontend | **Vercel** | Zero-config Next.js deployment, free tier, auto-deploys from GitHub |
| Backend | **Railway** | Simple container deployment, managed PostgreSQL, environment variables |
| Data Pipeline | **GitHub Actions** | Free scheduled jobs, no infrastructure to manage |
| ML Training | **Local / GitHub Actions** | GPU not required for gradient boosting; weekly retraining via cron |

The repository follows a monorepo structure with path-filtered CI: `backend/` triggers backend CI, `frontend/` triggers frontend CI, and `ml/` triggers model retraining workflows. Start with SQLite during development and migrate to PostgreSQL for production deployment.

---

## 8. Limitations, risks, and honest assessment

### Market efficiency is the fundamental obstacle

The academic evidence is clear: prediction markets are generally efficient. **Winkelmann et al. (2024, Journal of Sports Economics)** analyzed 155,000+ contests across 16 seasons and concluded that "no odds-based betting strategy will yield statistically significant long-term profits" after accounting for transaction costs. Even Polymarket, which claims **94% accuracy** one month before event resolution, benefits from rapid information aggregation across thousands of participants. Any systematic edge is likely to be small (2–5%), requiring hundreds of bets to confirm statistically and thousands to generate meaningful profit.

### Overfitting is the second-greatest threat

With ~1,230 NBA regular-season games per year, the feature-to-sample ratio is unfavorable for complex models. Adding features beyond the core ~10–15 risks fitting noise. Temporal overfitting—finding patterns that existed in 2019–2023 but not 2024+—is especially dangerous. Mitigation requires strict walk-forward validation, fractional Kelly sizing (which automatically reduces exposure when the model is wrong), and monitoring for model degradation via rolling accuracy and calibration checks.

### Data quality and operational risks

NBA.com's undocumented API can break without notice—endpoints have been deprecated mid-season (PlayByPlayV2, ScoreboardV2). Cloud IP blocking means the data pipeline must run locally or through residential proxies, creating a single point of failure. Injury report timing is imprecise; by the time lineups are confirmed, odds may have already adjusted. Polymarket odds can be manipulated by whales—during the 2024 US election, four accounts with ~$30M in positions caused price movements "larger than justified" (Nate Silver's assessment).

### Bankroll risk and gambler's ruin

Even with a genuine edge, variance can destroy a bankroll. At a 55% win rate on even-money bets with 10% bet sizing, the probability of ruin is **~13.4%**. Reducing to 2% bet sizing drops ruin probability to **~0.02%**. Full Kelly sizing produces 50%+ drawdowns routinely. The system must enforce hard limits: no more than 5% of bankroll on any single bet, no more than 25% total exposure, and fractional Kelly (0.25–0.50) on all positions.

### Legal and regulatory landscape

Polymarket acquired the CFTC-licensed QCEX exchange in December 2025, enabling regulated U.S. access for event contracts. However, the regulatory landscape for prediction market trading remains evolving. Users should verify their jurisdiction's laws regarding prediction market participation. Sports betting regulations vary by state, and the intersection of prediction markets with gambling law is not fully settled. This system should be used for research and paper trading initially.

---

## 9. Implementation roadmap: six phases to production

**Phase 1: Data collection and exploratory analysis (Weeks 1–2).** Set up the `nba_api` data pipeline to pull TeamEstimatedMetrics, LeagueGameLog, BoxScoreAdvancedV3, and BoxScoreFourFactorsV3 for the past 5 seasons. Build the Basketball Reference injury scraper. Implement the Polymarket Gamma/CLOB API integration. Store everything in SQLite. Conduct EDA: distribution of point differentials, home-court advantage trends, feature correlations, and missing data patterns.

**Phase 2: Baseline model (Weeks 3–4).** Implement the FiveThirtyEight Elo system with K=20, 100-point home bonus, MOV multiplier, and 75% year-to-year carryover. Build a logistic regression model on the Four Factors plus home/away and rest days. Establish baseline metrics: ~67% accuracy (Elo), ~64% accuracy (logistic regression), Brier scores, and calibration plots. These baselines define the bar that advanced models must clear.

**Phase 3: Advanced model and feature engineering (Weeks 5–7).** Engineer the full feature set: rolling team ratings (5/10/20-game windows), schedule density, travel distance, back-to-back indicators, injury-adjusted ratings, and head-to-head matchup history. Train XGBoost and LightGBM with Optuna hyperparameter optimization targeting log-loss. Apply Platt scaling calibration via CalibratedClassifierCV. Build the stacking ensemble. Run SHAP analysis to validate feature importance and prune uninformative features.

**Phase 4: Backtesting and validation (Weeks 8–9).** Implement walk-forward validation across 3+ seasons. Simulate the full betting pipeline with historical odds: edge detection, Kelly sizing, bankroll tracking. Compute all metrics: accuracy, Brier score, log-loss, calibration plots, ROI, Sharpe ratio, max drawdown. Compare against a "bet the market" baseline. If the model shows no consistent edge after rigorous backtesting, iterate on features and model architecture before proceeding.

**Phase 5: Web application MVP (Weeks 10–12).** Build the FastAPI backend with prediction, odds comparison, and recommendation endpoints. Implement the scheduled data pipeline (daily fetch + predict). Deploy the backend on Railway with PostgreSQL. Build the Next.js frontend with game cards, recommendation table, and performance tracker. Deploy on Vercel. Connect Polymarket live odds via the CLOB API.

**Phase 6: Live paper trading and iteration (Ongoing).** Run the system in paper-trading mode for at least one full month (~100+ games), tracking predictions against actual outcomes without risking capital. Monitor calibration drift, feature importance stability, and edge persistence. Compare model picks against Polymarket closing prices—consistently beating the closing line (positive CLV) is the strongest indicator of genuine skill. Only transition to real capital after demonstrating sustained positive ROI in paper trading across a statistically meaningful sample.

---

## Conclusion

Building a profitable NBA prediction system integrated with Polymarket is technically feasible but probabilistically challenging. The core tension is that prediction markets are efficient aggregators of information, and the academic literature consistently shows that beating them requires either superior data, superior modeling, or faster information processing—not just good models. The **realistic accuracy ceiling for pre-game NBA prediction is 65–70%** using public data, while the market itself operates at 69–73% implied accuracy.

The most promising path to edge lies not in raw accuracy but in **superior calibration**. A model that is merely 66% accurate but perfectly calibrated can identify +EV bets that a 70%-accurate but poorly calibrated model would miss. This is why the pipeline emphasizes Platt scaling, reliability diagrams, and Brier score over raw accuracy. The recommended technology stack—`nba_api` for data, XGBoost with SHAP for modeling, Polymarket's CLOB API for odds, FastAPI + Next.js for serving—provides a complete, free, open-source foundation. But the hardest part is not engineering; it is developing genuine, persistent statistical edge in one of the world's most scrutinized prediction domains. Begin with paper trading, measure relentlessly, and let the data determine whether the edge is real.