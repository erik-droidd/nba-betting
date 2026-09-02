"""CLI entry point for the NBA betting recommendation system."""

import typer
from rich.console import Console

from nba_betting.config import DEFAULT_BANKROLL, CURRENT_SEASON

app = typer.Typer(
    name="nba-betting",
    help="NBA betting recommendation system with Elo ratings and Polymarket integration.",
)
console = Console()


@app.command()
def predict(
    bankroll: float = typer.Option(DEFAULT_BANKROLL, help="Current bankroll in dollars"),
    model: str = typer.Option("auto", help="Model to use: elo, xgb, ensemble, auto"),
) -> None:
    """Generate betting recommendations for today's NBA games."""
    from nba_betting.data.nba_stats import fetch_todays_games, fetch_upcoming_games
    from nba_betting.data.polymarket import get_nba_odds
    from nba_betting.models.elo import get_current_elos, predict_home_win_prob
    from nba_betting.models.xgboost_model import load_model
    from nba_betting.models.calibration import load_calibrated_model
    from nba_betting.models.ensemble import ensemble_predict
    from nba_betting.betting.recommendations import generate_recommendations
    from nba_betting.display.console import display_recommendations, display_no_games

    console.print("[dim]Fetching today's scheduled games...[/dim]")
    games = fetch_todays_games()

    if not games:
        # No scheduled games left today — look ahead for the next game day
        console.print("[dim]No scheduled games remaining today. Looking for the next game day...[/dim]")
        games = fetch_upcoming_games(days_ahead=7)
        if games:
            next_day = games[0].get("game_time_utc", "")[:10] or "the next game day"
            console.print(f"[yellow]Showing {len(games)} game(s) scheduled for {next_day}.[/yellow]")

    if not games:
        display_no_games()
        return

    console.print(f"[dim]Found {len(games)} game(s). Fetching Elo ratings...[/dim]")
    elos = get_current_elos()

    if not elos:
        console.print("[red]No Elo ratings found. Run 'nba-betting sync' first.[/red]")
        raise typer.Exit(1)

    # Off/def split Elos — the model was trained on these (build_feature_matrix
    # joins the stored split ratings). Without them the prediction-time off/def
    # features degrade to the aggregate Elo, diverging from training AND from
    # the API path (which already passes them). Load once; injected per game.
    try:
        from nba_betting.models.elo import get_current_off_def_elos
        off_def_elos = get_current_off_def_elos()
    except Exception:
        off_def_elos = {}

    # Determine which model to use and build the shared prediction engine
    # (single source of truth for both predict paths — see
    # nba_betting/prediction_service.py; replaces the closure formerly
    # duplicated here and in api/routes.py, which had diverged — #29).
    use_model = model
    predict_fn = None
    engine = None

    if use_model in ("auto", "xgb", "ensemble"):
        console.print("[dim]Computing features for prediction...[/dim]")
        from nba_betting.prediction_service import PredictionEngine
        engine = PredictionEngine(games, off_def_elos, blend=(use_model != "xgb"))
        if engine.available:
            if use_model == "auto":
                use_model = "ensemble"
                console.print("[dim]Using ensemble model (Elo + XGBoost).[/dim]")
            predict_fn = engine.predict
        else:
            use_model = "elo"
            console.print("[dim]No trained XGBoost found. Using Elo model only.[/dim]")
            console.print("[dim]Run 'nba-betting train' to build the XGBoost model.[/dim]")

    # Sync injuries from ESPN
    console.print("[dim]Syncing injuries from ESPN...[/dim]")
    try:
        from nba_betting.data.injuries import sync_injuries_from_espn, load_injuries
        injuries = sync_injuries_from_espn()
        injured_count = len([i for i in injuries if i.status in ("Out", "Doubtful")])
        console.print(f"[dim]Loaded {len(injuries)} injuries ({injured_count} Out/Doubtful).[/dim]")
    except Exception as e:
        console.print(f"[yellow]Could not sync ESPN injuries: {e}[/yellow]")
        from nba_betting.data.injuries import load_injuries
        injuries = load_injuries()

    # Check starting lineups (ESPN probables, ~30min before tip)
    try:
        from nba_betting.data.lineups import fetch_probable_starters, apply_lineup_bumps
        starters = fetch_probable_starters()
        if starters:
            injuries = apply_lineup_bumps(injuries, starters)
            console.print(f"[dim]Lineup data found for {len(starters)} team(s) — injury impacts updated.[/dim]")
    except Exception:
        pass  # Non-critical; lineups may not be available yet

    # Fetch Polymarket odds
    console.print("[dim]Fetching Polymarket odds...[/dim]")
    try:
        market_odds = get_nba_odds()
        console.print(f"[dim]Found odds for {len(market_odds)} Polymarket market(s).[/dim]")
    except Exception as e:
        console.print(f"[yellow]Could not fetch Polymarket odds: {e}[/yellow]")
        market_odds = []

    # Fetch ESPN odds (fallback + spread/OU data)
    console.print("[dim]Fetching ESPN odds...[/dim]")
    try:
        from nba_betting.data.espn_odds import get_espn_odds
        espn_odds_data = get_espn_odds()
        console.print(f"[dim]Found odds for {len(espn_odds_data)} ESPN game(s).[/dim]")
    except Exception as e:
        console.print(f"[yellow]Could not fetch ESPN odds: {e}[/yellow]")
        espn_odds_data = []

    # Snapshot odds for line movement tracking
    try:
        from nba_betting.data.odds_tracker import snapshot_current_odds
        snaps = snapshot_current_odds(games, market_odds, espn_odds_data)
        if snaps:
            console.print(f"[dim]Saved {snaps} odds snapshot(s).[/dim]")
    except Exception:
        pass  # Non-critical

    # Get line movement data. Snapshots are filed under each game's ET
    # date (`snapshot_game_date`, the same key as Game.date / the closing
    # line lookups), so query with that key per game.
    line_movements = {}
    try:
        from nba_betting.data.odds_tracker import get_line_movement, snapshot_game_date
        from nba_betting.data.nba_stats import today_et
        from nba_betting.db.models import Team
        from nba_betting.db.session import get_session
        from sqlalchemy import select
        session = get_session()
        team_lookup = {t.abbreviation: t.id for t in session.execute(select(Team)).scalars().all()}
        session.close()
        _today = today_et()
        for g in games:
            h_id = team_lookup.get(g["home_team_abbr"])
            a_id = team_lookup.get(g["away_team_abbr"])
            if not (h_id and a_id):
                continue
            lm = get_line_movement(snapshot_game_date(g, _today), h_id, a_id)
            if lm.get("n_snapshots", 0) > 0:
                line_movements[(g["home_team_abbr"], g["away_team_abbr"])] = lm
    except Exception:
        pass  # Non-critical

    # Hand the engine the live context it predicts against (injuries +
    # line movements are finalized by now), then generate recommendations.
    if engine is not None:
        engine.injuries = injuries
        engine.line_movements = line_movements

    rolling_context = engine.rolling_context() if engine is not None else {}

    recommendations = generate_recommendations(
        games, elos, market_odds, bankroll,
        predict_fn=predict_fn,
        injuries=injuries,
        rolling_context=rolling_context,
        line_movements=line_movements,
        espn_odds=espn_odds_data,
        spread_total_predictions=engine.spread_total_predictions if engine else None,
        driver_contexts=engine.driver_contexts if engine else None,
        driver_model=engine.driver_model if engine else None,
        driver_feature_means=engine.feat_means if engine else None,
    )
    display_recommendations(recommendations, bankroll)

    # Auto-save predictions to history
    from nba_betting.betting.tracker import record_predictions
    saved = record_predictions(recommendations)
    if saved:
        console.print(f"[dim]Saved {saved} prediction(s) to history.[/dim]")


@app.command()
def train() -> None:
    """Train the XGBoost model with walk-forward validation."""
    from nba_betting.features.builder import build_feature_matrix
    from nba_betting.models.xgboost_model import (
        train_model, walk_forward_validate, save_model,
        _get_feature_cols, get_feature_importance,
    )
    from nba_betting.models.calibration import (
        calibrate_model, evaluate_calibration, save_calibrated_model,
    )
    import numpy as np

    console.print("[bold]Building feature matrix...[/bold]")
    X, y = build_feature_matrix()

    if X.empty:
        console.print("[red]No data available. Run 'nba-betting sync --seasons 3' first.[/red]")
        raise typer.Exit(1)

    feature_cols = _get_feature_cols(X)
    console.print(f"  {len(X)} games, {len(feature_cols)} features")
    console.print(f"  Home win rate: {y.mean():.1%}")

    # Walk-forward validation (return_oof=True so we can fit the meta-learner
    # on honest out-of-fold predictions rather than in-sample predictions)
    console.print("\n[bold]Walk-forward validation...[/bold]")
    results = walk_forward_validate(X, y, return_oof=True)

    if results["folds"]:
        from rich.table import Table
        table = Table(title="Walk-Forward Results", show_header=True, header_style="bold cyan")
        table.add_column("Fold", justify="center")
        table.add_column("Split", justify="center")
        table.add_column("Train", justify="right")
        table.add_column("Test", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Brier", justify="right")
        table.add_column("LogLoss", justify="right")

        for fold in results["folds"]:
            table.add_row(
                str(fold["fold"]),
                fold["split_date"],
                str(fold["train_size"]),
                str(fold["test_size"]),
                f"{fold['accuracy']:.1%}",
                f"{fold['brier_score']:.4f}",
                f"{fold['log_loss']:.4f}",
            )

        console.print(table)

        if results["aggregate"]:
            agg = results["aggregate"]
            console.print(
                f"\n[bold]Aggregate:[/bold] Accuracy={agg['accuracy']:.1%}, "
                f"Brier={agg['brier_score']:.4f}, LogLoss={agg['log_loss']:.4f} "
                f"({agg['total_games']} games)"
            )
    else:
        console.print("[yellow]Not enough data for walk-forward validation.[/yellow]")

    # Train final model on all data
    console.print("\n[bold]Training final model on all data...[/bold]")
    model = train_model(X, y)

    # Feature importance (permutation-based)
    console.print("\n[dim]Computing feature importance (permutation)...[/dim]")
    importance = get_feature_importance(model, feature_cols, X=X, y=y, top_n=10)
    console.print("\n[bold]Top 10 Features:[/bold]")
    max_imp = max((abs(imp) for _, imp in importance), default=1.0) or 1.0
    for name, imp in importance:
        bar_len = max(0, int((imp / max_imp) * 30)) if max_imp > 0 else 0
        bar = "#" * bar_len
        console.print(f"  {name:40s} {imp:+.4f} {bar}")

    # Calibrate using last 20% of data as calibration set
    console.print("\n[bold]Calibrating probabilities (isotonic regression)...[/bold]")
    n_cal = max(200, int(len(X) * 0.2))
    X_cal = X.iloc[-n_cal:][feature_cols]
    y_cal = y.iloc[-n_cal:].values
    X_train_part = X.iloc[:-n_cal][feature_cols]
    y_train_part = y.iloc[:-n_cal]

    # Retrain on the non-calibration portion for a proper calibration
    from nba_betting.models.xgboost_model import DEFAULT_PARAMS
    from sklearn.ensemble import HistGradientBoostingClassifier
    cal_params = {**DEFAULT_PARAMS, "early_stopping": False}
    cal_model = HistGradientBoostingClassifier(**cal_params)
    cal_model.fit(X_train_part, y_train_part)
    calibrated = calibrate_model(cal_model, X_cal, y_cal)

    # Calibration metrics — reported on the walk-forward OUT-OF-FOLD
    # predictions, never on X_cal. Isotonic regression interpolates its own
    # fit points, so evaluating `calibrated.predict_proba(X_cal)` on X_cal
    # yields a meaningless ECE≈0.0000 (the old, misleading output). The OOF
    # arrays were each scored by a model that never saw them.
    oof_elo = np.array(results.get("oof_elo_probs", []), dtype=float)
    oof_gbm = np.array(results.get("oof_gbm_cal_probs", []), dtype=float)
    oof_y = np.array(results.get("oof_y_true", []), dtype=int)
    have_oof = len(oof_y) >= 50
    if have_oof:
        raw_metrics = evaluate_calibration(oof_y, oof_elo)
        cal_metrics = evaluate_calibration(oof_y, oof_gbm)
        console.print(f"  Elo (out-of-fold): Brier={raw_metrics['brier_score']:.4f}, ECE={raw_metrics['ece']:.4f}")
        console.print(f"  GBM (out-of-fold): Brier={cal_metrics['brier_score']:.4f}, ECE={cal_metrics['ece']:.4f}")
    else:
        # Fallback for tiny datasets with no usable OOF: in-sample slice,
        # clearly labeled as optimistic.
        cal_metrics = evaluate_calibration(y_cal, calibrated.predict_proba(X_cal)[:, 1])
        console.print(f"  Calibrated (in-sample, optimistic): Brier={cal_metrics['brier_score']:.4f}, ECE={cal_metrics['ece']:.4f}")

    # Optimize the ensemble weight by grid-searching log-loss — on the SAME
    # honest out-of-fold predictions, NOT the isotonic fit slice. Scoring the
    # GBM in-sample (where isotonic looks perfect) while Elo is out-of-sample
    # is an unfair comparison that biased the weight toward the GBM (old:
    # w_elo≈0.30) even though the GBM is the weaker model out-of-sample.
    # Production uses this static weight via `ensemble_predict`, so getting it
    # right is worth ~+1pp accuracy / −0.03 log-loss on held-out games.
    console.print("\n[bold]Optimizing ensemble weight (Elo vs GBM)...[/bold]")
    from nba_betting.models.ensemble import learn_ensemble_weight, save_ensemble_weight
    if have_oof:
        best_weight, weight_table = learn_ensemble_weight(oof_elo, oof_gbm, oof_y)
    else:
        elo_cal_probs = X.iloc[-n_cal:]["elo_home_prob"].values.astype(float)
        gbm_cal_probs = calibrated.predict_proba(X_cal)[:, 1]
        best_weight, weight_table = learn_ensemble_weight(elo_cal_probs, gbm_cal_probs, y_cal)
    save_ensemble_weight(best_weight)
    console.print(f"  Optimal Elo weight: {best_weight:.2f} (GBM weight: {1 - best_weight:.2f})")
    sorted_weights = sorted(weight_table.items())
    pretty = "  ".join(f"w={w:.1f}: {ll:.4f}" for w, ll in sorted_weights)
    console.print(f"  [dim]{pretty}[/dim]")

    # Save both models (including feature means for prediction imputation)
    feature_means = X.attrs.get("feature_means", {})
    save_model(model, feature_cols, feature_means)
    save_calibrated_model(calibrated)

    # Fit stacked meta-learner on out-of-fold predictions from walk-forward.
    # The meta-learner learns game-dependent Elo/GBM blending weights; it
    # replaces the static grid-searched weight when present. Requires ≥ 200
    # OOF games to avoid overfitting a logistic regression on too few points.
    console.print("\n[bold]Fitting stacked meta-learner on OOF predictions...[/bold]")
    oof_elo = results.get("oof_elo_probs", [])
    oof_gbm = results.get("oof_gbm_cal_probs", [])
    oof_y = results.get("oof_y_true", [])
    if len(oof_y) >= 200:
        try:
            from nba_betting.models.stacking import fit_meta_model, save_meta_model
            artifact = fit_meta_model(
                np.array(oof_elo, dtype=float),
                np.array(oof_gbm, dtype=float),
                np.array(oof_y, dtype=int),
            )
            save_meta_model(artifact)
            console.print(
                f"  Fitted on {artifact['n_train']} OOF games — "
                f"train log-loss={artifact['train_log_loss']:.4f}"
            )
        except Exception as e:
            console.print(f"  [yellow]Skipped: {e}[/yellow]")
    else:
        console.print(
            f"  [dim]Skipped — {len(oof_y)} OOF games available, need ≥ 200[/dim]"
        )

    # Train spread + total regression heads on the same feature matrix.
    # Separate from the classifier: classification predicts the winner,
    # these predict the margin and the total.
    console.print("\n[bold]Training spread + total regression heads...[/bold]")
    try:
        from nba_betting.models.spreads_totals import (
            train_spread_total_regressors, save_regressors,
        )
        spread_m, total_m, reg_metrics = train_spread_total_regressors(X)
        save_regressors(spread_m, total_m, feature_cols)
        console.print(
            f"  Spread MAE: [cyan]{reg_metrics['spread_mae']:.2f} pts[/cyan]   "
            f"Total MAE: [cyan]{reg_metrics['total_mae']:.2f} pts[/cyan]   "
            f"(held-out 20%, n={reg_metrics['n_train']})"
        )
    except Exception as e:
        console.print(f"  [yellow]Skipped regression heads: {e}[/yellow]")

    console.print(f"\n[green]Models saved to trained_models/[/green]")
    console.print("[green]Run 'nba-betting predict' to use the trained model.[/green]")


@app.command()
def sync(
    seasons: int = typer.Option(1, help="Number of seasons to sync (current + N-1 prior)"),
) -> None:
    """Sync NBA game data and compute Elo ratings."""
    from nba_betting.data.nba_stats import sync_season
    from nba_betting.models.elo import compute_all_elos
    from nba_betting.db.session import init_db

    init_db()

    current_year = int(CURRENT_SEASON.split("-")[0])
    season_list = []
    for i in range(seasons - 1, -1, -1):
        year = current_year - i
        season_str = f"{year}-{str(year + 1)[-2:]}"
        season_list.append(season_str)

    from nba_betting.data.nba_stats import sync_player_game_stats

    total_new = 0
    for season in season_list:
        console.print(f"[dim]Syncing {season}...[/dim]")
        try:
            new = sync_season(season)
            total_new += new
            console.print(f"  Added [green]{new}[/green] new games")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
        # Player game logs (availability source) for the same season —
        # only games without player rows are fetched into the table.
        try:
            prow = sync_player_game_stats(season)
            if prow:
                console.print(f"  Added [green]{prow}[/green] player-game rows")
        except Exception as e:
            console.print(f"  [yellow]Player logs skipped: {e}[/yellow]")

    console.print(f"\n[bold]Total new games added: {total_new}[/bold]")

    console.print("[dim]Computing Elo ratings...[/dim]")
    elos = compute_all_elos()
    console.print(f"[green]Elo ratings computed for {len(elos)} teams.[/green]")

    # Odds snapshots are captured before their game exists in the games
    # table, so the importer can't attach a game_id at import time. Now
    # that new games are stored, attach them (and the game's ET date) so
    # closing-line / CLV / real-odds-backtest joins find them.
    try:
        from nba_betting.data.snapshot_jsonl import reresolve_existing_snapshots
        res = reresolve_existing_snapshots(only_unmatched=True)
        if res.get("updated"):
            console.print(
                f"[green]Attached {res['updated']} odds snapshot(s) to newly synced games.[/green]"
            )
    except Exception as e:
        console.print(f"[yellow]Snapshot re-resolution skipped: {e}[/yellow]")

    # Auto-resolve pending predictions
    from nba_betting.betting.tracker import update_results
    updated = update_results()
    if updated:
        console.print(f"[green]Resolved {updated} pending prediction(s) with game results.[/green]")


@app.command()
def elo() -> None:
    """Display current Elo ratings for all NBA teams."""
    from nba_betting.models.elo import get_current_elos
    from nba_betting.display.console import display_elo_ratings
    from nba_betting.db.models import Team
    from nba_betting.db.session import get_session
    from sqlalchemy import select

    elos = get_current_elos()
    if not elos:
        console.print("[red]No Elo ratings found. Run 'nba-betting sync' first.[/red]")
        raise typer.Exit(1)

    session = get_session()
    try:
        teams = session.execute(select(Team)).scalars().all()
        abbr_elos = {t.abbreviation: (t.current_elo or 1500.0) for t in teams}
    finally:
        session.close()

    display_elo_ratings(abbr_elos)


@app.command()
def backfill(
    seasons: int = typer.Option(5, help="Number of seasons to backfill"),
) -> None:
    """Backfill historical NBA data (multiple seasons)."""
    sync(seasons=seasons)


@app.command()
def injury(
    action: str = typer.Argument(help="add, remove, list, or clear"),
    player: str = typer.Argument(default="", help="Player name"),
    team: str = typer.Option("", help="Team abbreviation (e.g., LAL)"),
    status: str = typer.Option("Out", help="Out, Doubtful, Questionable, Probable"),
    impact: float = typer.Option(5.0, help="Impact rating 0-10 (10=MVP-level)"),
    reason: str = typer.Option("", help="Injury description"),
) -> None:
    """Manage injury list for prediction adjustments."""
    from nba_betting.data.injuries import (
        add_injury, remove_injury, clear_injuries, load_injuries,
    )

    if action == "sync":
        console.print("[dim]Syncing injuries from ESPN...[/dim]")
        from nba_betting.data.injuries import sync_injuries_from_espn
        injuries = sync_injuries_from_espn()
        out_count = len([i for i in injuries if i.status in ("Out", "Doubtful")])
        console.print(f"[green]Synced {len(injuries)} injuries ({out_count} Out/Doubtful).[/green]")
        # Show the table
        if injuries:
            from rich.table import Table
            table = Table(title="Current Injuries (from ESPN)", show_header=True, header_style="bold cyan")
            table.add_column("Player")
            table.add_column("Team")
            table.add_column("Status")
            table.add_column("Impact", justify="right")
            table.add_column("Reason", max_width=50)
            for i in sorted(injuries, key=lambda x: (-x.impact_rating, x.team_abbr)):
                table.add_row(i.player_name, i.team_abbr, i.status, f"{i.impact_rating:.1f}", i.reason[:50])
            console.print(table)
        return

    elif action == "add":
        if not player or not team:
            console.print("[red]Usage: nba-betting injury add 'Player Name' --team LAL --impact 8[/red]")
            raise typer.Exit(1)
        inj = add_injury(player, team, status, reason, impact)
        console.print(f"[green]Added: {inj.player_name} ({inj.team_abbr}) - {inj.status}, impact={inj.impact_rating}[/green]")

    elif action == "remove":
        if remove_injury(player):
            console.print(f"[green]Removed {player} from injury list.[/green]")
        else:
            console.print(f"[yellow]{player} not found in injury list.[/yellow]")

    elif action == "list":
        injuries = load_injuries()
        if not injuries:
            console.print("[dim]No injuries tracked.[/dim]")
            return
        from rich.table import Table
        table = Table(title="Current Injuries", show_header=True, header_style="bold cyan")
        table.add_column("Player")
        table.add_column("Team")
        table.add_column("Status")
        table.add_column("Impact", justify="right")
        table.add_column("Reason")
        for i in injuries:
            table.add_row(i.player_name, i.team_abbr, i.status, f"{i.impact_rating:.0f}", i.reason)
        console.print(table)

    elif action == "clear":
        clear_injuries()
        console.print("[green]All injuries cleared.[/green]")

    else:
        console.print(f"[red]Unknown action '{action}'. Use: add, remove, list, clear[/red]")


@app.command()
def performance() -> None:
    """Show historical prediction performance and ROI."""
    from nba_betting.betting.tracker import compute_performance, update_results

    console.print("[dim]Updating results...[/dim]")
    updated = update_results()
    if updated:
        console.print(f"  Updated {updated} prediction(s) with results")

    perf = compute_performance()

    if perf.get("resolved", 0) == 0:
        console.print("[yellow]No resolved predictions yet.[/yellow]")
        console.print("[dim]Run 'predict' to generate predictions, then 'sync' after games complete.[/dim]")
        return

    from rich.table import Table
    table = Table(title="Betting Performance", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Prediction Accuracy", f"{perf['prediction_accuracy']:.1%}")
    table.add_row("Total Bets", str(perf["total_bets"]))
    table.add_row("Win/Loss", f"{perf['wins']}/{perf['losses']}")
    table.add_row("Bet Win Rate", f"{perf['bet_win_rate']:.1%}")
    table.add_row("Total Wagered", f"${perf['total_wagered']:.2f}")
    table.add_row("Total Profit", f"${perf['total_profit']:+.2f}")
    table.add_row("ROI", f"{perf['roi']:+.1%}")
    table.add_row("Current Bankroll", f"${perf['current_bankroll']:.2f}")
    table.add_row("Max Drawdown", f"{perf['max_drawdown']:.1%}")

    # Tier 2.3 — show CLV metrics alongside ROI. Avg CLV > 0 with a
    # t-stat ≥ 1.5 is the gold-standard signal that the model has real
    # pricing edge, independent of single-bet outcome variance.
    avg_clv = perf.get("avg_clv")
    clv_t = perf.get("clv_tstat")
    clv_n = perf.get("clv_count", 0)
    if avg_clv is not None:
        clv_color = "green" if avg_clv > 0 else "red"
        table.add_row(
            "Avg CLV (logit)",
            f"[{clv_color}]{avg_clv:+.3f}[/{clv_color}] (n={clv_n})",
        )
        if clv_t is not None:
            t_color = "green" if clv_t >= 1.5 else ("yellow" if clv_t >= 0 else "red")
            table.add_row("CLV t-stat", f"[{t_color}]{clv_t:+.2f}[/{t_color}]")

    console.print(table)

    # Show calibration bins if available
    cal_bins = perf.get("calibration_bins", [])
    if cal_bins:
        from rich.table import Table as RichTable
        cal_table = RichTable(title="Calibration Check", show_header=True, header_style="bold cyan")
        cal_table.add_column("Prob Range")
        cal_table.add_column("Count", justify="right")
        cal_table.add_column("Avg Predicted", justify="right")
        cal_table.add_column("Avg Actual", justify="right")
        cal_table.add_column("Gap", justify="right")
        for b in cal_bins:
            gap = abs(b["avg_predicted"] - b["avg_actual"])
            gap_color = "green" if gap < 0.05 else "yellow" if gap < 0.1 else "red"
            cal_table.add_row(
                b["range"], str(b["count"]),
                f"{b['avg_predicted']:.1%}", f"{b['avg_actual']:.1%}",
                f"[{gap_color}]{gap:.1%}[/{gap_color}]",
            )
        console.print(cal_table)


@app.command()
def clv(limit: int = 20) -> None:
    """Show recent Closing Line Value detail (Tier 2.3).

    Surfaces per-bet CLV — how the price you bet at compared to the
    closing line. Sustained positive CLV is the strongest evidence of
    real modelling edge; ROI can stay negative for months on pure
    variance even when CLV is good.
    """
    from rich.table import Table
    from nba_betting.betting.tracker import load_history, update_results, compute_performance

    # Make sure closing lines are pulled in first.
    update_results()

    history = load_history()
    bets_with_clv = [r for r in history if r.clv is not None and r.bet_side != "NO BET"]
    if not bets_with_clv:
        console.print("[yellow]No CLV data yet.[/yellow]")
        console.print(
            "[dim]CLV requires at least two odds snapshots per game "
            "(one before tipoff, one after). Run `snapshot-odds` "
            "periodically to build history.[/dim]"
        )
        return

    # Sort by most recent first
    recent = sorted(bets_with_clv, key=lambda r: r.date, reverse=True)[:limit]

    table = Table(title=f"Recent CLV (last {len(recent)} bets)", show_header=True, header_style="bold cyan")
    table.add_column("Date")
    table.add_column("Game")
    table.add_column("Side")
    table.add_column("Bet Price", justify="right")
    table.add_column("Close Price", justify="right")
    table.add_column("CLV (logit)", justify="right")
    table.add_column("Result", justify="center")

    for r in recent:
        # Bet-side specific prices
        if r.bet_side == "HOME":
            bet_price = r.market_home_prob
            close_price = r.closing_market_prob
        else:
            bet_price = 1.0 - r.market_home_prob if r.market_home_prob else None
            close_price = (
                1.0 - r.closing_market_prob if r.closing_market_prob else None
            )

        clv_color = "green" if r.clv > 0 else ("red" if r.clv < 0 else "dim")
        if r.profit is None:
            result_str = "[dim]pending[/dim]"
        elif r.profit > 0:
            result_str = "[green]W[/green]"
        elif r.profit < 0:
            result_str = "[red]L[/red]"
        else:
            result_str = "-"

        table.add_row(
            r.date,
            f"{r.away_team}@{r.home_team}",
            r.bet_side,
            f"{bet_price:.1%}" if bet_price else "-",
            f"{close_price:.1%}" if close_price else "-",
            f"[{clv_color}]{r.clv:+.3f}[/{clv_color}]",
            result_str,
        )

    console.print(table)

    # Aggregate at the bottom — same numbers as `performance`.
    perf = compute_performance()
    avg_clv = perf.get("avg_clv")
    clv_t = perf.get("clv_tstat")
    clv_n = perf.get("clv_count", 0)
    if avg_clv is not None:
        summary = Table(title="CLV Summary", show_header=False)
        summary.add_column("Metric")
        summary.add_column("Value", justify="right")
        clv_color = "green" if avg_clv > 0 else "red"
        summary.add_row("Sample size", str(clv_n))
        summary.add_row("Average CLV (logit)", f"[{clv_color}]{avg_clv:+.3f}[/{clv_color}]")
        if clv_t is not None:
            t_color = "green" if clv_t >= 1.5 else ("yellow" if clv_t >= 0 else "red")
            summary.add_row("t-statistic", f"[{t_color}]{clv_t:+.2f}[/{t_color}]")
            if clv_t >= 1.5:
                summary.add_row("Interpretation", "[green]Significant positive edge[/green]")
            elif clv_t >= 0:
                summary.add_row("Interpretation", "[yellow]Neutral / not yet significant[/yellow]")
            else:
                summary.add_row("Interpretation", "[red]Negative CLV — model may lack edge[/red]")
        console.print(summary)


@app.command(name="market-eval")
def market_eval(
    n_splits: int = typer.Option(3, help="Walk-forward folds (July-1 season splits)."),
    min_games: int = typer.Option(300, help="Games with a closing line needed before a verdict is issued."),
) -> None:
    """Score the model against REAL closing lines (odds_snapshots).

    Walk-forward out-of-fold model probabilities / margins / totals are
    joined to the latest pre-game snapshot per game and compared with the
    market on the same games: moneyline Brier/log-loss (+ the market-
    shrinkage weight that minimises log-loss), spread MAE + cover rate of
    the model's picks, total MAE + hit rate. Every section reports n; the
    verdict is significance-gated and withheld below --min-games. This is
    the harness that will eventually set MARKET_SHRINKAGE_LAMBDA and decide
    whether spread/total picks deserve a stake — run it as the snapshot
    cron accumulates a season of lines.
    """
    from rich.table import Table
    from nba_betting.betting.market_eval import evaluate_against_market
    from nba_betting.config import MARKET_SHRINKAGE_LAMBDA

    console.print("[bold]Scoring walk-forward predictions against real closing lines...[/bold]")
    res = evaluate_against_market(n_splits=n_splits)
    ml, sp, tot = res["moneyline"], res["spread"], res["total"]

    t = Table(title=f"Moneyline vs closing line (n={ml['n']})", show_header=True, header_style="bold cyan")
    t.add_column("Series"); t.add_column("Brier", justify="right"); t.add_column("LogLoss", justify="right")
    if ml["n"]:
        t.add_row("model (OOF blend)", f"{ml['model_brier']:.4f}", f"{ml['model_ll']:.4f}")
        t.add_row("market (closing)", f"{ml['market_brier']:.4f}", f"{ml['market_ll']:.4f}")
        t.add_row(f"shrunk λ={MARKET_SHRINKAGE_LAMBDA:.2f} (live)", f"{ml['live_brier']:.4f}", f"{ml['live_ll']:.4f}")
        t.add_row(f"shrunk λ={ml['best_lambda']:.2f} (best)", f"{ml['best_brier']:.4f}", f"{ml['best_ll']:.4f}")
    console.print(t)
    if ml["n"]:
        console.print(
            f"  market − model paired-Brier t = {ml['t_market_vs_model']:+.2f} "
            f"(positive → market better); best λ vs live λ t = {ml['t_best_vs_live']:+.2f}"
        )
        console.print("  [dim]λ grid log-loss: " + "  ".join(f"{k:.1f}:{v:.4f}" for k, v in sorted(ml["lambda_table"].items())) + "[/dim]")

    for name, sec, thr in (("Spread", sp, "±1.5 pts"), ("Total", tot, "±2.5 pts")):
        t2 = Table(title=f"{name} vs closing line (n={sec['n']})", show_header=True, header_style="bold cyan")
        t2.add_column("Series"); t2.add_column("MAE", justify="right")
        if sec["n"]:
            t2.add_row("model", f"{sec['model_mae']:.2f}")
            t2.add_row("market", f"{sec['market_mae']:.2f}")
            t2.add_row("avg(model, market)", f"{sec['avg_mae']:.2f}")
        console.print(t2)
        if sec["n"]:
            console.print(
                f"  market − model paired-|err| t = {sec['t_market_vs_model']:+.2f} (positive → market better); "
                f"picks at {thr}: {sec['picks']} → {sec['hits']} hits ({sec['hit_rate']:.1%}), "
                f"break-even 52.4%"
            )

    n = ml["n"]
    if n < min_games:
        console.print(
            f"\n[yellow]Not yet informative: {n} games with a closing line "
            f"(need ≥ {min_games}). Keep the snapshot cron running; re-run next season.[/yellow]"
        )
    elif ml["t_best_vs_live"] >= 2.0 and abs(ml["best_lambda"] - MARKET_SHRINKAGE_LAMBDA) >= 0.1:
        console.print(
            f"\n[green]Verdict: λ={ml['best_lambda']:.2f} beats the live λ={MARKET_SHRINKAGE_LAMBDA:.2f} "
            f"(t={ml['t_best_vs_live']:+.2f}) — consider updating MARKET_SHRINKAGE_LAMBDA.[/green]"
        )
    else:
        console.print(
            f"\n[green]Verdict: live λ={MARKET_SHRINKAGE_LAMBDA:.2f} is within noise of the best "
            f"(t={ml['t_best_vs_live']:+.2f}); no change.[/green]"
        )


@app.command(name="predict-path-eval")
def predict_path_eval(split: str = "2025-07-01") -> None:
    """A/B the live prediction feature path on a walk-forward holdout.

    Replays historical games through the REAL build_prediction_features and
    compares the current (one-game-lagged) rolling lookup against an
    include-latest "correct" variant. The only predict-path validation we
    have — run it before changing any predict-time feature logic.
    """
    from nba_betting.betting.predict_path_eval import evaluate_predict_path
    console.print(f"[dim]Replaying predict path on games >= {split} (trains on earlier)...[/dim]")
    r = evaluate_predict_path(split=split)
    if not r.get("n"):
        console.print("[yellow]Not enough data to evaluate.[/yellow]")
        return
    from rich.table import Table
    t = Table(title=f"Predict-path eval — {r['n']} held-out games", header_style="bold cyan")
    t.add_column("Variant"); t.add_column("Accuracy", justify="right")
    t.add_column("Brier", justify="right"); t.add_column("LogLoss", justify="right")
    t.add_row("correct (include latest)", f"{r['correct']['accuracy']:.1%}",
              f"{r['correct']['brier']:.4f}", f"{r['correct']['log_loss']:.4f}")
    t.add_row("lagged (legacy rolling lookup)", f"{r['lagged']['accuracy']:.1%}",
              f"{r['lagged']['brier']:.4f}", f"{r['lagged']['log_loss']:.4f}")
    if "fresh_rest" in r:
        t.add_row("lagged + fresh rest (live, current)",
                  f"{r['fresh_rest']['accuracy']:.1%}",
                  f"{r['fresh_rest']['brier']:.4f}",
                  f"{r['fresh_rest']['log_loss']:.4f}")
    console.print(t)
    d = r["delta"]
    console.print(
        f"[dim]delta (correct − lagged): acc={d['accuracy']:+.4f}  "
        f"brier={d['brier']:+.4f}  logloss={d['log_loss']:+.4f}  | "
        f"mean |Δprob| from the lag = {r['mean_abs_prob_change']:.4f}[/dim]"
    )
    if "delta_fresh_rest" in r:
        df_ = r["delta_fresh_rest"]
        console.print(
            f"[dim]delta (fresh rest − lagged): acc={df_['accuracy']:+.4f}  "
            f"brier={df_['brier']:+.4f}  logloss={df_['log_loss']:+.4f}  "
            f"(paired-Brier t={r.get('fresh_rest_vs_lagged_tstat', 0):+.2f})  | "
            f"mean |Δprob| from stale rest = {r['mean_abs_prob_change_fresh_rest']:.4f}[/dim]"
        )
    # Verdict on the remaining candidate fix (include-latest rolling on top
    # of the live fresh-rest path), gated on the paired-Brier t-stat rather
    # than raw point deltas — those flip sign run-to-run at this sample size.
    t_corr = r.get("correct_vs_live_tstat", 0.0)
    if t_corr >= 2.0:
        console.print(
            f"[yellow]→ Include-latest rolling beats the live path "
            f"(paired-Brier t={t_corr:+.2f} ≥ 2); worth implementing.[/yellow]"
        )
    else:
        console.print(
            f"[green]→ Include-latest rolling vs live path is within noise "
            f"(paired-Brier t={t_corr:+.2f} < 2); no fix warranted.[/green]"
        )


@app.command()
def diagnose() -> None:
    """Validate the prediction pipeline and check for common issues."""
    from rich.table import Table
    from nba_betting.models.elo import get_current_elos, predict_home_win_prob
    from nba_betting.models.xgboost_model import load_model, load_feature_means
    from nba_betting.models.calibration import load_calibrated_model
    from nba_betting.models.ensemble import ensemble_predict
    from nba_betting.data.polymarket import get_nba_odds
    from nba_betting.data.nba_stats import fetch_todays_games

    console.print("[bold]Pipeline Diagnostics[/bold]\n")

    # 1. Check Elo ratings
    elos = get_current_elos()
    if not elos:
        console.print("[red]FAIL: No Elo ratings. Run 'sync' first.[/red]")
        return
    elo_vals = list(elos.values())
    console.print(f"[green]OK[/green] Elo ratings: {len(elo_vals)} teams, "
                  f"mean={sum(elo_vals)/len(elo_vals):.0f}, "
                  f"range=[{min(elo_vals):.0f}, {max(elo_vals):.0f}]")

    # 2. Check trained model
    result = load_model()
    if result:
        model, feature_cols = result
        console.print(f"[green]OK[/green] GBM model loaded: {len(feature_cols)} features")
    else:
        console.print("[yellow]WARN: No trained GBM model. Using Elo-only.[/yellow]")
        feature_cols = []

    calibrated = load_calibrated_model()
    if calibrated:
        console.print("[green]OK[/green] Calibrated model loaded")
    else:
        console.print("[yellow]WARN: No calibrated model[/yellow]")

    feat_means = load_feature_means()
    if feat_means:
        console.print(f"[green]OK[/green] Feature means loaded ({len(feat_means)} features)")
    else:
        console.print("[yellow]WARN: No feature means saved — prediction will use 0 for missing[/yellow]")

    # 3. Sample predictions (top vs bottom Elo team)
    console.print("\n[bold]Sample Predictions (Top vs Bottom Elo):[/bold]")
    sorted_elos = sorted(elos.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_elos) >= 2:
        top_id, top_elo = sorted_elos[0]
        bot_id, bot_elo = sorted_elos[-1]

        elo_prob = predict_home_win_prob(top_elo, bot_elo)
        console.print(f"  Best (home) vs Worst (away): Elo prob = {elo_prob:.1%}")

        elo_prob_rev = predict_home_win_prob(bot_elo, top_elo)
        console.print(f"  Worst (home) vs Best (away): Elo prob = {elo_prob_rev:.1%}")

        if calibrated and feature_cols:
            console.print("  (Ensemble prediction requires live game features — shown at predict time)")

    # 4. Check Polymarket odds
    console.print("\n[bold]Polymarket Odds Check:[/bold]")
    try:
        market_odds = get_nba_odds()
        console.print(f"  Found {len(market_odds)} market(s)")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Event")
        table.add_column("Team 1")
        table.add_column("Price 1", justify="right")
        table.add_column("Team 2")
        table.add_column("Price 2", justify="right")
        table.add_column("Sum", justify="right")

        for odds in market_odds[:10]:  # Show first 10
            teams = odds.get("teams", {})
            abbrs = list(teams.keys())
            if len(abbrs) == 2:
                p1 = teams[abbrs[0]]
                p2 = teams[abbrs[1]]
                table.add_row(
                    odds["event_title"][:40],
                    abbrs[0], f"{p1:.3f}",
                    abbrs[1], f"{p2:.3f}",
                    f"{p1+p2:.3f}",
                )
        console.print(table)
    except Exception as e:
        console.print(f"  [red]Error fetching odds: {e}[/red]")

    # 5. Check today's games
    console.print("\n[bold]Today's Games:[/bold]")
    games = fetch_todays_games()
    if games:
        for g in games:
            home = g["home_team_abbr"]
            away = g["away_team_abbr"]
            home_elo = elos.get(g["home_team_id"], 1500)
            away_elo = elos.get(g["away_team_id"], 1500)
            elo_p = predict_home_win_prob(home_elo, away_elo)
            console.print(f"  {away} @ {home}  |  Elo: {home}={home_elo:.0f} {away}={away_elo:.0f}  |  P(home)={elo_p:.1%}")
    else:
        console.print("  No games today")

    console.print("\n[green]Diagnostics complete.[/green]")


@app.command()
def backtest(
    bankroll: float = typer.Option(DEFAULT_BANKROLL, help="Starting bankroll"),
    splits: int = typer.Option(3, help="Number of walk-forward splits"),
    live_strategy: bool | None = typer.Option(
        None,
        "--live-strategy/--raw-model",
        help=(
            "Apply shrinkage + bet-side floor (live-equivalent) or raw model "
            "vs Elo proxy. Default follows --real-odds: on with real odds, "
            "off without (so we benchmark raw model quality instead of "
            "shrinking toward an Elo prior)."
        ),
    ),
    real_odds: bool = typer.Option(
        False,
        "--real-odds",
        help="Use historical Polymarket/ESPN snapshots from odds_snapshots table when available.",
    ),
) -> None:
    """Run historical backtest of the betting strategy."""
    from nba_betting.features.builder import build_feature_matrix
    from nba_betting.betting.backtest import run_backtest
    from rich.table import Table

    console.print("[bold]Building feature matrix...[/bold]")
    X, y = build_feature_matrix()

    if X.empty:
        console.print("[red]No data. Run 'sync --seasons 3' first.[/red]")
        raise typer.Exit(1)

    console.print(f"  {len(X)} games available for backtesting")

    effective_live = live_strategy if live_strategy is not None else real_odds
    mode_parts = []
    mode_parts.append("live-strategy" if effective_live else "raw-model")
    mode_parts.append("real-odds" if real_odds else "elo-proxy")
    console.print(f"[bold]Running backtest ({' + '.join(mode_parts)})...[/bold]")
    results = run_backtest(
        X, y,
        bankroll=bankroll,
        n_splits=splits,
        apply_live_strategy=effective_live,
        use_real_odds=real_odds,
    )

    summary = results["summary"]
    if not summary or summary.get("total_bets", 0) == 0:
        console.print("[yellow]No bets generated during backtest period.[/yellow]")
        return

    # Summary table
    table = Table(title="Backtest Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Total Bets", str(summary["total_bets"]))
    table.add_row("Win/Loss", f"{summary['wins']}/{summary['losses']}")
    table.add_row("Win Rate", f"{summary['win_rate']:.1%}")
    table.add_row("Total Wagered", f"${summary['total_wagered']:.2f}")
    table.add_row("Total Profit", f"${summary['total_profit']:+.2f}")
    table.add_row("ROI", f"{summary['roi']:+.1%}")
    table.add_row("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
    table.add_row("Final Bankroll", f"${summary['final_bankroll']:.2f}")
    table.add_row("Max Drawdown", f"{summary['max_drawdown']:.1%}")
    table.add_row("Avg Edge", f"{summary['avg_edge']:.1%}")
    table.add_row("Avg Bet Size", f"${summary['avg_bet_size']:.2f}")
    if "real_odds_coverage" in summary:
        table.add_row(
            "Real Odds Coverage",
            f"{summary['real_odds_hits']}/{summary['real_odds_hits'] + summary['real_odds_misses']} "
            f"({summary['real_odds_coverage']:.0%})",
        )

    console.print(table)

    # Badge distribution
    bets = results["bets"]
    badges = {}
    for b in bets:
        badges[b["badge"]] = badges.get(b["badge"], 0) + 1
    console.print("\n[bold]Signal Distribution:[/bold]")
    for badge, count in sorted(badges.items()):
        wins = sum(1 for b in bets if b["badge"] == badge and b["won"])
        console.print(f"  {badge:10s}: {count} bets, {wins}/{count} wins ({wins/count:.0%})")

    console.print(f"\n[dim]Bankroll: ${bankroll:.0f} → ${summary['final_bankroll']:.0f} "
                  f"({summary['roi']:+.1%} ROI over {summary['total_bets']} bets)[/dim]")


# Length of an NBA betting season in calendar days — Oct → mid-June, regular
# season plus playoffs. Used to project the backtest's observed bet density
# onto a "one season for me" forward-looking horizon in `simulate`.
_NBA_SEASON_DAYS = 240

# Minimum default horizon. Below ~20 bets the MC percentile distribution is
# noisy and the user gets less signal than they would from a single backtest.
_MIN_DEFAULT_HORIZON = 20


def _estimate_one_season_bets(bets: list[dict]) -> tuple[int, str]:
    """Estimate "how many bets does one NBA season produce for me?"

    Replaces the previous fixed 200-bet horizon, which was arbitrary and
    didn't adapt: a heavy bettor (high bet rate) saw a horizon ~half a
    season; a selective bettor (high edge threshold) saw three seasons in
    one horizon. Date-density-based estimation fixes both edges.

    Returns ``(horizon, reason)`` so the CLI can surface *why* the chosen
    horizon is what it is in the dim header line above the table.
    """
    from datetime import datetime

    pool_size = len(bets)
    try:
        dates = sorted({
            datetime.strptime(b["date"], "%Y-%m-%d").date()
            for b in bets if "date" in b
        })
    except (ValueError, TypeError):
        return pool_size, "full pool (bet dates unparseable)"

    if len(dates) < 2:
        return pool_size, "full pool (insufficient date range)"

    span_days = max((dates[-1] - dates[0]).days, 1)
    if span_days < 30:
        # Backtest spans less than a month — extrapolating bet density to
        # 8 months would overstate horizon by 10x+. Just use the pool.
        return pool_size, f"full pool (backtest spans only {span_days} days)"

    bets_per_day = pool_size / span_days
    estimated = round(bets_per_day * _NBA_SEASON_DAYS)
    horizon = max(_MIN_DEFAULT_HORIZON, min(estimated, pool_size))

    return (
        horizon,
        (
            f"≈ one NBA season at {bets_per_day:.2f} bets/day "
            f"(backtest density over {span_days} days)"
        ),
    )


@app.command()
def simulate(
    n_sims: int = typer.Option(60_000, help="Number of Monte Carlo simulations"),
    bankroll: float = typer.Option(DEFAULT_BANKROLL, help="Starting bankroll"),
    mode: str = typer.Option(
        "both",
        help=(
            "Simulation mode: 'empirical' resamples real backtest outcomes "
            "(honest bootstrap — preserves the actual win rate). "
            "'market_right' assumes the market price is the true probability "
            "(pessimistic efficient-market null). "
            "'both' runs both side-by-side for comparison."
        ),
    ),
    real_odds: bool = typer.Option(
        False,
        "--real-odds",
        help=(
            "Use real closing-line snapshots from odds_snapshots as the market "
            "price in the backtest (instead of the Elo proxy). Passes through "
            "to run_backtest(use_real_odds=...)."
        ),
    ),
    live_strategy: bool | None = typer.Option(
        None,
        "--live-strategy/--no-live-strategy",
        help=(
            "Apply the live system's Bayesian shrinkage + bet-side floor in "
            "the underlying backtest, so the simulated bet pool matches what "
            "`predict` would actually bet. Defaults to ON — the simulator's "
            "job is to project the live system, not raw model lift. Pass "
            "--no-live-strategy to bound raw model skill against the market "
            "proxy (use `backtest --raw-model` for the canonical version of "
            "that question)."
        ),
    ),
    horizon: int | None = typer.Option(
        None,
        "--horizon",
        help=(
            "Bets per simulated path. Defaults to the backtest's observed "
            "bet density projected onto one NBA season (~240 days, "
            "regular + playoffs) — so a heavy bettor gets a longer "
            "horizon than a selective one. Pass an explicit value "
            "(including the full bet-pool size for the original "
            "compounded behavior) to override."
        ),
    ),
) -> None:
    """Run Monte Carlo simulation of bankroll evolution.

    Uses **empirical bootstrap** over the backtest's realized bets by
    default — i.e. each simulated bet draws a real historical
    ``(p_model, p_market, won)`` tuple from the backtest output. This
    preserves the actually-observed win rate rather than the model's
    self-reported probability, which was the bug in the original
    implementation (it simulated ``won ~ Bernoulli(p_model)`` — a
    tautology that produced fictional trillion-dollar bankrolls).

    The optional ``market_right`` mode assumes the efficient-market
    null: each bet's outcome is drawn from ``Bernoulli(p_market)``. If
    the bankroll drifts down under this mode, it confirms our positive
    ROI in the empirical mode comes from real edge over the market
    (not an artifact of Kelly compounding). A positive gap between
    empirical and market-null median ROI is the diagnostic.

    See ``nba_betting/betting/montecarlo.py`` docstring for the
    full rationale behind the rewrite.
    """
    from nba_betting.features.builder import build_feature_matrix
    from nba_betting.betting.backtest import run_backtest
    from nba_betting.betting.montecarlo import simulate_bankroll
    from rich.table import Table

    if mode not in {"empirical", "market_right", "both"}:
        console.print(
            f"[red]Invalid mode {mode!r}. Use 'empirical', 'market_right', "
            f"or 'both'.[/red]"
        )
        raise typer.Exit(1)

    console.print("[bold]Building feature matrix...[/bold]")
    X, y = build_feature_matrix()

    if X.empty:
        console.print("[red]No data. Run 'sync --seasons 3' first.[/red]")
        raise typer.Exit(1)

    # Resolve --live-strategy / --no-live-strategy. Default ON: the
    # simulator's job is to project the live system, so the backtest should
    # mirror what `predict` would actually bet (shrinkage + floor + edge
    # gate). Users can opt out for raw-model bounding.
    effective_live_strategy = True if live_strategy is None else live_strategy

    console.print("[bold]Running backtest to get bet distribution...[/bold]")
    bt = run_backtest(
        X, y,
        bankroll=bankroll,
        use_real_odds=real_odds,
        apply_live_strategy=effective_live_strategy,
    )
    bets = bt["bets"]

    if not bets:
        console.print("[yellow]No bets from backtest — cannot simulate.[/yellow]")
        return

    model_probs = [b["model_prob"] for b in bets]
    market_probs = [b["market_prob"] for b in bets]
    won_outcomes = [bool(b["won"]) for b in bets]

    realized_wr = sum(won_outcomes) / len(won_outcomes)
    claimed_wr = sum(model_probs) / len(model_probs)
    console.print(
        f"[dim]Backtest: {len(bets)} bets, live_strategy="
        f"{effective_live_strategy}, real_odds={real_odds}. "
        f"Realized win rate {realized_wr:.3f}, model-claimed avg prob "
        f"{claimed_wr:.3f} (gap {realized_wr - claimed_wr:+.3f}).[/dim]"
    )

    # Resolve --horizon. Default: project the backtest's observed bet
    # density onto one NBA season (~240 days, regular + playoffs). This
    # adapts to the user's actual bet rate — a heavy bettor gets a longer
    # horizon than a selective one — instead of the prior fixed 200-bet
    # cap, which under-counted heavy bettors and over-counted selective
    # ones. See ARCHITECTURE.md §6.9 for the rationale.
    if horizon is None:
        n_bets_per_sim, horizon_reason = _estimate_one_season_bets(bets)
    else:
        n_bets_per_sim = horizon
        horizon_reason = "user-supplied via --horizon"
    if n_bets_per_sim <= 0:
        console.print(f"[red]--horizon must be positive, got {horizon}.[/red]")
        raise typer.Exit(1)
    console.print(
        f"[dim]Default horizon: {n_bets_per_sim} bets — "
        f"{horizon_reason}.[/dim]"
    )

    modes_to_run = (
        ["empirical", "market_right"] if mode == "both" else [mode]
    )

    all_results: dict[str, dict] = {}
    for m in modes_to_run:
        console.print(
            f"[bold]Running {n_sims:,} MC sims ({n_bets_per_sim} bets each, "
            f"bootstrapped from {len(bets)}-bet pool) — mode={m}...[/bold]"
        )
        all_results[m] = simulate_bankroll(
            model_probs, market_probs,
            won_outcomes=won_outcomes,
            mode=m,
            n_simulations=n_sims,
            n_bets_per_sim=n_bets_per_sim,
            initial_bankroll=bankroll,
        )

    # Inflated-edge banner. The single most-useful upfront context for the
    # user: an Elo-proxy backtest with log-growth/bet > ~0.003 is overstating
    # real-market edge by a large factor. Print it BEFORE the table so the
    # 100%-P(Profit) row doesn't get read as a forecast.
    emp = all_results.get("empirical")
    _INFLATED_LOG_GROWTH = 0.003  # ~+30bps/bet — generous floor for "real" edge
    if (
        emp is not None
        and not real_odds
        and emp["median_log_growth_per_bet"] > _INFLATED_LOG_GROWTH
    ):
        console.print(
            "[yellow bold]⚠ Using Elo proxy (no --real-odds). Per ARCHITECTURE "
            "§6.6, the Elo proxy is systematically weaker than a real "
            "efficient market — model edge against it is overstated. The "
            "numbers below are the bootstrap-arithmetic of that overstated "
            "per-bet edge, NOT a real-world forecast. Look at the gap to "
            "the Market-is-right column for the skill signal, and rerun "
            "with --real-odds once odds_snapshots covers the eval window.[/]"
        )

    table = Table(
        title="Monte Carlo Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric")
    headers = {
        "empirical": "Empirical (real outcomes)",
        "market_right": "Market-is-right (null)",
    }
    for m in modes_to_run:
        table.add_column(headers[m], justify="right")

    def col(key: str, fmt) -> list[str]:
        return [fmt(all_results[m][key]) for m in modes_to_run]

    as_dollars = lambda v: f"${v:,.2f}"
    as_pct = lambda v: f"{v:.1%}"
    as_pct_signed = lambda v: f"{v:+.1%}"
    # Per-bet log growth stays small (~0.005 for a real 0.5%/bet edge,
    # ~-0.001 for the market-null), so we show 4 decimals rather than %.
    as_logret = lambda v: f"{v:+.4f}"

    table.add_row(
        "Simulations",
        *[f"{all_results[m]['n_simulations']:,}" for m in modes_to_run],
    )
    table.add_row(
        "Bets per Sim",
        *[str(all_results[m]["n_bets_per_sim"]) for m in modes_to_run],
    )
    blank = ["" for _ in modes_to_run]
    table.add_row("", *blank)
    # Horizon-invariant skill metrics first — these are the honest read.
    # log-growth/bet doesn't compound with n_bets, so a real +0.005 edge
    # stays at +0.005 whether you sim 200 or 2000 bets. The compounded
    # bankroll percentiles below DO scale with horizon and are mainly a
    # bankroll-sizing illustration, not a forecast.
    table.add_row(
        "Median Log-Growth / Bet", *col("median_log_growth_per_bet", as_logret)
    )
    table.add_row(
        "Mean Log-Growth / Bet", *col("mean_log_growth_per_bet", as_logret)
    )
    table.add_row("P(Ruin)", *col("probability_of_ruin", as_pct))
    table.add_row("Median Max Drawdown", *col("median_max_drawdown", as_pct))
    table.add_row("Worst Max Drawdown", *col("worst_max_drawdown", as_pct))
    table.add_row("", *blank)
    # Horizon-dependent illustration (clearly labeled).
    horizon_label = f"@ horizon={n_bets_per_sim} bets"
    table.add_row(f"P(Profit) {horizon_label}", *col("probability_of_profit", as_pct))
    table.add_row(f"Median ROI {horizon_label}", *col("median_roi", as_pct_signed))
    table.add_row(f"Median Bankroll {horizon_label}", *col("median_final_bankroll", as_dollars))
    table.add_row(f"5th Percentile {horizon_label}", *col("pct_5", as_dollars))
    table.add_row(f"25th Percentile {horizon_label}", *col("pct_25", as_dollars))
    table.add_row(f"75th Percentile {horizon_label}", *col("pct_75", as_dollars))
    table.add_row(f"95th Percentile {horizon_label}", *col("pct_95", as_dollars))

    console.print(table)

    mkt = all_results.get("market_right")

    if emp is not None:
        if emp["probability_of_ruin"] > 0.1:
            console.print(
                "[red]WARNING: empirical ruin probability > 10%. "
                "Lower the Kelly fraction or raise the edge threshold.[/red]"
            )
    if mkt is not None and emp is not None:
        # Horizon-invariant gap: doesn't blow up with n_bets the way
        # median_roi does. Expect ~+0.005 for a real edge, ~0 or negative
        # under the null. The compounded numbers above are honest Kelly
        # math but look absurd; this is the per-bet statement of skill.
        log_gap = (
            emp["median_log_growth_per_bet"]
            - mkt["median_log_growth_per_bet"]
        )
        console.print(
            f"[dim]Skill signal (horizon-invariant): empirical log-growth/bet "
            f"vs market-null gap = {log_gap:+.4f} (~{log_gap * 100:+.2f}% per "
            f"bet). Positive gap is evidence of real edge over the market "
            f"used in this backtest.[/dim]"
        )
    if not real_odds:
        console.print(
            "[yellow]Note: backtest used the Elo proxy as 'market' (no "
            "historical Polymarket coverage). Rerun with --real-odds "
            "once odds_snapshots has coverage for the eval window for a "
            "live-equivalent simulation.[/yellow]"
        )


@app.command(name="snapshot-odds")
def snapshot_odds(
    jsonl: str = typer.Option(
        None,
        "--jsonl",
        help=(
            "Write to a JSONL file instead of the local DB. "
            "Intended for the GitHub Actions cron runner (the user is in "
            "Europe and asleep during the NBA overnight window). Use "
            "`import-snapshots` to load the file into the local DB. "
            "Pass a directory path; defaults to data/odds_snapshots/."
        ),
    ),
) -> None:
    """Capture a single snapshot of current Polymarket + ESPN odds.

    Two modes:

    * Default (no flag): writes directly to the local SQLite
      `odds_snapshots` table. Designed for cron / launchd on the user's
      own machine.
    * `--jsonl PATH`: appends one record per (game, source) to a JSONL
      file under ``PATH`` (a directory; defaults to
      ``data/odds_snapshots/``). No DB access — designed for the
      GitHub Actions runner which has no persistent SQLite. Pair with
      ``import-snapshots`` to load the file back locally.

    Exit code 0 always; warnings go to stdout for log parsing.
    """
    if jsonl is not None:
        # JSONL mode: DB-free path for the GH Actions cron runner.
        from nba_betting.data.snapshot_jsonl import (
            DEFAULT_SNAPSHOT_DIR,
            capture_snapshot_to_jsonl,
        )

        out_dir = jsonl if jsonl else str(DEFAULT_SNAPSHOT_DIR)
        result = capture_snapshot_to_jsonl(out_dir)
        # `notes` are informational (e.g. "used ESPN fallback" — which is
        # the expected path on GitHub Actions). They intentionally do NOT
        # escalate the status from ok → warn; a yellow "warn" label on an
        # expected path is the exact mis-signal that made us add this
        # distinction. Only real problems (network failures, empty slate)
        # land in `warnings` and flip the label.
        status = "ok" if not result.get("warnings") else "warn"
        console.print(
            f"[{'green' if status == 'ok' else 'yellow'}]"
            f"snapshot-odds[jsonl] {status}[/] "
            f"games={result.get('games', 0)} "
            f"written={result.get('written', 0)} "
            f"source={result.get('source', 'nba-api')} "
            f"poly={result.get('polymarket_lines', 0)} "
            f"espn={result.get('espn_lines', 0)} "
            f"path={result.get('path', '')}"
        )
        for n in result.get("notes", []):
            console.print(f"[cyan]  note: {n}[/cyan]")
        for w in result.get("warnings", []):
            console.print(f"[yellow]  warn: {w}[/yellow]")
        return

    from nba_betting.data.odds_tracker import capture_snapshot
    from nba_betting.db.session import init_db

    init_db()
    result = capture_snapshot()
    status = "ok" if not result.get("warnings") else "warn"
    console.print(
        f"[{'green' if status == 'ok' else 'yellow'}]"
        f"snapshot-odds {status}[/] "
        f"games={result.get('games', 0)} "
        f"saved={result.get('saved', 0)} "
        f"poly={result.get('polymarket_lines', 0)} "
        f"espn={result.get('espn_lines', 0)}"
    )
    for w in result.get("warnings", []):
        console.print(f"[yellow]  warn: {w}[/yellow]")


@app.command(name="snapshot-injuries")
def snapshot_injuries(
    jsonl: str = typer.Option(
        None,
        "--jsonl",
        help=(
            "Write today's (ET) full injury list to a JSONL file under this "
            "directory instead of the local DB — the DB-free mode the GitHub "
            "Actions runner uses. Defaults to data/injury_snapshots/. Load it "
            "locally with `import-snapshots`."
        ),
    ),
) -> None:
    """Capture today's ESPN injury report as a dated snapshot.

    Default (no flag): same as `injury sync` — refresh data/injuries.json and
    upsert today's rows into the local `historical_injuries` table.

    `--jsonl DIR`: DB-free; writes/refreshes `DIR/<ET-date>.jsonl` with the
    full league list (latest capture of the day wins; untouched when the
    list hasn't changed). This is what makes the injury training features
    accumulate without anyone running `predict` — see ARCHITECTURE §4.10.
    """
    if jsonl is not None:
        from nba_betting.data.injury_jsonl import (
            DEFAULT_INJURY_SNAPSHOT_DIR, capture_injuries_to_jsonl,
        )
        out_dir = jsonl if jsonl else str(DEFAULT_INJURY_SNAPSHOT_DIR)
        result = capture_injuries_to_jsonl(out_dir)
        status = "ok" if not result.get("warnings") else "warn"
        console.print(
            f"[{'green' if status == 'ok' else 'yellow'}]"
            f"snapshot-injuries[jsonl] {status}[/] "
            f"date={result.get('snapshot_date')} "
            f"players={result.get('players', 0)} "
            f"written={result.get('written', 0)} "
            f"unchanged={result.get('unchanged', False)} "
            f"path={result.get('path', '')}"
        )
        for w in result.get("warnings", []):
            console.print(f"[yellow]  warn: {w}[/yellow]")
        return

    from nba_betting.data.injuries import sync_injuries_from_espn
    from nba_betting.db.session import init_db

    init_db()
    injuries = sync_injuries_from_espn()
    out_count = len([i for i in injuries if i.status in ("Out", "Doubtful")])
    console.print(
        f"[green]snapshot-injuries ok[/] players={len(injuries)} out_or_doubtful={out_count}"
    )


@app.command(name="import-snapshots")
def import_snapshots(
    path: str = typer.Option(
        "data/odds_snapshots",
        "--path",
        help=(
            "Path to a JSONL file or directory produced by "
            "`snapshot-odds --jsonl`. Defaults to data/odds_snapshots/."
        ),
    ),
    pull: bool = typer.Option(
        False,
        "--pull",
        help=(
            "Run `git pull --ff-only` in the repo root before importing, "
            "so snapshots committed by the GitHub Actions cron are fetched "
            "to your local working copy first. Preferred daily flow: "
            "`python3 -m nba_betting import-snapshots --pull`."
        ),
    ),
    injuries_path: str = typer.Option(
        "data/injury_snapshots",
        "--injuries-path",
        help=(
            "Directory of per-day injury JSONL files produced by "
            "`snapshot-injuries --jsonl` (the GH Actions cron). Imported "
            "into `historical_injuries` after the odds files; skipped "
            "silently if the directory doesn't exist."
        ),
    ),
) -> None:
    """Import odds (and injury) snapshots from JSONL files into the local DB.

    Typical workflow (daily):
        python3 -m nba_betting import-snapshots --pull

    This fetches the GitHub Actions cron commits and loads them into
    your local SQLite in one step. Without `--pull` you'll need to run
    `git pull` yourself first, otherwise files committed by the runner
    won't be on your filesystem yet.

    Idempotent — re-running this against the same file imports 0 rows.
    Dedup is done on the natural key
    `(game_date, home_team_id, away_team_id, source, timestamp)`.
    """
    import subprocess
    from pathlib import Path as _Path
    from sqlalchemy import select, func
    from nba_betting.data.snapshot_jsonl import import_snapshots_jsonl
    from nba_betting.db.models import Team
    from nba_betting.db.session import get_session, init_db

    # Local exception used to bail out of the multi-step git flow below
    # without pyramid-ing if/else. Scoped inside the function so it
    # doesn't leak into the module namespace.
    class _SkipPull(Exception):
        pass

    if pull:
        # Pull in the repo that contains this module — not CWD. Users
        # frequently invoke the CLI from elsewhere (home dir, another
        # project) and we still want `--pull` to update the NBA-Betting
        # checkout. The repo root is two levels up from this file
        # (nba_betting/cli.py → repo root).
        repo_root = _Path(__file__).resolve().parent.parent

        # CRITICAL: the GH Actions cron commits snapshot JSONL only to
        # `main`. A plain `git pull --ff-only` (what we used to do) pulls
        # the CURRENT branch's upstream, which for a user sitting on a
        # feature branch returns "Already up to date" — truthful but
        # silently misses every new snapshot on main. That exact sharp
        # edge bit the user on 2026-04-22 and is why this is now
        # fast-forward-from-origin/main rather than a plain pull.
        def _run_git(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        try:
            # Step 1: fetch origin. Gets origin/main up to date without
            # touching the working copy. Safe on any branch state.
            fetch = _run_git("fetch", "origin", "main")
            if fetch.returncode != 0:
                console.print(
                    f"[yellow]git fetch origin main failed "
                    f"(rc={fetch.returncode}); continuing with existing files[/yellow]"
                )
                if fetch.stderr:
                    console.print(f"[dim]{fetch.stderr.strip()}[/dim]")
                raise _SkipPull()

            # Step 2: figure out whether the working copy is already at
            # or behind origin/main. If behind, fast-forward. If ahead
            # or diverged (user on an unmerged feature branch with local
            # commits), don't touch anything — warn and continue.
            branch_out = _run_git("rev-parse", "--abbrev-ref", "HEAD")
            branch = branch_out.stdout.strip() if branch_out.returncode == 0 else "?"

            # Is origin/main an ancestor of HEAD? If yes, local is
            # already at/ahead of main — no fast-forward needed.
            is_ancestor = _run_git(
                "merge-base", "--is-ancestor", "origin/main", "HEAD"
            ).returncode == 0
            if is_ancestor:
                console.print(
                    f"[green]git: already up to date with origin/main "
                    f"(branch={branch})[/]"
                )
            else:
                # origin/main has commits we don't — attempt FF merge.
                # This works from any branch whose HEAD is a strict
                # ancestor of origin/main (e.g. `main` itself, or a
                # feature branch already merged via PR).
                merge = _run_git("merge", "--ff-only", "origin/main")
                if merge.returncode == 0:
                    first = (merge.stdout or merge.stderr).strip().splitlines()
                    msg = first[0] if first else "fast-forwarded"
                    console.print(
                        f"[green]git: fast-forwarded {branch} → origin/main[/] {msg}"
                    )
                else:
                    # Fast-forward impossible: user has local commits
                    # not on main. Don't touch their branch — loudly
                    # tell them to switch to main and try again.
                    console.print(
                        f"[yellow]⚠ Snapshot JSONL files are committed to "
                        f"`main`, but you're on `{branch}` with local commits "
                        f"that diverge from main. `--pull` can't fast-forward "
                        f"safely — run `git checkout main && python3 -m "
                        f"nba_betting import-snapshots --pull` instead.[/yellow]"
                    )
                    if merge.stderr:
                        console.print(f"[dim]{merge.stderr.strip()}[/dim]")
        except _SkipPull:
            pass
        except FileNotFoundError:
            console.print("[yellow]git not found on PATH; skipping pull[/yellow]")
        except subprocess.TimeoutExpired:
            console.print("[yellow]git fetch/merge timed out; continuing with existing files[/yellow]")

    init_db()
    # Fresh-clone guard: without a populated Teams table, every record
    # falls through the abbr→id lookup and lands in `errors`. Fail fast
    # with a clear next step instead of letting the user stare at N
    # "unknown team" warnings.
    sess = get_session()
    try:
        team_count = sess.execute(select(func.count(Team.id))).scalar_one()
    finally:
        sess.close()
    if not team_count:
        console.print(
            "[red]Teams table is empty — run `python3 -m nba_betting sync` "
            "first so team abbreviations can resolve.[/red]"
        )
        raise typer.Exit(1)

    result = import_snapshots_jsonl(path)
    console.print(
        f"[green]import-snapshots ok[/] "
        f"files={result['files']} "
        f"records={result['records']} "
        f"imported={result['imported']} "
        f"skipped={result['skipped']} "
        f"errors={len(result['errors'])}"
    )
    for e in result["errors"][:10]:
        console.print(f"[yellow]  warn: {e}[/yellow]")
    if len(result["errors"]) > 10:
        console.print(f"[dim]  ({len(result['errors']) - 10} more errors suppressed)[/dim]")

    # Daily injury snapshots (same cron, separate directory). Each file
    # replaces its day in `historical_injuries`, so this is idempotent.
    if injuries_path and _Path(injuries_path).exists():
        from nba_betting.data.injury_jsonl import import_injuries_jsonl
        inj = import_injuries_jsonl(injuries_path)
        console.print(
            f"[green]import-injuries ok[/] "
            f"files={inj['files']} days={inj['days']} rows={inj['rows']} "
            f"errors={len(inj['errors'])}"
        )
        for e in inj["errors"][:10]:
            console.print(f"[yellow]  warn: {e}[/yellow]")


@app.command(name="sync-players")
def sync_players() -> None:
    """Sync player rosters and depth charts from ESPN."""
    from nba_betting.data.player_stats import sync_all_rosters
    from nba_betting.db.session import init_db

    init_db()
    console.print("[bold]Syncing player rosters from ESPN...[/bold]")
    console.print("[dim]This fetches rosters + depth charts for all 30 teams (rate-limited).[/dim]")
    total = sync_all_rosters()
    console.print(f"[green]Synced {total} players across 30 teams.[/green]")


@app.command(name="readiness-status")
def readiness_status() -> None:
    """Report how much historical injury + odds snapshot data we've
    accumulated, so the user knows whether the pipeline is ready to
    retrain with the new feature columns.

    The injury-as-of and real-odds backtest paths need *accumulated*
    historical rows to be useful — they start out empty and only become
    informative once `snapshot-odds` and `injury sync` have been running
    on a schedule for a while. This command prints distinct
    `snapshot_date` counts for each source so the user can decide when
    to rerun `train`/`backtest --real-odds`.

    Rough guidance printed alongside:
      <  5 days   — collect more data; features will be near-constant.
      5-30 days   — usable for diagnostics but not for retraining.
      >= 30 days  — enough variation to retrain with injury/odds features.
    """
    from sqlalchemy import select, func, distinct
    from nba_betting.db.models import (
        HistoricalInjury, OddsSnapshot,
    )
    from nba_betting.db.session import get_session
    from rich.table import Table

    session = get_session()
    try:
        # Distinct snapshot-date counts for each source. We use
        # `func.count(distinct(...))` so a busy day with 30 snapshots
        # still counts as 1 day of coverage — which is the number that
        # matters for "do we have enough variation to train on".
        inj_days = session.execute(
            select(func.count(distinct(HistoricalInjury.snapshot_date)))
        ).scalar() or 0
        inj_rows = session.execute(
            select(func.count(HistoricalInjury.id))
        ).scalar() or 0

        odds_days_total = session.execute(
            select(func.count(distinct(OddsSnapshot.game_date)))
        ).scalar() or 0
        odds_days_poly = session.execute(
            select(func.count(distinct(OddsSnapshot.game_date)))
            .where(OddsSnapshot.source == "polymarket")
        ).scalar() or 0
        odds_days_espn = session.execute(
            select(func.count(distinct(OddsSnapshot.game_date)))
            .where(OddsSnapshot.source == "espn")
        ).scalar() or 0
        odds_rows = session.execute(
            select(func.count(OddsSnapshot.id))
        ).scalar() or 0
    finally:
        session.close()

    def _tier(days: int) -> tuple[str, str]:
        if days >= 30:
            return "ready", "green"
        if days >= 5:
            return "partial", "yellow"
        return "cold", "red"

    inj_tier, inj_color = _tier(inj_days)
    odds_tier, odds_color = _tier(odds_days_total)

    table = Table(title="Pipeline Readiness", show_header=True, header_style="bold cyan")
    table.add_column("Stream")
    table.add_column("Distinct Days", justify="right")
    table.add_column("Rows", justify="right")
    table.add_column("Status", justify="center")

    table.add_row(
        "Historical injuries",
        str(inj_days),
        str(inj_rows),
        f"[{inj_color}]{inj_tier}[/]",
    )
    table.add_row(
        "Odds snapshots — total",
        str(odds_days_total),
        str(odds_rows),
        f"[{odds_color}]{odds_tier}[/]",
    )
    table.add_row("  • Polymarket days", str(odds_days_poly), "", "")
    table.add_row("  • ESPN days", str(odds_days_espn), "", "")

    console.print(table)

    # Actionable nudges
    console.print()
    if inj_days < 30:
        console.print(
            f"[yellow]Injury coverage is {inj_tier} ({inj_days} days).[/yellow] "
            f"Run [cyan]python3 -m nba_betting injury sync[/cyan] (or schedule it daily) "
            f"until you hit ~30 days, then rerun [cyan]train[/cyan] to let the model "
            f"learn injury-impact features."
        )
    if odds_days_total < 30:
        console.print(
            f"[yellow]Odds snapshot coverage is {odds_tier} ({odds_days_total} days).[/yellow] "
            f"Schedule [cyan]python3 -m nba_betting snapshot-odds[/cyan] every 30–60 "
            f"minutes during the NBA season. Once you hit ~30 days, "
            f"[cyan]backtest --real-odds[/cyan] produces a live-equivalent ROI."
        )
    if inj_days >= 30 and odds_days_total >= 30:
        console.print("[green]Both streams ready. Consider rerunning train + backtest --real-odds.[/green]")


@app.command()
def serve(
    port: int = typer.Option(8050, help="Port to run the web server on"),
) -> None:
    """Start the web dashboard server."""
    import uvicorn
    console.print(f"[bold]Starting NBA Betting dashboard at http://localhost:{port}[/bold]")
    uvicorn.run("nba_betting.api.app:app", host="0.0.0.0", port=port, reload=False)


@app.command()
def commands() -> None:
    """Display all available terminal commands."""
    console.print("[bold]NBA Betting System Commands[/bold]\n")
    cmds = [
        ("python3 -m nba_betting sync --seasons 3", "Fetch game data + compute Elo"),
        ("python3 -m nba_betting train", "Train GBM model + calibrate"),
        ("python3 -m nba_betting predict", "Today's recommendations + explanations"),
        ("python3 -m nba_betting elo", "Current Elo standings"),
        ("python3 -m nba_betting performance", "Historical accuracy + ROI + CLV"),
        ("python3 -m nba_betting clv", "Per-bet Closing Line Value detail"),
        ("python3 -m nba_betting backtest", "Simulate strategy on historical data"),
        ("python3 -m nba_betting simulate", "Monte Carlo bankroll simulation"),
        ("python3 -m nba_betting diagnose", "Validate prediction pipeline"),
        ("python3 -m nba_betting injury sync", "Auto-sync injuries from ESPN"),
        ("python3 -m nba_betting injury list", "View injury list"),
        ("python3 -m nba_betting injury add 'Name' --team LAL --impact 8", "Add injury manually"),
        ("python3 -m nba_betting sync-players", "Sync player rosters from ESPN"),
        ("python3 -m nba_betting snapshot-odds", "Capture Polymarket+ESPN snapshot (cron)"),
        ("python3 -m nba_betting readiness-status", "Report injury + odds snapshot coverage"),
        ("python3 -m nba_betting serve", "Launch web dashboard at localhost:8050"),
        ("python3 -m nba_betting commands", "Show this help"),
    ]
    for cmd, desc in cmds:
        console.print(f"  [cyan]{cmd:55s}[/cyan] {desc}")


def main():
    app()


if __name__ == "__main__":
    main()
