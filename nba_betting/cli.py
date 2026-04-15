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

    # Determine which model to use
    use_model = model
    xgb_loaded = None
    calibrated_loaded = None
    rolling_df = None

    if use_model in ("auto", "xgb", "ensemble"):
        calibrated_loaded = load_calibrated_model()
        if calibrated_loaded is None:
            xgb_result = load_model()
            if xgb_result:
                xgb_loaded = xgb_result
        if use_model == "auto":
            if calibrated_loaded or xgb_loaded:
                use_model = "ensemble"
                console.print("[dim]Using ensemble model (Elo + XGBoost).[/dim]")
            else:
                use_model = "elo"
                console.print("[dim]No trained XGBoost found. Using Elo model only.[/dim]")
                console.print("[dim]Run 'nba-betting train' to build the XGBoost model.[/dim]")

    # Build prediction function based on model choice
    predict_fn = None

    if use_model in ("xgb", "ensemble") and (calibrated_loaded or xgb_loaded):
        # Load rolling features for XGBoost prediction
        console.print("[dim]Computing features for prediction...[/dim]")
        from nba_betting.features.rolling import compute_rolling_features
        from nba_betting.features.four_factors import add_four_factors, add_opponent_rebound_data
        from nba_betting.features.rest_days import add_rest_features
        from nba_betting.features.builder import build_prediction_features

        rolling_df = compute_rolling_features()
        if not rolling_df.empty:
            rolling_df = add_four_factors(rolling_df)
            rolling_df = add_opponent_rebound_data(rolling_df)
            rolling_df = add_rest_features(rolling_df)

            # Add rolling Four Factors
            import pandas as pd
            rolling_df = rolling_df.sort_values(["team_id", "date", "game_id"])
            four_factor_cols = ["efg_pct", "tov_pct", "orb_pct", "ft_rate"]
            for team_id, team_df in rolling_df.groupby("team_id"):
                idx = team_df.index
                for col in four_factor_cols:
                    for w in (5, 10, 20):
                        roll_col = f"{col}_roll_{w}"
                        rolling_df.loc[idx, roll_col] = (
                            team_df[col].shift(1)
                            .rolling(window=w, min_periods=max(1, w // 2))
                            .mean().values
                        )

        # Get model and feature cols
        from nba_betting.models.xgboost_model import load_feature_means
        feat_means = load_feature_means()

        if calibrated_loaded:
            actual_model = calibrated_loaded
            # Feature cols from the base estimator
            result = load_model()
            feature_cols = result[1] if result else []
            # For per-prediction attribution we prefer the uncalibrated
            # base GBM: isotonic calibration is a monotonic post-hoc
            # transform, so it distorts the *magnitudes* of
            # leave-one-out deltas even though it preserves sign and
            # ranking. Running attribution on the raw tree ensemble
            # gives a more interpretable signal about which splits the
            # model actually leaned on.
            driver_model = (result[0] if result else actual_model)
        else:
            actual_model, feature_cols = xgb_loaded
            driver_model = actual_model

        # We stash the aligned feature row per (home_id, away_id) so
        # `generate_recommendations` can compute LOO-to-mean attribution
        # *lazily* — only for games that actually clear the edge + floor
        # gate. NO-BET rows don't render drivers anywhere, so computing
        # them eagerly here would just waste a predict_proba pass per
        # filtered game.
        driver_contexts: dict = {}
        spread_total_predictions: dict = {}

        # Load the spread/total regressors once at predict-time. If they
        # haven't been trained yet (old checkpoints) we silently fall
        # back to no spread/total picks.
        from nba_betting.models.spreads_totals import (
            load_regressors as _load_regs,
            predict_spread_total as _predict_st,
        )
        _regressors = _load_regs()

        def _xgb_predict(home_elo, away_elo, home_id=None, away_id=None):
            if rolling_df is None or rolling_df.empty or not feature_cols:
                return predict_home_win_prob(home_elo, away_elo)

            # Inject live line-movement features at prediction time. The
            # `line_movements` dict in the outer scope is populated BEFORE
            # `generate_recommendations` invokes predict_fn, so by the
            # time this closure runs the data is available. Falls back
            # to 0.0 (= "no movement data yet") if unavailable.
            extra = {}
            if home_id and away_id:
                # line_movements is keyed by (home_abbr, away_abbr); we
                # need to look it up. The closure can see the outer
                # `games` list to map IDs → abbrs.
                _game = next(
                    (g for g in games
                     if g["home_team_id"] == home_id and g["away_team_id"] == away_id),
                    None,
                )
                if _game:
                    lm = line_movements.get(
                        (_game["home_team_abbr"], _game["away_team_abbr"]), {},
                    )
                    extra["spread_movement"] = lm.get("spread_movement", 0.0)
                    extra["prob_movement"] = lm.get("prob_movement", 0.0)
                    extra["odds_disagreement"] = lm.get("odds_disagreement", 0.0)

            feat_row = build_prediction_features(
                home_id, away_id, rolling_df, home_elo, away_elo,
                feature_means=feat_means,
                extra_features=extra or None,
            )

            # Fall back to Elo if too many features are missing
            if feat_row is None:
                return predict_home_win_prob(home_elo, away_elo)

            # Align columns
            for col in feature_cols:
                if col not in feat_row.columns:
                    feat_row[col] = feat_means.get(col, 0) if feat_means else 0
            feat_row = feat_row[feature_cols]

            xgb_prob = actual_model.predict_proba(feat_row)[0, 1]

            # Stash the aligned row for lazy driver attribution in
            # `generate_recommendations`. We intentionally do NOT run
            # `compute_prediction_drivers` here — the recommendation
            # pipeline will only compute drivers for bets that clear
            # the edge threshold.
            if home_id is not None and away_id is not None:
                driver_contexts[(home_id, away_id)] = feat_row

            # Spread + total predictions share the same feature row.
            if _regressors is not None and home_id is not None and away_id is not None:
                try:
                    st = _predict_st(feat_row, _regressors)
                    spread_total_predictions[(home_id, away_id)] = st
                except Exception:
                    pass

            if use_model == "ensemble":
                elo_prob = predict_home_win_prob(home_elo, away_elo)
                return ensemble_predict(elo_prob, xgb_prob)
            return xgb_prob

        predict_fn = _xgb_predict

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

    # Get line movement data
    line_movements = {}
    try:
        from nba_betting.data.odds_tracker import get_line_movement
        from nba_betting.db.models import Team
        from nba_betting.db.session import get_session
        from sqlalchemy import select
        from datetime import date as date_type
        session = get_session()
        team_lookup = {t.abbreviation: t.id for t in session.execute(select(Team)).scalars().all()}
        session.close()
        for g in games:
            h_id = team_lookup.get(g["home_team_abbr"])
            a_id = team_lookup.get(g["away_team_abbr"])
            if h_id and a_id:
                lm = get_line_movement(date_type.today(), h_id, a_id)
                if lm.get("n_snapshots", 0) > 0:
                    line_movements[(g["home_team_abbr"], g["away_team_abbr"])] = lm
    except Exception:
        pass  # Non-critical

    # Build rolling context for explanations
    rolling_context = {}
    if rolling_df is not None and not rolling_df.empty:
        for team_id, team_df in rolling_df.groupby("team_id"):
            if not team_df.empty:
                rolling_context[team_id] = team_df.sort_values("date").iloc[-1].to_dict()

    recommendations = generate_recommendations(
        games, elos, market_odds, bankroll,
        predict_fn=predict_fn,
        injuries=injuries,
        rolling_context=rolling_context,
        line_movements=line_movements,
        espn_odds=espn_odds_data,
        spread_total_predictions=locals().get("spread_total_predictions"),
        driver_contexts=locals().get("driver_contexts"),
        driver_model=locals().get("driver_model"),
        driver_feature_means=locals().get("feat_means"),
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

    # Walk-forward validation
    console.print("\n[bold]Walk-forward validation...[/bold]")
    results = walk_forward_validate(X, y)

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

    # Evaluate calibration
    cal_probs = calibrated.predict_proba(X_cal)[:, 1]
    raw_probs = cal_model.predict_proba(X_cal)[:, 1]
    cal_metrics = evaluate_calibration(y_cal, cal_probs)
    raw_metrics = evaluate_calibration(y_cal, raw_probs)

    console.print(f"  Raw:        Brier={raw_metrics['brier_score']:.4f}, ECE={raw_metrics['ece']:.4f}")
    console.print(f"  Calibrated: Brier={cal_metrics['brier_score']:.4f}, ECE={cal_metrics['ece']:.4f}")

    # Optimize the ensemble weight by grid-searching log-loss on the
    # calibration slice. Compute Elo's per-row probability from the
    # elo_home_prob feature already in X (which is exactly Elo's
    # prediction with home-court advantage applied).
    console.print("\n[bold]Optimizing ensemble weight (Elo vs GBM)...[/bold]")
    from nba_betting.models.ensemble import learn_ensemble_weight, save_ensemble_weight
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

    total_new = 0
    for season in season_list:
        console.print(f"[dim]Syncing {season}...[/dim]")
        try:
            new = sync_season(season)
            total_new += new
            console.print(f"  Added [green]{new}[/green] new games")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")

    console.print(f"\n[bold]Total new games added: {total_new}[/bold]")

    console.print("[dim]Computing Elo ratings...[/dim]")
    elos = compute_all_elos()
    console.print(f"[green]Elo ratings computed for {len(elos)} teams.[/green]")

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


@app.command()
def simulate(
    n_sims: int = typer.Option(10_000, help="Number of Monte Carlo simulations"),
    bankroll: float = typer.Option(DEFAULT_BANKROLL, help="Starting bankroll"),
) -> None:
    """Run Monte Carlo simulation of bankroll evolution."""
    from nba_betting.features.builder import build_feature_matrix
    from nba_betting.betting.backtest import run_backtest
    from nba_betting.betting.montecarlo import simulate_bankroll
    from rich.table import Table

    console.print("[bold]Building feature matrix...[/bold]")
    X, y = build_feature_matrix()

    if X.empty:
        console.print("[red]No data. Run 'sync --seasons 3' first.[/red]")
        raise typer.Exit(1)

    console.print("[bold]Running backtest to get bet distribution...[/bold]")
    bt = run_backtest(X, y, bankroll=bankroll)
    bets = bt["bets"]

    if not bets:
        console.print("[yellow]No bets from backtest — cannot simulate.[/yellow]")
        return

    model_probs = [b["model_prob"] for b in bets]
    market_probs = [b["market_prob"] for b in bets]

    console.print(f"[bold]Running {n_sims:,} Monte Carlo simulations ({len(bets)} bets each)...[/bold]")
    results = simulate_bankroll(
        model_probs, market_probs,
        n_simulations=n_sims,
        initial_bankroll=bankroll,
    )

    table = Table(title="Monte Carlo Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Simulations", f"{results['n_simulations']:,}")
    table.add_row("Bets per Sim", str(results["n_bets_per_sim"]))
    table.add_row("", "")
    table.add_row("Median Final Bankroll", f"${results['median_final_bankroll']:.2f}")
    table.add_row("Mean Final Bankroll", f"${results['mean_final_bankroll']:.2f}")
    table.add_row("5th Percentile", f"${results['pct_5']:.2f}")
    table.add_row("25th Percentile", f"${results['pct_25']:.2f}")
    table.add_row("75th Percentile", f"${results['pct_75']:.2f}")
    table.add_row("95th Percentile", f"${results['pct_95']:.2f}")
    table.add_row("", "")
    table.add_row("P(Profit)", f"{results['probability_of_profit']:.1%}")
    table.add_row("P(Ruin)", f"{results['probability_of_ruin']:.1%}")
    table.add_row("Median ROI", f"{results['median_roi']:+.1%}")
    table.add_row("Median Max Drawdown", f"{results['median_max_drawdown']:.1%}")
    table.add_row("Worst Max Drawdown", f"{results['worst_max_drawdown']:.1%}")

    console.print(table)

    if results['probability_of_ruin'] > 0.1:
        console.print("[red]WARNING: High ruin probability. Consider reducing bet sizing.[/red]")
    if results['probability_of_profit'] > 0.6:
        console.print(f"[green]Strategy is profitable in {results['probability_of_profit']:.0%} of simulations.[/green]")


@app.command(name="snapshot-odds")
def snapshot_odds() -> None:
    """Capture a single snapshot of current Polymarket + ESPN odds.

    Designed for cron / launchd scheduling so the odds_snapshots table
    accumulates a real line-movement history. Typical schedule: every
    30 minutes between 08:00 and 23:00 local time during the NBA season.

    Exit code 0 always; warnings go to stdout for log parsing.
    """
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
        ("python3 -m nba_betting performance", "Historical accuracy + ROI"),
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
