"""Rich console output for betting recommendations."""
from __future__ import annotations

from datetime import date

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from nba_betting.betting.recommendations import BetRecommendation

console = Console()


def _badge_style(badge: str) -> str:
    if badge == "SUSPECT":
        return "bold red"
    elif badge == "STRONG":
        return "bold green"
    elif badge == "MODERATE":
        return "yellow"
    elif badge == "LEAN":
        return "dim yellow"
    return "dim"


def display_recommendations(
    recommendations: list[BetRecommendation],
    bankroll: float,
) -> None:
    """Display betting recommendations in a Rich table."""
    today = date.today().strftime("%B %d, %Y")

    console.print()
    console.print(
        Panel(
            f"[bold]NBA Betting Recommendations[/bold] — {today}\n"
            f"[dim]Bankroll: ${bankroll:,.2f} | Min Edge: 2% | Quarter-Kelly[/dim]",
            border_style="blue",
        )
    )
    console.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Matchup", style="white", min_width=15)
    table.add_column("Model", justify="center", min_width=8)
    table.add_column("Market", justify="center", min_width=8)
    table.add_column("Bet", justify="center", min_width=6)
    table.add_column("Edge", justify="right", min_width=7)
    table.add_column("EV/$1", justify="right", min_width=7)
    table.add_column("Kelly", justify="right", min_width=7)
    table.add_column("Size", justify="right", min_width=8)
    table.add_column("Signal", justify="center", min_width=10)

    for rec in recommendations:
        matchup = f"{rec.away_team} @ {rec.home_team}"

        # Show the *shrunken* probability — the same one the edge is
        # computed against — so model% / market% / edge% reconcile.
        # If shrinkage wasn't applied (no market), fall back to raw model.
        display_home_prob = (
            rec.shrunken_home_prob
            if rec.shrunken_home_prob is not None
            else rec.model_home_prob
        )
        model_str = f"{display_home_prob:.1%}"
        market_str = f"{rec.market_home_prob:.1%}" if rec.market_home_prob > 0 else "N/A"

        if rec.bet_side == "NO BET":
            bet_str = "—"
            edge_str = f"{rec.edge:+.1%}" if rec.edge != 0 else "—"
            ev_str = "—"
            kelly_str = "—"
            size_str = "—"
        else:
            bet_str = rec.home_team if rec.bet_side == "HOME" else rec.away_team
            edge_str = f"{rec.edge:+.1%}"
            ev_str = f"${rec.ev_per_dollar:.2f}"
            kelly_str = f"{rec.kelly_pct:.1%}"
            size_str = f"${rec.bet_size:.0f}"

        badge_text = Text(rec.badge, style=_badge_style(rec.badge))

        table.add_row(
            matchup,
            model_str,
            market_str,
            bet_str,
            edge_str,
            ev_str,
            kelly_str,
            size_str,
            badge_text,
        )

    console.print(table)

    # Show explanations below the table
    for rec in recommendations:
        has_alt_pick = (
            rec.spread_pick != "NO BET" or rec.total_pick != "NO BET"
        )
        if rec.bet_side != "NO BET" or has_alt_pick:
            spread_str = f" | Spread: {rec.spread:+.1f}" if rec.spread is not None else ""
            ou_str = f" | O/U: {rec.over_under:.1f}" if rec.over_under is not None else ""
            console.print(f"  [bold]{rec.away_team} @ {rec.home_team}[/bold]{spread_str}{ou_str}")
            if rec.badge == "SUSPECT":
                console.print(f"  [bold red]⚠ SUSPECT EDGE ({rec.edge:.1%})[/bold red] — Edge exceeds 15%, likely a data or model issue. "
                              f"Verify market odds manually before betting.")
            if rec.spread_pick != "NO BET":
                console.print(
                    f"  [cyan]Spread pick:[/cyan] {rec.spread_pick} "
                    f"[dim](model: {rec.predicted_spread:+.1f}, edge: {rec.spread_edge:+.1f} pts)[/dim]"
                )
            if rec.total_pick != "NO BET":
                console.print(
                    f"  [cyan]Total pick:[/cyan] {rec.total_pick} "
                    f"[dim](model: {rec.predicted_total:.1f}, edge: {rec.total_edge:+.1f} pts)[/dim]"
                )
            if rec.explanation:
                console.print(f"  [dim italic]{rec.explanation}[/dim italic]")
            console.print()

    # Summary
    bets = [r for r in recommendations if r.bet_side != "NO BET"]
    if bets:
        total_exposure = sum(r.bet_size for r in bets)
        console.print(
            f"\n[bold green]{len(bets)} actionable bet(s)[/bold green] | "
            f"Total exposure: ${total_exposure:.0f} "
            f"({total_exposure / bankroll:.1%} of bankroll)"
        )
    else:
        console.print("\n[dim]No +EV bets found for today's games.[/dim]")

    console.print()


def display_elo_ratings(elos: dict[str, float]) -> None:
    """Display current Elo ratings sorted by rating."""
    console.print()
    console.print(Panel("[bold]Current NBA Elo Ratings[/bold]", border_style="blue"))
    console.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", min_width=3)
    table.add_column("Team", min_width=5)
    table.add_column("Elo", justify="right", min_width=6)
    table.add_column("vs Avg", justify="right", min_width=7)

    sorted_elos = sorted(elos.items(), key=lambda x: x[1], reverse=True)

    for i, (team, elo) in enumerate(sorted_elos, 1):
        diff = elo - 1500
        diff_style = "green" if diff > 0 else "red" if diff < 0 else "dim"
        table.add_row(
            str(i),
            team,
            f"{elo:.0f}",
            Text(f"{diff:+.0f}", style=diff_style),
        )

    console.print(table)
    console.print()


def display_no_games() -> None:
    """Display message when no games are scheduled."""
    console.print()
    console.print("[yellow]No NBA games scheduled for today.[/yellow]")
    console.print()
