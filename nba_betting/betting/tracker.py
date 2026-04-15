"""Performance tracking: record predictions and compute historical ROI."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path

from nba_betting.config import DATA_DIR

HISTORY_FILE = DATA_DIR / "prediction_history.json"


@dataclass
class PredictionRecord:
    date: str
    home_team: str
    away_team: str
    model_home_prob: float
    market_home_prob: float
    bet_side: str  # "HOME", "AWAY", "NO BET"
    edge: float
    bet_size: float
    model_type: str = "unknown"  # "elo", "ensemble"
    # Filled in after game completes
    home_won: bool | None = None
    profit: float | None = None
    # --- CLV (Closing Line Value) tracking ---
    # Populated at prediction time from the earliest available snapshot.
    opening_market_prob: float | None = None
    # Populated at result-resolution time from the latest pre-game snapshot.
    closing_market_prob: float | None = None
    # CLV = (closing_prob_bet_side / opening_prob_bet_side) - 1
    # Positive → we bet at a better price than the close (edge was real).
    clv: float | None = None


def load_history() -> list[PredictionRecord]:
    """Load prediction history."""
    if not HISTORY_FILE.exists():
        return []
    data = json.loads(HISTORY_FILE.read_text())
    return [PredictionRecord(**r) for r in data]


def save_history(records: list[PredictionRecord]) -> None:
    """Save prediction history."""
    DATA_DIR.mkdir(exist_ok=True)
    HISTORY_FILE.write_text(json.dumps([asdict(r) for r in records], indent=2))


def record_predictions(recommendations) -> int:
    """Save today's predictions to history. Returns count of new records."""
    history = load_history()
    today = str(date.today())

    # Don't duplicate
    existing_keys = {(r.date, r.home_team, r.away_team) for r in history}

    # Fetch opening lines from snapshots for CLV tracking (best-effort).
    opening_probs: dict[tuple[str, str], float] = {}
    try:
        from nba_betting.data.odds_tracker import get_opening_line
        from nba_betting.db.models import Team
        from nba_betting.db.session import get_session
        from sqlalchemy import select

        session = get_session()
        _teams = {t.abbreviation: t.id for t in session.execute(select(Team)).scalars().all()}
        session.close()

        for rec in recommendations:
            h_id = _teams.get(rec.home_team)
            a_id = _teams.get(rec.away_team)
            if h_id and a_id:
                opening = get_opening_line(date.today(), h_id, a_id)
                if opening and opening.get("home_prob"):
                    opening_probs[(rec.home_team, rec.away_team)] = opening["home_prob"]
    except Exception:
        pass  # CLV is non-critical

    new_count = 0
    for rec in recommendations:
        key = (today, rec.home_team, rec.away_team)
        if key in existing_keys:
            continue

        history.append(PredictionRecord(
            date=today,
            home_team=rec.home_team,
            away_team=rec.away_team,
            model_home_prob=rec.model_home_prob,
            market_home_prob=rec.market_home_prob,
            bet_side=rec.bet_side,
            edge=rec.edge,
            bet_size=rec.bet_size,
            opening_market_prob=opening_probs.get(
                (rec.home_team, rec.away_team),
            ),
        ))
        new_count += 1

    save_history(history)
    return new_count


def update_results() -> int:
    """Update historical predictions with actual game outcomes.

    Looks up completed games in the database and fills in results.
    Also populates closing_market_prob and CLV when snapshot data exists.
    Returns count of records updated.
    """
    from sqlalchemy import select
    from nba_betting.db.models import Game, Team
    from nba_betting.db.session import get_session

    history = load_history()
    if not history:
        return 0

    session = get_session()
    try:
        teams = {t.abbreviation: t.id for t in session.execute(select(Team)).scalars().all()}

        updated = 0
        for record in history:
            if record.home_won is not None:
                continue  # Already resolved

            # Look up game result
            home_id = teams.get(record.home_team)
            away_id = teams.get(record.away_team)
            if not home_id or not away_id:
                continue

            game = session.execute(
                select(Game).where(
                    Game.home_team_id == home_id,
                    Game.away_team_id == away_id,
                    Game.date == datetime.strptime(record.date, "%Y-%m-%d").date(),
                    Game.home_score.isnot(None),
                )
            ).scalars().first()

            if not game:
                continue

            record.home_won = game.home_win

            # Calculate profit
            if record.bet_side == "NO BET":
                record.profit = 0.0
            elif record.bet_side == "HOME":
                if game.home_win:
                    decimal_odds = 1.0 / record.market_home_prob if record.market_home_prob > 0 else 0
                    record.profit = record.bet_size * (decimal_odds - 1)
                else:
                    record.profit = -record.bet_size
            elif record.bet_side == "AWAY":
                market_away = 1.0 - record.market_home_prob if record.market_home_prob > 0 else 0
                if not game.home_win:
                    decimal_odds = 1.0 / market_away if market_away > 0 else 0
                    record.profit = record.bet_size * (decimal_odds - 1)
                else:
                    record.profit = -record.bet_size

            # CLV: fetch closing line (latest pre-game snapshot).
            if record.closing_market_prob is None and record.bet_side != "NO BET":
                try:
                    from nba_betting.data.odds_tracker import get_closing_line
                    game_date = datetime.strptime(record.date, "%Y-%m-%d").date()
                    closing = get_closing_line(game_date, home_id, away_id)
                    if closing and closing.get("home_prob"):
                        record.closing_market_prob = closing["home_prob"]

                        # Compute CLV: compare closing price to the price
                        # we actually bet at (market_home_prob at bet time).
                        # Positive CLV = we got a better price than the close.
                        opening_prob = record.opening_market_prob or record.market_home_prob
                        close_p = record.closing_market_prob
                        if record.bet_side == "HOME" and opening_prob > 0:
                            # Closing home prob higher → home "got shorter"
                            # → we bet at a better price (lower implied %).
                            record.clv = (close_p / opening_prob) - 1.0
                        elif record.bet_side == "AWAY" and opening_prob < 1:
                            open_away = 1.0 - opening_prob
                            close_away = 1.0 - close_p
                            if open_away > 0:
                                record.clv = (close_away / open_away) - 1.0
                except Exception:
                    pass  # CLV is non-critical

            updated += 1

        save_history(history)
        return updated
    finally:
        session.close()


def compute_performance() -> dict:
    """Compute performance metrics from prediction history."""
    history = load_history()
    resolved = [r for r in history if r.home_won is not None]
    bets = [r for r in resolved if r.bet_side != "NO BET"]

    if not resolved:
        return {"total_predictions": len(history), "resolved": 0}

    # Prediction accuracy (did model pick the right side?)
    correct = 0
    for r in resolved:
        if r.model_home_prob >= 0.5 and r.home_won:
            correct += 1
        elif r.model_home_prob < 0.5 and not r.home_won:
            correct += 1

    accuracy = correct / len(resolved) if resolved else 0

    # Betting performance
    total_wagered = sum(abs(r.bet_size) for r in bets)
    total_profit = sum(r.profit for r in bets if r.profit is not None)
    roi = total_profit / total_wagered if total_wagered > 0 else 0

    wins = sum(1 for r in bets if r.profit is not None and r.profit > 0)
    losses = sum(1 for r in bets if r.profit is not None and r.profit < 0)
    win_rate = wins / len(bets) if bets else 0

    # Bankroll tracking
    bankroll_curve = [1000.0]  # Starting bankroll
    for r in sorted(bets, key=lambda x: x.date):
        if r.profit is not None:
            bankroll_curve.append(bankroll_curve[-1] + r.profit)

    max_bankroll = max(bankroll_curve)
    max_drawdown = 0.0
    peak = bankroll_curve[0]
    for val in bankroll_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    # Calibration check: bin predictions by probability
    calibration_bins = []
    if resolved:
        import numpy as np
        probs = np.array([r.model_home_prob for r in resolved])
        actuals = np.array([1.0 if r.home_won else 0.0 for r in resolved])
        bin_edges = np.linspace(0, 1, 6)  # 5 bins
        for i in range(len(bin_edges) - 1):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if i == len(bin_edges) - 2:
                mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
            n_in = mask.sum()
            if n_in > 0:
                calibration_bins.append({
                    "range": f"{bin_edges[i]:.0%}-{bin_edges[i+1]:.0%}",
                    "count": int(n_in),
                    "avg_predicted": round(float(probs[mask].mean()), 3),
                    "avg_actual": round(float(actuals[mask].mean()), 3),
                })

    # CLV metrics — the gold-standard for "did we actually have edge?"
    # Positive avg CLV means we consistently bet at better prices than
    # the close, which decouples from bet-outcome variance.
    clv_values = [r.clv for r in bets if r.clv is not None]
    avg_clv = None
    clv_tstat = None
    if clv_values:
        import numpy as np
        clv_arr = np.array(clv_values)
        avg_clv = round(float(clv_arr.mean()), 4)
        if len(clv_arr) >= 3 and clv_arr.std() > 0:
            clv_tstat = round(
                float(clv_arr.mean() / (clv_arr.std() / np.sqrt(len(clv_arr)))),
                2,
            )

    return {
        "total_predictions": len(history),
        "resolved": len(resolved),
        "prediction_accuracy": round(accuracy, 4),
        "total_bets": len(bets),
        "wins": wins,
        "losses": losses,
        "bet_win_rate": round(win_rate, 4),
        "total_wagered": round(total_wagered, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 4),
        "current_bankroll": round(bankroll_curve[-1], 2),
        "max_drawdown": round(max_drawdown, 4),
        "calibration_bins": calibration_bins,
        # CLV metrics
        "avg_clv": avg_clv,
        "clv_tstat": clv_tstat,
        "clv_count": len(clv_values),
    }
