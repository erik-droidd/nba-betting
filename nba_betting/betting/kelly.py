"""Kelly Criterion bet sizing with drawdown awareness.

Standard quarter-Kelly is static — it sizes the same bet after a 5-game
losing streak as after a hot week. This module adds an optional
drawdown multiplier that shrinks the Kelly fraction when recent
realized ROI diverges from expectations, reducing tail risk during
cold streaks without requiring a manual bankroll override.
"""
from __future__ import annotations

from nba_betting.config import KELLY_FRACTION, MAX_BET_PCT, MAX_EXPOSURE_PCT


def kelly_fraction(
    prob: float,
    market_price: float,
    lambda_: float = KELLY_FRACTION,
    drawdown_mult: float = 1.0,
) -> float:
    """Compute fractional Kelly bet size as fraction of bankroll.

    Args:
        prob: Model's estimated win probability.
        market_price: Polymarket implied probability / price.
        lambda_: Fraction of full Kelly to use (0.25 = quarter-Kelly).
        drawdown_mult: Multiplier applied to the fractional Kelly output
            after all other math. 1.0 = normal sizing. <1.0 = shrink
            after recent drawdown. See ``compute_drawdown_multiplier``.

    Returns:
        Recommended bet as fraction of bankroll (0 to MAX_BET_PCT).
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0

    decimal_odds = 1.0 / market_price
    b = decimal_odds - 1.0  # Net odds
    q = 1.0 - prob

    if b <= 0:
        return 0.0

    full_kelly = (b * prob - q) / b
    if full_kelly <= 0:
        return 0.0

    fractional = lambda_ * full_kelly * max(0.0, min(1.0, drawdown_mult))
    return min(fractional, MAX_BET_PCT)


def compute_bet_size(
    bankroll: float,
    prob: float,
    market_price: float,
    lambda_: float = KELLY_FRACTION,
    drawdown_mult: float = 1.0,
) -> float:
    """Compute dollar bet size."""
    fraction = kelly_fraction(prob, market_price, lambda_, drawdown_mult)
    return round(bankroll * fraction, 2)


def compute_drawdown_multiplier(
    recent_roi: float,
    lookback_bets: int = 0,
    mild_threshold: float = -0.05,
    severe_threshold: float = -0.10,
    mild_mult: float = 0.50,
    severe_mult: float = 0.25,
    min_bets: int = 5,
) -> float:
    """Return a Kelly multiplier based on recent bet ROI.

    After a losing streak, shrinking position size reduces variance
    and prevents compounding drawdown. This implements a simple
    two-tier step-function:

    - ``recent_roi >= mild_threshold``: no adjustment (1.0)
    - ``mild_threshold > recent_roi >= severe_threshold``: half Kelly
    - ``recent_roi < severe_threshold``: quarter Kelly

    Args:
        recent_roi: Realized ROI over last ``lookback_bets`` settled
            bets. E.g. -0.07 = 7% loss on wagered capital.
        lookback_bets: Number of bets in the lookback window. If below
            ``min_bets``, returns 1.0 (not enough data).
        mild_threshold: ROI threshold for first reduction step.
        severe_threshold: ROI threshold for aggressive reduction.
        mild_mult: Kelly multiplier at mild drawdown.
        severe_mult: Kelly multiplier at severe drawdown.
        min_bets: Minimum resolved bets required before activating.

    Returns:
        Multiplier in ``[severe_mult, 1.0]``.
    """
    if lookback_bets < min_bets:
        return 1.0

    if recent_roi < severe_threshold:
        return severe_mult
    if recent_roi < mild_threshold:
        return mild_mult
    return 1.0


def get_recent_roi(lookback: int = 10) -> tuple[float, int]:
    """Compute ROI over the most recent ``lookback`` settled bets.

    Returns ``(roi, count)`` where ``count`` is the number of resolved
    bets actually used (may be < ``lookback`` if history is short).
    """
    try:
        from nba_betting.betting.tracker import load_history
        history = load_history()
        settled = [
            r for r in reversed(history)
            if r.bet_side != "NO BET" and r.profit is not None
        ][:lookback]
        if not settled:
            return 0.0, 0
        wagered = sum(abs(r.bet_size) for r in settled)
        profit = sum(r.profit for r in settled)
        roi = profit / wagered if wagered > 0 else 0.0
        return roi, len(settled)
    except Exception:
        return 0.0, 0


def check_exposure(
    current_exposure: float,
    new_bet: float,
    bankroll: float,
) -> bool:
    """Check if adding a new bet would exceed max exposure."""
    total = current_exposure + new_bet
    return (total / bankroll) <= MAX_EXPOSURE_PCT
