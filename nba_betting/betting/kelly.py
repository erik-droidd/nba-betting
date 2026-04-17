"""Kelly Criterion bet sizing with drawdown awareness.

Standard quarter-Kelly is static — it sizes the same bet after a 5-game
losing streak as after a hot week. This module adds an optional
drawdown multiplier that shrinks the Kelly fraction when recent
realized ROI diverges from expectations, reducing tail risk during
cold streaks without requiring a manual bankroll override.
"""
from __future__ import annotations

from nba_betting.config import KELLY_FRACTION, MAX_BET_PCT, MAX_EXPOSURE_PCT


def signal_dependent_lambda(
    base_lambda: float,
    edge: float,
    clv_tstat: float | None = None,
    model_market_disagreement: float | None = None,
) -> float:
    """Tier 2.5 — scale base Kelly fraction by signal strength.

    The static quarter-Kelly assumes every bet is equally informative.
    In reality, a 2% edge on a game the model agrees with the market on
    (i.e., edge came from a small price inefficiency) is qualitatively
    different from a 8% edge where the model strongly disagrees — the
    latter has more idiosyncratic risk and should be sized somewhat
    smaller despite the higher edge point estimate.

    This function multiplies ``base_lambda`` by three soft factors:

    - **edge_factor**: taper aggressively beyond ~10% edges (suspicious
      territory — usually mispriced odds or stale lines).
    - **clv_factor**: amplify lambda when we have measurable CLV skill
      (t-stat > 1.5). If CLV is negative, shrink it.
    - **disagreement_factor**: shrink lambda when the model and market
      strongly disagree (|prob_diff| > 0.15) — high idiosyncratic risk.

    Returns a lambda in ``[base_lambda * 0.25, base_lambda * 1.25]`` so
    the signal scaling can at most quarter or 1.25x the static rate.
    """
    # Edge taper: multiplicative Gaussian around the sweet spot at 4-5%.
    # The static lambda fully applies in the 2–8% band; beyond ~15% it
    # tapers toward 0.5x to guard against mispriced lines.
    if edge <= 0:
        return base_lambda * 0.5
    if edge < 0.02:
        edge_factor = 0.6  # barely-there edge
    elif edge <= 0.08:
        edge_factor = 1.0
    elif edge <= 0.12:
        edge_factor = 0.9
    elif edge <= 0.20:
        edge_factor = 0.75
    else:
        edge_factor = 0.5

    # CLV factor: if we have no CLV history (bootstrap phase), leave
    # unchanged. Otherwise reward a positive-t-stat and penalize negative.
    if clv_tstat is None:
        clv_factor = 1.0
    elif clv_tstat >= 1.5:
        clv_factor = 1.15
    elif clv_tstat <= -1.5:
        clv_factor = 0.7
    else:
        clv_factor = 1.0

    # Disagreement factor: if |edge| > 0.15 and the model strongly
    # disagrees with the market the idiosyncratic risk is higher.
    if model_market_disagreement is None:
        disagree_factor = 1.0
    elif abs(model_market_disagreement) >= 0.20:
        disagree_factor = 0.65
    elif abs(model_market_disagreement) >= 0.15:
        disagree_factor = 0.85
    else:
        disagree_factor = 1.0

    scaled = base_lambda * edge_factor * clv_factor * disagree_factor
    # Clamp to ±quarter of the base so a single-game signal can't blow up sizing.
    lo = base_lambda * 0.25
    hi = base_lambda * 1.25
    return max(lo, min(hi, scaled))


def kelly_fraction(
    prob: float,
    market_price: float,
    lambda_: float = KELLY_FRACTION,
    drawdown_mult: float = 1.0,
    clv_tstat: float | None = None,
) -> float:
    """Compute fractional Kelly bet size as fraction of bankroll.

    Args:
        prob: Model's estimated win probability.
        market_price: Polymarket implied probability / price.
        lambda_: Fraction of full Kelly to use (0.25 = quarter-Kelly).
        drawdown_mult: Multiplier applied to the fractional Kelly output
            after all other math. 1.0 = normal sizing. <1.0 = shrink
            after recent drawdown. See ``compute_drawdown_multiplier``.
        clv_tstat: Optional CLV t-statistic for Tier 2.5 signal-dependent
            sizing. When provided, lambda is scaled up/down based on
            demonstrated closing-line-value skill.

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

    # Tier 2.5 — scale lambda by signal strength when we have the inputs.
    edge = prob - market_price
    effective_lambda = signal_dependent_lambda(
        lambda_,
        edge=edge,
        clv_tstat=clv_tstat,
        model_market_disagreement=edge,
    )

    fractional = effective_lambda * full_kelly * max(0.0, min(1.0, drawdown_mult))
    return min(fractional, MAX_BET_PCT)


def compute_bet_size(
    bankroll: float,
    prob: float,
    market_price: float,
    lambda_: float = KELLY_FRACTION,
    drawdown_mult: float = 1.0,
    clv_tstat: float | None = None,
) -> float:
    """Compute dollar bet size."""
    fraction = kelly_fraction(prob, market_price, lambda_, drawdown_mult, clv_tstat=clv_tstat)
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
