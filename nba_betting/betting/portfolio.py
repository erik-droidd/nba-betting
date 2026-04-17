"""Slate-level Kelly optimization (Tier 2.4).

Single-game Kelly sizes each bet as if it were the only game. On a full
NBA slate of 10 bets, per-game Kellys often sum past the prudent
exposure cap, and they ignore correlation: three "West Coast favorites
in back-to-backs" are not independent outcomes — if the fatigue story
is wrong for one it's often wrong for all of them.

The portfolio Kelly maximizes expected log-bankroll across the whole
slate jointly, with an exposure cap and an optional correlation matrix.
For independent bets with no cap it reduces to standard per-game Kelly;
the value-add is the joint optimization when bets overlap or when the
raw Kelly sum exceeds the exposure budget.

Implementation: convex NLP via scipy.optimize.minimize (SLSQP), with
fraction vector ``f_i`` in ``[0, MAX_BET_PCT]`` and sum-exposure
constraint ``Σ f_i ≤ MAX_EXPOSURE_PCT``. The objective is
``- E[log(1 + Σ f_i * r_i)]`` approximated via Monte Carlo over the
joint Bernoulli distribution.

Falls back to per-game Kelly with a proportional exposure haircut if
scipy isn't available or the optimizer fails.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nba_betting.betting.kelly import kelly_fraction
from nba_betting.config import KELLY_FRACTION, MAX_BET_PCT, MAX_EXPOSURE_PCT


@dataclass
class BetCandidate:
    """One candidate bet for slate-level optimization.

    ``prob`` = model P(side wins). ``market_price`` = market implied
    price for that side (0-1). ``side`` is just a label echoed in the
    output; the optimizer ignores it.
    """
    id: str
    prob: float
    market_price: float
    side: str = ""


def _sample_joint_outcomes(
    probs: np.ndarray,
    correlation: np.ndarray | None,
    n_samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Monte-Carlo joint outcome draws for the Kelly expectation.

    With ``correlation=None`` we sample independent Bernoulli. With a
    correlation matrix, we use a Gaussian copula: sample Normal(0, Σ),
    threshold each coordinate at Φ⁻¹(1 - p_i) to recover Bernoulli with
    the correct marginals and approximately the requested correlations.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(probs)

    if correlation is None:
        # Independent Bernoulli
        draws = rng.random((n_samples, n))
        return (draws < probs[None, :]).astype(float)

    # Gaussian copula
    from scipy.stats import norm
    try:
        L = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError:
        # Non-PSD — fall back to independent
        return _sample_joint_outcomes(probs, None, n_samples, rng)
    Z = rng.standard_normal((n_samples, n)) @ L.T
    thresholds = norm.ppf(1 - probs)
    return (Z > thresholds[None, :]).astype(float)


def _expected_log_growth(
    f: np.ndarray,
    probs: np.ndarray,
    net_odds: np.ndarray,
    samples: np.ndarray,
) -> float:
    """E[log(1 + Σ f_i * (b_i * win_i - loss_i))] over MC samples.

    ``samples[j, i]`` is 1 if bet i won in sample j, else 0. The returns
    are ``b_i`` on win and ``-1`` on loss.
    """
    # Per-sample return: samples * b - (1 - samples) * 1 = samples*(b+1) - 1
    per_sample_return = samples * (net_odds + 1.0) - 1.0  # shape (S, N)
    portfolio_return = per_sample_return @ f  # shape (S,)
    # Clip to avoid log(0) or worse if f is pathological
    growth = np.log1p(np.clip(portfolio_return, -0.999, None))
    return float(growth.mean())


def optimize_slate(
    bets: list[BetCandidate],
    correlation: np.ndarray | None = None,
    kelly_lambda: float = KELLY_FRACTION,
    max_bet_pct: float = MAX_BET_PCT,
    max_exposure_pct: float = MAX_EXPOSURE_PCT,
    drawdown_mult: float = 1.0,
) -> dict:
    """Solve the joint Kelly optimization for a slate of candidate bets.

    Args:
        bets: List of ``BetCandidate`` — each must have prob > market_price
            to be worth betting. Losing-edge bets are kept in the input
            but will be assigned zero fraction.
        correlation: Optional N×N correlation matrix between bet outcomes.
            Defaults to identity (independent). Must be PSD; non-PSD
            inputs silently fall back to independent.
        kelly_lambda: Fractional-Kelly scale (e.g. 0.25 = quarter-Kelly).
        max_bet_pct: Per-bet cap on fraction of bankroll.
        max_exposure_pct: Total cross-bet exposure cap.
        drawdown_mult: Overall multiplier (e.g. 0.5 in drawdown mode).

    Returns:
        ``{"fractions": np.ndarray[N], "objective": float,
           "fallback": bool, "reason": str}``.
        ``fallback=True`` means the optimizer couldn't solve and we
        returned haircut per-game Kelly instead.
    """
    n = len(bets)
    if n == 0:
        return {"fractions": np.zeros(0), "objective": 0.0, "fallback": False, "reason": "empty"}

    probs = np.array([b.prob for b in bets], dtype=float)
    prices = np.array([b.market_price for b in bets], dtype=float)

    # Filter: keep only positive-EV bets (others get f=0 in output).
    positive_mask = (prices > 0) & (prices < 1) & (probs > prices)

    # Per-bet fallback Kelly fractions (scaled by lambda + drawdown).
    per_bet_kelly = np.array([
        kelly_fraction(b.prob, b.market_price, lambda_=kelly_lambda, drawdown_mult=drawdown_mult)
        for b in bets
    ])

    # Attempt joint optimization
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Haircut fallback: scale per-bet kellys proportionally to fit the cap.
        fractions = _apply_exposure_cap(per_bet_kelly, max_exposure_pct)
        return {
            "fractions": fractions,
            "objective": float("nan"),
            "fallback": True,
            "reason": "scipy unavailable",
        }

    if positive_mask.sum() == 0:
        return {
            "fractions": np.zeros(n),
            "objective": 0.0,
            "fallback": False,
            "reason": "no positive EV",
        }

    # Work with only positive-EV bets for the optimization, then rebuild
    # the full-length output vector at the end. Keeps the search space
    # tight and avoids the optimizer getting distracted by zero-EV tails.
    sub_probs = probs[positive_mask]
    sub_prices = prices[positive_mask]
    sub_net_odds = (1.0 / sub_prices) - 1.0
    sub_correlation = None
    if correlation is not None:
        idx = np.where(positive_mask)[0]
        sub_correlation = np.asarray(correlation)[np.ix_(idx, idx)]

    samples = _sample_joint_outcomes(
        sub_probs, sub_correlation, n_samples=2000,
    )

    # Objective: negative expected log growth (scipy minimizes).
    def neg_elog(f):
        return -_expected_log_growth(f, sub_probs, sub_net_odds, samples)

    x0 = np.clip(per_bet_kelly[positive_mask], 0.0, max_bet_pct)
    # If the warm-start already exceeds the exposure cap, scale it down.
    if x0.sum() > max_exposure_pct:
        x0 = x0 * (max_exposure_pct / x0.sum())

    bounds = [(0.0, max_bet_pct)] * len(x0)
    constraints = [{"type": "ineq", "fun": lambda f: max_exposure_pct - np.sum(f)}]

    try:
        result = minimize(
            neg_elog, x0=x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 80, "ftol": 1e-6},
        )
        if not result.success:
            raise RuntimeError(result.message)
        # Quarter-Kelly-style shrinkage on the full-Kelly optimum.
        # Without this, the optimizer finds the aggressive full-Kelly
        # sizing; users configured kelly_lambda for a reason.
        sub_fractions = result.x * kelly_lambda * drawdown_mult
        sub_fractions = np.clip(sub_fractions, 0.0, max_bet_pct)
        # Re-apply the exposure cap post-haircut.
        if sub_fractions.sum() > max_exposure_pct:
            sub_fractions = sub_fractions * (max_exposure_pct / sub_fractions.sum())

        fractions = np.zeros(n)
        fractions[positive_mask] = sub_fractions
        return {
            "fractions": fractions,
            "objective": float(-result.fun),
            "fallback": False,
            "reason": "",
        }
    except Exception as e:
        fractions = _apply_exposure_cap(per_bet_kelly, max_exposure_pct)
        return {
            "fractions": fractions,
            "objective": float("nan"),
            "fallback": True,
            "reason": f"optimizer failed: {e}",
        }


def _apply_exposure_cap(per_bet_kelly: np.ndarray, max_exposure: float) -> np.ndarray:
    """Proportionally shrink per-bet Kellys so the total ≤ ``max_exposure``."""
    total = float(per_bet_kelly.sum())
    if total <= max_exposure or total == 0:
        return per_bet_kelly.copy()
    return per_bet_kelly * (max_exposure / total)


def build_simple_correlation(bets: list[BetCandidate], same_day_rho: float = 0.05) -> np.ndarray:
    """Build a light correlation matrix based on shared-day covariance.

    Without more signal (division, conference, shared injury) we assume a
    small positive correlation between same-day NBA bets (rho ≈ 0.05) —
    league-wide effects like referee trends, news cycles, or basketball
    narratives tend to correlate outcomes mildly. Overriding this with a
    richer matrix (e.g., computed from historical residuals) is a future
    improvement.
    """
    n = len(bets)
    corr = np.eye(n)
    # Mild positive correlation everywhere off-diagonal.
    off = same_day_rho
    corr[~np.eye(n, dtype=bool)] = off
    return corr
