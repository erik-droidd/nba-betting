"""Bayesian shrinkage of model probabilities toward market log-odds.

Why this exists
---------------
NBA moneyline markets — especially Polymarket and the major US books —
are highly efficient. The closing market line typically beats almost any
publicly-available model. Treating model and market as peers and naively
betting whenever the model disagrees produces tons of phantom edges,
because most apparent disagreements are model noise rather than genuine
mispricings.

The standard quant fix is to treat the market as a strong **prior** and
the model as a **likelihood**, and combine them in **log-odds (logit)
space**. This is mathematically equivalent to a Bayesian update where the
posterior log-odds is a precision-weighted average of the prior and the
evidence. In practice it means: small model-vs-market gaps get pulled
back toward the market; large gaps survive but are dampened proportional
to how much we trust the model.

The shrinkage strength `lambda` (in [0,1]) controls how much weight the
market gets:

    posterior_logit = (1 - lambda) * model_logit + lambda * market_logit

- lambda = 0  → pure model (current behavior)
- lambda = 1  → pure market (no bets ever)
- lambda = 0.5 → equal weight; large gaps dampened ~50%

For a system whose model is roughly competitive with the market (ECE
~0.03, Brier ~0.22 on held-out folds) but produces tons of >15% edges
on a closing line, lambda in [0.4, 0.7] is typical. We default to 0.6 —
biased toward the market, which is the right Bayesian stance when the
prior is strong.
"""
from __future__ import annotations

import math


def _logit(p: float) -> float:
    """Numerically stable scalar logit. Clips to [1e-6, 1-1e-6]."""
    p = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def shrink_to_market(
    model_prob: float,
    market_prob: float,
    lambda_market: float = 0.6,
) -> float:
    """Bayesian shrinkage of model probability toward market in log-odds space.

    Args:
        model_prob: Model's P(home win) (post-injury adjustment).
        market_prob: Market's implied P(home win). Must be in (0, 1).
        lambda_market: Weight on the market prior in [0, 1]. Higher = trust
            the market more. Default 0.6 (market-leaning, conservative).

    Returns:
        Shrunken P(home win). If market_prob is 0 or 1 (degenerate), returns
        the unshrunken model_prob to avoid sending it to a tail.
    """
    if model_prob <= 0 or model_prob >= 1:
        return model_prob
    if market_prob <= 0 or market_prob >= 1:
        return model_prob
    lam = max(0.0, min(1.0, lambda_market))
    z = (1.0 - lam) * _logit(model_prob) + lam * _logit(market_prob)
    return _sigmoid(z)
