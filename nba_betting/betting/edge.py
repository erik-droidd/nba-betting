"""Edge calculation and +EV identification."""

from nba_betting.config import MIN_EDGE_THRESHOLD, SUSPICIOUS_EDGE_THRESHOLD


def compute_edge(model_prob: float, market_price: float) -> float:
    """Compute betting edge: P_model * decimal_odds - 1.

    Args:
        model_prob: Model's estimated probability of the outcome (0-1).
        market_price: Polymarket price / implied probability (0-1).

    Returns:
        Edge as a decimal (e.g., 0.05 = 5% edge).
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    decimal_odds = 1.0 / market_price
    return model_prob * decimal_odds - 1.0


def expected_value_per_dollar(model_prob: float, market_price: float) -> float:
    """Expected profit per dollar wagered."""
    return compute_edge(model_prob, market_price)


def is_positive_ev(edge: float, threshold: float = MIN_EDGE_THRESHOLD) -> bool:
    """Check if edge exceeds minimum threshold."""
    return edge >= threshold


def is_suspicious_edge(edge: float, threshold: float = SUSPICIOUS_EDGE_THRESHOLD) -> bool:
    """Check if edge is suspiciously large (potential data/model issue)."""
    return edge >= threshold


def confidence_badge(edge: float) -> str:
    """Assign a confidence badge based on edge magnitude."""
    if edge >= SUSPICIOUS_EDGE_THRESHOLD:
        return "SUSPECT"
    elif edge >= 0.05:
        return "STRONG"
    elif edge >= 0.03:
        return "MODERATE"
    elif edge >= MIN_EDGE_THRESHOLD:
        return "LEAN"
    else:
        return "NO BET"
