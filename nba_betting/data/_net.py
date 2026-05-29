"""Shared network resilience: retry transient failures with backoff + logging.

The NBA / ESPN / Polymarket endpoints are flaky — ``stats.nba.com`` throttles
datacenter IPs, and all three time out under load. Before this, a single
transient blip made a fetch return ``[]`` / raise, which upstream code turned
into "no games" or "N/A odds" — indistinguishable from a genuinely empty
slate, and silent. ``with_retries`` retries a few times with exponential
backoff and **logs a warning when it ultimately gives up**, so silent
degradation becomes diagnosable (issue: data-layer robustness review).
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger("nba_betting.data")

T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    backoff: float = 0.6,
    what: str = "request",
) -> T:
    """Call ``fn`` with up to ``attempts`` tries and exponential backoff.

    Re-raises the last exception if every attempt fails (callers keep their
    own graceful-degradation handling), but logs a warning first so the
    failure isn't silent. Backoff between attempt ``i`` and ``i+1`` is
    ``backoff * 2**i`` seconds.
    """
    last_exc: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — transient network/API errors
            last_exc = e
            if i < attempts - 1:
                logger.debug(
                    "%s attempt %d/%d failed (%s); retrying", what, i + 1, attempts, e
                )
                time.sleep(backoff * (2 ** i))
    logger.warning("%s failed after %d attempts: %s", what, attempts, last_exc)
    raise last_exc  # type: ignore[misc]
