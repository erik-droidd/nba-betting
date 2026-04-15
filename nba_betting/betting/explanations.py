"""Generate natural-language explanations for betting recommendations.

Template-based (no LLM) — examines feature differentials, injuries,
market disagreement, and momentum to produce 2-3 sentence commentary.
"""
from __future__ import annotations

from nba_betting.data.injuries import PlayerInjury


def generate_explanation(
    rec,  # BetRecommendation
    home_stats: dict | None = None,
    away_stats: dict | None = None,
    injuries: list[PlayerInjury] | None = None,
    line_movement: dict | None = None,
) -> str:
    """Generate a 2-3 sentence explanation for a bet recommendation.

    Args:
        rec: BetRecommendation with model_home_prob, market_home_prob, etc.
        home_stats: Latest rolling stats for home team (from rolling_df).
        away_stats: Latest rolling stats for away team.
        injuries: Current injury list.
        line_movement: Line movement data for this game.

    Returns:
        Human-readable explanation string.
    """
    parts = []
    bet_team = rec.home_team if rec.bet_side == "HOME" else rec.away_team
    opp_team = rec.away_team if rec.bet_side == "HOME" else rec.home_team

    if rec.bet_side == "NO BET":
        parts.append(f"No significant edge found between {rec.away_team} and {rec.home_team}.")
        return " ".join(parts)

    # 1. Market disagreement (always include).
    # Use the shrunken probability — the same one rec.edge was computed
    # from — so "model% vs market% = edge%" actually reconciles for the
    # user. Falls back to the raw model prob if shrinkage wasn't applied
    # (e.g. no market price was available for this game).
    raw_home = rec.shrunken_home_prob if rec.shrunken_home_prob is not None else rec.model_home_prob
    model_pct = raw_home if rec.bet_side == "HOME" else (1 - raw_home)
    market_pct = rec.market_home_prob if rec.bet_side == "HOME" else (1 - rec.market_home_prob)
    if market_pct > 0:
        parts.append(
            f"Model gives {bet_team} a {model_pct:.0%} win probability "
            f"vs the market's {market_pct:.0%} — a {rec.edge:.1%} edge."
        )

    # 2. Primary driver — prefer the model's own top feature attribution
    # (see nba_betting/models/drivers.py). Fall back to the rolling-stat
    # heuristic when no drivers are attached (pure-Elo path, or computation
    # failed).
    driver = _driver_from_attribution(rec)
    if not driver:
        driver = _identify_primary_driver(rec, home_stats, away_stats)
    if driver:
        parts.append(driver)

    # 3. Injury context
    if injuries:
        injury_note = _injury_context(rec, injuries, bet_team, opp_team)
        if injury_note:
            parts.append(injury_note)

    # 4. Momentum / trend
    trend = _trend_signal(rec, home_stats, away_stats, bet_team)
    if trend:
        parts.append(trend)

    # 5. Line movement
    if line_movement and line_movement.get("n_snapshots", 0) >= 2:
        lm_note = _line_movement_note(line_movement, bet_team, rec)
        if lm_note:
            parts.append(lm_note)

    return " ".join(parts) if parts else f"Model favors {bet_team} with a {rec.edge:.1%} edge."


def _driver_from_attribution(rec) -> str:
    """Cite the top feature the MODEL itself relied on for this pick.

    Uses the `rec.drivers` list produced by
    `nba_betting.models.drivers.compute_prediction_drivers`. Each entry
    is `(feature_name, delta_toward_home, feature_value)`. We want the
    top feature whose delta points *toward the bet side* — that is the
    driver actually responsible for the pick, and the one worth citing
    to the user.

    If every driver points the wrong way (rare; usually means the bet
    survived on shrinkage math alone), fall back to the heuristic
    disagreement caveat produced by `_identify_primary_driver`.
    """
    drivers = getattr(rec, "drivers", None)
    if not drivers:
        return ""

    # Positive delta = pushes prediction toward home win. If we're betting
    # HOME, we want drivers with delta > 0; if betting AWAY, delta < 0.
    bet_is_home = rec.bet_side == "HOME"
    bet_team = rec.home_team if bet_is_home else rec.away_team
    direction = 1 if bet_is_home else -1

    # Noise guard: a LOO-to-mean delta below ~0.5pp isn't really "the
    # model relying on this feature" — it's numerical jitter from
    # swapping one column for its mean. Pretending a 0.1pp shift is a
    # driver makes the explanation sentence misleading, so we require a
    # meaningful signal before citing it and fall back to the heuristic
    # when nothing clears the floor.
    MIN_DRIVER_DELTA = 0.005

    agreeing = [
        d for d in drivers
        if d[1] * direction > 0 and abs(d[1]) >= MIN_DRIVER_DELTA
    ]
    if not agreeing:
        return ""

    # Drivers come in |delta|-descending; the first agreeing one is the
    # strongest thing actually supporting the bet.
    from nba_betting.models.drivers import humanize_feature
    name, delta, value = agreeing[0]
    label = humanize_feature(name)
    support = abs(delta)

    # Translate the delta into a human-scale phrase. Small deltas are
    # "slight", big ones "primary". We avoid citing raw ±probability
    # numbers because they're unintuitive.
    if support >= 0.03:
        strength = "primary driver"
    elif support >= 0.01:
        strength = "key factor"
    else:
        strength = "top contributing factor"

    # Include the current value for context, formatted a few different
    # ways depending on whether it looks like a probability, a
    # differential, or a raw count.
    if "%" in label or "probability" in label or "rate" in label:
        value_str = f"{value:.1%}" if -1.5 <= value <= 1.5 else f"{value:+.2f}"
    elif "differential" in label or "Elo" in label:
        value_str = f"{value:+.2f}"
    else:
        value_str = f"{value:.2f}"

    return (
        f"Model's {strength} for {bet_team}: {label} ({value_str}) "
        f"— shifts P(home) by {delta:+.1%}."
    )


def _identify_primary_driver(rec, home_stats: dict | None, away_stats: dict | None) -> str:
    """Find the top contributing feature differential.

    Prefers signals that *agree* with the bet side. If every signal points
    against the bet, returns a caveat sentence flagging the disagreement
    rather than confidently parroting a driver that contradicts the call.
    This happens because the model is multi-feature — Elo, rest, four
    factors, etc. can outweigh raw rolling +/-.
    """
    if not home_stats or not away_stats:
        return ""

    # bet_side directionality: HOME bet → we want signals where home is "better".
    bet_team = rec.home_team if rec.bet_side == "HOME" else rec.away_team
    bet_is_home = rec.bet_side == "HOME"

    # Tuples: (magnitude, agrees_with_bet, sentence)
    signals = []

    # Plus/minus (most direct performance indicator)
    for window in [5, 10]:
        h_pm = home_stats.get(f"plus_minus_roll_{window}")
        a_pm = away_stats.get(f"plus_minus_roll_{window}")
        if h_pm is not None and a_pm is not None:
            diff = h_pm - a_pm
            if abs(diff) > 3:
                home_is_better = diff > 0
                better = rec.home_team if home_is_better else rec.away_team
                worse = rec.away_team if home_is_better else rec.home_team
                better_pm = h_pm if home_is_better else a_pm
                worse_pm = a_pm if home_is_better else h_pm
                agrees = home_is_better == bet_is_home
                signals.append((
                    abs(diff),
                    agrees,
                    f"{better} outperforms {worse} by {abs(diff):.1f} pts "
                    f"in {window}-game rolling +/- ({better}: {better_pm:+.1f}, {worse}: {worse_pm:+.1f})."
                ))

    # Effective FG%
    for window in [5, 10]:
        h_efg = home_stats.get(f"efg_pct_roll_{window}")
        a_efg = away_stats.get(f"efg_pct_roll_{window}")
        if h_efg is not None and a_efg is not None:
            diff = h_efg - a_efg
            if abs(diff) > 0.02:
                home_is_better = diff > 0
                better = rec.home_team if home_is_better else rec.away_team
                agrees = home_is_better == bet_is_home
                signals.append((
                    abs(diff) * 50,  # Scale to compare with +/- magnitude
                    agrees,
                    f"{better} has a {abs(diff):.1%} eFG% advantage over the last {window} games."
                ))

    # Turnover rate (lower is better)
    for window in [5, 10]:
        h_tov = home_stats.get(f"tov_pct_roll_{window}")
        a_tov = away_stats.get(f"tov_pct_roll_{window}")
        if h_tov is not None and a_tov is not None:
            diff = a_tov - h_tov  # Positive means home has lower TOV (better)
            if abs(diff) > 0.02:
                home_is_better = diff > 0
                better = rec.home_team if home_is_better else rec.away_team
                worse = rec.away_team if home_is_better else rec.home_team
                better_tov = h_tov if home_is_better else a_tov
                worse_tov = a_tov if home_is_better else h_tov
                agrees = home_is_better == bet_is_home
                signals.append((
                    abs(diff) * 30,
                    agrees,
                    f"{better} takes better care of the ball "
                    f"({better}: {better_tov:.1%} vs {worse}: {worse_tov:.1%} TOV% over {window} games)."
                ))

    if not signals:
        return ""

    # Prefer the strongest signal that *agrees* with the bet side.
    agreeing = [s for s in signals if s[1]]
    if agreeing:
        agreeing.sort(key=lambda x: x[0], reverse=True)
        return agreeing[0][2]

    # All available signals contradict the bet — flag the disagreement
    # rather than confidently citing a stat that points the wrong way.
    signals.sort(key=lambda x: x[0], reverse=True)
    contrary = signals[0][2]
    return f"Note: surface stats favor the other side ({contrary}) — model relies on Elo, rest, and four-factors to overrule."


def _injury_context(rec, injuries: list[PlayerInjury], bet_team: str, opp_team: str) -> str:
    """Generate injury-related explanation."""
    # Key injuries on the opposing team (helps the bet)
    opp_out = [i for i in injuries if i.team_abbr == opp_team and i.impact_rating >= 5.0]
    bet_out = [i for i in injuries if i.team_abbr == bet_team and i.impact_rating >= 5.0]

    parts = []
    if opp_out:
        names = ", ".join(f"{i.player_name} ({i.status})" for i in opp_out[:2])
        parts.append(f"{opp_team} weakened by {names}.")
    if bet_out:
        names = ", ".join(f"{i.player_name} ({i.status})" for i in bet_out[:2])
        parts.append(f"Note: {bet_team} also missing {names}.")

    return " ".join(parts)


def _trend_signal(rec, home_stats: dict | None, away_stats: dict | None, bet_team: str) -> str:
    """Detect momentum — compare 5-game vs 20-game rolling."""
    if not home_stats or not away_stats:
        return ""

    if rec.bet_side == "HOME":
        pm_5 = home_stats.get("plus_minus_roll_5")
        pm_20 = home_stats.get("plus_minus_roll_20")
    else:
        pm_5 = away_stats.get("plus_minus_roll_5")
        pm_20 = away_stats.get("plus_minus_roll_20")

    if pm_5 is not None and pm_20 is not None:
        diff = pm_5 - pm_20
        if diff > 3:
            return f"{bet_team} trending up: {pm_5:+.1f} last 5 vs {pm_20:+.1f} over 20 games."
        elif diff < -3:
            return f"Caution: {bet_team} in a dip, {pm_5:+.1f} last 5 vs {pm_20:+.1f} over 20."

    return ""


def _line_movement_note(line_movement: dict, bet_team: str, rec) -> str:
    """Comment on line movement if significant."""
    prob_move = line_movement.get("prob_movement", 0)
    spread_move = line_movement.get("spread_movement", 0)
    disagree = line_movement.get("odds_disagreement", 0)

    parts = []
    if abs(prob_move) > 0.03:
        direction = "toward" if prob_move > 0 else "away from"
        parts.append(f"Line moving {direction} the home team ({prob_move:+.1%}).")

    if disagree > 0.05:
        parts.append(f"Polymarket and ESPN odds disagree by {disagree:.1%} — possible mispricing.")

    return " ".join(parts)
