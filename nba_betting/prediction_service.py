"""Shared prediction engine for both `cli.predict` and the API route.

The model-load → rolling-feature pipeline → predict closure used to be
duplicated in `cli.py` and `api/routes.py`. The two drifted — the CLI silently
omitted split off/def Elos for a while (#29), feeding the primary path degraded
features. This module is the single source of truth so the paths cannot diverge
again, and it owns the per-slate precompute caches (#3/#4).

Usage::

    engine = PredictionEngine(games, off_def_elos, blend=True)
    if engine.available:
        engine.injuries = injuries            # set before predicting
        engine.line_movements = line_movements
        recs = generate_recommendations(..., predict_fn=engine.predict,
                                         driver_contexts=engine.driver_contexts,
                                         driver_model=engine.driver_model,
                                         driver_feature_means=engine.feat_means,
                                         spread_total_predictions=engine.spread_total_predictions)
"""
from __future__ import annotations

from nba_betting.models.elo import predict_home_win_prob
from nba_betting.models.ensemble import ensemble_predict


class PredictionEngine:
    """Loads the model + rolling features once and predicts per game.

    ``available`` is False when no trained GBM/calibrated model exists — the
    caller should fall back to Elo-only. ``blend`` controls whether ``predict``
    returns the Elo+GBM log-odds ensemble (True, the default and the API
    behavior) or the raw calibrated GBM probability (False, the CLI's
    ``--model xgb`` mode).
    """

    def __init__(self, games, off_def_elos=None, *, blend: bool = True):
        self.games = games
        self.off_def_elos = off_def_elos or {}
        self.blend = blend
        # Context the caller fills in before predicting (late-bound, like the
        # old closures' outer-scope variables).
        self.injuries: list = []
        self.line_movements: dict = {}
        # Outputs the recommendation pipeline reads back.
        self.driver_contexts: dict = {}
        self.spread_total_predictions: dict = {}
        # Per-slate precompute caches (#3/#4), filled lazily on first predict.
        self._idx_cache: dict = {}
        # Model state.
        self.available = False
        self.model_name = "elo"
        self.rolling_df = None
        self.feature_cols: list = []
        self.feat_means: dict = {}
        self.driver_model = None
        self._actual_model = None
        self._regressors = None
        self._load()

    def _load(self) -> None:
        from nba_betting.models.calibration import load_calibrated_model
        from nba_betting.models.xgboost_model import load_model, load_feature_means

        calibrated = load_calibrated_model()
        xgb_result = load_model()  # (estimator, feature_cols) or None — call ONCE
        if not (calibrated or xgb_result):
            return  # available stays False → caller uses Elo only

        from nba_betting.features.rolling import compute_rolling_features
        from nba_betting.features.four_factors import (
            add_four_factors, add_opponent_rebound_data,
        )
        from nba_betting.features.rest_days import add_rest_features

        rolling_df = compute_rolling_features()
        if not rolling_df.empty:
            rolling_df = add_four_factors(rolling_df)
            rolling_df = add_opponent_rebound_data(rolling_df)
            rolling_df = add_rest_features(rolling_df)
            # Rolling Four Factors — vectorized groupby.transform, matching the
            # training path in builder.build_feature_matrix.
            rolling_df = rolling_df.sort_values(["team_id", "date", "game_id"])
            for col in ("efg_pct", "tov_pct", "orb_pct", "ft_rate"):
                shifted = rolling_df.groupby("team_id", sort=False)[col].shift(1)
                for w in (5, 10, 20):
                    mp = max(1, w // 2)
                    rolling_df[f"{col}_roll_{w}"] = (
                        shifted.groupby(rolling_df["team_id"], sort=False)
                        .transform(lambda s, _w=w, _m=mp: s.rolling(_w, min_periods=_m).mean())
                    )
        self.rolling_df = rolling_df

        # The calibrated wrapper is the prediction model; feature_cols come from
        # the base estimator's joblib payload. Driver attribution prefers the
        # uncalibrated base GBM (isotonic distorts LOO-delta magnitudes).
        self._actual_model = calibrated if calibrated else xgb_result[0]
        self.feature_cols = xgb_result[1] if xgb_result else []
        self.feat_means = load_feature_means() or {}
        self.driver_model = xgb_result[0] if xgb_result else self._actual_model

        from nba_betting.models.spreads_totals import load_regressors
        self._regressors = load_regressors()
        self.model_name = "ensemble"
        self.available = True

    def rolling_context(self) -> dict:
        """`{team_id: latest_rolling_row}` for explanation generation."""
        ctx: dict = {}
        if self.rolling_df is not None and not self.rolling_df.empty:
            for team_id, tdf in self.rolling_df.groupby("team_id"):
                if not tdf.empty:
                    ctx[team_id] = tdf.sort_values("date").iloc[-1].to_dict()
        return ctx

    def predict(self, home_elo, away_elo, home_id=None, away_id=None) -> float:
        from nba_betting.features.builder import (
            build_prediction_features,
            compute_latest_stats_index,
            compute_injury_impact_index,
        )
        from nba_betting.models.spreads_totals import predict_spread_total

        rolling_df = self.rolling_df
        if rolling_df is None or rolling_df.empty or not self.feature_cols:
            return predict_home_win_prob(home_elo, away_elo)

        # Lazily build the per-slate indices on first use (injuries are synced
        # by the caller before the first predict call).
        if "latest" not in self._idx_cache:
            self._idx_cache["latest"] = compute_latest_stats_index(rolling_df)
            self._idx_cache["injury"] = compute_injury_impact_index()

        extra: dict = {}
        if home_id and away_id:
            _game = next(
                (g for g in self.games
                 if g["home_team_id"] == home_id and g["away_team_id"] == away_id),
                None,
            )
            if _game:
                lm = self.line_movements.get(
                    (_game["home_team_abbr"], _game["away_team_abbr"]), {},
                )
                extra["spread_movement"] = lm.get("spread_movement", 0.0)
                extra["prob_movement"] = lm.get("prob_movement", 0.0)
                extra["odds_disagreement"] = lm.get("odds_disagreement", 0.0)
                try:
                    from nba_betting.features.player_impact import (
                        compute_player_impact_features,
                    )
                    extra.update(compute_player_impact_features(
                        home_id, away_id, self.injuries,
                        home_abbr=_game["home_team_abbr"],
                        away_abbr=_game["away_team_abbr"],
                    ))
                except Exception:
                    pass  # Non-critical; model trained with 0 as neutral value

        h_off_def = self.off_def_elos.get(home_id) if home_id else None
        a_off_def = self.off_def_elos.get(away_id) if away_id else None
        feat_row = build_prediction_features(
            home_id, away_id, rolling_df, home_elo, away_elo,
            feature_means=self.feat_means,
            extra_features=extra or None,
            home_elo_off=h_off_def[0] if h_off_def else None,
            home_elo_def=h_off_def[1] if h_off_def else None,
            away_elo_off=a_off_def[0] if a_off_def else None,
            away_elo_def=a_off_def[1] if a_off_def else None,
            latest_stats_by_team=self._idx_cache.get("latest"),
            injury_impacts=self._idx_cache.get("injury"),
        )
        if feat_row is None:
            return predict_home_win_prob(home_elo, away_elo)

        for col in self.feature_cols:
            if col not in feat_row.columns:
                feat_row[col] = self.feat_means.get(col, 0) if self.feat_means else 0
        feat_row = feat_row[self.feature_cols]

        xgb_prob = self._actual_model.predict_proba(feat_row)[0, 1]

        if home_id is not None and away_id is not None:
            self.driver_contexts[(home_id, away_id)] = feat_row
            if self._regressors is not None:
                try:
                    self.spread_total_predictions[(home_id, away_id)] = (
                        predict_spread_total(feat_row, self._regressors)
                    )
                except Exception:
                    pass

        if self.blend:
            return ensemble_predict(predict_home_win_prob(home_elo, away_elo), xgb_prob)
        return xgb_prob
