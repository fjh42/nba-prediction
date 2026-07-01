"""
Walk-forward refit loop (Phase 1.2) -- the model track's backtest harness.
=========================================================================
The Elo backtest (run_elo_backtest) updates a running rating game-by-game. A
parametric model (logistic -> XGBoost) is different: it has no per-game state, it
learns WEIGHTS from a training set. So its walk-forward form is a REFIT loop --
train on the past, predict a held-out block, advance -- not an incremental update.

WHY per-season expanding window (the decision, defensibly)
----------------------------------------------------------
- The training set's only job is to estimate the feature->win relationship (the
  weights). That relationship is ~stable within the sample, so MORE training rows
  = lower-variance weights. Train on ALL strictly-prior seasons, refit at each
  season boundary.
- This does NOT make a mid-season prediction stale: recency already lives in the
  ROLLING features (form/ppg/ortg are last-10 windows recomputed per game), not in
  the training set. Shrinking the window to a recent chunk would just discard rows
  and add variance -- a hyperparameter the backtest can test later, not the default.
- The first season has no prior data -> it is UNSCORED (the model-track analog of
  Elo's warmup: games still exist, they just aren't graded).

WHY leak-free: season s is predicted ONLY by a model fit on seasons < s, so no game
ever informs a model that predicts it (or any earlier game). A FRESH estimator per
refit means no fitted state bleeds across the boundary.

WHY model-agnostic: make_estimator() returns anything with fit(X, y)/predict_proba,
so Phase 1.4 (XGBoost) and 1.5 (a PyTorch wrapper) reuse this loop untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_by_season(games, X, feature_cols, make_estimator,
                           label_col="home_win", season_col="season"):
    """Per-season expanding-window walk-forward refit.

    games : canonical frame carrying `season_col` + `label_col`, row-aligned to X.
    X     : feature frame in the SAME row order, holding `feature_cols`.
    make_estimator : zero-arg factory returning a FRESH estimator (fit/predict_proba).

    Returns (scored, preds_full):
      preds_full : `games` + a `pred` column; NaN on the unscored first season(s).
      scored     : preds_full with the unscored rows dropped -- the view you grade,
                   mirroring run_elo_backtest's (scored, ...) contract.
    """
    games = games.reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = games[label_col].to_numpy()
    preds = np.full(len(games), np.nan)

    # Season ids sort chronologically (SEASON_ID ints, or 'YYYY-YY' strings), so a
    # plain sort gives strict time order -- exactly what "train on the past" needs.
    for s in sorted(games[season_col].unique()):
        train_mask = (games[season_col] < s).to_numpy()
        test_mask = (games[season_col] == s).to_numpy()
        if not train_mask.any():
            continue  # first season: no prior data -> leave NaN (unscored)
        est = make_estimator()  # FRESH each refit -> no cross-season state leak
        est.fit(X.loc[train_mask, feature_cols], y[train_mask])
        # predict_proba[:, 1] = P(home_win=1); classes_ are sorted so col 1 is label 1
        preds[test_mask] = est.predict_proba(X.loc[test_mask, feature_cols])[:, 1]

    out = games.copy()
    out["pred"] = preds
    scored = out[~np.isnan(preds)].reset_index(drop=True)
    return scored, out
