"""
Walk-forward refit-loop contract (Phase 1.2).
============================================
A SPY estimator (records the game_ids it was trained on; counts fit calls) lets
these run fully OFFLINE -- no sklearn, no network -- the same discipline as the
other suites. Three guarantees, one test each:

  A. NO FUTURE LEAK   -> a season is trained ONLY on strictly-earlier seasons.
  B. FRESH REFIT      -> exactly one fit per scored season (no carried state).
  C. UNSCORED WARMUP  -> the earliest season has no prior data -> NaN / dropped.

We smuggle `game_id` in as the estimator's single "feature" so the spy can log the
exact identities it trained on -- that is what makes the leak assertion concrete.

Run:  cd nba && python ../tests/test_walkforward.py   (bare-assert runner)
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nba"))
from walkforward import walk_forward_by_season  # noqa: E402


class SpyEstimator:
    """Minimal fit/predict_proba stand-in. Appends each fit's training game_ids to a
    shared log (fit-call order) and returns a constant 0.5 so preds are checkable."""

    def __init__(self, log):
        self.log = log

    def fit(self, X, y):
        self.log.append(list(X["game_id"]))   # game_id is the smuggled-in feature col
        return self

    def predict_proba(self, X):
        p = np.full(len(X), 0.5)
        return np.column_stack([1 - p, p])    # [:,1] = 0.5, the loop reads col 1


def _toy():
    """3 seasons x 2 games; season ids sort chronologically like real SEASON_IDs."""
    games = pd.DataFrame({
        "game_id": [1, 2, 3, 4, 5, 6],
        "season":  [22022, 22022, 22023, 22023, 22024, 22024],
        "home_win": [1, 0, 1, 0, 1, 0],
    })
    X = games[["game_id"]].copy()   # the spy's lone "feature" is the id, so it can log it
    return games, X


ID_TO_SEASON = {1: 22022, 2: 22022, 3: 22023, 4: 22023, 5: 22024, 6: 22024}
SCORED_SEASONS = [22023, 22024]     # first season (22022) is the unscored warmup


def test_no_future_leak():
    """Each scored season's training ids belong to STRICTLY-EARLIER seasons only."""
    games, X = _toy()
    log = []
    walk_forward_by_season(games, X, ["game_id"], lambda: SpyEstimator(log))
    assert len(log) == len(SCORED_SEASONS), log
    for train_ids, s in zip(log, SCORED_SEASONS):
        assert all(ID_TO_SEASON[i] < s for i in train_ids), (s, train_ids)
    # concretely: 22023 trained on {1,2}; 22024 trained on {1,2,3,4} (expanding window)
    assert log[0] == [1, 2], log[0]
    assert log[1] == [1, 2, 3, 4], log[1]


def test_fresh_refit_once_per_scored_season():
    """Exactly one fit per scored season -- proves a fresh estimator each boundary."""
    games, X = _toy()
    log = []
    walk_forward_by_season(games, X, ["game_id"], lambda: SpyEstimator(log))
    assert len(log) == 2, f"expected one fit per scored season, got {len(log)}"


def test_first_season_unscored():
    """The earliest season has no prior data -> NaN in full, dropped from scored."""
    games, X = _toy()
    scored, full = walk_forward_by_season(games, X, ["game_id"], lambda: SpyEstimator([]))
    assert full.loc[full["season"] == 22022, "pred"].isna().all(), "warmup must be NaN"
    assert full.loc[full["season"] > 22022, "pred"].notna().all(), "scored must be filled"
    assert set(scored["game_id"]) == {3, 4, 5, 6}, scored["game_id"].tolist()
    assert np.allclose(scored["pred"].to_numpy(), 0.5), scored["pred"].tolist()


if __name__ == "__main__":
    test_no_future_leak()
    test_fresh_refit_once_per_scored_season()
    test_first_season_unscored()
    print("All walk-forward tests passed.")
