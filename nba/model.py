"""
Logistic baseline (Phase 1.3) -- the model track's first predictor; job: beat Elo.
=================================================================================
Assembles a small, defensible feature set, runs it through the Phase 1.2 per-season
walk-forward refit loop, and scores it head-to-head against Elo on the SAME games.

FEATURES (7) -- six home-minus-away diffs + Elo's own pre-game probability
  d_form, d_ppg, d_papg, d_ortg, d_drtg, d_rest, elo_pred
WHY diffs: each raw pair (home_x, away_x) is symmetric; only the DIFFERENCE bears on
  who wins. One signed number per matchup edge -> half the dimensionality and one
  interpretable coefficient each.
WHY feed elo_pred in: Elo already compresses results-history into one calibrated
  number. Handing it to the model means the model only has to learn what the rich
  stats add BEYOND Elo -- a strictly easier job than rebuilding Elo from scratch.
  It is pre-game (run_elo_backtest, warmup_games=0), so it carries no leakage.
WHY this Pipeline: SimpleImputer(median) fills early-window NaNs (a team's first
  games); StandardScaler puts days-of-rest and points-per-100 on one scale so L2
  regularization is even-handed; LogisticRegression is the linear baseline. All three
  are fit INSIDE each per-season refit, so the imputer/scaler never see the future.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import (accuracy, brier, calibration_table, load_advanced,
                  load_games, log_loss, run_elo_backtest)
from eval_log import log_run
from features import build_features
from walkforward import walk_forward_by_season

PAIRS = ["form", "ppg", "papg", "ortg", "drtg", "rest"]
FEATURE_COLS = [f"d_{p}" for p in PAIRS] + ["elo_pred"]


def assemble_features(games, advanced):
    """games + advanced tidy frame -> (feats, X). `feats` is build_features' output
    (all game columns + the 12 home_/away_ features) plus an `elo_pred` column; `X`
    is the 7-column model matrix, row-aligned to `feats`.

    Elo comes from a warmup_games=0 backtest so EVERY game gets a pre-game pred, then
    joins by game_id (not row position) to stay order-robust."""
    feats = build_features(games, advanced)                 # 12 features, home_*/away_*
    elo_full, _ = run_elo_backtest(games, warmup_games=0)   # pre-game pred on all games
    elo = elo_full[["game_id", "pred"]].rename(columns={"pred": "elo_pred"})
    feats = feats.merge(elo, on="game_id", how="left")      # left merge preserves order

    X = pd.DataFrame({f"d_{p}": feats[f"home_{p}"] - feats[f"away_{p}"] for p in PAIRS})
    X["elo_pred"] = feats["elo_pred"].to_numpy()
    return feats, X


def make_logistic():
    """Fresh leak-safe pipeline per refit: impute -> scale -> logistic."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+",
                    default=["2022-23", "2023-24", "2024-25"])
    args = ap.parse_args()

    games = load_games(args.seasons)
    advanced = load_advanced(args.seasons)
    feats, X = assemble_features(games, advanced)

    # Pass `feats` (carries season + home_win + game_id, aligned to X) as the frame.
    scored, _ = walk_forward_by_season(feats, X, FEATURE_COLS, make_logistic)

    # Fair head-to-head: grade Elo on exactly the games the model scored (drop the
    # unscored first season from Elo too), else the two run on different samples.
    elo_full, _ = run_elo_backtest(games, warmup_games=0)
    elo_scored = elo_full[elo_full["game_id"].isin(set(scored["game_id"]))]

    p, y = scored["pred"].to_numpy(), scored["home_win"].to_numpy()
    pe, ye = elo_scored["pred"].to_numpy(), elo_scored["home_win"].to_numpy()

    print(f"\n=== LOGISTIC vs ELO  (scored on {len(scored)} games; first season is warmup) ===")
    print(f"  Logistic   acc {accuracy(p, y):.4f} | Brier {brier(p, y):.4f} | logloss {log_loss(p, y):.4f}")
    print(f"  Elo        acc {accuracy(pe, ye):.4f} | Brier {brier(pe, ye):.4f} | logloss {log_loss(pe, ye):.4f}")
    print(f"  home base  acc {accuracy(np.ones_like(y), y):.4f} | "
          f"Brier {brier(np.full_like(y, y.mean(), dtype=float), y):.4f}  (must beat)")

    print("\n=== LOGISTIC CALIBRATION ===")
    print(calibration_table(p, y).to_string(index=False))

    logged = log_run(
        model="logistic_v1",
        config={"features": FEATURE_COLS, "refit": "per_season_expanding",
                "C": 1.0, "seasons": args.seasons},
        n_scored=len(scored),
        accuracy=accuracy(p, y), brier=brier(p, y), log_loss=log_loss(p, y),
        notes="Phase 1.3 logistic on 6 diffs + elo_pred",
    )
    print(f"\nLogged run {logged['run_id']} (logistic_v1).")


if __name__ == "__main__":
    main()
