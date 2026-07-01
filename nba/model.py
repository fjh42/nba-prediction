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
from walkforward import walk_forward_by_season, walk_forward_sliding

PAIRS = ["form", "ppg", "papg", "ortg", "drtg", "rest"]
# Two feature parametrizations over the SAME 12 underlying signals:
#  diffs -> one home-minus-away number per pair (linear-friendly, interpretable).
#  raw   -> both sides kept separate, so a tree can find asymmetries/interactions the
#           diff pre-collapses. Same build_features cols underneath; only the SHAPE
#           differs, so a diffs-vs-raw gap is about representation, not new data.
DIFF_COLS = [f"d_{p}" for p in PAIRS] + ["elo_pred"]
RAW_COLS = [f"{side}_{p}" for p in PAIRS for side in ("home", "away")] + ["elo_pred"]
FEATURE_SETS = {"diffs": DIFF_COLS, "raw": RAW_COLS}


def assemble_features(games, advanced, feature_set="diffs"):
    """games + advanced tidy frame -> (feats, X, cols). `feats` is build_features' output
    (all game columns + the 12 home_/away_ features) plus an `elo_pred` column; `X` is the
    model matrix for the chosen `feature_set`, row-aligned to `feats`; `cols` is its column
    list (for logging).

    Elo comes from a warmup_games=0 backtest so EVERY game gets a pre-game pred, then
    joins by game_id (not row position) to stay order-robust."""
    feats = build_features(games, advanced)                 # 12 features, home_*/away_*
    elo_full, _ = run_elo_backtest(games, warmup_games=0)   # pre-game pred on all games
    elo = elo_full[["game_id", "pred"]].rename(columns={"pred": "elo_pred"})
    feats = feats.merge(elo, on="game_id", how="left")      # left merge preserves order

    if feature_set == "diffs":
        X = pd.DataFrame({f"d_{p}": feats[f"home_{p}"] - feats[f"away_{p}"] for p in PAIRS})
    else:  # raw: keep both sides so the model sees each edge separately
        X = feats[[f"{side}_{p}" for p in PAIRS for side in ("home", "away")]].copy()
    X["elo_pred"] = feats["elo_pred"].to_numpy()
    return feats, X, FEATURE_SETS[feature_set]


def make_logistic():
    """Fresh leak-safe pipeline per refit: impute -> scale -> logistic."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])


def make_xgb():
    """Fresh XGBoost classifier. Lazy-import (like data.fetch_games' nba_api) so model.py
    and its logistic path stay importable without xgboost installed.

    No imputer/scaler here -- the defensible difference from the logistic pipeline: trees
    split on thresholds (scale-invariant) and XGBoost learns a default branch for missing
    values natively, so the early-window NaNs the logistic path had to fill are fed raw.
    Shallow trees (depth 3) suit ~2500 rows -- guards overfitting on a small tabular set."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        max_depth=3, n_estimators=200, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=-1, random_state=0,
    )


# model key -> (factory, logged name, hyperparams for the eval_runs config, notes)
MODELS = {
    "logistic": (make_logistic, "logistic_v1", {"C": 1.0}, "Phase 1.3 logistic"),
    "xgb": (make_xgb, "xgboost_v1",
            {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.05,
             "subsample": 0.8, "colsample_bytree": 0.8},
            "Phase 1.4 XGBoost"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+",
                    default=["2022-23", "2023-24", "2024-25"])
    ap.add_argument("--model", choices=list(MODELS), default="logistic",
                    help="which model track to run through the same refit loop")
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="diffs",
                    help="diffs (home-away) or raw (both sides) over the same 12 signals")
    ap.add_argument("--refit", choices=["season", "sliding"], default="season",
                    help="expanding per-season (all prior), or a trailing-K sliding window")
    ap.add_argument("--window", type=int, default=1230,
                    help="[sliding] trailing games each refit trains on (K)")
    ap.add_argument("--refit-every", type=int, default=100, dest="refit_every",
                    help="[sliding] games predicted per refit block")
    args = ap.parse_args()
    make_est, model_name, hp, base_note = MODELS[args.model]

    games = load_games(args.seasons)
    advanced = load_advanced(args.seasons)
    feats, X, feature_cols = assemble_features(games, advanced, args.features)

    # feats carries season + home_win + game_id, aligned to X.
    if args.refit == "season":
        scored, _ = walk_forward_by_season(feats, X, feature_cols, make_est)
        refit_desc = "per_season_expanding"
    else:
        scored, _ = walk_forward_sliding(feats, X, feature_cols, make_est,
                                         window=args.window, refit_every=args.refit_every)
        refit_desc = f"sliding_w{args.window}_r{args.refit_every}"

    # FIXED eval set: grade every run (any refit strategy, any K) on the SAME 2455
    # season-2+ games -- the Elo warmup=1230 scored set. Every sliding K has >=K prior
    # games before that set starts, so all runs predict an identical sample and the
    # K-to-K / vs-expanding deltas are pure signal, not different denominators.
    eval_ids = set(run_elo_backtest(games, warmup_games=1230)[0]["game_id"])
    scored = scored[scored["game_id"].isin(eval_ids)]

    elo_full, _ = run_elo_backtest(games, warmup_games=0)
    elo_scored = elo_full[elo_full["game_id"].isin(set(scored["game_id"]))]

    p, y = scored["pred"].to_numpy(), scored["home_win"].to_numpy()
    pe, ye = elo_scored["pred"].to_numpy(), elo_scored["home_win"].to_numpy()

    label = model_name.replace("_v1", "")
    print(f"\n=== {label.upper()} ({refit_desc}) vs ELO  (graded on {len(scored)} games) ===")
    print(f"  {label:<9}  acc {accuracy(p, y):.4f} | Brier {brier(p, y):.4f} | logloss {log_loss(p, y):.4f}")
    print(f"  {'elo':<9}  acc {accuracy(pe, ye):.4f} | Brier {brier(pe, ye):.4f} | logloss {log_loss(pe, ye):.4f}")
    print(f"  {'home base':<9}  acc {accuracy(np.ones_like(y), y):.4f} | "
          f"Brier {brier(np.full_like(y, y.mean(), dtype=float), y):.4f}  (must beat)")

    print(f"\n=== {label.upper()} CALIBRATION ===")
    print(calibration_table(p, y).to_string(index=False))

    feat_desc = "6 home-away diffs + elo_pred" if args.features == "diffs" else "raw 12 + elo_pred"
    logged = log_run(
        model=model_name,
        config={"features": feature_cols, "feature_set": args.features,
                "refit": refit_desc, **hp, "seasons": args.seasons},
        n_scored=len(scored),
        accuracy=accuracy(p, y), brier=brier(p, y), log_loss=log_loss(p, y),
        notes=f"{base_note} on {feat_desc}",
    )
    print(f"\nLogged run {logged['run_id']} ({model_name}, {args.features}, {refit_desc}).")


if __name__ == "__main__":
    main()
