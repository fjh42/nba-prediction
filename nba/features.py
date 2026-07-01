"""
Feature pipeline (Phase 1.1) -- the model track's input layer.
=============================================================
Turns the canonical one-row-per-game frame into leak-safe, past-only features
for the model track (logistic -> XGBoost). Elo stays feature-free; ALL rich
signals live here, and the whole track's job is to beat the Elo baseline.

THE LEAK GUARANTEE (the only thing that makes these features valid)
------------------------------------------------------------------
Every feature for game i must depend ONLY on each team's strictly-earlier
games. Two ways that can break, both closed here:
  * FUTURE leak  -> sorting by date before rolling: a window can't reach ahead.
  * SAME-GAME leak -> .shift(1) before .rolling(): the window can't see row i.
Enforced by tests/test_features_leak.py (prefix-invariance + current-row
immunity + first-game-NaN). If those pass, no future or current result can
reach a feature row -- the exact walk-forward guarantee, mechanically checked.

WHY LONG FORM
-------------
A team's history is split across home_* and away_* columns, so you cannot roll
over it in the wide frame. We melt to one row per (team, game) -- every team's
games land in a single time-ordered column -- compute the rolling stats there,
then join the per-team features back onto the wide game row as home_*/away_*.

v1 FEATURES (defensible, and all derivable from the cache's columns)
  rest  : days since that team's previous game (fatigue / schedule edge)
  form  : rolling win rate, last 10 (recent strength; absorbs regime changes
          faster than Elo -- e.g. a star trade -- which is why it complements Elo)
  ppg   : rolling points scored / game, last 10 (offense proxy)
  papg  : rolling points allowed / game, last 10 (defense proxy)

ADVANCED FEATURES (optional `advanced` arg; the tidy long frame from
data.fetch_advanced / _tidy_advanced)
  ortg  : rolling mean, last 10, of that team's official per-game OFF_RATING
  drtg  : rolling mean, last 10, of that team's official per-game DEF_RATING
Official off/def ratings are points per 100 POSSESSIONS, so they are
pace-adjusted -- the real efficiency signal that ppg/papg only approximate. We
KEEP ppg/papg alongside ortg/drtg, not instead of: raw scoring and per-100
efficiency diverge for fast/slow teams, so both carry orthogonal information and
the model can weigh them. Ratings are pulled once (data.fetch_advanced) and
rolled HERE through the same shift(1)->rolling helper, so the leak guarantee is
identical to the other features. When `advanced` is None, output is unchanged.
"""
from __future__ import annotations

import pandas as pd

WINDOW = 10  # form/scoring lookback; a hyperparameter the backtest will tune


def _past_mean(grouped):
    """Mean of the last WINDOW games STRICTLY BEFORE the current one.
    shift(1) drops the current game (no same-game leak); min_periods=1 keeps
    partial early windows so only a team's first-ever game is NaN."""
    return grouped.transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())


def build_features(games: pd.DataFrame,
                   advanced: pd.DataFrame | None = None) -> pd.DataFrame:
    """Canonical one-row-per-game frame -> same rows + past-only feature columns.

    advanced=None (default) -> the original 8 columns: {home,away}_{rest,form,
    ppg,papg}. Kept byte-for-byte identical so existing callers/tests are unaffected.
    advanced=<tidy long frame from data._tidy_advanced> -> 4 more columns
    ({home,away}_{ortg,drtg}) = 12 total, rolled through the SAME leak-safe helper."""
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])

    # --- melt to one row per (team, game); pts_against is the opponent's pts ---
    home = pd.DataFrame({
        "game_id": games["game_id"], "date": games["date"], "side": "home",
        "team": games["home_team"],
        "pts_for": games["home_pts"], "pts_against": games["away_pts"],
        "win": games["home_win"],
    })
    away = pd.DataFrame({
        "game_id": games["game_id"], "date": games["date"], "side": "away",
        "team": games["away_team"],
        "pts_for": games["away_pts"], "pts_against": games["home_pts"],
        "win": 1 - games["home_win"],
    })
    # sort by date so "rolling" means "in time order"; game_id breaks same-day ties
    long = (pd.concat([home, away], ignore_index=True)
            .sort_values(["team", "date", "game_id"])
            .reset_index(drop=True))

    g = long.groupby("team", sort=False)
    long["rest"] = g["date"].diff().dt.days          # gap to prior game (no leak: dates only)
    long["form"] = _past_mean(g["win"])
    long["ppg"] = _past_mean(g["pts_for"])
    long["papg"] = _past_mean(g["pts_against"])

    feat = ["rest", "form", "ppg", "papg"]

    # --- optional: roll official off/def ratings, exactly like ppg/papg above ---
    if advanced is not None:
        # Take ONLY the rating columns: a wide merge would collide on date/season
        # and silently fork the long frame's own date column the rolling depends on.
        # drop_duplicates is a guard, not cosmetics: a duplicate (game_id, team) in
        # `advanced` would fan a left-merge into 2 rows per game -> phantom history
        # that corrupts the rolling window. One rating per (team, game) is the contract.
        adv = advanced[["game_id", "team", "off_rating", "def_rating"]]
        # Fail loud on a real duplicate (e.g. two fetches concatenated) instead of
        # letting drop_duplicates silently keep one -- the caller should fix upstream.
        dups = adv.duplicated(["game_id", "team"]).sum()
        if dups:
            raise ValueError(f"{dups} duplicate (game_id, team) rows in advanced; "
                             "did you concat two fetches? expected one rating per team-game")
        # Coerce the join key to the long frame's dtype: load_games reads game_id back
        # as int64 and _tidy_advanced matches that, but a string-keyed `games` would make
        # merge() HARD-crash on mixed str/int64 -- coerce so the contract is robust.
        if adv["game_id"].dtype != long["game_id"].dtype:
            adv = adv.assign(game_id=adv["game_id"].astype(long["game_id"].dtype))
        # Left merge keeps EVERY long row even if advanced misses some games. A missing
        # game contributes NaN to the shifted series; rolling().mean()'s default
        # skipna=True drops it, so ortg/drtg still reflects whatever prior ratings exist
        # (no NaN cascade). Only a team's first-ever game -- no prior rows at all -- is
        # truly NaN. Re-sort after merge so rolling still runs in (team, date, game_id)
        # time order -- merge may not preserve it, and an out-of-order window would leak.
        long = (long.merge(adv, on=["game_id", "team"], how="left")
                .sort_values(["team", "date", "game_id"])
                .reset_index(drop=True))
        g = long.groupby("team", sort=False)  # regroup: the frame above is a new object
        long["ortg"] = _past_mean(g["off_rating"])   # shift(1)->rolling: identical leak guard
        long["drtg"] = _past_mean(g["def_rating"])
        feat = feat + ["ortg", "drtg"]

    # --- join per-team features back onto the wide game row ---
    home_feat = (long[long["side"] == "home"][["game_id"] + feat]
                 .rename(columns={c: f"home_{c}" for c in feat}))
    away_feat = (long[long["side"] == "away"][["game_id"] + feat]
                 .rename(columns={c: f"away_{c}" for c in feat}))
    out = (games
           .merge(home_feat, on="game_id", how="left")
           .merge(away_feat, on="game_id", how="left"))
    return out
