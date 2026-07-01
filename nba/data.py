"""
NBA Match-Outcome Prediction - Week 1 Starter
=============================================
Build the "measuring stick" BEFORE any ML model. This script pulls NBA games,
computes an Elo baseline, and scores it with a leak-free walk-forward backtest.

Why this first: once this loop exists, every later question ("how much history
helps?", "does rest matter?", "is XGBoost better than Elo?") becomes ONE more
row in the results table instead of an open-ended worry. The harness is the
instrument; the model comes in Week 3.

USAGE
-----
  pip install nba_api pandas numpy
  python nba_elo_backtest.py                      # default: 7 recent seasons
  python nba_elo_backtest.py --seasons 2022-23 2023-24 2024-25
Pulled data is cached to nba_games_cache.csv, so you only hit the API once.

NOTE: nba_api is imported lazily (inside fetch_games), so every other function
in this file can be unit-tested without the package or a network connection.
"""

from __future__ import annotations
import argparse
import math
import os
import time

import numpy as np
import pandas as pd

from eval_log import log_run

CACHE = "../data/interim/nba_games_cache.csv"
ADV_CACHE = "../data/interim/nba_advanced_cache.csv"  # sibling cache; advanced stats stay OUT of the Elo games cache so the Elo path keeps eating ONLY results
HOME_ADV = 100.0   # Elo points of home-court advantage (~60% home win rate)
K = 20.0           # Elo update speed
CARRY = 0.75       # season-to-season carryover (regress 25% toward the mean)
MEAN_ELO = 1500.0

# --------------------------------------------------------------------------- #
# 1. DATA                                                                      #
# --------------------------------------------------------------------------- #
def fetch_games(seasons, pause=0.6):
    """Pull regular-season games via nba_api. Lazy import keeps the rest of the
    file testable without the package installed."""
    from nba_api.stats.endpoints import leaguegamelog
    frames = []
    for season in seasons:
        print(f"  pulling {season} ...")
        raw = leaguegamelog.LeagueGameLog(
            season=season, season_type_all_star="Regular Season"
        ).get_data_frames()[0]
        frames.append(raw)
        time.sleep(pause)  # be polite to stats.nba.com, avoid rate limiting
    return _to_one_row_per_game(pd.concat(frames, ignore_index=True))


def _to_one_row_per_game(raw):
    """nba_api returns TWO rows per game (one per team). Collapse to one row with
    home/away/result. 'vs.' in MATCHUP = home, '@' = away."""
    raw = raw.copy()
    raw["is_home"] = ~raw["MATCHUP"].str.contains("@")
    home = raw[raw["is_home"]]
    away = raw[~raw["is_home"]]
    m = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))
    out = pd.DataFrame({
        "game_id": m["GAME_ID"],
        "date": pd.to_datetime(m["GAME_DATE_home"]),
        "season": m["SEASON_ID_home"],
        "home_team": m["TEAM_ABBREVIATION_home"],
        "away_team": m["TEAM_ABBREVIATION_away"],
        "home_pts": m["PTS_home"],
        "away_pts": m["PTS_away"],
    })
    out["home_win"] = (out["home_pts"] > out["away_pts"]).astype(int)
    return out.sort_values("date").reset_index(drop=True)


def load_games(seasons):
    if os.path.exists(CACHE):
        print(f"Loading cached games from {CACHE}")
        return pd.read_csv(CACHE, parse_dates=["date"])
    df = fetch_games(seasons)
    df.to_csv(CACHE, index=False)
    print(f"Cached {len(df)} games to {CACHE}")
    return df

# --------------------------------------------------------------------------- #
# 1b. ADVANCED STATS  (off/def rating + pace; own frame, own cache)            #
# --------------------------------------------------------------------------- #
def _find_col(df, name):
    """Case-insensitively resolve `name` to a real column, else raise listing what
    IS there. Why defensive: the advanced column names (OFF_RATING/DEF_RATING/PACE)
    are INFERRED, not yet confirmed from a live pull -- so match loosely and fail
    LOUD rather than silently grab the wrong column. (On the first live run, print
    df.columns to confirm the real names.)"""
    lowered = [c.lower() for c in df.columns]
    # Guard the loose match: if two columns collide case-insensitively (e.g. PACE and
    # pace after a concat), a last-writer-wins dict would silently pick one -- which
    # defeats the whole "never grab the wrong column" purpose. Fail loud instead.
    dupes = {c for c in lowered if lowered.count(c) > 1}
    if dupes:
        raise ValueError(f"ambiguous case-insensitive columns {dupes} in {list(df.columns)}")
    lookup = {c.lower(): c for c in df.columns}
    if name.lower() not in lookup:
        raise ValueError(
            f"expected column '{name}' not found; available columns: {list(df.columns)}")
    return lookup[name.lower()]


def _tidy_advanced(raw):
    """Raw TeamGameLogs (Advanced) -> tidy LONG frame, one row per (team, game):
    game_id, date, season, team, off_rating, def_rating, pace. Pure (no nba_api, no
    network) so it is unit-testable offline, the same way _to_one_row_per_game is.

    Note vs games: TeamGameLogs is ALREADY team-level (one row per team per game),
    so there is NO home/away collapse here -- this is exactly the long shape the
    feature pipeline rolls over. Every column goes through _find_col so a renamed
    field surfaces as a clear ValueError instead of a silent wrong column."""
    raw = raw.copy()
    out = pd.DataFrame({
        # Cast to int to MATCH the games cache: nba_api GAME_IDs are zero-padded strings
        # ("0022200001"), but load_games' CSV round-trip reads game_id back as int64
        # (22200001, zeros stripped). The feature pipeline joins advanced->games on
        # game_id, so both sides must share one representation or the join silently
        # finds zero matches. int is the common ground.
        "game_id": raw[_find_col(raw, "GAME_ID")].astype("int64"),
        "date": pd.to_datetime(raw[_find_col(raw, "GAME_DATE")]),
        "season": raw[_find_col(raw, "SEASON_YEAR")],
        "team": raw[_find_col(raw, "TEAM_ABBREVIATION")],   # = team identity, mirrors games' abbreviations
        "off_rating": raw[_find_col(raw, "OFF_RATING")],
        "def_rating": raw[_find_col(raw, "DEF_RATING")],
        "pace": raw[_find_col(raw, "PACE")],
    })
    # sort by (team, date, game_id) -> already in the time order rolling features need
    return out.sort_values(["team", "date", "game_id"]).reset_index(drop=True)


def fetch_advanced(seasons, pause=0.6):
    """Pull per-game ADVANCED team stats (off/def rating, pace) via nba_api. Lazy
    import keeps the rest of the file testable without the package installed.
    OMIT team_id so ALL teams come back in ONE call per season -- that is the whole
    point: ~1 call/season instead of 30, far gentler on stats.nba.com."""
    from nba_api.stats.endpoints import teamgamelogs
    if not seasons:
        raise ValueError("seasons list is empty")  # else pd.concat([]) raises an opaque error
    frames = []
    for season in seasons:
        print(f"  pulling advanced {season} ...")
        raw = teamgamelogs.TeamGameLogs(
            season=season,
            season_type_nullable="Regular Season",
            # NOTE (verified via inspect.signature): TeamGameLogs really does reuse this
            # *_player_game_logs_* kwarg name for its MeasureType param -- it is NOT a typo,
            # do not "correct" it to *_team_game_logs_*. "Advanced" yields OFF/DEF_RATING+PACE.
            measure_type_player_game_logs_nullable="Advanced",
        ).get_data_frames()[0]
        frames.append(raw)
        time.sleep(pause)  # be polite to stats.nba.com, avoid rate limiting
    return _tidy_advanced(pd.concat(frames, ignore_index=True))


def load_advanced(seasons):
    if os.path.exists(ADV_CACHE):
        print(f"Loading cached advanced stats from {ADV_CACHE}")
        return pd.read_csv(ADV_CACHE, parse_dates=["date"])
    df = fetch_advanced(seasons)
    os.makedirs(os.path.dirname(ADV_CACHE), exist_ok=True)  # don't lose a fresh API pull to a missing dir
    df.to_csv(ADV_CACHE, index=False)
    print(f"Cached {len(df)} advanced rows to {ADV_CACHE}")
    return df

# --------------------------------------------------------------------------- #
# 2. ELO  (pure functions -- unit-testable)                                    #
# --------------------------------------------------------------------------- #
def expected_home_win(elo_home, elo_away, home_adv=HOME_ADV):
    """Pre-game probability the home team wins, from the Elo difference."""
    return 1.0 / (1.0 + 10 ** (-(elo_home + home_adv - elo_away) / 400.0))


def update_elo(elo_home, elo_away, home_win, k=K, home_adv=HOME_ADV):
    """Standard Elo update. Home gains exactly what away loses (zero-sum)."""
    exp = expected_home_win(elo_home, elo_away, home_adv)
    delta = k * (home_win - exp)
    return elo_home + delta, elo_away - delta

# --------------------------------------------------------------------------- #
# 3. WALK-FORWARD BACKTEST  (pure -- this is the anti-leakage core)            #
# --------------------------------------------------------------------------- #
def run_elo_backtest(games, warmup_games=0, k=K, home_adv=HOME_ADV, carry=CARRY):
    """Process games strictly in DATE order. For each game we predict from the
    CURRENT ratings (pre-game => zero leakage), record the prediction, and only
    THEN update the ratings with the result. That ordering is what makes this a
    valid walk-forward backtest rather than a leaky random split.

    Returns (scored_games_with_pred_column, final_ratings_dict).
    `warmup_games` drops the first N games from scoring while ratings are still
    immature (they are still used to build the ratings, just not graded)."""
    ratings, last_season = {}, None
    preds = np.full(len(games), np.nan)
    for i, row in enumerate(games.itertuples(index=False)):
        if carry is not None and last_season is not None and row.season != last_season:
            for t in ratings:                       # new season: regress to mean
                ratings[t] = carry * ratings[t] + (1 - carry) * MEAN_ELO
        last_season = row.season
        eh = ratings.get(row.home_team, MEAN_ELO)
        ea = ratings.get(row.away_team, MEAN_ELO)
        preds[i] = expected_home_win(eh, ea, home_adv)          # predict (pre-game)
        nh, na = update_elo(eh, ea, row.home_win, k, home_adv)  # then learn
        ratings[row.home_team], ratings[row.away_team] = nh, na
    out = games.copy()
    out["pred"] = preds
    return out.iloc[warmup_games:].reset_index(drop=True), ratings

# --------------------------------------------------------------------------- #
# 4. METRICS  (pure -- probability quality, not just right/wrong)              #
# --------------------------------------------------------------------------- #
def brier(p, y):
    """Mean squared error of probabilities. 0 = perfect, 0.25 = coin flip."""
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(p, y, eps=1e-15):
    """Punishes confident-and-wrong harshly."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(p, y):
    return float(np.mean((np.asarray(p, float) > 0.5).astype(int) == np.asarray(y)))


def calibration_table(p, y, bins=10):
    """Do the games you called X% actually win ~X% of the time?"""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges, rows = np.linspace(0, 1, bins + 1), []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum() == 0:
            continue
        rows.append({"bucket": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                     "pred_mean": round(float(p[m].mean()), 3),
                     "actual_winrate": round(float(y[m].mean()), 3)})
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# 5. MAIN                                                                      #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+",
                    default=["2018-19", "2019-20", "2020-21", "2021-22",
                             "2022-23", "2023-24", "2024-25"])
    ap.add_argument("--warmup", type=int, default=1230,
                    help="games to skip from scoring while ratings warm up (~1 season)")
    args = ap.parse_args()

    games = load_games(args.seasons)
    print(f"\nLoaded {len(games)} games | "
          f"{games['date'].min().date()} -> {games['date'].max().date()}")

    # ---- DATA SPIKE: the go/no-go checks you run before trusting anything ----
    print("\n=== DATA SPIKE CHECKS ===")
    print(f"  rows ................ {len(games)}")
    print(f"  duplicate game_ids .. {games['game_id'].duplicated().sum()}  (want 0)")
    print(f"  any nulls ........... {int(games.isnull().sum().sum())}  (want 0)")
    print(f"  home win rate ....... {games['home_win'].mean():.3f}  (sanity ~0.55-0.60)")

    scored, final_ratings = run_elo_backtest(games, warmup_games=args.warmup)
    p, y = scored["pred"].values, scored["home_win"].values

    print(f"\n=== ELO BACKTEST  (scored on {len(scored)} games) ===")
    print(f"  Accuracy ... {accuracy(p, y):.4f}")
    print(f"  Brier ...... {brier(p, y):.4f}   (lower better; 0.25 = coin flip)")
    print(f"  Log loss ... {log_loss(p, y):.4f}")

    print("\n=== BASELINES (what you must beat) ===")
    print(f"  'home always wins' accuracy : {accuracy(np.ones_like(y), y):.4f}")
    print(f"  constant base-rate Brier .... {brier(np.full_like(y, y.mean(), float), y):.4f}")

    print("\n=== CALIBRATION ===")
    print(calibration_table(p, y).to_string(index=False))

    print("\nTop 5 teams by final Elo (sanity check):")
    for t, r in sorted(final_ratings.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {t}: {r:.0f}")

    # Record this run in the experiment log (config + the 3 probability metrics).
    logged = log_run(
        model="elo_baseline",
        config={"K": K, "home_adv": HOME_ADV, "carry": CARRY,
                "warmup": args.warmup, "seasons": args.seasons},
        n_scored=len(scored),
        accuracy=accuracy(p, y), brier=brier(p, y), log_loss=log_loss(p, y),
        notes="Phase 0 measuring stick",
    )
    print(f"\nLogged run {logged['run_id']} to eval log.")

    print("\nGO/NO-GO: if the checks above look sane and Elo beats the baselines,"
          "\nyou have a working measuring stick. Next: add features + a model (Week 3).")


if __name__ == "__main__":
    main()
