"""
Leak contract for the feature pipeline (Phase 1.1).
==================================================
A feature is leak-free iff each game's features depend ONLY on that team's
strictly-earlier games. Two independent ways that can break, one test each:

  A. FUTURE leak  -> features change when later games are added/removed.
  B. SAME-GAME leak -> features change when THIS game's outcome is scrambled.

If build_features passes both, no future result and no current result can
reach the feature row. That is exactly the walk-forward guarantee, enforced.

Run:  cd nba && conda run -n analytics python -m pytest ../tests -q
  (or: python ../tests/test_features_leak.py  for the bare-assert runner)
"""
from __future__ import annotations
import sys
import os

import numpy as np
import pandas as pd

# import the pipeline the developer writes in nba/features.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nba"))
from features import build_features  # noqa: E402

FEATURE_COLS = ["home_rest", "away_rest", "home_form", "away_form",
                "home_ppg", "away_ppg", "home_papg", "away_papg"]


def _toy_games():
    """Small deterministic schedule: 4 teams, interleaved home/away, so every
    team plays both roles and has a clear game history."""
    rows = [
        # game_id, date, season, home, away, home_pts, away_pts
        ("g1", "2024-01-01", "2023-24", "AAA", "BBB", 110, 100),
        ("g2", "2024-01-02", "2023-24", "CCC", "DDD",  95, 105),
        ("g3", "2024-01-04", "2023-24", "AAA", "CCC", 120,  90),
        ("g4", "2024-01-05", "2023-24", "BBB", "DDD", 100, 101),
        ("g5", "2024-01-08", "2023-24", "DDD", "AAA", 115, 110),
        ("g6", "2024-01-09", "2023-24", "CCC", "BBB",  99,  98),
        ("g7", "2024-01-11", "2023-24", "AAA", "DDD", 108, 109),
        ("g8", "2024-01-12", "2023-24", "BBB", "CCC", 100,  97),
    ]
    df = pd.DataFrame(rows, columns=["game_id", "date", "season",
                                     "home_team", "away_team",
                                     "home_pts", "away_pts"])
    df["date"] = pd.to_datetime(df["date"])
    df["home_win"] = (df["home_pts"] > df["away_pts"]).astype(int)
    return df


def _toy_advanced():
    """Tidy LONG advanced frame aligned to _toy_games(): one row per (team, game)
    with the SAME game_id/team keys, so a merge on [game_id, team] lands cleanly.
    Mirrors data._tidy_advanced's shape (game_id, date, season, team, off_rating,
    def_rating, pace). Ratings are hand-picked round numbers so a rolling-10 mean is
    just the average of a team's prior values -- checkable by eye in the assertions."""
    rows = [
        # game_id, team, off_rating, def_rating   (the cols build_features actually uses)
        ("g1", "AAA", 110.0, 100.0),
        ("g1", "BBB", 100.0, 110.0),
        ("g2", "CCC",  95.0, 105.0),
        ("g2", "DDD", 105.0,  95.0),
        ("g3", "AAA", 120.0,  90.0),
        ("g3", "CCC",  90.0, 120.0),
        ("g4", "BBB", 108.0, 102.0),
        ("g4", "DDD", 102.0, 108.0),
        ("g5", "DDD", 115.0, 112.0),
        ("g5", "AAA", 112.0, 115.0),
        ("g6", "CCC",  99.0,  98.0),
        ("g6", "BBB",  98.0,  99.0),
        ("g7", "AAA", 104.0, 106.0),
        ("g7", "DDD", 106.0, 104.0),
        ("g8", "BBB", 101.0,  97.0),
        ("g8", "CCC",  97.0, 101.0),
    ]
    adv = pd.DataFrame(rows, columns=["game_id", "team",
                                      "off_rating", "def_rating"])
    # carry date/season too, like the real tidy frame -- proves build_features keeps
    # only the rating cols and doesn't collide on the long frame's own date column.
    g2date = {g: d for g, d in zip(_toy_games()["game_id"], _toy_games()["date"])}
    adv["date"] = adv["game_id"].map(g2date)
    adv["season"] = "2023-24"
    adv["pace"] = 100.0
    return adv


ADV_COLS = ["home_ortg", "away_ortg", "home_drtg", "away_drtg"]


def test_no_future_leak():
    """PREFIX INVARIANCE: a game's features must be identical whether or not
    games that happen AFTER it exist in the frame. Compute on the full schedule,
    then on every prefix; the last row of each prefix must match the full run.
    If a window reaches into the future, adding later games shifts past rows."""
    full = build_features(_toy_games()).sort_values("game_id").reset_index(drop=True)
    games = _toy_games()
    for k in range(1, len(games) + 1):
        prefix = build_features(games.iloc[:k].copy())
        last = prefix.sort_values("game_id").reset_index(drop=True).iloc[k - 1]
        ref = full.iloc[k - 1]
        for c in FEATURE_COLS:
            a, b = last[c], ref[c]
            assert (pd.isna(a) and pd.isna(b)) or np.isclose(a, b), (
                f"future leak: {c} for game {ref['game_id']} changed when later "
                f"games were added ({b} -> {a})")


def test_no_same_game_leak():
    """CURRENT-ROW IMMUNITY: scrambling THIS game's own points/label must not
    change THIS game's features -- they may only use prior games. Catches a
    missing .shift(1) (the window swallowing its own row)."""
    base = build_features(_toy_games()).sort_values("game_id").reset_index(drop=True)
    scrambled_src = _toy_games()
    i = 6  # game g7, a team with history on both sides
    scrambled_src.loc[i, ["home_pts", "away_pts"]] = [999, 0]
    scrambled_src.loc[i, "home_win"] = 1
    scrambled = build_features(scrambled_src).sort_values("game_id").reset_index(drop=True)
    gid = base.iloc[i]["game_id"]
    for c in FEATURE_COLS:
        a, b = base.iloc[i][c], scrambled.iloc[i][c]
        assert (pd.isna(a) and pd.isna(b)) or np.isclose(a, b), (
            f"same-game leak: {c} for game {gid} changed when its OWN result "
            f"was scrambled ({a} -> {b})")


def test_first_game_per_team_is_nan():
    """A team's first-ever game has no past -> rolling features must be NaN.
    A non-NaN here means the window grabbed the current (or a future) game."""
    out = build_features(_toy_games())
    # g1 is the first game for AAA (home) and BBB (away)
    g1 = out[out["game_id"] == "g1"].iloc[0]
    for c in ["home_form", "away_form", "home_ppg", "away_ppg",
              "home_papg", "away_papg"]:
        assert pd.isna(g1[c]), f"{c} should be NaN on a team's first game"


def test_advanced_no_future_leak():
    """PREFIX INVARIANCE for ortg/drtg: rolling the official off/def ratings must
    obey the same future-blindness as ppg/papg. Full advanced frame is passed every
    time (left-merge ignores the unmatched future games), so any drift in an early
    game's ortg/drtg can only come from a window reaching ahead."""
    adv = _toy_advanced()
    full = (build_features(_toy_games(), advanced=adv)
            .sort_values("game_id").reset_index(drop=True))
    games = _toy_games()
    for k in range(1, len(games) + 1):
        prefix = build_features(games.iloc[:k].copy(), advanced=adv)
        last = prefix.sort_values("game_id").reset_index(drop=True).iloc[k - 1]
        ref = full.iloc[k - 1]
        for c in ADV_COLS:
            a, b = last[c], ref[c]
            assert (pd.isna(a) and pd.isna(b)) or np.isclose(a, b), (
                f"future leak: {c} for game {ref['game_id']} changed when later "
                f"games were added ({b} -> {a})")


def test_advanced_no_same_game_leak():
    """CURRENT-ROW IMMUNITY for ortg/drtg: scrambling THIS game's OWN off/def rating
    must not move its ortg/drtg feature -- those may only average prior games. Proves
    the shift(1) is in force on the advanced branch too (a missing one would let the
    window swallow the current rating)."""
    adv = _toy_advanced()
    base = (build_features(_toy_games(), advanced=adv)
            .sort_values("game_id").reset_index(drop=True))
    i = 6                      # game g7: home=AAA, away=DDD, both have prior history
    gid = base.iloc[i]["game_id"]
    scrambled_adv = adv.copy()
    mask = (scrambled_adv["game_id"] == gid)   # this game's OWN rating rows (AAA + DDD)
    scrambled_adv.loc[mask, ["off_rating", "def_rating"]] = [999.0, 0.0]
    scrambled = (build_features(_toy_games(), advanced=scrambled_adv)
                 .sort_values("game_id").reset_index(drop=True))
    for c in ADV_COLS:
        a, b = base.iloc[i][c], scrambled.iloc[i][c]
        assert (pd.isna(a) and pd.isna(b)) or np.isclose(a, b), (
            f"same-game leak: {c} for game {gid} changed when its OWN rating "
            f"was scrambled ({a} -> {b})")


def test_advanced_first_game_per_team_is_nan():
    """A team's first-ever game has no past ratings -> ortg/drtg must be NaN.
    Non-NaN here means the window grabbed the current (or a future) rating."""
    out = build_features(_toy_games(), advanced=_toy_advanced())
    g1 = out[out["game_id"] == "g1"].iloc[0]   # g1 = first game for AAA (home), BBB (away)
    for c in ADV_COLS:
        assert pd.isna(g1[c]), f"{c} should be NaN on a team's first game"


def test_back_compat_no_advanced():
    """BACK-COMPAT: with no `advanced` arg the output must be EXACTLY the original 8
    feature columns -- no ortg/drtg leaking in -- so every existing caller/test that
    pins the 8-feature schema keeps passing unchanged."""
    out = build_features(_toy_games())
    cols = set(out.columns)
    assert set(FEATURE_COLS).issubset(cols), "the original 8 features must still be present"
    for c in ADV_COLS:
        assert c not in cols, f"{c} must NOT appear when advanced is None (8-feature contract)"


def test_advanced_partial_coverage_skips_missing():
    """Games ABSENT from `advanced` must NOT cascade NaN: rolling().mean() skips the
    missing rating (skipna=True), so the feature still reflects prior known ratings.
    Drop g3 from advanced and check AAA (plays g1, g3, g5):
      - g3 home_ortg (AAA) stays 110.0 -- its own missing g3 rating is irrelevant
        anyway (shift(1) excludes the current game); only the g1 prior counts.
      - g5 away_ortg (AAA) drops 115.0 -> 110.0: full data averages g1(110)+g3(120);
        with g3 missing only g1 survives. This is the skip, not a NaN."""
    full = build_features(_toy_games(), advanced=_toy_advanced())
    partial_adv = _toy_advanced()
    partial_adv = partial_adv[partial_adv["game_id"] != "g3"]   # drop both g3 rows
    part = build_features(_toy_games(), advanced=partial_adv)

    g5_full = full[full["game_id"] == "g5"].iloc[0]
    g5_part = part[part["game_id"] == "g5"].iloc[0]
    g3_part = part[part["game_id"] == "g3"].iloc[0]
    assert np.isclose(g5_full["away_ortg"], 115.0), g5_full["away_ortg"]   # mean(110,120)
    assert np.isclose(g5_part["away_ortg"], 110.0), g5_part["away_ortg"]   # g3 skipped -> just g1
    assert np.isclose(g3_part["home_ortg"], 110.0), g3_part["home_ortg"]   # AAA's g1 prior, not NaN


if __name__ == "__main__":
    test_no_future_leak()
    test_no_same_game_leak()
    test_first_game_per_team_is_nan()
    test_advanced_no_future_leak()
    test_advanced_no_same_game_leak()
    test_advanced_first_game_per_team_is_nan()
    test_advanced_partial_coverage_skips_missing()
    test_back_compat_no_advanced()
    print("All leak tests passed.")
