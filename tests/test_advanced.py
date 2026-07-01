"""
Tidy contract for the advanced-stats ingestion (data.py).
========================================================
fetch_advanced does the network pull; the RESHAPE lives in the pure helper
_tidy_advanced (mirroring _to_one_row_per_game). We test only that pure helper,
so these run fully OFFLINE -- no nba_api, no stats.nba.com. This is the same
discipline the file is built around: keep the reshape pure so it is unit-testable
without the package installed.

What we pin down:
  1. happy path  -> raw TeamGameLogs(Advanced) cols become the tidy long schema,
                    lowercased, dates parsed, sorted by (team, date, game_id).
  2. case-insensitivity -> _find_col resolves real columns regardless of casing
                    (the advanced names are INFERRED, so matching must be loose).
  3. fail loud   -> a missing advanced column raises ValueError listing what IS
                    there, never a silent wrong-column grab.

Run:  cd nba && conda run -n analytics python -m pytest ../tests -q
  (or: python ../tests/test_advanced.py  for the bare-assert runner)
"""
from __future__ import annotations
import sys
import os

import numpy as np
import pandas as pd

# import the helpers from nba/data.py (pure; no nba_api touched at import time)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nba"))
from data import _tidy_advanced, _find_col  # noqa: E402

TIDY_COLS = ["game_id", "date", "season", "team",
             "off_rating", "def_rating", "pace"]


def _raw_advanced(cols=None):
    """Tiny synthetic frame mimicking TeamGameLogs(measure_type='Advanced'):
    one row per (team, game), two teams x two games, intentionally out of order
    so the sort is exercised. `cols` lets a test rename/drop columns."""
    rows = [
        # GAME_ID, GAME_DATE, SEASON_YEAR, TEAM_ABBREVIATION, OFF_RATING, DEF_RATING, PACE
        ("0022300002", "2024-01-03", "2023-24", "BOS", 118.1, 108.4, 99.2),
        ("0022300001", "2024-01-01", "2023-24", "BOS", 115.0, 110.0, 98.0),
        ("0022300001", "2024-01-01", "2023-24", "NYK", 110.0, 115.0, 98.0),
        ("0022300002", "2024-01-03", "2023-24", "NYK", 109.5, 117.3, 99.2),
    ]
    df = pd.DataFrame(rows, columns=["GAME_ID", "GAME_DATE", "SEASON_YEAR",
                                     "TEAM_ABBREVIATION", "OFF_RATING",
                                     "DEF_RATING", "PACE"])
    if cols is not None:
        df = df.rename(columns=cols)
    return df


def test_tidy_schema_and_values():
    """Raw advanced frame -> exact tidy long schema, lowercased, dates parsed."""
    out = _tidy_advanced(_raw_advanced())
    assert list(out.columns) == TIDY_COLS, out.columns.tolist()
    assert pd.api.types.is_datetime64_any_dtype(out["date"]), "date must be parsed"
    # game_id is cast to int to match the games cache's in-memory dtype (the join key)
    assert pd.api.types.is_integer_dtype(out["game_id"]), out["game_id"].dtype
    # BOS game 1 row carried its values through untouched
    bos1 = out[(out["team"] == "BOS") & (out["game_id"] == 22300001)].iloc[0]
    assert np.isclose(bos1["off_rating"], 115.0)
    assert np.isclose(bos1["def_rating"], 110.0)
    assert np.isclose(bos1["pace"], 98.0)
    assert bos1["season"] == "2023-24"


def test_sorted_by_team_date_gameid():
    """Output is in (team, date, game_id) order -- the time order rolling
    features depend on, regardless of raw row order."""
    out = _tidy_advanced(_raw_advanced())
    # Explicit expected order (NOT a re-sort of `out` -- that would pass even if the
    # helper never sorted). Fixture is fed out-of-order; correct (team,date,game_id)
    # order is: BOS 01-01, BOS 01-03, NYK 01-01, NYK 01-03.
    assert out["team"].tolist() == ["BOS", "BOS", "NYK", "NYK"], out["team"].tolist()
    assert out["game_id"].tolist() == [22300001, 22300002, 22300001, 22300002], \
        out["game_id"].tolist()


def test_case_insensitive_columns():
    """_find_col / _tidy_advanced resolve columns regardless of casing -- the
    advanced names are inferred, so the match must be loose."""
    raw = _raw_advanced(cols={
        "OFF_RATING": "off_rating", "DEF_RATING": "Def_Rating", "PACE": "pace",
        "GAME_ID": "Game_Id",
    })
    out = _tidy_advanced(raw)
    assert list(out.columns) == TIDY_COLS
    assert len(out) == 4
    # _find_col directly, too
    assert _find_col(raw, "PACE") == "pace"


def test_missing_column_raises():
    """Dropping an advanced column must raise ValueError that LISTS what IS there
    (fail loud, never silently grab the wrong column)."""
    raw = _raw_advanced().drop(columns=["PACE"])
    try:
        _tidy_advanced(raw)
    except ValueError as e:
        msg = str(e)
        assert "PACE" in msg, msg
        assert "available columns" in msg, msg
    else:
        raise AssertionError("expected ValueError when PACE column is missing")


def test_cache_roundtrip_preserves_join_key():
    """The riskiest real path is tidy -> to_csv -> read_csv (what load_advanced does).
    Assert game_id survives as an integer matching the games-cache representation and
    date comes back as datetime -- otherwise the feature-pipeline join silently breaks."""
    import tempfile
    out = _tidy_advanced(_raw_advanced())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "adv.csv")
        out.to_csv(path, index=False)                       # exactly as load_advanced writes
        back = pd.read_csv(path, parse_dates=["date"])      # exactly as load_advanced reads
    assert pd.api.types.is_integer_dtype(back["game_id"]), back["game_id"].dtype
    assert back["game_id"].tolist() == [22300001, 22300002, 22300001, 22300002]
    assert pd.api.types.is_datetime64_any_dtype(back["date"]), "date lost its dtype on reload"


if __name__ == "__main__":
    test_tidy_schema_and_values()
    test_sorted_by_team_date_gameid()
    test_case_insensitive_columns()
    test_missing_column_raises()
    test_cache_roundtrip_preserves_join_key()
    print("OK")
