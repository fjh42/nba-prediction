"""
Experiment log (local CSV stage).
=================================
One row per meaningful run: reproducible config + the three probability
metrics. Why a log at all: it turns "did change X help?" into a table lookup
instead of memory. This is the SAME shape that moves to Supabase `eval_runs`
in Phase 2 (read/written via the Supabase MCP) -- start the discipline now.

Kept dependency-light and pure-ish (one file append, no network) so it stays
testable and trivially portable to a DB insert later.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

import pandas as pd

EVAL_LOG = "../data/eval_runs.csv"

COLUMNS = ["run_id", "model", "config", "n_scored",
           "accuracy", "brier", "log_loss", "notes"]


def log_run(model, config, n_scored, accuracy, brier, log_loss, notes="",
            path=EVAL_LOG):
    """Append one experiment row. `config` is any dict (params that define the
    run); it is JSON-serialized so the row is self-describing and reproducible.
    Returns the appended row as a dict."""
    row = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "model": model,
        "config": json.dumps(config, sort_keys=True),
        "n_scored": int(n_scored),
        "accuracy": round(float(accuracy), 4),
        "brier": round(float(brier), 4),
        "log_loss": round(float(log_loss), 4),
        "notes": notes,
    }
    header = not os.path.exists(path)
    pd.DataFrame([row], columns=COLUMNS).to_csv(
        path, mode="a", header=header, index=False)
    return row
