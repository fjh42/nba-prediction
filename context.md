# NBA Match Intelligence — Project Context

> Context primer for Claude Code. Read this fully before doing anything.

---

## ⚠️ WORKING AGREEMENT (read first, applies to every session)

The developer (Francisco, Cornell CS junior, targeting Data Science / AI-ML / FDE roles) is building this project **to learn and to be able to defend every design decision in an interview.** The explicit anti-goal is "vibecoding" — accepting generated code he doesn't understand. So:

1. **Do NOT generate large chunks of finished code unprompted.** Build incrementally, one concept at a time.
2. **Explain the *why* before the *how*.** Every design decision (why walk-forward, why Brier, why a feature is windowed) gets a one-line rationale he can repeat in an interview.
3. **Make him write the load-bearing pieces.** Offer the spec and intuition; let him write the core logic, then review it. Scaffolding/plumbing you can write; the parts he must defend, he writes.
4. **When he says "just do it," still narrate the decision** so he can follow.
5. **Prefer raw implementations over frameworks** in v1 (no LangChain/LlamaIndex for RAG yet) so the moving parts are visible.
6. If you catch yourself about to hand over a black box, stop and turn it into a teach-then-build step.

He already understands these three concepts (calibrate your explanations to this level — don't re-explain from zero):
- **Elo**: a self-correcting strength rating; updates move proportionally to how *surprising* a result is.
- **Walk-forward / leakage**: predict from past-only ratings, record, *then* update — in strict time order — because at prediction time you only have the past. Never a random split for time-series.
- **Probability metrics**: grade the probability (Brier, log loss, calibration), not just the side; always compare to a baseline.

---

## What this project is

A **hybrid** NBA system with two halves:
1. **Prediction** — predict game outcomes (win/loss + calibrated probability), rigorously backtested.
2. **LLM briefing (RAG)** — generate a grounded, cited explanation of each prediction, with a faithfulness eval.

It deliberately spans **Data Science** (modeling, backtesting, calibration) and **AI engineering / FDE** (RAG, grounded generation, evals). Built so the **predictor ships standalone** if time runs short; the LLM layer is phase 2 and cuttable.

---

## Current status (as of June 2026)

- ✅ `nba_elo_backtest.py` written and verified on synthetic data (Elo recovers true team ordering; beats coin-flip and base-rate Brier). This is the **Week-1 "measuring stick."**
- ✅ Have a CSV of games **2022-onward, ~3600 rows (~3 seasons)**.
- ⏭️ **Immediate next step:** run the Elo backtest on the real CSV and interpret accuracy / Brier / calibration vs the home-team baseline (~58%). That completes Week 1.
- 🔭 NBA is in offseason until October — build/backtest on past seasons now; live demo arrives during fall interview season.

---

## Architecture (the key idea: TWO parallel predictors)

```
DATA (nba_api: results, box scores, advanced stats; + news/injury text for phase 2)
      │
STORAGE (CSV now → Postgres/Supabase + pgvector once schema stabilizes)
      │
      ├──► ELO BASELINE  — eats ONLY match results (+ home edge). No stats go in here.
      │
      └──► MODEL TRACK   — feature pipeline (rolling off/def rtg, pace, rest, form,
                            injuries, + Elo as a feature) → logistic → XGBoost
      │
EVAL #1: walk-forward backtest — Model vs Elo vs home-baseline (Brier, log loss, calibration);
         also tunes "how much history" (training depth) and "form window" (decay half-life)
      │
LLM BRIEFING (phase 2): retrieve stats+text → grounded "why model favors X" briefing w/ citations
      │
EVAL #2: faithfulness — LLM-as-judge + numeric fact-check vs box scores (auto-flag hallucinated stats)
      │
SHOWCASE: Streamlit — pick a game → calibrated probabilities + grounded briefing
```

**The decision that resolves most confusion:** advanced stats and box scores do **NOT** go into Elo. Elo is the minimal, robust baseline (results only). All rich features live in the **model track**, whose entire job is to beat the Elo baseline.

---

## Data schema (canonical, one row per game)

| column | meaning |
|---|---|
| `game_id` | unique game id |
| `date` | game date (sortable; backtest depends on correct ordering) |
| `season` | season id (used for offseason rating regression) |
| `home_team`, `away_team` | team abbreviations |
| `home_pts`, `away_pts` | final scores |
| `home_win` | 1 if home won else 0 (the label) |

If the source CSV has two rows per game (raw `nba_api` LeagueGameLog), collapse with the `_to_one_row_per_game` logic in the starter script.

---

## Tech stack & conventions

- **Python 3.10+**, `pandas`, `numpy`. `nba_api` for data ([github.com/swar/nba_api](https://github.com/swar/nba_api)).
- Model: `scikit-learn` (logistic) → `xgboost`. RAG phase: an LLM API + `pgvector` (Supabase).
- App: `streamlit`.
- **Always cache API pulls** (stats.nba.com rate-limits; `time.sleep` between calls).
- **Keep pure functions pure** (Elo update, metrics) so they're unit-testable without the network. The starter script does this — preserve the pattern.
- Each meaningful change gets logged to an `eval_runs` record (config + Brier/log-loss/accuracy) once storage exists — this is the experiment-tracking discipline.

---

## Key design decisions (defensible, keep these straight)

- **Walk-forward, never a random split** — prevents future→past leakage. Non-negotiable.
- **Warmup period** — early games build ratings but aren't scored (ratings immature).
- **Stats feed the model, not Elo.**
- **Windows are tuned, not guessed** — training depth and form-window/decay half-life are hyperparameters chosen by the backtest.
- **Regime changes (big trades, e.g. a star traded) are a known limitation** — basic Elo won't see them until results arrive; recent-form features absorb them faster; an injury/roster flag can be added later. Name it honestly; don't pretend to solve it perfectly.
- **Raw Elo is mildly overconfident** (found in verification) → a calibration step is a real, reportable improvement.
- **Storage follows the schema** — CSV until the data model stabilizes, then Postgres/Supabase; pgvector unifies structured stats + text embeddings for RAG.

---

## Roadmap

- **Phase 0 (now):** run Elo backtest on the real CSV; confirm it beats baselines; read calibration.
- **Phase 1 (model):** feature pipeline (leak-safe rolling windows) → logistic → XGBoost; beat Elo on the backtest. Pull more seasons to enable the history-depth experiment. Add calibration.
- **Phase 2 (storage):** move to Postgres/Supabase; tables for games, team_game_stats, features, predictions, eval_runs.
- **Phase 3 (RAG briefing):** documents+embeddings (pgvector); retrieve stats+text; grounded generation with citations.
- **Phase 4 (eval #2):** faithfulness — LLM-as-judge + automated numeric fact-check vs box scores.
- **Phase 5 (showcase):** Streamlit app; README with architecture diagram + results/calibration table.

---

## Definition of done

1. Deployed, documented predictor with a working walk-forward backtest + calibration table, beating Elo and the home baseline.
2. (Stretch) LLM briefing layer with a faithfulness eval incl. numeric fact-check.
3. A decision log / README the developer can talk through for an hour.

**Target resume bullet:** "Built and backtested an NBA game-outcome model via a leak-free walk-forward harness; benchmarked against Elo and home-court baselines with calibrated probabilities; added a grounded LLM briefing layer with a faithfulness eval that auto-flags hallucinated stats against box-score data."
