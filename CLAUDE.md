# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working agreement

This is a **learn-first portfolio project targeting DS/ML roles**. Code ships with its reasoning intact, never as a black box. Operate accordingly:

- **Explain the *why* before the *how*.** Every design decision gets a one-line, interview-ready rationale alongside the code.
- **Prefer raw implementations over frameworks in v1** (no LangChain/LlamaIndex for RAG yet) so moving parts stay visible.
- Even when told "just do it," narrate the decision.

> `context.md` is a personal reference — do not read or edit it. CLAUDE.md is the single source of truth for Claude.

## Architecture (big picture)

A **hybrid** system with two **core** halves (both ship; the predictor is built to stand alone as a resilience property, but the RAG layer is committed, not cuttable):

1. **Prediction** — calibrated win/loss probability, rigorously backtested.
2. **LLM briefing (RAG, Phase 3)** — grounded, cited explanation per prediction, with a faithfulness eval. Core deliverable.

The **only** optional/cuttable piece in the whole project is the Phase-1.5 PyTorch DL comparison model.

The structural decision that resolves most confusion — **two parallel predictors**:

- **Elo baseline** — eats *only* match results + home edge. No box scores or advanced stats. Minimal, robust yardstick.
- **Model track** — feature pipeline (rolling off/def rtg, pace, rest, form, injuries, + Elo as a feature) → logistic → **XGBoost**. Its entire job is to *beat the Elo baseline*. All rich stats live here, never in Elo.

Everything is graded by the **walk-forward backtest** (`run_elo_backtest` in `nba/data.py`): process games in strict date order, predict from current (pre-game) ratings, record, *then* update. This ordering is what makes it leak-free — **never a random split** for this time-series. Same harness scores any model's probabilities via `brier` / `log_loss` / `accuracy` / `calibration_table`.

### Model scope (decided, see memory `dl-scope-optional`)

- **Core predictor is XGBoost (gradient-boosted trees), not a neural net** — GBMs beat deep nets on this ~3600-row tabular data.
- **PyTorch is optional Phase-1.5 only**: a neural comparison model run through the *same* harness, gated behind the shipped XGBoost core. Never put DL on the critical path or propose it as the primary model.

### Storage

CSV now → **Supabase** once schema stabilizes (`pgvector` reserved for RAG). `eval_runs` is the experiment log, read/written via the **Supabase MCP** (`.mcp.json`). Every meaningful change = one `eval_runs` row (config + Brier/log-loss/accuracy).

## Key design decisions (defensible — keep straight)

- **Walk-forward, never a random split** — prevents future→past leakage. Non-negotiable.
- **Warmup period** — early games build ratings but aren't scored (ratings still immature).
- **Stats feed the model, not Elo** — Elo stays the minimal results-only baseline.
- **Core model is gradient-boosted trees, not a neural net** — on ~3600 tabular rows GBMs reliably beat deep nets (Grinsztajn et al. 2022). PyTorch MLP is an optional comparison only.
- **Windows are tuned, not guessed** — training depth and form-window/decay half-life are hyperparameters chosen by the backtest.
- **Regime changes (e.g. star traded) are a known limitation** — basic Elo won't see them until results arrive; recent-form features absorb them faster; an injury/roster flag can be added later. Name it honestly.
- **Raw Elo is mildly overconfident** (found in verification) → a calibration step is a real, reportable improvement.
- **`eval_runs` lives in Supabase, queried via MCP** — a queryable experiment log, not just storage.

## Roadmap

- **Phase 0 (now):** run Elo backtest on the real CSV; confirm it beats baselines; read calibration.
- **Phase 1 (model):** feature pipeline (leak-safe rolling windows) → logistic → XGBoost; beat Elo. Build the **walk-forward refit loop** here (parametric models retrain on an expanding window); Phase 1.5 reuses it. Add calibration.
- **Phase 1.5 (optional, AFTER core ships):** PyTorch MLP comparison model, same refit loop + harness. Expected to tie/lose on tabular data; document *why*. Fully cuttable.
- **Phase 2 (storage):** move to Supabase (games, team_game_stats, features, predictions, eval_runs). pgvector reserved for RAG.
- **Phase 3 (RAG briefing):** documents+embeddings (pgvector); retrieve stats+text; grounded generation with citations.
- **Phase 4 (eval #2):** faithfulness — LLM-as-judge + automated numeric fact-check vs box scores.
- **Phase 5 (showcase):** Streamlit app; README with architecture diagram + results/calibration table.

## Definition of done

1. Deployed, documented predictor with a working walk-forward backtest + calibration table, beating Elo and the home baseline.
2. Grounded LLM briefing layer with a faithfulness eval incl. numeric fact-check. **Core, not stretch.**
3. A decision log / README thorough enough to talk through for an hour.
4. (Optional) PyTorch DL comparison model (Phase 1.5), scored by the same harness against XGBoost.

## Code conventions

- **Keep pure functions pure** (Elo update, metrics) so they're unit-testable with no network. `nba/data.py` establishes this — preserve it.
- **Lazy-import `nba_api`** (inside `fetch_games`) so the rest of the file is testable without the package installed.
- **Always cache API pulls** — stats.nba.com rate-limits; `time.sleep` between calls.
- Canonical schema = one row per game (`game_id, date, season, home_team, away_team, home_pts, away_pts, home_win`). `nba_api` returns two rows/game; `_to_one_row_per_game` collapses them (`vs.` = home, `@` = away).

## Commands

No `requirements.txt`/`pyproject.toml` yet — install deps ad hoc:
```
pip install nba_api pandas numpy        # core backtest
# later phases: scikit-learn xgboost streamlit torch
```

Run the Elo backtest (the Week-1 "measuring stick"). **Run from the `nba/` directory** — `data.py`'s `CACHE = "../data/interim/nba_games_cache.csv"` is relative to the working dir:
```
cd nba
python data.py                                       # default: 7 recent seasons
python data.py --seasons 2022-23 2023-24 2024-25     # specific seasons
python data.py --warmup 1230                          # games skipped from scoring while ratings warm up
```
First run hits the API and writes the cache; subsequent runs load `data/interim/nba_games_cache.csv` (no network).

Tests: `tests/` exists but is empty. The pure-function design is meant for offline unit tests (pytest-style) of Elo + metrics — no test runner is configured yet.

> Note: docstrings/plan refer to a `nba_elo_backtest.py` starter; the actual implementation lives in `nba/data.py`.
