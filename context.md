# NBA Match Intelligence — Project Context

> Context primer for Claude Code. Read this fully before doing anything.

\---

## WORKING AGREEMENT (read first, applies to every session)

The developer (Francisco, Cornell CS junior, targeting Data Science / AI-ML / FDE roles) is building this project **to learn and to be able to defend every design decision in an interview.** The explicit anti-goal is "vibecoding" — shipping code he can't defend. He learns through **explanation, not authorship**: Claude writes the code; the reasoning behind each decision is what must land. So:

1. **Explain the *why* before the <i>how</i>.** Every design decision (why walk-forward, why Brier, why a feature is windowed) gets a one-line rationale he can repeat in an interview.
2. **When he says "just do it," still narrate the decision** so he can follow.
3. **Prefer raw implementations over frameworks** in v1 (no LangChain/LlamaIndex for RAG yet) so the moving parts are visible.
4. If you catch yourself about to hand over a black box, stop and explain the reasoning behind it before moving on.

He already understands these three concepts (calibrate your explanations to this level — don't re-explain from zero):

* **Elo**: a self-correcting strength rating; updates move proportionally to how *surprising* a result is.
* **Walk-forward / leakage**: predict from past-only ratings, record, *then* update — in strict time order — because at prediction time you only have the past. Never a random split for time-series.
* **Probability metrics**: grade the probability (Brier, log loss, calibration), not just the side; always compare to a baseline.

\---

## What this project is

A **hybrid** NBA system with two halves:

1. **Prediction** — predict game outcomes (win/loss + calibrated probability), rigorously backtested.
2. **LLM briefing (RAG)** — generate a grounded, cited explanation of each prediction, with a faithfulness eval.

It deliberately spans **Data Science** (modeling, backtesting, calibration) and **AI engineering / FDE** (RAG, grounded generation, evals). **Both halves are core deliverables and both ship** — the predictor is built to stand alone as a resilience property, but the RAG briefing layer is committed, not cuttable. The **only** optional extra in the project is the PyTorch neural comparison model (Phase 1.5) — added only after the core XGBoost predictor ships, never on the critical path.

\---

## Current status (as of June 2026)

* `nba\_elo\_backtest.py` written and verified on synthetic data (Elo recovers true team ordering; beats coin-flip and base-rate Brier). This is the **Week-1 "measuring stick."**
* Have a CSV of games **2022-onward, \~3600 rows (\~3 seasons)**.
* **Immediate next step:** run the Elo backtest on the real CSV and interpret accuracy / Brier / calibration vs the home-team baseline (\~58%). That completes Week 1.
* NBA is in offseason until October — build/backtest on past seasons now; live demo arrives during fall interview season.

\---

## Architecture (the key idea: TWO parallel predictors)

```
DATA (nba\_api: results, box scores, advanced stats; + news/injury text for phase 2)
      │
STORAGE (CSV now → Supabase + pgvector once schema stabilizes)
      │
      ├──► ELO BASELINE  — eats ONLY match results (+ home edge). No stats go in here.
      │
      └──► MODEL TRACK   — feature pipeline (rolling off/def rtg, pace, rest, form,
                            injuries, + Elo as a feature) → logistic → XGBoost  (CORE)
                            └─ (optional, Phase 1.5) PyTorch MLP — comparison model,
                               SAME harness; expected to tie/lose XGBoost (tabular)
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

\---

## Data schema (canonical, one row per game)

|column|meaning|
|-|-|
|`game\_id`|unique game id|
|`date`|game date (sortable; backtest depends on correct ordering)|
|`season`|season id (used for offseason rating regression)|
|`home\_team`, `away\_team`|team abbreviations|
|`home\_pts`, `away\_pts`|final scores|
|`home\_win`|1 if home won else 0 (the label)|

If the source CSV has two rows per game (raw `nba\_api` LeagueGameLog), collapse with the `\_to\_one\_row\_per\_game` logic in the starter script.

\---

## Tech stack \& conventions

* **Python 3.10+**, `pandas`, `numpy`. `nba\_api` for data ([github.com/swar/nba\_api](https://github.com/swar/nba_api)).
* Model: `scikit-learn` (logistic) → `xgboost` (the core predictor). RAG phase: an LLM API + `pgvector` (Supabase).
* `pytorch` — **optional only** (Phase 1.5): a neural comparison model scored by the same backtest harness. *Not* the core predictor — gradient-boosted trees beat neural nets on small tabular data, so the NN exists to be benchmarked, not to ship.
* **Supabase MCP** (`.mcp.json`) is the channel for reading/writing `eval\_runs` — the experiment log lives in Supabase, queried directly via MCP.
* App: `streamlit`.
* **Always cache API pulls** (stats.nba.com rate-limits; `time.sleep` between calls).
* **Keep pure functions pure** (Elo update, metrics) so they're unit-testable without the network. The starter script does this — preserve the pattern.
* Each meaningful change gets logged to an `eval\_runs` record (config + Brier/log-loss/accuracy) once storage exists — this is the experiment-tracking discipline.

\---

## Key design decisions (defensible, keep these straight)

* **Walk-forward, never a random split** — prevents future→past leakage. Non-negotiable.
* **Warmup period** — early games build ratings but aren't scored (ratings immature).
* **Stats feed the model, not Elo.**
* **Core model is gradient-boosted trees, not a neural net** — on ~3600 tabular rows, GBMs reliably outperform deep nets (Grinsztajn et al. 2022). A PyTorch MLP is added *only* as an optional comparison so the backtest can show it ties/loses XGBoost — defensible to demo, indefensible as the primary model.
* **`eval\_runs` lives in Supabase, queried via MCP** — not just storage; it's the queryable experiment log that enforces the experiment-tracking discipline.
* **Windows are tuned, not guessed** — training depth and form-window/decay half-life are hyperparameters chosen by the backtest.
* **Regime changes (big trades, e.g. a star traded) are a known limitation** — basic Elo won't see them until results arrive; recent-form features absorb them faster; an injury/roster flag can be added later. Name it honestly; don't pretend to solve it perfectly.
* **Raw Elo is mildly overconfident** (found in verification) → a calibration step is a real, reportable improvement.
* **Storage follows the schema** — CSV until the data model stabilizes, then Supabase; pgvector unifies structured stats + text embeddings for RAG.

\---

## Roadmap

* **Phase 0 (now):** run Elo backtest on the real CSV; confirm it beats baselines; read calibration.
* **Phase 1 (model):** feature pipeline (leak-safe rolling windows) → logistic → XGBoost; beat Elo on the backtest. Pull more seasons to enable the history-depth experiment. Add calibration. Build the **walk-forward refit loop** here (parametric models retrain on an expanding window) — Phase 1.5 reuses it.
* **Phase 1.5 (optional, AFTER core ships):** neural comparison model — a PyTorch MLP trained in the same walk-forward refit loop, scored against XGBoost/Elo/home-baseline by the same harness. Expected outcome: ties or loses on tabular data; document *why*. ~6–10 hrs. Fully cuttable.
* **Phase 2 (storage):** move to Supabase; tables for games, team\_game\_stats, features, predictions, eval\_runs. `eval\_runs` is the experiment log, read/written via the Supabase MCP. pgvector reserved for the RAG phase.
* **Phase 3 (RAG briefing):** documents+embeddings (pgvector); retrieve stats+text; grounded generation with citations.
* **Phase 4 (eval #2):** faithfulness — LLM-as-judge + automated numeric fact-check vs box scores.
* **Phase 5 (showcase):** Streamlit app; README with architecture diagram + results/calibration table.

\---

## Definition of done

1. Deployed, documented predictor with a working walk-forward backtest + calibration table, beating Elo and the home baseline.
2. Grounded LLM briefing layer with a faithfulness eval incl. numeric fact-check. **Core, not stretch.**
3. A decision log / README the developer can talk through for an hour.
4. (Stretch / optional) PyTorch DL comparison model (Phase 1.5), scored by the same harness against XGBoost.

**Target resume bullet:** "Built and backtested an NBA game-outcome model via a leak-free walk-forward harness; benchmarked against Elo and home-court baselines with calibrated probabilities; added a grounded LLM briefing layer with a faithfulness eval that auto-flags hallucinated stats against box-score data."

