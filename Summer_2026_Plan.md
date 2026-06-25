# Francisco — Summer 2026 Plan

**Window:** Jun 22 – Aug 1, 2026 (6 internship weeks) · **Effort:** 10–15 hrs/week on top of the internship · **Targets:** AI/ML + FDE jobs *and* Cornell ML/ORIE masters (keep both open) · **Grad:** Summer 2027

---

## The decision (final)

One flagship project, built deliberately — you make every architecture call; AI is the pair you argue with, not autocomplete. Skip certificates, n8n, PowerBI, and extra courses this summer; they're motion, not progress. Stop collecting project ideas — this is the one.

**Flagship: NBA Match Intelligence (Hybrid)**
A system that (1) predicts NBA game outcomes with a properly backtested ML model, and (2) generates a grounded, LLM-written briefing explaining each prediction with citations. It deliberately spans both skill sets you're keeping open:
- **Data Science** — feature engineering, modeling, backtesting, calibration.
- **AI engineering / FDE** — RAG, grounded LLM generation, and a faithfulness eval.

Why hybrid: classic ML alone signals "DS" but you can already show that (bio XGBoost). The LLM layer is the portfolio gap and the FDE-relevant signal. Built in order so the **predictor ships standalone** if the summer gets tight — the LLM layer is phase 2 and cuttable.

**Why NBA over the World Cup:** far more data (thousands of games/season → better, more stable models), a cleaner binary target (no draws), a richer text corpus for the briefing layer, and — critically — it's **live during your fall/winter interview season** (NBA tips off in October), so you can demo on tonight's games when it actually counts in front of a recruiter. Bonus: a real NBA project **replaces the fabricated NBA API line** on your resume with something true.

---

## Two reframes that unblock the project

**Offseason now is fine — you're building the engine this summer, demoing it in the fall.** The NBA is in its offseason until October, so all summer you backtest on past seasons (no live games needed to build or prove the system). The live demo arrives exactly when you're interviewing. That's better timing than a tournament that ends mid-summer.

**"How much history matters" is not a guess — it's an experiment.** Don't pick a cutoff by gut. Use time-decay weighting (recent games weighted more, via an exponential half-life you *tune*), and let the backtest tell you which window predicts held-out games best. Your uncertainty IS the research question, and the harness is the instrument that answers it. (The starter script already gives you that instrument — see Week 1.)

---

## Two parallel tracks

**Track A — Internship (turn dead time into resume lines).** Don't wait to be assigned work. Each rotation, ask for one real, messy problem and ship something small they actually use. Goal by Aug 1: one quantified bullet per rotation.

**Track B — Flagship project (10–15 hrs/week).** Milestones below. Each weekend, write a 3–5 sentence "what I decided and why" log — this becomes your interview talking points and the project README.

---

## Architecture (each stage = a decision you can defend)

**Data** — `nba_api` (the Python wrapper for official stats.nba.com): game logs, box scores, advanced stats (offensive/defensive rating, pace). Thousands of games per season across many seasons. The starter script handles the pull + caching.

**Features** — Elo rating, rolling offensive/defensive rating, pace, recent form (last-10), rest days / back-to-backs, home/away, injuries; time-decay weighting so all history contributes but recency dominates.

**Model** — Climb the ladder: Elo baseline → logistic regression → XGBoost. Binary outcome (win/loss), so it's a clean classification problem. The baseline isn't a throwaway; *beating it* is your headline result.

**Eval harness #1 — the backtest (model).** Walk-forward (train on the past, predict the future, never a random split), leak-free. Score with Brier + log loss (probability quality), calibration (do your 70% calls win ~70%?), and accuracy vs. the "home team wins" baseline (~58–60%) and Elo. Also runs the "how much history" experiment. *Note from the verification run: raw Elo is mildly overconfident, so adding a calibration step is a real, defensible improvement to report.* A natural extension is a **live win-probability** model from the play-by-play log.

**LLM briefing layer (phase 2).** RAG over game recaps, injury reports, and team news → LLM generates a grounded "why the model favors Boston" briefing with citations. NBA has a rich text corpus (daily games, heavy recap/injury coverage), so retrieval has plenty to work with.

**Eval harness #2 — faithfulness (text).** Is every claim in the briefing supported by retrieved context? LLM-as-judge + citation grounding, validated against your own labels on a sample. **NBA superpower:** because you *have* the structured box-score stats, you can programmatically catch hallucinated numbers (briefing says "30 PPG", data says 24 → auto-flag). Two eval harnesses = double rigor and the thing almost nobody builds.

**Showcase** — Streamlit app: pick a game → calibrated win probabilities + the grounded briefing. Clean README with architecture diagram and a results/calibration table.

---

## Week-by-week (mapped to your rotations)

### Weeks 1–2 · Jun 22 – Jul 3 · Rotation: Frontend / App Dev
- **Internship:** Learn the stack, be useful, ask the lead: "what's a small thing nobody's had time to fix?" Ship one fix or component.
- **Project — run the measuring stick (already written for you):** `pip install nba_api pandas numpy`, then `python nba_elo_backtest.py`. It pulls the seasons, builds Elo, runs the walk-forward backtest, and prints accuracy / Brier / calibration + the data-spike go/no-go checks. Your only job week 1: get it running, read the numbers, confirm Elo beats the baselines. *No ML model yet.*

### Weeks 3–4 · Jul 6 – Jul 17 · Rotation: Data & Analytics ⭐
- **Internship (your highest-value rotation):** Ask for a dataset nobody's analyzed. Produce one analysis or dashboard. Get a name + a result you can quote.
- **Project — the real model (this is the shippable standalone core):** Engineer features (off/def rating, pace, rest, B2B, form), climb from logistic regression to XGBoost, measuring each change against the backtest instead of eyeballing. Run the "how much history / which decay half-life" experiment and the calibration step. Write up what you found.

### Weeks 5–6 · Jul 20 – Aug 1 · Rotation: DevSecOps
- **Internship:** Learn the CI/CD + security side (directly relevant to FDE — you'll deploy into client environments). Ship one small pipeline or security improvement if you can.
- **Project — phase 2 + ship:** Build the LLM briefing layer (RAG + grounded generation) and the faithfulness eval (including the automated numeric fact-check). Deploy the Streamlit showcase. Write the README + one short public post about the dual eval harnesses. *If time is short, cut the LLM layer — you still have a complete, backtested predictor.*

---

## Interview prep (woven in, ~1 hr/week)

Generate prep from the project, don't cram it. Each milestone, answer out loud: "Walk me through the architecture." / "Why walk-forward, not a random split?" / "Why Brier over accuracy?" / "How do you know the briefing isn't hallucinating?" Your decision log = the script; your two eval harnesses = the answers. Plus 2–3 LeetCode-style problems/week and a refresher on ML fundamentals (bias-variance, embeddings, evaluation metrics, calibration).

---

## Action item — fix the resume before sending it anywhere

The **NBA Player Comparison API** line on the DataScience resume describes a project that doesn't currently work and that you can't defend in an interview — that's a liability, not an asset. Before the resume goes out: **cut it**, or **replace it** with this flagship once the predictor core is shipped. A shorter all-true resume beats a longer one with a landmine.

---

## What to say NO to this summer
- ❌ Certificates (Google AI Essentials, etc.) — low signal for these roles.
- ❌ Learning n8n / PowerBI "just in case" — learn tools on demand.
- ❌ Collecting more project ideas / starting a second project — finish this one.
- ✅ Optional, *only if* the flagship ships early: text-to-SQL with destructive-query guardrails (strong FDE project #2); or a shot-quality (xG-style) model folded in as an enhancement.

---

## Resources
- Evals: [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) and [LLM Evals: Everything You Need to Know](https://hamel.dev/blog/posts/evals-faq/) — Hamel Husain.
- Agents/architecture: [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) and [Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — Anthropic.
- LLM/RAG roadmap: [mlabonne/llm-course](https://github.com/mlabonne/llm-course). Build v1 without a heavy framework so you actually learn the parts.

---

## Definition of done by Aug 1
1. A deployed, documented predictor with a working **walk-forward backtest** + calibration table, benchmarked against Elo and the home baseline.
2. (Stretch) The LLM briefing layer with a **faithfulness eval** (incl. automated numeric fact-check).
3. A decision log / README you can talk through for an hour.
4. One quantified resume bullet from each internship rotation.
5. The NBA line cut or replaced on the DataScience resume.

**Resume bullet this produces (draft):** "Built and backtested an NBA game-outcome model over N seasons via a leak-free walk-forward harness; benchmarked against Elo and home-court baselines with calibrated probabilities (Brier [X]); added a grounded LLM briefing layer with a faithfulness eval that auto-flags hallucinated stats against box-score data."
