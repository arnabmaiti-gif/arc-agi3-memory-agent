# Self-Improving Memory Agent for ARC-AGI-3

**HUD × YC "Frontier RSI/RL Environments" Hackathon**

An agent that **learns from its own past play** on ARC-AGI-3 by distilling
natural-language *hindsight* into a small, **parametric** two-tier memory — and
recalling the right hint, conditioned on what it's seeing, on its next attempt.

> The reward is sparse, but the **experience is rich.** Instead of waiting for a
> scalar reward signal, a teacher model retrospects over each play-through in
> *natural language* — "you spammed a blocked direction; verify the player moved
> and switch if it didn't" — and that feedback is compiled into the weights of a
> small memory model that the frozen policy queries as it plays.

---

## The idea

Frontier models score **<1%** on ARC-AGI-3 — the benchmark is built to reward
*learning, not memorizing*. Our entry keeps the **policy frozen** (Claude, via
the HUD gateway) and makes the *agent around it* improve, the way a person does:
by remembering what worked and what didn't, and recalling it when a similar
situation recurs.

Crucially, the memory is **parametric** (LoRA weights), not a hand-written notes
file — and it's **retrieved by scene**, because for novel situations *relevance ≠
text similarity*. A small model *learns the mapping* from "what I'm seeing" to
"what I should know."

## Architecture — a 4-tier memory

| Tier | What | Form | Updated |
|---|---|---|---|
| 0 | Universal priors | text (system prompt) | once |
| 1 | Per-game invariants | text (auto-promoted rules) | when a rule recurs across levels |
| 2 | **LTM** — long-term, transferable | **Qwen3-4B LoRA** | batch SFT (consolidation) |
| 3 | **STM** — short-term, this level | **Qwen3-1.7B LoRA** | live, between attempts; reset per level |

The two parametric tiers form a **cascade** (episodic → semantic):

```
current scene (structured text)
   → STM  → nuance   (level-specific observation: "exit corridor opens right near x=40")
   → LTM(scene+nuance) → note   (transferable instruction: "when up is walled, sweep right then up")
   → injected into the policy's tool stream
```

STM supplies *what's specific here*; LTM turns scene+nuance into *actionable,
transferable guidance*. This mirrors complementary-learning-systems
(hippocampus → neocortex): on level-clear, STM's nuances consolidate into LTM,
then STM resets.

## How it works (the pipeline)

```
frame ─► scene.py ─────────► structured text (objects + frame-delta), NOT pixels
            (grid→text)        e.g. "darkred 5x3 block@(36,48); changed: 15 cells green->darkred ..."
                                     │
env.py (ARC_TRACE) ─► trajectory traces (scene + actions + reasoning + outcome)
                                     │
retrospect.py ─► GPT-5.2 (reasoning=high) hindsight ─► (scene → nuance / note) datapoints
   teacher cites the pivotal step; we attach that step's real scene.py encoding
                                     │
build_datasets.py ─► LTM set (scene+nuance→note) + per-level STM sets + held-out eval
                                     │
modal_app.py ─► LoRA-SFT on Modal (A100), serve STM+LTM, swappable adapters on a Volume
                                     │
mem_client.py ◄─ env.py queries the served cascade PER STEP and injects the note
```

The whole memory representation is **text** in a *shared cross-game vocabulary*
(`N cells in WxH block@(x,y)`, `changed: A->B`, `available=[...]`), so patterns
("a block that swaps with the background on a directional action is the player")
can transfer between games — something raw pixels or free-form prose wouldn't give.

## Repo layout

| File | Role |
|---|---|
| `scene.py` | deterministic grid → structured-text encoder (the memory's input) |
| `env.py` | HUD ARC-AGI-3 environment + gated trace logging + per-step memory injection |
| `retrospect.py` | GPT-5.2 teacher: trajectory → `(scene→nuance/note)` datapoints |
| `mem_format.py` | single source of truth for STM/LTM prompts (train == inference) |
| `build_datasets.py` | corpus → LTM / per-level STM / held-out eval sets |
| `modal_app.py` | LoRA train + serve (Qwen3-1.7B STM, Qwen3-4B LTM) on Modal |
| `mem_client.py` | client `env.py` uses to query the served cascade |
| `ab_run.py` / `ab_compare.py` | baseline-vs-memory A/B harness |
| `eval.py` | held-out "right-note" metric (adapter vs base, GPT-judged) |
| `rsi_loop.py` | autonomous self-improvement loop (play → retrospect → retrain → repeat) |

Everything memory-related is **gated** (`ARC_MEMORY`, `ARC_TRACE`, `ARC_MEM_MODEL`)
so a baseline run is byte-identical to the stock environment.

## Results (honest)

See `report.html` (the demo) for the live numbers and the rigorous analysis.
Headline, stated plainly:

- The **full system works end-to-end**: scene→text, hindsight retrospection,
  two-tier LoRA training/serving, the STM→LTM cascade, and per-step injection
  into the live policy are all verified.
- The cascade produces **concrete, grounded** guidance (e.g. *"verify the player
  bbox actually changes; if it doesn't you hit a wall — switch direction instead
  of spamming the blocked action"*).
- **Performance lift on ARC scores is the open question.** An early A/B "win"
  turned out to be **noise** (the same note appeared in 4 attempts; only 1
  cleared) — we caught and reported that rather than overclaiming. We then made
  notes concrete + injection per-step and re-ran; `report.html` carries the
  current `rsi_loop` success-rate curve.

This is a benchmark where frontier models score <1%; we treat a *working
self-improvement system* + an *honest read of whether it moves the needle* as the
contribution.

## How to run

```bash
uv venv && uv pip install -e .          # deps (hud, anthropic, openai, modal, pillow)
cp .env.example .env                    # add HUD_API_KEY, ARC_API_KEY, OPENAI_API_KEY, MODAL_TOKEN_*

# 1. harvest trajectories (baseline play, with tracing)
ARC_TRACE=1 uv run python self_improve.py ls20 2

# 2. retrospect into datapoints, build datasets
uv run python retrospect.py --all
uv run python build_datasets.py

# 3. train + serve the memory on Modal
uv run modal deploy modal_app.py
uv run modal run modal_app.py::train_ltm
uv run modal run modal_app.py::train_stm --stm-file ls20-9607627b__L1

# 4. A/B: baseline vs memory-on
ARC_TRACE=1 uv run python ab_run.py ls20 5 claude-opus-4-7 30 baseline
ARC_TRACE=1 ARC_MEM_MODEL=1 ARC_STM_TAG=stm__ls20-9607627b__L1 \
    uv run python ab_run.py ls20 5 claude-opus-4-7 30 memory
uv run python ab_compare.py ls20

# or the full autonomous self-improvement loop:
uv run python rsi_loop.py
```

## Limitations & honest findings

- **Serving is unoptimized** (HF `generate`, two sequential passes) → ~6–9s per
  cascade; production would use vLLM (~1s) to make per-step injection cheap.
- **Per-step injection bloats the policy's context** (notes + frames accumulate);
  a windowed/transient injection is the right fix.
- **ARC-AGI-3 is largely a spatial-reasoning bottleneck**, which declarative
  memory can't fix; the lever that *can* help is **procedural** memory
  ("you did X, it failed, do Y") — which is what we now generate.
- N is small, so single-game A/B results are **directional, not conclusive**.

## Roadmap

vLLM serving · enable policy extended-thinking · windowed context · more games ·
larger-N evaluation · live STM↔LTM consolidation across levels.

---
*Stack: HUD `hud-python` (gateway → Claude policy) · ARC Prize official runner ·
GPT-5.2 teacher · Qwen3 (1.7B/4B) LoRA on Modal A100 · PEFT/Transformers.*
