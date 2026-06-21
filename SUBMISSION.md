# Hackathon Submission — Self-Improving Memory Agent (ARC-AGI-3)

## Submit these two links on aivalley.io

| Field | Link |
|---|---|
| **Demo URL** (required) | https://claude.ai/code/artifact/ec59f432-dea7-4265-98b8-04032891a89d |
| **GitHub repo** (required) | https://github.com/arnabmaiti-gif/arc-agi3-memory-agent |
| Video / slides | optional |

**Before submitting:** the repo is **private** — make it public (or add the
judges) so the link opens for them:
```
gh repo edit arnabmaiti-gif/arc-agi3-memory-agent --visibility public --accept-visibility-change-consequences
```
The demo URL is a private Claude Artifact — open it and use **Share** to make it
viewable, or it may already be shareable via the link.

## One-paragraph pitch

A frozen Claude policy that **learns from its own past play** on ARC-AGI-3. After
each attempt, a GPT-5.2 teacher retrospects the trajectory in *natural language*
("you spammed a blocked direction — verify the player moved, else switch") and
compiles that hindsight into a small **two-tier parametric memory** (Qwen3-1.7B
short-term + Qwen3-4B long-term, LoRA on Modal). At inference the memory is
**retrieved per-scene** and injected into the agent every step. It's RSI through
*memory*, not reward — and we report honestly whether it moves a benchmark where
frontier models score <1%.

## What's genuinely demonstrated

- End-to-end working system: grid→structured-text encoder, hindsight
  retrospection, two-tier LoRA train+serve on Modal, STM→LTM cascade, per-step
  injection into the live policy. All verified.
- The cascade produces concrete, grounded, *procedural* guidance.
- **Honest evaluation:** an early "win" was shown to be noise (same note, 1/4
  cleared) and reported as such; we then made notes concrete + per-step and ran
  a self-improvement loop (`rsi_loop.py`) whose success-rate curve is in the demo.

## Talking points (4-min presentation)

1. **The unlock** — reward is sparse, but experience is rich; turn it into
   natural-language hindsight, then into *weights*.
2. **Two representations** — the policy sees pixels; the memory reads structure
   (shared cross-game vocabulary → transfer). *(opening visual in the demo)*
3. **The cascade** — episodic STM → semantic LTM, mirroring complementary
   learning systems; consolidate on level-clear, reset STM.
4. **Rigor** — we caught our own false positive. On a <1% benchmark, a working
   self-improvement system + an honest read is the contribution.
5. **Roadmap** — vLLM serving, windowed context, extended-thinking policy,
   larger-N eval.

*(Full architecture + how-to-run: see `README.md`. Live results: the demo URL
or re-run `uv run python report_v2.py`.)*
