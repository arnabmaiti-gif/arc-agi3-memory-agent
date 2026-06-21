"""Self-improvement (RSI) loop for ARC-AGI-3.

Play ONE game repeatedly. With memory on, the agent's own end-of-game rules
summary is persisted (memory/<game_id>.md) and replayed into the next
attempt's prompt, so it carries what it learned forward — and we watch the
score climb across attempts (the learning curve).

    # memory ON  (self-improving):
    ARC_MEMORY=1 uv run python self_improve.py ls20 5
    # baseline   (no memory, fresh context every attempt):
    uv run python self_improve.py ls20 5

    positional args: <game prefix or full id> [iters] [model] [max_steps]

Mirrors run_bench.py's agent construction (HUD gateway → only HUD_API_KEY
needed) and Taskset/LocalRuntime usage.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

_ENV_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ENV_DIR))

from env import GAME_IDS, _mem_path, play
from hud.agents import create_agent
from hud.eval import LocalRuntime, Taskset


def _resolve(game: str) -> str:
    if game in GAME_IDS:
        return game
    for g in GAME_IDS:
        if g.startswith(game) or g.split("-")[0] == game:
            return g
    opts = ", ".join(g.split("-")[0] for g in GAME_IDS)
    raise SystemExit(f"unknown game '{game}'. options: {opts}")


def _make_agent(model: str, max_steps: int):
    # Same fallback as run_bench.py: if the model isn't in the gateway registry
    # yet, build a ClaudeAgent against the inference proxy directly.
    try:
        return create_agent(model, max_steps=max_steps)
    except ValueError:
        if not model.startswith("claude"):
            raise
        from hud.agents.claude import ClaudeAgent
        from hud.agents.types import ClaudeConfig
        from hud.utils.gateway import build_gateway_client

        return ClaudeAgent(ClaudeConfig(
            model=model,
            model_client=build_gateway_client("anthropic"),
            max_steps=max_steps,
        ))


async def main() -> None:
    game = _resolve(sys.argv[1] if len(sys.argv) > 1 else "ls20")
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    model = sys.argv[3] if len(sys.argv) > 3 else "claude-opus-4-7"
    max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    mem_on = os.environ.get("ARC_MEMORY", "").lower() in ("1", "true", "yes", "on")
    label = "MEMORY (self-improving)" if mem_on else "BASELINE (no memory)"

    # Start each curve from a clean slate so the learning signal is honest.
    mem = _mem_path(game)
    best = mem.with_suffix(".best")
    for p in (mem, best):
        if p.exists():
            p.unlink()

    print(f"=== {label} | game={game} | iters={iters} | model={model} | max_steps={max_steps} ===")
    if not mem_on:
        print("(memory OFF — set ARC_MEMORY=1 to enable the self-improvement loop)")

    curve: list[float] = []
    rulebooks: list[str] = []
    for i in range(1, iters + 1):
        t = play(game_id=game)
        t.slug = f"arc-agi-3-{game.split('-')[0]}-i{i}"
        agent = _make_agent(model, max_steps)
        job = await Taskset(f"rsi-{game}", [t]).run(
            agent,
            runtime=LocalRuntime(str(_ENV_DIR / "env.py")),
            max_concurrent=1,
        )
        runs = job.results.get(t.slug, [])
        reward = sum(r.reward for r in runs) / len(runs) if runs else 0.0
        curve.append(reward)
        rb = mem.read_text() if mem.exists() else ""
        rulebooks.append(rb)
        print(f"  attempt {i}: reward={reward:.4f}   (rulebook: {len(rb)} chars)")

    print(f"\n{label} curve: " + "  ".join(f"{c:.3f}" for c in curve))
    print(f"best={max(curve):.4f}  final={curve[-1]:.4f}  mean={sum(curve)/len(curve):.4f}")

    # Persist the run so report.py can chart it (memory vs baseline).
    runs_dir = _ENV_DIR / "runs"
    runs_dir.mkdir(exist_ok=True)
    kind = "memory" if mem_on else "baseline"
    record = {
        "kind": kind,
        "game": game,
        "model": model,
        "max_steps": max_steps,
        "curve": curve,
        "rulebooks": rulebooks,
    }
    out = runs_dir / f"{kind}_{game.split('-')[0]}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"run saved -> {out}")
    if mem.exists():
        print(f"learned rulebook -> {mem}")


if __name__ == "__main__":
    asyncio.run(main())
