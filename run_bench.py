"""The official ARC-AGI-3 public bench on HUD.

Fans the 25 games out as HUD rollouts (one fresh env process each via
LocalRuntime). Every game runs on the official runner and closes its own
official scorecard; this runner aggregates the official per-game records
(written by each rollout to results/) into the bench report.

    uv run python run_bench.py [model] [max_steps] [concurrency]
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

_ENV_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ENV_DIR))

from env import GAME_IDS, play
from hud.agents import create_agent
from hud.eval import LocalRuntime, Taskset


async def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-7"
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else 7

    results_dir = _ENV_DIR / "results"
    shutil.rmtree(results_dir, ignore_errors=True)

    tasks = []
    for gid in GAME_IDS:
        t = play(game_id=gid)  # each rollout owns its official scorecard
        t.slug = f"arc-agi-3-{gid.split('-')[0]}"
        t.columns = {"game": gid.split("-")[0], "bench": "arc-agi-3-public"}
        tasks.append(t)

    try:
        agent = create_agent(model, max_steps=max_steps)
    except ValueError:
        # Model not in the gateway *registry* yet, but the inference proxy may
        # still serve it (registry lag) — construct the agent directly.
        if not model.startswith("claude"):
            raise
        from hud.agents.claude import ClaudeAgent
        from hud.agents.types import ClaudeConfig
        from hud.utils.gateway import build_gateway_client

        agent = ClaudeAgent(ClaudeConfig(
            model=model,
            model_client=build_gateway_client("anthropic"),
            max_steps=max_steps,
        ))
    job = await Taskset("arc-agi-3-public", tasks).run(
        agent,
        runtime=LocalRuntime(str(_ENV_DIR / "env.py")),
        max_concurrent=concurrency,
    )

    print("\n=== HUD rewards (per game) ===")
    results = job.results  # slug -> [runs]; group-safe, no positional zip
    for task in tasks:
        runs = results.get(task.slug, [])
        mean = sum(r.reward for r in runs) / len(runs) if runs else 0.0
        print(f"  {task.slug:<22} reward={mean:.4f}")
    print(f"\nHUD mean reward: {job.reward:.4f}")

    # Aggregate the official per-game scorecard records the rollouts persisted.
    records = []
    for f in sorted(results_dir.glob("*.json")) if results_dir.exists() else []:
        records.append(json.loads(f.read_text()))
    if records:
        print(f"\n=== OFFICIAL PER-GAME SCORECARDS ({len(records)}/25) ===")
        total = 0.0
        for r in records:
            e = r["entry"]
            total += float(e.get("score") or 0.0)
            print(
                f"  {e.get('id'):<22} score={e.get('score')} "
                f"levels={e.get('levels_completed')} actions={e.get('actions')} "
                f"completed={e.get('completed')} card={r['card_id'][:8]}"
            )
        print(f"\nofficial mean score: {total / len(records):.4f}")
        out = _ENV_DIR / "scorecard.json"
        out.write_text(json.dumps(records, indent=2, default=str))
        print(f"full report: {out}")


if __name__ == "__main__":
    asyncio.run(main())
