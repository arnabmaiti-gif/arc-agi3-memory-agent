"""Tractability scan: play a subset of ARC-AGI-3 games once each (parallel) to
find which the agent can make progress on (reward > 0). Picks the demo game
for the self-improvement experiment.

    uv run python scan.py [max_steps] [game_prefix ...]
    # e.g. uv run python scan.py 50 ls20 ft09 sp80

Baseline only (no memory). Mirrors run_bench.py's machinery on a subset.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_ENV_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ENV_DIR))

from env import play
from hud.eval import LocalRuntime, Taskset
from self_improve import _make_agent, _resolve

DEFAULT_SUBSET = ["ls20", "ft09", "sp80", "vc33", "su15", "tn36"]


async def main() -> None:
    args = sys.argv[1:]
    max_steps = int(args[0]) if args and args[0].isdigit() else 50
    prefixes = [a for a in args if not a.isdigit()] or DEFAULT_SUBSET
    model = "claude-opus-4-7"
    concurrency = 3

    games = [_resolve(p) for p in prefixes]
    tasks = []
    for gid in games:
        t = play(game_id=gid)
        t.slug = f"scan-{gid.split('-')[0]}"
        tasks.append(t)

    print(f"=== tractability scan | {len(games)} games | max_steps={max_steps} | model={model} ===")
    agent = _make_agent(model, max_steps)
    job = await Taskset("arc-scan", tasks).run(
        agent,
        runtime=LocalRuntime(str(_ENV_DIR / "env.py")),
        max_concurrent=concurrency,
    )

    out: dict[str, float] = {}
    print("\n=== results ===")
    for t in tasks:
        runs = job.results.get(t.slug, [])
        r = sum(x.reward for x in runs) / len(runs) if runs else 0.0
        out[t.slug] = r
        flag = "  <-- progress!" if r > 0 else ""
        print(f"  {t.slug:<16} reward={r:.4f}{flag}")

    ranked = sorted(out.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\nbest: {ranked[0][0]} = {ranked[0][1]:.4f}")
    (_ENV_DIR / "runs").mkdir(exist_ok=True)
    (_ENV_DIR / "runs" / "scan.json").write_text(json.dumps(out, indent=2))
    print(f"saved -> {_ENV_DIR / 'runs' / 'scan.json'}")


if __name__ == "__main__":
    asyncio.run(main())
