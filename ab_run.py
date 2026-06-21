"""Run K attempts of one game under one condition; record per-attempt metrics.

Condition is set by the *environment* (so env.py imports the right gates in
this fresh process): baseline = no flags; memory = ARC_MEM_MODEL=1 [+ARC_STM_TAG].
ARC_TRACE=1 must be set so we can parse max-level / action-count / whether the
memory note was actually injected.

    ARC_TRACE=1 uv run python ab_run.py <game> <K> <model> <max_steps> <cond>
    ARC_TRACE=1 ARC_MEM_MODEL=1 ARC_STM_TAG=stm__ls20-9607627b__L1 \
        uv run python ab_run.py ls20 4 claude-opus-4-7 30 memory
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from env import play
from hud.eval import LocalRuntime, Taskset
from self_improve import _make_agent, _resolve


def _newest_trace(game_id: str) -> Path | None:
    ts = sorted((_DIR / "traces").glob(f"{game_id}*.trace.jsonl"))
    return ts[-1] if ts else None


def _parse_trace(path: Path) -> dict:
    maxlvl, nacts, mem, reward = 0, 0, None, None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        k = rec.get("kind")
        if k == "act":
            nacts += 1
            for s in (rec.get("steps") or []):
                maxlvl = max(maxlvl, s.get("levels") or 0)
        elif k == "memory":
            mem = rec.get("note")
        elif k == "end":
            reward = rec.get("reward")
    return {"max_level": maxlvl, "n_acts": nacts, "mem_injected": bool(mem), "mem_note": mem, "reward": reward}


async def main() -> None:
    game = _resolve(sys.argv[1])
    K = int(sys.argv[2])
    model = sys.argv[3]
    max_steps = int(sys.argv[4])
    cond = sys.argv[5]

    results = []
    for i in range(1, K + 1):
        t = play(game_id=game, max_actions=int(os.environ.get("ARC_MAX_ACTIONS", "80")))
        t.slug = f"ab-{game.split('-')[0]}-{cond}-i{i}"
        agent = _make_agent(model, max_steps)
        job = await Taskset(f"ab-{game}", [t]).run(
            agent, runtime=LocalRuntime(str(_DIR / "env.py")), max_concurrent=1)
        runs = job.results.get(t.slug, [])
        reward = sum(r.reward for r in runs) / len(runs) if runs else 0.0
        tr = _newest_trace(game)
        m = _parse_trace(tr) if tr else {}
        m["reward"] = round(reward, 4)
        results.append(m)
        print(f"  {cond} attempt {i}: reward={reward:.3f} maxLvl={m.get('max_level')} "
              f"acts={m.get('n_acts')} mem={'Y' if m.get('mem_injected') else 'N'}", flush=True)

    out = _DIR / "runs" / f"ab_{game.split('-')[0]}_{cond}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"game": game, "cond": cond, "model": model, "max_steps": max_steps, "attempts": results}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
