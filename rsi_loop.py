"""Autonomous RSI improvement loop for the ARC-AGI-3 memory agent.

Each CYCLE:
  1. play K memory attempts (per-step injection) on the game,
  2. retrospect the NEW traces into datapoints,
  3. rebuild datasets, retrain STM (every cycle) + LTM (every Nth),
  4. log the cleared-L1 success rate.
Repeats so we can watch whether success rate climbs as the parametric memory
accumulates experience. Heavy timestamped logging, robust per-cycle, bounded by
wall-clock. Run with .env sourced (HUD/ARC/OPENAI/MODAL tokens in env):

    set -a; source .env; set +a
    uv run python rsi_loop.py
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

_DIR = Path(__file__).resolve().parent
GAME = os.environ.get("RSI_GAME", "ls20")
STM_FILE = os.environ.get("RSI_STM_FILE", "ls20-9607627b__L1")
STM_TAG = "stm__" + STM_FILE
K = int(os.environ.get("RSI_K", "3"))
MAX_CYCLES = int(os.environ.get("RSI_CYCLES", "8"))
MAX_MIN = float(os.environ.get("RSI_MAX_MIN", "300"))     # wall-clock budget (min)
MODEL = os.environ.get("RSI_MODEL", "claude-opus-4-7")
MAX_STEPS = int(os.environ.get("RSI_MAX_STEPS", "30"))
LTM_EVERY = int(os.environ.get("RSI_LTM_EVERY", "2"))     # retrain LTM every Nth cycle

T0 = time.time()


def log(msg: str) -> None:
    print(f"[rsi +{(time.time() - T0) / 60:5.1f}m {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], env_extra: dict | None = None, label: str = "") -> int:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"  # stream subprocess prints (ab_run per-attempt)
    if env_extra:
        env.update(env_extra)
    log(f"$ {label or ' '.join(cmd[:5])}")
    return subprocess.run(cmd, env=env, cwd=str(_DIR)).returncode


def ls_traces() -> set:
    return set((_DIR / "traces").glob(f"{GAME}*.trace.jsonl"))


def cleared(cond: str) -> tuple[int, int]:
    p = _DIR / "runs" / f"ab_{GAME}_{cond}.json"
    if not p.exists():
        return 0, 0
    a = json.loads(p.read_text())["attempts"]
    return sum(1 for x in a if (x.get("max_level") or 0) >= 1), len(a)


def main() -> None:
    log(f"START game={GAME} K={K} cycles={MAX_CYCLES} budget={MAX_MIN}m model={MODEL} stm={STM_TAG}")
    curve = []
    for c in range(1, MAX_CYCLES + 1):
        if (time.time() - T0) / 60 > MAX_MIN:
            log(f"wall-clock budget hit; stopping before cycle {c}")
            break
        log(f"================ CYCLE {c}/{MAX_CYCLES} ================")
        try:
            before = ls_traces()
            cond = f"mem_c{c}"
            run(["uv", "run", "python", "ab_run.py", GAME, str(K), MODEL, str(MAX_STEPS), cond],
                {"ARC_TRACE": "1", "ARC_MEM_MODEL": "1", "ARC_MEM_EVERY": "1", "ARC_STM_TAG": STM_TAG},
                label=f"play {K} memory attempts")
            cl, n = cleared(cond)
            curve.append({"cycle": c, "cleared": cl, "n": n,
                          "rate": round(cl / n, 3) if n else 0.0, "elapsed_min": round((time.time() - T0) / 60, 1)})
            (_DIR / "runs" / "rsi_curve.json").write_text(json.dumps(curve, indent=2))
            log(f"CYCLE {c} RESULT: cleared L1 {cl}/{n}   |   curve: "
                + " ".join(f"c{x['cycle']}={x['cleared']}/{x['n']}" for x in curve))

            new = sorted(ls_traces() - before)
            if new:
                run(["uv", "run", "python", "retrospect.py", *[str(p) for p in new]],
                    label=f"retrospect {len(new)} new traces")
            run(["uv", "run", "python", "build_datasets.py"], label="build_datasets")
            run(["uv", "run", "modal", "run", "modal_app.py::train_stm", "--stm-file", STM_FILE],
                label="retrain STM")
            if c % LTM_EVERY == 0:
                run(["uv", "run", "modal", "run", "modal_app.py::train_ltm"], label="retrain LTM")
            log(f"CYCLE {c} done ({(time.time() - T0) / 60:.1f}m elapsed)")
        except Exception as exc:
            log(f"CYCLE {c} ERROR: {exc!r} — continuing")
    log("DONE. final success-rate curve: "
        + " ".join(f"c{x['cycle']}={x['cleared']}/{x['n']}" for x in curve))


if __name__ == "__main__":
    main()
