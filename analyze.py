"""Competence metrics from recordings — signal beyond binary level completion.

Reads the official jsonl recordings (each line = {"timestamp", "data": frame}).
Even when no level completes (the ARC-AGI-3 norm), these expose whether the
agent is progressing: how far it got, how much it explored, how much it looped.

    uv run python analyze.py [game_prefix]   # default: all recordings
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

_ENV_DIR = Path(__file__).resolve().parent
_REC = _ENV_DIR / "recordings"


def _grid_hash(frame) -> str | None:
    if not frame:
        return None
    grid = frame[-1] if isinstance(frame, list) else frame
    return hashlib.md5(json.dumps(grid).encode()).hexdigest()[:8]


def analyze(path: Path) -> dict:
    frames = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        d = rec.get("data", rec) if isinstance(rec, dict) else None
        if isinstance(d, dict) and "levels_completed" in d:
            frames.append(d)

    states = [f.get("state") for f in frames]
    grids = [_grid_hash(f.get("frame")) for f in frames]
    grids = [g for g in grids if g]
    seen, repeats = set(), 0
    for g in grids:
        if g in seen:
            repeats += 1
        seen.add(g)
    avail = Counter(tuple(f.get("available_actions") or []) for f in frames)
    max_levels = max((f.get("levels_completed") or 0 for f in frames), default=0)
    terminal = next((s for s in states if s in ("WIN", "GAME_OVER")), None)
    return {
        "game": path.name.split(".")[0],
        "actions": len(frames),
        "max_levels": max_levels,
        "won": "WIN" in states,
        "terminal_seen": terminal or states[-1] if states else "?",
        "distinct_states": len(set(grids)),
        "revisit_frac": round(repeats / len(grids), 3) if grids else 0.0,
        "available_actions": dict(avail),
    }


def main() -> None:
    pref = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(_REC.glob("*.jsonl"))
    if pref:
        files = [f for f in files if f.name.startswith(pref)]
    if not files:
        raise SystemExit(f"no recordings in {_REC}")
    for f in files:
        m = analyze(f)
        print(
            f"{m['game']:<18} acts={m['actions']:>3} maxLvl={m['max_levels']} won={m['won']!s:<5} "
            f"distinct={m['distinct_states']:>3} revisit={m['revisit_frac']:<5} "
            f"avail={list(m['available_actions'])}"
        )


if __name__ == "__main__":
    main()
