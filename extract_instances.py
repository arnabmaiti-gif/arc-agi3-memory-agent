"""Pull real (grid, nuance, note, next-action) instances from a memory run's
trace into data/memory_instances.json, for the visual 'memory in action' demo.

Each trace 'memory' record now carries the exact grid the agent faced + the
injected STM nuance / LTM note; we pair it with the agent's next act() so the
demo can show: board → injected note → what the agent did.

    uv run python extract_instances.py [trace.jsonl] [N]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def extract(path: Path, want: int = 4) -> list[dict]:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = []
    for i, r in enumerate(recs):
        if r.get("kind") != "memory" or not r.get("grid") or not r.get("note"):
            continue
        nxt = next((recs[j] for j in range(i + 1, len(recs)) if recs[j].get("kind") == "act"), {})
        steps = nxt.get("steps") or []
        out.append({
            "grid": r["grid"],
            "nuance": r.get("nuance") or "",
            "note": r["note"],
            "action": nxt.get("actions") or "",
            "result": (f"-> level {steps[-1].get('levels')} / {steps[-1].get('state')}" if steps else ""),
        })
        if len(out) >= want:
            break
    return out


def main() -> None:
    if len(sys.argv) > 1:
        traces = [Path(sys.argv[1])]
    else:  # newest trace that actually has grid-bearing memory records
        traces = sorted((_DIR / "traces").glob("*.trace.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    for t in traces:
        inst = extract(t, want)
        if inst:
            (_DIR / "data" / "memory_instances.json").write_text(json.dumps(inst))
            print(f"{len(inst)} instances from {t.name} -> data/memory_instances.json")
            for x in inst:
                print(f"  note: {x['note'][:90]}  | did: {x['action'][:24]} {x['result']}")
            return
    print("no grid-bearing memory instances found (run a memory attempt after grid-capture was added)")


if __name__ == "__main__":
    main()
