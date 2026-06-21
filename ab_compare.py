"""Compare baseline vs memory A/B runs for a game (reads runs/ab_<game>_*.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def _load(game: str, cond: str):
    p = _DIR / "runs" / f"ab_{game}_{cond}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _summ(d: dict) -> dict:
    a = d["attempts"]
    rew = [x.get("reward") or 0 for x in a]
    lv = [x.get("max_level") or 0 for x in a]
    acts = [x.get("n_acts") or 0 for x in a]
    inj = sum(1 for x in a if x.get("mem_injected"))
    return {"n": len(a), "reward": sum(rew) / len(a), "maxlvl": sum(lv) / len(a),
            "clearedL1": sum(1 for l in lv if l >= 1), "acts": sum(acts) / len(a), "injected": inj}


def main() -> None:
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    b, m = _load(game, "baseline"), _load(game, "memory")
    print(f"=== A/B: {game} ===")
    sb = sm = None
    for label, d in [("baseline", b), ("memory", m)]:
        if not d:
            print(f"{label:9} (missing)")
            continue
        s = _summ(d)
        if label == "baseline":
            sb = s
        else:
            sm = s
        extra = f" injected={s['injected']}/{s['n']}" if label == "memory" else ""
        print(f"{label:9} n={s['n']}  reward={s['reward']:.3f}  maxLvl={s['maxlvl']:.2f}  "
              f"clearedL1={s['clearedL1']}/{s['n']}  acts={s['acts']:.1f}{extra}")
    if sb and sm:
        print(f"\nlift (memory - baseline): reward {sm['reward'] - sb['reward']:+.3f}  |  "
              f"maxLvl {sm['maxlvl'] - sb['maxlvl']:+.2f}  |  clearedL1 {sm['clearedL1'] - sb['clearedL1']:+d}")


if __name__ == "__main__":
    main()
