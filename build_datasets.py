"""data/corpus.jsonl -> SFT datasets for the two-tier memory (+ held-out eval).

  data/train/ltm.jsonl              all train games: (scene, nuance) -> note
  data/train/stm/<game>__L<lvl>.jsonl  per game/level: (scene) -> nuance
  data/eval/holdout.jsonl           held-out games: scene + gold note/nuance
                                    (the low-noise "right-note" metric set)

Held-out games are kept ENTIRELY out of LTM training so the eval measures
generalization, not memorization.

    uv run python build_datasets.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from mem_format import ltm_chat, stm_chat

HOLDOUT = {"tn36"}  # games withheld from LTM training -> clean generalization metric


def main() -> None:
    ex_dir = _DIR / "data" / "examples"
    files = sorted(ex_dir.glob("*.jsonl")) if ex_dir.exists() else []
    if not files:
        raise SystemExit("no data/examples/*.jsonl yet — run retrospect.py first")
    rows = []
    for f in files:
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    train = [r for r in rows if r["game_id"].split("-")[0] not in HOLDOUT]
    held = [r for r in rows if r["game_id"].split("-")[0] in HOLDOUT]

    out = _DIR / "data" / "train"
    (out / "stm").mkdir(parents=True, exist_ok=True)
    ev = _DIR / "data" / "eval"
    ev.mkdir(parents=True, exist_ok=True)

    # LTM: every train row -> (scene, nuance) -> note
    with open(out / "ltm.jsonl", "w") as fh:
        for r in train:
            fh.write(json.dumps({"messages": ltm_chat(r["scene"], r["nuance"], r["note"])}) + "\n")

    # STM: grouped per (game, level) -> (scene) -> nuance (one fast-adapter set each)
    by_level: dict[tuple, list] = defaultdict(list)
    for r in train:
        by_level[(r["game_id"], r["level"])].append(r)
    for (g, lv), rs in by_level.items():
        with open(out / "stm" / f"{g}__L{lv}.jsonl", "w") as fh:
            for r in rs:
                fh.write(json.dumps({"messages": stm_chat(r["scene"], r["nuance"])}) + "\n")

    # Eval: held-out scenes + gold answers for the right-note metric
    with open(ev / "holdout.jsonl", "w") as fh:
        for r in held:
            fh.write(json.dumps({
                "game_id": r["game_id"], "level": r["level"], "scene": r["scene"],
                "gold_nuance": r["nuance"], "gold_note": r["note"], "note_type": r["note_type"],
            }) + "\n")

    print(f"LTM   : {len(train)} rows            -> data/train/ltm.jsonl")
    print(f"STM   : {len(by_level)} game/level files  -> data/train/stm/")
    print(f"eval  : {len(held)} held-out rows      -> data/eval/holdout.jsonl  (held-out: {', '.join(sorted(HOLDOUT))})")
    print("train games :", ", ".join(sorted({r["game_id"] for r in train})) or "(none)")
    print("STM groups  :", ", ".join(f"{g}/L{lv}({len(rs)})" for (g, lv), rs in sorted(by_level.items())))


if __name__ == "__main__":
    main()
