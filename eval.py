"""Held-out 'right-note' metric — the low-noise signal that the memory learned.

For each held-out scene (game tn36, never seen in LTM training), generate the
note with the LoRA adapter AND with the bare base model, then have GPT-5.2
judge each against the gold note. Reports adapter vs base mean — a clean number
that doesn't depend on ARC's noisy reward.

    set -a; source .env; set +a
    uv run python eval.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from retrospect import _load_dotenv

_JUDGE_SYS = (
    "You score how well a CANDIDATE note captures the GOLD note's key, "
    "actionable knowledge for the given ARC-AGI-3 SCENE. 1.0 = captures the "
    "core knowledge; 0.5 = partially; 0.0 = misses it or is wrong/generic. "
    'Respond ONLY JSON: {"score": <float 0..1>, "reason": "<short>"}'
)


def judge(client, model, scene, gold, cand) -> float:
    user = f"SCENE:\n{scene}\n\nGOLD NOTE:\n{gold}\n\nCANDIDATE NOTE:\n{cand}"
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _JUDGE_SYS},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        reasoning_effort="low",  # scoring is easy; keep it fast/cheap
    )
    try:
        return float(json.loads(r.choices[0].message.content or "{}").get("score"))
    except Exception:
        return 0.0


def main() -> None:
    _load_dotenv()
    import modal
    from openai import OpenAI

    rows = [json.loads(l) for l in
            (_DIR / "data" / "eval" / "holdout.jsonl").read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit("no held-out eval rows (data/eval/holdout.jsonl is empty)")

    Memory = modal.Cls.from_name("arc-memory", "Memory")()
    Memory.refresh_ltm.remote()  # ensure latest adapter is loaded
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("OPENAI_TEACHER_MODEL", "gpt-5.2")

    out = []
    for i, ex in enumerate(rows, 1):
        cmp = Memory.ltm_compare.remote(ex["scene"], "")  # LTM-only, no STM nuance
        sa = judge(client, model, ex["scene"], ex["gold_note"], cmp["adapter"])
        sb = judge(client, model, ex["scene"], ex["gold_note"], cmp["base"])
        out.append({"note_type": ex["note_type"], "adapter": sa, "base": sb,
                    "adapter_note": cmp["adapter"], "base_note": cmp["base"],
                    "gold_note": ex["gold_note"]})
        print(f"{i:>2}/{len(rows)} [{ex['note_type']:<17}] adapter={sa:.2f}  base={sb:.2f}")

    a = sum(r["adapter"] for r in out) / len(out)
    b = sum(r["base"] for r in out) / len(out)
    print(f"\n=== HELD-OUT RIGHT-NOTE SCORE (n={len(out)}) ===")
    print(f"  adapter (trained LTM): {a:.3f}")
    print(f"  base    (Qwen3-4B)   : {b:.3f}")
    print(f"  lift                 : {a - b:+.3f}")
    (_DIR / "data" / "eval_result.json").write_text(
        json.dumps({"adapter_mean": a, "base_mean": b, "n": len(out), "rows": out}, indent=2))
    print("saved -> data/eval_result.json")


if __name__ == "__main__":
    main()
