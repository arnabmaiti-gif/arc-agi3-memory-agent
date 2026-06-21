"""Hindsight retrospection → (scene → note) training data.

The reward is sparse, but the *experience* is rich. After an attempt, an
independent Teacher (GPT) reads the whole trajectory — the scene at each
decision, what the agent did and why, how the board changed, and how it
ended — and writes down the knowledge that would have helped at each point:
rules it had to discover, info it lacked, strategies worth trying, dead ends
to avoid. That becomes a dataset of (scene → note) pairs we fine-tune the
memory model on. This is hindsight relabeling, done in language space.

    # build the teacher prompt from a trajectory and SEE it (no API call):
    uv run python retrospect.py traces/ls20-9607627b.<ts>.trace.jsonl --dry-run
    # with OPENAI_API_KEY in .env, actually call the teacher and append data:
    uv run python retrospect.py recordings/ls20-9607627b.<id>.recording.jsonl

Works off either a rich trace (env.py with ARC_TRACE=1: scene+actions+
reasoning) or a raw official recording (scene sequence only, no reasoning).
Appends examples to data/corpus.jsonl (the LTM corpus) and writes a
per-source file. Each example is tagged with its level so STM (per-level) and
LTM (cross-level) training sets can be split downstream.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ENV_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ENV_DIR))

from scene import describe_delta, describe_scene, frames_from_recording

NOTE_TYPES = ("rule", "missing_info", "candidate_strategy", "failed_strategy")
_DATA = _ENV_DIR / "data"


# ─── .env loading (no python-dotenv dependency) ────────────────────────

def _load_dotenv() -> None:
    p = _ENV_DIR / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ─── trajectory loading (trace OR recording) ───────────────────────────

def _load_trace(path: Path) -> dict:
    start, end, steps = {}, {}, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        k = r.get("kind")
        if k == "start":
            start = r
        elif k == "end":
            end = r
        elif k == "act":
            last = r["steps"][-1] if r.get("steps") else {}
            steps.append({
                "scene_before": r.get("scene_before", ""),
                "actions": r.get("actions", ""),
                "reasoning": r.get("reasoning", ""),
                "scene_after": r.get("scene_after", ""),
                "levels": last.get("levels"),
                "state": last.get("state"),
            })
    game_id = start.get("game_id") or end.get("game_id") or path.name.split(".")[0]
    return {
        "game_id": game_id,
        "win_levels": start.get("win_levels"),
        "reward": end.get("reward"),
        "rulebook": end.get("answer") or _read_rulebook(game_id),
        "steps": steps,
        "source": "trace",
    }


def _load_recording(path: Path) -> dict:
    frames = frames_from_recording(path)
    game_id = path.name.split(".")[0]
    steps, prev = [], None
    for f in frames:
        steps.append({
            "scene_before": "",
            "actions": "",
            "reasoning": "",
            "scene_after": describe_scene(
                f["grid"], prev, state=f.get("state"), levels=f.get("levels"),
                win_levels=f.get("win_levels"), available=f.get("available")),
            "levels": f.get("levels"),
            "state": f.get("state"),
        })
        prev = f["grid"]
    win_levels = frames[0].get("win_levels") if frames else None
    max_lvl = max((s["levels"] or 0 for s in steps), default=0)
    return {
        "game_id": game_id,
        "win_levels": win_levels,
        "reward": round(max_lvl / win_levels, 4) if win_levels else None,
        "rulebook": _read_rulebook(game_id),
        "steps": steps,
        "source": "recording",
    }


def _read_rulebook(game_id: str) -> str:
    p = _ENV_DIR / "memory" / f"{game_id}.md"
    try:
        return p.read_text().strip() if p.exists() else ""
    except Exception:
        return ""


def load_trajectory(path: Path) -> dict:
    return _load_trace(path) if ".trace." in path.name else _load_recording(path)


# ─── compress a trajectory into a teacher-readable transcript ──────────

def _delta_line(scene_after: str) -> str:
    for ln in scene_after.splitlines():
        if ln.startswith("changed:") or ln.startswith("no change"):
            return ln
    return ""


def compress(traj: dict, max_steps: int = 40) -> str:
    steps = traj["steps"]
    # Keep the informative steps: those that changed the board or the level.
    keep, prev_lvl = [], 0
    for i, s in enumerate(steps):
        dl = _delta_line(s["scene_after"])
        leveled = s.get("levels") is not None and s["levels"] != prev_lvl
        if leveled or dl.startswith("changed:") or s.get("state") in ("WIN", "GAME_OVER") or i == 0:
            keep.append((i, s, leveled))
        prev_lvl = s.get("levels") if s.get("levels") is not None else prev_lvl
    if len(keep) > max_steps:  # sample evenly, always keep first + last
        idx = sorted({0, len(keep) - 1, *(round(j * (len(keep) - 1) / (max_steps - 1)) for j in range(max_steps))})
        keep = [keep[j] for j in idx]

    lines = []
    opening = steps[0]["scene_before"] or steps[0]["scene_after"]
    lines.append("OPENING SCENE:\n" + opening + "\n")
    for i, s, leveled in keep:
        if leveled:
            lines.append(f"*** reached level {s['levels']} ***")
        bits = [f"step {i}"]
        if s["actions"]:
            bits.append(f"actions={s['actions']}")
        if s["reasoning"]:
            bits.append(f'reason="{s["reasoning"][:160]}"')
        dl = _delta_line(s["scene_after"])
        if dl:
            bits.append(dl)
        if s.get("state") in ("WIN", "GAME_OVER"):
            bits.append(f"STATE={s['state']}")
        lines.append("  " + " | ".join(bits))
    return "\n".join(lines)


# ─── teacher prompt ────────────────────────────────────────────────────

_SYSTEM = (
    "You are an expert analyst of ARC-AGI-3 grid puzzle games doing HINDSIGHT "
    "analysis. An agent (a vision LLM) just played a game and we want to give "
    "its NEXT attempt the knowledge it lacked. The trajectory is given as "
    "NUMBERED STEPS. The OPENING SCENE and each step's board come from a "
    "deterministic encoder: objects are connected color regions written as "
    "'N cells in WxH block@(x,y)', and 'changed:' lines show what moved or "
    "appeared between frames. Each step also shows the actions taken and the "
    "agent's stated reasoning.\n\n"
    "The agent now has a BFS pathfinding TOOL (plan_path) that handles the "
    "navigation, so memory must capture what the tool CANNOT infer — game "
    "SEMANTICS, OBJECTIVES, and GOTCHAS — NOT turn-by-turn directions. For EACH "
    "example produce TWO outputs grounded in the SAME cited step:\n"
    "  - nuance (short-term, level-specific): a concrete fact about THIS level for "
    "using the tool — the floor color, the player's identity, the collectible and "
    "exit locations (cite colors + coordinates), required order, or a gotcha. "
    "E.g. 'floor is green; collectible is the blue cross ~(20,32); exit is the Heh "
    "tile up top — collect the cross before the exit'.\n"
    "  - note (long-term, transferable): a reusable rule for ANY level/game — how "
    "to identify floor vs wall vs player vs collectible vs exit, the ACTION->"
    "direction mapping, objective order, or when to call plan_path. E.g. 'the "
    "player is the block that moves on a directional action; the floor is the "
    "large region it traverses; collect the collectible before the exit; feed "
    "plan_path the floor color'.\n"
    "  Do NOT give turn-by-turn navigation ('go up then left') — the pathfinder does that.\n\n"
    "Also give the STEP number (we attach that step's encoded board as the "
    "scene the memory keys on), the level, and a note_type:\n"
    "  - rule: a game mechanic that appears reliably true.\n"
    "  - missing_info: a fact the agent did not know and had to discover.\n"
    "  - candidate_strategy: a concrete approach worth trying next attempt.\n"
    "  - failed_strategy: something that wasted moves or caused GAME_OVER.\n\n"
    "Be specific and grounded — cite colors, positions, actions from the "
    "encoded board; no generic platitudes ('explore carefully'). Output 5-15 "
    "examples. Respond ONLY with JSON: {\"examples\":[{\"step\":<int>,"
    "\"level\":<int>,\"note_type\":\"<one of the four>\",\"nuance\":"
    "\"<level-specific observation>\",\"note\":\"<generalized guidance>\"}]}"
)


def build_messages(traj: dict) -> list[dict]:
    outcome = (
        f"game_id={traj['game_id']} | win_levels={traj.get('win_levels')} | "
        f"final_reward={traj.get('reward')}"
    )
    rb = traj.get("rulebook") or "(none)"
    user = (
        f"OUTCOME: {outcome}\n\n"
        f"THE AGENT'S OWN END-OF-GAME NOTES (may be partial or wrong):\n{rb}\n\n"
        f"TRAJECTORY:\n{compress(traj)}\n\n"
        "Now produce the hindsight training examples as specified."
    )
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user}]


# ─── teacher call + validation ─────────────────────────────────────────

def call_teacher(messages: list[dict], model: str) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=150.0, max_retries=2)
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    effort = os.environ.get("OPENAI_REASONING_EFFORT", "").strip()
    if effort:
        # Reasoning models (gpt-5.x / o-series) take reasoning_effort and only
        # support the default temperature, so we don't send temperature.
        kwargs["reasoning_effort"] = effort
    else:
        kwargs["temperature"] = 0.4
    resp = client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    return data.get("examples", []) if isinstance(data, dict) else []


def _scene_for_step(traj: dict, step) -> str:
    """The real scene.py encoding of the cited step — attached, never invented."""
    steps = traj["steps"]
    if not steps:
        return ""
    i = int(step) if isinstance(step, (int, float)) else 0
    i = max(0, min(i, len(steps) - 1))
    s = steps[i]
    return s.get("scene_after") or s.get("scene_before") or ""


def _level_for_step(traj: dict, step) -> int:
    """Level being PLAYED at this step (levels_completed + 1) — derived from the
    actual frame so it stays consistent with the attached scene, rather than
    trusting the teacher's claimed level."""
    steps = traj["steps"]
    if steps and isinstance(step, (int, float)) and 0 <= int(step) < len(steps):
        lv = steps[int(step)].get("levels")
        return (int(lv) if isinstance(lv, (int, float)) else 0) + 1
    return 1


def validate(examples: list[dict], traj: dict) -> list[dict]:
    out = []
    for e in examples:
        if not isinstance(e, dict):
            continue
        nt = str(e.get("note_type", "")).strip()
        nuance = str(e.get("nuance", "")).strip()   # STM target (scene -> nuance)
        note = str(e.get("note", "")).strip()        # LTM target (scene+nuance -> note)
        scene = _scene_for_step(traj, e.get("step"))  # canonical encoder output
        if nt not in NOTE_TYPES or not nuance or not note or not scene:
            continue
        out.append({
            "game_id": traj["game_id"],
            "level": _level_for_step(traj, e.get("step")),
            "step": int(e["step"]) if isinstance(e.get("step"), (int, float)) else None,
            "note_type": nt,
            "scene": scene,
            "nuance": nuance,
            "note": note,
        })
    return out


def _est_tokens(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages) // 4


_DATA_EX = _DATA / "examples"


def _targets(arg: str) -> list[Path]:
    p = Path(arg)
    if arg in ("--all", "all"):
        return sorted((_ENV_DIR / "traces").glob("*.trace.jsonl"))
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    return [p]


MIN_ACTS = 15   # skip degenerate traces (short diagnostics, gateway-truncated)
PER_GAME = 4    # cap per game so one game (ls20) can't flood the corpus


def _trace_stats(path: Path) -> tuple[int, int]:
    n, mx = 0, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") == "act":
            n += 1
            for s in (r.get("steps") or []):
                mx = max(mx, s.get("levels") or 0)
    return n, mx


def _curate(traces: list[Path]) -> list[Path]:
    """Keep substantive traces, balanced per game, preferring those that made
    progress — avoids flooding the corpus with redundant/degenerate runs."""
    by: dict[str, list] = {}
    for p in traces:
        n, mx = _trace_stats(p)
        if n < MIN_ACTS:
            continue
        by.setdefault(p.name.split("-")[0], []).append((mx, n, p))
    out: list[Path] = []
    for g, lst in by.items():
        lst.sort(key=lambda t: (-t[0], -t[1]))  # progress first, then length
        out += [p for _, _, p in lst[:PER_GAME]]
    return sorted(out)


def process_one(path: Path, model: str) -> list[dict]:
    """Retrospect one trajectory -> write data/examples/<stem>.examples.jsonl."""
    traj = load_trajectory(path)
    examples = validate(call_teacher(build_messages(traj), model), traj)
    if not examples:
        return []
    _DATA_EX.mkdir(parents=True, exist_ok=True)
    out = _DATA_EX / f"{path.stem}.examples.jsonl"
    with open(out, "w") as fh:
        for e in examples:
            fh.write(json.dumps(e) + "\n")
    return examples


def main() -> None:
    _load_dotenv()
    argv = sys.argv[1:]
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    want_all = "--all" in flags
    if not args and not want_all:
        raise SystemExit("usage: retrospect.py <trace|recording|dir|--all> [--dry-run] [--force]")
    model = os.environ.get("OPENAI_TEACHER_MODEL", "gpt-5.2")
    have_key = bool(os.environ.get("OPENAI_API_KEY"))
    dry = "--dry-run" in flags or not have_key

    files: list[Path] = []
    if want_all:
        allt = sorted((_ENV_DIR / "traces").glob("*.trace.jsonl"))
        files += _curate(allt)
        print(f"--all: curated {len(files)}/{len(allt)} traces "
              f"(>= {MIN_ACTS} acts, top {PER_GAME}/game by progress)")
    for a in args:
        files += _targets(a)
    if not files:
        raise SystemExit("no input files matched")

    if dry:
        traj = load_trajectory(files[0])
        messages = build_messages(traj)
        print(f"# DRY RUN ({'no OPENAI_API_KEY in .env' if not have_key else 'forced'}) — "
              f"teacher={model}, ~{_est_tokens(messages)} input tokens, {len(files)} file(s)\n")
        print("================ SYSTEM ================\n" + messages[0]["content"])
        print("\n================ USER ================\n" + messages[1]["content"])
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    _DATA_EX.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files
            if "--force" in flags or not (_DATA_EX / f"{f.stem}.examples.jsonl").exists()]
    print(f"processing {len(todo)} file(s) with {model} (concurrency 5) ...", flush=True)
    total = 0
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(process_one, f, model): f for f in todo}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                ex = fut.result()
            except Exception as exc:
                print(f"FAILED {f.name}: {exc}", flush=True)
                continue
            total += len(ex)
            print(f"  done {f.name}: {len(ex)} examples", flush=True)
    print(f"\nTOTAL: {total} examples across {len(todo)} file(s) -> data/examples/")


if __name__ == "__main__":
    main()
