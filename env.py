"""ARC-AGI-3 on HUD v6, running on the official Arc Prize runner.

The play loop is the official one: a vendored copy of the ARC-AGI-3-Agents
``Agent`` class (``arc_agents/``, MIT) runs its unmodified ``main()`` loop in a
thread per game — frames history, action counter, MAX_ACTIONS, jsonl
recordings, cleanup — while HUD's model supplies actions through MCP tools
bridged into ``choose_action``. Scoring is the official scorecard
(``open_scorecard`` → ``make(game, scorecard_id)`` → ``close_scorecard``), and
every action carries the model's ``reasoning`` metadata exactly like the
official LLM agents do.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import queue
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from arc_agi import Arcade, OperationMode
from arcengine.enums import GameAction, GameState

from fastmcp import FastMCP

from hud.agents.types import ContentResult
from hud.capabilities import Capability
from hud.environment import Environment

from arc_agents.agent import Agent as ArcAgent

logger = logging.getLogger("arc-agi-3-env")

_ENV_DIR = Path(__file__).resolve().parent
if str(_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(_ENV_DIR))  # so `import scene` works under the runtime
# The official Recorder reads this; keep its jsonl recordings with the env.
os.environ.setdefault("RECORDINGS_DIR", str(_ENV_DIR / "recordings"))

# ─── natural-language rulebook memory (gated by ARC_MEMORY) ────────────
# The agent's own end-of-game summary is persisted per game and replayed
# into the next attempt's prompt — a self-authored, in-context "rulebook"
# that lets the policy carry what it learned forward across attempts.
# Fully gated: with ARC_MEMORY unset the prompt is byte-identical to the
# original, so a baseline run is unaffected.
_MEM_DIR = _ENV_DIR / "memory"
_MEMORY_ON = os.environ.get("ARC_MEMORY", "").lower() in ("1", "true", "yes", "on")
# Parametric memory model (Modal-served STM+LTM cascade), injected at attempt
# start. Separate gate so it composes with / is independent of the text rulebook.
_MEM_MODEL_ON = os.environ.get("ARC_MEM_MODEL", "").lower() in ("1", "true", "yes", "on")
# Planning-strategy guidance appended to the task prompt (gated) — targets the
# observed failure (spamming blocked moves) and pushes past level 1.
_STRATEGY_ON = os.environ.get("ARC_STRATEGY", "").lower() in ("1", "true", "yes", "on")


def _strategy_suffix() -> str:
    if not _STRATEGY_ON:
        return ""
    return (
        "\n\nSTRATEGY — follow this every attempt:\n"
        "1. SURVEY first: name the background, the walls, the player sprite (the "
        "object that moves when you act), the collectible/target, and the exit.\n"
        "2. PLAN a route to the collectible, then to the exit, before moving.\n"
        "3. VERIFY each move: after an act, check the player's cells actually "
        "changed position. If they did NOT, you hit a wall/boundary — do not "
        "repeat that direction; try a different one.\n"
        "4. After a level clears the board changes — immediately re-survey, find "
        "the new collectible + exit, and keep going.\n"
        "5. Use your FULL action budget to clear as many levels as you can."
    )


# ─── pathfinding tool view (gated by ARC_TOOLS) ────────────────────────
_TOOLS_ON = os.environ.get("ARC_TOOLS", "").lower() in ("1", "true", "yes", "on")


def _toolview(agent: "HudBridgeAgent") -> str:
    """Structured object list (with coordinates) + ASCII map, so the agent can
    read the floor color and player/target coordinates to feed plan_path."""
    if not _TOOLS_ON:
        return ""
    try:
        from scene import ascii_map, describe_scene

        f = agent.frames[-1]
        grid = f.frame[-1]
        return ("\n\n--- STRUCTURED VIEW (use with plan_path) ---\n"
                + describe_scene(grid, state=f.state.name, levels=f.levels_completed,
                                 win_levels=f.win_levels, available=list(f.available_actions or []))
                + "\n\n" + ascii_map(grid, downsample=2))
    except Exception:
        return ""


def _tools_suffix() -> str:
    if not _TOOLS_ON:
        return ""
    return (
        "\n\nPATHFINDING TOOL: you have plan_path(walkable, player_x, player_y, "
        "target_x, target_y) — a BFS solver for the hard part (navigation). From "
        "the STRUCTURED VIEW + ASCII map in each result, identify (a) the FLOOR "
        "color the player moves through, (b) the player's cell, (c) the target "
        "(collect the collectible first, then the exit). Call plan_path to get a "
        "cell route, execute it as directional moves (ACTION1-4) verifying the "
        "player actually shifts, and re-call plan_path after a few moves or if blocked."
    )


def _mem_path(game_id: str) -> Path:
    return _MEM_DIR / f"{game_id}.md"


def _read_memory(game_id: str) -> str:
    if not _MEMORY_ON:
        return ""
    p = _mem_path(game_id)
    try:
        return p.read_text().strip() if p.exists() else ""
    except Exception:
        return ""


def _best_path(game_id: str) -> Path:
    return _MEM_DIR / f"{game_id}.best"


def _read_best(game_id: str) -> float:
    p = _best_path(game_id)
    try:
        return float(p.read_text().strip()) if p.exists() else -1.0
    except Exception:
        return -1.0


def _write_memory(game_id: str, text: str, reward: float) -> None:
    """Ratchet memory: keep the rulebook from the best attempt so far.

    Last-write-wins lets a later, doubt-filled attempt clobber a winning
    rulebook (we saw exactly that). Only overwrite when this attempt did as
    well or better, so hard-won lessons are never lost.
    """
    if not _MEMORY_ON or not text or not text.strip():
        return
    try:
        _MEM_DIR.mkdir(exist_ok=True)
        if reward >= _read_best(game_id):
            _mem_path(game_id).write_text(text.strip() + "\n")
            _best_path(game_id).write_text(str(reward))
    except Exception as exc:
        logger.warning("memory write failed for %s: %s", game_id, exc)


def _memory_suffix(game_id: str) -> str:
    prior = _read_memory(game_id)
    if not prior:
        return ""
    return (
        "\n\n--- YOUR NOTES FROM PREVIOUS ATTEMPTS AT THIS GAME ---\n"
        "(You wrote these. Use them as a starting point, but verify against "
        "what you actually observe, and refine them in your final summary.)\n"
        f"{prior}\n"
        "--- END NOTES ---"
    )


def _maybe_inject_memory(agent: "HudBridgeAgent") -> str:
    """Per-step parametric memory: query the STM->LTM cascade on the CURRENT
    scene each tool call and surface the note whenever it changes (deduped, so
    it doesn't repeat). This puts the relevant hint in front of the agent at the
    moment it faces the scenario — not one stale note at the start.

    Gated by ARC_MEM_MODEL; never raises. ARC_STM_TAG selects this level's STM
    adapter for the cascade; absent it, LTM-only.
    """
    if not _MEM_MODEL_ON or _session is None:
        return ""
    # Throttle: query every ARC_MEM_EVERY-th tool call (1 = every call). Keeps
    # per-step latency + context bloat down on long attempts.
    _session["mem_calls"] = _session.get("mem_calls", 0) + 1
    every = int(os.environ.get("ARC_MEM_EVERY", "1") or "1")
    if every > 1 and (_session["mem_calls"] - 1) % every != 0:
        return ""
    try:
        import mem_client

        scene = _scene_of(agent.frames[-1] if agent.frames else None)
        if not scene:
            return ""
        stm_tag = os.environ.get("ARC_STM_TAG") or None
        res = mem_client.memory_note(scene, stm_tag, timeout=150)
        if not res or not res.get("note"):
            return ""
        note = res["note"]
        if note == _session.get("last_note"):
            return ""  # unchanged since last surfaced — don't repeat it
        _session["last_note"] = note
        _trace({"kind": "memory", "stm_tag": stm_tag,
                "nuance": res.get("nuance"), "note": note})
        block = "\n\n[MEMORY — hint from past attempts at this game; verify against what you see]\n"
        if res.get("nuance"):
            block += f"- noticed: {res['nuance']}\n"
        block += f"- try: {note}"
        return block
    except Exception as exc:
        _trace({"kind": "mem_error", "error": repr(exc)})
        logger.warning("memory injection failed: %r", exc)
        return ""

# ─── trajectory tracing (gated by ARC_TRACE) ───────────────────────────
# The clean per-attempt trace the raw recordings can't give us: the scene
# description at each decision, the actions + the model's reasoning, and the
# resulting state. This is the substrate the GPT teacher retrospects over to
# manufacture (scene -> note) training data. Side-channel only — it never
# touches the prompt, so baseline behaviour and reward are unchanged.
_TRACE_DIR = _ENV_DIR / "traces"
_TRACE_ON = os.environ.get("ARC_TRACE", "").lower() in ("1", "true", "yes", "on")


def _scene_of(frame: Any, prev_grid: Any = None, action: str | None = None) -> str:
    if frame is None or not getattr(frame, "frame", None):
        return ""
    try:
        from scene import describe_scene

        return describe_scene(
            frame.frame[-1], prev_grid,
            state=frame.state.name, levels=frame.levels_completed,
            win_levels=frame.win_levels,
            available=list(frame.available_actions or []), action=action,
        )
    except Exception:
        return ""


def _trace_write(path: str | None, record: dict) -> None:
    if not path:
        return
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.debug("trace write failed: %s", exc)


def _trace(record: dict) -> None:
    if not _TRACE_ON or _session is None:
        return
    _trace_write(_session.get("trace"), record)


# ─── the 25 public bench games ─────────────────────────────────────────

GAME_IDS = [
    "lp85-305b61c3", "cn04-2fe56bfb", "ft09-0d8bbf25", "sp80-589a99af",
    "dc22-fdcac232", "cd82-fb555c5d", "su15-1944f8ab", "ls20-9607627b",
    "vc33-5430563c", "m0r0-492f87ba", "ka59-38d34dbb", "lf52-271a04aa",
    "tu93-0768757b", "tn36-ef4dde99", "r11l-495a7899", "bp35-0a0ad940",
    "tr87-cd924810", "s5i5-18d95033", "ar25-0c556536", "sb26-7fbdac44",
    "sc25-635fd71a", "g50t-5849a774", "sk48-d8078629", "re86-8af5384d",
    "wa30-ee6fef47",
]

# ─── official toolkit: one Arcade + one scorecard per process ──────────

_arcade: Arcade | None = None
_card_id: str | None = None
_card_owned = False  # we opened it (close at grade) vs. passed in by a runner


def _arc() -> Arcade:
    global _arcade
    if _arcade is None:
        _arcade = Arcade(
            operation_mode=OperationMode(os.environ.get("ARC_OPERATION_MODE", "normal")),
            environments_dir=str(_ENV_DIR / "environment_files"),
            recordings_dir=str(_ENV_DIR / "recordings"),
        )
    return _arcade


def _scorecard(scorecard_id: str | None, tags: list[str]) -> str:
    global _card_id, _card_owned
    if scorecard_id:
        _card_id, _card_owned = scorecard_id, False
    elif _card_id is None:
        _card_id = _arc().open_scorecard(tags=tags)
        _card_owned = True
    return _card_id


# ─── the official runner, bridged ──────────────────────────────────────

_STOP = object()


class HudBridgeAgent(ArcAgent):
    """The official ``Agent`` with HUD's model as the policy.

    ``main()`` (the official loop) runs unmodified in a thread; HUD's model
    pushes ``GameAction``s through ``in_q`` and receives each resulting
    ``FrameData`` on ``out_q``. ``choose_action`` is the designed extension
    point; ``take_action`` is only intercepted for the stop sentinel.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.in_q: queue.Queue[Any] = queue.Queue()
        self.out_q: queue.Queue[Any] = queue.Queue()
        self._stop = False
        super().__init__(*args, **kwargs)

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return self._stop or latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[Any], latest_frame: Any) -> GameAction:
        item = self.in_q.get()
        if item is _STOP:
            self._stop = True
            return GameAction.RESET  # never executed: take_action skips on stop
        return item

    def take_action(self, action: GameAction) -> Any:
        if self._stop:
            return None
        try:
            frame = super().take_action(action)
            self.out_q.put(frame)
            return frame
        except Exception as exc:  # surface engine errors to the tool instead of dying
            self.out_q.put(exc)
            raise


_session: dict[str, Any] | None = None  # {"agent", "thread", "game_id"}


def _start_session(game_id: str, scorecard_id: str | None, max_actions: int) -> None:
    global _session
    card = _scorecard(scorecard_id, tags=["hud", game_id.split("-")[0]])
    wrapper = _arc().make(game_id, scorecard_id=card)
    agent = HudBridgeAgent(
        card_id=card,
        game_id=game_id,
        agent_name="hud",
        ROOT_URL=os.environ.get("ARC_ROOT_URL", "https://three.arcprize.org"),
        record=True,
        arc_env=wrapper,
        tags=["hud"],
    )
    agent.MAX_ACTIONS = max_actions
    thread = threading.Thread(target=agent.main, daemon=True, name=f"arc-{game_id}")
    thread.start()
    _session = {"agent": agent, "thread": thread, "game_id": game_id}
    if _TRACE_ON:
        try:
            _TRACE_DIR.mkdir(exist_ok=True)
            _session["trace"] = str(_TRACE_DIR / f"{game_id}.{int(time.time())}.trace.jsonl")
        except Exception as exc:
            logger.debug("trace init failed: %s", exc)


def _finish_session() -> Any:
    """Stop the official loop (its cleanup runs) and return the last frame."""
    global _session
    if _session is None:
        return None
    agent: HudBridgeAgent = _session["agent"]
    if _session["thread"].is_alive():
        agent.in_q.put(_STOP)
        _session["thread"].join(timeout=10)
    frame = agent.frames[-1] if agent.frames else None
    _session = None
    return frame


# ─── rendering ─────────────────────────────────────────────────────────

_PALETTE = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (87, 36, 194), (46, 26, 71),
    (255, 255, 255), (61, 153, 112), (57, 204, 204), (1, 255, 112),
]
_SCALE = 6  # 64 -> 384 px


def _render(frame: Any) -> str | None:
    if frame is None or not frame.frame:
        return None
    grid = frame.frame[-1]
    h, w = len(grid), len(grid[0])
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        row = grid[y]
        for x in range(w):
            px[x, y] = _PALETTE[int(row[x]) % 16]
    img = img.resize((w * _SCALE, h * _SCALE), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _status(agent: HudBridgeAgent) -> str:
    f = agent.frames[-1]
    acts = ", ".join(f"ACTION{a}" for a in (f.available_actions or []))
    return (
        f"state={f.state.name} | levels={f.levels_completed}/{f.win_levels} "
        f"| actions={agent.action_counter}/{agent.MAX_ACTIONS} | available=[{acts or '?'}]"
    )


# ─── MCP tools the model drives ────────────────────────────────────────

server = FastMCP(name="arc-agi-3")

_TOKEN = re.compile(r"(RESET|ACTION[1-7])(?:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\))?", re.I)


def _content(text: str, image: str | None = None) -> list:
    """A tool result: text plus the rendered frame when present (SDK helper)."""
    return ContentResult(output=text, base64_image=image).to_content_blocks()


@server.tool
async def look() -> list:
    """Return the current game frame as an image plus state summary."""
    if _session is None:
        return _content("No active game.")
    agent = _session["agent"]
    mem = await asyncio.to_thread(_maybe_inject_memory, agent)
    return _content(_status(agent) + mem + _toolview(agent), _render(agent.frames[-1]))


@server.tool
async def act(actions: str, reasoning: str = "") -> list:
    """Execute up to 12 game actions in order, then return the resulting frame.

    `actions`: space/comma-separated ACTION1..ACTION7 tokens — ACTION6 takes
    click coordinates as ACTION6(x,y), 0-63 — plus RESET to restart the
    attempt. Example: "ACTION1 ACTION1 ACTION6(31,40)".
    `reasoning`: a short note on your plan; it is attached to each action and
    appears on the official scorecard.
    """
    if _session is None:
        return _content("No active game.")
    agent: HudBridgeAgent = _session["agent"]
    tokens = _TOKEN.findall(actions or "")
    if not tokens:
        return _content("No valid actions. Use ACTION1..ACTION7, ACTION6(x,y), RESET.")

    before = agent.frames[-1] if agent.frames else None
    before_grid = before.frame[-1] if (before and before.frame) else None
    log: list[str] = []
    steps: list[dict[str, Any]] = []
    for name, x, y in tokens[:12]:
        if not _session["thread"].is_alive():
            log.append("game loop finished")
            break
        name = name.upper()
        action = GameAction.RESET if name == "RESET" else GameAction[name]
        data: dict[str, Any] = {"game_id": _session["game_id"]}
        if name == "ACTION6" and x:
            # Clamp to the 64x64 grid: an out-of-range click otherwise fails
            # GameAction validation and can kill the whole session.
            data.update(x=min(63, max(0, int(x))), y=min(63, max(0, int(y))))
        action.set_data(data)
        # The official reasoning metadata, shaped like the reference LLM agents.
        action.reasoning = {
            "agent": "hud",
            "action_chosen": name,
            "reasoning": reasoning[:500],
            "game_context": {
                "levels": agent.frames[-1].levels_completed,
                "state": agent.frames[-1].state.name,
                "action_counter": agent.action_counter,
            },
        }
        agent.in_q.put(action)
        try:
            result = await asyncio.to_thread(agent.out_q.get, True, 30)
        except Exception:
            log.append(f"{name}: timed out waiting for the engine")
            break
        if isinstance(result, Exception):
            log.append(f"{name}: engine error: {result}")
            break
        note = f"{name}" + (f"({x},{y})" if name == "ACTION6" and x else "")
        log.append(f"{note} -> levels={result.levels_completed} state={result.state.name}")
        steps.append({"action": note, "levels": result.levels_completed,
                      "state": result.state.name})
        if result.state is GameState.WIN:
            log.append("WIN: all levels complete")
            break
        if result.state is GameState.GAME_OVER:
            log.append("GAME_OVER: send RESET to retry")
            break
        if agent.action_counter >= agent.MAX_ACTIONS:
            log.append("action budget exhausted")
            break

    _trace({
        "kind": "act",
        "action_counter": agent.action_counter,
        "scene_before": _scene_of(before),
        "actions": actions,
        "reasoning": reasoning,
        "steps": steps,
        "scene_after": _scene_of(agent.frames[-1] if agent.frames else None,
                                 before_grid, action=actions),
    })
    mem = await asyncio.to_thread(_maybe_inject_memory, agent)
    return _content("\n".join(log) + "\n" + _status(agent) + mem + _toolview(agent),
                    _render(agent.frames[-1]))


@server.tool
async def plan_path(walkable: str, player_x: int, player_y: int,
                    target_x: int, target_y: int) -> list:
    """BFS shortest route for the player to a target over walkable floor cells.

    `walkable`: floor color the player moves through — a name ('green') or int.
    (player_x, player_y): the player's current cell; (target_x, target_y): the
    destination (collectible or exit). Returns a turn-by-turn route in grid cells;
    execute it as directional moves (ACTION1-4), verifying the player shifts.
    """
    if _session is None:
        return _content("No active game.")
    agent: HudBridgeAgent = _session["agent"]
    try:
        from scene import bfs_path, path_segments

        grid = agent.frames[-1].frame[-1]
        path = bfs_path(grid, walkable, (player_x, player_y), (target_x, target_y))
        if not path:
            return _content(
                f"No path over '{walkable}' floor from ({player_x},{player_y}) to "
                f"({target_x},{target_y}). Check the floor color, or pick a reachable "
                "sub-goal (a floor cell nearer the target).")
        segs = path_segments(path)
        route = ", ".join(f"{d} {n}" for d, n in segs)
        _trace({"kind": "plan_path", "walkable": walkable, "from": [player_x, player_y],
                "to": [target_x, target_y], "route": route})
        return _content(
            f"Route ({len(path)} cells, {len(segs)} legs): {route}.\n"
            "Execute as directional moves toward each leg; after each act verify the "
            "player block moved that way (if not, you hit a wall — re-call plan_path).")
    except Exception as exc:
        return _content(f"plan_path error: {exc}")


# ─── the HUD environment ───────────────────────────────────────────────

env = Environment(name="arc-agi-3")

_server_task: asyncio.Task[None] | None = None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


_PORT = _free_port()


async def _listening(host: str, port: int, timeout: float = 15.0) -> None:
    """Block until host:port accepts a connection — call before publishing."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), 0.5):
                return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"nothing listening on {host}:{port}")


@env.initialize
async def _up() -> None:
    global _server_task
    if _server_task is None:
        # Loopback bind: the control channel forwards loopback capabilities, so
        # only port 8765 needs publishing in Docker.
        _server_task = asyncio.create_task(
            server.run_async(transport="http", host="127.0.0.1", port=_PORT),
        )
        await _listening("127.0.0.1", _PORT)
    env.add_capability(Capability.mcp(name="game", url=f"http://127.0.0.1:{_PORT}/mcp"))


@env.shutdown
async def _down() -> None:
    global _server_task
    if _server_task is not None:
        _server_task.cancel()
        _server_task = None


@env.template()
async def play(game_id: str = "ls20-9607627b", scorecard_id: str = "",
               max_actions: int = 80):
    """Play one ARC-AGI-3 game on the official runner; reward is the official score."""
    _start_session(game_id, scorecard_id or None, max_actions)
    agent: HudBridgeAgent = _session["agent"]
    first = agent.frames[-1]
    _trace({"kind": "start", "game_id": game_id, "win_levels": first.win_levels,
            "max_actions": max_actions, "mem_model_on": _MEM_MODEL_ON,
            "stm_tag_env": os.environ.get("ARC_STM_TAG"), "scene": _scene_of(first)})
    prompt = (
        f"You are playing an unknown ARC-AGI-3 grid game (id: {game_id}) on the "
        f"official Arc Prize runner. It has {first.win_levels or '?'} levels; "
        "complete as many as you can.\n\n"
        "The board is a 64x64 grid of colors. Discover the mechanics by acting "
        "and watching how the frame changes.\n\n"
        "Tools:\n"
        "- look(): current frame image + state.\n"
        "- act(actions, reasoning): run up to 12 actions in order, e.g. "
        '"ACTION1 ACTION1 ACTION3" or "ACTION6(31,40)", with a short note of '
        "your current plan (it is recorded on the official scorecard).\n\n"
        "Actions: ACTION1-ACTION4 are usually movement/direction keys, ACTION5 "
        "interact/confirm, ACTION6(x,y) clicks grid cell (x right, y down, "
        "0-63), ACTION7 is game-specific. RESET restarts the current attempt "
        "(use it after GAME_OVER).\n\n"
        f"You have a budget of {max_actions} actions (the official agent "
        "limit). Batch several actions per act() call. Start with look(). When "
        "done or out of budget, reply with a short summary of the game's rules."
    )
    answer = yield prompt + _strategy_suffix() + _tools_suffix() + _memory_suffix(game_id)
    trace_path = _session.get("trace") if (_TRACE_ON and _session) else None
    last = _finish_session()
    # Official scoring: per-game entries materialize when the card closes. We
    # close the card we opened; a runner-supplied card is closed by the runner,
    # so grade from the final frame (same engine numbers) in that case.
    reward = 0.0
    if last is not None and last.win_levels:
        reward = 1.0 if last.state is GameState.WIN else round(
            last.levels_completed / last.win_levels, 4)
    if _card_owned:
        try:
            closed = _arc().close_scorecard(_card_id)
            envs = closed.model_dump().get("environments") or [] if closed else []
            entry = next((e for e in envs if e.get("id") == game_id), None)
            if entry:
                logger.info("official scorecard %s entry: %s", _card_id, entry)
                # The official per-game score (per-level partial credit, 0..1).
                reward = max(0.0, min(1.0, float(entry.get("score") or 0.0)))
                if entry.get("completed"):
                    reward = 1.0
                # Persist the official record so a bench runner can aggregate
                # across rollout processes (the local scorecard manager does
                # not merge plays from other processes into a shared card).
                results = _ENV_DIR / "results"
                results.mkdir(exist_ok=True)
                (results / f"{game_id}.json").write_text(
                    json.dumps({"card_id": _card_id, "entry": entry}, default=str),
                )
        except Exception as exc:
            logger.warning("scorecard close failed (%s); using frame-based reward", exc)
    # Persist the agent's rulebook keyed on this attempt's reward (ratchet:
    # only replaces the stored rulebook when this attempt did as well or better).
    if isinstance(answer, str):
        _write_memory(game_id, answer, reward)
    _trace_write(trace_path, {"kind": "end", "game_id": game_id, "reward": reward,
                              "answer": answer if isinstance(answer, str) else None})
    yield reward
