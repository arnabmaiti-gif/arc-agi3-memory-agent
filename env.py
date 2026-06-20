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
# The official Recorder reads this; keep its jsonl recordings with the env.
os.environ.setdefault("RECORDINGS_DIR", str(_ENV_DIR / "recordings"))

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
    return _content(_status(agent), _render(agent.frames[-1]))


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

    log: list[str] = []
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
        if result.state is GameState.WIN:
            log.append("WIN: all levels complete")
            break
        if result.state is GameState.GAME_OVER:
            log.append("GAME_OVER: send RESET to retry")
            break
        if agent.action_counter >= agent.MAX_ACTIONS:
            log.append("action budget exhausted")
            break

    return _content("\n".join(log) + "\n" + _status(agent), _render(agent.frames[-1]))


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
    answer = yield (
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
    yield reward
