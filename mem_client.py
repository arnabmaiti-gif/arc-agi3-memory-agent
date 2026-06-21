"""Client: query the deployed Modal memory model from inside env.py.

Gated by ARC_MEM_MODEL. Any failure (app not deployed, cold-start too slow,
network) returns None within `timeout` so a game eval NEVER stalls or breaks —
the agent simply runs without the parametric note that turn.

Returns the STM->LTM cascade result {'nuance': ..., 'note': ...}. For Rung 1
we call with stm_tag=None (LTM only); Rung 2 passes a per-level STM tag.
"""

from __future__ import annotations

import concurrent.futures
import functools
import os

APP_NAME = "arc-memory"

_ON = os.environ.get("ARC_MEM_MODEL", "").lower() in ("1", "true", "yes", "on")
LAST_ERROR = ""  # last failure traceback, for diagnosis from the caller


def enabled() -> bool:
    return _ON


def stm_tag(game_id: str, level: int) -> str:
    """Adapter tag for a game/level STM — must match modal_app train() tags."""
    return f"stm__{game_id}__L{level}"


@functools.lru_cache(maxsize=1)
def _handle():
    import modal  # lazy: never imported unless ARC_MEM_MODEL is on

    return modal.Cls.from_name(APP_NAME, "Memory")()


def memory_note(scene: str, stm_tag: str | None = None, timeout: float = 60.0) -> dict | None:
    """Cascade scene (+ optional STM tag) -> {'nuance','note'} or None on any failure."""
    global LAST_ERROR
    if not _ON or not scene:
        return None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(lambda: _handle().cascade.remote(scene, stm_tag))
            res = fut.result(timeout=timeout)
        return res if isinstance(res, dict) and res.get("note") else None
    except Exception as exc:
        import traceback
        LAST_ERROR = traceback.format_exc()
        return None
