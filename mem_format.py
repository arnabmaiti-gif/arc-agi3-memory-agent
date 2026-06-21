"""Shared prompt/format for the two-tier memory model (STM + LTM).

Imported by build_datasets.py (to render training rows), the Modal serving
app (to format inference requests), and any client. Single source of truth so
TRAINING == INFERENCE — if the prompt drifts between them the model degrades.

  STM (Qwen3-1.7B):  scene            -> nuance   (level-specific observation)
  LTM (Qwen3-4B):    scene + nuance   -> note     (generalized guidance)
"""

from __future__ import annotations

STM_SYSTEM = (
    "You are SHORT-TERM memory for the ARC-AGI-3 level currently being played. "
    "Given the current SCENE (a deterministic encoding of the board: objects as "
    "'N cells in WxH block@(x,y)' and 'changed:' deltas), recall ONE terse, "
    "level-specific observation about what is true here right now."
)

LTM_SYSTEM = (
    "You are LONG-TERM memory for an ARC-AGI-3 agent. Given the current SCENE "
    "(encoded board) and a level-specific NUANCE, output ONE concise, actionable "
    "note for the agent — an identification cue, the objective, a strategy, or a "
    "caution — phrased to transfer beyond this exact layout. <= 2 sentences."
)


def stm_user(scene: str) -> str:
    return f"SCENE:\n{scene}"


def ltm_user(scene: str, nuance: str) -> str:
    return f"SCENE:\n{scene}\nNUANCE: {nuance.strip() or '(none)'}"


def stm_chat(scene: str, nuance: str) -> list[dict]:
    """Training/inference messages for STM (assistant turn = the nuance)."""
    return [
        {"role": "system", "content": STM_SYSTEM},
        {"role": "user", "content": stm_user(scene)},
        {"role": "assistant", "content": nuance},
    ]


def ltm_chat(scene: str, nuance: str, note: str) -> list[dict]:
    """Training/inference messages for LTM (assistant turn = the note)."""
    return [
        {"role": "system", "content": LTM_SYSTEM},
        {"role": "user", "content": ltm_user(scene, nuance)},
        {"role": "assistant", "content": note},
    ]
