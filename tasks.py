"""The ARC-AGI-3 public bench as a HUD taskset: one task per game."""

from env import GAME_IDS, env, play

__all__ = ["env", "tasks"]


def _mint() -> list:
    out = []
    for gid in GAME_IDS:
        t = play(game_id=gid)
        t.slug = f"arc-agi-3-{gid.split('-')[0]}"
        t.columns = {"game": gid.split("-")[0], "bench": "arc-agi-3-public"}
        out.append(t)
    return out


tasks = _mint()
