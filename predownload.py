"""Vendor all 25 ARC-AGI-3 games into environment_files/ (run at image build)."""

from arc_agi import Arcade

from env import GAME_IDS, _ENV_DIR

arc = Arcade(environments_dir=str(_ENV_DIR / "environment_files"))
for gid in GAME_IDS:
    arc.make(gid)
    print("downloaded", gid)
print("all", len(GAME_IDS), "games vendored")
