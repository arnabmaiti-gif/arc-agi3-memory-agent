"""Deterministic structured-scene encoder for ARC-AGI-3 grids.

The policy (Claude) sees the rendered image; the *memory model* sees this:
a compact, abstracted text description built straight from the raw 64x64
integer grid the engine already gives us. Rules in these games are about
objects and how they change, so we surface exactly that — connected
components by color (position/size) and the frame-to-frame delta (what moved,
appeared, or vanished). No vision model, no lossy prose, near-lossless for
what a rule actually depends on.

Used three ways:
  - as the *query* the memory model conditions on (scene -> note),
  - as the per-step context appended to act()'s tool result (hero injection),
  - as the substrate the GPT teacher retrospects over to manufacture data.

Pure stdlib so it runs anywhere (env.py, Modal, offline dataset builds).
"""

from __future__ import annotations

from collections import Counter, deque

# Names roughly matching env.py's _PALETTE (index = color int 0..15).
COLOR_NAMES = [
    "black", "blue", "red", "green", "yellow", "gray", "magenta", "orange",
    "skyblue", "darkred", "purple", "navy", "white", "seagreen", "cyan", "limegreen",
]

Grid = list  # list[list[int]]


def _name(c: int) -> str:
    return COLOR_NAMES[c % 16] if 0 <= c < 16 else f"c{c}"


def _components(grid: Grid, color: int) -> list[dict]:
    """4-connected regions of `color`. Returns bbox, size, centroid per region."""
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    out: list[dict] = []
    for sy in range(h):
        for sx in range(w):
            if grid[sy][sx] != color or seen[sy][sx]:
                continue
            q = deque([(sy, sx)])
            seen[sy][sx] = True
            cells = 0
            x0 = x1 = sx
            y0 = y1 = sy
            sumx = sumy = 0
            while q:
                y, x = q.popleft()
                cells += 1
                sumx += x
                sumy += y
                x0, x1 = min(x0, x), max(x1, x)
                y0, y1 = min(y0, y), max(y1, y)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        q.append((ny, nx))
            out.append({
                "size": cells,
                "bbox": (x0, y0, x1, y1),
                "centroid": (round(sumx / cells), round(sumy / cells)),
                "wh": (x1 - x0 + 1, y1 - y0 + 1),
            })
    out.sort(key=lambda r: -r["size"])
    return out


def _shape(comp: dict) -> str:
    w, h = comp["wh"]
    if comp["size"] == w * h:
        return f"{w}x{h} block" if (w > 1 or h > 1) else "1 cell"
    return f"{comp['size']} cells in {w}x{h}"


def describe_grid(grid: Grid, *, max_objs: int = 10) -> str:
    """One line per color group: the objects on the board, biggest first."""
    h, w = len(grid), len(grid[0])
    counts = Counter(c for row in grid for c in row)
    bg, _ = counts.most_common(1)[0]
    lines = [f"{w}x{h} grid, background={_name(bg)}"]
    shown = 0
    for color, total in counts.most_common():
        if color == bg:
            continue
        comps = _components(grid, color)
        n = len(comps)
        if n == 1:
            c = comps[0]
            lines.append(f"  {_name(color)}: {_shape(c)} at {c['centroid']}")
        else:
            head = comps[: min(4, n)]
            locs = "; ".join(f"{_shape(c)}@{c['centroid']}" for c in head)
            tail = f" (+{n - len(head)} more)" if n > len(head) else ""
            lines.append(f"  {_name(color)}: {total} cells in {n} objects: {locs}{tail}")
        shown += 1
        if shown >= max_objs:
            lines.append(f"  ... (+{sum(1 for c in counts if c != bg) - shown} more colors)")
            break
    return "\n".join(lines)


def describe_delta(prev: Grid, grid: Grid, *, max_groups: int = 8) -> str:
    """What changed vs the previous frame, grouped by (old->new) color."""
    h, w = len(grid), len(grid[0])
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y in range(h):
        pr, gr = prev[y], grid[y]
        for x in range(w):
            if pr[x] != gr[x]:
                groups.setdefault((pr[x], gr[x]), []).append((x, y))
    if not groups:
        return "no change since last frame"
    parts = []
    for (a, b), cells in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:max_groups]:
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        parts.append(f"{len(cells)} cells {_name(a)}->{_name(b)} in bbox{bbox}")
    extra = len(groups) - min(len(groups), max_groups)
    tail = f" (+{extra} more change-groups)" if extra > 0 else ""
    return "changed: " + "; ".join(parts) + tail


def describe_scene(
    grid: Grid,
    prev: Grid | None = None,
    *,
    state: str | None = None,
    levels: int | None = None,
    win_levels: int | None = None,
    available: list[int] | None = None,
    action: str | None = None,
) -> str:
    """Full scene description: objects + delta + game meta. The memory query."""
    blocks = [describe_grid(grid)]
    if prev is not None:
        blocks.append(describe_delta(prev, grid))
    meta = []
    if state is not None:
        meta.append(f"state={state}")
    if levels is not None:
        meta.append(f"levels={levels}/{win_levels if win_levels is not None else '?'}")
    if available is not None:
        meta.append("available=[" + ",".join(f"ACTION{a}" for a in available) + "]")
    if action is not None:
        meta.append(f"last_action={action}")
    if meta:
        blocks.append(" | ".join(meta))
    return "\n".join(blocks)


# ─── recording adapter (for offline dataset builds / testing) ───────────

def frames_from_recording(path) -> list[dict]:
    """Parse an official .recording.jsonl into [{grid, state, levels, ...}]."""
    import json
    from pathlib import Path

    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line).get("data")
        except Exception:
            continue
        if not isinstance(d, dict) or "frame" not in d:
            continue
        fr = d["frame"]
        grid = fr[-1] if (isinstance(fr, list) and fr and isinstance(fr[0], list)) else fr
        if not (isinstance(grid, list) and grid and isinstance(grid[0], list)):
            continue
        out.append({
            "grid": grid,
            "state": d.get("state"),
            "levels": d.get("levels_completed"),
            "win_levels": d.get("win_levels"),
            "available": d.get("available_actions"),
        })
    return out


def scene_sequence(frames: list[dict]) -> list[str]:
    """Scene descriptions for a trajectory, each with delta vs the prior frame."""
    out = []
    prev = None
    for f in frames:
        out.append(describe_scene(
            f["grid"], prev,
            state=f.get("state"), levels=f.get("levels"),
            win_levels=f.get("win_levels"), available=f.get("available"),
        ))
        prev = f["grid"]
    return out


def _to_color_int(c) -> int | None:
    """Accept a color int or a name ('green', 'yellow', ...)."""
    if isinstance(c, int):
        return c
    s = str(c).strip().lower()
    if s.isdigit():
        return int(s)
    return COLOR_NAMES.index(s) if s in COLOR_NAMES else None


def _nearest_walkable(grid, wset, x, y, R: int = 10):
    """Nearest cell to (x,y) whose color is walkable (player/target usually sit
    on top of floor, so we snap to the adjacent floor)."""
    from collections import deque

    h, w = len(grid), len(grid[0])
    q, seen = deque([(x, y)]), {(x, y)}
    while q:
        cx, cy = q.popleft()
        if 0 <= cx < w and 0 <= cy < h and grid[cy][cx] in wset:
            return (cx, cy)
        if abs(cx - x) + abs(cy - y) > R:
            continue
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nb = (cx + dx, cy + dy)
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return (x, y)


def bfs_path(grid, walkable, start, goal, max_nodes: int = 40000):
    """Shortest 4-connected path over cells whose color is in `walkable`.
    Player/target usually sit on floor, so start & goal are snapped to the
    nearest walkable cell first. `walkable` = a color int/name or a list.
    Returns a list of (x, y) or None. y is row (down), x is col (right)."""
    from collections import deque

    h, w = len(grid), len(grid[0])
    if isinstance(walkable, (list, tuple, set)):
        wset = {_to_color_int(c) for c in walkable}
    else:
        wset = {_to_color_int(walkable)}
    wset.discard(None)
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    if 0 <= sy < h and 0 <= sx < w and grid[sy][sx] not in wset:
        sx, sy = _nearest_walkable(grid, wset, sx, sy)
    if 0 <= gy < h and 0 <= gx < w and grid[gy][gx] not in wset:
        gx, gy = _nearest_walkable(grid, wset, gx, gy)

    def ok(x, y):
        return 0 <= x < w and 0 <= y < h and (grid[y][x] in wset or (x, y) == (gx, gy))

    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    q = deque([(sx, sy)])
    prev = {(sx, sy): None}
    n = 0
    while q and n < max_nodes:
        x, y = q.popleft()
        n += 1
        if (x, y) == (gx, gy):
            path, cur = [], (x, y)
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if (nx, ny) not in prev and ok(nx, ny):
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
    return None


def path_segments(path) -> list[tuple[str, int]]:
    """Compress a cell path to (direction, cells) segments (up/down/left/right)."""
    if not path or len(path) < 2:
        return []

    def d(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if dx:
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"

    segs, cur, cnt = [], d(path[0], path[1]), 1
    for i in range(1, len(path) - 1):
        nd = d(path[i], path[i + 1])
        if nd == cur:
            cnt += 1
        else:
            segs.append((cur, cnt))
            cur, cnt = nd, 1
    segs.append((cur, cnt))
    return segs


_ASCII = {  # color int -> single char for the ASCII map
    None: "?",
}


def ascii_map(grid, downsample: int = 2) -> str:
    """Compact ASCII of the board: each color -> a distinct char, optionally
    block-downsampled (each downsample x downsample cell -> majority color).
    Returns the map plus a legend so the agent can read the maze as a grid."""
    from collections import Counter

    h, w = len(grid), len(grid[0])
    ds = max(1, downsample)
    chars = {}
    nxt = [ord("a")]

    def ch(c):
        if c not in chars:
            chars[c] = _name(c)[0].upper() if _name(c)[0].isalpha() else "#"
            # ensure uniqueness
            used = set(chars.values())
            base = chars[c]
            if list(chars.values()).count(base) > 1:
                while chr(nxt[0]) in used:
                    nxt[0] += 1
                chars[c] = chr(nxt[0])
                nxt[0] += 1
        return chars[c]

    rows = []
    for by in range(0, h, ds):
        line = []
        for bx in range(0, w, ds):
            block = [grid[y][x] for y in range(by, min(by + ds, h))
                     for x in range(bx, min(bx + ds, w))]
            line.append(ch(Counter(block).most_common(1)[0][0]))
        rows.append("".join(line))
    legend = ", ".join(f"{chars[c]}={_name(c)}" for c in sorted(chars))
    return f"ASCII map ({w // ds}x{h // ds}, each char = {ds}x{ds} cells)\nlegend: {legend}\n" + "\n".join(rows)


if __name__ == "__main__":  # quick smoke test on a real recording
    import sys
    from pathlib import Path

    recs = sorted((Path(__file__).resolve().parent / "recordings").glob("*.jsonl"))
    if not recs:
        raise SystemExit("no recordings to test on")
    path = recs[0] if len(sys.argv) < 2 else sys.argv[1]
    frames = frames_from_recording(path)
    print(f"recording: {Path(path).name}  ({len(frames)} frames)\n")
    # Show the opening scene and the first few frames that actually changed.
    print("=== frame 0 (opening) ===")
    print(describe_scene(frames[0]["grid"], None, state=frames[0]["state"],
                         levels=frames[0]["levels"], win_levels=frames[0]["win_levels"],
                         available=frames[0]["available"]))
    shown = 0
    for i in range(1, len(frames)):
        d = describe_delta(frames[i - 1]["grid"], frames[i]["grid"])
        if d != "no change since last frame":
            print(f"\n=== frame {i} delta ===")
            print(d)
            shown += 1
            if shown >= 4:
                break
    changed = sum(
        1 for i in range(1, len(frames))
        if describe_delta(frames[i - 1]["grid"], frames[i]["grid"]) != "no change since last frame"
    )
    print(f"\n{changed}/{len(frames) - 1} frame-steps changed the grid")
