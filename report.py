"""Render a self-contained HTML report for the self-improvement demo.

Reads runs/baseline_<game>.json and runs/memory_<game>.json (written by
self_improve.py) and produces report.html: a learning-curve chart (baseline
vs memory) plus the agent's rulebook evolving attempt by attempt.

    uv run python report.py ls20

No third-party deps; the chart is hand-built inline SVG so the page is one
portable file (good for the demo / submission).
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

_ENV_DIR = Path(__file__).resolve().parent
_W, _H, _PAD = 760, 340, 48
_BASE, _MEM, _GRID, _INK, _BG = "#8a8f98", "#5b8cff", "#262a33", "#e6e8ec", "#0f1115"


def _load(game: str, kind: str) -> dict | None:
    p = _ENV_DIR / "runs" / f"{kind}_{game}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _xy(i: int, n: int, v: float) -> tuple[float, float]:
    x = _PAD + (i * (_W - 2 * _PAD) / max(1, n - 1))
    y = (_H - _PAD) - v * (_H - 2 * _PAD)  # v in 0..1
    return x, y


def _polyline(curve: list[float], color: str, dash: bool = False) -> str:
    n = len(curve)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (_xy(i, n, v) for i, v in enumerate(curve)))
    da = ' stroke-dasharray="6 5"' if dash else ""
    line = f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"{da}/>'
    if dash:
        return line
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
        for x, y in (_xy(i, n, v) for i, v in enumerate(curve))
    )
    return line + dots


def _chart(base: dict | None, mem: dict | None) -> str:
    n = max(len(base["curve"]) if base else 0, len(mem["curve"]) if mem else 0, 2)
    grid = ""
    for k in range(5):  # y gridlines 0, .25, .5, .75, 1
        v = k / 4
        _, y = _xy(0, n, v)
        grid += (
            f'<line x1="{_PAD}" y1="{y:.1f}" x2="{_W-_PAD}" y2="{y:.1f}" stroke="{_GRID}"/>'
            f'<text x="{_PAD-10}" y="{y+4:.1f}" fill="{_BASE}" font-size="12" text-anchor="end">{v:.2f}</text>'
        )
    xlabels = ""
    for i in range(n):
        x, _ = _xy(i, n, 0)
        xlabels += f'<text x="{x:.1f}" y="{_H-_PAD+22:.1f}" fill="{_BASE}" font-size="12" text-anchor="middle">{i+1}</text>'
    lines = ""
    if base:
        lines += _polyline(base["curve"], _BASE)
    if mem:
        cummax, m = [], 0.0
        for v in mem["curve"]:
            m = max(m, v)
            cummax.append(m)
        lines += _polyline(cummax, "#9b8cff", dash=True)  # best-so-far (ratchet)
        lines += _polyline(mem["curve"], _MEM)
    return (
        f'<svg viewBox="0 0 {_W} {_H}" width="100%" style="max-width:{_W}px">'
        f"{grid}{xlabels}{lines}"
        f'<text x="{_W/2}" y="{_H-8}" fill="{_BASE}" font-size="13" text-anchor="middle">attempt #</text>'
        f"</svg>"
    )


def _rulebook_blocks(mem: dict | None) -> str:
    if not mem:
        return '<p class="muted">No memory run yet. Run: <code>ARC_MEMORY=1 uv run python self_improve.py &lt;game&gt;</code></p>'
    out = []
    prev = ""
    for i, rb in enumerate(mem["rulebooks"], 1):
        rb = rb or ""
        tag = "new" if rb and rb != prev else ("same" if rb else "empty")
        out.append(
            f'<div class="rb"><div class="rb-h">after attempt {i} '
            f'<span class="pill {tag}">{len(rb)} chars</span></div>'
            f'<pre>{html.escape(rb) or "(no notes yet)"}</pre></div>'
        )
        prev = rb
    return "\n".join(out)


def _stat(d: dict | None) -> str:
    if not d:
        return "—"
    c = d["curve"]
    return f"best {max(c):.3f} · final {c[-1]:.3f} · mean {sum(c)/len(c):.3f}"


def main() -> None:
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    base, mem = _load(game, "baseline"), _load(game, "memory")
    if not base and not mem:
        raise SystemExit(f"no runs for '{game}' in runs/. Run self_improve.py first.")

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>ARC-AGI-3 · learning from natural-language feedback</title>
<style>
 body{{background:{_BG};color:{_INK};font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:40px}}
 .wrap{{max-width:860px;margin:0 auto}}
 h1{{font-size:24px;margin:0 0 4px}} .sub{{color:{_BASE};margin:0 0 28px}}
 .card{{background:#161922;border:1px solid {_GRID};border-radius:14px;padding:22px;margin:0 0 22px}}
 .legend span{{display:inline-flex;align-items:center;gap:7px;margin-right:20px;font-size:13px}}
 .dot{{width:12px;height:12px;border-radius:50%;display:inline-block}}
 .stats{{display:flex;gap:24px;margin-top:10px;font-size:13px;color:{_BASE}}}
 .stats b{{color:{_INK}}}
 .rb{{border-top:1px solid {_GRID};padding:14px 0}} .rb:first-child{{border-top:none}}
 .rb-h{{font-size:13px;color:{_BASE};margin-bottom:8px}}
 pre{{background:#0b0d12;border:1px solid {_GRID};border-radius:8px;padding:12px;white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace;margin:0}}
 .pill{{font-size:11px;padding:2px 8px;border-radius:20px;background:{_GRID};color:{_INK}}}
 .pill.new{{background:#1d3a6b;color:#cfe0ff}} .pill.empty{{opacity:.5}}
 code{{background:{_GRID};padding:1px 6px;border-radius:5px;font-size:13px}} .muted{{color:{_BASE}}}
</style></head><body><div class="wrap">
 <h1>Agents that learn from natural-language feedback</h1>
 <p class="sub">ARC-AGI-3 · game <code>{html.escape(game)}</code> · the agent writes its own rulebook and carries it forward across attempts</p>
 <div class="card">
   <div class="legend"><span><i class="dot" style="background:{_MEM}"></i>memory (per attempt)</span>
   <span><i class="dot" style="background:#9b8cff"></i>memory best-so-far</span>
   <span><i class="dot" style="background:{_BASE}"></i>baseline (no memory)</span></div>
   {_chart(base, mem)}
   <div class="stats"><div>memory: <b>{_stat(mem)}</b></div><div>baseline: <b>{_stat(base)}</b></div></div>
 </div>
 <div class="card"><h2 style="margin-top:0;font-size:17px">The rulebook, evolving</h2>{_rulebook_blocks(mem)}</div>
</div></body></html>"""

    out = _ENV_DIR / "report.html"
    out.write_text(page)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
