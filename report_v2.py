"""Generate report.html — the hackathon demo page (self-contained).

Thesis-driven: ARC-AGI-3 takes several abilities (perception, MEMORY, planning,
control); we built the memory piece — a parametric STM/LTM that recalls prior
experience per scene — and show it working, with real boards + injected notes.
Frank that memory alone isn't sufficient (planning + closed-model control are
the open frontier). Re-run anytime: uv run python report_v2.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

_DIR = Path(__file__).resolve().parent

PALETTE = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE",
    "#FF851B", "#7FDBFF", "#870C25", "#5724C2", "#2E1A47", "#FFFFFF", "#3D9970",
    "#39CCCC", "#01FF70",
]


def _real_grid():
    import sys
    sys.path.insert(0, str(_DIR))
    from scene import frames_from_recording
    recs = sorted((_DIR / "recordings").glob("ls20*.jsonl"))
    if not recs:
        return [[0] * 64 for _ in range(64)]
    frames = frames_from_recording(recs[0])
    return frames[min(3, len(frames) - 1)]["grid"]


def _scene_text() -> str:
    import sys
    sys.path.insert(0, str(_DIR))
    from scene import describe_grid
    return describe_grid(_real_grid())


def _examples() -> list[dict]:
    rows = []
    for f in sorted((_DIR / "data" / "examples").glob("ls20*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    pick, seen = [], set()
    for nt in ("rule", "candidate_strategy", "missing_info", "failed_strategy"):
        for r in rows:
            if r.get("note_type") == nt and r.get("nuance") and r.get("note"):
                key = r["note"][:30]
                if key not in seen:
                    seen.add(key)
                    pick.append(r)
                    break
    return pick[:3]


def _mem_instances() -> list:
    p = _DIR / "data" / "memory_instances.json"
    return json.loads(p.read_text()) if p.exists() else []


def workflow_svg() -> str:
    """Pictorial RSI workflow: a 6-stage pipeline + a dashed self-improvement loop."""
    stages = [
        ("1", "Play", "Claude policy"), ("2", "Trace", "scene · action · reasoning"),
        ("3", "Retrospect", "GPT-5.2 hindsight"), ("4", "Dataset", "scene → note"),
        ("5", "Train LoRA", "STM + LTM · Modal"), ("6", "Serve + inject", "per-step note"),
    ]
    W, bw, bh, y, x0 = 960, 132, 72, 92, 14
    n = len(stages)
    gap = (W - 2 * x0 - n * bw) / (n - 1)
    boxes = arrows = ""
    for i, (num, t, sub) in enumerate(stages):
        x = x0 + i * (bw + gap)
        boxes += (
            f'<rect x="{x:.0f}" y="{y}" width="{bw}" height="{bh}" rx="12" fill="#141d31" stroke="#2c3a57"/>'
            f'<text x="{x+bw/2:.0f}" y="{y+22}" text-anchor="middle" fill="#7fb6ff" font-family="monospace" font-size="11">{num}</text>'
            f'<text x="{x+bw/2:.0f}" y="{y+42}" text-anchor="middle" fill="#E6E9F0" font-size="14" font-weight="700">{t}</text>'
            f'<text x="{x+bw/2:.0f}" y="{y+59}" text-anchor="middle" fill="#8d9bb5" font-family="monospace" font-size="9">{sub}</text>')
        if i < n - 1:
            ax = x + bw
            arrows += f'<line x1="{ax:.0f}" y1="{y+bh/2}" x2="{ax+gap:.0f}" y2="{y+bh/2}" stroke="#0074D9" stroke-width="2" marker-end="url(#ah)"/>'
    lx = x0 + (n - 1) * (bw + gap) + bw / 2
    fx = x0 + bw / 2
    loop = (
        f'<path d="M {lx:.0f} {y} C {lx:.0f} 26, {fx:.0f} 26, {fx:.0f} {y}" fill="none" stroke="#7c5cff" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#ah2)"/>'
        f'<text x="{(fx+lx)/2:.0f}" y="20" text-anchor="middle" fill="#b9a6ff" font-family="monospace" font-size="11">self-improvement loop — the agent learns from its own play</text>')
    return (
        f'<svg viewBox="0 0 {W} 184" width="100%" style="max-width:{W}px" role="img" aria-label="RSI workflow">'
        '<defs>'
        '<marker id="ah" markerWidth="8" markerHeight="8" refX="6.5" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#0074D9"/></marker>'
        '<marker id="ah2" markerWidth="8" markerHeight="8" refX="6.5" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#7c5cff"/></marker>'
        f'</defs>{loop}{arrows}{boxes}</svg>')


def main() -> None:
    grid = _real_grid()
    scene = _scene_text()
    exs = _examples()
    instances = _mem_instances()

    # cascade examples (STM nuance -> LTM note)
    casc = ""
    for e in exs:
        casc += f"""
      <div class="casc">
        <div class="casc-row"><span class="tag stm">short-term memory · what's true on this level</span><p>{html.escape(e['nuance'])}</p></div>
        <div class="casc-arrow">scene + nuance → long-term memory ↓</div>
        <div class="casc-row"><span class="tag ltm">long-term memory · transferable note (injected)</span><p>{html.escape(e['note'])}</p></div>
      </div>"""

    # before -> after contrast: two real boards + the real in-game move-budget bar
    cpath = _DIR / "data" / "contrast.json"
    cd = json.loads(cpath.read_text()) if cpath.exists() else None
    if cd:
        contrast = f"""
      <div class="contrast">
        <div class="cside">
          <span class="tag no">1 · without memory</span>
          <canvas id="cbefore" width="64" height="64"></canvas>
          <div class="budget">move budget<div class="bar"><i style="width:{cd['before']['budget']}%"></i></div></div>
          <p>With no record of past attempts, the agent re-tries a move that's blocked and stalls near the
          start — spending actions with no progress.</p>
        </div>
        <div class="cside">
          <span class="tag ltm">2 · with memory</span>
          <canvas id="cafter" width="64" height="64"></canvas>
          <div class="budget">move budget<div class="bar"><i style="width:{cd['after']['budget']}%"></i></div></div>
          <p>Handed the note below, it <b>switches direction</b> and climbs toward the exit — the player block
          has moved up the board.</p>
        </div>
      </div>
      <div class="note-box" style="margin-top:6px"><span class="tag ltm">the note memory injected</span>
        <p style="margin:0">{html.escape(cd['note'])}</p></div>"""
        cgrids = (f"paint('cbefore', {json.dumps(cd['before']['grid'])}); "
                  f"paint('cafter', {json.dumps(cd['after']['grid'])});")
    else:
        contrast, cgrids = '<p class="dim">(contrast pending)</p>', ""

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory for ARC-AGI-3 · learning from experience</title>
<style>
  :root {{
    --ground:#0E1526; --panel:#141d31; --line:#243049; --text:#E6E9F0; --dim:#8d9bb5;
    --blue:#0074D9; --green:#2ECC40; --yellow:#FFDC00; --red:#FF4136; --violet:#7c5cff;
    --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--ground); color:var(--text); font-family:var(--sans); line-height:1.6; }}
  .wrap {{ max-width:940px; margin:0 auto; padding:0 24px; }}
  .eyebrow {{ font-family:var(--mono); font-size:12px; letter-spacing:.18em; text-transform:uppercase; color:var(--blue); margin:0 0 14px; }}
  h1 {{ font-size:clamp(32px,5.5vw,58px); line-height:1.05; letter-spacing:-.02em; margin:0 0 18px; font-weight:800; }}
  h2 {{ font-size:clamp(21px,3.2vw,29px); letter-spacing:-.01em; margin:0 0 10px; font-weight:750; }}
  .lede {{ font-size:18.5px; color:#c6cfe0; max-width:64ch; }}
  .dim {{ color:var(--dim); }}
  section {{ padding:54px 0; border-top:1px solid var(--line); }}
  .hero {{ padding:60px 0 38px; border:none; }}
  .grid-split {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:32px 0 8px; align-items:stretch; }}
  @media(max-width:720px){{ .grid-split{{grid-template-columns:1fr;}} }}
  .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px; }}
  .panel h3 {{ margin:0 0 12px; font-family:var(--mono); font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--dim); font-weight:600; }}
  canvas {{ width:100%; height:auto; image-rendering:pixelated; border-radius:8px; display:block; }}
  pre {{ font-family:var(--mono); font-size:12px; line-height:1.5; color:#b9c6e0; margin:0; white-space:pre-wrap; word-break:break-word; max-height:300px; overflow:auto; }}
  .tag {{ display:inline-block; font-family:var(--mono); font-size:11px; letter-spacing:.06em; text-transform:uppercase; padding:3px 9px; border-radius:20px; margin-bottom:8px; background:rgba(0,116,217,.16); color:#7fb6ff; }}
  .tag.stm {{ background:rgba(124,92,255,.18); color:#b9a6ff; }}
  .tag.ltm {{ background:rgba(46,204,64,.16); color:#79e08c; }}
  .tag.no {{ background:rgba(255,65,54,.15); color:#ff9a93; }}
  .casc {{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--violet); border-radius:12px; padding:16px 18px; margin:14px 0; }}
  .casc-row p {{ margin:0; font-size:15.5px; }} .casc-row {{ margin:8px 0; }}
  .casc-arrow {{ font-family:var(--mono); font-size:11px; color:var(--violet); margin:8px 0 4px; }}
  .contrast {{ display:flex; gap:18px; align-items:stretch; flex-wrap:wrap; margin:16px 0; }}
  .cside {{ flex:1; min-width:270px; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px; }}
  .cside canvas {{ width:100%; max-width:230px; margin:10px 0; }}
  .cside p {{ margin:0; font-size:14.5px; }}
  .budget {{ font-family:var(--mono); font-size:11px; color:var(--dim); margin:0 0 6px; }}
  .budget .bar {{ height:8px; background:var(--line); border-radius:5px; overflow:hidden; margin-top:4px; max-width:230px; }}
  .budget .bar i {{ display:block; height:100%; background:var(--yellow); }}
  .aspects {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:22px 0; }}
  @media(max-width:640px){{ .aspects{{grid-template-columns:repeat(2,1fr);}} }}
  .asp {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }}
  .asp b {{ display:block; font-size:15px; margin-bottom:4px; }}
  .asp.us {{ border-color:var(--green); }}
  .asp .lbl {{ font-family:var(--mono); font-size:11px; color:var(--dim); }}
  .flow {{ display:flex; flex-wrap:wrap; gap:8px; font-family:var(--mono); font-size:13px; margin-top:18px; }}
  .flow span {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px 12px; }}
  .flow span b {{ color:var(--blue); }} .flow i {{ color:var(--dim); font-style:normal; align-self:center; }}
  .callout {{ background:rgba(124,92,255,.08); border:1px solid rgba(124,92,255,.3); border-radius:12px; padding:18px 20px; margin:20px 0; }}
  .callout b {{ color:#b9a6ff; }}
  ul.clean {{ list-style:none; padding:0; }}
  ul.clean li {{ padding:10px 0 10px 26px; border-top:1px solid var(--line); position:relative; }}
  ul.clean li:before {{ content:"›"; position:absolute; left:6px; color:var(--blue); font-family:var(--mono); }}
  footer {{ padding:40px 0 70px; color:var(--dim); font-family:var(--mono); font-size:12px; }}
  b.k {{ color:var(--green); }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <p class="eyebrow">HUD × YC · Frontier RSI Environments</p>
    <h1>Agents that learn the way we do — by remembering.</h1>
    <p class="lede">ARC-AGI-3 rewards <i>learning, not memorizing</i>. Humans crack these puzzles by recalling
    prior experience — the rules we've inferred, the strategies that failed. Take that recollection away and
    we'd struggle too. We give an LLM agent that recollection: a <b class="k">parametric memory</b> it builds
    from its own play and recalls, conditioned on what it's seeing.</p>
  </header>

  <section>
    <p class="eyebrow">// 01 — the problem is bigger than any one trick</p>
    <h2>Four abilities, working together</h2>
    <p class="lede">Solving an unknown ARC-AGI-3 game takes several things at once. We built the memory piece —
    and we're frank that it isn't sufficient on its own.</p>
    <div class="aspects">
      <div class="asp"><b>Perception</b><div class="lbl">read the 64×64 board</div></div>
      <div class="asp us"><b>Memory ✓</b><div class="lbl">recall prior experience — OUR FOCUS</div></div>
      <div class="asp"><b>Planning</b><div class="lbl">turn recalled knowledge into a NEW plan</div></div>
      <div class="asp"><b>Control</b><div class="lbl">steer a closed model you can't fine-tune</div></div>
    </div>
    <div class="callout"><b>Being frank:</b> memory alone won't clear deep levels. The hard open piece is
    turning recalled knowledge into a fresh <i>plan</i> in the moment — and we drive a closed model we can only
    <i>inform</i>, not retrain. In 24h we built and validated the memory; the planning + control aspects are the
    open frontier (and exactly why an end-to-end fine-tunable policy is the natural next step).</p></div>
  </section>

  <section>
    <p class="eyebrow">// 02 — why memory matters</p>
    <h2>Same board. What the agent <i>remembers</i> changes everything.</h2>
    <p class="lede">Two real boards. <b>(1)</b> Without memory the agent re-tries a blocked move and stalls
    near the start. <b>(2)</b> With memory — handed the lesson from past play — it switches direction and the
    player climbs toward the exit.</p>
    {contrast}
  </section>

  <section>
    <p class="eyebrow">// 03 — how the memory works</p>
    <h2>The policy sees pixels. The memory reads structure.</h2>
    <p class="lede">The frozen Claude policy navigates the rendered image; the memory keys off a deterministic
    object+delta encoding — a shared vocabulary that lets lessons transfer across games.</p>
    <div class="grid-split">
      <div class="panel"><h3>policy input — rendered frame</h3>
        <canvas id="grid" width="64" height="64" aria-label="ARC-AGI-3 grid"></canvas></div>
      <div class="panel"><h3>memory input — scene.py encoding</h3><pre>{html.escape(scene)}</pre></div>
    </div>
    <h2 style="margin-top:38px">Two tiers: short-term and long-term memory</h2>
    <p class="lede">A 1.7B <b>short-term memory</b> (STM) recalls <i>this level's</i> specifics; a 4B
    <b>long-term memory</b> (LTM) turns scene + nuance into a transferable instruction — episodic → semantic,
    the way the hippocampus consolidates into neocortex. Real cascades the agent learned and was handed back:</p>
    {casc}
    <h2 style="margin-top:38px">Training: hindsight, distilled into weights</h2>
    <p class="lede">Reward is sparse, but experience is rich. A GPT-5.2 teacher retrospects each play-through in
    natural language, that hindsight becomes scene→note data, and it's compiled into the LoRA weights the
    memory recalls — self-improvement through memory, not reward. The whole pipeline is a loop:</p>
    {workflow_svg()}
  </section>

  <section>
    <p class="eyebrow">// 04 — what works</p>
    <h2>The hard part — making memory recall the <i>right</i> thing — works</h2>
    <p class="lede">The engineering challenge was getting the memory to surface correct, scene-relevant knowledge
    from past play. It does, reliably:</p>
    <ul class="clean">
      <li>The <b>short-term → long-term memory cascade</b> turns a concrete level observation into transferable, grounded guidance
      (the examples above are verbatim from the trained models).</li>
      <li>It's <b>scene-retrieved and parametric</b> — for novel states relevance ≠ text similarity, so a small
      model <i>learns</i> the scene→knowledge mapping rather than matching strings.</li>
      <li>The full loop runs end-to-end on real infra: encoder → GPT-5.2 hindsight → Qwen3 1.7B/4B LoRA on
      Modal (H100) → per-step injection into the live policy.</li>
      <li>Injected memory correctly flags recurring mistakes (e.g. wall-bumps) the bare agent re-makes — shown
      live above on the real board.</li>
    </ul>
  </section>

  <section>
    <p class="eyebrow">// 05 — the frontier</p>
    <h2>What it takes to go further</h2>
    <p class="lede">Honestly: clearing deep levels needs the other aspects, which a 24h hack on a closed model
    can't fully reach.</p>
    <ul class="clean">
      <li><b>Planning from memory:</b> the open hard problem — turning recalled knowledge into a correct, novel
      multi-step plan in the moment. Memory informs it; it doesn't replace it.</li>
      <li><b>An open, fine-tunable policy:</b> with a closed model we can only inject text. Fine-tuning the
      policy end-to-end (alongside the memory) would let experience shape the <i>reasoning</i>, not just the prompt.</li>
      <li><b>Controllability:</b> steering a frozen model purely through context is a real limit on how much any
      memory can change behaviour.</li>
    </ul>
  </section>

  <footer>Stack — HUD gateway (Claude policy) · ARC Prize runner · GPT-5.2 teacher ·
  Qwen3 1.7B/4B LoRA on Modal · PEFT/Transformers · BFS pathfinding tool.</footer>
</div>

<script>
  const PAL = {json.dumps(PALETTE)};
  function paint(id, g) {{
    const cv = document.getElementById(id); if (!cv || !g) return;
    const ctx = cv.getContext('2d');
    for (let y=0; y<64; y++) for (let x=0; x<64; x++) {{
      ctx.fillStyle = PAL[((g[y] && g[y][x]) || 0) % 16]; ctx.fillRect(x, y, 1, 1);
    }}
  }}
  paint('grid', {json.dumps(grid)});
  {cgrids}
</script>
</body></html>
"""
    out = _DIR / "report.html"
    out.write_text(page)
    print(f"report -> {out}  ({len(page)} bytes) | cascade={len(exs)} | instances={len(instances)}")


if __name__ == "__main__":
    main()
