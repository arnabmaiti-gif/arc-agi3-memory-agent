"""Generate slides.html — a self-contained, keyboard-navigable deck for the pitch.

Reuses the demo's data (real board, cascade example, board->note instance).
    uv run python slides.py    ->    slides.html   (publish as an Artifact)
Navigate with ← / → / Space; click the dots.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from report_v2 import PALETTE, _examples, _mem_instances, _real_grid, workflow_svg

_DIR = Path(__file__).resolve().parent


def main() -> None:
    grid = _real_grid()
    exs = _examples()
    inst = _mem_instances()
    casc = exs[0] if exs else {"nuance": "—", "note": "—"}
    cpath = _DIR / "data" / "contrast.json"
    cd = json.loads(cpath.read_text()) if cpath.exists() else None
    before_grid = json.dumps(cd["before"]["grid"]) if cd else json.dumps(grid)
    after_grid = json.dumps(cd["after"]["grid"]) if cd else json.dumps(grid)
    note = html.escape(cd["note"]) if cd else "if a move only changes non-player tiles, it's a wall — switch direction."
    before_budget = cd["before"]["budget"] if cd else 16
    after_budget = cd["after"]["budget"] if cd else 88

    def budget_bar(pct: int) -> str:
        return (f'<div style="font-family:var(--mono);font-size:12px;color:var(--dim);max-width:200px;margin:6px auto 0">'
                f'move budget left'
                f'<div style="height:9px;background:var(--line);border-radius:5px;overflow:hidden;margin-top:4px">'
                f'<i style="display:block;height:100%;width:{pct}%;background:var(--yellow)"></i></div></div>')

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory for ARC-AGI-3 — pitch</title>
<style>
  :root {{
    --ground:#0E1526; --panel:#141d31; --line:#243049; --text:#E6E9F0; --dim:#8d9bb5;
    --blue:#0074D9; --green:#2ECC40; --yellow:#FFDC00; --red:#FF4136; --violet:#7c5cff;
    --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
  }}
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; height:100%; background:var(--ground); color:var(--text);
    font-family:var(--sans); overflow:hidden; }}
  .slide {{ position:fixed; inset:0; display:none; flex-direction:column; justify-content:center;
    padding:7vh 9vw; animation:fade .35s ease; }}
  .slide.on {{ display:flex; }}
  @keyframes fade {{ from{{opacity:0; transform:translateY(8px)}} to{{opacity:1; transform:none}} }}
  .eyebrow {{ font-family:var(--mono); font-size:13px; letter-spacing:.2em; text-transform:uppercase; color:var(--blue); margin:0 0 18px; }}
  h1 {{ font-size:clamp(34px,6vw,68px); line-height:1.04; letter-spacing:-.02em; margin:0 0 18px; font-weight:800; }}
  h2 {{ font-size:clamp(26px,4vw,44px); line-height:1.08; letter-spacing:-.015em; margin:0 0 18px; font-weight:780; }}
  p.big {{ font-size:clamp(17px,2.1vw,23px); color:#c6cfe0; max-width:30ch; }}
  .sub {{ font-size:clamp(16px,2vw,21px); color:var(--dim); max-width:60ch; }}
  .row {{ display:flex; gap:28px; align-items:center; flex-wrap:wrap; }}
  .col {{ flex:1; min-width:260px; }}
  .grid4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:10px; max-width:880px; }}
  @media(max-width:760px){{ .grid4{{grid-template-columns:repeat(2,1fr);}} }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px; }}
  .card.us {{ border-color:var(--green); }}
  .card b {{ display:block; font-size:18px; margin-bottom:6px; }}
  .card .l {{ font-family:var(--mono); font-size:12px; color:var(--dim); }}
  canvas {{ image-rendering:pixelated; border-radius:10px; border:1px solid var(--line); width:300px; height:300px; }}
  pre {{ font-family:var(--mono); font-size:13px; color:#b9c6e0; background:var(--panel); border:1px solid var(--line);
    border-radius:12px; padding:16px; margin:0; white-space:pre-wrap; }}
  .tag {{ display:inline-block; font-family:var(--mono); font-size:12px; letter-spacing:.06em; text-transform:uppercase;
    padding:4px 10px; border-radius:20px; margin-bottom:8px; }}
  .tag.stm {{ background:rgba(124,92,255,.18); color:#b9a6ff; }}
  .tag.ltm {{ background:rgba(46,204,64,.16); color:#79e08c; }}
  .tag.no {{ background:rgba(255,65,54,.15); color:#ff9a93; }}
  .flow {{ display:flex; flex-wrap:wrap; gap:10px; font-family:var(--mono); font-size:15px; margin-top:8px; }}
  .flow span {{ background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:9px 13px; }}
  .flow span b {{ color:var(--blue); }} .flow i {{ color:var(--dim); font-style:normal; align-self:center; }}
  .note-box {{ border-left:3px solid var(--green); padding-left:14px; }}
  .no-box {{ border-left:3px solid var(--red); padding-left:14px; }}
  .links {{ font-family:var(--mono); font-size:15px; line-height:2; }}
  .links a {{ color:var(--green); }}
  .k {{ color:var(--green); }} .v {{ color:var(--violet); }}
  nav {{ position:fixed; bottom:20px; left:0; right:0; display:flex; justify-content:center; align-items:center; gap:10px; z-index:5; }}
  .dot {{ width:9px; height:9px; border-radius:50%; background:var(--line); border:none; cursor:pointer; padding:0; }}
  .dot.on {{ background:var(--blue); }}
  .count {{ position:fixed; top:22px; right:28px; font-family:var(--mono); font-size:13px; color:var(--dim); }}
  .hint {{ position:fixed; bottom:20px; right:28px; font-family:var(--mono); font-size:12px; color:var(--dim); }}
</style>
</head>
<body>
<div class="count"><span id="cur">1</span> / 6</div>

<section class="slide on">
  <p class="eyebrow">HUD × YC · Frontier Recursive Self-Improvement Environments</p>
  <h1>Agents that learn the way we do —<br>by <span class="v">remembering</span>.</h1>
  <p class="sub">A self-improving <b class="k">parametric memory</b> for ARC-AGI-3. The agent distills its own
  play into weights and recalls the right lesson, conditioned on what it sees — <b>recursive self-improvement</b>
  through memory, not reward.</p>
</section>

<section class="slide">
  <p class="eyebrow">// the problem</p>
  <h2>ARC-AGI-3 rewards <span class="v">learning, not memorizing</span></h2>
  <p class="sub">Frontier models score &lt;1%. Humans crack it by recalling prior experience — inferred rules,
  strategies that failed. Take that recollection away and we'd struggle too. Solving it takes four abilities:</p>
  <div class="grid4">
    <div class="card"><b>Perception</b><div class="l">read the 64×64 board</div></div>
    <div class="card us"><b>Memory ✓</b><div class="l">recall prior experience — OUR FOCUS</div></div>
    <div class="card"><b>Planning</b><div class="l">recalled knowledge → a NEW plan</div></div>
    <div class="card"><b>Control</b><div class="l">steer a closed, frozen policy</div></div>
  </div>
</section>

<section class="slide">
  <p class="eyebrow">// how it works</p>
  <h2>The policy sees pixels. The <span class="v">memory reads structure</span>.</h2>
  <div class="row">
    <canvas id="g1" width="64" height="64"></canvas>
    <div class="col">
      <p class="sub" style="margin-bottom:14px">A frozen Claude policy navigates the image. The memory keys off a
      deterministic object+delta encoding — a shared vocabulary so lessons transfer across games.</p>
      <div class="card">
        <span class="tag stm">short-term memory · 1.7B · this level</span>
        <p style="margin:0 0 12px;font-size:15px">{html.escape(casc['nuance'])}</p>
        <span class="tag ltm">long-term memory · 4B · transferable note</span>
        <p style="margin:0;font-size:15px">{html.escape(casc['note'])}</p>
      </div>
    </div>
  </div>
</section>

<section class="slide">
  <p class="eyebrow">// training</p>
  <h2>Reward is sparse. <span class="v">Experience is rich.</span></h2>
  <p class="sub">A GPT-5.2 teacher retrospects each play-through in natural language — "you re-tried a blocked move;
  here's the rule" — and that hindsight is compiled into LoRA weights the memory recalls.</p>
  <div style="margin-top:30px;max-width:980px">{workflow_svg()}</div>
</section>

<section class="slide">
  <p class="eyebrow">// it works</p>
  <h2>Same level. What it <span class="v">remembers</span> changes the move.</h2>
  <div class="row" style="align-items:flex-end;gap:22px">
    <div style="text-align:center">
      <span class="tag no">1 · without memory</span>
      <canvas id="g2" width="64" height="64" style="display:block;margin:10px auto"></canvas>
      {budget_bar(before_budget)}
      <p class="sub" style="font-size:14px;max-width:230px">re-issues the blocked move; the player never leaves the shaft</p>
    </div>
    <div style="font-size:34px;color:var(--violet);padding-bottom:80px">→</div>
    <div style="text-align:center">
      <span class="tag ltm">2 · with memory</span>
      <canvas id="g3" width="64" height="64" style="display:block;margin:10px auto"></canvas>
      {budget_bar(after_budget)}
      <p class="sub" style="font-size:14px;max-width:230px">switches direction; the player navigates into the room</p>
    </div>
  </div>
  <div class="note-box" style="margin-top:18px;max-width:62ch"><span class="tag ltm">the injected note</span>
    <p style="margin:0;font-size:15px">{note}</p></div>
</section>

<section class="slide">
  <p class="eyebrow">// the frontier — being frank</p>
  <h2>Memory is <span class="k">necessary</span>, not sufficient</h2>
  <p class="sub">We built and validated the memory in 24h. Clearing deep levels also needs <b>planning</b> (turning
  recall into a new plan) and a <b>fine-tunable policy</b> — we can only <i>inform</i> a closed model, not retrain
  its reasoning. That's the next frontier.</p>
  <div class="links" style="margin-top:24px">
    demo &nbsp;<a href="https://arnabmaiti-gif.github.io/arc-agi3-memory-agent/">arnabmaiti-gif.github.io/arc-agi3-memory-agent</a><br>
    code &nbsp;<a href="https://github.com/arnabmaiti-gif/arc-agi3-memory-agent">github.com/arnabmaiti-gif/arc-agi3-memory-agent</a>
  </div>
</section>

<nav id="dots"></nav>
<div class="hint">← / → / space</div>

<script>
  const PAL = {json.dumps(PALETTE)};
  function paint(id, g) {{
    const cv = document.getElementById(id); if (!cv || !g) return;
    const ctx = cv.getContext('2d');
    for (let y=0; y<64; y++) for (let x=0; x<64; x++) {{
      ctx.fillStyle = PAL[((g[y] && g[y][x]) || 0) % 16]; ctx.fillRect(x, y, 1, 1);
    }}
  }}
  paint('g1', {json.dumps(grid)});
  paint('g2', {before_grid});
  paint('g3', {after_grid});

  const slides = [...document.querySelectorAll('.slide')];
  const dots = document.getElementById('dots');
  slides.forEach((_, i) => {{
    const b = document.createElement('button'); b.className = 'dot' + (i===0?' on':'');
    b.onclick = () => go(i); dots.appendChild(b);
  }});
  let cur = 0;
  function go(n) {{
    cur = Math.max(0, Math.min(slides.length-1, n));
    slides.forEach((s,i) => s.classList.toggle('on', i===cur));
    [...dots.children].forEach((d,i) => d.classList.toggle('on', i===cur));
    document.getElementById('cur').textContent = cur+1;
  }}
  document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowRight' || e.key === ' ') {{ e.preventDefault(); go(cur+1); }}
    else if (e.key === 'ArrowLeft') {{ go(cur-1); }}
  }});
</script>
</body></html>
"""
    out = _DIR / "slides.html"
    out.write_text(page)
    print(f"slides -> {out}  ({len(page)} bytes)")


if __name__ == "__main__":
    main()
