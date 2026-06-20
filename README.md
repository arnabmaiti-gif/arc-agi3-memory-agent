# ARC-AGI-3 on HUD

The [ARC-AGI-3](https://arcprize.org/arc-agi/3) public benchmark — 25 interactive grid games — as a HUD v6 environment, **running on the official Arc Prize runner**: the local [ARCEngine toolkit](https://docs.arcprize.org/) plays the games, the vendored [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) `Agent` loop (MIT) drives each session, and scoring comes from official scorecards.

## How it maps

- **Their runner, unmodified** — each game runs the official `Agent.main()` loop (vendored in `arc_agents/`): frames history, action counter, `MAX_ACTIONS`, jsonl recordings, cleanup. HUD's model is the policy, pushing actions through a queue bridge (`HudBridgeAgent.choose_action`).
- **Official scoring** — `open_scorecard` → `make(game, scorecard_id)` → `close_scorecard`; the reward is the official per-game `score` (per-level partial credit). `run_bench.py` keeps **one scorecard across all 25 games**, like the reference Swarm runner.
- **Official metadata** — every action carries the model's `reasoning` dict, shaped like the reference LLM agents, so it appears on the scorecard.
- **HUD surface** — one `Environment` with an `mcp` capability (`look()` / `act(actions, reasoning)` returning frame images), one generative `play(game_id, scorecard_id, max_actions)` task, 25 task rows in `tasks.py`. Each rollout gets a fresh env process (`LocalRuntime`), so games run in parallel.

## Run it

```bash
# one game (needs HUD_API_KEY)
uv run hud eval tasks.py claude \
    --model claude-opus-4-7 --gateway --task-ids arc-agi-3-ls20 --max-steps 30 -y

# the official bench: one scorecard, 25 games, parallel rollouts
uv run python run_bench.py claude-opus-4-7 30 4
```

`run_bench.py` prints HUD per-game rewards, the official scorecard report (also saved to `scorecard.json`), and the scorecard id.

## Package it

Deploy straight to the platform (builds the image, vendors the games, publishes the env):

```bash
hud deploy .
```

Or build and run the container yourself — it serves the control channel on `8765`:

```bash
docker build -t arc-agi-3-env .   # vendors all 25 games at build; offline at runtime
docker run -d -p 8765:8765 --name arc arc-agi-3-env
hud client info --url tcp://127.0.0.1:8765   # identity, capabilities, the 25 tasks
```

## Files

| File | Role |
|------|------|
| `env.py` | Engine bridge to the official runner, MCP game tools, the `play` task |
| `arc_agents/` | Vendored official Agent/Recorder/tracing (MIT, arcprize/ARC-AGI-3-Agents) |
| `tasks.py` | The 25-game public bench as task rows |
| `run_bench.py` | Official single-scorecard bench runner |
| `predownload.py` / `Dockerfile` | Image build: vendor games at build, serve offline via `hud serve env:env` |
