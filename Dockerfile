FROM python:3.12-slim

WORKDIR /app

# hud-python v6 is not on PyPI yet; install from the v6 branch.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "hud-python[agents] @ git+https://github.com/hud-evals/hud-python@v6" \
    arc-agi pillow

COPY env.py tasks.py predownload.py ./
COPY arc_agents ./arc_agents

# Vendor the 25 public bench games into the image (network at build time only).
RUN python predownload.py

# Play fully offline at runtime.
ENV ARC_OPERATION_MODE=offline

# Only the control channel is published; the MCP game server binds a loopback
# port and is forwarded through it (a container publishes just the one port).
EXPOSE 8765

CMD ["hud", "serve", "env:env", "--host", "0.0.0.0", "--port", "8765"]
