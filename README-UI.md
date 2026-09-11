# ShortStack (native app)

A native Qt (PySide6) desktop app for the `podman-compose` stack (tailscale,
ollama, open-webui, tor, qdrant, searxng, comfyui). No browser, no server --
just a window.

## What it does

- **Start** – runs `podman-compose pull` then `up -d`.
- **Stop** – runs `podman-compose stop --all`. Graceful: containers stay in
  place (not removed), so **Start** brings them back up quickly.
- **Restart** – stops all log/GPU watchers, then `down` + `up -d` (full
  recreate; asks for confirmation first).
- **Kill** – stops all watchers, then `podman-compose kill --all` + `down`
  (force-stops immediately and removes the containers; asks for
  confirmation first).
- **Service log toggles** – one checkbox per container. Checking it starts
  `podman-compose logs -f <service>` and streams it into the box beneath
  (ANSI color codes are stripped so it renders as plain readable text).
  Unchecking it **terminates that process** — it's not just hidden.
- **GPU monitor** – a single lightweight `nvidia-smi` query every 2 seconds
  while the toggle is on, parsed into fixed gauges (utilization bar, memory
  bar, temperature, power draw). It does **not** run a continuous streaming
  process and does **not** accumulate a scrolling log — the numbers just
  update in place. Turning the toggle off stops the timer, so there are
  zero further GPU queries. The checkbox is disabled automatically if
  `nvidia-smi` isn't on `PATH`.
- **Container status table** – built from `podman ps -a` structured output
  (one row per service, colored by state: green=running, gray=exited,
  yellow=created/paused, red=error), refreshed every 4 seconds. This
  replaces a dumped `podman-compose ps` text blob with data the app parses
  and renders itself.
- Closing the window stops every active watcher and the GPU timer first, so
  nothing is left running in the background.

## Setup

1. Copy this whole `llm-stack-app/` folder into the **same directory** as
   your `docker-compose.yaml` and `start.sh` (the app assumes
   `docker-compose.yaml` is a sibling of `app.py` — set the `COMPOSE_DIR`
   environment variable if you'd rather keep it elsewhere).

2. Install dependencies (Qt bindings, pulled in via pip — no system package
   manager needed):

   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

3. Run it:

   ```bash
   python3 app.py
   ```

## Optional: add it to your app menu

1. Edit `llm-stack-control.desktop` and replace
   `/absolute/path/to/llm-stack-app/run.sh` with the real path, e.g.
   `/home/you/llm-stack/llm-stack-app/run.sh`.
2. Copy it into your user applications directory:

   ```bash
   cp llm-stack-control.desktop ~/.local/share/applications/
   ```
3. It should now show up in your desktop's application launcher as
   "ShortStack".

## Optional: build a standalone binary

If you'd rather not depend on a system Python + pip install, you can bundle
it into a single executable with PyInstaller:

```bash
pip install pyinstaller --break-system-packages
pyinstaller --onefile --windowed --name llm-stack-control app.py
```

The binary will land in `dist/llm-stack-control`. Copy it next to your
`docker-compose.yaml` (or set `COMPOSE_DIR`) and run it directly.

## Notes

- Uses the same `--podman-args=--root=<dir>/containers-storage` override
  your `start.sh` used, so it points at the same storage root.
- This is a separate control surface from `start.sh`'s tmux setup — both
  just call `podman-compose`, so you can use either at any time (though
  running two `logs -f` tails on the same container from both at once is
  harmless, just redundant).
- Restart/Kill stop all watchers first to avoid leaving a `logs -f` process
  attached to a container that's about to disappear.
- No web server, no open port — everything runs as local subprocesses of
  the app itself.
