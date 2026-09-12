#!/usr/bin/env python3
"""
ShortStack -- native Linux desktop app.

Run this in the same directory as your docker-compose.yaml (or set
COMPOSE_DIR). Gives you:

    - Start / Stop / Restart / Kill buttons for the whole podman-compose stack.
    - An expandable row per service showing live logs, container status, and individual start/stop/restart/kill controls.
      Expanding starts the underlying `podman-compose logs -f` process;
      collapsing terminates it.
    - A GPU panel that polls `nvidia-smi` once every 2 seconds while its
      toggle is on, and renders the numbers into fixed labels/bars -- it
      never runs a continuous streaming process and never accumulates text.
      Turning the toggle off stops the polling outright.
    - A host CPU and RAM panel that updates alongside the GPU monitor.
    - Container status is displayed inline beside each service and refreshed
      from structured `podman ps` output.

Usage:
    pip install -r requirements.txt --break-system-packages
    python3 app.py
"""

import os
import re
import shutil
import subprocess
import sys
import threading

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QToolButton,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMPOSE_DIR = os.environ.get("COMPOSE_DIR", "/mnt/data/LLM")
PODMAN_ROOT = os.environ.get("PODMAN_ROOT", os.path.join(COMPOSE_DIR, "containers-storage"))
# Keep podman-compose and direct `podman` status queries on the same storage root.
# The compose wrapper accepts the forwarded argument as a single string.
BASE_CMD = ["podman-compose", f"--podman-args=--root {PODMAN_ROOT}"]

SERVICES = ["tailscale", "ollama", "open-webui", "tor", "qdrant", "searxng", "comfyui"]

HAS_NVIDIA_SMI = shutil.which("nvidia-smi") is not None
HAS_PODMAN_COMPOSE = shutil.which("podman-compose") is not None
HAS_PODMAN = shutil.which("podman") is not None

MAX_LOG_LINES = 400
GPU_POLL_MS = 2000
HOST_POLL_MS = 2000
STATUS_POLL_MS = 4000

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def strip_ansi(text):
    return ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Signal bus -- background threads emit these, the main window listens.
# Qt marshals cross-thread signal emissions onto the UI thread automatically.
# ---------------------------------------------------------------------------

class Bus(QObject):
    log_line = Signal(str, str)          # service name, line
    watcher_toggled = Signal(str, bool)  # service name, enabled
    watcher_error = Signal(str, str)     # service name, message
    compose_line = Signal(str)
    compose_busy = Signal(bool, str)     # busy, label
    container_status = Signal(dict)      # {service: (state, status_text)}
    gpu_stats = Signal(dict)             # parsed nvidia-smi fields
    gpu_error = Signal(str)
    host_stats = Signal(dict)            # parsed host cpu/ram fields
    host_error = Signal(str)


bus = Bus()

host_stats_lock = threading.Lock()
last_cpu_sample = None  # (total, idle)


# ---------------------------------------------------------------------------
# Log watchers -- one `podman-compose logs -f <service>` process per toggle.
# ---------------------------------------------------------------------------

watchers_lock = threading.Lock()
watchers = {name: {"process": None, "enabled": False} for name in SERVICES}


def _reader_loop(name, proc):
    try:
        for line in proc.stdout:
            bus.log_line.emit(name, strip_ansi(line.rstrip("\n")))
    except Exception:
        pass


def start_watcher(name):
    with watchers_lock:
        w = watchers[name]
        if w["enabled"] and w["process"] and w["process"].poll() is None:
            return
        try:
            proc = subprocess.Popen(
                BASE_CMD + ["logs", "-f", "--tail", "50", name],
                cwd=COMPOSE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            bus.watcher_error.emit(name, str(e))
            return
        w["process"] = proc
        w["enabled"] = True
        threading.Thread(target=_reader_loop, args=(name, proc), daemon=True).start()
    bus.watcher_toggled.emit(name, True)


def stop_watcher(name, notify=True):
    with watchers_lock:
        w = watchers[name]
        w["enabled"] = False
        proc = w["process"]
        w["process"] = None
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        except Exception:
            pass
    if notify:
        bus.watcher_toggled.emit(name, False)


def stop_all_watchers():
    for name in SERVICES:
        stop_watcher(name)


# ---------------------------------------------------------------------------
# GPU polling -- a single one-shot `nvidia-smi` query per tick, no
# long-running subprocess. The UI timer that drives this is only started
# while the toggle is on, so turning it off means zero further queries.
# ---------------------------------------------------------------------------

GPU_FIELDS = "utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw"


def poll_gpu_once():
    def worker():
        try:
            proc = subprocess.run(
                ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            line = (proc.stdout or "").strip().splitlines()
            if not line:
                bus.gpu_error.emit(proc.stderr.strip() or "no output from nvidia-smi")
                return
            parts = [p.strip() for p in line[0].split(",")]
            if len(parts) != 6:
                bus.gpu_error.emit(f"unexpected nvidia-smi output: {line[0]}")
                return
            util_gpu, util_mem, mem_used, mem_total, temp, power = (float(p) for p in parts)
            bus.gpu_stats.emit(
                {
                    "util_gpu": util_gpu,
                    "util_mem": util_mem,
                    "mem_used": mem_used,
                    "mem_total": mem_total,
                    "temp": temp,
                    "power": power,
                }
            )
        except Exception as e:
            bus.gpu_error.emit(str(e))

    threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Container status -- one structured `podman ps` call, parsed, instead of
# a dumped `podman-compose ps` text block.
# ---------------------------------------------------------------------------


def poll_host_once():
    def read_cpu_sample():
        with open("/proc/stat", "r", encoding="utf-8") as fh:
            fields = fh.readline().split()
        if not fields or fields[0] != "cpu":
            raise RuntimeError("unexpected /proc/stat format")
        values = [float(v) for v in fields[1:]]
        while len(values) < 7:
            values.append(0.0)
        total = sum(values[:7])
        idle = values[3] + values[4]
        return total, idle

    def read_mem_sample():
        meminfo = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, rest = line.split(":", 1)
                meminfo[key] = int(rest.strip().split()[0])
        total_kib = meminfo["MemTotal"]
        avail_kib = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        used_kib = max(total_kib - avail_kib, 0)
        used_pct = (used_kib / total_kib * 100.0) if total_kib else 0.0
        return used_kib / 1024.0, total_kib / 1024.0, used_pct

    def worker():
        global last_cpu_sample
        try:
            total, idle = read_cpu_sample()
            with host_stats_lock:
                previous = last_cpu_sample
                last_cpu_sample = (total, idle)
            if previous is None:
                cpu_pct = 0.0
            else:
                prev_total, prev_idle = previous
                total_delta = max(total - prev_total, 0.0)
                idle_delta = max(idle - prev_idle, 0.0)
                busy_delta = max(total_delta - idle_delta, 0.0)
                cpu_pct = (busy_delta / total_delta * 100.0) if total_delta else 0.0

            mem_used_mib, mem_total_mib, mem_pct = read_mem_sample()
            bus.host_stats.emit({
                "cpu_pct": cpu_pct,
                "mem_used_mib": mem_used_mib,
                "mem_total_mib": mem_total_mib,
                "mem_pct": mem_pct,
            })
        except Exception as e:
            bus.host_error.emit(str(e))

    threading.Thread(target=worker, daemon=True).start()

def refresh_container_status():
    def worker():
        result = {name: ("unknown", "not found") for name in SERVICES}
        try:
            # Match the same Podman storage root used by podman-compose.
            # Without this, `podman ps` can inspect a different local store
            # and report every compose service as missing.
            proc = subprocess.run(
                [
                    "podman",
                    f"--root={PODMAN_ROOT}",
                    "ps",
                    "-a",
                    "--format",
                    "{{.Names}}|{{.State}}|{{.Status}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout or "podman ps failed").strip())

            for line in (proc.stdout or "").splitlines():
                parts = line.split("|", 2)
                if len(parts) != 3:
                    continue
                cname, state, status_text = parts
                if cname in result:
                    result[cname] = (state.lower(), status_text)
        except Exception as e:
            message = str(e)
            for name in SERVICES:
                result[name] = ("error", message)
        bus.container_status.emit(result)

    threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Compose lifecycle (stack + per-service up / stop / restart / kill)
# ---------------------------------------------------------------------------

compose_busy_flag = threading.Event()


def _run_sequence(cmds):
    for cmd in cmds:
        bus.compose_line.emit(f"$ {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd, cwd=COMPOSE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            for line in proc.stdout:
                bus.compose_line.emit(strip_ansi(line.rstrip("\n")))
            proc.wait()
            if proc.returncode != 0:
                bus.compose_line.emit(f"[exited with code {proc.returncode}]")
                return "error"
        except FileNotFoundError as e:
            bus.compose_line.emit(f"[error] {e}")
            return "error"
    return "ok"


def _run_action(label, cmds):
    if compose_busy_flag.is_set():
        return False
    compose_busy_flag.set()
    bus.compose_busy.emit(True, label)

    def worker():
        _run_sequence(cmds)
        compose_busy_flag.clear()
        bus.compose_busy.emit(False, label)
        refresh_container_status()

    threading.Thread(target=worker, daemon=True).start()
    return True


def _service_cmd(name, action):
    if action == "up":
        return BASE_CMD + ["up", "-d", name]
    return BASE_CMD + [action, name]


def action_start():
    return _run_action("up", [BASE_CMD + ["up", "-d"]])


def action_restart():
    threading.Thread(target=stop_all_watchers, daemon=True).start()
    return _run_action("restart", [BASE_CMD + ["restart"]])


def action_kill():
    # Force: SIGKILL every service immediately, then remove the containers.
    threading.Thread(target=stop_all_watchers, daemon=True).start()
    return _run_action("kill", [BASE_CMD + ["kill", "--all"]])


def action_service(name, action):
    if action in {"restart", "kill", "stop"}:
        threading.Thread(target=stop_watcher, args=(name,), daemon=True).start()
    return _run_action(f"{name} {action}", [_service_cmd(name, action)])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

MONO = QFont("Monospace")
MONO.setStyleHint(QFont.TypeWriter)
MONO.setPointSize(10)

STATE_COLORS = {
    "running": "#3fb950",
    "exited": "#8b96a5",
    "created": "#d29922",
    "paused": "#d29922",
    "error": "#f85149",
    "unknown": "#8b96a5",
}


def make_log_box():
    box = QPlainTextEdit()
    box.setReadOnly(True)
    box.setFont(MONO)
    box.setMaximumBlockCount(MAX_LOG_LINES)
    box.setFixedHeight(140)
    box.setVisible(False)
    box.setStyleSheet(
        "QPlainTextEdit { background:#0a0e14; color:#c9d1d9; border:1px solid #262d38; border-radius:6px; }"
    )
    return box


def hline():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color:#262d38;")
    return line


def styled_bar():
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setTextVisible(True)
    bar.setFixedHeight(16)
    bar.setStyleSheet(
        "QProgressBar { background:#0a0e14; border:1px solid #262d38; border-radius:6px; "
        "color:#e6edf3; text-align:center; font-size:10px; }"
        "QProgressBar::chunk { background:#4f8cff; border-radius:5px; }"
    )
    return bar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ShortStack")
        self.resize(780, 860)

        self.log_boxes = {}
        self.checkboxes = {}
        self.status_labels = {}
        self.service_action_buttons = {}

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # --- Header: title + action buttons ---
        header = QHBoxLayout()
        header.addStretch()

        self.btn_up = QPushButton("Up")
        self.btn_down = QPushButton("Down")
        self.btn_restart = QPushButton("Restart")
        self.btn_kill = QPushButton("Kill")
        self.btn_up.setToolTip("podman-compose up -d")
        self.btn_down.setToolTip("podman-compose down -- tear down the stack and remove its containers")
        self.btn_restart.setToolTip("podman-compose restart -- restart the existing containers")
        self.btn_kill.setToolTip("podman-compose kill --all -- force-stop the containers")
        self.btn_up.setStyleSheet("background:#3fb950; color:#0d1117; font-weight:600; padding:6px 14px;")
        self.btn_down.setStyleSheet("background:#4f8cff; color:#0d1117; font-weight:600; padding:6px 14px;")
        self.btn_restart.setStyleSheet("background:#d29922; color:#0d1117; font-weight:600; padding:6px 14px;")
        self.btn_kill.setStyleSheet("background:#f85149; color:#0d1117; font-weight:600; padding:6px 14px;")
        self.btn_up.clicked.connect(self.on_up)
        self.btn_down.clicked.connect(self.on_down)
        self.btn_restart.clicked.connect(self.on_restart)
        self.btn_kill.clicked.connect(self.on_kill)
        for b in (self.btn_up, self.btn_down, self.btn_restart, self.btn_kill):
            header.addWidget(b)
        outer.addLayout(header)

        if not HAS_PODMAN_COMPOSE:
            warn = QLabel("⚠ podman-compose not found on PATH -- buttons/toggles will error until it's installed.")
            warn.setStyleSheet("color:#f85149; font-size:12px;")
            outer.addWidget(warn)

        # Scrollable content area for the panels
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)
        outer.addWidget(scroll)

        # --- GPU monitor: fixed gauges, not a scrolling log ---
        content_layout.addWidget(
            self._section_label(
                "GPU monitor",
                "Polls nvidia-smi once every 2s while on. Off means zero queries -- no background process at all.",
            )
        )
        gpu_row = QHBoxLayout()
        self.gpu_checkbox = QCheckBox("Enable GPU polling")
        self.gpu_checkbox.setEnabled(HAS_NVIDIA_SMI)
        self.gpu_checkbox.stateChanged.connect(self.on_gpu_toggle)
        gpu_row.addWidget(self.gpu_checkbox)
        gpu_row.addStretch()
        content_layout.addLayout(gpu_row)
        if not HAS_NVIDIA_SMI:
            no_gpu = QLabel("nvidia-smi not found on this host")
            no_gpu.setStyleSheet("color:#8b96a5; font-size:11px;")
            content_layout.addWidget(no_gpu)

        self.gpu_panel = QWidget()
        self.gpu_panel.setVisible(False)
        gpu_grid = QGridLayout(self.gpu_panel)
        gpu_grid.setContentsMargins(0, 6, 0, 6)

        self.gpu_util_bar = styled_bar()
        self.gpu_mem_bar = styled_bar()
        self.gpu_temp_label = QLabel("-- °C")
        self.gpu_power_label = QLabel("-- W")
        self.gpu_mem_label = QLabel("-- / -- MiB")

        gpu_grid.addWidget(QLabel("GPU utilization"), 0, 0)
        gpu_grid.addWidget(self.gpu_util_bar, 0, 1)
        gpu_grid.addWidget(QLabel("Memory"), 1, 0)
        gpu_grid.addWidget(self.gpu_mem_bar, 1, 1)
        gpu_grid.addWidget(self.gpu_mem_label, 1, 2)
        gpu_grid.addWidget(QLabel("Temperature"), 2, 0)
        gpu_grid.addWidget(self.gpu_temp_label, 2, 1)
        gpu_grid.addWidget(QLabel("Power draw"), 3, 0)
        gpu_grid.addWidget(self.gpu_power_label, 3, 1)
        for lbl in (self.gpu_temp_label, self.gpu_power_label, self.gpu_mem_label):
            lbl.setStyleSheet("color:#c9d1d9; font-family:monospace; font-size:12px;")
        content_layout.addWidget(self.gpu_panel)

        # --- Host monitor: CPU and RAM alongside GPU ---
        content_layout.addWidget(
            self._section_label(
                "Host monitor",
                "Shows host CPU and RAM utilization alongside the GPU. CPU and RAM refresh on the same 2s tick.",
            )
        )
        host_grid = QGridLayout()
        host_grid.setContentsMargins(0, 6, 0, 6)

        self.host_cpu_bar = styled_bar()
        self.host_ram_bar = styled_bar()
        self.host_cpu_label = QLabel("-- %")
        self.host_ram_label = QLabel("-- / -- MiB")

        host_grid.addWidget(QLabel("CPU utilization"), 0, 0)
        host_grid.addWidget(self.host_cpu_bar, 0, 1)
        host_grid.addWidget(self.host_cpu_label, 0, 2)
        host_grid.addWidget(QLabel("Memory"), 1, 0)
        host_grid.addWidget(self.host_ram_bar, 1, 1)
        host_grid.addWidget(self.host_ram_label, 1, 2)
        for lbl in (self.host_cpu_label, self.host_ram_label):
            lbl.setStyleSheet("color:#c9d1d9; font-family:monospace; font-size:12px;")
        content_layout.addLayout(host_grid)
        content_layout.addWidget(hline())

        # --- Services + expandable logs/status ---
        content_layout.addWidget(
            self._section_label(
                "Services",
                "Click a service to expand its live logs. Each row now has its own start/stop/restart/kill controls.",
            )
        )
        for svc in SERVICES:
            row, log_box = self._watcher_row(svc, svc)
            content_layout.addLayout(row)
            content_layout.addWidget(log_box)
        content_layout.addWidget(hline())

        # --- Compose output ---
        content_layout.addWidget(self._section_label("Last command output", ""))
        self.compose_output = QPlainTextEdit()
        self.compose_output.setReadOnly(True)
        self.compose_output.setFont(MONO)
        self.compose_output.setMaximumBlockCount(MAX_LOG_LINES)
        self.compose_output.setFixedHeight(160)
        self.compose_output.setStyleSheet(
            "QPlainTextEdit { background:#0a0e14; color:#c9d1d9; border:1px solid #262d38; border-radius:6px; }"
        )
        content_layout.addWidget(self.compose_output)
        content_layout.addStretch()

        # --- Wire up bus signals ---
        bus.log_line.connect(self.on_log_line)
        bus.watcher_toggled.connect(self.on_watcher_toggled)
        bus.watcher_error.connect(self.on_watcher_error)
        bus.compose_line.connect(self.on_compose_line)
        bus.compose_busy.connect(self.on_compose_busy)
        bus.container_status.connect(self.on_container_status)
        bus.gpu_stats.connect(self.on_gpu_stats)
        bus.gpu_error.connect(self.on_gpu_error)
        bus.host_stats.connect(self.on_host_stats)
        bus.host_error.connect(self.on_host_error)

        # --- GPU polling timer: only runs while the toggle is on ---
        self.gpu_timer = QTimer(self)
        self.gpu_timer.timeout.connect(poll_gpu_once)

        # --- Host polling timer: always-on CPU/RAM sampling ---
        self.host_timer = QTimer(self)
        self.host_timer.timeout.connect(poll_host_once)
        self.host_timer.start(HOST_POLL_MS)
        poll_host_once()

        # --- Periodic container status refresh ---
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(refresh_container_status)
        self.status_timer.start(STATUS_POLL_MS)
        refresh_container_status()
    # -- UI builder helpers --

    def _section_label(self, title, hint):
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 8, 0, 4)
        t = QLabel(title)
        t.setStyleSheet("font-size:14px; font-weight:600;")
        layout.addWidget(t)
        if hint:
            h = QLabel(hint)
            h.setStyleSheet("color:#8b96a5; font-size:11px;")
            h.setWordWrap(True)
            layout.addWidget(h)
        return wrap

    def _service_button(self, text, tooltip, color, callback):
        btn = QToolButton()
        btn.setText(text)
        btn.setAutoRaise(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setFont(MONO)
        btn.setStyleSheet(
            "QToolButton {"
            f"border:1px solid {color}; padding:2px 8px; border-radius:6px; color:{color}; background:transparent; font-size:10px;"
            "}"
            f"QToolButton:hover {{ background:{color}; color:#0d1117; }}"
            "QToolButton:disabled { color:#586069; border-color:#30363d; background:transparent; }"
        )
        btn.setToolTip(tooltip)
        btn.clicked.connect(callback)
        return btn

    def _watcher_row(self, name, label_text):
        row = QHBoxLayout()
        row.setContentsMargins(0, 1, 0, 1)
        row.setSpacing(8)

        toggle = QToolButton()
        toggle.setText(f"▸ {label_text}")
        toggle.setCheckable(True)
        toggle.setAutoRaise(True)
        toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        toggle.setFont(MONO)
        toggle.setStyleSheet(
            "QToolButton { border:none; padding:3px 2px; text-align:left; color:#e6edf3; }"
            "QToolButton:hover { color:#58a6ff; }"
        )
        toggle.clicked.connect(lambda checked, n=name: self.on_toggle(n, checked))
        row.addWidget(toggle)

        status = QLabel("querying…")
        status.setStyleSheet("color:#8b96a5; font-size:11px;")
        status.setFont(MONO)
        row.addWidget(status)
        row.addStretch()

        buttons = []
        buttons.append(self._service_button("Up", f"Start {name}", "#3fb950", lambda _, n=name: self.on_service_action(n, "up")))
        buttons.append(self._service_button("Stop", f"Stop {name}", "#8b96a5", lambda _, n=name: self.on_service_action(n, "stop")))
        buttons.append(self._service_button("Restart", f"Restart {name}", "#d29922", lambda _, n=name: self.on_service_action(n, "restart")))
        buttons.append(self._service_button("Kill", f"Kill {name}", "#f85149", lambda _, n=name: self.on_service_action(n, "kill")))
        self.service_action_buttons[name] = buttons
        for btn in buttons:
            row.addWidget(btn)

        self.checkboxes[name] = toggle
        self.status_labels[name] = status
        log_box = make_log_box()
        self.log_boxes[name] = log_box
        return row, log_box

    # -- Button handlers --

    def on_up(self):
        action_start()

    def on_down(self):
        if QMessageBox.question(
            self,
            "Down stack",
            "Bring the whole stack down? This stops and removes the containers; persistent volumes are kept.",
        ) == QMessageBox.Yes:
            _run_action("down", [BASE_CMD + ["down"]])

    def on_restart(self):
        if QMessageBox.question(
            self,
            "Restart stack",
            "Restart the whole stack? This stops and starts the existing containers.",
        ) == QMessageBox.Yes:
            action_restart()

    def on_kill(self):
        if QMessageBox.question(
            self,
            "Kill stack",
            "Kill the whole stack now? This force-stops all containers immediately.",
        ) == QMessageBox.Yes:
            action_kill()

    def on_service_action(self, name, action):
        if action == "restart" and QMessageBox.question(
            self,
            f"Restart {name}",
            f"Restart {name}? This stops and starts only that service.",
        ) != QMessageBox.Yes:
            return
        if action == "kill" and QMessageBox.question(
            self,
            f"Kill {name}",
            f"Kill {name}? This force-stops only that service.",
        ) != QMessageBox.Yes:
            return
        action_service(name, action)

    def on_toggle(self, name, checked):
        if checked:
            threading.Thread(target=start_watcher, args=(name,), daemon=True).start()
        else:
            threading.Thread(target=stop_watcher, args=(name,), daemon=True).start()

    def on_gpu_toggle(self, state):
        if state != 0:
            self.gpu_panel.setVisible(True)
            poll_gpu_once()
            self.gpu_timer.start(GPU_POLL_MS)
        else:
            self.gpu_timer.stop()
            self.gpu_panel.setVisible(False)

    # -- Bus signal handlers (run on the UI thread) --

    def on_log_line(self, name, line):
        box = self.log_boxes.get(name)
        if box is None:
            return
        sb = box.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        box.appendPlainText(line)
        if at_bottom:
            sb.setValue(sb.maximum())

    def on_watcher_toggled(self, name, enabled):
        cb = self.checkboxes.get(name)
        box = self.log_boxes.get(name)
        if cb is not None:
            cb.blockSignals(True)
            cb.setChecked(enabled)
            cb.setText(("▾ " if enabled else "▸ ") + name)
            cb.blockSignals(False)
        if box is not None:
            box.setVisible(enabled)
            if enabled:
                box.clear()

    def on_watcher_error(self, name, message):
        cb = self.checkboxes.get(name)
        if cb is not None:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.setText(f"▸ {name}")
            cb.blockSignals(False)
        box = self.log_boxes.get(name)
        if box is not None:
            box.setVisible(False)
        QMessageBox.warning(self, "Couldn't start watcher", f"{name}: {message}")

    def on_compose_line(self, line):
        sb = self.compose_output.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        self.compose_output.appendPlainText(line)
        if at_bottom:
            sb.setValue(sb.maximum())

    def on_compose_busy(self, busy, label):
        for b in (self.btn_up, self.btn_down, self.btn_restart, self.btn_kill):
            b.setEnabled(not busy)
        for buttons in self.service_action_buttons.values():
            for b in buttons:
                b.setEnabled(not busy)
        if busy:
            self.compose_output.clear()

    def on_container_status(self, result):
        for svc in SERVICES:
            state, status_text = result.get(svc, ("unknown", "not found"))
            label = self.status_labels.get(svc)
            if label is None:
                continue
            label.setText(status_text if status_text else state)
            label.setStyleSheet(
                f"color:{STATE_COLORS.get(state, '#8b96a5')}; font-size:11px;"
            )

    def on_gpu_stats(self, stats):
        self.gpu_util_bar.setValue(int(stats["util_gpu"]))
        self.gpu_util_bar.setFormat(f"{stats['util_gpu']:.0f}%")
        mem_pct = int((stats["mem_used"] / stats["mem_total"]) * 100) if stats["mem_total"] else 0
        self.gpu_mem_bar.setValue(mem_pct)
        self.gpu_mem_bar.setFormat(f"{mem_pct}%")
        self.gpu_mem_label.setText(f"{stats['mem_used']:.0f} / {stats['mem_total']:.0f} MiB")
        self.gpu_temp_label.setText(f"{stats['temp']:.0f} °C")
        self.gpu_power_label.setText(f"{stats['power']:.1f} W")

    def on_gpu_error(self, message):
        self.gpu_temp_label.setText("error")
        self.gpu_power_label.setText(message[:60])

    def on_host_stats(self, stats):
        self.host_cpu_bar.setValue(int(stats["cpu_pct"]))
        self.host_cpu_bar.setFormat(f"{stats['cpu_pct']:.0f}%")
        self.host_ram_bar.setValue(int(stats["mem_pct"]))
        self.host_ram_bar.setFormat(f"{stats['mem_pct']:.0f}%")
        self.host_cpu_label.setText(f"{stats['cpu_pct']:.0f}%")
        self.host_ram_label.setText(f"{stats['mem_used_mib']:.0f} / {stats['mem_total_mib']:.0f} MiB")

    def on_host_error(self, message):
        self.host_cpu_label.setText("error")
        self.host_ram_label.setText(message[:60])

    def closeEvent(self, event):
        stop_all_watchers()
        self.gpu_timer.stop()
        self.host_timer.stop()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget { background:#0d1117; color:#e6edf3; } QLabel { color:#e6edf3; }")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()