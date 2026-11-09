"""Start, wait for and stop one vLLM server (serve/serve.py) from the run scripts."""

import os
import signal
import subprocess
import time
from pathlib import Path

import requests

from vlm_street_audit.config import PROJECT_ROOT

SERVE_DIR = PROJECT_ROOT / "serve"


# Long, because wsl_yield can pause the server while it loads (SIGSTOP) and the clock keeps running:
# 30 min was too short twice on 2026-10-29/30. A server that exits early is still caught at once.
def wait_until_ready(base_url: str, process: subprocess.Popen, timeout_s: float = 6 * 3600) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}; see its log")
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("vLLM did not become ready in time")


def start_server(model_key: str, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    return subprocess.Popen(
        [str(SERVE_DIR / ".venv" / "bin" / "python"), "serve.py", model_key],
        cwd=SERVE_DIR,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop_server(process: subprocess.Popen) -> None:
    """Stop vLLM's whole process group: its engine runs in a child process that holds the GPU memory."""
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=120)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        pass
    time.sleep(5)  # give the engine child a moment to exit after the API server
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
