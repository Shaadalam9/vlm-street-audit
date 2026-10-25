"""Run the street-audit task with each VLM on the clip sets.

Each model is served by vLLM (serve/serve.py) one at a time; with --start-server
this script starts it, waits until it answers, runs every clip and stops it again.
Answers go to outputs/vlm/<set>/<model>/<clip_id>.json; clips already answered are
skipped, so a stopped run resumes where it left off.

Run on wsl5090:
    uv run python scripts/run_vlm.py --start-server --models all --sets jaad cities
    uv run python scripts/run_vlm.py --models qwen3.8-27b --sets cities --limit 3   # server already up
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from vlm_street_audit.config import PROJECT_ROOT, load_config, project_path
from vlm_street_audit.vlm import PROMPT_VERSION, ask, build_messages, load_models

logger = logging.getLogger("run_vlm")
SERVE_DIR = PROJECT_ROOT / "serve"


def wait_until_ready(base_url: str, process: subprocess.Popen, timeout_s: float = 1800) -> None:
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
    return subprocess.Popen([str(SERVE_DIR / ".venv" / "bin" / "python"), "serve.py", model_key], cwd=SERVE_DIR,
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)


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


def run_set(model: dict, set_name: str, settings: dict, base_url: str, limit: int | None) -> tuple[int, int]:
    clips = pd.read_csv(project_path(settings["clip_sets_dir"]) / f"{set_name}.csv")
    if limit:
        clips = clips.head(limit)
    out_dir = project_path(settings["vlm_output_dir"]) / set_name / model["key"]
    out_dir.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    with requests.Session() as session:
        for clip in clips.itertuples():
            path = out_dir / f"{clip.clip_id}.json"
            if path.is_file():
                continue
            frames = [PROJECT_ROOT / f for f in clip.frames.split(";")]
            record = {"clip_id": clip.clip_id, "set": set_name, "role": clip.role, "model": model["key"],
                      "checkpoint": model["checkpoint"], "revision": model.get("revision", ""),
                      "prompt_version": PROMPT_VERSION, "frames": clip.frames.split(";")}
            try:
                record.update(ask(model["key"], build_messages(frames, float(clip.seconds),
                                                               int(settings["vlm_image_max_side"])),
                                  base_url, session=session))
            except requests.RequestException as error:
                record.update({"answer": None, "error": f"request failed: {error}"})
            failed += record.get("answer") is None
            done += 1
            partial = path.with_suffix(".part")
            partial.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
            partial.replace(path)
            logger.info("%s %s %s: %s", model["key"], set_name, clip.clip_id,
                        record.get("error") or f"{len(record['answer']['issues'])} issues, {record['seconds']} s")
    return done, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--sets", nargs="+", default=["jaad", "cities"])
    parser.add_argument("--start-server", action="store_true", help="start and stop vLLM for each model")
    parser.add_argument("--limit", type=int, help="first N clips of each set only (smoke test)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_config()
    models = load_models()
    server = models.pop("server")
    keys = list(models) if args.models == ["all"] else args.models
    base_url = f"http://127.0.0.1:{server['port']}"
    for key in keys:
        process = None
        if args.start_server:
            logger.info("starting vLLM for %s", key)
            process = start_server(key, PROJECT_ROOT / "logs" / f"vllm-{key}.log")
        try:
            if process is not None:
                wait_until_ready(base_url, process)
            for set_name in args.sets:
                done, failed = run_set(models[key], set_name, settings, base_url, args.limit)
                logger.info("%s %s: %d answered, %d without a valid answer", key, set_name, done, failed)
        finally:
            if process is not None:
                stop_server(process)
    return 0


if __name__ == "__main__":
    sys.exit(main())
