"""Run the street-audit task with each VLM on the clip sets.

Each model is served by vLLM (serve/serve.py) one at a time; with --start-server
this script starts it, waits until it answers, runs every clip and stops it again.
Answers go to outputs/vlm/<prompt variant>/<set>/<model>/<clip_id>.json; clips already answered are
skipped, so a stopped run resumes where it left off.

Run on wsl5090:
    uv run python scripts/run_vlm.py --start-server --models all --sets jaad cities
    uv run python scripts/run_vlm.py --models qwen3.8-27b --sets cities --limit 3   # server already up
    uv run python scripts/run_vlm.py --start-server --prompt audit --tag audit-1frame --frames 3   # robustness
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from vlm_street_audit.config import PROJECT_ROOT, load_config, project_path
from vlm_street_audit.vlm import VARIANTS, ask, build_messages, issues_of, load_models, prompt_version

logger = logging.getLogger("run_vlm")
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


def full_condition(settings: dict, condition: dict | None) -> dict:
    return {"temperature": 0.0, "seed": 0, "frames": None,
            "max_side": int(settings["vlm_image_max_side"])} | (condition or {})


def pending(model_key: str, set_name: str, settings: dict, limit: int | None, variant: str, only: set[str] | None,
            condition: dict | None = None, tag: str | None = None) -> list:
    """The clips of a set this model has not yet answered validly with this prompt and condition."""
    condition = full_condition(settings, condition)
    clips = pd.read_csv(project_path(settings["clip_sets_dir"]) / f"{set_name}.csv")
    if only:
        clips = clips[clips.clip_id.isin(only)]
    if limit:
        clips = clips.head(limit)
    version = prompt_version(variant)
    out_dir = project_path(settings["vlm_output_dir"]) / (tag or variant) / set_name / model_key
    todo = []
    for clip in clips.itertuples():
        path = out_dir / f"{clip.clip_id}.json"
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if (previous.get("prompt_version") == version and previous.get("answer") is not None
                    and previous.get("condition", condition) == condition):
                continue  # answered with this prompt and condition already; otherwise ask again
        todo.append(clip)
    return todo


def run_set(model: dict, set_name: str, settings: dict, base_url: str, limit: int | None, variant: str,
            only: set[str] | None, parallel: int = 1, timeout_s: float = 600, condition: dict | None = None,
            tag: str | None = None) -> tuple[int, int]:
    """Answer every clip of a set; `condition` (temperature, seed, frames, max_side) varies the request."""
    todo = pending(model["key"], set_name, settings, limit, variant, only, condition, tag)
    condition = full_condition(settings, condition)
    version = prompt_version(variant)
    out_dir = project_path(settings["vlm_output_dir"]) / (tag or variant) / set_name / model["key"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def answer(clip) -> dict:
        names = clip.frames.split(";")
        if condition["frames"]:  # 1-based positions of the frames to keep, e.g. [3] for the middle one
            names = [names[i - 1] for i in condition["frames"]]
        record = {"clip_id": clip.clip_id, "set": set_name, "role": clip.role, "model": model["key"],
                  "checkpoint": model["checkpoint"], "revision": model.get("revision", ""),
                  "prompt_variant": variant, "prompt_version": version, "condition": condition, "frames": names}
        try:
            record.update(ask(model["key"], build_messages([PROJECT_ROOT / f for f in names], float(clip.seconds),
                                                           condition["max_side"], variant),
                              base_url, timeout_s=timeout_s, max_tokens=int(model.get("max_answer_tokens", 2048)),
                              variant=variant, temperature=condition["temperature"], seed=condition["seed"],
                              thinking_switch=bool(model.get("thinking_switch", True))))
        except requests.RequestException as error:
            record.update({"answer": None, "error": f"request failed: {error}"})
        path = out_dir / f"{clip.clip_id}.json"
        partial = path.with_suffix(".part")
        partial.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        partial.replace(path)
        return record

    done = failed = 0
    # Several requests at once: vLLM batches them, which raises throughput on the shared GPU.
    # Answers stay the same in substance (temperature 0); batching can shift the last decimals.
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        for record in pool.map(answer, todo):
            done += 1
            failed += record.get("answer") is None
            logger.info("%s %s %s [%d/%d]: %s", model["key"], set_name, record["clip_id"], done, len(todo),
                        record.get("error") or f"{len(issues_of(record['answer']))} issues, {record['seconds']} s")
    return done, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--sets", nargs="+", default=["jaad", "cities"])
    parser.add_argument("--start-server", action="store_true", help="start and stop vLLM for each model")
    parser.add_argument("--limit", type=int, help="first N clips of each set only (smoke test)")
    parser.add_argument("--prompt", choices=sorted(VARIANTS), default="open", help="prompt variant")
    parser.add_argument("--timeout", type=float, default=600,
                        help="seconds to wait for one answer (longer when the GPU is shared)")
    parser.add_argument("--clips", type=Path, help="file with one clip_id per line: run only these")
    # Robustness conditions; each run with one goes to its own folder (--tag).
    parser.add_argument("--tag", help="output folder name instead of the prompt variant's (e.g. audit-1frame)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", help="1-based positions of the clip's frames to send")
    parser.add_argument("--max-side", type=int, help="longest image side in pixels (default: vlm_image_max_side)")
    args = parser.parse_args()
    condition = {"temperature": args.temperature, "seed": args.seed, "frames": args.frames}
    if args.max_side:
        condition["max_side"] = args.max_side
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_config()
    models = load_models()
    server = models.pop("server")
    keys = list(models) if args.models == ["all"] else args.models
    base_url = f"http://127.0.0.1:{server['port']}"
    only = set(args.clips.read_text().split()) if args.clips else None
    failures: list[str] = []
    for key in keys:
        process = None
        # A resumed run (or its retry pass) need not load a model that has nothing left to answer.
        if not any(pending(key, set_name, settings, args.limit, args.prompt, only, condition, args.tag)
                   for set_name in args.sets):
            logger.info("%s: nothing left to answer, not started", key)
            continue
        if args.start_server:
            logger.info("starting vLLM for %s", key)
            process = start_server(key, PROJECT_ROOT / "logs" / f"vllm-{key}.log")
        try:
            if process is not None:
                wait_until_ready(base_url, process)
            for set_name in args.sets:
                done, failed = run_set(models[key], set_name, settings, base_url, args.limit, args.prompt, only,
                                       int(server.get("parallel_requests", 1)), args.timeout, condition, args.tag)
                logger.info("%s %s: %d answered, %d without a valid answer", key, set_name, done, failed)
        except (RuntimeError, TimeoutError) as error:
            # One model failing to start must not stop the others.
            failures.append(key)
            logger.error("%s skipped: %s", key, error)
        finally:
            if process is not None:
                stop_server(process)
    if failures:
        logger.error("models that did not run: %s", ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
