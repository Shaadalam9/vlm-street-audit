"""Start vLLM for one model of configs/models.toml, on 127.0.0.1 only.

Run on wsl5090, from this folder:
    uv run python serve.py qwen3.8-27b          # foreground
    uv run python serve.py qwen3.8-27b --print  # only print the command

One model at a time: the GPU also holds the CROWD workers (~6 GB), so each server
takes `gpu_memory_utilization` of the 32 GB card, set in models.toml.
"""

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "configs" / "models.toml"


def command(key: str) -> list[str]:
    config = tomllib.loads(MODELS.read_text(encoding="utf-8"))
    server = config["server"]
    models = {m["key"]: m for m in config["model"]}
    if key not in models:
        raise SystemExit(f"Unknown model {key!r}; choose from {sorted(models)}")
    model = models[key]
    # The vllm entry point of this environment, which need not be on PATH.
    executable = Path(sys.executable).with_name("vllm")
    args = [
        str(executable) if executable.is_file() else "vllm", "serve", model["checkpoint"],
        "--served-model-name", key,
        "--host", "127.0.0.1", "--port", str(server["port"]),
        "--gpu-memory-utilization", str(model.get("gpu_memory_utilization", server["gpu_memory_utilization"])),
        "--max-model-len", str(model.get("max_model_len", server["max_model_len"])),
        "--limit-mm-per-prompt", json.dumps({"image": server["max_images"], "video": 0}),
        "--seed", "0",
    ]
    if model.get("revision"):
        args += ["--revision", model["revision"]]
    if model.get("trust_remote_code"):
        args.append("--trust-remote-code")
    kv_cache_gib = model.get("kv_cache_gib", server.get("kv_cache_gib"))
    if kv_cache_gib:
        args += ["--kv-cache-memory-bytes", str(int(kv_cache_gib * 2**30))]
    args += server.get("vllm_args", []) + model.get("vllm_args", [])
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--print", action="store_true", help="print the command instead of running it")
    args = parser.parse_args()
    cmd = command(args.model)
    # Triton compiles small C helpers at run time and needs a C compiler.
    os.environ.setdefault("CC", str(Path(__file__).resolve().with_name("cc")))
    for name, value in tomllib.loads(MODELS.read_text(encoding="utf-8"))["server"].get("env", {}).items():
        os.environ.setdefault(name, str(value))
    print(" ".join(cmd), flush=True)
    if not args.print:
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
