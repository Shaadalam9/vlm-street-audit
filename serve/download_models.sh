#!/usr/bin/env bash
# Download the VLM weights at the exact revisions in configs/models.toml, and nothing else.
#
#   bash serve/download_models.sh                  # every model in configs/models.toml
#   bash serve/download_models.sh molmo2-8b ...    # only these keys
#
# Run on the GPU machine after `cd serve && uv sync`; the files go to the Hugging Face cache (HF_HOME, default
# ~/.cache/huggingface), where vLLM finds them. Gated models need their terms accepted on the model's
# Hugging Face page and `uv run --project serve hf auth login` first. The six models take about 130 GB.
# If downloads stall (seen on WSL2 with several parallel connections), set HF_MAX_WORKERS=1.
set -euo pipefail
cd "$(dirname "$0")/.."

# key, checkpoint and revision of each requested model, one per line
models=$(uv run --frozen --project serve python - "$@" <<'EOF'
import sys, tomllib
models = {m["key"]: m for m in tomllib.load(open("configs/models.toml", "rb"))["model"]}
unknown = [k for k in sys.argv[1:] if k not in models]
if unknown:
    sys.exit(f"unknown model key(s): {', '.join(unknown)}; known: {', '.join(models)}")
for key in sys.argv[1:] or models:
    print(key, models[key]["checkpoint"], models[key]["revision"])
EOF
)

while read -r key checkpoint revision; do
    echo "== $key: $checkpoint @ $revision"
    uv run --frozen --project serve hf download "$checkpoint" --revision "$revision" \
        --max-workers "${HF_MAX_WORKERS:-8}"
    if [ "$key" = "kanana-1.5-v-3b" ]; then  # its fixed tokeniser (see the script's docstring)
        uv run --frozen --project serve python serve/prepare_tokenizers.py
    fi
done <<< "$models"
