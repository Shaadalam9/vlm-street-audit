"""Build the local tokeniser folders some models need (data/tokenizers/, not tracked in git).

Kanana-1.5-V 3B: its repository loads a slow tokeniser class that the structured-output engines cannot read, and
with Transformers 5 its fast class rebuilds the tokeniser from the bare vocabulary, without tokenizer.json's byte-level
pre-tokeniser and decoder, so text is split into single characters. This copies the tokeniser files at the pinned
revision, selects the fast class and builds it from tokenizer.json. vLLM loads the result through `--tokenizer`
(configs/models.toml); the model's weights, config and processor still come from Kakao's repository.

Run on the GPU machine (serve/download_models.sh does it after downloading Kanana):
    uv run --project serve python serve/prepare_tokenizers.py
"""

import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
KANANA = "kakaocorp/kanana-1.5-v-3b-instruct"
KANANA_REVISION = "2e00ef13ccec2e99459a8eada18a1bfd05bff44b"

ORIGINAL_INIT = """class KananaVTokenizerFast(PreTrainedTokenizerFast, KananaVTokenizerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
"""
FIXED_INIT = """class KananaVTokenizerFast(PreTrainedTokenizerFast, KananaVTokenizerMixin):
    def __init__(self, **kwargs):
        # Local fix (vlm-street-audit, 2026-11-09): with Transformers 5 this class rebuilt its tokeniser from the bare
        # vocabulary, without tokenizer.json's byte-level pre-tokeniser and decoder, so text was split into single
        # characters. Building it from tokenizer.json keeps both.
        tokenizer_json = os.path.join(kwargs.get("name_or_path") or "", "tokenizer.json")
        if "tokenizer_object" not in kwargs and os.path.isfile(tokenizer_json):
            from tokenizers import Tokenizer
            kwargs["tokenizer_object"] = Tokenizer.from_file(tokenizer_json)
        super().__init__(**kwargs)
"""


def kanana(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "tokenization.py", "config.json"):
        shutil.copyfile(hf_hub_download(KANANA, name, revision=KANANA_REVISION), out / name)
    config = json.loads((out / "tokenizer_config.json").read_text(encoding="utf-8"))
    config["tokenizer_class"] = "KananaVTokenizerFast"  # the repository names its slow class
    (out / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    code = (out / "tokenization.py").read_text(encoding="utf-8")
    if ORIGINAL_INIT not in code:
        raise SystemExit("tokenization.py differs from the reviewed revision; check it before patching")
    code = code.replace("import logging\n", "import logging\nimport os\n", 1).replace(ORIGINAL_INIT, FIXED_INIT)
    (out / "tokenization.py").write_text(code, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    kanana(ROOT / "data" / "tokenizers" / "kanana-1.5-v-fast")
