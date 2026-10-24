"""Settings from `default.config` (tracked) and `config` (local), credentials from `secret`.

`config` is optional and may override any setting of `default.config`; an unknown
setting in it is an error, so a typo cannot silently fall back to the default.
`secret` is never committed: copy `default.secret` to `secret` and fill it in.
"""

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    settings = json.loads((root / "default.config").read_text(encoding="utf-8"))
    local = root / "config"
    if local.is_file():
        overrides = json.loads(local.read_text(encoding="utf-8"))
        unknown = sorted(set(overrides) - set(settings))
        if unknown:
            raise KeyError(f"Unknown settings in 'config': {unknown}")
        settings.update(overrides)
    return settings


def project_path(value: str, root: Path = PROJECT_ROOT) -> Path:
    """A path setting, relative to the project root unless absolute."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_secret(root: Path = PROJECT_ROOT) -> dict[str, str]:
    path = root / "secret"
    if not path.is_file():
        raise FileNotFoundError("No 'secret' file: copy 'default.secret' to 'secret' and fill it in.")
    return {key: str(value or "") for key, value in json.loads(path.read_text(encoding="utf-8")).items()}
