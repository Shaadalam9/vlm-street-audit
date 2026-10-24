# AGENTS.md: conventions for anyone (or any coding agent) working on this repo

[README.md](README.md) describes the project. These rules apply to all code and text here.

## Rules

- **British English** everywhere: code comments, docstrings, identifiers, log and error
  messages, figure titles and labels, docs and commit messages (colour, analyse, localise,
  behaviour, centre, metre, licence as a noun). Library names keep their own spelling
  (`color=` in Plotly, `normalize` in a third-party API).
- **Plotly for every figure.** No matplotlib or seaborn. Build figures with
  `plotly.graph_objects` or `plotly.express`, style them with `vlm_street_audit.figures.style`
  and save them with `vlm_street_audit.figures.save_figure` (HTML in `outputs/figures/`,
  static copies through Kaleido when asked for).
- **uv for everything**: `uv add` / `uv remove` for dependencies (commit `uv.lock` with
  `pyproject.toml`), `uv run` to run code, `uv sync --frozen` on the GPU machine. Never bare pip.
- **On-device models only.** All inference runs locally (vLLM, Hugging Face `transformers`,
  Ollama); no cloud model APIs. Footage and frames never leave the machine. The models are
  listed in `configs/models.toml`.
- **Secrets stay out of git.** File-server credentials go in `secret` (copied from
  `default.secret`), local settings in `config` (copied from `default.config`); both are
  ignored. Never print, log or commit their contents.
- **Do not change the city sample casually.** `data/city_sample/` is a probability sample
  (n = 75, alpha = 0.5, seed 20260818). Redrawing changes most cities; never swap a city for
  a convenient neighbour.

## Layout

```
src/vlm_street_audit/   package: config, file_server (CROWD videos and track CSVs), mapping,
                        snippets + semantics (choosing and cutting snippets), figures
scripts/                sample_cities.py, fetch_data.py, make_snippets.py, plot_city_sample.py
configs/                models.toml (the VLMs), snippet_prompts.json (snippet ranking)
data/                   mapping, city sample; videos/frames/bbox are downloaded, not tracked
tests/                  uv run pytest
```

## Data access

Videos and YOLO track CSVs come from the CROWD file server (`files.mobility-squad.com`, an
authenticated HTTP server; its login is called `ftp_username` / `ftp_password`). Use
`FileServer.from_config()`: it takes local drives first (`video_dirs`, `bbox_dirs`, present on
lin4080), then the download folders, then the server. The sample holds ~5,500 videos, so
fetch a selection (`scripts/fetch_data.py --videos-per-city N`), not everything.

## Snippets

`scripts/make_snippets.py` follows opticarvis-global's candidate selection: windows inside
the mapped segments, SigLIP2 keyframe margins over routine-scene prompts, packet energy
as tie break, one window per event, plus random windows. Keep the random windows: they
are the unbiased part of the sample. Changing `configs/snippet_prompts.json` changes which
snippets the VLMs see, so record such changes in the commit message.
