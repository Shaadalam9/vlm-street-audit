# VLM Street Audit

Using vision-language models (VLMs) to analyse real-world street footage, identify
pedestrian-related problems, localise them in the scene, and suggest street improvements.

## Background

The original concept was a browser-based interface in which people pause street footage,
mark a location in the frame, and describe an observed pedestrian issue and a possible
intervention. The project has since moved away from human participants: the task is now
carried out entirely by VLMs/LLMs.

## Goal

Given street footage, a VLM should:

1. **Detect** pedestrian or infrastructure issues (e.g. missing or blocked pavements,
   unsafe crossings, poor visibility, conflicts with vehicles).
2. **Localise** each issue in the frame (timestamp + bounding box or point).
3. **Describe** the issue in plain language.
4. **Suggest** a concrete improvement or intervention.

## On-device models only

All model inference runs locally (e.g. Hugging Face `transformers` or Ollama).
No cloud APIs: footage and frames never leave the machine.

### Models (RTX 5090, 32 GB)

Several VLMs from different developers do the same task on the same frames, one
model at a time. Details and checkpoints in [`configs/models.toml`](configs/models.toml).

| Model | Developer | Run as | Weights | Why it is in the set |
|---|---|---|---|---|
| Qwen3.8-27B | Alibaba | 4-bit (W4A16) | ~17 GB | Newest and strongest general model; images and video |
| Gemma 4 31B | Google | 4-bit (Google's QAT) | ~19 GB | Second large model from another lineage; detection and pointing |
| Molmo2-8B | Ai2 | BF16 | ~18 GB | Localisation specialist (points, video tracks); open training data |
| InternVL3.5-8B | Shanghai AI Lab | BF16 | ~17 GB | Fourth lineage; an older baseline |

- 27-31B models do not fit 32 GB in BF16 (~55-62 GB), so they run as 4-bit checkpoints.
- Served with vLLM (Linux/WSL2, CUDA 12.8+ for Blackwell) on localhost through its
  OpenAI-compatible API, so one client works for every model. Ollama works too, but its
  GGUF quantisations and image resizing differ per model, which makes coordinates harder
  to compare.
- Fixed per run: frames, prompt, image resolution, temperature 0, and each model's
  checkpoint revision.

## Research questions (draft)

- How well can VLMs identify and localise pedestrian-related problems in real-world street footage?
- How plausible, specific, and actionable are the improvement suggestions they generate?
- How consistent are these observations across models, prompts, and repeated runs?
- Can VLMs (or LLM judges) review, correct, or refine each other's observations, standing in for the human review step?

## Setup

The project uses [uv](https://docs.astral.sh/uv/) (Python 3.12):

```bash
uv sync
cp default.secret secret      # then fill in the file-server login (never commit it)
uv run pytest                 # tests, no network needed
```

Settings live in `default.config`; to change one locally, copy it to `config` and edit
that (an unknown setting there is an error). Code and text follow [AGENTS.md](AGENTS.md):
British English, Plotly for every figure, uv for every package operation.

## Getting the footage

Videos and CROWD's YOLO track CSVs come from the CROWD file server
(`files.mobility-squad.com`). It is an authenticated HTTP server, though its login is
stored as `ftp_username` / `ftp_password` in `secret`, as in the other CROWD projects.
`vlm_street_audit.file_server.FileServer` looks in this order:

1. the local drives (`video_dirs`, `bbox_dirs`; they exist on lin4080),
2. the download folders (`data/videos/`, `data/bbox/`),
3. the server: the file's URL under each alias (`tue4`, `tue5`, `tue3`, `tue2`, `tue1` for
   videos, `data` for track files), then a walk of the server's folder pages.

Downloads are written to a `.part` file and renamed only when complete.

The 75 cities hold about 5,500 videos, so fetch a selection: the videos with the most
qualifying footage per city.

```bash
uv run python scripts/fetch_data.py --dry-run                   # what would be fetched
uv run python scripts/fetch_data.py --videos-per-city 1 --tracks
uv run python scripts/fetch_data.py --country Japan --videos-per-city 3
```

## Figures

All figures are made with Plotly and saved by `vlm_street_audit.figures.save_figure` to
`outputs/figures/` as HTML (static PNG/PDF through Kaleido on request).

```bash
uv run python scripts/plot_city_sample.py    # world map of the drawn cities
```

## Choosing the cities

Cities, and with them the countries, are drawn from the CROWD mapping
(`data/mapping_original.csv`, copied from
[opticarvis-global](https://github.com/Shaadalam9/opticarvis-global/blob/main/docs/mapping_original.csv))
with the same design as opticarvis-global (see its
[docs/CITY_SAMPLING.md](https://github.com/Shaadalam9/opticarvis-global/blob/main/docs/CITY_SAMPLING.md)):
a spatially balanced probability sample drawn with the Local Pivotal Method
([lpm-sampling](https://github.com/M-Colley/lpm-sampling)), with inclusion
probability proportional to population^0.5 (opticarvis-global uses 0.75; at 75
cities 0.5 nearly doubles the effective sample size, 42 vs 24, for ~8% sparser coverage).

- Eligibility comes first: at least 1 h of daytime footage of vehicle type 0, valid
  coordinates and population. Filtering after the draw would break the probabilities.
- No one-city-per-country rule: large countries get several cities, small ones at most one.
- `1/pi` is each city's design weight, kept in `design_manifest.csv`.
- A drawn city with no usable footage is reported as missing, never swapped for a neighbour.

```bash
uv run python scripts/sample_cities.py            # defaults: n=75, seed 20260818, alpha 0.5, LPM2
uv run python scripts/sample_cities.py --help     # change n, alpha, seed, method, filters
```

The design follows opticarvis-global, which draws 150; this project draws 75 cities
(43 countries with the defaults).
Output in `data/city_sample/`:

| File | Contents |
|---|---|
| `mapping.csv` | The drawn cities in mapping format, holding only the qualifying footage |
| `design_manifest.csv` | The drawn cities with `pi` and `design_weight` |
| `eligible_frame_with_pi.csv` | Every eligible city with its inclusion probability |
| `selected_cities_by_country.csv` | Cities drawn per country |
| `diagnostics.txt` | Spatial balance, nearest-city distance, effective sample size, weights |

## Structure

```
configs/       models.toml (the VLMs and how they run)
scripts/       sample_cities.py, fetch_data.py, plot_city_sample.py
src/vlm_street_audit/       config, file_server, mapping, figures
tests/         pytest, no network
default.config / default.secret   settings and an empty login (local copies: config, secret)
data/mapping_original.csv   the CROWD mapping (sampling frame)
data/city_sample/           the drawn sample and its design files
data/videos/   downloaded footage (not tracked in git)
data/bbox/     downloaded YOLO track CSVs (not tracked in git)
data/frames/   extracted frames (not tracked in git)
prompts/       prompt templates for detection, localisation and suggestions
notebooks/     exploration and analysis
outputs/       model responses and results (not tracked in git)
```

## Licence

MIT, see [LICENSE](LICENSE). Security reports: see [SECURITY.md](SECURITY.md).
