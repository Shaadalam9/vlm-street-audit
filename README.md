# What Pedestrians Lack Around the World

**Auditing street infrastructure in 75 cities with on-device vision-language models**

[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20with-uv-6340ac.svg)](https://docs.astral.sh/uv/)
[![Paper: IHIET-AI 2027](https://img.shields.io/badge/paper-IHIET--AI%202027-2a78d6.svg)](#citation)

Code, model answers and analysis for the paper

> Md Shadab Alam, Alexandros Rouchitsas and Pavlo Bazilinskyy (2027).
> *What Pedestrians Lack Around the World: Auditing Street Infrastructure in 75 Cities with On-Device
> Vision-Language Models.* Human Interaction and Emerging Technologies (IHIET-AI 2027), Palma de Mallorca,
> Spain. Accepted for oral presentation.

Six open vision-language models (VLMs) from four countries, run locally on one GPU, audit pedestrian
infrastructure in dashcam clips from 75 cities on six continents. They code each street with items from two established instruments,
the [iRAP](https://irap.org) road attributes and the MAPS-Mini streetscape audit, list what is missing, point
to it in the frame and suggest a change to the street. No footage leaves the machine.

![What pedestrians lack in each of the 75 cities, with six of the streets](docs/figures/paper/fig1_world.png)

**Figure 1.** What pedestrians lack in each of the 75 cities (needs on which at least four of the six models agree
in at least one of the city's two clips), with six of the streets and the point each model gave (ring) on the
frame it named. Suggestions: (a) a continuous sidewalk; (b) a marked crossing with dropped kerbs at the sign;
(c) remove the cart and debris to clear the path; (d) regulating the stalls and a no-parking zone for motorcycles;
(e) a marked crossing with a central refuge island. (f) A false report: the bridge's footway runs behind the barrier
and fence, separated from the roadway. Three of six models made it, short of the 4-of-6 consensus.

## Contents

- [How it works](#how-it-works)
- [Main findings](#main-findings)
- [Data and materials](#data-and-materials)
- [Models](#models)
- [Reproducing the study](#reproducing-the-study)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [Licence](#licence)

## How it works

```mermaid
flowchart LR
    A["CROWD dashcam mapping<br/>(cities worldwide)"] --> B["75 cities<br/>spatially balanced<br/>probability sample"]
    B --> C["150 city clips<br/>10 s, 5 frames<br/>(best-ranked + random)"]
    J["120 JAAD clips<br/>with human labels"] --> D
    C --> D["6 VLMs on one GPU<br/>iRAP + MAPS-Mini items<br/>JSON schema enforced"]
    D --> E["Needs, points<br/>and suggestions"]
    D --> R["Robustness runs<br/>reworded, 1 frame,<br/>640 px, T = 0.7"]
    E --> F["2 VLM judges<br/>rate each need"]
    D --> G["Analysis<br/>consensus, κ,<br/>bootstrap over cities"]
    R --> G
    F --> G
```

1. **Cities.** 75 cities in 43 countries are drawn from the CROWD mapping of dashcam videos to cities with the
   Local Pivotal Method, a spatially balanced probability sample, with inclusion probability proportional to
   the square root of population ([`scripts/data/sample_cities.py`](scripts/data/sample_cities.py)).
2. **Clips.** For each city, the video with the most daytime car footage is cut into 10 s windows. Windows are
   ranked with SigLIP2 image–text embeddings for pedestrian-relevant scenes, and one random window is added,
   so each city gives one best-ranked and one random clip ([`scripts/data/make_snippets.py`](scripts/data/make_snippets.py),
   [`scripts/data/build_clip_sets.py`](scripts/data/build_clip_sets.py)). 120 clips from the JAAD dataset, with human labels,
   check validity.
3. **Audit.** Each model sees five frames, 2 s apart, and codes the street as a road-safety auditor would code a
   100 m segment: road, each side's footway, and the crossing (iRAP and MAPS-Mini items). From the item answers,
   eight needs follow by fixed rules, and the model lists each one with evidence, a point on the frame, a standard
   measure and a one-sentence suggestion. vLLM enforces the JSON schema during decoding, so every answer
   holds every item ([`src/vlm_street_audit/vlm.py`](src/vlm_street_audit/vlm.py)).
4. **Robustness.** The audit is repeated with a reworded prompt, one frame instead of five, half the
   resolution (640 px) and twice at temperature 0.7 ([`scripts/audit/run_vlm.py`](scripts/audit/run_vlm.py)).
5. **Judges.** The two larger models rate every listed need: is the deficit visible, would the suggestion fix it,
   is it specific to the street, is it feasible, and does it tell people how to behave
   ([`scripts/audit/judge_suggestions.py`](scripts/audit/judge_suggestions.py)).
6. **Analysis.** An item counts for a clip when at least two thirds of the models agree (4 of 6). Cities are
   the unit, with 95% intervals from a bootstrap over cities ([`scripts/analysis/analyse_audit.py`](scripts/analysis/analyse_audit.py)).

An answer, abridged (Qwen3.8-27B on the Kinshasa clip of Figure 1e; the full file is
[`outputs/vlm/audit/cities/qwen3.8-27b/Q3WvZW9zBk8_00851000_010000.json`](outputs/vlm/audit/cities/qwen3.8-27b/Q3WvZW9zBk8_00851000_010000.json)):

```json
{
  "road": {"area_type": "urban", "land_use": "mixed", "number_of_lanes": "2", "street_lighting": "none", "...": "..."},
  "left_side": {"footway": "informal_path_1m_or_more", "obstruction": "partly_blocked", "trip_hazards": "some"},
  "crossing": {"crossing_demand": true, "crossing_facility": "none", "crossing_quality": "not_applicable",
               "kerb_ramps": "no_crossing"},
  "needed": [
    {"need": "no_crossing", "where": "road", "frame": 1, "x": 500, "y": 750,
     "evidence": "There is a high demand for crossing due to the market stalls and buildings, but no marked or purpose-built crossing is present.",
     "suggestion": "Install a marked pedestrian crossing with a central refuge island to provide a safe place for people to cross the road."},
    "..."
  ]
}
```

Points are integers from 0 to 1000 on the named frame, x from the left and y from the top.

## Main findings

**The crossing item is valid where it can be checked.** On JAAD's labelled clips, the iRAP crossing-facility
item finds marked crossings with a balanced accuracy of 0.86–0.92 for five of the six models; the smallest,
Kanana-1.5-V 3B, reaches only 0.57.

**A crossing where people have reason to cross is the most common gap.** By consensus, street lighting is the most
common feature (83% of cities), a formal footway on both sides is found in 51%, and a crossing facility in only 22%.
In 39 of the 75 cities at least one clip needs a crossing, in 22 a footway, and in 12 both (Figure 1).

**The gaps differ by continent.** A crossing is needed in 57% of African cities and 23% of North American ones. A
footway is needed in 50% of African and 23% of North American cities, against 10–12% in Europe and Asia: in Africa
basic provision is often absent, while many North American roads are built without footways.

**Needs follow road deaths.** The number of needs per clip rises with the country's road deaths per 100,000
(Spearman's ρ = 0.33, 95% CI 0.11 to 0.52) and falls with median age (−0.43) and literacy (−0.44). These are
country-level figures that track national income, so they describe a pattern, not a cause.

![The gaps by continent, and against road deaths](docs/figures/paper/fig2_gaps.png)

**Figure 2.** (a) Share of cities needing a crossing where people have reason to cross, and a footway, by continent
(number of cities in brackets; Oceania, one city, is left out). (b) Missing items per street clip (consensus of the
six models, averaged over each city's two clips) against the country's road deaths per 100,000 people, for 73
cities; the line joins the means of each quarter of the cities.

**The models agree somewhat less outside Europe and North America, and not at all on footway type.** Agreement on
the crossing items is lower for clips from the rest of the world (reason to cross κ = 0.23 against 0.35; the
intervals overlap). Footway type (κ = 0.00), trip hazards and kerb ramps are not reliable in any region. On one
Kingston market street, all six models see people walking in the road, a reason to cross and no crossing, but give
five different answers for each footway.

![One street in Kingston, as coded by the six models](docs/figures/paper/fig3_one_street.png)

**Figure 3.** One street in Kingston, Jamaica, as coded by the six models: they agree on what is easy to see and
give five different answers for each footway.

![Agreement between the models, by region](docs/figures/paper/fig4_agreement.png)

**Figure 4.** Agreement between the six models on each item (Fleiss' κ) for clips from Europe and North America (60)
and from the rest of the world (90). The shaded band marks agreement below 0.2.

**Only the two largest models are sound auditors.** The judges see 81% of the needs listed by Qwen3.8 and Gemma 4
in the frames, against 28–31% for Molmo2, InternVL3.5 and Ministral 3, and Gemma 4's needs always follow from its
own items; Kanana lists no needs at all. Rewording the prompt or halving the resolution changes few answers (mean κ
0.68 and 0.73); one frame (0.56) and resampling (0.50) change more.

<p align="center"><img src="docs/figures/paper/fig5_models.png" width="560" alt="The models as auditors"></p>

**Figure 5.** The five models that listed needs, as auditors: share of their listed needs that the judges rated
visible in the frames (mean of two judges) against the share backed by the model's own item answers.

All tables (with 95% intervals) are in [`outputs/vlm/analysis_audit/tables.md`](outputs/vlm/analysis_audit/tables.md).
Further figures, each with an interactive HTML version, are in [`docs/figures/audit/`](docs/figures/audit/):

| Figure | Shows |
|---|---|
| [`jaad_validity`](docs/figures/audit/jaad_validity.png) | Scene facts and the crossing item against JAAD's labels, per model |
| [`present_by_continent`](docs/figures/audit/present_by_continent.png) | Infrastructure present, by continent |
| [`needed_by_continent`](docs/figures/audit/needed_by_continent.png) | Infrastructure needed, by continent |
| [`country_indicators`](docs/figures/audit/country_indicators.png) | Spearman's ρ with road deaths, median age, Gini and literacy |
| [`agreement_by_region`](docs/figures/audit/agreement_by_region.png) | Fleiss' κ per item and region, with intervals |
| [`robustness`](docs/figures/audit/robustness.png) | Cohen's κ under each changed condition |
| [`suggestion_quality`](docs/figures/audit/suggestion_quality.png) | Judges' ratings per auditor model |

## Data and materials

| What | Where |
|---|---|
| Video clips (150 city clips, 10 s each), their frames, and a snapshot of the code | [Dropbox folder](https://www.dropbox.com/scl/fo/cl1e4cgdjlj43i5gy4gsy/ALxGI5IgWYm9RRxrfey5AYA?rlkey=cu2udsf5ooey7cp2ff6efu4ut&st=zjngk7m6&dl=0) |
| Every model answer on the city clips: main audit and the four robustness runs (5,400 files) | [`outputs/vlm/<run>/cities/<model>/`](outputs/vlm/) |
| Every model answer on the JAAD clips: main audit, reworded, one frame, half resolution (2,880 files) | [`outputs/vlm/<run>/jaad/<model>/`](outputs/vlm/) |
| The clip sets and JAAD's labels for the 120 JAAD clips | [`data/clip_sets/`](data/clip_sets/) |
| The seven frames shown in Figures 1 and 3 | [`docs/figures/paper/frames/`](docs/figures/paper/frames/) |
| Every judge rating (1,206 rated answers, 1,874 needs rated by both judges) | [`outputs/vlm/judge/<judge>/<auditor>/`](outputs/vlm/judge/) |
| The tables behind every number, as CSV | [`outputs/vlm/analysis_audit/`](outputs/vlm/analysis_audit/) |
| The city sample and its design (inclusion probabilities, weights, diagnostics) | [`data/city_sample/`](data/city_sample/) |
| The sampling frame (CROWD mapping of videos to cities) | [`data/mapping_original.csv`](data/mapping_original.csv) |

Each answer file holds the model's checkpoint and revision, the prompt version (a hash of prompt and schema), the
condition (temperature, seed, frames, image size), the frame paths, the raw text and the parsed answer. A clip's
id is `<YouTube video id>_<start ms>_<length ms>`, so each clip can be traced to its source video. The footage itself
comes from YouTube via the CROWD dataset; this repository holds no video. JAAD's clips and labels come from
[the JAAD dataset](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/) (Rasouli et al., 2017; annotations under the
MIT licence, clips under CC BY 4.0); `data/clip_sets/jaad_labels.json` holds its labels for the 120 sampled clips,
and the JAAD frames themselves are not included.

## Models

All models run on one NVIDIA RTX 5090 (32 GB) with [vLLM](https://github.com/vllm-project/vllm) through its
OpenAI-compatible API on localhost, at temperature 0, each pinned to a checkpoint revision in
[`configs/models.toml`](configs/models.toml).

| Model | Developer (country) | Run as | Checkpoint |
|---|---|---|---|
| Qwen3.8-27B | Alibaba (China) | 4-bit, W4A16 | [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) |
| Gemma 4 31B | Google (US) | 4-bit, quantisation-aware | [google/gemma-4-31B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) |
| Molmo2-8B | Allen Institute for AI (US) | BF16 | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) |
| InternVL3.5-8B | Shanghai AI Laboratory (China) | BF16 | [OpenGVLab/InternVL3_5-8B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) |
| Ministral 3 14B | Mistral AI (France) | FP8, the developer's release | [mistralai/Ministral-3-14B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) |
| Kanana-1.5-V 3B | Kakao (South Korea) | BF16 | [kakaocorp/kanana-1.5-v-3b-instruct](https://huggingface.co/kakaocorp/kanana-1.5-v-3b-instruct) |

Ministral runs with Mistral's own tokeniser and weight format, as Mistral documents it; with the Hugging Face
tokeniser its answers were valid JSON but nonsense. Kanana runs with a fast tokeniser built by
[`serve/prepare_tokenizers.py`](serve/prepare_tokenizers.py), since its own cannot be read by the structured-output
engine.

Six further models were tried and excluded (details in [`configs/models.toml`](configs/models.toml)):

| Model | Developer (country) | Why excluded |
|---|---|---|
| EXAONE 4.5 33B | LG AI Research (South Korea) | Answered text correctly but returned nonsense for every image (vLLM's own code); every image request failed through the Transformers backend |
| HyperCLOVA X SEED Vision 3B | Naver (South Korea) | Not supported by vLLM after 0.28, and its config does not load in Transformers 5 |
| Aya Vision 8B | Cohere (Canada) | Degenerate audit: 148 of 150 city answers listed all eight needs, none with a point; JAAD at chance (balanced accuracy 0.50–0.58) |
| North-Micro-Vision | Cohere (Canada) | Degenerate audit: every need at point (0, 0); JAAD zebra crossings at chance (0.57) |
| Chitrarth | Krutrim (India) | Not supported by vLLM; served with its own code, one image only. Degenerate audit: no footway and no lighting in all 120 JAAD clips, every need at point (0, 0); crossing item at chance (0.50) |
| Falcon2-11B-VLM | TII (UAE) | Cannot be served by vLLM 0.32 (FP8 fails in its image projector; neither vLLM's own nor the Transformers implementation runs its Falcon language model) |

## Reproducing the study

### Setup

The project uses [uv](https://docs.astral.sh/uv/) with Python 3.12; vLLM has its own environment in
[`serve/`](serve/).

```bash
uv sync
uv run pytest                  # tests, no network or GPU needed
cd serve && uv sync            # vLLM, on the GPU machine (Linux or WSL2, CUDA 12.8+ for Blackwell)
bash download_models.sh       # the weights at the pinned revisions (~130 GB)
```

[`serve/download_models.sh`](serve/download_models.sh) fetches exactly the checkpoints and revisions in [`configs/models.toml`](configs/models.toml)
(pass model keys to fetch only some) and builds the local tokeniser Kanana needs. Gated models need their terms
accepted on Hugging Face and `uv run --project serve hf auth login` first.

Settings are in [`default.config`](default.config); to change one locally, copy it to `config` and edit that (an
unknown setting is an error). Downloading CROWD footage needs a login to the CROWD file server: copy
`default.secret` to `secret` and fill it in (both `config` and `secret` are ignored by git).

### The analysis from the published answers

Everything the analysis reads is in this repository (the answers, the judge ratings, the clip sets with JAAD's
labels, the city sample and the frames shown in the figures), so a fresh clone recomputes every table and figure of
the paper and this README, without a GPU or the footage:

```bash
uv sync
uv run python scripts/analysis/analyse_audit.py    # tables in outputs/vlm/analysis_audit/, figures in docs/figures/audit/
uv run python scripts/analysis/paper_figures.py    # the paper's five figures in docs/figures/paper/
git status                                          # nothing changes: the tables and figures are the published ones
```

The bootstraps use fixed seeds, so the tables come out identical; the figures do too where Arial is installed (other
fonts shift the PNGs' pixels, not their content). Static PNGs need a Chromium browser for Kaleido
(set `BROWSER_PATH` if Chrome is not installed, or pass `--no-png` to `analyse_audit.py` for HTML only), and the
maps need an internet connection for Plotly's world outlines.

### The full pipeline

```bash
uv run python scripts/data/sample_cities.py                         # draw the 75 cities (seed 20260818, alpha 0.5)
uv run python scripts/data/fetch_data.py --videos-per-city 1        # fetch each city's video with the most footage
uv run python scripts/data/make_snippets.py --videos-per-city 1     # rank 10 s windows, cut clips and frames
uv run python scripts/data/build_clip_sets.py                       # city and JAAD clip sets
uv run python scripts/audit/run_vlm.py --start-server --models all --sets jaad cities          # main audit
uv run python scripts/audit/run_vlm.py --start-server --prompt audit_reworded --tag audit-reworded
uv run python scripts/audit/run_vlm.py --start-server --tag audit-1frame --frames 3
uv run python scripts/audit/run_vlm.py --start-server --tag audit-640px --max-side 640
uv run python scripts/audit/run_vlm.py --start-server --tag audit-t07-s1 --temperature 0.7 --seed 1 --sets cities
uv run python scripts/audit/run_vlm.py --start-server --tag audit-t07-s2 --temperature 0.7 --seed 2 --sets cities
uv run python scripts/audit/judge_suggestions.py --start-server       # Qwen3.8 and Gemma 4 rate every listed need
uv run python scripts/analysis/analyse_audit.py
uv run python scripts/analysis/paper_figures.py
```

`run_vlm.py --start-server` starts vLLM for each model in turn, answers every clip and stops it; answered clips
are skipped, so a stopped run resumes where it left off. On one RTX 5090, each model's share of the audit (1,380
answers) takes about half a day to a day.

<details>
<summary><b>How the cities are drawn</b></summary>

Cities are drawn from the CROWD mapping ([`data/mapping_original.csv`](data/mapping_original.csv), from
[opticarvis-global](https://github.com/Shaadalam9/opticarvis-global)) with the Local Pivotal Method
([lpm-sampling](https://github.com/M-Colley/lpm-sampling)), a spatially balanced probability sample, with inclusion
probability proportional to population^0.5 (at 75 cities, 0.5 nearly doubles the effective sample size against 0.75,
42 against 24, for about 8% sparser coverage).

- Eligibility comes first: at least 1 h of daytime car footage, valid coordinates and population. Filtering after the
  draw would break the probabilities.
- No one-city-per-country rule: large countries get several cities, small ones at most one.
- `1/pi` is each city's design weight, kept in `design_manifest.csv`.
- A drawn city with no usable footage would be reported as missing, never swapped for a neighbour (none was).

| File in `data/city_sample/` | Contents |
|---|---|
| `mapping.csv` | The drawn cities in mapping format, holding only the qualifying footage |
| `design_manifest.csv` | The drawn cities with `pi` and `design_weight` |
| `eligible_frame_with_pi.csv` | Every eligible city with its inclusion probability |
| `selected_cities_by_country.csv` | Cities drawn per country |
| `diagnostics.txt` | Spatial balance, nearest-city distance, effective sample size, weights |

</details>

<details>
<summary><b>How the clips are chosen</b></summary>

1. **Windows**: 10 s windows every 5 s inside each mapped daytime car segment.
2. **Semantic rank**: SigLIP2 (`google/siglip2-base-patch16-224`) embeds keyframes at least 2 s apart. Each window
   is scored by how much the prompt groups in [`configs/snippet_prompts.json`](configs/snippet_prompts.json)
   (footway condition, obstruction, crossings, pedestrian exposure, accessibility, vulnerable users) beat the best
   routine-scene prompt (motorway, countryside, tunnel, tidy street) on the same keyframes.
3. **Tie break**: packet energy, the encoded bytes per second from ffprobe.
4. **One window per event**: the best window claims its neighbours within 30 s.
5. **Per video**: the best-ranked events plus one random window. The random windows are not chosen for problems,
   so they show what an ordinary stretch looks like and keep prevalence estimates honest.

Each snippet is written as an H.264 clip and JPEG frames, with every snippet's scores in `data/snippets/snippets.csv`.
Downloaded videos are deleted once cut (`--keep-downloads` keeps them). All numbers are settings in
`default.config` (`snippet_*`, `semantic_*`).

</details>

## Repository structure

```
configs/
  models.toml              the VLMs: checkpoints, revisions, serving settings
  snippet_prompts.json     prompts that rank the 10 s windows
data/
  mapping_original.csv     the sampling frame (CROWD mapping)
  city_sample/             the drawn cities and their design files
docs/figures/
  paper/                   the paper's six figures (PNG and interactive HTML)
  audit/                   the further analysis figures
outputs/vlm/
  audit*/<set>/<model>/    the model answers (main run and robustness runs; JAAD and city clips)
  judge/<judge>/<auditor>/ the judges' ratings
  analysis_audit/          every table, as CSV and Markdown
scripts/
  data/                    sample_cities, plot_city_sample, fetch_data, make_snippets, build_clip_sets
  audit/                   run_vlm (the audit), judge_suggestions (the judges)
  analysis/                analyse_audit (tables and figures), paper_figures
serve/                     vLLM environment (run on the GPU machine): serve.py starts one model,
                           download_models.sh fetches the pinned weights, prepare_tokenizers.py
src/vlm_street_audit/      prompt and schema (vlm), vLLM start and stop (server), statistics (evaluation),
                           snippets, semantics, JAAD labels, CROWD file server, configuration, figure style
tests/                     pytest, no network or GPU
```

Downloaded videos, snippets, frames and logs stay out of git. Code and text follow [AGENTS.md](AGENTS.md):
British English, Plotly for every figure, uv for every package operation, on-device models only.

## Citation

If you use this code or data, please cite the paper:

```bibtex
@inproceedings{alam2027pedestrians,
  author    = {Alam, Md Shadab and Rouchitsas, Alexandros and Bazilinskyy, Pavlo},
  title     = {What Pedestrians Lack Around the World: Auditing Street Infrastructure in 75 Cities
               with On-Device Vision-Language Models},
  booktitle = {Human Interaction and Emerging Technologies (IHIET-AI 2027): Artificial Intelligence
               and Future Applications},
  address   = {Palma de Mallorca, Spain},
  publisher = {AHFE International},
  year      = {2027},
  note      = {Accepted for publication}
}
```

Citation metadata is also in [`CITATION.cff`](CITATION.cff).

## Licence

The code is released under the MIT licence, see [LICENSE](LICENSE). The model answers and ratings are released under
the same terms. The video footage belongs to its creators on YouTube and is not redistributed here. Security reports:
see [SECURITY.md](SECURITY.md).

Contact: Md Shadab Alam, Eindhoven University of Technology, [m.s.alam@tue.nl](mailto:m.s.alam@tue.nl).
