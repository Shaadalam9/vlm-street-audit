120 JAAD clips and 150 city clips (75 cities) answered by all four models.

### JAAD: balanced accuracy against the labels

| Item | Qwen3.8-27B | Gemma 4 31B | Molmo2-8B | InternVL3.5-8B |
|---|---|---|---|---|
| Zebra crossing | 0.92 | 0.90 | 0.92 | 0.91 |
| Traffic signal | 0.98 | 0.73 | 0.82 | 0.95 |
| Pedestrians visible | 0.90 | 0.99 | 0.98 | 0.99 |
| Pedestrians in road | 0.71 | 0.64 | 0.67 | 0.62 |
| Crossing facility (iRAP item) | 0.92 | 0.88 | 0.92 | 0.89 |

The crossing-facility item counts a marked, signalised or raised crossing and is scored against JAAD's crossing label.


### Infrastructure present, by continent (share of cities; at least 3 of 4 models agree)

|  | Asia (32) | Europe (15) | North America (15) | Africa (7) | South America (5) | Oceania (1) | All (75) |
|---|---|---|---|---|---|---|---|
| Formal footway on both sides | 42% (27%–58%) | 57% (40%–73%) | 43% (23%–63%) | 7% (0%–21%) | 50% (20%–80%) | 50% | 43% (33%–51%) |
| Footway set back ≥ 1 m or behind a barrier | 52% (36%–67%) | 60% (40%–80%) | 47% (27%–67%) | 29% (7%–43%) | 60% (30%–90%) | 50% | 51% (41%–59%) |
| Footways free of obstruction | 54% (37%–72%) | 42% (19%–65%) | 62% (38%–85%) | 62% (25%–100%) | 80% (60%–100%) | 100% | 57% (46%–67%) |
| Crossing facility | 22% (12%–33%) | 20% (7%–33%) | 30% (10%–53%) | 7% (0%–21%) | 50% (20%–80%) | 50% | 24% (17%–31%) |
| Signalised or grade-separated crossing | 9% (3%–17%) | 7% (0%–17%) | 17% (3%–33%) | 0% (0%–0%) | 20% (0%–40%) | 50% | 11% (6%–15%) |
| Refuge island | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 0% (0%–0%) |
| Kerb ramps at the crossing | 0% (0%–0%) | 33% (0%–100%) | 0% (0%–0%) | – | 0% (0%–0%) | – | 7% (0%–21%) |
| Street lighting | 94% (88%–98%) | 83% (67%–97%) | 70% (50%–87%) | 57% (36%–79%) | 80% (40%–100%) | 100% | 83% (75%–89%) |
| Traffic calming | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 0% (0%–0%) |
| Pedestrian fencing | 8% (2%–16%) | 3% (0%–10%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 4% (1%–8%) |
| Bike facility | 3% (0%–8%) | 0% (0%–0%) | 10% (0%–20%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 3% (1%–7%) |
| Transit stop | 12% (5%–20%) | 3% (0%–10%) | 3% (0%–10%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 7% (3%–11%) |

### Infrastructure needed, by continent (share of cities)

|  | Asia (32) | Europe (15) | North America (15) | Africa (7) | South America (5) | Oceania (1) | All (75) |
|---|---|---|---|---|---|---|---|
| Footway (none or informal) | 12% (5%–22%) | 10% (0%–20%) | 30% (17%–47%) | 50% (21%–79%) | 20% (0%–40%) | 50% | 20% (14%–27%) |
| Clear a blocked footway | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 0% (0%–0%) |
| Repair trip hazards | 5% (0%–12%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 10% (0%–30%) | 0% | 3% (0%–6%) |
| Crossing where people need one | 47% (33%–59%) | 40% (20%–63%) | 27% (7%–47%) | 64% (36%–93%) | 40% (10%–70%) | 0% | 42% (33%–51%) |
| Better crossing | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 0% (0%–0%) |
| Refuge on a wide road | 5% (0%–9%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 10% (0%–30%) | 0% | 3% (1%–5%) |
| Kerb ramps | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 0% (0%–0%) |
| Street lighting | 3% (0%–8%) | 7% (0%–17%) | 17% (3%–33%) | 21% (0%–50%) | 0% (0%–0%) | 0% | 8% (3%–13%) |

Cities are the unit: each city's two clips are averaged, and intervals come from a bootstrap over cities. Oceania is one city.


### Infrastructure and country indicators (Spearman's rho over cities, 95% bootstrap interval)

|  | Road deaths per 100k (n = 73) | Median age (n = 75) | Gini index (n = 74) | Literacy rate (n = 51) |
|---|---|---|---|---|
| Formal footway on both sides | -0.25 (-0.46 to -0.03) | 0.42 (0.19 to 0.61) | -0.13 (-0.34 to 0.09) | 0.50 (0.28 to 0.67) |
| Footway set back ≥ 1 m or behind a barrier | -0.20 (-0.42 to 0.04) | 0.37 (0.15 to 0.57) | -0.16 (-0.37 to 0.07) | 0.33 (0.08 to 0.55) |
| Footways free of obstruction | 0.03 (-0.23 to 0.28) | 0.01 (-0.27 to 0.30) | 0.01 (-0.25 to 0.26) | -0.03 (-0.37 to 0.30) |
| Crossing facility | -0.21 (-0.42 to 0.02) | 0.28 (0.07 to 0.48) | -0.01 (-0.23 to 0.23) | 0.19 (-0.06 to 0.43) |
| Signalised or grade-separated crossing | -0.02 (-0.24 to 0.20) | 0.04 (-0.15 to 0.23) | 0.09 (-0.14 to 0.31) | 0.18 (-0.08 to 0.41) |
| Street lighting | -0.24 (-0.43 to -0.03) | 0.35 (0.14 to 0.53) | -0.26 (-0.46 to -0.02) | 0.23 (-0.07 to 0.48) |
| Pedestrian fencing | 0.12 (-0.07 to 0.29) | 0.07 (-0.15 to 0.25) | -0.00 (-0.14 to 0.12) | 0.23 (0.06 to 0.41) |
| Bike facility | 0.11 (-0.03 to 0.26) | 0.06 (-0.05 to 0.17) | 0.17 (0.02 to 0.31) | – |
| Transit stop | -0.07 (-0.32 to 0.19) | 0.23 (0.04 to 0.39) | -0.27 (-0.44 to -0.09) | 0.34 (0.13 to 0.51) |
| Needed: Footway (none or informal) | 0.18 (-0.05 to 0.40) | -0.37 (-0.57 to -0.15) | 0.28 (0.06 to 0.48) | -0.55 (-0.73 to -0.30) |
| Needed: Crossing where people need one | 0.28 (0.06 to 0.48) | -0.30 (-0.50 to -0.09) | -0.05 (-0.29 to 0.19) | -0.36 (-0.58 to -0.10) |
| Needed: Street lighting | 0.19 (-0.01 to 0.36) | -0.24 (-0.43 to -0.02) | 0.14 (-0.09 to 0.35) | -0.25 (-0.49 to 0.06) |
| Number of needs | 0.41 (0.21 to 0.58) | -0.43 (-0.61 to -0.20) | 0.17 (-0.08 to 0.40) | -0.54 (-0.72 to -0.29) |

Country figures from the city sample (CROWD mapping); a positive rho means the item is more common where the indicator is higher.


### Agreement between the four models, by region (Fleiss' kappa, city clips, 95% interval)

| Item | All clips | Europe & North America (60 clips) | Rest of the world (90 clips) |
|---|---|---|---|
| Area type | 0.68 | 0.64 (0.42 to 0.78) | 0.73 (0.12 to 0.94) |
| Land use | 0.42 | 0.42 (0.28 to 0.54) | 0.40 (0.29 to 0.51) |
| Lanes | 0.33 | 0.26 (0.14 to 0.37) | 0.31 (0.21 to 0.41) |
| Median | 0.12 | 0.01 (-0.09 to 0.08) | 0.18 (0.09 to 0.25) |
| Street lighting | 0.35 | 0.45 (0.28 to 0.59) | 0.26 (0.13 to 0.39) |
| Traffic calming | -0.00 | -0.00 (-0.02 to -0.00) | – |
| Pedestrian fencing | 0.23 | 0.22 (0.04 to 0.37) | 0.22 (0.10 to 0.32) |
| Transit stops | 0.46 | 0.53 (-0.01 to 0.83) | 0.43 (0.26 to 0.57) |
| Bike facility | 0.48 | 0.42 (0.23 to 0.64) | 0.55 (0.16 to 0.83) |
| Footway (per side) | 0.02 | 0.01 (-0.05 to 0.05) | 0.02 (-0.02 to 0.07) |
| Footway obstruction (per side) | 0.11 | 0.09 (0.03 to 0.13) | 0.12 (0.07 to 0.17) |
| Trip hazards (per side) | 0.05 | 0.04 (-0.03 to 0.10) | 0.05 (-0.01 to 0.10) |
| Reason to cross | 0.34 | 0.42 (0.25 to 0.57) | 0.25 (0.08 to 0.39) |
| Crossing facility | 0.53 | 0.61 (0.49 to 0.73) | 0.47 (0.39 to 0.56) |
| Crossing quality | 0.65 | 0.73 (0.60 to 0.85) | 0.59 (0.47 to 0.70) |
| Kerb ramps | -0.04 | -0.04 (-0.12 to 0.02) | -0.04 (-0.09 to 0.00) |

### Listed needs against the item answers, by region

| Model | Region | Needs per clip | Item needs listed | Listed needs backed by an item | Answers fully consistent |
|---|---|---|---|---|---|
| Qwen3.8-27B | Europe & North America | 1.4 | 86% | 80% | 57% |
| Qwen3.8-27B | Rest of the world | 1.6 | 83% | 75% | 33% |
| Gemma 4 31B | Europe & North America | 1.4 | 94% | 100% | 90% |
| Gemma 4 31B | Rest of the world | 1.8 | 95% | 100% | 90% |
| Molmo2-8B | Europe & North America | 0.6 | 67% | 24% | 43% |
| Molmo2-8B | Rest of the world | 0.6 | 88% | 28% | 52% |
| InternVL3.5-8B | Europe & North America | 1.2 | 40% | 25% | 15% |
| InternVL3.5-8B | Rest of the world | 1.5 | 53% | 37% | 21% |

### Robustness: Cohen's kappa with the main answers (city clips, mean of the four models)

| Item | Reworded prompt | One frame | Half resolution | Resampled (T = 0.7) |
|---|---|---|---|---|
| Area type | 0.84 | 0.87 | 0.91 | 0.86 |
| Land use | 0.89 | 0.74 | 0.85 | 0.54 |
| Lanes | 0.83 | 0.76 | 0.87 | 0.63 |
| Median | 0.83 | 0.65 | 0.79 | 0.46 |
| Street lighting | 0.91 | 0.62 | 0.86 | 0.53 |
| Traffic calming | 0.33 | 0.00 | 0.00 | 0.00 |
| Pedestrian fencing | 0.64 | 0.61 | 0.65 | 0.55 |
| Transit stops | 0.92 | 0.52 | 0.78 | 0.72 |
| Bike facility | 0.83 | 0.38 | 0.68 | 0.56 |
| Footway | 0.47 | 0.46 | 0.73 | 0.33 |
| Footway obstruction | 0.58 | 0.51 | 0.73 | 0.43 |
| Trip hazards | 0.54 | 0.52 | 0.71 | 0.42 |
| Reason to cross | 0.82 | 0.63 | 0.77 | 0.66 |
| Crossing facility | 0.77 | 0.68 | 0.76 | 0.66 |
| Crossing quality | 0.78 | 0.69 | 0.81 | 0.72 |
| Kerb ramps | 0.55 | 0.43 | 0.65 | 0.41 |

### Robustness by region (mean Cohen's kappa over items and models, city clips)

| Change | Europe & North America | Rest of the world |
|---|---|---|
| Reworded prompt | 0.75 | 0.73 |
| One frame | 0.61 | 0.57 |
| Half resolution | 0.75 | 0.77 |
| Resampled (T = 0.7) | 0.57 | 0.54 |

"Resampled" compares two runs at temperature 0.7 with each other, not with the main run.


### Suggestions: share rated yes by each judge (city clips)


#### Judge: Qwen3.8-27B

| Criterion | Qwen3.8-27B (n = 307) | Gemma 4 31B (n = 282) | Molmo2-8B (n = 289) | InternVL3.5-8B (n = 375) |
|---|---|---|---|---|
| Deficit visible | 70% | 71% | 22% | 13% |
| Fixes the deficit | 97% | 98% | 96% | 67% |
| Specific to the street | 41% | 18% | 3% | 0% |
| Feasible | 13% | 10% | 1% | 1% |
| About behaviour | 0% | 0% | 0% | 0% |

#### Judge: Gemma 4 31B

| Criterion | Qwen3.8-27B (n = 307) | Gemma 4 31B (n = 282) | Molmo2-8B (n = 289) | InternVL3.5-8B (n = 375) |
|---|---|---|---|---|
| Deficit visible | 92% | 91% | 34% | 47% |
| Fixes the deficit | 96% | 99% | 95% | 72% |
| Specific to the street | 61% | 45% | 9% | 6% |
| Feasible | 76% | 76% | 51% | 38% |
| About behaviour | 0% | 0% | 0% | 0% |

#### Judged quality by region (both judges, all auditors)

| Criterion | Europe & North America (n = 896) | Rest of the world (n = 1610) |
|---|---|---|
| Deficit visible | 51% | 54% |
| Fixes the deficit | 85% | 90% |
| Specific to the street | 19% | 24% |
| Feasible | 32% | 32% |
| About behaviour | 0% | 0% |

#### Agreement between the judges (1253 suggestions rated by both)

| Criterion | Same rating | Cohen's kappa |
|---|---|---|
| Deficit visible | 71% | 0.44 |
| Fixes the deficit | 95% | 0.77 |
| Specific to the street | 80% | 0.42 |
| Feasible | 47% | 0.07 |
| About behaviour | 100% | – |

#### Self-preference: a judge's yes-share minus the other judge's, in points

| Judge | Criterion | On its own model | On the two neutral models |
|---|---|---|---|
| Qwen3.8-27B | Deficit visible | -22 | -23 |
| Qwen3.8-27B | Fixes the deficit | +1 | -2 |
| Qwen3.8-27B | Specific to the street | -21 | -6 |
| Qwen3.8-27B | Feasible | -62 | -43 |
| Qwen3.8-27B | About behaviour | +0 | +0 |
| Gemma 4 31B | Deficit visible | +20 | +23 |
| Gemma 4 31B | Fixes the deficit | +1 | +2 |
| Gemma 4 31B | Specific to the street | +27 | +6 |
| Gemma 4 31B | Feasible | +67 | +43 |
| Gemma 4 31B | About behaviour | +0 | +0 |
