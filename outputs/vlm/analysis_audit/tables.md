120 JAAD clips and 150 city clips (75 cities) answered by all six models.

### JAAD: balanced accuracy against the labels

| Item | Qwen3.8-27B | Gemma 4 31B | Molmo2-8B | InternVL3.5-8B | Ministral 3 14B | Kanana-1.5-V 3B |
|---|---|---|---|---|---|---|
| Zebra crossing | 0.92 | 0.90 | 0.92 | 0.91 | 0.86 | 0.67 |
| Traffic signal | 0.98 | 0.73 | 0.82 | 0.95 | 0.82 | 0.90 |
| Pedestrians visible | 0.90 | 0.99 | 0.98 | 0.99 | 0.98 | 0.87 |
| Pedestrians in road | 0.71 | 0.64 | 0.67 | 0.62 | 0.57 | 0.76 |
| Crossing facility (iRAP item) | 0.92 | 0.88 | 0.92 | 0.89 | 0.86 | 0.57 |

The crossing-facility item counts a marked, signalised or raised crossing and is scored against JAAD's crossing label.


### Infrastructure present, by continent (share of cities; two thirds of the models agree)

|  | Asia (32) | Europe (15) | North America (15) | Africa (7) | South America (5) | Oceania (1) | All (75) |
|---|---|---|---|---|---|---|---|
| Formal footway on both sides | 52% (36%–67%) | 60% (43%–77%) | 50% (30%–70%) | 14% (0%–36%) | 80% (60%–100%) | 50% | 51% (41%–61%) |
| Footway set back ≥ 1 m or behind a barrier | 53% (38%–69%) | 60% (40%–80%) | 47% (27%–67%) | 29% (7%–43%) | 60% (30%–90%) | 50% | 51% (42%–60%) |
| Footways free of obstruction | 60% (40%–76%) | 50% (23%–77%) | 82% (59%–100%) | 67% (0%–100%) | 80% (60%–100%) | 100% | 65% (54%–76%) |
| Crossing facility | 19% (9%–30%) | 20% (7%–33%) | 27% (10%–47%) | 7% (0%–21%) | 50% (20%–80%) | 50% | 22% (15%–29%) |
| Signalised or grade-separated crossing | 6% (2%–12%) | 7% (0%–17%) | 7% (0%–20%) | 0% (0%–0%) | 20% (0%–40%) | 50% | 7% (3%–12%) |
| Refuge island | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 0% (0%–0%) |
| Kerb ramps at the crossing | 12% (0%–38%) | 20% (0%–60%) | 25% (0%–75%) | – | 0% (0%–0%) | – | 16% (0%–34%) |
| Street lighting | 95% (89%–100%) | 83% (67%–97%) | 70% (50%–87%) | 50% (29%–71%) | 90% (70%–100%) | 100% | 83% (77%–89%) |
| Traffic calming | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 0% (0%–0%) |
| Pedestrian fencing | 5% (0%–12%) | 3% (0%–10%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 3% (0%–7%) |
| Bike facility | 3% (0%–8%) | 0% (0%–0%) | 7% (0%–17%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 3% (1%–5%) |
| Transit stop | 20% (11%–30%) | 3% (0%–10%) | 0% (0%–0%) | 0% (0%–0%) | 10% (0%–30%) | 0% | 10% (5%–15%) |

### Infrastructure needed, by continent (share of cities)

|  | Asia (32) | Europe (15) | North America (15) | Africa (7) | South America (5) | Oceania (1) | All (75) |
|---|---|---|---|---|---|---|---|
| Footway (none or informal) | 12% (5%–22%) | 10% (0%–20%) | 23% (10%–40%) | 50% (21%–79%) | 10% (0%–30%) | 0% | 17% (11%–25%) |
| Clear a blocked footway | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 0% (0%–0%) |
| Repair trip hazards | 2% (0%–5%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 10% (0%–30%) | 0% | 1% (0%–3%) |
| Crossing where people need one | 42% (28%–56%) | 37% (17%–57%) | 23% (7%–43%) | 57% (36%–79%) | 40% (10%–70%) | 0% | 38% (29%–47%) |
| Better crossing | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 0% (0%–0%) |
| Refuge on a wide road | 5% (0%–9%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 10% (0%–30%) | 0% | 3% (1%–5%) |
| Kerb ramps | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% | 0% (0%–0%) |
| Street lighting | 3% (0%–8%) | 3% (0%–10%) | 10% (0%–27%) | 14% (0%–36%) | 0% (0%–0%) | 0% | 5% (2%–9%) |

Cities are the unit: each city's two clips are averaged, and intervals come from a bootstrap over cities. Oceania is one city.


### Infrastructure and country indicators (Spearman's rho over cities, 95% bootstrap interval)

|  | Road deaths per 100k (n = 73) | Median age (n = 75) | Gini index (n = 74) | Literacy rate (n = 51) |
|---|---|---|---|---|
| Formal footway on both sides | -0.17 (-0.39 to 0.07) | 0.43 (0.22 to 0.61) | -0.04 (-0.27 to 0.21) | 0.43 (0.19 to 0.62) |
| Footway set back ≥ 1 m or behind a barrier | -0.18 (-0.41 to 0.06) | 0.38 (0.16 to 0.58) | -0.16 (-0.37 to 0.08) | 0.34 (0.10 to 0.56) |
| Footways free of obstruction | 0.01 (-0.28 to 0.27) | -0.06 (-0.33 to 0.25) | -0.01 (-0.27 to 0.27) | -0.07 (-0.43 to 0.29) |
| Crossing facility | -0.24 (-0.45 to -0.01) | 0.25 (0.03 to 0.44) | 0.00 (-0.23 to 0.24) | 0.11 (-0.15 to 0.36) |
| Signalised or grade-separated crossing | -0.11 (-0.37 to 0.15) | 0.08 (-0.14 to 0.28) | 0.02 (-0.22 to 0.25) | 0.00 (-0.21 to 0.22) |
| Street lighting | -0.27 (-0.46 to -0.06) | 0.34 (0.13 to 0.52) | -0.25 (-0.46 to -0.02) | 0.31 (0.01 to 0.58) |
| Transit stop | -0.01 (-0.26 to 0.24) | 0.15 (-0.05 to 0.34) | -0.24 (-0.43 to -0.02) | 0.21 (-0.04 to 0.43) |
| Needed: Footway (none or informal) | 0.22 (-0.01 to 0.42) | -0.39 (-0.59 to -0.17) | 0.27 (0.06 to 0.46) | -0.53 (-0.72 to -0.27) |
| Needed: Crossing where people need one | 0.20 (-0.01 to 0.41) | -0.34 (-0.52 to -0.12) | 0.03 (-0.20 to 0.26) | -0.29 (-0.53 to -0.04) |
| Needed: Street lighting | 0.19 (-0.02 to 0.37) | -0.20 (-0.41 to 0.05) | 0.03 (-0.20 to 0.25) | -0.24 (-0.50 to 0.11) |
| Number of needs | 0.33 (0.11 to 0.52) | -0.43 (-0.61 to -0.21) | 0.20 (-0.05 to 0.43) | -0.44 (-0.66 to -0.17) |

Country figures from the city sample (CROWD mapping); a positive rho means the item is more common where the indicator is higher.


### Agreement between the six models, by region (Fleiss' kappa, city clips, 95% interval)

| Item | All clips | Europe & North America (60 clips) | Rest of the world (90 clips) |
|---|---|---|---|
| Area type | 0.46 | 0.48 (0.27 to 0.61) | 0.40 (0.08 to 0.60) |
| Land use | 0.30 | 0.31 (0.19 to 0.42) | 0.27 (0.19 to 0.36) |
| Lanes | 0.28 | 0.19 (0.11 to 0.26) | 0.28 (0.21 to 0.35) |
| Median | 0.06 | -0.02 (-0.08 to 0.03) | 0.08 (0.03 to 0.13) |
| Street lighting | 0.21 | 0.20 (0.10 to 0.29) | 0.20 (0.11 to 0.29) |
| Traffic calming | -0.00 | -0.00 (-0.01 to -0.00) | -0.00 (-0.01 to -0.00) |
| Pedestrian fencing | 0.23 | 0.21 (0.05 to 0.34) | 0.23 (0.14 to 0.30) |
| Transit stops | 0.27 | 0.26 (0.07 to 0.49) | 0.25 (0.15 to 0.33) |
| Bike facility | 0.41 | 0.32 (0.13 to 0.49) | 0.52 (0.14 to 0.82) |
| Footway (per side) | 0.00 | -0.00 (-0.04 to 0.03) | 0.00 (-0.03 to 0.03) |
| Footway obstruction (per side) | 0.05 | 0.05 (0.01 to 0.08) | 0.05 (0.01 to 0.08) |
| Trip hazards (per side) | 0.01 | 0.01 (-0.04 to 0.04) | 0.01 (-0.03 to 0.04) |
| Reason to cross | 0.29 | 0.35 (0.22 to 0.46) | 0.23 (0.13 to 0.33) |
| Crossing facility | 0.42 | 0.46 (0.35 to 0.56) | 0.40 (0.33 to 0.47) |
| Crossing quality | 0.39 | 0.45 (0.36 to 0.55) | 0.35 (0.26 to 0.42) |
| Kerb ramps | -0.00 | -0.01 (-0.06 to 0.02) | -0.00 (-0.04 to 0.03) |

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
| Ministral 3 14B | Europe & North America | 1.6 | 94% | 57% | 28% |
| Ministral 3 14B | Rest of the world | 2.0 | 92% | 61% | 29% |
| Kanana-1.5-V 3B | Europe & North America | 0.6 | 0% | – | 70% |
| Kanana-1.5-V 3B | Rest of the world | 0.6 | 0% | – | 67% |

### Robustness: Cohen's kappa with the main answers (city clips, mean of the six models)

| Item | Reworded prompt | One frame | Half resolution | Resampled (T = 0.7) |
|---|---|---|---|---|
| Area type | 0.84 | 0.82 | 0.91 | 0.80 |
| Land use | 0.85 | 0.72 | 0.84 | 0.55 |
| Lanes | 0.82 | 0.74 | 0.84 | 0.61 |
| Median | 0.69 | 0.65 | 0.79 | 0.44 |
| Street lighting | 0.83 | 0.57 | 0.79 | 0.47 |
| Traffic calming | 0.39 | 0.00 | 0.22 | 0.17 |
| Pedestrian fencing | 0.53 | 0.64 | 0.66 | 0.48 |
| Transit stops | 0.86 | 0.53 | 0.75 | 0.70 |
| Bike facility | 0.78 | 0.38 | 0.70 | 0.54 |
| Footway | 0.48 | 0.46 | 0.69 | 0.28 |
| Footway obstruction | 0.47 | 0.48 | 0.67 | 0.35 |
| Trip hazards | 0.48 | 0.47 | 0.66 | 0.32 |
| Reason to cross | 0.76 | 0.61 | 0.74 | 0.61 |
| Crossing facility | 0.72 | 0.60 | 0.73 | 0.57 |
| Crossing quality | 0.72 | 0.60 | 0.76 | 0.66 |
| Kerb ramps | 0.51 | 0.41 | 0.64 | 0.36 |

### Robustness by region (mean Cohen's kappa over items and models, city clips)

| Change | Europe & North America | Rest of the world |
|---|---|---|
| Reworded prompt | 0.70 | 0.67 |
| One frame | 0.58 | 0.54 |
| Half resolution | 0.73 | 0.73 |
| Resampled (T = 0.7) | 0.53 | 0.49 |

"Resampled" compares two runs at temperature 0.7 with each other, not with the main run.


### Suggestions: share rated yes by each judge (city clips)


#### Judge: Qwen3.8-27B

| Criterion | Qwen3.8-27B (n = 307) | Gemma 4 31B (n = 282) | Molmo2-8B (n = 289) | InternVL3.5-8B (n = 375) | Ministral 3 14B (n = 621) |
|---|---|---|---|---|---|
| Deficit visible | 70% | 71% | 22% | 13% | 13% |
| Fixes the deficit | 97% | 98% | 96% | 67% | 97% |
| Specific to the street | 41% | 18% | 3% | 0% | 2% |
| Feasible | 13% | 10% | 1% | 1% | 2% |
| About behaviour | 0% | 0% | 0% | 0% | 0% |

#### Judge: Gemma 4 31B

| Criterion | Qwen3.8-27B (n = 307) | Gemma 4 31B (n = 282) | Molmo2-8B (n = 289) | InternVL3.5-8B (n = 375) | Ministral 3 14B (n = 621) |
|---|---|---|---|---|---|
| Deficit visible | 92% | 91% | 34% | 47% | 49% |
| Fixes the deficit | 96% | 99% | 95% | 72% | 94% |
| Specific to the street | 61% | 45% | 9% | 6% | 7% |
| Feasible | 76% | 76% | 51% | 38% | 54% |
| About behaviour | 0% | 0% | 0% | 0% | 0% |

#### Judged quality by region (both judges, all auditors)

| Criterion | Europe & North America (n = 1356) | Rest of the world (n = 2392) |
|---|---|---|
| Deficit visible | 44% | 47% |
| Fixes the deficit | 88% | 92% |
| Specific to the street | 13% | 18% |
| Feasible | 30% | 31% |
| About behaviour | 0% | 0% |

#### Agreement between the judges (1874 suggestions rated by both)

| Criterion | Same rating | Cohen's kappa |
|---|---|---|
| Deficit visible | 68% | 0.40 |
| Fixes the deficit | 95% | 0.70 |
| Specific to the street | 83% | 0.40 |
| Feasible | 47% | 0.06 |
| About behaviour | 100% | 1.00 |

#### Self-preference: a judge's yes-share minus the other judge's, in points

| Judge | Criterion | On its own model | On the models that do not judge |
|---|---|---|---|
| Qwen3.8-27B | Deficit visible | -22 | -28 |
| Qwen3.8-27B | Fixes the deficit | +1 | -0 |
| Qwen3.8-27B | Specific to the street | -21 | -5 |
| Qwen3.8-27B | Feasible | -62 | -46 |
| Qwen3.8-27B | About behaviour | +0 | +0 |
| Gemma 4 31B | Deficit visible | +20 | +28 |
| Gemma 4 31B | Fixes the deficit | +1 | +0 |
| Gemma 4 31B | Specific to the street | +27 | +5 |
| Gemma 4 31B | Feasible | +67 | +46 |
| Gemma 4 31B | About behaviour | +0 | +0 |
