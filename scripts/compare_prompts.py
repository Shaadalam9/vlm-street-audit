"""Compare two prompt variants on the clips both have answered.

Per model: problem types reported, pavement answers, problems on clips where the
model saw nobody on foot, and (for JAAD clips) the scene facts against the labels.

Run:  uv run python scripts/compare_prompts.py open checklist
"""

import argparse
import json
import sys
from collections import Counter

import pandas as pd

from vlm_street_audit.config import load_config, project_path
from vlm_street_audit.evaluation import SCENE_BOOLEANS, load_answers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args()
    settings = load_config()
    root = project_path(settings["vlm_output_dir"])
    truth = json.loads((project_path(settings["clip_sets_dir"]) / "jaad_labels.json").read_text())

    frames = []
    for variant in (args.first, args.second):
        for set_name in ("cities", "jaad"):
            answers = load_answers(root / variant, set_name)
            if not answers.empty:
                frames.append(answers.assign(variant=variant, set=set_name))
    data = pd.concat(frames, ignore_index=True)
    data = data[data.valid]  # failed requests (e.g. timed out while the server was paused) have no answer
    shared = data.groupby(["model", "clip_id"]).variant.nunique()
    data = data.set_index(["model", "clip_id"]).loc[shared[shared == 2].index].reset_index()
    print(f"{data.clip_id.nunique()} clips answered with both prompts\n")

    rows = []
    for (model, variant), group in data.groupby(["model", "variant"]):
        cats = Counter(issue["category"] for issues in group.issues for issue in issues)
        pavement = Counter(list(group.scene_pavement_left) + list(group.scene_pavement_right))
        empty = group[~group.scene_pedestrians_visible.astype(bool)]
        row = {"model": model, "prompt": variant, "clips": len(group),
               "issues": sum(cats.values()), "types_used": len(cats),
               "problem_types": ", ".join(f"{k} {v}" for k, v in cats.most_common()),
               "pavement_answers": ", ".join(f"{k} {v}" for k, v in pavement.most_common()),
               "empty_clips_with_problem": f"{int(empty.issues.map(bool).sum())}/{len(empty)}"}
        jaad = group[group.set == "jaad"]
        for field in SCENE_BOOLEANS:
            if len(jaad):
                row[f"jaad_{field}"] = float(
                    (jaad[f"scene_{field}"].astype(bool) == [truth[c][field] for c in jaad.clip_id]).mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        for model, group in table.groupby("model"):
            print(f"== {model}")
            print(group.drop(columns="model").T.to_string(header=False), "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
