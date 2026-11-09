"""Have VLM judges rate the needed infrastructure and suggestions of the street audit.

For every answer of the "audit" run on the city clips, a judge model sees the same frames
and the auditor's "needed" entries, here called deficits (what is missing, where, evidence,
measure, suggestion), and rates each:
is the deficit really there, does the suggestion fix it, is it specific to this street,
is it feasible, and does it change the street rather than tell people how to behave.
One request per answer covers all its deficits. Two judges from different developers
(Qwen and Gemma) rate everything, so the analysis can compare them and check whether a
judge favours its own model's suggestions.

Ratings go to outputs/vlm/judge/<judge>/<auditor model>/<clip_id>.json; finished ones are skipped.

Run on wsl5090:
    uv run python scripts/audit/judge_suggestions.py --start-server --judges qwen3.8-27b gemma4-31b
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

from vlm_street_audit.config import PROJECT_ROOT, load_config, project_path
from vlm_street_audit.server import start_server, stop_server, wait_until_ready
from vlm_street_audit.vlm import encode_image, load_models

logger = logging.getLogger("judge_suggestions")
SOURCE = "audit"
JUDGE_VERSION = "2026-10-28"

RATING = {
    "type": "object",
    "additionalProperties": False,
    "required": ["deficit_visible", "addresses_deficit", "specific", "feasible", "about_behaviour", "reason"],
    "properties": {
        "deficit_visible": {"enum": ["yes", "no", "cannot_tell"]},
        "addresses_deficit": {"type": "boolean"},
        "specific": {"enum": ["specific", "generic"]},
        "feasible": {"enum": ["yes", "no", "unclear"]},
        "about_behaviour": {"type": "boolean"},
        "reason": {"type": "string", "maxLength": 200},
    },
}

PROMPT = """You are an experienced street-design reviewer checking another auditor's work. The {n} images are \
the frames the auditor saw, from a dashcam clip of one street. The auditor coded the street with the iRAP road \
attributes and MAPS-Mini and reported the missing infrastructure (deficits) below, each with a suggested change.

{deficits}

Rate each deficit, in the same order:
- deficit_visible: do the frames show this deficit at that place? "cannot_tell" if they do not show enough.
- addresses_deficit: would the suggested change fix this deficit?
- specific: "specific" if the suggestion refers to this street (its layout, a feature you can see, where the \
change goes); "generic" if it would read the same for any street.
- feasible: could a city realistically make this change on this street, given the space you see? "unclear" if \
the frames do not show enough.
- about_behaviour: true if the suggestion tells people how to behave (walk carefully, use another route) instead \
of changing the street.
- reason: one short sentence on your ratings.

Judge only from the frames. Answer with JSON only, in British English."""


def deficit_text(issues: list[dict]) -> str:
    return "\n".join(
        f"{i}. {d['need']} ({d['where']}). Evidence: {d['evidence']} Measure: {d['measure']}. "
        f"Suggestion: {d['suggestion']}"
        for i, d in enumerate(issues, start=1)
    )


def judge_one(judge: str, record: dict, base_url: str, max_side: int, timeout_s: float, max_tokens: int) -> dict:
    deficits = record["answer"]["needed"]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ratings"],
        "properties": {
            "ratings": {"type": "array", "items": RATING, "minItems": len(deficits), "maxItems": len(deficits)}
        },
    }
    content = [
        {"type": "image_url", "image_url": {"url": encode_image(PROJECT_ROOT / f, max_side)}} for f in record["frames"]
    ]
    content.append({"type": "text", "text": PROMPT.format(n=len(record["frames"]), deficits=deficit_text(deficits))})
    body = {
        "model": judge,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": "judge", "schema": schema, "strict": True}},
    }
    result = {
        "clip_id": record["clip_id"],
        "auditor": record["model"],
        "judge": judge,
        "judge_version": JUDGE_VERSION,
        "deficits": deficits,
    }
    try:
        response = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=timeout_s)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"] or ""
        result.update({"ratings": json.loads(text)["ratings"], "error": None})
    except (requests.RequestException, json.JSONDecodeError, KeyError) as error:
        result.update({"ratings": None, "error": str(error)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judges", nargs="+", default=["qwen3.8-27b", "gemma4-31b"])
    parser.add_argument("--start-server", action="store_true", help="start and stop vLLM for each judge")
    parser.add_argument("--limit", type=int, help="first N answers per auditor only (smoke test)")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_config()
    models = load_models()
    server = models.pop("server")
    base_url = f"http://127.0.0.1:{server['port']}"
    root = project_path(settings["vlm_output_dir"])
    records = []
    for auditor_dir in sorted((root / SOURCE / "cities").glob("*")):
        answered = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(auditor_dir.glob("*.json"))]
        answered = [r for r in answered if r.get("answer") and r["answer"]["needed"]]
        records += answered[: args.limit] if args.limit else answered
    failures = []
    for judge in args.judges:
        out = root / "judge" / judge
        todo = []
        for record in records:
            path = out / record["model"] / f"{record['clip_id']}.json"
            if path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("ratings"):
                continue
            todo.append(record)
        logger.info("%s: %d answers to judge", judge, len(todo))
        if not todo:
            continue
        process = start_server(judge, PROJECT_ROOT / "logs" / f"vllm-{judge}.log") if args.start_server else None
        try:
            if process is not None:
                wait_until_ready(base_url, process)

            def rate(record: dict) -> dict:
                # Up to 8 short ratings; Gemma's 4k context must hold the frames and the deficits too.
                max_tokens = min(1200, int(models[judge].get("max_answer_tokens", 2048)))
                result = judge_one(
                    judge, record, base_url, int(settings["vlm_image_max_side"]), args.timeout, max_tokens
                )
                path = out / record["model"] / f"{record['clip_id']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
                return result

            with ThreadPoolExecutor(max_workers=int(server.get("parallel_requests", 1))) as pool:
                for done, result in enumerate(pool.map(rate, todo), start=1):
                    logger.info(
                        "%s judged %s %s [%d/%d]%s",
                        judge,
                        result["auditor"],
                        result["clip_id"],
                        done,
                        len(todo),
                        f": {result['error']}" if result["error"] else "",
                    )
        except (RuntimeError, TimeoutError) as error:
            failures.append(judge)
            logger.error("%s skipped: %s", judge, error)
        finally:
            if process is not None:
                stop_server(process)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
