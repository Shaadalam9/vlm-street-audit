"""The street-audit task for the VLMs: prompt, output schema and an OpenAI-compatible client.

Every model gets the same frames, the same prompt and the same JSON schema, which
vLLM enforces while decoding (structured output), so answers are comparable field
by field. The `scene` part holds plain facts that can be checked against labelled
data (JAAD); the rest is the audit itself. Two prompt variants: "open" (a free list
of up to 5 issues) and "checklist" (a verdict for every problem type).

Coordinates are integers 0-1000 on the frame named by `frame` (1 = first frame
sent), x from the left edge, y from the top, whatever the model's native format.
"""

import base64
import hashlib
import io
import json
import time
import tomllib
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

MODELS_TOML = PROJECT_ROOT / "configs" / "models.toml"

ISSUE_CATEGORIES = [
    "pavement_condition",     # missing, broken, narrow or uneven footway
    "pavement_obstruction",   # parked cars, stalls, bins, works on the footway
    "crossing_facilities",    # no, faded or badly placed crossing; no signal or refuge
    "pedestrian_exposure",    # people walking or waiting in the carriageway
    "accessibility",          # high kerbs, steps, no dropped kerb or tactile paving
    "vulnerable_users",       # children, older people, crowds close to fast traffic
    "other",
]

MEASURES = [
    "build_or_widen_pavement",
    "repair_pavement_surface",
    "clear_pavement_obstruction",
    "enforce_or_redesign_parking",
    "add_zebra_crossing",
    "add_signalised_crossing",
    "add_refuge_island",
    "raised_crossing_or_traffic_calming",
    "dropped_kerbs_and_tactile_paving",
    "lower_speed_limit",
    "protective_bollards_or_barriers",
    "other",
]

TRI_STATE = ["present", "absent", "unclear"]

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene", "issues"],
    "properties": {
        "scene": {
            "type": "object",
            "additionalProperties": False,
            "required": ["road_type", "pavement_left", "pavement_right", "zebra_crossing_visible",
                         "traffic_signal_visible", "pedestrians_visible", "pedestrians_in_carriageway"],
            "properties": {
                "road_type": {"enum": ["urban_street", "residential_street", "arterial_road", "motorway",
                                       "rural_road", "parking_area", "other"]},
                "pavement_left": {"enum": TRI_STATE},
                "pavement_right": {"enum": TRI_STATE},
                "zebra_crossing_visible": {"type": "boolean"},
                "traffic_signal_visible": {"type": "boolean"},
                "pedestrians_visible": {"type": "boolean"},
                "pedestrians_in_carriageway": {"type": "boolean"},
            },
        },
        "issues": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "description", "frame", "x", "y", "severity", "measure",
                             "suggestion", "confidence"],
                "properties": {
                    "category": {"enum": ISSUE_CATEGORIES},
                    "description": {"type": "string", "maxLength": 300},
                    "frame": {"type": "integer", "minimum": 1, "maximum": 10},
                    "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "y": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "severity": {"enum": ["low", "medium", "high"]},
                    "measure": {"enum": MEASURES},
                    "suggestion": {"type": "string", "maxLength": 300},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}

PROMPT = """You are auditing a street for pedestrians. The {n} images are frames from a {seconds:.0f}-second \
dashcam clip, in time order, about {step:.0f} s apart (frame 1 is the first).

Ignore anything that is not part of the street: the car's own bonnet or dashboard, reflections on the \
windscreen, and any date, time, speed or logo overlaid on the picture. People on bicycles, scooters or \
motorbikes are not pedestrians.

1. Scene facts, over all frames:
   - road_type: the kind of road the car is on.
   - pavement_left / pavement_right: is there a footway (pavement) beside the road on that side? \
"unclear" if you cannot see that side well enough.
   - zebra_crossing_visible: a marked pedestrian crossing (zebra or striped crossing) is visible in any frame.
   - traffic_signal_visible: a traffic light is visible in any frame.
   - pedestrians_visible: at least one person on foot is visible.
   - pedestrians_in_carriageway: at least one person on foot is walking, standing or crossing in the \
roadway used by vehicles.

2. Issues: up to 5 problems for people on foot that you can actually see, most serious first. \
For each, give the category, a short description of what you see, the frame where it is clearest, \
and the point (x, y) on that frame where it is, as integers from 0 to 1000 (x from the left edge, y from \
the top). Add the severity, the standard measure that would fix it best, a one-sentence suggestion \
specific to this place, and your confidence from 0 to 1. A suggestion must change the street (its design, \
layout, crossings, signals, parking or enforcement), never tell people how to behave, and must match the \
measure you chose. Report only what is visible; if no person on foot is visible, report no issue that \
depends on one. If the street works well for pedestrians, return an empty issue list.

Answer with JSON only, in British English."""

PROMPT_VERSION = hashlib.sha256((PROMPT + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:12]

# ---- "checklist" variant -------------------------------------------------------------
# The open prompt above let every model report almost only pedestrian exposure (first run,
# 2026-10-25): problems only where a person was visible, pavements nearly always "unclear".
# This variant asks for a verdict on every problem type, evidence first, and judges the
# street even when it is empty. Its scene facts are the same, except that the pavement
# answer "unclear" becomes "not_visible", allowed only when that side is out of view.

CATEGORY_DEFINITIONS = {
    "pavement_condition": "Is there a usable footway along the road? A problem if a footway is missing on a "
                          "side where people would walk (bare ground, verge, ditch or the carriageway's edge "
                          "instead), or is broken, uneven or too narrow for two people to pass.",
    "pavement_obstruction": "Is the footway kept clear? A problem if parked vehicles, stalls, goods, bins, "
                            "poles, signs or works block it or force people off it.",
    "crossing_facilities": "Can people cross safely where they would need to? A problem if a wide or busy road, "
                           "junction or bus stop has no marked crossing, the markings are faded, the crossing is "
                           "far too long without a refuge island, or a signalled junction has no pedestrian signal.",
    "pedestrian_exposure": "Are people on foot exposed to traffic? Needs at least one visible person on foot: a "
                           "problem if they walk, wait or cross in the carriageway, or have no separation from "
                           "fast traffic.",
    "accessibility": "Could someone in a wheelchair or with a pushchair get along and across? A problem if kerbs "
                     "are high with no dropped kerb, there are steps, gaps or barriers, or no tactile paving at a "
                     "crossing.",
    "vulnerable_users": "Are children, older people or crowds close to traffic? Needs visible people: a problem "
                        "if a school, market, stop or crowd sits next to traffic without protection.",
}

_CHECK = {
    "type": "object",
    "additionalProperties": False,
    "required": ["evidence", "verdict", "frame", "x", "y", "severity", "measure", "suggestion", "confidence"],
    "properties": {
        "evidence": {"type": "string", "maxLength": 250},
        "verdict": {"enum": ["problem", "no_problem", "cannot_tell"]},
        "frame": {"type": "integer", "minimum": 0, "maximum": 10},
        "x": {"type": "integer", "minimum": 0, "maximum": 1000},
        "y": {"type": "integer", "minimum": 0, "maximum": 1000},
        "severity": {"enum": ["none", "low", "medium", "high"]},
        "measure": {"enum": ["none"] + MEASURES},
        "suggestion": {"type": "string", "maxLength": 300},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

CHECKLIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene", "checks"],
    "properties": {
        "scene": {
            **SCHEMA["properties"]["scene"],
            "properties": {
                **SCHEMA["properties"]["scene"]["properties"],
                "pavement_left": {"enum": ["present", "absent", "not_visible"]},
                "pavement_right": {"enum": ["present", "absent", "not_visible"]},
            },
        },
        "checks": {
            "type": "object",
            "additionalProperties": False,
            "required": list(CATEGORY_DEFINITIONS),
            "properties": {category: _CHECK for category in CATEGORY_DEFINITIONS},
        },
    },
}

CHECKLIST_PROMPT = """You are auditing a street for pedestrians. The {n} images are frames from a {seconds:.0f}-second \
dashcam clip, in time order, about {step:.0f} s apart (frame 1 is the first).

Ignore anything that is not part of the street: the car's own bonnet or dashboard, reflections on the \
windscreen, and any date, time, speed, map or logo overlaid on the picture. People on bicycles, scooters or \
motorbikes are not pedestrians.

1. Scene facts, over all frames:
   - road_type: the kind of road the car is on.
   - pavement_left / pavement_right: "present" if a raised or clearly separated footway runs along that side; \
"absent" if you can see that side of the road but there is no such footway (bare ground, verge, ditch, \
parked cars or the carriageway's edge); "not_visible" only if that side is out of view.
   - zebra_crossing_visible: a marked pedestrian crossing (zebra or striped crossing) is visible in any frame.
   - traffic_signal_visible: a traffic light is visible in any frame.
   - pedestrians_visible: at least one person on foot is visible.
   - pedestrians_in_carriageway: at least one person on foot is walking, standing or crossing in the \
roadway used by vehicles.

2. Checks: judge the street itself for each of the six problem types below, one by one. A street can fail \
people on foot even when nobody is walking there: judge the infrastructure you can see. For each type, first \
write the evidence (what you see that decides it), then the verdict: "problem", "no_problem", or \
"cannot_tell" if the frames do not show enough. For a problem, give the frame where it is clearest and the \
point (x, y) on that frame as integers from 0 to 1000 (x from the left edge, y from the top), the severity, \
the standard measure that would fix it best, a one-sentence suggestion specific to this place that changes \
the street (never what people should do) and matches the measure, and your confidence from 0 to 1. For \
"no_problem" or "cannot_tell", use frame 0, x 0, y 0, severity "none", measure "none" and an empty \
suggestion.

{definitions}

Answer with JSON only, in British English."""

VARIANTS: dict[str, tuple[str, dict]] = {
    "open": (PROMPT, SCHEMA),
    "checklist": (CHECKLIST_PROMPT.replace("{definitions}", "\n".join(
        f"   - {category}: {text}" for category, text in CATEGORY_DEFINITIONS.items())), CHECKLIST_SCHEMA),
}


def prompt_version(variant: str) -> str:
    prompt, schema = VARIANTS[variant]
    return hashlib.sha256((prompt + json.dumps(schema, sort_keys=True)).encode()).hexdigest()[:12]


def issues_of(answer: dict) -> list[dict]:
    """The reported problems of either variant, as issue dicts (checklist: verdict "problem")."""
    if "checks" in answer:
        return [{"category": category, "description": check["evidence"], **{k: check[k] for k in (
            "frame", "x", "y", "severity", "measure", "suggestion", "confidence")}}
            for category, check in answer["checks"].items() if check["verdict"] == "problem"]
    return answer.get("issues", [])


def load_models() -> dict[str, dict]:
    config = tomllib.loads(MODELS_TOML.read_text(encoding="utf-8"))
    return {"server": config["server"], **{m["key"]: m for m in config["model"]}}


def encode_image(path: Path, max_side: int) -> str:
    """JPEG data URL, downscaled so the longer side is at most `max_side` pixels."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def build_messages(frames: list[Path], seconds: float, max_side: int, variant: str = "open") -> list[dict]:
    step = seconds / max(1, len(frames))
    content: list[dict] = [{"type": "image_url", "image_url": {"url": encode_image(f, max_side)}} for f in frames]
    prompt = VARIANTS[variant][0]
    content.append({"type": "text", "text": prompt.format(n=len(frames), seconds=seconds, step=step)})
    return [{"role": "user", "content": content}]


def ask(model_key: str, messages: list[dict], base_url: str, timeout_s: float = 600,
        max_tokens: int = 2048, session=None, variant: str = "open") -> dict:
    """One request; returns the raw text, the parsed answer (or None) and timing."""
    import requests

    body: dict[str, Any] = {
        "model": model_key,
        "messages": messages,
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": "street_audit", "schema": VARIANTS[variant][1],
                                                                     "strict": True}},
        # Qwen and other "thinking" models: answer directly, so the output is the JSON alone.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    response = (session or requests).post(f"{base_url}/v1/chat/completions", json=body, timeout=timeout_s)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"]["content"] or ""
    try:
        parsed = json.loads(text)
        error = None
    except json.JSONDecodeError as exc:
        parsed, error = None, f"invalid JSON: {exc}"
    return {"text": text, "answer": parsed, "error": error, "seconds": round(elapsed, 2),
            "usage": data.get("usage", {}), "finish_reason": data["choices"][0].get("finish_reason")}
