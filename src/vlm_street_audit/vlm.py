"""The street-audit task for the VLMs: prompt, output schema and an OpenAI-compatible client.

Every model gets the same frames, the same prompt and the same JSON schema, which
vLLM enforces while decoding (structured output), so answers are comparable field
by field. The `scene` part holds plain facts that can be checked against labelled
data (JAAD); the rest is the audit itself: iRAP's pedestrian attributes plus MAPS-Mini items
("audit"), and the same task in other words ("audit_reworded"), to test how much the wording
matters. (Two earlier prompt variants, a free list of issues and a checklist of problem types,
are in the repository's history up to commit e3020d3.)

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

# ---- The audit ------------------------------------------------------------------------
# Standard instruments rather than problem types of our own (2026-10-28): the
# pedestrian attributes of the iRAP Star Rating (iRAP Coding Manual, Version 5.3, drive-on-left edition, 2022:
# area type, lanes, median, lighting, pedestrian fencing, footway provision per side, crossing
# facility and its quality), which are coded the same way in over 100 countries, plus the
# MAPS-Mini items iRAP lacks (Sallis et al. 2015, Preventing Chronic Disease 12:150098; land use,
# transit stops, bike facility, trip hazards, kerb ramps), and the footway obstruction common in
# many cities. iRAP codes 100 m segments, about what a 10 s dashcam clip covers. Some iRAP
# options are merged (raised crossings with and without a refuge or markings; median types).
# Every answer that marks infrastructure as missing comes back as a "needed" entry with a
# point, a standard measure and a suggestion, so suggestions are tied to an item.

NOT_VISIBLE = "not_visible"
ROAD_TYPES = ["urban_street", "residential_street", "arterial_road", "motorway", "rural_road", "parking_area", "other"]
ROAD_ITEMS: dict[str, tuple[str, list[str]]] = {
    "area_type": ("Urban if buildings line the road, rural otherwise.", ["urban", "rural"]),
    "land_use": (
        "Mostly homes (residential), shops and offices (commercial), both (mixed), or other.",
        ["residential", "commercial", "mixed", "other"],
    ),
    "number_of_lanes": (
        "Lanes for motor traffic across the whole road, both directions.",
        ["1", "2", "3", "4_or_more"],
    ),
    "median": ("What separates the two directions.", ["none", "painted", "physical", NOT_VISIBLE]),
    "street_lighting": ("Street lights along the road.", ["none", "some", "ample"]),
    "traffic_calming": ("Speed humps, raised tables, chicanes or narrowings.", ["present", "not_present"]),
    "pedestrian_fencing": (
        "Fences or barriers along the road that stop people crossing anywhere.",
        ["present", "not_present"],
    ),
    "transit_stops": ("Public transport stops (bus, tram) visible.", ["0", "1", "2_or_more"]),
    "bike_facility": ("A designated bike lane or path.", ["none", "painted_line", "physical_barrier"]),
}
FOOTWAY = [
    "physical_barrier",
    "sidewalk_3m_or_more",
    "sidewalk_1_to_3m",
    "sidewalk_under_1m",
    "informal_path_1m_or_more",
    "informal_path_under_1m",
    "none",
    NOT_VISIBLE,
]
SIDE_ITEMS: dict[str, tuple[str, list[str]]] = {
    "footway": (
        "The footway on this side. A sidewalk is purpose-built, continuous and usable; its distance is from "
        "the edge of the nearest driving lane (parking and shoulder included); physical_barrier means "
        "bollards, a fence, wall or hedge that would stop a car (a kerb is not one). An informal path is a "
        "trodden strip without a sidewalk, or a sidewalk so blocked that it is only partly usable. none: "
        "people must walk in the road.",
        FOOTWAY,
    ),
    "obstruction": (
        "Parked vehicles, stalls, goods or works on the footway.",
        ["none", "partly_blocked", "blocked", "no_footway", NOT_VISIBLE],
    ),
    "trip_hazards": (
        "Broken or uneven sections that are major trip hazards.",
        ["none", "some", "no_footway", NOT_VISIBLE],
    ),
}
CROSSING_FACILITIES = [
    "grade_separated",
    "signalised_with_refuge",
    "signalised",
    "marked_with_refuge",
    "marked",
    "raised",
    "refuge_only",
    "none",
]
CROSSING_ITEMS: dict[str, tuple[str, list[str]]] = {
    "crossing_facility": (
        "The purpose-built crossing in view (the riskiest if several). signalised needs signals "
        "for pedestrians; a refuge is a proper island midway; grade_separated is a footbridge or "
        "underpass; raised is a speed-reducing platform.",
        CROSSING_FACILITIES,
    ),
    "crossing_quality": (
        "poor if drivers would not stop, the crossing is hard to see or it is obstructed.",
        ["adequate", "poor", "not_applicable"],
    ),
    "kerb_ramps": ("Dropped kerbs at the crossing.", ["both_kerbs", "one_kerb", "none", "no_crossing", NOT_VISIBLE]),
}
INFORMAL_OR_NONE = {"informal_path_1m_or_more", "informal_path_under_1m", "none"}
NEEDED: dict[str, str] = {
    "no_footway": "footway is an informal path or none",
    "blocked_footway": "obstruction is blocked",
    "trip_hazards": "trip_hazards is some",
    "no_crossing": "crossing_demand is true and crossing_facility is none",
    "poor_crossing": "crossing_quality is poor",
    "no_refuge_on_wide_road": "4_or_more lanes and a signalised, marked or raised crossing without a refuge",
    "no_kerb_ramps": "kerb_ramps is none",
    "no_street_lighting": "an urban road with street_lighting none",
}
MEASURES_AUDIT = [
    "build_footway",
    "widen_or_separate_footway",
    "clear_footway",
    "repair_footway",
    "add_marked_crossing",
    "add_signalised_crossing",
    "add_refuge_island",
    "add_raised_crossing",
    "improve_crossing",
    "add_dropped_kerbs",
    "add_street_lighting",
    "traffic_calming",
    "other",
]


def _enum(options: list[str]) -> dict:
    return {"enum": options}


def _block(items: dict[str, tuple[str, list[str]]], extra: dict | None = None) -> dict:
    properties = {**(extra or {}), **{item: _enum(options) for item, (_, options) in items.items()}}
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene", "road", "left_side", "right_side", "crossing", "needed"],
    "properties": {
        "scene": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "road_type",
                "zebra_crossing_visible",
                "traffic_signal_visible",
                "pedestrians_visible",
                "pedestrians_in_carriageway",
            ],
            "properties": {
                "road_type": _enum(ROAD_TYPES),
                **{
                    fact: {"type": "boolean"}
                    for fact in (
                        "zebra_crossing_visible",
                        "traffic_signal_visible",
                        "pedestrians_visible",
                        "pedestrians_in_carriageway",
                    )
                },
            },
        },
        "road": _block(ROAD_ITEMS),
        "left_side": _block(SIDE_ITEMS),
        "right_side": _block(SIDE_ITEMS),
        "crossing": _block(CROSSING_ITEMS, {"crossing_demand": {"type": "boolean"}}),
        "needed": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["need", "where", "evidence", "frame", "x", "y", "measure", "suggestion"],
                "properties": {
                    "need": _enum(list(NEEDED)),
                    "where": _enum(["left_side", "right_side", "crossing", "road"]),
                    "evidence": {"type": "string", "maxLength": 160},
                    "frame": {"type": "integer", "minimum": 1, "maximum": 10},
                    "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "y": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "measure": _enum(MEASURES_AUDIT),
                    "suggestion": {"type": "string", "maxLength": 220},
                },
            },
        },
    },
}


def _item_lines(items: dict[str, tuple[str, list[str]]]) -> str:
    return "\n".join(f"   - {item}: {question} ({' / '.join(options)})" for item, (question, options) in items.items())


_SCENE = """   - road_type: the kind of road the car is on.
   - zebra_crossing_visible: a marked pedestrian crossing (zebra or striped crossing) is visible in any frame.
   - traffic_signal_visible: a traffic light is visible in any frame.
   - pedestrians_visible: at least one person on foot is visible.
   - pedestrians_in_carriageway: at least one person on foot is walking, standing or crossing in the roadway \
used by vehicles."""
_NEEDED = "; ".join(f"{need} ({rule})" for need, rule in NEEDED.items())

AUDIT_PROMPT = f"""You are a trained road-safety auditor coding the street in {{intro}} Use the iRAP road \
attribute definitions and the MAPS-Mini street audit, as in a standard inspection of a 100 m road segment.

Code only what you can see. Ignore the car's own bonnet or dashboard, reflections on the windscreen, and any date, \
time, speed, map or logo overlaid on the picture. People on bicycles, scooters or motorbikes are not pedestrians.

1. scene, over all frames:
{_SCENE}

2. road, for the stretch of road the car drives along:
{_item_lines(ROAD_ITEMS)}

3. left_side and right_side, each side of that road separately. Use "no_footway" for obstruction and trip_hazards \
when the footway is "none", and "not_visible" only when that side is out of view.
{_item_lines(SIDE_ITEMS)}

4. crossing: crossing_demand is true if people have a reason to cross here (a junction, a stop, shops or homes on \
both sides, or people seen crossing). Then code the crossing facility in view.
{_item_lines(CROSSING_ITEMS)}

5. needed: one entry for each of these answers you gave, and for nothing else: {_NEEDED}. For each, give where it \
is, the evidence you see, the frame where it is clearest and the point (x, y) on that frame as integers from 0 to \
1000 (x from the left edge, y from the top), the standard measure that would fix it, and a one-sentence suggestion \
specific to this place that changes the street (never what people should do) and matches the measure. If no \
answer calls for one, return an empty list.

Answer with JSON only, in British English."""

# Same task, items and schema, written differently: the robustness check for wording.
AUDIT_PROMPT_REWORDED = f"""Below you will find {{intro}} Please inspect this stretch of road (about 100 m) the \
way a trained road-safety auditor would, following the iRAP coding of road attributes and the MAPS-Mini \
streetscape audit, and record how well it serves people on foot.

Every answer must rest on what the pictures show. The vehicle's own bonnet or dashboard, windscreen reflections and \
on-screen overlays (date, time, speed, maps, logos) are not part of the street. Cyclists, scooter riders and \
motorcyclists do not count as pedestrians.

Part A, scene (true if seen in any frame):
{_SCENE}

Part B, road, describing the road the vehicle is on:
{_item_lines(ROAD_ITEMS)}

Part C, left_side and right_side, judged separately for the two sides of that road. Where a side has no footway at \
all ("none"), obstruction and trip_hazards are "no_footway"; keep "not_visible" for a side you cannot see.
{_item_lines(SIDE_ITEMS)}

Part D, crossing. Set crossing_demand to true when people have cause to cross here (junction, stop, shops or \
homes on both sides, or people crossing), then describe the crossing facility you can see.
{_item_lines(CROSSING_ITEMS)}

Part E, needed. Add exactly one entry per answer that matches one of these rules, and none otherwise: {_NEEDED}. \
Each entry names where it is, what you see that shows it, the clearest frame, the point (x, y) on that frame (whole \
numbers 0-1000, x measured from the left, y from the top), the standard measure that would remedy it, and one \
sentence proposing a change to this particular street that fits that measure (a change to the street, not advice \
to people). With no matching answers, the list is empty.

Reply in British English, with JSON only."""


def frames_intro(n: int, seconds: float, step: float) -> str:
    """How the frames are introduced; one frame is a still, not a clip."""
    if n == 1:
        return "this image, a still from a dashcam video."
    return (
        f"these {n} images, frames from a {seconds:.0f}-second dashcam clip in time order, about {step:.0f} s "
        f"apart (frame 1 is the first)."
    )


VARIANTS: dict[str, tuple[str, dict]] = {
    "audit": (AUDIT_PROMPT, AUDIT_SCHEMA),
    "audit_reworded": (AUDIT_PROMPT_REWORDED, AUDIT_SCHEMA),
}


def expected_needs(answer: dict) -> set[tuple[str, str]]:
    """The (need, where) pairs the item answers call for."""
    expected = set()
    for side in ("left_side", "right_side"):
        values = answer[side]
        if values["footway"] in INFORMAL_OR_NONE:
            expected.add(("no_footway", side))
        if values["obstruction"] == "blocked":
            expected.add(("blocked_footway", side))
        if values["trip_hazards"] == "some":
            expected.add(("trip_hazards", side))
    crossing, road = answer["crossing"], answer["road"]
    facility = crossing["crossing_facility"]
    if crossing["crossing_demand"] and facility == "none":
        expected.add(("no_crossing", "crossing"))
    if crossing["crossing_quality"] == "poor":
        expected.add(("poor_crossing", "crossing"))
    if road["number_of_lanes"] == "4_or_more" and facility in {"signalised", "marked", "raised"}:
        expected.add(("no_refuge_on_wide_road", "crossing"))
    if crossing["kerb_ramps"] == "none":
        expected.add(("no_kerb_ramps", "crossing"))
    if road["area_type"] == "urban" and road["street_lighting"] == "none":
        expected.add(("no_street_lighting", "road"))
    return expected


def prompt_version(variant: str) -> str:
    prompt, schema = VARIANTS[variant]
    return hashlib.sha256((prompt + json.dumps(schema, sort_keys=True)).encode()).hexdigest()[:12]


def issues_of(answer: dict) -> list[dict]:
    """The answer's "needed" entries as flat issue dicts (category = the need)."""
    return [
        {
            "category": d["need"],
            "description": d["evidence"],
            **{k: d[k] for k in ("where", "frame", "x", "y", "measure", "suggestion")},
        }
        for d in answer.get("needed", [])
    ]


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


def build_messages(frames: list[Path], seconds: float, max_side: int, variant: str = "audit") -> list[dict]:
    step = seconds / max(1, len(frames))
    content: list[dict] = [{"type": "image_url", "image_url": {"url": encode_image(f, max_side)}} for f in frames]
    prompt = VARIANTS[variant][0]
    content.append(
        {
            "type": "text",
            "text": prompt.format(
                n=len(frames), seconds=seconds, step=step, intro=frames_intro(len(frames), seconds, step)
            ),
        }
    )
    return [{"role": "user", "content": content}]


def ask(
    model_key: str,
    messages: list[dict],
    base_url: str,
    timeout_s: float = 600,
    max_tokens: int = 2048,
    session=None,
    variant: str = "audit",
    temperature: float = 0,
    seed: int = 0,
    schema: dict | None = None,
    thinking_switch: bool = True,
) -> dict:
    """One request; returns the raw text, the parsed answer (or None) and timing.

    `thinking_switch=False` leaves out chat_template_kwargs, for models served with Mistral's own tokenizer,
    which rejects them (and have no thinking mode to switch off)."""
    import requests

    body: dict[str, Any] = {
        "model": model_key,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "street_audit", "schema": schema or VARIANTS[variant][1], "strict": True},
        },
    }
    if thinking_switch:  # Qwen, EXAONE and other "thinking" models: answer directly, the output is the JSON alone
        body["chat_template_kwargs"] = {"enable_thinking": False}
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
    return {
        "text": text,
        "answer": parsed,
        "error": error,
        "seconds": round(elapsed, 2),
        "usage": data.get("usage", {}),
        "finish_reason": data["choices"][0].get("finish_reason"),
    }
