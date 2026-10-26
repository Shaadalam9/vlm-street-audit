import json

from vlm_street_audit.jaad import labels
from vlm_street_audit.vlm import ISSUE_CATEGORIES, MEASURES, PROMPT, SCHEMA, build_messages

TRAFFIC = """<traffic_scene><road_type>street</road_type>
<frame id="0" ped_crossing="0" ped_sign="0" stop_sign="0" traffic_light="n/a" />
<frame id="30" ped_crossing="1" ped_sign="0" stop_sign="0" traffic_light="red" />
<frame id="60" ped_crossing="0" ped_sign="0" stop_sign="0" traffic_light="n/a" />
</traffic_scene>"""

ANNOTATIONS = """<annotations><meta><task><size>90</size>
<original_size><width>1000</width><height>500</height></original_size></task></meta>
<track label="pedestrian">
 <box frame="0" outside="0" xtl="100" ytl="100" xbr="200" ybr="300"><attribute name="cross">not-crossing</attribute></box>
 <box frame="60" outside="0" xtl="500" ytl="100" xbr="600" ybr="300"><attribute name="cross">crossing</attribute></box>
</track>
<track label="people"><box frame="0" outside="0" xtl="0" ytl="0" xbr="10" ybr="10"/></track>
</annotations>"""


def write_jaad(tmp_path):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations_traffic").mkdir()
    (tmp_path / "annotations" / "video_0001.xml").write_text(ANNOTATIONS)
    (tmp_path / "annotations_traffic" / "video_0001_traffic.xml").write_text(TRAFFIC)
    return tmp_path


def test_jaad_labels_only_use_the_frames_sent(tmp_path):
    jaad = write_jaad(tmp_path)
    early = labels(jaad, "video_0001", [0])
    assert early["pedestrians_visible"] and not early["zebra_crossing_visible"]
    assert not early["pedestrians_in_carriageway"] and not early["traffic_signal_visible"]
    full = labels(jaad, "video_0001", [0, 30, 60])
    assert full["zebra_crossing_visible"] and full["traffic_signal_visible"] and full["pedestrians_in_carriageway"]
    # Boxes are scaled to 0-1000 like the VLM points; "people" groups are not pedestrians.
    assert full["boxes"]["60"] == [{"x1": 500, "y1": 200, "x2": 600, "y2": 600, "crossing": True}]
    assert len(full["boxes"]["0"]) == 1


def test_schema_lists_match_the_constants():
    issue = SCHEMA["properties"]["issues"]["items"]["properties"]
    assert issue["category"]["enum"] == ISSUE_CATEGORIES and issue["measure"]["enum"] == MEASURES
    assert set(SCHEMA["properties"]["scene"]["required"]) == set(SCHEMA["properties"]["scene"]["properties"])
    json.dumps(SCHEMA)


def test_messages_hold_the_frames_then_the_prompt(tmp_path):
    from PIL import Image

    frames = []
    for i in range(3):
        path = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (2000, 1000)).save(path)
        frames.append(path)
    content = build_messages(frames, 10, 1280)[0]["content"]
    assert [c["type"] for c in content] == ["image_url"] * 3 + ["text"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "3 images" in content[-1]["text"] and "10-second" in content[-1]["text"]
    assert "{" not in PROMPT.format(n=1, seconds=1, step=1)


def test_checklist_variant_covers_every_problem_type():
    from vlm_street_audit.vlm import CATEGORY_DEFINITIONS, VARIANTS, prompt_version

    prompt, schema = VARIANTS["checklist"]
    assert set(schema["properties"]["checks"]["required"]) == set(CATEGORY_DEFINITIONS)
    assert set(CATEGORY_DEFINITIONS) == set(ISSUE_CATEGORIES) - {"other"}
    assert all(f"- {category}:" in prompt for category in CATEGORY_DEFINITIONS)
    assert "{definitions}" not in prompt and "{" not in prompt.format(n=1, seconds=1, step=1)
    assert prompt_version("open") != prompt_version("checklist")
    # The scene facts stay comparable with the open variant, except the pavement options.
    assert schema["properties"]["scene"]["required"] == SCHEMA["properties"]["scene"]["required"]


def test_issues_of_turns_problem_verdicts_into_issues():
    from vlm_street_audit.vlm import issues_of

    check = {"evidence": "no footway on the right", "verdict": "problem", "frame": 2, "x": 800, "y": 700,
             "severity": "high", "measure": "build_or_widen_pavement", "suggestion": "Build a footway.",
             "confidence": 0.8}
    answer = {"checks": {"pavement_condition": check, "accessibility": {**check, "verdict": "cannot_tell"}}}
    issues = issues_of(answer)
    assert [i["category"] for i in issues] == ["pavement_condition"]
    assert issues[0]["description"] == "no footway on the right"
    assert issues_of({"issues": [{"category": "other"}]}) == [{"category": "other"}]
