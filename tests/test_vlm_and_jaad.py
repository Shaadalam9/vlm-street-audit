import json

from vlm_street_audit.jaad import labels
from vlm_street_audit.vlm import AUDIT_SCHEMA, MEASURES_AUDIT, NEEDED, build_messages

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
    needed = AUDIT_SCHEMA["properties"]["needed"]["items"]["properties"]
    assert needed["need"]["enum"] == list(NEEDED) and needed["measure"]["enum"] == MEASURES_AUDIT
    for part in ("scene", "road", "left_side", "right_side", "crossing"):
        block = AUDIT_SCHEMA["properties"][part]
        assert set(block["required"]) == set(block["properties"])
    json.dumps(AUDIT_SCHEMA)


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


def audit_answer(**changes):
    side = {"footway": "sidewalk_1_to_3m", "obstruction": "none", "trip_hazards": "none"}
    answer = {"scene": {"road_type": "urban_street", "zebra_crossing_visible": True, "traffic_signal_visible": False,
                        "pedestrians_visible": True, "pedestrians_in_carriageway": False},
              "road": {"area_type": "urban", "land_use": "commercial", "number_of_lanes": "2", "median": "none",
                       "street_lighting": "some", "traffic_calming": "not_present", "pedestrian_fencing": "not_present",
                       "transit_stops": "1", "bike_facility": "none"},
              "left_side": dict(side), "right_side": dict(side),
              "crossing": {"crossing_demand": True, "crossing_facility": "marked", "crossing_quality": "adequate",
                           "kerb_ramps": "both_kerbs"},
              "needed": []}
    for path, value in changes.items():
        part, item = path.split("__")
        answer[part][item] = value
    return answer


def test_audit_prompts_fill_in_and_share_one_schema():
    from vlm_street_audit.vlm import CROSSING_ITEMS, NEEDED, ROAD_ITEMS, SIDE_ITEMS, VARIANTS, prompt_version

    for variant in ("audit", "audit_reworded"):
        prompt, schema = VARIANTS[variant]
        for n in (1, 5):
            text = prompt.format(n=n, seconds=10, step=2, intro="INTRO")
            assert "{" not in text and "INTRO" in text
        assert all(f"- {item}:" in prompt for item in {**ROAD_ITEMS, **SIDE_ITEMS, **CROSSING_ITEMS})
        assert all(need in prompt for need in NEEDED)
        json.dumps(schema)
    assert VARIANTS["audit"][1] is VARIANTS["audit_reworded"][1]
    assert prompt_version("audit") != prompt_version("audit_reworded")


def test_one_frame_is_introduced_as_a_still(tmp_path):
    from PIL import Image

    path = tmp_path / "f.jpg"
    Image.new("RGB", (64, 32)).save(path)
    text = build_messages([path], 10, 640, "audit")[0]["content"][-1]["text"]
    assert "a still from a dashcam video" in text and "1 images" not in text


def test_expected_needs_follow_the_item_answers():
    from vlm_street_audit.vlm import expected_needs, issues_of

    assert expected_needs(audit_answer()) == set()
    answer = audit_answer(left_side__footway="informal_path_under_1m", right_side__obstruction="blocked",
                          road__number_of_lanes="4_or_more", crossing__kerb_ramps="none",
                          road__street_lighting="none")
    assert expected_needs(answer) == {("no_footway", "left_side"), ("blocked_footway", "right_side"),
                                      ("no_refuge_on_wide_road", "crossing"), ("no_kerb_ramps", "crossing"),
                                      ("no_street_lighting", "road")}
    # No crossing is a need only where people have a reason to cross; dark rural roads are not flagged.
    assert expected_needs(audit_answer(crossing__crossing_facility="none", crossing__crossing_demand=False,
                                       road__area_type="rural", road__street_lighting="none")) == set()
    answer["needed"] = [{"need": "no_footway", "where": "left_side", "evidence": "trodden verge", "frame": 2,
                         "x": 100, "y": 800, "measure": "build_footway", "suggestion": "Build a footway."}]
    assert [(i["category"], i["where"]) for i in issues_of(answer)] == [("no_footway", "left_side")]
