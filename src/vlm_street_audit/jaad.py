"""JAAD as a labelled check for the scene facts the VLMs report.

JAAD (York University) labels every frame of its 346 clips: whether a pedestrian
crossing (zebra) is visible, the traffic light state, each pedestrian's box and,
for pedestrians with behaviour labels, whether they are crossing. Per clip it gives
the road type. The labels used here, taken over the same frames the VLMs see:

    zebra_crossing_visible      any frame with ped_crossing = 1
    traffic_signal_visible      any frame whose traffic_light is not n/a
    pedestrians_visible         any pedestrian box that is not outside the frame
    pedestrians_in_carriageway  any pedestrian whose `cross` is "crossing"
                                (a lower bound: people walking along the road are not labelled)
    parking_area                road_type is parking_lot or garage
"""

import xml.etree.ElementTree as ET
from pathlib import Path

FPS = 30  # JAAD clips are 30 fps; annotation frame n is at n / 30 s


def frame_count(jaad_dir: Path, video: str) -> int:
    root = ET.parse(jaad_dir / "annotations" / f"{video}.xml").getroot()
    return int(root.findtext(".//meta/task/size"))


def labels(jaad_dir: Path, video: str, frames: list[int]) -> dict:
    """Clip-level labels over the given frame numbers, plus pedestrian boxes per frame."""
    wanted = set(frames)
    traffic = ET.parse(jaad_dir / "annotations_traffic" / f"{video}_traffic.xml").getroot()
    road_type = traffic.findtext("road_type")
    zebra = signal = False
    for frame in traffic.findall("frame"):
        if int(frame.get("id")) in wanted:
            zebra |= frame.get("ped_crossing") == "1"
            signal |= frame.get("traffic_light") not in (None, "n/a")
    boxes: dict[int, list[dict]] = {f: [] for f in frames}
    crossing = False
    root = ET.parse(jaad_dir / "annotations" / f"{video}.xml").getroot()
    width = int(root.findtext(".//meta/task/original_size/width") or 1920)
    height = int(root.findtext(".//meta/task/original_size/height") or 1080)
    for track in root.findall("track"):
        if track.get("label") not in ("pedestrian", "ped"):
            continue
        for box in track.findall("box"):
            number = int(box.get("frame"))
            if number not in wanted or box.get("outside") == "1":
                continue
            attributes = {a.get("name"): a.text for a in box.findall("attribute")}
            is_crossing = attributes.get("cross") == "crossing"
            crossing |= is_crossing
            boxes[number].append({
                # Corners scaled to 0-1000, like the VLM points.
                "x1": round(1000 * float(box.get("xtl")) / width), "y1": round(1000 * float(box.get("ytl")) / height),
                "x2": round(1000 * float(box.get("xbr")) / width), "y2": round(1000 * float(box.get("ybr")) / height),
                "crossing": is_crossing,
            })
    return {
        "road_type": road_type,
        "zebra_crossing_visible": zebra,
        "traffic_signal_visible": signal,
        "pedestrians_visible": any(boxes.values()),
        "pedestrians_in_carriageway": crossing,
        "parking_area": road_type in ("parking_lot", "garage"),
        "boxes": {str(f): b for f, b in boxes.items()},
    }
