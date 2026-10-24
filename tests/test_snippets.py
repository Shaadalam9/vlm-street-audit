import shutil
import subprocess

import numpy as np
import pytest

from vlm_street_audit.semantics import load_prompts, score_windows
from vlm_street_audit.config import PROJECT_ROOT
from vlm_street_audit.snippets import (
    Window,
    extract_snippet,
    make_windows,
    normalise_intervals,
    probe_packets,
    robust_scores,
    select_windows,
)


def test_intervals_are_sorted_merged_and_cleaned():
    assert normalise_intervals([(30, 40), (0, 10), (5, 12), (50, 50), (60, float("nan"))]) == [(0, 12), (30, 40)]


def test_windows_stay_inside_segments():
    packets = [(t / 10, 1000) for t in range(0, 400)]  # 0-40 s, 10 packets a second
    windows = make_windows("v", [(0, 25), (30, 40)], packets, length_s=10, stride_s=5)
    assert [(w.start_s, w.end_s) for w in windows] == [(0, 10), (5, 15), (10, 20), (15, 25), (30, 40)]


def test_busier_windows_score_higher_on_packet_energy():
    packets = [(t / 10, 5000 if 100 <= t < 200 else 1000) for t in range(0, 400)]
    windows = make_windows("v", [(0, 40)], packets, length_s=10, stride_s=10)
    assert max(windows, key=lambda w: w.packet_score).start_s == 10
    assert robust_scores([5, 5, 5]) == [0.0, 0.0, 0.0]


def _windows(scores):
    return [Window("v", 0, 1000, start, start + 10, semantic_score=score)
            for start, score in zip(range(0, 10 * len(scores), 10), scores)]


def test_selection_keeps_one_window_per_event_and_adds_random_ones():
    # Windows at 0..190 s; the best two are neighbours (100, 110), the third best is far away (0).
    scores = [0.5] + [0.0] * 9 + [0.9, 0.8] + [0.0] * 8
    chosen = select_windows(_windows(scores), ranked_per_video=2, random_per_video=2, min_gap_s=30, seed=1)
    ranked = [w.start_s for w, kind in chosen if kind == "ranked"]
    random_ = [w.start_s for w, kind in chosen if kind == "random"]
    assert ranked == [100, 0]
    assert len(random_) == 2
    starts = sorted(w.start_s for w, _ in chosen)
    assert all(b - a >= 30 for a, b in zip(starts, starts[1:]))


def test_random_choice_is_reproducible():
    pick = lambda: [w.start_s for w, k in select_windows(_windows([0.0] * 40), 1, 3, 30, seed=7) if k == "random"]
    assert pick() == pick()


def test_semantic_margin_is_measured_against_the_routine_baseline():
    # Keyframe at t=5 looks like the "crossing" prompt, at t=15 like the routine prompt.
    texts = {"crossing": np.array([[1.0, 0.0]]), "negative": np.array([[0.0, 1.0]])}
    windows = [Window("v", 0, 20, 0, 10), Window("v", 0, 20, 10, 20)]
    score_windows(windows, np.array([5.0, 15.0]), np.array([[1.0, 0.1], [0.1, 1.0]]), texts, top_k=1)
    assert windows[0].semantic_score > 0 > windows[1].semantic_score
    assert set(windows[0].group_scores) == {"crossing"}


def test_shipped_prompts_are_valid():
    prompts = load_prompts(PROJECT_ROOT / "configs" / "snippet_prompts.json")
    assert len(prompts["positive_groups"]) == 6


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_probe_and_extract_on_a_generated_video(tmp_path):
    video = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
                    "-t", "12", "-c:v", "libx264", "-g", "10", str(video)], check=True)
    packets = probe_packets(video, [(2.0, 8.0)])
    assert packets and all(2.0 <= t < 8.0 for t, _ in packets)
    frames = extract_snippet(video, 2.0, 4.0, tmp_path / "out" / "clip.mp4", tmp_path / "out" / "clip", 1)
    assert (tmp_path / "out" / "clip.mp4").stat().st_size > 0
    assert len(frames) == 4
