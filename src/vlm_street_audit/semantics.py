"""SigLIP2 keyframe scores for ranking snippet windows (from opticarvis-global's candidate_semantics.py).

Keyframes (at least `min_frame_gap_s` apart) are decoded inside the mapped segments
only, embedded once and cached per video. A window's score for a prompt group is
the mean of its `top_k` highest keyframe margins, where a keyframe's margin is its
best similarity to the group's prompts minus its best similarity to the routine-scene
(negative) prompts. Subtracting that baseline on the same frame removes most of the
image-text similarity bias that has nothing to do with the prompt. The window's
semantic score is its best group's score.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .snippets import Window

CACHE_VERSION = 1


def load_prompts(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    groups, negatives = config.get("positive_groups"), config.get("negative_prompts")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("positive_groups must be a non-empty object")
    for group, prompts in groups.items():
        if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError(f"Prompt group {group} must hold non-empty prompts")
    if not isinstance(negatives, list) or not negatives or not all(isinstance(p, str) and p.strip() for p in negatives):
        raise ValueError("negative_prompts must hold non-empty prompts")
    return config


def prompts_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def unit_rows(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def top_k_mean(values: np.ndarray, k: int) -> float:
    count = min(max(1, k), values.size)
    return float(np.mean(np.partition(values, values.size - count)[-count:]))


def score_windows(windows: list[Window], times: np.ndarray, images: np.ndarray, texts: dict[str, np.ndarray],
                  top_k: int = 2) -> None:
    """Set `semantic_score` and `group_scores` of each window from keyframe embeddings."""
    if len(times) == 0:
        return
    images = unit_rows(images)
    baseline = np.max(images @ unit_rows(texts["negative"]).T, axis=1)
    margins = {group: np.max(images @ unit_rows(emb).T, axis=1) - baseline
               for group, emb in texts.items() if group != "negative"}
    for window in windows:
        inside = np.flatnonzero((times >= window.start_s) & (times < window.end_s))
        if not len(inside):
            middle = (window.start_s + window.end_s) / 2
            inside = np.asarray([int(np.argmin(np.abs(times - middle)))])
        window.group_scores = {group: round(top_k_mean(m[inside], top_k), 6) for group, m in margins.items()}
        window.semantic_score = max(window.group_scores.values())


class KeyframeEncoder:
    """SigLIP2 image/text encoder, loaded on first use and shared across videos."""

    def __init__(self, model_id: str, batch_size: int = 32, min_frame_gap_s: float = 2.0) -> None:
        self.model_id = model_id
        self.batch_size = max(1, batch_size)
        self.min_frame_gap_s = max(0.0, min_frame_gap_s)
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(
            self.model_id, dtype=torch.float16 if self._device == "cuda" else torch.float32).to(self._device).eval()

    @staticmethod
    def _pooled(output):
        for name in ("pooler_output", "image_embeds", "text_embeds"):
            value = getattr(output, name, None)
            if value is not None:
                return value
        return output

    def encode_texts(self, prompts: list[str]) -> np.ndarray:
        self._load()
        inputs = self._processor(text=list(prompts), padding="max_length", return_tensors="pt").to(self._device)
        with self._torch.inference_mode():
            features = self._pooled(self._model.get_text_features(**inputs))
        return unit_rows(features.float().cpu().numpy())

    def encode_images(self, images: list) -> np.ndarray:
        self._load()
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with self._torch.inference_mode():
            features = self._pooled(self._model.get_image_features(**inputs))
        return unit_rows(features.float().cpu().numpy())

    def encode_keyframes(self, video: Path, intervals: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        """Times and embeddings of keyframes inside the intervals, `min_frame_gap_s` apart."""
        import av

        times: list[float] = []
        batches: list[np.ndarray] = []
        pending: list = []
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"
            for start, end in intervals:
                container.seek(int(start / float(stream.time_base)), backward=True, any_frame=False, stream=stream)
                last = None
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    time_s = float(frame.pts * stream.time_base)
                    if time_s < start:
                        continue
                    if time_s >= end:
                        break
                    if last is not None and time_s < last + self.min_frame_gap_s - 1e-9:
                        continue
                    last = time_s
                    times.append(time_s)
                    pending.append(frame.to_image())
                    if len(pending) >= self.batch_size:
                        batches.append(self.encode_images(pending))
                        pending = []
        if pending:
            batches.append(self.encode_images(pending))
        if not batches:
            return np.zeros(0), np.zeros((0, 0), dtype=np.float32)
        return np.asarray(times), np.concatenate(batches)


class SemanticScorer:
    """Prompt embeddings plus per-video keyframe embeddings cached as .npz."""

    def __init__(self, prompts_path: Path, cache_dir: Path, model_id: str, batch_size: int = 32,
                 min_frame_gap_s: float = 2.0, top_k: int = 2) -> None:
        self.prompts = load_prompts(prompts_path)
        self.prompts_hash = prompts_hash(self.prompts)
        self.cache_dir = Path(cache_dir)
        self.top_k = top_k
        self.encoder = KeyframeEncoder(model_id, batch_size, min_frame_gap_s)
        self._texts: dict[str, np.ndarray] | None = None

    def texts(self) -> dict[str, np.ndarray]:
        if self._texts is None:
            self._texts = {group: self.encoder.encode_texts(prompts)
                           for group, prompts in self.prompts["positive_groups"].items()}
            self._texts["negative"] = self.encoder.encode_texts(self.prompts["negative_prompts"])
        return self._texts

    def _cache_key(self, video: Path, intervals: list[tuple[float, float]]) -> str:
        stat = video.stat()
        payload = json.dumps([CACHE_VERSION, self.encoder.model_id, self.encoder.min_frame_gap_s, stat.st_size,
                              stat.st_mtime_ns, intervals])
        return hashlib.sha256(payload.encode()).hexdigest()

    def keyframes(self, video_id: str, video: Path, intervals: list[tuple[float, float]]):
        path = self.cache_dir / f"{video_id}.npz"
        key = self._cache_key(video, intervals)
        if path.is_file():
            with np.load(path, allow_pickle=False) as cached:
                if str(cached["key"]) == key:
                    return cached["times"], cached["embeddings"]
        times, embeddings = self.encoder.encode_keyframes(video, intervals)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part.npz")
        np.savez_compressed(partial, key=np.asarray(key), times=times, embeddings=embeddings)
        partial.replace(path)
        return times, embeddings

    def score(self, video_id: str, video: Path, intervals: list[tuple[float, float]], windows: list[Window]) -> None:
        times, embeddings = self.keyframes(video_id, video, intervals)
        score_windows(windows, times, embeddings, self.texts(), self.top_k)


def top_group(window: Window) -> str:
    if not window.group_scores:
        return ""
    return max(window.group_scores, key=lambda g: window.group_scores[g])


def finite_or_blank(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.6f}"
