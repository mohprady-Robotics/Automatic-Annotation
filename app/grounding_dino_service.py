from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

from PIL import Image


DetectionMap = Dict[str, List[Tuple[List[float], float]]]
OpenVocabDetections = List[Tuple[str, List[float], float]]


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _normalize_label(label: str) -> str:
    collapsed = re.sub(r"\s+", " ", label.strip().lower())
    return collapsed


def _as_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


@lru_cache(maxsize=1)
def _load_model_bundle():
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except Exception:
        return None

    model_id = os.getenv("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny").strip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        model = model.to(device)
        model.eval()
    except Exception:
        return None

    return {
        "torch": torch,
        "processor": processor,
        "model": model,
        "device": device,
    }


def _match_label(det_label: str, expected_labels: List[str]) -> str | None:
    normalized_det = _normalize_label(det_label)
    if not normalized_det:
        return None

    for expected in expected_labels:
        if expected in normalized_det or normalized_det in expected:
            return expected
    return None


def detect_boxes_for_labels(frame_path: str, labels: Iterable[str]) -> DetectionMap | None:
    """
    Returns detections keyed by normalized label.
    Each value is [(bbox_xyxy, score), ...].
    """
    if not _env_enabled("ENABLE_GROUNDING_DINO", default=True):
        return None

    bundle = _load_model_bundle()
    if bundle is None:
        return None

    normalized_labels = sorted({_normalize_label(label) for label in labels if label and label.strip()})
    if not normalized_labels:
        return None

    query = " . ".join(normalized_labels) + " ."
    box_threshold = _as_float("GROUNDING_DINO_BOX_THRESHOLD", 0.30)
    text_threshold = _as_float("GROUNDING_DINO_TEXT_THRESHOLD", 0.25)

    torch = bundle["torch"]
    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]

    try:
        with Image.open(frame_path) as image:
            rgb_image = image.convert("RGB")
            inputs = processor(images=rgb_image, text=query, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[rgb_image.size[::-1]],  # (height, width)
            )[0]
    except Exception:
        return None

    detections: DetectionMap = {label: [] for label in normalized_labels}
    scores = result.get("scores", [])
    labels_out = result.get("labels", [])
    boxes_out = result.get("boxes", [])

    for score, det_label, box in zip(scores, labels_out, boxes_out):
        if hasattr(score, "item"):
            score_value = float(score.item())
        else:
            score_value = float(score)

        if hasattr(box, "tolist"):
            box_values = [float(value) for value in box.tolist()]
        else:
            box_values = [float(value) for value in box]

        expected = _match_label(str(det_label), normalized_labels)
        if expected is None:
            continue
        detections.setdefault(expected, []).append((box_values, score_value))

    return detections


def detect_boxes_for_text_prompt(
    frame_path: str,
    text_prompt: str,
    box_threshold: float | None = None,
    text_threshold: float | None = None,
    top_k: int = 20,
) -> OpenVocabDetections | None:
    """
    Open-vocabulary detections for an arbitrary text prompt.
    Returns [(det_label, bbox_xyxy, score), ...] sorted by score descending.
    """
    if not _env_enabled("ENABLE_GROUNDING_DINO", default=True):
        return None

    bundle = _load_model_bundle()
    if bundle is None:
        return None

    prompt = text_prompt.strip().lower()
    if not prompt:
        return None
    if not prompt.endswith("."):
        prompt = f"{prompt} ."

    resolved_box_threshold = box_threshold if box_threshold is not None else _as_float("GROUNDING_DINO_BOX_THRESHOLD", 0.30)
    resolved_text_threshold = text_threshold if text_threshold is not None else _as_float("GROUNDING_DINO_TEXT_THRESHOLD", 0.25)

    torch = bundle["torch"]
    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]

    try:
        with Image.open(frame_path) as image:
            rgb_image = image.convert("RGB")
            inputs = processor(images=rgb_image, text=prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=float(resolved_box_threshold),
                text_threshold=float(resolved_text_threshold),
                target_sizes=[rgb_image.size[::-1]],  # (height, width)
            )[0]
    except Exception:
        return None

    detections: OpenVocabDetections = []
    scores = result.get("scores", [])
    labels_out = result.get("labels", [])
    boxes_out = result.get("boxes", [])

    for score, det_label, box in zip(scores, labels_out, boxes_out):
        if hasattr(score, "item"):
            score_value = float(score.item())
        else:
            score_value = float(score)

        if hasattr(box, "tolist"):
            box_values = [float(value) for value in box.tolist()]
        else:
            box_values = [float(value) for value in box]

        detections.append((str(det_label).strip().lower(), box_values, score_value))

    detections.sort(key=lambda item: item[2], reverse=True)
    return detections[:top_k]
