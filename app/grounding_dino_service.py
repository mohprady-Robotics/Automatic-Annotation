from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Tuple

from PIL import Image
from app.sam_mask_service import segment_box_to_polygon


DetectionMap = Dict[str, List[Tuple[List[float], float]]]
OpenVocabDetections = List[Tuple[str, List[float], float, List[List[float]] | None]]

_MODEL_BUNDLE: dict | None = None
_MODEL_ID: str | None = None
_LAST_ERROR: str | None = None


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


def _default_model_id() -> str:
    return os.getenv("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny").strip()


def _set_last_error(message: str | None) -> None:
    global _LAST_ERROR
    _LAST_ERROR = message.strip() if message else None


def get_last_grounding_dino_error() -> str:
    return _LAST_ERROR or "Unknown Grounding DINO error."


def _safe_from_pretrained(loader, model_id: str):
    # Some model versions require trust_remote_code while others reject it.
    try:
        return loader.from_pretrained(model_id, trust_remote_code=True)
    except TypeError:
        return loader.from_pretrained(model_id)


def _load_model_bundle():
    global _MODEL_BUNDLE, _MODEL_ID
    model_id = _default_model_id()
    if _MODEL_BUNDLE is not None and _MODEL_ID == model_id:
        return _MODEL_BUNDLE

    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except Exception as exc:
        _set_last_error(f"Missing Grounding DINO dependencies: {exc}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = _safe_from_pretrained(AutoProcessor, model_id)
        model = _safe_from_pretrained(AutoModelForZeroShotObjectDetection, model_id)
        model = model.to(device)
        model.eval()
    except Exception as exc:
        _set_last_error(f"Failed loading model '{model_id}': {exc}")
        return None

    bundle = {
        "torch": torch,
        "processor": processor,
        "model": model,
        "device": device,
    }
    _MODEL_BUNDLE = bundle
    _MODEL_ID = model_id
    _set_last_error(None)
    return bundle


def _move_inputs_to_device(inputs, device: str):
    if hasattr(inputs, "to"):
        try:
            return inputs.to(device)
        except Exception:
            pass
    if isinstance(inputs, dict):
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved
    return inputs


def _post_process_detection(
    processor,
    outputs,
    input_ids,
    box_threshold: float,
    text_threshold: float,
    target_size: tuple[int, int],
):
    errors: List[str] = []
    attempts = [
        dict(
            outputs=outputs,
            input_ids=input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[target_size],
        ),
        dict(
            outputs=outputs,
            input_ids=input_ids,
            threshold=box_threshold,
            target_sizes=[target_size],
        ),
    ]
    for kwargs in attempts:
        try:
            return processor.post_process_grounded_object_detection(**kwargs)[0]
        except Exception as exc:
            errors.append(str(exc))

    if hasattr(processor, "post_process_object_detection"):
        try:
            return processor.post_process_object_detection(outputs, threshold=box_threshold, target_sizes=[target_size])[0]
        except Exception as exc:
            errors.append(str(exc))

    raise RuntimeError(" ; ".join(errors))


def grounding_dino_status() -> tuple[bool, str]:
    if not _env_enabled("ENABLE_GROUNDING_DINO", default=True):
        return False, "ENABLE_GROUNDING_DINO must be set to 1."
    if _load_model_bundle() is None:
        return (
            False,
            (
                f"Unable to load Grounding DINO model '{_default_model_id()}'. "
                f"Install dependencies and ensure model download is available. Details: {get_last_grounding_dino_error()}"
            ),
        )
    return True, "ok"


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
    is_ready, _reason = grounding_dino_status()
    if not is_ready:
        return None
    bundle = _load_model_bundle()

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
            inputs = processor(images=rgb_image, text=query, return_tensors="pt")
            inputs = _move_inputs_to_device(inputs, device)
            with torch.inference_mode():
                outputs = model(**inputs)
            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            result = _post_process_detection(
                processor=processor,
                outputs=outputs,
                input_ids=input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_size=rgb_image.size[::-1],  # (height, width)
            )
    except Exception as exc:
        _set_last_error(f"Label detection failed: {exc}")
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
    use_sam_masks: bool = True,
) -> OpenVocabDetections | None:
    """
    Open-vocabulary detections for an arbitrary text prompt.
    Returns [(det_label, bbox_xyxy, score, polygon_or_none), ...] sorted by score descending.
    """
    is_ready, _reason = grounding_dino_status()
    if not is_ready:
        return None
    bundle = _load_model_bundle()

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
            inputs = processor(images=rgb_image, text=prompt, return_tensors="pt")
            inputs = _move_inputs_to_device(inputs, device)
            with torch.inference_mode():
                outputs = model(**inputs)
            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            result = _post_process_detection(
                processor=processor,
                outputs=outputs,
                input_ids=input_ids,
                box_threshold=float(resolved_box_threshold),
                text_threshold=float(resolved_text_threshold),
                target_size=rgb_image.size[::-1],  # (height, width)
            )
    except Exception as exc:
        _set_last_error(f"Prompt detection failed: {exc}")
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

        det_label_norm = str(det_label).strip().lower()
        if use_sam_masks:
            polygon = segment_box_to_polygon(frame_path=frame_path, bbox_xyxy=box_values)
            if polygon and len(polygon) >= 3:
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                box_values = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
        detections.append((det_label_norm, box_values, score_value, polygon if use_sam_masks else None))

    detections.sort(key=lambda item: item[2], reverse=True)
    return detections[:top_k]
