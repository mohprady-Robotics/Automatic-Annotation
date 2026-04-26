from __future__ import annotations

import os
from typing import List

import numpy as np
from PIL import Image

_SAM_BUNDLE: dict | None = None
_SAM_MODEL_ID: str | None = None


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _load_sam_bundle():
    global _SAM_BUNDLE, _SAM_MODEL_ID
    model_id = os.getenv("SAM_MODEL_ID", "facebook/sam-vit-base").strip()
    if _SAM_BUNDLE is not None and _SAM_MODEL_ID == model_id:
        return _SAM_BUNDLE

    try:
        import torch
        from transformers import SamModel, SamProcessor
    except Exception:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = SamProcessor.from_pretrained(model_id)
        model = SamModel.from_pretrained(model_id).to(device)
        model.eval()
    except Exception:
        return None

    bundle = {"torch": torch, "processor": processor, "model": model, "device": device}
    _SAM_BUNDLE = bundle
    _SAM_MODEL_ID = model_id
    return bundle


def sam_mask_available() -> bool:
    if not _env_enabled("ENABLE_SAM_MASKS", default=True):
        return False
    return _load_sam_bundle() is not None


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except Exception:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest_label).astype(np.uint8)
    return component


def _mask_to_polygon(mask: np.ndarray, max_points: int = 120) -> List[List[float]] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 3 or len(ys) < 3:
        return None

    try:
        import cv2

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        epsilon = max(1.0, 0.005 * cv2.arcLength(largest, closed=True))
        approx = cv2.approxPolyDP(largest, epsilon, closed=True)
        polygon = [[float(point[0][0]), float(point[0][1])] for point in approx]
        if len(polygon) < 3:
            return None
        if len(polygon) > max_points:
            step = max(1, len(polygon) // max_points)
            polygon = polygon[::step][:max_points]
        return polygon
    except Exception:
        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max() + 1)
        y2 = float(ys.max() + 1)
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def segment_box_to_polygon(
    frame_path: str,
    bbox_xyxy: List[float],
) -> List[List[float]] | None:
    """
    Uses SAM to segment inside a box prompt and returns a polygon approximation.
    Returns None when SAM is unavailable or segmentation fails.
    """
    if not sam_mask_available():
        return None

    bundle = _load_sam_bundle()
    if bundle is None:
        return None

    torch = bundle["torch"]
    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]

    try:
        with Image.open(frame_path) as image:
            rgb = image.convert("RGB")
            input_boxes = [[[float(v) for v in bbox_xyxy]]]
            inputs = processor(images=rgb, input_boxes=input_boxes, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)
            elif isinstance(inputs, dict):
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)

            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            mask_candidates = masks[0]
            if hasattr(mask_candidates, "cpu"):
                mask_candidates = mask_candidates.cpu().numpy()
            else:
                mask_candidates = np.asarray(mask_candidates)

            if mask_candidates.ndim == 2:
                best_mask = mask_candidates
            else:
                best_idx = 0
                try:
                    iou_scores = outputs.iou_scores
                    if hasattr(iou_scores, "detach"):
                        iou_scores = iou_scores.detach()
                    if hasattr(iou_scores, "cpu"):
                        iou_scores = iou_scores.cpu()
                    iou_np = np.asarray(iou_scores).reshape(-1)
                    if len(iou_np) > 0:
                        best_idx = int(np.argmax(iou_np))
                except Exception:
                    best_idx = 0
                best_idx = max(0, min(best_idx, mask_candidates.shape[0] - 1))
                best_mask = mask_candidates[best_idx]

            mask = best_mask > 0
            mask = _largest_connected_component(mask.astype(np.uint8))
            polygon = _mask_to_polygon(mask)
            return polygon
    except Exception:
        return None


def _to_device_batch(inputs, device: str):
    if hasattr(inputs, "to"):
        try:
            return inputs.to(device)
        except Exception:
            pass
    if isinstance(inputs, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    return inputs


def _select_best_mask(mask_candidates, outputs) -> np.ndarray:
    if mask_candidates.ndim == 2:
        return mask_candidates
    best_idx = 0
    try:
        iou_scores = outputs.iou_scores
        if hasattr(iou_scores, "detach"):
            iou_scores = iou_scores.detach()
        if hasattr(iou_scores, "cpu"):
            iou_scores = iou_scores.cpu()
        iou_np = np.asarray(iou_scores).reshape(-1)
        if len(iou_np) > 0:
            best_idx = int(np.argmax(iou_np))
    except Exception:
        best_idx = 0
    best_idx = max(0, min(best_idx, mask_candidates.shape[0] - 1))
    return mask_candidates[best_idx]


def segment_with_clicks_to_polygon(
    frame_path: str,
    positive_points: List[List[float]],
    negative_points: List[List[float]] | None = None,
    input_box: List[float] | None = None,
) -> List[List[float]] | None:
    """
    Interactive SAM refinement using positive/negative clicks and optional box prompt.
    Returns a polygon approximation for the best predicted mask.
    """
    if not sam_mask_available():
        return None
    if not positive_points:
        return None

    bundle = _load_sam_bundle()
    if bundle is None:
        return None

    torch = bundle["torch"]
    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]

    negatives = negative_points or []
    all_points = [[float(x), float(y)] for x, y in positive_points] + [
        [float(x), float(y)] for x, y in negatives
    ]
    labels = [1] * len(positive_points) + [0] * len(negatives)
    if not all_points:
        return None

    try:
        with Image.open(frame_path) as image:
            rgb = image.convert("RGB")
            kwargs = {
                "images": rgb,
                "input_points": [all_points],
                "input_labels": [labels],
                "return_tensors": "pt",
            }
            if input_box and len(input_box) == 4:
                kwargs["input_boxes"] = [[[float(v) for v in input_box]]]

            inputs = processor(**kwargs)
            inputs = _to_device_batch(inputs, device)

            with torch.inference_mode():
                outputs = model(**inputs)

            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            mask_candidates = masks[0]
            if hasattr(mask_candidates, "cpu"):
                mask_candidates = mask_candidates.cpu().numpy()
            else:
                mask_candidates = np.asarray(mask_candidates)

            best_mask = _select_best_mask(mask_candidates, outputs)
            mask = best_mask > 0
            mask = _largest_connected_component(mask.astype(np.uint8))
            return _mask_to_polygon(mask)
    except Exception:
        return None
