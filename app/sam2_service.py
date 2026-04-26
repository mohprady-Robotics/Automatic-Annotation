from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from app.grounding_dino_service import detect_boxes_for_labels
from app.models import Annotation


@dataclass
class _TrackerState:
    track_id: int
    label: str
    bbox: List[float]
    shape_type: str
    polygon: List[List[float]] | None


def _annotation_bbox(annotation: Annotation) -> List[float]:
    if annotation.shape_type == "bbox":
        assert annotation.bbox is not None
        return [float(value) for value in annotation.bbox]
    assert annotation.polygon is not None
    xs = [point[0] for point in annotation.polygon]
    ys = [point[1] for point in annotation.polygon]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _clamp_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(x1 + 1.0, min(x2, width))
    y2 = max(y1 + 1.0, min(y2, height))
    return [x1, y1, x2, y2]


def _load_gray(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _compute_shift(prev_gray: np.ndarray, curr_gray: np.ndarray, max_shift: int = 24) -> Tuple[int, int]:
    # Lightweight translation-only search used as a fallback when SAM2 is unavailable.
    best_score = None
    best_shift = (0, 0)
    prev = prev_gray.astype(np.float32)
    curr = curr_gray.astype(np.float32)
    for dy in range(-max_shift, max_shift + 1, 2):
        for dx in range(-max_shift, max_shift + 1, 2):
            y1_prev = max(0, dy)
            y2_prev = min(prev.shape[0], prev.shape[0] + dy)
            x1_prev = max(0, dx)
            x2_prev = min(prev.shape[1], prev.shape[1] + dx)

            y1_curr = max(0, -dy)
            y2_curr = min(curr.shape[0], curr.shape[0] - dy)
            x1_curr = max(0, -dx)
            x2_curr = min(curr.shape[1], curr.shape[1] - dx)
            if y2_prev - y1_prev < 16 or x2_prev - x1_prev < 16:
                continue
            prev_crop = prev[y1_prev:y2_prev, x1_prev:x2_prev]
            curr_crop = curr[y1_curr:y2_curr, x1_curr:x2_curr]
            diff = np.mean(np.abs(prev_crop - curr_crop))
            if best_score is None or diff < best_score:
                best_score = diff
                best_shift = (dx, dy)
    return best_shift


def _iou_xyxy(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _refine_with_grounding_dino(
    frame_path: str,
    states: List[_TrackerState],
    width: int,
    height: int,
) -> bool:
    labels = [state.label for state in states]
    detections = detect_boxes_for_labels(frame_path=frame_path, labels=labels)
    if not detections:
        return False

    used_detection_idx: Dict[str, set[int]] = {}
    any_refined = False
    min_iou = float(os.getenv("GROUNDING_DINO_MIN_IOU", "0.05"))
    blend = float(os.getenv("GROUNDING_DINO_BLEND", "0.70"))
    score_weight = float(os.getenv("GROUNDING_DINO_SCORE_WEIGHT", "0.20"))

    for state in states:
        label_key = " ".join(state.label.strip().lower().split())
        candidates = detections.get(label_key, [])
        if not candidates:
            continue

        best_idx = -1
        best_score = -1.0
        best_iou = -1.0
        best_box = None
        for idx, (candidate_box, det_score) in enumerate(candidates):
            if idx in used_detection_idx.setdefault(label_key, set()):
                continue
            candidate_clamped = _clamp_bbox(candidate_box, width, height)
            iou = _iou_xyxy(state.bbox, candidate_clamped)
            combined_score = iou + score_weight * float(det_score)
            if combined_score > best_score:
                best_score = combined_score
                best_iou = iou
                best_idx = idx
                best_box = candidate_clamped

        if best_box is None or best_iou < min_iou:
            continue

        x1 = state.bbox[0] * (1.0 - blend) + best_box[0] * blend
        y1 = state.bbox[1] * (1.0 - blend) + best_box[1] * blend
        x2 = state.bbox[2] * (1.0 - blend) + best_box[2] * blend
        y2 = state.bbox[3] * (1.0 - blend) + best_box[3] * blend
        state.bbox = _clamp_bbox([x1, y1, x2, y2], width, height)
        used_detection_idx[label_key].add(best_idx)
        any_refined = True

    return any_refined


def _propagate_fallback(frame_paths: List[str], key_frame_index: int, key_annotations: List[Annotation]) -> tuple[Dict[int, List[Annotation]], bool]:
    with Image.open(frame_paths[key_frame_index]) as key_image:
        width, height = key_image.size

    tracked_objects = [
        _TrackerState(
            track_id=annotation.track_id,
            label=annotation.label,
            bbox=_annotation_bbox(annotation),
            shape_type=annotation.shape_type,
            polygon=[list(point) for point in annotation.polygon] if annotation.polygon else None,
        )
        for annotation in key_annotations
    ]

    result: Dict[int, List[Annotation]] = {}
    grounding_used = False

    for frame_index, _ in enumerate(frame_paths):
        if frame_index == key_frame_index:
            result[frame_index] = [
                Annotation(
                    id=f"{annotation.track_id}_{frame_index}",
                    track_id=annotation.track_id,
                    label=annotation.label,
                    frame_index=frame_index,
                    shape_type=annotation.shape_type,
                    source="manual",
                    bbox=annotation.bbox,
                    polygon=annotation.polygon,
                )
                for annotation in key_annotations
            ]

    # Forward propagation
    prev_gray = _load_gray(frame_paths[key_frame_index])
    states_fwd = [
        _TrackerState(
            track_id=state.track_id,
            label=state.label,
            bbox=list(state.bbox),
            shape_type=state.shape_type,
            polygon=[list(point) for point in state.polygon] if state.polygon else None,
        )
        for state in tracked_objects
    ]
    for frame_index in range(key_frame_index + 1, len(frame_paths)):
        curr_gray = _load_gray(frame_paths[frame_index])
        dx, dy = _compute_shift(prev_gray, curr_gray)
        frame_annotations = []
        for state in states_fwd:
            shifted = [state.bbox[0] + dx, state.bbox[1] + dy, state.bbox[2] + dx, state.bbox[3] + dy]
            clamped = _clamp_bbox(shifted, width, height)
            state.bbox = clamped
        frame_refined = _refine_with_grounding_dino(
            frame_path=frame_paths[frame_index],
            states=states_fwd,
            width=width,
            height=height,
        )
        if frame_refined:
            grounding_used = True
        for state in states_fwd:
            frame_annotations.append(
                Annotation(
                    id=f"{state.track_id}_{frame_index}",
                    track_id=state.track_id,
                    label=state.label,
                    frame_index=frame_index,
                    shape_type="bbox",
                    source="sam2_fallback_grounding_dino" if frame_refined else "sam2_fallback",
                    bbox=state.bbox,
                    polygon=None,
                )
            )
        result[frame_index] = frame_annotations
        prev_gray = curr_gray

    # Backward propagation
    prev_gray = _load_gray(frame_paths[key_frame_index])
    states_bwd = [
        _TrackerState(
            track_id=state.track_id,
            label=state.label,
            bbox=list(state.bbox),
            shape_type=state.shape_type,
            polygon=[list(point) for point in state.polygon] if state.polygon else None,
        )
        for state in tracked_objects
    ]
    for frame_index in range(key_frame_index - 1, -1, -1):
        curr_gray = _load_gray(frame_paths[frame_index])
        dx, dy = _compute_shift(prev_gray, curr_gray)
        frame_annotations = []
        for state in states_bwd:
            shifted = [state.bbox[0] + dx, state.bbox[1] + dy, state.bbox[2] + dx, state.bbox[3] + dy]
            clamped = _clamp_bbox(shifted, width, height)
            state.bbox = clamped
        frame_refined = _refine_with_grounding_dino(
            frame_path=frame_paths[frame_index],
            states=states_bwd,
            width=width,
            height=height,
        )
        if frame_refined:
            grounding_used = True
        for state in states_bwd:
            frame_annotations.append(
                Annotation(
                    id=f"{state.track_id}_{frame_index}",
                    track_id=state.track_id,
                    label=state.label,
                    frame_index=frame_index,
                    shape_type="bbox",
                    source="sam2_fallback_grounding_dino" if frame_refined else "sam2_fallback",
                    bbox=state.bbox,
                    polygon=None,
                )
            )
        result[frame_index] = frame_annotations
        prev_gray = curr_gray

    return result, grounding_used


def _try_import_sam2() -> bool:
    try:
        import sam2  # noqa: F401

        return True
    except Exception:
        return False


def _mask_to_bbox(mask: np.ndarray) -> List[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return [x1, y1, x2, y2]


def _add_sam2_prompt(predictor, inference_state, key_frame_index: int, annotation: Annotation) -> None:
    bbox = _annotation_bbox(annotation)
    # Official SAM2 APIs are evolving; support the common "box prompt" signatures.
    if hasattr(predictor, "add_new_points_or_box"):
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=key_frame_index,
            obj_id=annotation.track_id,
            box=np.array(bbox, dtype=np.float32),
        )
    elif hasattr(predictor, "add_new_box"):
        predictor.add_new_box(
            inference_state=inference_state,
            frame_idx=key_frame_index,
            obj_id=annotation.track_id,
            box=np.array(bbox, dtype=np.float32),
        )
    else:
        raise RuntimeError("Installed SAM2 predictor does not expose a supported box prompt API.")


def _propagate_with_sam2(
    frame_paths: List[str],
    key_frame_index: int,
    key_annotations: List[Annotation],
) -> Dict[int, List[Annotation]] | None:
    if not _try_import_sam2():
        return None

    sam2_cfg = os.getenv("SAM2_MODEL_CFG")
    sam2_ckpt = os.getenv("SAM2_CHECKPOINT")
    if not sam2_cfg or not sam2_ckpt:
        return None
    sam2_cfg = sam2_cfg.strip()
    sam2_ckpt = sam2_ckpt.strip()
    if not Path(sam2_ckpt).exists():
        return None

    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except Exception:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with Image.open(frame_paths[key_frame_index]) as image:
        width, height = image.size

    tracks = {
        annotation.track_id: {
            "label": annotation.label,
            "seed_bbox": _annotation_bbox(annotation),
            "shape_type": annotation.shape_type,
            "seed_polygon": annotation.polygon,
        }
        for annotation in key_annotations
    }

    # SAM2 video predictor expects a frame directory. We create a deterministic staging
    # directory so this can run for arbitrary user-provided image paths.
    with tempfile.TemporaryDirectory(prefix="sam2_frames_") as tmp_dir:
        staged_dir = Path(tmp_dir)
        for frame_index, frame_path in enumerate(frame_paths):
            src = Path(frame_path)
            extension = src.suffix.lower() or ".jpg"
            dst = staged_dir / f"{frame_index:05d}{extension}"
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        try:
            predictor = build_sam2_video_predictor(sam2_cfg, sam2_ckpt, device=device)
            inference_state = predictor.init_state(video_path=str(staged_dir))
            for annotation in key_annotations:
                _add_sam2_prompt(
                    predictor=predictor,
                    inference_state=inference_state,
                    key_frame_index=key_frame_index,
                    annotation=annotation,
                )

            raw_per_frame: Dict[int, Dict[int, List[float]]] = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                frame_bboxes: Dict[int, List[float]] = {}
                for idx, obj_id in enumerate(out_obj_ids):
                    mask = out_mask_logits[idx]
                    if hasattr(mask, "detach"):
                        mask = mask.detach()
                    if hasattr(mask, "cpu"):
                        mask = mask.cpu()
                    if hasattr(mask, "numpy"):
                        mask = mask.numpy()
                    mask_np = np.asarray(mask)
                    if mask_np.ndim > 2:
                        mask_np = np.squeeze(mask_np)
                    bbox = _mask_to_bbox(mask_np > 0.0)
                    if bbox is None:
                        continue
                    frame_bboxes[int(obj_id)] = _clamp_bbox(bbox, width, height)
                raw_per_frame[int(out_frame_idx)] = frame_bboxes
        except Exception:
            return None

    # Densify tracks so every frame has an annotation for each seeded object.
    frame_count = len(frame_paths)
    dense_tracks: Dict[int, List[List[float] | None]] = {
        track_id: [None] * frame_count for track_id in tracks
    }
    for frame_idx, frame_data in raw_per_frame.items():
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        for track_id, bbox in frame_data.items():
            if track_id in dense_tracks:
                dense_tracks[track_id][frame_idx] = bbox

    for track_id, per_frame in dense_tracks.items():
        seed_bbox = tracks[track_id]["seed_bbox"]
        per_frame[key_frame_index] = _clamp_bbox(seed_bbox, width, height)

        # Forward fill.
        last = None
        for frame_idx in range(frame_count):
            if per_frame[frame_idx] is None and last is not None:
                per_frame[frame_idx] = list(last)
            elif per_frame[frame_idx] is not None:
                last = per_frame[frame_idx]

        # Backward fill.
        last = None
        for frame_idx in range(frame_count - 1, -1, -1):
            if per_frame[frame_idx] is None and last is not None:
                per_frame[frame_idx] = list(last)
            elif per_frame[frame_idx] is not None:
                last = per_frame[frame_idx]

        for frame_idx in range(frame_count):
            if per_frame[frame_idx] is None:
                per_frame[frame_idx] = _clamp_bbox(seed_bbox, width, height)

    result: Dict[int, List[Annotation]] = {}
    for frame_idx in range(frame_count):
        if frame_idx == key_frame_index:
            result[frame_idx] = [
                Annotation(
                    id=f"{annotation.track_id}_{frame_idx}",
                    track_id=annotation.track_id,
                    label=annotation.label,
                    frame_index=frame_idx,
                    shape_type=annotation.shape_type,
                    source="manual",
                    bbox=annotation.bbox,
                    polygon=annotation.polygon,
                )
                for annotation in key_annotations
            ]
            continue

        frame_annotations: List[Annotation] = []
        for track_id, track in tracks.items():
            bbox = dense_tracks[track_id][frame_idx]
            assert bbox is not None
            frame_annotations.append(
                Annotation(
                    id=f"{track_id}_{frame_idx}",
                    track_id=track_id,
                    label=track["label"],
                    frame_index=frame_idx,
                    shape_type="bbox",
                    source="sam2",
                    bbox=bbox,
                    polygon=None,
                )
            )
        result[frame_idx] = frame_annotations

    return result


def propagate_annotations(
    frame_paths: List, key_frame_index: int, key_annotations: List[Annotation]
) -> tuple[str, Dict[int, List[Annotation]]]:
    normalized_paths = [str(path) for path in frame_paths]

    sam2_result = _propagate_with_sam2(
        frame_paths=normalized_paths,
        key_frame_index=key_frame_index,
        key_annotations=key_annotations,
    )
    if sam2_result is not None:
        return "sam2", sam2_result

    result, grounding_used = _propagate_fallback(normalized_paths, key_frame_index, key_annotations)
    if grounding_used:
        return "sam2_fallback_grounding_dino", result
    return "sam2_fallback", result
