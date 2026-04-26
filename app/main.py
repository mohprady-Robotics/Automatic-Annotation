from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.models import (
    Annotation,
    ExportRequest,
    ExportResponse,
    PropagateRequest,
    PropagateResponse,
    SessionCreateResponse,
)
from app.sam2_service import propagate_annotations


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
EXPORT_DIR = BASE_DIR / "app" / "data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SessionCreateRequest(BaseModel):
    image_dir: str = Field(min_length=1)


@dataclass
class SessionState:
    session_id: str
    frame_paths: List[Path]
    width: int
    height: int


SESSIONS: Dict[str, SessionState] = {}


app = FastAPI(title="SAM2 Annotation App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists() or not image_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {image_dir}")
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise HTTPException(
            status_code=400,
            detail=f"No supported images found in {image_dir}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    return paths


def _read_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return width, height


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/session/create", response_model=SessionCreateResponse)
def create_session(request: SessionCreateRequest) -> SessionCreateResponse:
    image_dir = Path(request.image_dir).expanduser().resolve()
    frame_paths = _list_images(image_dir)
    width, height = _read_image_size(frame_paths[0])
    session_id = uuid4().hex
    SESSIONS[session_id] = SessionState(
        session_id=session_id,
        frame_paths=frame_paths,
        width=width,
        height=height,
    )
    frame_urls = [f"/api/sessions/{session_id}/frames/{index}" for index in range(len(frame_paths))]
    return SessionCreateResponse(
        session_id=session_id,
        frame_count=len(frame_paths),
        width=width,
        height=height,
        frame_urls=frame_urls,
    )


@app.get("/api/sessions/{session_id}/frames/{frame_index}")
def get_frame(session_id: str, frame_index: int) -> FileResponse:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    if frame_index < 0 or frame_index >= len(session.frame_paths):
        raise HTTPException(status_code=404, detail=f"Frame index out of range: {frame_index}")
    return FileResponse(session.frame_paths[frame_index])


@app.post("/api/propagate", response_model=PropagateResponse)
def propagate(request: PropagateRequest) -> PropagateResponse:
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown session: {request.session_id}")
    if request.key_frame_index < 0 or request.key_frame_index >= len(session.frame_paths):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid key frame index {request.key_frame_index}.",
        )
    if not request.annotations:
        raise HTTPException(status_code=400, detail="At least one annotation is required.")

    backend, result = propagate_annotations(
        frame_paths=session.frame_paths,
        key_frame_index=request.key_frame_index,
        key_annotations=request.annotations,
    )
    frames = {str(frame_index): annotations for frame_index, annotations in result.items()}
    return PropagateResponse(session_id=request.session_id, backend=backend, frames=frames)


@app.post("/api/export", response_model=ExportResponse)
def export_annotations(request: ExportRequest) -> ExportResponse:
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown session: {request.session_id}")

    ordered_frames = sorted(request.annotations_by_frame.items(), key=lambda entry: int(entry[0]))
    all_annotations: List[dict] = []
    for _, annotations in ordered_frames:
        all_annotations.extend(annotation.model_dump() for annotation in annotations)

    track_summary: Dict[int, dict] = {}
    for annotation in all_annotations:
        track_id = int(annotation["track_id"])
        item = track_summary.setdefault(
            track_id,
            {
                "track_id": track_id,
                "label": annotation["label"],
                "frames": [],
                "shape_types": set(),
            },
        )
        item["frames"].append(int(annotation["frame_index"]))
        item["shape_types"].add(annotation["shape_type"])

    normalized_tracks = []
    for track in sorted(track_summary.values(), key=lambda entry: entry["track_id"]):
        unique_frames = sorted(set(track["frames"]))
        normalized_tracks.append(
            {
                "track_id": track["track_id"],
                "label": track["label"],
                "frames": unique_frames,
                "first_frame": unique_frames[0] if unique_frames else None,
                "last_frame": unique_frames[-1] if unique_frames else None,
                "shape_types": sorted(track["shape_types"]),
            }
        )

    payload = {
        "session_id": request.session_id,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "images": [
            {
                "frame_index": index,
                "file_name": path.name,
                "file_path": str(path),
            }
            for index, path in enumerate(session.frame_paths)
        ],
        "annotations": all_annotations,
        "annotations_by_frame": {
            frame: [annotation.model_dump() for annotation in annotations]
            for frame, annotations in ordered_frames
        },
        "tracks": normalized_tracks,
    }
    filename = f"{request.session_id}_annotations.json"
    file_path = EXPORT_DIR / filename
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ExportResponse(
        session_id=request.session_id,
        filename=filename,
        download_url=f"/api/exports/{filename}",
    )


@app.get("/api/exports/{filename}")
def download_export(filename: str) -> FileResponse:
    file_path = EXPORT_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Export not found: {filename}")
    return FileResponse(file_path)
