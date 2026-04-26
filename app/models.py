from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


ShapeType = Literal["bbox", "polygon"]


class Annotation(BaseModel):
    id: str
    track_id: int = Field(ge=1)
    label: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    shape_type: ShapeType
    source: Literal["manual", "sam2", "sam2_fallback", "sam2_fallback_grounding_dino"] = "manual"
    bbox: Optional[List[float]] = None
    polygon: Optional[List[List[float]]] = None

    @model_validator(mode="after")
    def validate_geometry(self) -> "Annotation":
        if self.shape_type == "bbox":
            if not self.bbox or len(self.bbox) != 4:
                raise ValueError("bbox annotations require 4 values: [x1, y1, x2, y2].")
            if self.polygon:
                raise ValueError("bbox annotations cannot include polygon geometry.")
        if self.shape_type == "polygon":
            if not self.polygon or len(self.polygon) < 3:
                raise ValueError("polygon annotations require at least 3 points.")
            if self.bbox:
                raise ValueError("polygon annotations cannot include bbox geometry.")
        return self


class SessionCreateResponse(BaseModel):
    session_id: str
    frame_count: int
    width: int
    height: int
    frame_urls: List[str]


class PropagateRequest(BaseModel):
    session_id: str
    key_frame_index: int = Field(ge=0)
    annotations: List[Annotation]


class PropagateResponse(BaseModel):
    session_id: str
    backend: Literal["sam2", "sam2_fallback", "sam2_fallback_grounding_dino"]
    frames: Dict[str, List[Annotation]]


class ExportRequest(BaseModel):
    session_id: str
    annotations_by_frame: Dict[str, List[Annotation]]


class ExportResponse(BaseModel):
    session_id: str
    filename: str
    download_url: str


class OpenVocabDetectRequest(BaseModel):
    session_id: str
    frame_index: int = Field(ge=0)
    text_prompt: str = Field(min_length=1)
    box_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    text_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=1, le=200)
    use_sam_masks: bool = True


class OpenVocabDetection(BaseModel):
    label: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(min_length=4, max_length=4)
    polygon: Optional[List[List[float]]] = None


class OpenVocabDetectResponse(BaseModel):
    session_id: str
    frame_index: int
    detections: List[OpenVocabDetection]
