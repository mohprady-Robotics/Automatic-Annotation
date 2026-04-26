const state = {
  sessionId: null,
  frameUrls: [],
  imageWidth: 0,
  imageHeight: 0,
  currentFrame: 0,
  tool: "rect",
  isDrawing: false,
  draftRect: null,
  draftPolygon: [],
  imageElement: null,
  selectedAnnotationId: null,
  nextTrackId: 1,
  annotationsByFrame: {},
  openVocabDetectionsByFrame: {},
  openVocabSelectedDetectionIds: {},
  samRefine: {
    enabled: false,
    clickMode: "positive",
    pointsByFrame: {},
    previewByFrame: {},
  },
};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const imageDirInput = document.getElementById("imageDirInput");
const loadSessionBtn = document.getElementById("loadSessionBtn");
const sessionMeta = document.getElementById("sessionMeta");
const labelInput = document.getElementById("labelInput");
const rectToolBtn = document.getElementById("rectToolBtn");
const polygonToolBtn = document.getElementById("polygonToolBtn");
const selectToolBtn = document.getElementById("selectToolBtn");
const finishPolygonBtn = document.getElementById("finishPolygonBtn");
const clearFrameBtn = document.getElementById("clearFrameBtn");
const propagateBtn = document.getElementById("propagateBtn");
const exportBtn = document.getElementById("exportBtn");
const statusText = document.getElementById("statusText");
const downloadLink = document.getElementById("downloadLink");
const prevFrameBtn = document.getElementById("prevFrameBtn");
const nextFrameBtn = document.getElementById("nextFrameBtn");
const frameCounter = document.getElementById("frameCounter");
const selectedInfo = document.getElementById("selectedInfo");
const deleteSelectedBtn = document.getElementById("deleteSelectedBtn");
const annotationList = document.getElementById("annotationList");
const openVocabPromptInput = document.getElementById("openVocabPromptInput");
const openVocabBoxThresholdInput = document.getElementById("openVocabBoxThresholdInput");
const openVocabTextThresholdInput = document.getElementById("openVocabTextThresholdInput");
const openVocabUseSamMasksInput = document.getElementById("openVocabUseSamMasksInput");
const detectOpenVocabBtn = document.getElementById("detectOpenVocabBtn");
const addSelectedDetectionsBtn = document.getElementById("addSelectedDetectionsBtn");
const openVocabDetections = document.getElementById("openVocabDetections");
const samRefineModeInput = document.getElementById("samRefineModeInput");
const samPositiveModeBtn = document.getElementById("samPositiveModeBtn");
const samNegativeModeBtn = document.getElementById("samNegativeModeBtn");
const runSamRefineBtn = document.getElementById("runSamRefineBtn");
const clearSamClicksBtn = document.getElementById("clearSamClicksBtn");
const addSamMaskBtn = document.getElementById("addSamMaskBtn");
const samClicksInfo = document.getElementById("samClicksInfo");

function setStatus(text) {
  statusText.textContent = `Status: ${text}`;
}

function setTool(tool) {
  state.tool = tool;
  rectToolBtn.classList.toggle("active", tool === "rect");
  polygonToolBtn.classList.toggle("active", tool === "polygon");
  selectToolBtn.classList.toggle("active", tool === "select");
  canvas.style.cursor = tool === "select" ? "pointer" : "crosshair";
}

function currentFrameAnnotations() {
  const key = String(state.currentFrame);
  if (!state.annotationsByFrame[key]) {
    state.annotationsByFrame[key] = [];
  }
  return state.annotationsByFrame[key];
}

function generateAnnotationId(trackId, frameIndex) {
  return `${trackId}_${frameIndex}_${Math.random().toString(16).slice(2, 8)}`;
}

function currentFrameSamPoints() {
  const key = String(state.currentFrame);
  if (!state.samRefine.pointsByFrame[key]) {
    state.samRefine.pointsByFrame[key] = { positive: [], negative: [] };
  }
  return state.samRefine.pointsByFrame[key];
}

function currentFrameSamPreview() {
  return state.samRefine.previewByFrame[String(state.currentFrame)] || null;
}

function setSamClickMode(mode) {
  state.samRefine.clickMode = mode;
  samPositiveModeBtn.classList.toggle("active", mode === "positive");
  samNegativeModeBtn.classList.toggle("active", mode === "negative");
}

function refreshSamClicksInfo() {
  const points = currentFrameSamPoints();
  samClicksInfo.textContent = `SAM clicks: ${points.positive.length} positive, ${points.negative.length} negative`;
}

function currentFrameOpenVocabDetections() {
  const key = String(state.currentFrame);
  return state.openVocabDetectionsByFrame[key] || [];
}

function currentFrameSelectedDetectionIds() {
  const key = String(state.currentFrame);
  if (!state.openVocabSelectedDetectionIds[key]) {
    state.openVocabSelectedDetectionIds[key] = [];
  }
  return state.openVocabSelectedDetectionIds[key];
}

function getPointerPosition(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = state.imageWidth / rect.width;
  const scaleY = state.imageHeight / rect.height;
  return {
    x: (evt.clientX - rect.left) * scaleX,
    y: (evt.clientY - rect.top) * scaleY,
  };
}

function clampPoint(point) {
  if (state.imageWidth <= 0 || state.imageHeight <= 0) {
    return { x: 0, y: 0 };
  }
  return {
    x: Math.max(0, Math.min(point.x, state.imageWidth - 1)),
    y: Math.max(0, Math.min(point.y, state.imageHeight - 1)),
  };
}

function normalizeRect(rect) {
  const x1 = Math.min(rect.x1, rect.x2);
  const y1 = Math.min(rect.y1, rect.y2);
  const x2 = Math.max(rect.x1, rect.x2);
  const y2 = Math.max(rect.y1, rect.y2);
  if (x2 - x1 < 2 || y2 - y1 < 2) {
    return null;
  }
  return [x1, y1, x2, y2];
}

function drawPolygon(points, color, lineWidth = 2, fillAlpha = 0.2) {
  if (points.length === 0) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i][0], points[i][1]);
  }
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
  const fillColor = color.replace(")", `, ${fillAlpha})`).replace("rgb", "rgba");
  ctx.fillStyle = fillColor;
  ctx.fill();
}

function colorForTrack(trackId) {
  const hue = (trackId * 47) % 360;
  return `hsl(${hue}, 88%, 55%)`;
}

function hslToRgbString(hsl) {
  const m = hsl.match(/hsl\((\d+),\s*(\d+)%?,\s*(\d+)%?\)/);
  if (!m) return "rgb(59,130,246)";
  const h = Number(m[1]) / 360;
  const s = Number(m[2]) / 100;
  const l = Number(m[3]) / 100;

  function hue2rgb(p, q, t) {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  }

  let r;
  let g;
  let b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.imageElement) return;
  ctx.drawImage(state.imageElement, 0, 0, state.imageWidth, state.imageHeight);

  for (const detection of currentFrameOpenVocabDetections()) {
    const isSelected = currentFrameSelectedDetectionIds().includes(detection.id);
    if (detection.polygon && detection.polygon.length >= 3) {
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeStyle = isSelected ? "#16a34a" : "#f59e0b";
      drawPolygon(detection.polygon, ctx.strokeStyle, ctx.lineWidth, 0.12);
      const x0 = detection.polygon[0][0];
      const y0 = detection.polygon[0][1];
      ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
      ctx.fillRect(x0, Math.max(0, y0 - 18), 200, 18);
      ctx.fillStyle = "#111827";
      ctx.font = "12px Arial";
      ctx.fillText(`det:${detection.label} (${detection.score.toFixed(2)})`, x0 + 4, Math.max(12, y0 - 5));
    } else {
      const [x1, y1, x2, y2] = detection.bbox;
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeStyle = isSelected ? "#16a34a" : "#f59e0b";
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
      ctx.fillRect(x1, Math.max(0, y1 - 18), 200, 18);
      ctx.fillStyle = "#111827";
      ctx.font = "12px Arial";
      ctx.fillText(`det:${detection.label} (${detection.score.toFixed(2)})`, x1 + 4, Math.max(12, y1 - 5));
    }
  }

  const samPreview = currentFrameSamPreview();
  if (samPreview && samPreview.polygon && samPreview.polygon.length >= 3) {
    drawPolygon(samPreview.polygon, "rgb(168, 85, 247)", 3, 0.22);
  }

  for (const annotation of currentFrameAnnotations()) {
    const isSelected = annotation.id === state.selectedAnnotationId;
    const baseColor = hslToRgbString(colorForTrack(annotation.track_id));
    ctx.lineWidth = isSelected ? 4 : 2;
    ctx.strokeStyle = baseColor;
    ctx.fillStyle = "rgba(255,255,255,0.95)";

    if (annotation.shape_type === "bbox" && annotation.bbox) {
      const [x1, y1, x2, y2] = annotation.bbox;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillRect(x1, Math.max(0, y1 - 18), 130, 18);
      ctx.fillStyle = "#111827";
      ctx.font = "12px Arial";
      ctx.fillText(`${annotation.label} #${annotation.track_id}`, x1 + 4, Math.max(12, y1 - 5));
    } else if (annotation.shape_type === "polygon" && annotation.polygon) {
      drawPolygon(annotation.polygon, baseColor, isSelected ? 4 : 2, 0.18);
      const [x0, y0] = annotation.polygon[0];
      ctx.fillStyle = "rgba(255,255,255,0.95)";
      ctx.fillRect(x0, Math.max(0, y0 - 18), 130, 18);
      ctx.fillStyle = "#111827";
      ctx.font = "12px Arial";
      ctx.fillText(`${annotation.label} #${annotation.track_id}`, x0 + 4, Math.max(12, y0 - 5));
    }
  }

  if (state.draftRect) {
    const normalized = normalizeRect(state.draftRect);
    if (normalized) {
      const [x1, y1, x2, y2] = normalized;
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
    }
  }

  if (state.draftPolygon.length > 0) {
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(state.draftPolygon[0][0], state.draftPolygon[0][1]);
    for (let i = 1; i < state.draftPolygon.length; i += 1) {
      ctx.lineTo(state.draftPolygon[i][0], state.draftPolygon[i][1]);
    }
    ctx.stroke();
    for (const point of state.draftPolygon) {
      ctx.beginPath();
      ctx.arc(point[0], point[1], 3, 0, Math.PI * 2);
      ctx.fillStyle = "#ef4444";
      ctx.fill();
    }
  }

  if (state.samRefine.enabled) {
    const samPoints = currentFrameSamPoints();
    for (const point of samPoints.positive) {
      ctx.beginPath();
      ctx.arc(point[0], point[1], 4, 0, Math.PI * 2);
      ctx.fillStyle = "#22c55e";
      ctx.fill();
      ctx.strokeStyle = "#14532d";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    for (const point of samPoints.negative) {
      ctx.beginPath();
      ctx.arc(point[0], point[1], 4, 0, Math.PI * 2);
      ctx.fillStyle = "#ef4444";
      ctx.fill();
      ctx.strokeStyle = "#7f1d1d";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    const preview = currentFrameSamPreview();
    if (preview && preview.polygon && preview.polygon.length >= 3) {
      drawPolygon(preview.polygon, "rgb(168,85,247)", 2, 0.16);
    }
  }
}

function refreshOpenVocabDetectionsList() {
  const detections = currentFrameOpenVocabDetections();
  const selectedIds = currentFrameSelectedDetectionIds();
  openVocabDetections.innerHTML = "";
  if (detections.length === 0) {
    openVocabDetections.innerHTML = "<p>No detections on this frame.</p>";
    return;
  }

  detections.forEach((detection) => {
    const row = document.createElement("div");
    row.className = "detection-row";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selectedIds.includes(detection.id);
    checkbox.addEventListener("change", () => {
      const next = new Set(currentFrameSelectedDetectionIds());
      if (checkbox.checked) {
        next.add(detection.id);
      } else {
        next.delete(detection.id);
      }
      state.openVocabSelectedDetectionIds[String(state.currentFrame)] = [...next];
      render();
    });

    const meta = document.createElement("div");
    meta.className = "meta";
    const [x1, y1, x2, y2] = detection.bbox;
    const geometryText = detection.polygon && detection.polygon.length >= 3
      ? `polygon_points=${detection.polygon.length}`
      : `bbox=[${x1.toFixed(1)}, ${y1.toFixed(1)}, ${x2.toFixed(1)}, ${y2.toFixed(1)}]`;
    meta.innerHTML = `<strong>${detection.label}</strong>
      <span>score=${detection.score.toFixed(3)}</span>
      <span>${geometryText}</span>`;

    row.appendChild(checkbox);
    row.appendChild(meta);
    openVocabDetections.appendChild(row);
  });
}

function refreshAnnotationList() {
  const annotations = currentFrameAnnotations();
  annotationList.innerHTML = "";
  if (annotations.length === 0) {
    annotationList.innerHTML = "<p>No annotations for this frame.</p>";
    return;
  }
  annotations.forEach((annotation) => {
    const row = document.createElement("div");
    row.className = "annotation-row";
    row.textContent = `${annotation.label} (#${annotation.track_id}) - ${annotation.shape_type} - ${annotation.source}`;
    row.style.borderColor = annotation.id === state.selectedAnnotationId ? "#2563eb" : "#e5e7eb";
    row.addEventListener("click", () => {
      state.selectedAnnotationId = annotation.id;
      updateSelectedInfo();
      render();
      refreshAnnotationList();
    });
    annotationList.appendChild(row);
  });
}

function updateSelectedInfo() {
  const annotations = currentFrameAnnotations();
  const selected = annotations.find((annotation) => annotation.id === state.selectedAnnotationId);
  if (!selected) {
    selectedInfo.textContent = "None selected.";
    return;
  }
  selectedInfo.textContent = `ID=${selected.id}, track=${selected.track_id}, label=${selected.label}, source=${selected.source}`;
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0];
    const yi = polygon[i][1];
    const xj = polygon[j][0];
    const yj = polygon[j][1];
    const intersect = yi > point.y !== yj > point.y
      && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function selectAnnotationAtPoint(point) {
  const annotations = currentFrameAnnotations();
  for (let i = annotations.length - 1; i >= 0; i -= 1) {
    const annotation = annotations[i];
    if (annotation.shape_type === "bbox" && annotation.bbox) {
      const [x1, y1, x2, y2] = annotation.bbox;
      if (point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2) {
        return annotation.id;
      }
    } else if (annotation.shape_type === "polygon" && annotation.polygon) {
      if (pointInPolygon(point, annotation.polygon)) {
        return annotation.id;
      }
    }
  }
  return null;
}

async function loadFrame(frameIndex) {
  if (!state.sessionId) return;
  state.currentFrame = Math.max(0, Math.min(frameIndex, state.frameUrls.length - 1));
  const image = new Image();
  image.src = state.frameUrls[state.currentFrame];
  image.crossOrigin = "anonymous";
  await image.decode();
  state.imageElement = image;
  if (!state.imageWidth || !state.imageHeight) {
    state.imageWidth = image.naturalWidth || image.width;
    state.imageHeight = image.naturalHeight || image.height;
  }
  canvas.width = state.imageWidth;
  canvas.height = state.imageHeight;
  frameCounter.textContent = `Frame ${state.currentFrame + 1} / ${state.frameUrls.length}`;
  state.selectedAnnotationId = null;
  state.draftPolygon = [];
  state.draftRect = null;
  refreshSamClicksInfo();
  updateSelectedInfo();
  refreshAnnotationList();
  refreshOpenVocabDetectionsList();
  render();
}

function addRectangle(rect) {
  const label = labelInput.value.trim();
  if (!label) {
    setStatus("enter a label before drawing");
    return;
  }
  const trackId = state.nextTrackId;
  state.nextTrackId += 1;
  currentFrameAnnotations().push({
    id: generateAnnotationId(trackId, state.currentFrame),
    track_id: trackId,
    label,
    frame_index: state.currentFrame,
    shape_type: "bbox",
    source: "manual",
    bbox: rect,
    polygon: null,
  });
  refreshAnnotationList();
  setStatus("added rectangle annotation");
  render();
}

function addPolygon(points) {
  const label = labelInput.value.trim();
  if (!label) {
    setStatus("enter a label before drawing");
    return;
  }
  if (points.length < 3) {
    setStatus("polygon requires at least 3 points");
    return;
  }
  const trackId = state.nextTrackId;
  state.nextTrackId += 1;
  currentFrameAnnotations().push({
    id: generateAnnotationId(trackId, state.currentFrame),
    track_id: trackId,
    label,
    frame_index: state.currentFrame,
    shape_type: "polygon",
    source: "manual",
    bbox: null,
    polygon: points,
  });
  state.draftPolygon = [];
  refreshAnnotationList();
  setStatus("added polygon annotation");
  render();
}

canvas.addEventListener("mousedown", (evt) => {
  if (!state.imageElement) return;
  const point = clampPoint(getPointerPosition(evt));

  if (state.samRefine.enabled) {
    const samPoints = currentFrameSamPoints();
    const slot = state.samRefine.clickMode === "negative" ? samPoints.negative : samPoints.positive;
    slot.push([point.x, point.y]);
    state.samRefine.previewByFrame[String(state.currentFrame)] = null;
    refreshSamClicksInfo();
    render();
    return;
  }

  if (state.tool === "select") {
    state.selectedAnnotationId = selectAnnotationAtPoint(point);
    updateSelectedInfo();
    refreshAnnotationList();
    render();
    return;
  }

  if (state.tool === "rect") {
    state.isDrawing = true;
    state.draftRect = { x1: point.x, y1: point.y, x2: point.x, y2: point.y };
  } else if (state.tool === "polygon") {
    state.draftPolygon.push([point.x, point.y]);
    render();
  }
});

canvas.addEventListener("mousemove", (evt) => {
  if (!state.imageElement || state.tool !== "rect" || !state.isDrawing || !state.draftRect) return;
  const point = clampPoint(getPointerPosition(evt));
  state.draftRect.x2 = point.x;
  state.draftRect.y2 = point.y;
  render();
});

canvas.addEventListener("mouseup", () => {
  if (state.tool !== "rect" || !state.isDrawing || !state.draftRect) return;
  state.isDrawing = false;
  const normalized = normalizeRect(state.draftRect);
  state.draftRect = null;
  if (normalized) {
    addRectangle(normalized);
  }
  render();
});

finishPolygonBtn.addEventListener("click", () => {
  if (state.tool !== "polygon") {
    setStatus("switch to polygon tool first");
    return;
  }
  addPolygon([...state.draftPolygon]);
});

clearFrameBtn.addEventListener("click", () => {
  state.annotationsByFrame[String(state.currentFrame)] = [];
  state.selectedAnnotationId = null;
  updateSelectedInfo();
  refreshAnnotationList();
  render();
  setStatus("cleared current frame annotations");
});

deleteSelectedBtn.addEventListener("click", () => {
  if (!state.selectedAnnotationId) {
    setStatus("select an annotation first");
    return;
  }
  const key = String(state.currentFrame);
  state.annotationsByFrame[key] = currentFrameAnnotations().filter(
    (annotation) => annotation.id !== state.selectedAnnotationId,
  );
  state.selectedAnnotationId = null;
  updateSelectedInfo();
  refreshAnnotationList();
  render();
  setStatus("deleted selected annotation");
});

prevFrameBtn.addEventListener("click", () => {
  if (!state.sessionId) return;
  loadFrame(state.currentFrame - 1);
});

nextFrameBtn.addEventListener("click", () => {
  if (!state.sessionId) return;
  loadFrame(state.currentFrame + 1);
});

rectToolBtn.addEventListener("click", () => setTool("rect"));
polygonToolBtn.addEventListener("click", () => setTool("polygon"));
selectToolBtn.addEventListener("click", () => setTool("select"));

loadSessionBtn.addEventListener("click", async () => {
  const imageDir = imageDirInput.value.trim();
  if (!imageDir) {
    setStatus("please enter image directory path");
    return;
  }
  try {
    setStatus("loading session...");
    const response = await fetch("/api/session/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_dir: imageDir }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to create session.");
    }
    state.sessionId = payload.session_id;
    state.frameUrls = payload.frame_urls;
    state.imageWidth = payload.width;
    state.imageHeight = payload.height;
    state.currentFrame = 0;
    state.nextTrackId = 1;
    state.annotationsByFrame = {};
    state.openVocabDetectionsByFrame = {};
    state.openVocabSelectedDetectionIds = {};
    state.samRefine.pointsByFrame = {};
    state.samRefine.previewByFrame = {};
    sessionMeta.textContent = `session=${payload.session_id} | frames=${payload.frame_count} | size=${payload.width}x${payload.height}`;
    await loadFrame(0);
    setStatus("session loaded");
    downloadLink.hidden = true;
  } catch (error) {
    setStatus(error.message);
  }
});

detectOpenVocabBtn.addEventListener("click", async () => {
  if (!state.sessionId) {
    setStatus("load a session first");
    return;
  }
  const prompt = openVocabPromptInput.value.trim();
  if (!prompt) {
    setStatus("enter a text prompt first");
    return;
  }

  const boxThreshold = Number(openVocabBoxThresholdInput.value);
  const textThreshold = Number(openVocabTextThresholdInput.value);
  if (Number.isNaN(boxThreshold) || Number.isNaN(textThreshold)) {
    setStatus("threshold values must be numbers");
    return;
  }

  try {
    setStatus("running open-vocab detection...");
    const response = await fetch("/api/open_vocab/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        frame_index: state.currentFrame,
        text_prompt: prompt,
        box_threshold: boxThreshold,
        text_threshold: textThreshold,
        top_k: 30,
        use_sam_masks: Boolean(openVocabUseSamMasksInput.checked),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Open-vocab detection failed.");
    }

    const key = String(state.currentFrame);
    state.openVocabDetectionsByFrame[key] = payload.detections.map((det, index) => ({
      id: `det_${state.currentFrame}_${index}`,
      label: det.label,
      score: Number(det.score),
      bbox: det.bbox,
      polygon: det.polygon || null,
    }));
    state.openVocabSelectedDetectionIds[key] = state.openVocabDetectionsByFrame[key].map((det) => det.id);
    refreshOpenVocabDetectionsList();
    render();
    setStatus(`detected ${payload.detections.length} open-vocab boxes`);
  } catch (error) {
    setStatus(error.message);
  }
});

addSelectedDetectionsBtn.addEventListener("click", () => {
  if (!state.sessionId) {
    setStatus("load a session first");
    return;
  }
  const detections = currentFrameOpenVocabDetections();
  const selected = new Set(currentFrameSelectedDetectionIds());
  const chosen = detections.filter((det) => selected.has(det.id));
  if (chosen.length === 0) {
    setStatus("select at least one detection");
    return;
  }

  for (const detection of chosen) {
    const trackId = state.nextTrackId;
    state.nextTrackId += 1;
    const annotationBase = {
      id: generateAnnotationId(trackId, state.currentFrame),
      track_id: trackId,
      label: detection.label || openVocabPromptInput.value.trim() || "detected_object",
      frame_index: state.currentFrame,
      source: "manual",
    };
    if (detection.polygon && detection.polygon.length >= 3) {
      currentFrameAnnotations().push({
        ...annotationBase,
        shape_type: "polygon",
        bbox: null,
        polygon: detection.polygon,
      });
    } else {
      currentFrameAnnotations().push({
        ...annotationBase,
        shape_type: "bbox",
        bbox: detection.bbox,
        polygon: null,
      });
    }
  }

  refreshAnnotationList();
  render();
  setStatus(`added ${chosen.length} detection(s) as annotations`);
});

samRefineModeInput.addEventListener("change", () => {
  state.samRefine.enabled = Boolean(samRefineModeInput.checked);
  setStatus(state.samRefine.enabled ? "SAM click mode enabled" : "SAM click mode disabled");
  render();
});

samPositiveModeBtn.addEventListener("click", () => setSamClickMode("positive"));
samNegativeModeBtn.addEventListener("click", () => setSamClickMode("negative"));

clearSamClicksBtn.addEventListener("click", () => {
  const key = String(state.currentFrame);
  state.samRefine.pointsByFrame[key] = { positive: [], negative: [] };
  state.samRefine.previewByFrame[key] = null;
  refreshSamClicksInfo();
  render();
  setStatus("cleared SAM clicks on current frame");
});

runSamRefineBtn.addEventListener("click", async () => {
  if (!state.sessionId) {
    setStatus("load a session first");
    return;
  }
  const points = currentFrameSamPoints();
  if (points.positive.length === 0) {
    setStatus("add at least one positive click");
    return;
  }
  try {
    setStatus("running interactive SAM refinement...");
    let inputBox = null;
    const selected = currentFrameAnnotations().find((annotation) => annotation.id === state.selectedAnnotationId);
    if (selected) {
      if (selected.shape_type === "bbox" && selected.bbox) {
        inputBox = selected.bbox;
      } else if (selected.shape_type === "polygon" && selected.polygon && selected.polygon.length >= 3) {
        const xs = selected.polygon.map((point) => point[0]);
        const ys = selected.polygon.map((point) => point[1]);
        inputBox = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
      }
    }
    const response = await fetch("/api/sam/refine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        frame_index: state.currentFrame,
        positive_points: points.positive,
        negative_points: points.negative,
        input_box: inputBox,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "SAM refinement failed.");
    }
    state.samRefine.previewByFrame[String(state.currentFrame)] = {
      polygon: payload.polygon,
      bbox: payload.bbox,
    };
    render();
    setStatus("SAM refinement preview ready");
  } catch (error) {
    setStatus(error.message);
  }
});

addSamMaskBtn.addEventListener("click", () => {
  if (!state.sessionId) {
    setStatus("load a session first");
    return;
  }
  const preview = currentFrameSamPreview();
  if (!preview || !preview.polygon || preview.polygon.length < 3) {
    setStatus("run SAM refinement first");
    return;
  }
  const label = labelInput.value.trim() || "refined_object";
  const trackId = state.nextTrackId;
  state.nextTrackId += 1;
  currentFrameAnnotations().push({
    id: generateAnnotationId(trackId, state.currentFrame),
    track_id: trackId,
    label,
    frame_index: state.currentFrame,
    shape_type: "polygon",
    source: "manual",
    bbox: null,
    polygon: preview.polygon,
  });
  refreshAnnotationList();
  render();
  setStatus("added SAM refined mask as polygon annotation");
});

propagateBtn.addEventListener("click", async () => {
  if (!state.sessionId) {
    setStatus("load a session first");
    return;
  }
  const keyFrameAnnotations = currentFrameAnnotations();
  if (keyFrameAnnotations.length === 0) {
    setStatus("draw at least one annotation on current frame");
    return;
  }
  try {
    setStatus("running propagation...");
    const response = await fetch("/api/propagate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        key_frame_index: state.currentFrame,
        annotations: keyFrameAnnotations,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Propagation failed.");
    }
    state.annotationsByFrame = payload.frames;
    state.samRefine.previewByFrame[String(state.currentFrame)] = null;
    let maxTrack = 0;
    Object.values(state.annotationsByFrame).forEach((frameAnnotations) => {
      frameAnnotations.forEach((annotation) => {
        maxTrack = Math.max(maxTrack, Number(annotation.track_id || 0));
      });
    });
    state.nextTrackId = maxTrack + 1;
    await loadFrame(state.currentFrame);
    setStatus(`propagation complete via ${payload.backend}`);
  } catch (error) {
    setStatus(error.message);
  }
});

exportBtn.addEventListener("click", async () => {
  if (!state.sessionId) {
    setStatus("load a session first");
    return;
  }
  try {
    setStatus("exporting json...");
    const response = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        annotations_by_frame: state.annotationsByFrame,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Export failed.");
    }
    downloadLink.href = payload.download_url;
    downloadLink.hidden = false;
    downloadLink.textContent = `Download ${payload.filename}`;
    setStatus("export complete");
  } catch (error) {
    setStatus(error.message);
  }
});

setTool("rect");
setSamClickMode("positive");
refreshSamClicksInfo();
setStatus("idle");
