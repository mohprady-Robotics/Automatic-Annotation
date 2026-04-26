# Automatic Annotation Webapp (Notebook/Colab Friendly)

This project provides a local webapp for batch image annotation with:

- Manual annotation on images using:
  - Rectangular bounding boxes
  - Polygon drawing
- Per-object label names
- A **Propagate** action that applies SAM2-style propagation to all frames in the batch
  - Uses real SAM2 when configured
  - Uses Grounding DINO-guided fallback tracking when SAM2 is not available
  - Refines boxes to segmentation polygons with SAM masks when available
  - Grounding DINO is mandatory for propagation and open-vocabulary detection
- Manual cleanup after propagation (delete/add annotations)
- JSON export containing:
  - all per-frame annotations
  - label + track IDs
  - per-track frame coverage summary

---

## 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> Optional (for public Colab access): `pip install pyngrok`

---

## 2) Run locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- `http://127.0.0.1:8000` (local machine)

In the UI:

1. Enter an image directory path available to the backend (example: `/content/images` on Colab).
2. Click **Load Session**.
3. Draw manual labels on a key frame.
4. Use **Open-vocabulary detect (Grounding DINO + SAM)**:
   - enter a text prompt (for example: `person . bicycle . dog`)
   - run detection on current frame
   - optionally enable SAM mask generation to get polygon masks
   - select suggested detections and add them as manual annotations
5. Click **Propagate with SAM2**.
6. Review each frame and manually add/delete as needed.
7. Click **Export JSON** and download the final output.

---

## 3) Run from Google Colab / Jupyter

### 3.1 Full SAM2 + Grounding DINO setup (install + checkpoint download + env update)

This downloads a SAM 2.1 checkpoint and writes `.env.colab` automatically.
By default, it also enables Grounding DINO-based box refinement.

```python
# Choose tiny, small, base_plus, or large
!bash scripts/setup_sam2_colab.sh tiny
```

After running the script, start the app:

```python
from notebook_launcher import launch
info = launch(port=8000, image_dir="/content/images")
info
```

`notebook_launcher` auto-loads `.env.colab`, so SAM2 config and checkpoint paths are picked up automatically.

For Colab, if you need public access:

```python
!pip install -q pyngrok
from notebook_launcher import launch
info = launch(port=8000, image_dir="/content/images", enable_ngrok=True, ngrok_auth_token="YOUR_TOKEN")
info
```

Use `info["public_url"]` (or `info["local_url"]`) to open the app.

---

## 4) SAM2 environment details

The app reads environment variables from:

1. `.env`
2. `.env.colab`
3. optional override file specified via `ANNOTATION_APP_ENV_FILE`

Required for real SAM2 propagation:

- `SAM2_MODEL_CFG` (config path, e.g. `configs/sam2.1/sam2.1_hiera_t.yaml`)
- `SAM2_CHECKPOINT` (absolute checkpoint file path)

If SAM2 is missing or misconfigured, propagation continues with Grounding DINO-guided fallback tracking.

### Grounding DINO requirement

Grounding DINO is required by the backend for:
- fallback propagation (box refinement/tracking guidance)
- open-vocabulary text prompt detections in the GUI

SAM mask extraction is used for higher-accuracy segmentation polygons:
- open-vocabulary detections can include polygons (`use_sam_masks=true`)
- propagation can emit polygon annotations when SAM masks are available

If Grounding DINO is unavailable, `/api/propagate` and `/api/open_vocab/detect` return an error.

The GUI also supports **open-vocabulary prompt detections** through Grounding DINO:
- panel: `2b) Open-vocabulary detect (Grounding DINO)`
- endpoint: `POST /api/open_vocab/detect`
- detections can be selected and converted into manual annotations before propagation.

Environment variables:

- `ENABLE_GROUNDING_DINO` (must be `1`)
- `GROUNDING_DINO_MODEL_ID` (default `IDEA-Research/grounding-dino-tiny`)
- `GROUNDING_DINO_BOX_THRESHOLD`
- `GROUNDING_DINO_TEXT_THRESHOLD`
- `GROUNDING_DINO_MIN_IOU`
- `GROUNDING_DINO_BLEND`
- `GROUNDING_DINO_SCORE_WEIGHT`
- `ENABLE_SAM_MASKS` (`1` to enable SAM polygon extraction)
- `SAM_MODEL_ID` (default `facebook/sam-vit-base`)

When Grounding DINO refinement is used, propagation backend is reported as
`sam2_fallback_grounding_dino`.

### Setup script options

```bash
# Download one model (default: large)
bash scripts/setup_sam2_colab.sh tiny

# Download all SAM2.1 checkpoints
DOWNLOAD_ALL=1 bash scripts/setup_sam2_colab.sh large

# Reinstall SAM2 with full dependency resolution (may be heavy)
FORCE_REINSTALL=1 bash scripts/setup_sam2_colab.sh tiny

# Skip dependency install and only download/write env
SKIP_DEP_INSTALL=1 bash scripts/setup_sam2_colab.sh small

```

---

## 5) Export format

JSON export includes:

- `images`: frame index and source file path/name
- `annotations`: flattened list (with `track_id`, `label`, geometry, `source`)
- `annotations_by_frame`: grouped by frame index
- `tracks`: summary per track (`first_frame`, `last_frame`, covered frames)

Each annotation has one of:

- `shape_type = "bbox"` with `bbox = [x1, y1, x2, y2]`
- `shape_type = "polygon"` with `polygon = [[x, y], ...]`
