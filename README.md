# Automatic Annotation Webapp (Notebook/Colab Friendly)

This project provides a local webapp for batch image annotation with:

- Manual annotation on images using:
  - Rectangular bounding boxes
  - Polygon drawing
- Per-object label names
- A **Propagate** action that applies SAM2-style propagation to all frames in the batch
  - Uses real SAM2 when configured
  - Falls back to a built-in tracker if SAM2 is not available
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
4. Click **Propagate with SAM2**.
5. Review each frame and manually add/delete as needed.
6. Click **Export JSON** and download the final output.

---

## 3) Run from Google Colab / Jupyter

Create a cell:

```python
!pip install -q -r requirements.txt
```

Then:

```python
from notebook_launcher import launch
info = launch(port=8000, image_dir="/content/images")
info
```

For Colab, if you need public access:

```python
!pip install -q pyngrok
from notebook_launcher import launch
info = launch(port=8000, image_dir="/content/images", enable_ngrok=True, ngrok_auth_token="YOUR_TOKEN")
info
```

Use `info["public_url"]` (or `info["local_url"]`) to open the app.

---

## 4) Enable real SAM2 propagation (optional)

The app can call SAM2 if it is installed and these env vars are set:

- `SAM2_MODEL_CFG` (model config path, e.g. yaml)
- `SAM2_CHECKPOINT` (checkpoint path, e.g. `.pt`)

Example:

```bash
export SAM2_MODEL_CFG="/path/to/sam2_hiera_l.yaml"
export SAM2_CHECKPOINT="/path/to/sam2_hiera_large.pt"
```

If SAM2 is missing or misconfigured, the app still works via the fallback propagator.

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
