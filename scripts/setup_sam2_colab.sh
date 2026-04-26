#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:-large}"
DOWNLOAD_ALL="${DOWNLOAD_ALL:-0}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
SKIP_DEP_INSTALL="${SKIP_DEP_INSTALL:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints/sam2.1"
ENV_FILE="${ROOT_DIR}/.env.colab"

declare -A CHECKPOINT_URLS
CHECKPOINT_URLS[tiny]="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
CHECKPOINT_URLS[small]="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
CHECKPOINT_URLS[base_plus]="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
CHECKPOINT_URLS[large]="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

declare -A CONFIG_NAMES
CONFIG_NAMES[tiny]="configs/sam2.1/sam2.1_hiera_t.yaml"
CONFIG_NAMES[small]="configs/sam2.1/sam2.1_hiera_s.yaml"
CONFIG_NAMES[base_plus]="configs/sam2.1/sam2.1_hiera_b+.yaml"
CONFIG_NAMES[large]="configs/sam2.1/sam2.1_hiera_l.yaml"

download_file() {
  local url="$1"
  local destination="$2"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$destination" "$url"
    return
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$url" -o "$destination"
    return
  fi
  echo "Neither wget nor curl is available. Install one of them first."
  exit 1
}

download_checkpoint() {
  local size="$1"
  local url="${CHECKPOINT_URLS[$size]}"
  local filename
  filename="$(basename "$url")"
  local destination="${CHECKPOINT_DIR}/${filename}"
  if [[ -f "$destination" ]]; then
    echo "Checkpoint already present: $destination"
    return
  fi
  echo "Downloading ${filename}..."
  download_file "$url" "$destination"
}

if [[ "$MODEL_SIZE" != "tiny" && "$MODEL_SIZE" != "small" && "$MODEL_SIZE" != "base_plus" && "$MODEL_SIZE" != "large" ]]; then
  echo "Invalid model size '${MODEL_SIZE}'. Use one of: tiny, small, base_plus, large."
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

if [[ "$SKIP_DEP_INSTALL" != "1" ]]; then
  echo "Installing app dependencies..."
  python3 -m pip install -U pip
  python3 -m pip install -r "${ROOT_DIR}/requirements.txt"

  # Keep existing torch/torchvision from Colab when possible to avoid forcing large reinstalls.
  if [[ "$FORCE_REINSTALL" == "1" ]]; then
    echo "Force reinstalling SAM2 with dependencies..."
    python3 -m pip install --upgrade "git+https://github.com/facebookresearch/sam2.git"
  else
    echo "Installing SAM2 helper deps and SAM2 package..."
    python3 -m pip install --upgrade "hydra-core>=1.3.2" "iopath>=0.1.10" "tqdm>=4.66.1"
    python3 -m pip install --upgrade --no-deps "git+https://github.com/facebookresearch/sam2.git"
  fi

  echo "Installing Grounding DINO + SAM mask dependencies..."
  python3 -m pip install --upgrade transformers accelerate timm opencv-python
else
  echo "Skipping dependency installation (SKIP_DEP_INSTALL=1)."
fi

if [[ "$DOWNLOAD_ALL" == "1" ]]; then
  for size in tiny small base_plus large; do
    download_checkpoint "$size"
  done
else
  download_checkpoint "$MODEL_SIZE"
fi

SELECTED_CKPT="${CHECKPOINT_DIR}/sam2.1_hiera_${MODEL_SIZE}.pt"
SELECTED_CFG="${CONFIG_NAMES[$MODEL_SIZE]}"

cat > "$ENV_FILE" <<EOF
SAM2_MODEL_CFG=${SELECTED_CFG}
SAM2_CHECKPOINT=${SELECTED_CKPT}
ENABLE_GROUNDING_DINO=1
GROUNDING_DINO_MODEL_ID=IDEA-Research/grounding-dino-tiny
GROUNDING_DINO_BOX_THRESHOLD=0.30
GROUNDING_DINO_TEXT_THRESHOLD=0.25
GROUNDING_DINO_MIN_IOU=0.05
GROUNDING_DINO_BLEND=0.70
GROUNDING_DINO_SCORE_WEIGHT=0.20
ENABLE_SAM_MASKS=1
SAM_MODEL_ID=facebook/sam-vit-base
EOF

echo
echo "Setup complete."
echo "Wrote env file: ${ENV_FILE}"
echo "Model config: ${SELECTED_CFG}"
echo "Model checkpoint: ${SELECTED_CKPT}"
