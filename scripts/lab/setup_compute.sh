#!/usr/bin/env bash
# setup_compute.sh — run once on EACH compute machine (from the repo root of a clone).
# Creates the Python 3.11 training environment, caches the ImageNet ViT weights on the
# shared storage, then runs the CPU smoke test and the GPU/data preflight.
#
#   bash scripts/lab/setup_compute.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/lab/lab.env

# ── 1. uv (Python + dependency manager; installs into ~/.local/bin, no root needed) ──
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ── 2. environment (Python 3.11 is fetched by uv if the system lacks it) ──
export UV_PROJECT_ENVIRONMENT="$CB2_VENV"
uv sync --extra train
PY="$CB2_VENV/bin/python"

if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  if [[ -n "${TORCH_INDEX:-}" ]]; then
    echo "default torch cannot see a GPU; reinstalling torch from $TORCH_INDEX"
    uv pip install --python "$PY" --reinstall torch torchvision --index-url "$TORCH_INDEX"
  else
    echo "WARNING: torch cannot see a GPU. Check 'nvidia-smi'; if the driver is older than the"
    echo "         torch CUDA build, set TORCH_INDEX in lab.env (e.g. .../whl/cu121) and re-run."
  fi
fi
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

# ── 3. ImageNet weights → shared TORCH_HOME (needs internet once, on any machine) ──
mkdir -p "$TORCH_HOME"
"$PY" - "$MODEL" <<'EOF' || echo "WARNING: could not download ImageNet weights (no internet?). Run this script once on a machine with internet."
import sys
from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights
w = {"vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1, "vit_l_16": ViT_L_16_Weights.IMAGENET1K_V1}
for name in {"vit_b_16", sys.argv[1]} & set(w):
    w[name].get_state_dict(progress=False)
    print("cached", name)
EOF

# ── 4. CPU smoke test (uses the dataset's demo.parquet, no internet needed) ──
DEMO="$CB2_DATA/demo.parquet"
if [[ -f "$DEMO" ]]; then
  OMP_NUM_THREADS=4 "$PY" scripts/training/smoke_test.py --demo "$DEMO"
else
  echo "skip smoke test: $DEMO not found (run prepare_data.sh on the data machine first)"
fi

# ── 5. preflight: GPUs, dataset on the shared storage, caches, permissions ──
"$PY" scripts/training/check_env.py --data "$CB2_DATA" --runs "$CB2_RUNS" \
  --ngpu "$NGPU" --num-workers "$WORKERS" --models "$MODEL" ${PARTIAL_OK:+--partial-ok} || {
  echo; echo "preflight reported FAIL items above - fix them before training."; exit 1; }
echo; echo "compute machine ready. Start training with: bash scripts/lab/train.sh"
