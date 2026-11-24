#!/usr/bin/env bash
# prepare_data.sh — run on the DATA machine. Downloads ConnectomeBench2 v2 into the shared
# storage and verifies it. Needs only python3 (>=3.9) + internet; no GPU, no CUDA.
#
#   bash scripts/lab/prepare_data.sh                 # full v2 (~235 GB, 903 shards)
#   MAX_SHARDS=4 bash scripts/lab/prepare_data.sh    # small subset to test the pipeline first
#
# Safe to re-run: already-downloaded files are skipped, interrupted downloads resume.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/lab/lab.env

MAX_SHARDS=${MAX_SHARDS:-}
DATA=${DATA:-$CB2_DATA}

need_gb=240
[[ -n "$MAX_SHARDS" ]] && need_gb=$(( MAX_SHARDS * 3 / 4 + 2 ))
mkdir -p "$DATA" "$HF_HOME"
free_gb=$(df -Pk "$DATA" | awk 'NR==2 {print int($4/1024/1024)}')
echo "target: $DATA  (free ${free_gb} GB, need ~${need_gb} GB)"
if (( free_gb < need_gb )); then echo "ERROR: not enough space on $DATA"; exit 1; fi

# Small throw-away environment just for the download.
VENV=${DATA_VENV:-$HOME/.venvs/cb2-data}
DEPS=("huggingface_hub>=0.24" pyarrow pandas numpy)
if ! "$VENV/bin/python" -c "import huggingface_hub, pyarrow, pandas" 2>/dev/null; then
  rm -rf "$VENV"
  if python3 -m venv "$VENV" 2>/dev/null; then
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q "${DEPS[@]}"
  else
    # no python3-venv (needs root to install) -> use uv, which installs into ~/.local/bin
    rm -rf "$VENV"
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
      echo "python3-venv missing; installing uv into ~/.local/bin ..."
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" -q "${DEPS[@]}"
  fi
fi

ARGS=(--out "$DATA" --revision v2 --workers "${DL_WORKERS:-8}")
[[ -n "$MAX_SHARDS" ]] && ARGS+=(--max-shards "$MAX_SHARDS")
"$VENV/bin/python" scripts/training/download_dataset.py "${ARGS[@]}"

# sha256 check of every shard against shards.csv (reads all data once; ~10-30 min for the full set)
if [[ "${VERIFY:-1}" == "1" ]]; then
  "$VENV/bin/python" - "$DATA" "${MAX_SHARDS:-}" <<'EOF'
import csv, hashlib, sys
from pathlib import Path
root, max_shards = Path(sys.argv[1]), sys.argv[2]
rows = list(csv.DictReader(open(root / "shards.csv")))
present = [r for r in rows if (root / r["path"]).exists()]
bad = []
for i, r in enumerate(present, 1):
    h = hashlib.sha256()
    with open(root / r["path"], "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    if h.hexdigest() != r["sha256"]:
        bad.append(r["path"])
    if i % 50 == 0:
        print(f"  verified {i}/{len(present)}", flush=True)
missing = len(rows) - len(present)
print(f"verified {len(present)} shards, {len(bad)} corrupt, {missing} not downloaded"
      + (" (expected with MAX_SHARDS)" if max_shards else ""))
if bad:
    print("corrupt (delete and re-run):", *bad, sep="\n  ")
    sys.exit(1)
if missing and not max_shards:
    sys.exit(1)
EOF
fi

# Private by default (owner only). SHARE_DATA=1 makes the dataset readable by other users.
if [[ -n "${SHARE_DATA:-}" ]]; then chmod -R a+rX "$DATA" 2>/dev/null || true
else chmod -R go-rwx "$DATA" 2>/dev/null || true; fi
echo
echo "Dataset ready at $DATA"
du -sh "$DATA" 2>/dev/null || true
