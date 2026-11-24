#!/usr/bin/env bash
# train.sh — run on a COMPUTE machine: preflight → train → test evaluation → ablations → tables.
# Reads the dataset from shared storage, writes everything to $CB2_RUNS/$NAME.
#
#   bash scripts/lab/train.sh                                  # ViT-B, 1 GPU, paper recipe
#   NGPU=2 BS=512 NAME=vitb_2gpu bash scripts/lab/train.sh
#   MODEL=vit_l_16 NGPU=3 BS=768 bash scripts/lab/train.sh
#   EPOCHS=1 MAX_SAMPLES=5000 NAME=quick bash scripts/lab/train.sh   # ~short end-to-end test
#
# Long runs: start inside tmux/screen (or nohup) so an SSH disconnect does not kill them.
# Re-running the same NAME resumes from its last checkpoint.
#
# SLURM: submit with `sbatch scripts/lab/train.sh` after uncommenting/adjusting:
#SBATCH --job-name=cb2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=cb2-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/lab/lab.env
source "$CB2_VENV/bin/activate"

NAME=${NAME:-${MODEL}_full}
RUN_DIR="$CB2_RUNS/$NAME"
mkdir -p "$RUN_DIR"
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# Record exactly what ran.
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "git: $(git rev-parse HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | wc -l) dirty files"
  echo "NGPU=$NGPU MODEL=$MODEL BS=$BS EPOCHS=$EPOCHS WORKERS=$WORKERS MAX_SAMPLES=${MAX_SAMPLES:-}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
} >> "$RUN_DIR/run_info.txt"

python scripts/training/check_env.py --data "$CB2_DATA" --runs "$CB2_RUNS" \
  --ngpu "$NGPU" --num-workers "$WORKERS" --models "$MODEL" ${PARTIAL_OK:+--partial-ok} \
  | tee "$RUN_DIR/preflight.txt"

EXTRA="${EXTRA:-}"
EVAL_EXTRA="${EVAL_EXTRA:-}"
if [[ -n "${MAX_SAMPLES:-}" ]]; then   # quick test: cap train rows, val rows and test rows
  EXTRA="$EXTRA --max-samples $MAX_SAMPLES --max-val-samples $(( MAX_SAMPLES / 5 + 1 ))"
  EVAL_EXTRA="$EVAL_EXTRA --max-val-samples $(( MAX_SAMPLES / 5 + 1 ))"
fi
if (( EPOCHS <= 5 )); then EXTRA="$EXTRA --warmup-epochs 0"; fi   # paper warm-up is 5 epochs

DATA="$CB2_DATA" NGPU="$NGPU" MODEL="$MODEL" BS="$BS" EPOCHS="$EPOCHS" WORKERS="$WORKERS" \
NAME="$NAME" CKPT_DIR="$RUN_DIR/checkpoints" OUT="$RUN_DIR/results" EXTRA="$EXTRA" EVAL_EXTRA="$EVAL_EXTRA" \
  bash scripts/training/reproduce_paper.sh

echo; echo "done: tables in $RUN_DIR/results/analysis/report.md"
