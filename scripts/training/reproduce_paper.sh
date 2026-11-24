#!/usr/bin/env bash
# reproduce_paper.sh — ViT-B headline run + test-set evaluation + Table 1 ablations.
#
# Run on the GPU machine from the repo root:
#   DATA=/data/ConnectomeBench2 NGPU=1 bash scripts/training/reproduce_paper.sh
#
# Paper reference (test, per-sample bAcc @0.5): ViT-B 97.0 split / 93.0 merge, mIoU 0.696.
# Budget: ~15 H100-hours for ViT-B at bs 256 (paper App. B.2).
set -euo pipefail

DATA=${DATA:?set DATA to the dataset root (contains train/ val/ test/)}
NGPU=${NGPU:-1}
MODEL=${MODEL:-vit_b_16}
BS=${BS:-256}                    # GLOBAL batch size; lr = 1e-4 * sqrt(BS/32)
EPOCHS=${EPOCHS:-20}
WORKERS=${WORKERS:-16}
NAME=${NAME:-${MODEL}_full}
CKPT_DIR=${CKPT_DIR:-checkpoints_neurips}
OUT=${OUT:-results/${NAME}}
CFG=src/neurips_training/configs/full_4sp.yaml
TRAIN=src/neurips_training/training.py
EXTRA=${EXTRA:-}                 # extra training flags, e.g. EXTRA="--wandb --run-name vitb"
# EVAL_EXTRA: extra evaluation flags, e.g. EVAL_EXTRA="--max-val-samples 2000" for a quick test

mkdir -p "$OUT/analysis"

# ── 1. train (resumes automatically if a *_last.pt exists) ──
RESUME=""
if [[ -f "$CKPT_DIR/${NAME}_last.pt" ]]; then RESUME="--resume $CKPT_DIR/${NAME}_last.pt"; fi
if [[ "$NGPU" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nproc-per-node="$NGPU")
else
  LAUNCH=(python)
fi
"${LAUNCH[@]}" "$TRAIN" --blend-config "$CFG" --parquet-root "$DATA" \
  --model "$MODEL" --batch-size "$BS" --epochs "$EPOCHS" --num-workers "$WORKERS" \
  --checkpoint-dir "$CKPT_DIR" --checkpoint-name "$NAME" $RESUME $EXTRA \
  2>&1 | tee -a "$OUT/train.log"

BEST="$CKPT_DIR/${NAME}_best.pt"
EVAL=(python "$TRAIN" --mode evaluate --blend-config "$CFG" --parquet-root "$DATA"
      --checkpoint "$BEST" --eval-split test --num-workers "$WORKERS" --batch-size 256 ${EVAL_EXTRA:-})

# ── 2. held-out test set, full model (+ CLS features for calibration / OOD work) ──
"${EVAL[@]}" --export-predictions "$OUT/test_full.parquet" 2>&1 | tee "$OUT/eval_full.log"

# ── 3. Table 1 inference-time modality & channel ablations ──
"${EVAL[@]}" --modalities geom             --export-predictions "$OUT/test_geom_only.parquet"
"${EVAL[@]}" --modalities geom --ablate shape   --export-predictions "$OUT/test_geom_minus_shape.parquet"
"${EVAL[@]}" --modalities geom --ablate normals --export-predictions "$OUT/test_geom_minus_normals.parquet"
"${EVAL[@]}" --modalities geom --ablate masks   --export-predictions "$OUT/test_geom_minus_masks.parquet"
"${EVAL[@]}" --modalities em               --export-predictions "$OUT/test_em_only.parquet"

# ── 4. tables (Table 1, 6, 7, 8 + calibration) ──
python scripts/training/analyze_predictions.py \
  "$OUT"/test_full.parquet "$OUT"/test_geom_only.parquet "$OUT"/test_geom_minus_shape.parquet \
  "$OUT"/test_geom_minus_normals.parquet "$OUT"/test_geom_minus_masks.parquet "$OUT"/test_em_only.parquet \
  --out-dir "$OUT/analysis" | tee "$OUT/analysis/report.md"
