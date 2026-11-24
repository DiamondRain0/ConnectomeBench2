# ConnectomeBench2

Data generation pipeline + training code for the NeurIPS submission on automated neural-circuit proofreading.

## Getting started

### 1. Clone

```bash
git clone https://github.com/timfarkas/ConnectomeBench2.git
cd ConnectomeBench2
```

### 2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create the environment

```bash
uv sync --extra train            # training / evaluation on the released dataset (GPU box)
uv sync --extra datagen          # data generation from CAVE + mesh rendering
uv sync --extra train --extra datagen   # both
source .venv/bin/activate
```

Python 3.11 is pinned (FM-stack / `pcg-skel` constraint). Training only needs the `train`
extra (torch, torchvision, scikit-learn, huggingface_hub, wandb); the rendering stack
(`pygfx`, `octarine3d`, CAVE clients) is only needed for `datagen`.

### 4. Set `PYTHONPATH`

Source code lives under `src/data_generation/`. Imports like `from connectome.operation_bank import ...` need that on the path:

```bash
export PYTHONPATH=src/data_generation:$PYTHONPATH
```

(Or add it to a per-project `.envrc` / shell init.)

### 5. CAVE auth + env vars

All cloud data access goes through [CAVEclient](https://github.com/CAVEconnectome/CAVEclient). Get a token via the [auth tutorial](https://caveconnectome.github.io/CAVEclient/tutorials/authentication/), then copy the example env file and fill it in:

```bash
cp env.sh.example env.sh
# edit env.sh and paste your token(s)
source env.sh
```

You may need two separate tokens — `CAVE_API_TOKEN` for MICrONS (mouse) + FlyWire (fly), and `CAVE_API_TOKEN_DAF` for H01 (human) + fish1 (zebrafish). If your account is provisioned for all four datastacks, the same value works for both.

`env.sh.example` also documents an optional `CACHE_DIR` (defaults to `./.cache`) for the mesh / EM diskcaches.

### 6. (Linux only) headless rendering deps

`pygfx` / `octarine3d` render via WGPU offscreen. On Linux you need system EGL libs:

```bash
sudo apt-get install -y libegl1 libgl1 libglx-mesa0 mesa-utils
```

macOS works out of the box.

## Usage

### Build an operation bank (mouse, full split)

```bash
python scripts/data_generation/build_operation_bank.py build \
    --species mouse \
    --target-count full \
    --output datasets/mouse/splits/train/operation_bank.jsonl
```

Replicate for `fly`, `human`, `zebrafish`. See `--help` for split filters (path length, ops/mm density, staleness).

### Render 5 mouse `endpoint_error_corr` samples

```bash
python -m training.training_data_renderer \
    --config configs/renderer_config_neurips_unified.yaml \
    --species mouse \
    --max-ops 5 \
    --input datasets/mouse/splits/train/operation_bank.jsonl \
    --output test_outputs/
```

Outputs a per-task parquet + `images/` (geometry `.npy` + EM cardinal/oblique `.png` slices).
The unified config renders both geometry flavours (`render_modes: [geometry, geometry_single]`);
add the split-mask labels for split edits with a second pass:

```bash
python -m training.training_data_renderer --config configs/renderer_config_neurips_unified.yaml     --species mouse --input ... --output test_outputs/ --split-mask-enrich
```

### Package into the ConnectomeBench2 v2 format

The packaging scripts produce exactly the schema of the released v2 dataset
(`jeffbbrown2/ConnectomeBench2@v2`), so new or re-generated samples can be mixed with it.

**1. Split table (once).** v2 assigns train/val/test per 80 µm cube. The authors' per-cube
hash is not public, so the cube→split table is recovered from the released rows; samples in
cubes covered by the release get the release's split, unseen cubes get a deterministic
75/12.5/12.5 hash:

```bash
python scripts/data_generation/neurips/build_v2_cube_splits.py     --dataset /data/ConnectomeBench2 --out configs/cube_splits_v2   # or --hf to stream metadata only
```

It prints a *purity* per species (fraction of released samples that agree with their cube's
split) and exits non-zero below 0.99. The grid origin comes from CloudVolume (needs `env.sh`)
or is inferred from the data (`--origin infer`).

**2. Combine → v2 semantics → HF shards:**

```bash
python scripts/data_generation/neurips/build_giga_parquet.py --out-root <render_root>
python scripts/data_generation/neurips/postprocess_giga_parquet.py --parquet <render_root>/combined.parquet
python scripts/data_generation/neurips/build_parquet_dataset.py --parquet <render_root>/combined.parquet --out-root <hf_root>
```

`build_giga_parquet.py` picks up every `<render_root>/<subdir>/questions.parquet`.
`postprocess_giga_parquet.py` maps archives to `unified_<sp>` / `unified_controls_<sp>`, derives
`sample_type`, picks the dual/single `geometry` per task, renames EM to `em_*_before`, keeps split
masks for split edits, sets nullable labels / `task_routing` / `has_after_mask`, and assigns the
v2 split (rows missing the needed geometry or any EM view are dropped, as in v2).
`build_parquet_dataset.py` writes the v2 columns, converting split masks to v2's
0 = A / 1 = B / 255 = background encoding.

Offline check of the whole chain (no CAVE, no rendering; compares against the released schema):

```bash
python scripts/data_generation/neurips/smoke_test_v2_packaging.py --reference /data/ConnectomeBench2/demo.parquet
```

`configs/cube_splits/` holds the old v1 splits (50 µm cubes, 80/10/10); they are not used for v2.

## Reproducing the paper (training on the released dataset)

All training code lives in `src/neurips_training/`; helper scripts in `scripts/training/`.
**Lab setup with a data machine + compute machines on shared storage: see [docs/LAB_SETUP.md](docs/LAB_SETUP.md).**
The loader auto-detects the dataset schema; the paper's numbers use the **v2** release
(HF branch `v2`, 716,485 rows, ~230 GB).

### 0. Smoke test (CPU, ~1-2 minutes, no GPU needed)

```bash
python scripts/training/smoke_test.py
```

Builds a fake train/val/test tree from the 20-row `demo.parquet` and runs training,
checkpoint + resume, test evaluation with exports, an ablation and the analysis script.

### 1. Download the dataset

```bash
python scripts/training/download_dataset.py --out /data/ConnectomeBench2          # full (~230 GB)
python scripts/training/download_dataset.py --out /data/cb2_small --max-shards 4  # small debug subset
python scripts/training/download_dataset.py --out /data/ConnectomeBench2 --splits test --verify
```

### 2. Train the multi-task ViT

Paper recipe (App. B.2) is the default: 20 epochs, AdamW, lr = 1e-4·sqrt(global_bs/32),
5 warmup epochs + cosine, wd 0.05 / drop-path 0.1 (ViT-B) or 0.1 / 0.25 (ViT-L), label
smoothing 0.1, bf16, sqrt species×task blend with fly/zebrafish edit weights 0.7/0.5 and
class balancing, all augmentations. `--batch-size` is the GLOBAL batch size.

```bash
# single GPU, ViT-B (paper: ~15 H100-hours)
python src/neurips_training/training.py \
    --blend-config src/neurips_training/configs/full_4sp.yaml \
    --parquet-root /data/ConnectomeBench2 --batch-size 256

# multi-GPU, ViT-L (paper: 3xH100, bs 768)
torchrun --standalone --nproc-per-node=3 src/neurips_training/training.py \
    --blend-config src/neurips_training/configs/full_4sp.yaml \
    --parquet-root /data/ConnectomeBench2 --model vit_l_16 --batch-size 768

# resume after an interruption
python src/neurips_training/training.py ... --resume checkpoints_neurips/neurips_run_last.pt
```

Checkpoints: `<name>_best.pt` (best val headline = mean of split-error bAcc, merge-error
bAcc, mask mIoU) and `<name>_last.pt` (with optimizer state). Per-epoch metrics are appended
to `<name>_metrics.jsonl`; `--wandb` is optional.

### 3. Evaluate on the held-out test set and build the tables

```bash
python src/neurips_training/training.py --mode evaluate \
    --blend-config src/neurips_training/configs/full_4sp.yaml --parquet-root /data/ConnectomeBench2 \
    --checkpoint checkpoints_neurips/neurips_run_best.pt --eval-split test \
    --export-predictions results/test_full.parquet

# Table 1 inference-time ablations
...  --modalities geom                      # "ViT-B, geom only"
...  --modalities geom --ablate shape       # "-shape"   (silhouette + depth zeroed)
...  --modalities geom --ablate normals     # "-normals"
...  --modalities geom --ablate masks       # "-masks"
...  --modalities em                        # "ViT-B, EM only"

python scripts/training/analyze_predictions.py results/test_*.parquet --out-dir results/analysis
```

`analyze_predictions.py` prints Table 1 / 6 / 7 / 8-style results (per-sample bAcc at 0.5
with 95% op-clustered bootstrap CIs, mIoU, per-species TPR/TNR, high-confidence coverage)
and calibration (15-bin adaptive ECE, normalised sharpness, ROC-AUC, 12-bin reliability CSVs).

Everything above in one go: `DATA=/data/ConnectomeBench2 NGPU=1 bash scripts/training/reproduce_paper.sh`.

### 4. Other experiments

* **Data scaling (Fig. 2b):** `--max-samples N --data-subsample-seed S --seed S` (N = 100 … all).
* **Leave-one-species-out (Fig. 4c):** train with `--include-species mouse,fly,human`, evaluate with
  `--eval-species zebrafish`; export CLS features with `--export-features` (eval on `--eval-split train`
  for the in-distribution fit) and run `scripts/training/mahalanobis_ood.py`.
* **Random / foundation-model init (Fig. 5a):** `--init-source none` / `--init-source fm_checkpoint --fm-checkpoint ...`.

### Changes relative to the original code release

The released training code targeted the **v1** dataset (`v1-may06`) and could not load v2.
Fixed / added:

* **v2 schema support** (auto-detected): single `geometry` column whose dual/single semantics
  follow `sample_type`, `em_*_before` columns, split-mask GT from `split_mask_{front,side,top}`
  (0 = piece A, 1 = piece B, 255 = background), nullable labels, `has_after_mask`.
* **Leakage guard** from the dataset card: `em_best` is dropped for merge-error identification
  (`--keep-em-best-for-single` restores it); EM masks are fused for single-segment rows.
* **Per-op aggregation key** is `combined_sample_hash` (unique); `source_archive_sample_hash`
  is not unique across archives and could merge different samples.
* **Sampling = paper:** each epoch draws N rows with replacement according to the blend
  weights (streaming Poisson equivalent of `WeightedRandomSampler`). The old accept/reject
  scheme produced ~10x smaller epochs (`--epoch-size-mode accept` keeps it).
* **Epoch counter shared with persistent DataLoader workers** (previously every epoch reused
  epoch-0 shard order and sample selection).
* **Item shuffle buffer** so batches are not clumps of views of the same operation.
* **Evaluation uses the full split** (previously val was randomly subsampled by the class-balance
  accept rolls) and **rank-0 eval sees all shards** under DDP (previously only its 1/world share).
* **`--eval-split test`**, per-species metrics, TPR/TNR, ROC-AUC, adaptive ECE, sharpness,
  per-view prediction/feature export, inference-time ablations, analysis + Mahalanobis scripts.
* **DDP:** per-rank epoch length (scheduler length was off by world_size) and synchronized
  stopping (ranks with fewer batches previously deadlocked the others).
* **Hyper-parameters = paper App. B.2** (lr reference bs 32, ViT-L wd 0.1 / drop-path 0.25,
  20 epochs, per-archive blend weights); CLI values are no longer silently overwritten.
* **Mask-decoder view embedding** (App. B.2) is now applied and trained (`view_emb` existed but
  was unused and outside every optimizer group). `--no-view-embedding` restores old behaviour.
* Bilinear (input) / nearest (mask) interpolation for spatial augmentation; resume support;
  atomic checkpoint writes; dependency fixes (`torch`, `torchvision`, `scikit-learn`,
  `huggingface_hub` were missing; the unrelated `torchrun` PyPI package was removed).

Not included in this repository (described in the paper but not released): the
self-supervised foundation-model pretraining (App. B.4), the active-learning experiments
(App. B.5), mask → supervoxel multicut evaluation (App. B.3, needs `pychunkedgraph`),
64-view test-time view scaling (needs re-rendering), and occlusion-sensitivity maps.

## License

- **Code** is released under the [MIT License](LICENSE).
- **Data** (the [`jeffbbrown2/ConnectomeBench2`](https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2) HuggingFace dataset built by this pipeline) is released under the licenses of the respective upstream connectomic sources — see the dataset card and the [Citation](#citation) section below.

## Citation

If you use ConnectomeBench2, please cite:

```
Brown, J., Farkas, T., Razgar, G., Boyden, E. S.
ConnectomeBench2: A unified benchmark for automated connectomic proofreading.
(2026, in submission). Brown J. and Farkas T. contributed equally as first authors.
```

Please also cite the upstream connectome sources used by this dataset:

- **MICrONS** (mouse cortex): <https://www.microns-explorer.org/cortical-mm3>
- **FlyWire** (Drosophila): <https://flywire.ai/>
- **H01** (human cortex): <https://h01-release.storage.googleapis.com/landing.html>
- **Zebrafish** (fish1): <https://fish1-release.storage.googleapis.com/index.html>
