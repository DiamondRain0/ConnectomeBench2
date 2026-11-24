"""data.py — NeurIPS streamlined training data layer.

Loads ConnectomeBench2 from HuggingFace-format parquet shards
(`jeffbbrown2/ConnectomeBench2` on the HF Hub, or the output of
`scripts/data_generation/neurips/build_parquet_dataset.py`).

Two on-disk schemas are supported and auto-detected from the first shard:

  v2 (June 2026, HF branch `v2`, 716,485 rows) — what the paper reports on
    - `geometry`  : npz (3, 7, 224, 224) f16. DUAL-segment for merge_edit /
                    adjacent_control / synapse_control, SINGLE-segment for
                    split_edit / junction_control.
    - `em_{xy,xz,yz,best}_before` : HF Image struct, (224, 224, 3) uint8 PNG
                    (ch0 = EM, ch1 = seg A, ch2 = seg B).
    - `split_mask_{front,side,top}` : HF Image struct, (224, 224) uint8 PNG,
                    per-view post-split GT for split_edit rows. Encoding:
                    0 = piece A, 1 = piece B, 255 = background.
    - flags: `has_em`, `has_after_mask`; labels are NULL for rows the task does
      not apply to.

  v1 (`v1-may06` tag, 401,170 rows) — what the original release code targeted
    - `geometry` (dual) + `geometry_single`; `em_{xy,xz,yz,best}`;
      `has_single_mask` / `has_dual_mask`; mask GT = dual geometry ch5/6.

Loading model
-------------
`IterableDataset`. The full dataset is ~230 GB, so image bytes are never held
in memory; each (rank, worker) consumer streams its own subset of shards.

Per epoch (training mode):
  1. Shuffle shard order (deterministic on (seed, epoch)).
  2. Partition shards over all (rank, worker) consumers.
  3. Stream record batches; per row, draw how many times it is used this epoch
     from Poisson(lambda_row), where lambda_row = N * w_row / sum(w). This is
     the streaming equivalent of the paper's per-epoch
     `WeightedRandomSampler(weights, num_samples=N, replacement=True)`.
     (`epoch_size_mode="accept"` restores the original release behaviour:
     accept each row once with probability w_row / max(w), which yields much
     smaller epochs.)
  4. Every accepted row yields one item per view (3 geom + 3–4 EM). Items pass
     through a shuffle buffer so a batch is not made of clumps of views of the
     same operation.

Evaluation mode (`sample_all=True`): every surviving row is yielded exactly
once, no shuffling, no rank partitioning — so val/test metrics are computed on
the full held-out split, as in the paper.

Routing semantics:
  task_routing                         head       cls_label                          mask GT
  -----------------------------------  ---------  ---------------------------------  -------
  false_split_correction               endpoint   false_split_correction_label       no
  false_merge_identification (+ split_ junction   false_merge_identification_label   yes iff
    mask_generation)                                                                  split_mask_generation

Leakage guards (from the dataset card):
  - junction-routed rows use the single-segment geometry; mask channels are
    zeroed in the input.
  - junction-routed EM: seg A/B are fused into A, B zeroed, and `em_best` is
    dropped by default (it is chosen using both labels and can leak GT).
"""
from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

# ───────────────────────────── Constants ──────────────────────────────────

IMAGE_SIZE = 224
NUM_GEOM_VIEWS = 3
NUM_GEOM_CHANNELS = 7

# Geometry channel layout (must match renderer + model.py)
CH_SILH, CH_DEPTH, CH_NX, CH_NY, CH_NZ, CH_MASK_A, CH_MASK_B = range(7)

# EM PNG channel layout (R/G/B)
CH_EM_GREY, CH_EM_MASK_A, CH_EM_MASK_B = 0, 1, 2

# Task IDs
TASK_ENDPOINT = 0   # split-error correction (should these two segments merge?)
TASK_JUNCTION = 1   # merge-error identification (does this segment contain a false merge?)

# Modality IDs
MODALITY_GEOM = 0
MODALITY_EM = 1

# EM view names (index in this tuple == view_idx for EM items)
EM_VIEW_NAMES = ("xy", "xz", "yz", "best")
EM_COLUMNS_V1 = tuple(f"em_{v}" for v in EM_VIEW_NAMES)
EM_COLUMNS_V2 = tuple(f"em_{v}_before" for v in EM_VIEW_NAMES)
EM_VIEWS = EM_COLUMNS_V1  # backwards-compatible export
EM_BEST_IDX = 3

GEOM_VIEW_NAMES = ("front", "side", "top")
SPLIT_MASK_COLUMNS = tuple(f"split_mask_{v}" for v in GEOM_VIEW_NAMES)

# v2 split-mask PNG encoding
SPLIT_MASK_PIECE_A = 0
SPLIT_MASK_PIECE_B = 1

# Sample type → kind (legacy 3-cell axis: edits / junction / synapse)
SAMPLE_TYPE_TO_KIND: dict[str, str] = {
    "merge_edit":        "edits",
    "split_edit":        "edits",
    "adjacent_control":  "edits",
    "junction_control":  "junction",
    "synapse_control":   "synapse",
}

# HF readme uses "validation" for the val split; the shard dir is "val".
SPLIT_ALIASES: dict[str, str] = {
    "train":      "train",
    "val":        "val",
    "validation": "val",
    "test":       "test",
}

VALID_BLEND_STRATEGIES = (
    "natural", "uniform_species", "uniform_type", "uniform_cell", "sqrt", "custom",
)
VALID_EPOCH_SIZE_MODES = ("full", "accept")

# Inference-time channel ablations (Table 1 of the paper).
#   shape   → silhouette + depth (geometry)
#   normals → nx, ny, nz (geometry)
#   masks   → geometry mask A/B and EM seg A/B
VALID_ABLATIONS = ("shape", "normals", "masks", "silhouette", "depth")

_METADATA_COLUMNS_COMMON: tuple[str, ...] = (
    "combined_sample_hash",
    "source_archive_sample_hash",
    "source_archive",
    "sample_type",
    "same_neuron",
    "species",
    "has_em",
    "task_routing",
    "false_split_correction_label",
    "false_merge_identification_label",
)
METADATA_COLUMNS_V1 = _METADATA_COLUMNS_COMMON + ("has_single_mask", "has_dual_mask")
METADATA_COLUMNS_V2 = _METADATA_COLUMNS_COMMON + ("has_after_mask",)
METADATA_COLUMNS = METADATA_COLUMNS_V1  # backwards-compatible export

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def detect_schema(shard: str | Path) -> str:
    """Return "v2" or "v1" from a shard's column names."""
    names = set(pq.read_schema(str(shard)).names)
    if "em_xy_before" in names or "has_after_mask" in names:
        return "v2"
    if "geometry_single" in names or "em_xy" in names:
        return "v1"
    raise ValueError(f"cannot detect ConnectomeBench2 schema version from {shard}: {sorted(names)}")


# ───────────────────────────── Blend config ───────────────────────────────

@dataclass
class BlendConfig:
    """Sampling blend across (species, kind) cells.

    Strategies:
      - natural          per-item uniform
      - uniform_species  equalise mass per species
      - uniform_type     equalise mass per kind
      - uniform_cell     equalise mass per (species, kind) cell
      - sqrt             cell mass ∝ sqrt(N_cell), uniform within cell (paper default)
      - custom           start from natural, expect manual `weights` overrides

    `weights` are multiplicative per-cell overrides keyed by (species, cell);
    the paper uses fly edits 0.7 and zebrafish edits 0.5 (see full_4sp.yaml).
    """
    strategy: str = "sqrt"
    cell_axis: str = "kind"  # "kind" (3-cell legacy) | "sample_type" (5-cell)
    weights: dict[tuple[str, str], float] = field(default_factory=dict)
    include_species: list[str] | None = None
    include_sample_types: list[str] | None = None
    class_balance: bool = True


def load_blend_config(yaml_path: str | Path) -> BlendConfig:
    """Parse a training yaml (parquet / filter / blend / class_balance)."""
    import yaml
    raw = yaml.safe_load(Path(yaml_path).read_text())
    blend_raw = raw.get("blend", {}) or {}
    filt_raw = raw.get("filter", {}) or {}

    strategy = blend_raw.get("strategy", "sqrt")
    if strategy not in VALID_BLEND_STRATEGIES:
        raise ValueError(f"unknown blend strategy: {strategy!r}")
    cell_axis = blend_raw.get("cell_axis", "kind")
    if cell_axis not in ("kind", "sample_type"):
        raise ValueError(f"unknown cell_axis: {cell_axis!r}")

    weights: dict[tuple[str, str], float] = {}
    for w in blend_raw.get("weights", []) or []:
        weights[(w["species"], w["cell"])] = float(w["weight"])

    return BlendConfig(
        strategy=strategy,
        cell_axis=cell_axis,
        weights=weights,
        include_species=filt_raw.get("include_species"),
        include_sample_types=filt_raw.get("include_sample_types"),
        class_balance=bool(raw.get("class_balance", True)),
    )


# ──────────────────────── Routing (column lookups) ────────────────────────

def _is_null(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(isinstance(v, float) and math.isnan(v))
    except TypeError:
        return False


def route_row(row: Any) -> tuple[int, int, bool] | None:
    """Map a parquet row (dict or pd.Series) → (task_id, cls_label, has_mask_gt) or None.

      - false_merge_identification (split_edit / junction_control)
            → TASK_JUNCTION, label = false_merge_identification_label
            has_mask_gt = "split_mask_generation" in routing
      - false_split_correction (merge_edit / synapse_control / adjacent_control)
            → TASK_ENDPOINT, label = false_split_correction_label
      - otherwise, or if the label is null → None (row is dropped)
    """
    routing = row["task_routing"]
    routing = list(routing) if routing is not None else []
    if "false_merge_identification" in routing:
        lab = row["false_merge_identification_label"]
        if _is_null(lab):
            return None
        return (TASK_JUNCTION, int(lab), "split_mask_generation" in routing)
    if "false_split_correction" in routing:
        lab = row["false_split_correction_label"]
        if _is_null(lab):
            return None
        return (TASK_ENDPOINT, int(lab), False)
    return None


def synapse_label_from_row(row: Any) -> tuple[bool, int]:
    """(has_synapse_label, synapse_label) for the auxiliary synapse head.

      synapse_control                     → (True, 1)   # real synapse between two neurons
      merge_edit (same neuron)            → (True, 0)   # pair is one neuron, not a synapse
      else                                → (False, 0)  # undefined; gated out of the loss
    """
    st = row["sample_type"]
    if st == "synapse_control":
        return (True, 1)
    if st == "merge_edit" and bool(row["same_neuron"]):
        return (True, 0)
    return (False, 0)


# ───────────────────────────── Dataset ────────────────────────────────────

class NeurIPSDataset(IterableDataset):
    """Streaming dataset over HF parquet shards. One yielded item = one (row, view).

    Args:
        parquet_root: directory containing `{split}/{split}-NNNNN.parquet`.
        split: "train" | "val" | "validation" | "test".
        blend: include filters + per-cell sampling weights + class balance.
        modalities: subset of ("geom", "em").
        augment: enable CPU-side training augmentations.
        max_samples: cap on distinct rows (applied before view expansion).
        data_subsample_seed: if set, the max_samples subset is random (seeded),
            otherwise the first rows are used.
        seed: RNG seed for shard order / sampling / shuffling.
        sample_all: evaluation mode — yield every surviving row exactly once
            (no weighting, no shuffle, no DDP partitioning).
        epoch_size_mode: "full" (paper: N draws with replacement per epoch) or
            "accept" (original release: each row at most once, p = w / max w).
        shuffle_buffer: item-level shuffle buffer size (training only; 0 = off).
        drop_em_best_for_single: drop `em_best` views for junction-routed rows.
        ablate: channels zeroed at load time (see VALID_ABLATIONS).
        partition_by_rank: split shards across DDP ranks (training). Must be
            False for rank-0-only evaluation.
    """

    def __init__(
        self,
        parquet_root: str | Path,
        split: str,
        blend: BlendConfig | None = None,
        modalities: tuple[str, ...] = ("geom", "em"),
        augment: bool = False,
        max_samples: int | None = None,
        data_subsample_seed: int | None = None,
        seed: int = 42,
        shuffle_within_shard: bool = True,
        sample_all: bool = False,
        epoch_size_mode: str = "full",
        shuffle_buffer: int = 0,
        drop_em_best_for_single: bool = True,
        ablate: Iterable[str] = (),
        partition_by_rank: bool = True,
        read_batch_rows: int = 32,
    ):
        self.parquet_root = Path(parquet_root)
        canon = SPLIT_ALIASES.get(split, split)
        self.split = canon
        self.blend = blend or BlendConfig()
        self.modalities = tuple(modalities)
        bad_mod = set(self.modalities) - {"geom", "em"}
        if bad_mod or not self.modalities:
            raise ValueError(f"modalities must be a non-empty subset of (geom, em); got {self.modalities}")
        self.augment = bool(augment)
        self.seed = int(seed)
        self.shuffle_within_shard = bool(shuffle_within_shard) and not sample_all
        self.sample_all = bool(sample_all)
        if epoch_size_mode not in VALID_EPOCH_SIZE_MODES:
            raise ValueError(f"epoch_size_mode must be one of {VALID_EPOCH_SIZE_MODES}")
        self.epoch_size_mode = epoch_size_mode
        self.shuffle_buffer = 0 if sample_all else max(0, int(shuffle_buffer))
        self.drop_em_best_for_single = bool(drop_em_best_for_single)
        self.ablate = tuple(ablate or ())
        bad = set(self.ablate) - set(VALID_ABLATIONS)
        if bad:
            raise ValueError(f"unknown ablation(s) {sorted(bad)}; valid: {VALID_ABLATIONS}")
        self.partition_by_rank = bool(partition_by_rank) and not sample_all
        self.read_batch_rows = int(read_batch_rows)

        # Epoch counter in shared memory so persistent DataLoader workers see
        # `ds.epoch = k` set from the main process (a plain attribute would be
        # frozen in each worker's copy → identical sampling every epoch).
        self._epoch_t = torch.zeros(1, dtype=torch.int64).share_memory_()

        # ── discover shards ──
        shard_dir = self.parquet_root / canon
        self.shards: list[Path] = sorted(shard_dir.glob(f"{canon}-*.parquet"))
        if not self.shards:
            raise FileNotFoundError(
                f"no shards at {shard_dir}/{canon}-*.parquet "
                f"(known split aliases: {sorted(SPLIT_ALIASES)})"
            )
        self.schema = detect_schema(self.shards[0])
        if self.schema == "v2":
            self.em_columns = EM_COLUMNS_V2
            meta_cols = METADATA_COLUMNS_V2
        else:
            self.em_columns = EM_COLUMNS_V1
            meta_cols = METADATA_COLUMNS_V1

        # ── cheap metadata-only scan (no image bytes) ──
        meta_df = self._read_metadata(list(meta_cols))
        meta_df = self._annotate(meta_df)
        meta_df = self._filter_df(meta_df, self.blend).reset_index(drop=True)

        # ── optional row-level subsample (BEFORE view expansion) ──
        if max_samples is not None and len(meta_df) > max_samples:
            if data_subsample_seed is not None:
                rng = np.random.default_rng(int(data_subsample_seed))
                idx = np.sort(rng.permutation(len(meta_df))[:max_samples])
            else:
                idx = np.arange(max_samples)
            meta_df = meta_df.iloc[idx].reset_index(drop=True)

        hashes = meta_df["combined_sample_hash"].astype(str).to_numpy()
        n_items = self._row_n_items(meta_df).astype(np.int64)

        # ── per-row expected draws per epoch (or accept probability) ──
        if self.sample_all or len(meta_df) == 0:
            rate = np.ones(len(meta_df), dtype=np.float64)
        else:
            kinds = (meta_df["sample_type"] if self.blend.cell_axis == "sample_type"
                     else meta_df["_kind"]).astype(str).to_numpy()
            w = compute_row_weights(
                species=meta_df["species"].astype(str).to_numpy(),
                kinds=kinds,
                task_ids=meta_df["_task_id"].to_numpy(),
                cls_labels=meta_df["_cls_label"].to_numpy(),
                strategy=self.blend.strategy,
                cell_overrides=self.blend.weights,
                class_balance=self.blend.class_balance,
            )
            if self.epoch_size_mode == "full":
                tot = float(w.sum())
                rate = (w * len(w) / tot) if tot > 0 else np.zeros_like(w)
            else:
                mx = float(w.max()) if len(w) else 0.0
                rate = (w / mx) if mx > 0 else np.zeros_like(w)

        self._row_rate: dict[str, float] = dict(zip(hashes.tolist(), rate.tolist()))
        self._row_n_items_map: dict[str, int] = dict(zip(hashes.tolist(), n_items.tolist()))
        self._surviving_hashes: frozenset[str] = frozenset(hashes.tolist())
        self._approx_len_global = int(round(float((rate * n_items).sum())))
        self.n_rows = int(len(meta_df))
        self._meta_df_for_introspection = meta_df

    # ─── metadata ─────────────────────────────────────────────────────────

    def _read_metadata(self, cols: list[str]) -> pd.DataFrame:
        """Row metadata for all shards of this split.

        Fast path: the release's `metadata/<split>.parquet` sidecar (a few MB,
        one file) — important on network storage, where touching every shard
        footer from every DDP rank is slow. Rows are restricted to shards that
        exist locally, so partial downloads (--max-shards) work. Falls back to
        scanning the shards themselves.
        """
        sidecar = self.parquet_root / "metadata" / f"{self.split}.parquet"
        if sidecar.exists():
            try:
                names = set(pq.read_schema(str(sidecar)).names)
                if set(cols) <= names and "shard" in names:
                    df = pq.read_table(str(sidecar), columns=[*cols, "shard"]).to_pandas()
                    have = {f"{self.split}/{s.name}" for s in self.shards}
                    shard_col = df["shard"].astype(str).str.replace("\\", "/", regex=False)
                    df = df[shard_col.isin(have)].drop(columns=["shard"])
                    if len(df):
                        return df.reset_index(drop=True)
            except Exception as e:  # corrupt / foreign sidecar → slow path
                print(f"[data] sidecar {sidecar} unusable ({e}); scanning shards", flush=True)
        return pq.read_table([str(s) for s in self.shards], columns=cols).to_pandas()

    # ─── epoch (shared across workers) ────────────────────────────────────

    @property
    def epoch(self) -> int:
        return int(self._epoch_t.item())

    @epoch.setter
    def epoch(self, ep: int) -> None:
        self._epoch_t.fill_(int(ep))

    def set_epoch(self, ep: int) -> None:
        self.epoch = ep

    # ─── metadata annotation / filtering / weighting ──────────────────────

    def _annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add `_task_id`, `_cls_label`, `_has_mask_gt`, `_kind` (vectorised routing)."""
        routing = df["task_routing"].map(lambda r: tuple(r) if r is not None else ())
        is_jn = routing.map(lambda r: "false_merge_identification" in r).to_numpy(bool)
        is_ep = routing.map(lambda r: "false_split_correction" in r).to_numpy(bool) & ~is_jn
        has_mask_route = routing.map(lambda r: "split_mask_generation" in r).to_numpy(bool)

        jn_lab = pd.to_numeric(df["false_merge_identification_label"], errors="coerce").to_numpy(float)
        ep_lab = pd.to_numeric(df["false_split_correction_label"], errors="coerce").to_numpy(float)
        label = np.where(is_jn, jn_lab, np.where(is_ep, ep_lab, np.nan))
        task = np.where(is_jn, TASK_JUNCTION, np.where(is_ep, TASK_ENDPOINT, -1))
        task = np.where(np.isnan(label), -1, task)

        if self.schema == "v2":
            has_after = df["has_after_mask"].fillna(False).to_numpy(bool)
            has_mask = is_jn & has_mask_route & has_after
        else:
            has_dual = df["has_dual_mask"].fillna(False).to_numpy(bool)
            has_mask = is_jn & has_mask_route & has_dual

        df = df.copy()
        df["_task_id"] = task.astype(np.int64)
        df["_cls_label"] = np.nan_to_num(label, nan=0.0).astype(np.int64)
        df["_has_mask_gt"] = has_mask
        df["_kind"] = df["sample_type"].map(SAMPLE_TYPE_TO_KIND).fillna("other")
        return df

    @staticmethod
    def _filter_df(df: pd.DataFrame, blend: BlendConfig) -> pd.DataFrame:
        mask = df["_task_id"].to_numpy() >= 0
        if blend.include_species is not None:
            mask &= df["species"].isin(blend.include_species).to_numpy()
        if blend.include_sample_types is not None:
            mask &= df["sample_type"].isin(blend.include_sample_types).to_numpy()
        return df.loc[mask]

    def _row_n_items(self, df: pd.DataFrame) -> np.ndarray:
        """How many per-view items each row yields."""
        n = np.zeros(len(df), dtype=np.int64)
        is_jn = df["_task_id"].to_numpy() == TASK_JUNCTION
        if "geom" in self.modalities:
            if self.schema == "v2":
                n += NUM_GEOM_VIEWS  # geometry always present in v2
            else:
                single = df["has_single_mask"].fillna(False).to_numpy(bool)
                dual = df["has_dual_mask"].fillna(False).to_numpy(bool)
                n += np.where(np.where(is_jn, single, dual), NUM_GEOM_VIEWS, 0)
        if "em" in self.modalities:
            has_em = df["has_em"].fillna(False).to_numpy(bool)
            n_em = np.where(is_jn & self.drop_em_best_for_single, len(EM_VIEW_NAMES) - 1, len(EM_VIEW_NAMES))
            n += np.where(has_em, n_em, 0)
        return n

    # ─── IterableDataset interface ────────────────────────────────────────

    def _world(self) -> tuple[int, int]:
        if self.partition_by_rank and dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
        return 1, 0

    def __len__(self) -> int:
        """Approximate # items per epoch *for this rank* (sampling adds noise)."""
        world_size, _ = self._world()
        return max(1, self._approx_len_global // world_size)

    def _read_columns(self) -> list[str]:
        cols = ["combined_sample_hash", "sample_type", "same_neuron", "species",
                "task_routing", "false_split_correction_label",
                "false_merge_identification_label"]
        if "geom" in self.modalities:
            cols.append("geometry")
            if self.schema == "v1":
                cols += ["geometry_single", "has_single_mask", "has_dual_mask"]
            else:
                cols += ["has_after_mask", *SPLIT_MASK_COLUMNS]
        if "em" in self.modalities:
            cols += ["has_em", *self.em_columns]
        return cols

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = get_worker_info()
        if worker_info is None:
            wid, nworkers = 0, 1
        else:
            wid, nworkers = int(worker_info.id), int(worker_info.num_workers)
        world_size, rank = self._world()
        epoch = self.epoch

        shards = list(self.shards)
        if not self.sample_all:
            random.Random(_mix_seed(self.seed, epoch, "shards")).shuffle(shards)
        consumer_idx = rank * nworkers + wid
        n_consumers = world_size * nworkers
        my_shards = shards[consumer_idx::n_consumers]

        rng = random.Random(_mix_seed(self.seed, epoch, rank, wid, "iter"))
        np_rng = np.random.default_rng(_mix_seed(self.seed, epoch, rank, wid, "draws"))

        items = self._iter_items(my_shards, rng, np_rng)
        if self.shuffle_buffer > 0:
            items = _shuffle_buffer(items, self.shuffle_buffer, rng)
        yield from items

    def _iter_items(self, shards: list[Path], rng: random.Random,
                    np_rng: np.random.Generator) -> Iterator[dict[str, Any]]:
        cols = self._read_columns()
        for shard_path in shards:
            pf = pq.ParquetFile(str(shard_path))
            avail = set(pf.schema_arrow.names)
            read_cols = [c for c in cols if c in avail]
            for rb in pf.iter_batches(batch_size=self.read_batch_rows, columns=read_cols):
                rows = rb.to_pylist()
                if self.shuffle_within_shard:
                    rng.shuffle(rows)
                for row in rows:
                    h = str(row["combined_sample_hash"])
                    if h not in self._surviving_hashes:
                        continue
                    n_draw = self._n_draws(h, rng, np_rng)
                    for _ in range(n_draw):
                        yield from self._materialize_row(row)

    def _n_draws(self, h: str, rng: random.Random, np_rng: np.random.Generator) -> int:
        if self.sample_all:
            return 1
        r = self._row_rate.get(h, 0.0)
        if r <= 0.0:
            return 0
        if self.epoch_size_mode == "full":
            return int(np_rng.poisson(r))
        return 1 if (r >= 1.0 or rng.random() < r) else 0

    # ─── per-row materialisation ──────────────────────────────────────────

    def _materialize_row(self, row: dict[str, Any]) -> Iterator[dict[str, Any]]:
        routed = route_row(row)
        if routed is None:
            return
        task_id, cls_label, has_mask_route = routed
        has_syn, syn_label = synapse_label_from_row(row)
        sample_type = row["sample_type"]
        common = dict(
            task_id=task_id, cls_label=cls_label,
            has_synapse_label=has_syn, synapse_label=syn_label,
            op_id=str(row["combined_sample_hash"]),
            species=row["species"], sample_type=sample_type,
            kind=SAMPLE_TYPE_TO_KIND.get(sample_type, "other"),
        )

        if "geom" in self.modalities:
            yield from self._geom_items(row, task_id, has_mask_route, common)

        if "em" in self.modalities and bool(row.get("has_em", False)):
            for vi, col in enumerate(self.em_columns):
                if task_id == TASK_JUNCTION and vi == EM_BEST_IDX and self.drop_em_best_for_single:
                    continue
                blob = row.get(col)
                if not _blob_present(blob):
                    continue
                yield self._build_em_sample(blob, vi, task_id, common)

    def _geom_items(self, row: dict[str, Any], task_id: int, has_mask_route: bool,
                    common: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if self.schema == "v2":
            src_blob = row.get("geometry")
            if src_blob is None:
                return
            src = _decode_npz(src_blob)
            gt_views: list[np.ndarray | None] = [None] * NUM_GEOM_VIEWS
            if has_mask_route and bool(row.get("has_after_mask", False)):
                blobs = [row.get(c) for c in SPLIT_MASK_COLUMNS]
                if all(_blob_present(b) for b in blobs):
                    gt_views = [_split_mask_png_to_target(b) for b in blobs]
        else:
            if task_id == TASK_JUNCTION:
                if not bool(row.get("has_single_mask", False)):
                    return
                src = _decode_npz(row["geometry_single"])
            else:
                if not bool(row.get("has_dual_mask", False)):
                    return
                src = _decode_npz(row["geometry"])
            gt_views = [None] * NUM_GEOM_VIEWS
            if has_mask_route and bool(row.get("has_dual_mask", False)):
                dual = _decode_npz(row["geometry"])
                gt_views = [_dual_geometry_to_target(dual[v]) for v in range(NUM_GEOM_VIEWS)]

        for v in range(NUM_GEOM_VIEWS):
            yield self._build_geom_sample(src[v], gt_views[v], v, task_id, common)

    def _build_geom_sample(self, view_arr: np.ndarray, mask_gt: np.ndarray | None,
                           view_idx: int, task_id: int, common: dict[str, Any]) -> dict[str, Any]:
        view = view_arr.astype(np.float32, copy=True)  # (7, H, W)
        if task_id == TASK_JUNCTION:
            # Leakage guard (as in the original release): the single-segment
            # input carries no mask channels (mask A == silhouette anyway).
            view[CH_MASK_A] = 0.0
            view[CH_MASK_B] = 0.0
        if self.ablate:
            if "shape" in self.ablate or "silhouette" in self.ablate:
                view[CH_SILH] = 0.0
            if "shape" in self.ablate or "depth" in self.ablate:
                view[CH_DEPTH] = 0.0
            if "normals" in self.ablate:
                view[CH_NX:CH_NZ + 1] = 0.0
            if "masks" in self.ablate:
                view[CH_MASK_A] = 0.0
                view[CH_MASK_B] = 0.0
        x = torch.from_numpy(view)
        H, W = x.shape[-2:]
        has_mask_gt = mask_gt is not None
        mask_gt_t = (torch.from_numpy(mask_gt.astype(np.int64)) if has_mask_gt
                     else torch.zeros((H, W), dtype=torch.int64))
        if self.augment:
            x, mask_gt_t = _train_aug(x, mask_gt_t, MODALITY_GEOM)
        return {
            "input": x, "mask_label": mask_gt_t, "modality": MODALITY_GEOM,
            "has_mask_gt": has_mask_gt, "view_idx": view_idx, **common,
        }

    def _build_em_sample(self, blob: Any, view_idx: int, task_id: int,
                         common: dict[str, Any]) -> dict[str, Any]:
        png = _hf_image_bytes(blob)
        img = Image.open(io.BytesIO(png)).convert("RGB")
        rgb = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)
        rgb = np.ascontiguousarray(rgb)

        if task_id == TASK_JUNCTION:
            # Single-segment task: fuse A|B into A, zero B (dataset-card guidance).
            rgb[CH_EM_MASK_A] = np.maximum(rgb[CH_EM_MASK_A], rgb[CH_EM_MASK_B])
            rgb[CH_EM_MASK_B] = 0.0
        if "masks" in self.ablate:
            rgb[CH_EM_MASK_A] = 0.0
            rgb[CH_EM_MASK_B] = 0.0

        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        H, W = rgb.shape[-2:]
        x_arr = np.zeros((NUM_GEOM_CHANNELS, H, W), dtype=np.float32)
        x_arr[:3] = rgb
        x = torch.from_numpy(x_arr)
        mask_gt_t = torch.zeros((H, W), dtype=torch.int64)
        if self.augment:
            x, mask_gt_t = _train_aug(x, mask_gt_t, MODALITY_EM)
        return {
            "input": x, "mask_label": mask_gt_t, "modality": MODALITY_EM,
            "has_mask_gt": False, "view_idx": view_idx, **common,
        }


# ───────────────────────────── Helpers ────────────────────────────────────

def _mix_seed(*parts: Any) -> int:
    """Stable hash → int (Python's `hash()` is randomised between processes)."""
    import hashlib
    s = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "big")


def _shuffle_buffer(items: Iterator[dict[str, Any]], size: int,
                    rng: random.Random) -> Iterator[dict[str, Any]]:
    """Reservoir-style shuffle buffer. Inputs are held as float16 to bound RAM
    (~0.7 MB per item); they are converted back to float32 on emit."""
    buf: list[dict[str, Any]] = []
    for it in items:
        it["input"] = it["input"].to(torch.float16)
        if len(buf) < size:
            buf.append(it)
            continue
        j = rng.randrange(size)
        out, buf[j] = buf[j], it
        out["input"] = out["input"].to(torch.float32)
        yield out
    rng.shuffle(buf)
    for out in buf:
        out["input"] = out["input"].to(torch.float32)
        yield out


def _decode_npz(blob: Any) -> np.ndarray:
    """Decode a geometry npz blob (bytes / HF struct / path) → ndarray (`arr_0`)."""
    if blob is None:
        raise ValueError("npz blob is None")
    if isinstance(blob, dict):
        blob = blob.get("bytes") if blob.get("bytes") is not None else blob.get("path")
    if isinstance(blob, (str, Path)):
        return np.load(str(blob))["arr_0"]
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return np.load(io.BytesIO(bytes(blob)))["arr_0"]
    raise TypeError(f"unrecognised npz blob type: {type(blob).__name__}")


def _hf_image_bytes(blob: Any) -> bytes:
    """Raw PNG bytes from an HF Image cell ({bytes, path} struct or raw bytes)."""
    if blob is None:
        raise ValueError("image blob is None")
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return bytes(blob)
    if isinstance(blob, dict):
        if blob.get("bytes") is not None:
            return bytes(blob["bytes"])
        if blob.get("path") is not None:
            return Path(blob["path"]).read_bytes()
    raise TypeError(f"unrecognised image blob type: {type(blob).__name__}")


def _blob_present(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, dict):
        return val.get("bytes") is not None or val.get("path") is not None
    return True


def _split_mask_png_to_target(blob: Any) -> np.ndarray:
    """v2 split-mask PNG (0=A, 1=B, 255=bg) → int64 target (0=bg, 1=A, 2=B)."""
    a = np.asarray(Image.open(io.BytesIO(_hf_image_bytes(blob))))
    if a.ndim == 3:
        a = a[..., 0]
    out = np.zeros(a.shape, dtype=np.int64)
    out[a == SPLIT_MASK_PIECE_A] = 1
    out[a == SPLIT_MASK_PIECE_B] = 2
    return out


def _dual_geometry_to_target(dv: np.ndarray) -> np.ndarray:
    """v1: dual geometry view (7, H, W) → target from mask ch5/6; overlap → bg."""
    mask_a = dv[CH_MASK_A] > 0.5
    mask_b = dv[CH_MASK_B] > 0.5
    out = np.zeros(mask_a.shape, dtype=np.int64)
    out[mask_a & ~mask_b] = 1
    out[mask_b & ~mask_a] = 2
    return out


def _row_has_em_view(row: Any, em_suffix: str) -> bool:  # kept for API compat
    return _blob_present(row.get(em_suffix))


# ───────────────────────────── Loaders ────────────────────────────────────

def download_hf_dataset(
    repo_id: str = "jeffbbrown2/ConnectomeBench2",
    cache_dir: str | Path | None = None,
    revision: str | None = "v2",
    allow_patterns: list[str] | None = None,
    local_dir: str | Path | None = None,
) -> Path:
    """Materialise the HF parquet-shards dataset locally; return the local root."""
    from huggingface_hub import snapshot_download
    p = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(local_dir) if local_dir else None,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    return Path(p)


# ────────────────────── Augmentation (CPU path) ───────────────────────────

def _train_aug(
    x: torch.Tensor,
    mask_gt: torch.Tensor,
    modality: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Modality-aware training augs (paper Appendix B.1).

      spatial (joint with mask): hflip p=0.5, rotate U(-25°, 25°), translate ±15%
          — bilinear for the input, nearest for the mask
      photometric (input only, geom: depth+normals, EM: grey channel):
          brightness U(0.6,1.4), contrast U(0.6,1.4), gaussian blur p=0.4 σ∈[0.1,2]
      occlusion: random erase p=0.4, box 10–30%, all channels
      channel-group dropout (geom only), p=0.15 each: silhouette / depth / normals
    """
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode

    H, W = x.shape[-2:]
    m = mask_gt.unsqueeze(0).to(torch.float32)  # (1, H, W)

    if random.random() < 0.5:
        x = TF.hflip(x)
        m = TF.hflip(m)

    angle = random.uniform(-25.0, 25.0)
    tx = int(random.uniform(-0.15, 0.15) * W)
    ty = int(random.uniform(-0.15, 0.15) * H)
    x = TF.affine(x, angle=angle, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0],
                  interpolation=InterpolationMode.BILINEAR)
    m = TF.affine(m, angle=angle, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0],
                  interpolation=InterpolationMode.NEAREST)
    x = x.contiguous()

    if modality == MODALITY_GEOM:
        ch_lo, ch_hi = CH_DEPTH, CH_NZ + 1   # depth + normals
    else:
        ch_lo, ch_hi = CH_EM_GREY, CH_EM_GREY + 1

    bright = random.uniform(0.6, 1.4)
    contr = random.uniform(0.6, 1.4)
    sub = x[ch_lo:ch_hi] * bright
    sub_mean = sub.mean(dim=(-2, -1), keepdim=True)
    x[ch_lo:ch_hi] = (sub - sub_mean) * contr + sub_mean

    if random.random() < 0.4:
        sigma = random.uniform(0.1, 2.0)
        x[ch_lo:ch_hi] = TF.gaussian_blur(x[ch_lo:ch_hi], kernel_size=[5, 5], sigma=[sigma, sigma])

    if random.random() < 0.4:
        eh = int(H * random.uniform(0.10, 0.30))
        ew = int(W * random.uniform(0.10, 0.30))
        if eh > 0 and ew > 0:
            y0 = random.randint(0, max(0, H - eh))
            x0 = random.randint(0, max(0, W - ew))
            x[:, y0:y0 + eh, x0:x0 + ew] = 0.0

    if modality == MODALITY_GEOM:
        for chans in ([CH_SILH], [CH_DEPTH], [CH_NX, CH_NY, CH_NZ]):
            if random.random() < 0.15:
                x[chans] = 0.0

    return x, m.squeeze(0).round().to(torch.long)


# ─────────────────── Blend weight strategies ──────────────────────────────

def compute_row_weights(
    species: np.ndarray,
    kinds: np.ndarray,
    task_ids: np.ndarray,
    cls_labels: np.ndarray,
    strategy: str,
    cell_overrides: dict[tuple[str, str], float] | None = None,
    class_balance: bool = True,
) -> np.ndarray:
    """Per-row sampling weight (unnormalised, float64). Cell = (species, kind)."""
    if strategy not in VALID_BLEND_STRATEGIES:
        raise ValueError(f"unknown blend strategy: {strategy!r}")
    cell_overrides = cell_overrides or {}
    n = len(species)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    cells = np.char.add(np.char.add(species.astype(str), "|"), kinds.astype(str))
    unique_cells, cell_inverse, cell_counts = np.unique(cells, return_inverse=True, return_counts=True)
    counts = cell_counts.astype(np.float64)
    n_species_total = len(np.unique(species))
    n_kinds_total = len(np.unique(kinds))

    if strategy in ("natural", "custom"):
        cell_w = 1.0 / counts
    elif strategy == "uniform_species":
        sp_count = np.array([(species == c.split("|", 1)[0]).sum() for c in unique_cells], dtype=np.float64)
        cell_w = 1.0 / (n_species_total * sp_count)
    elif strategy == "uniform_type":
        k_count = np.array([(kinds == c.split("|", 1)[1]).sum() for c in unique_cells], dtype=np.float64)
        cell_w = 1.0 / (n_kinds_total * k_count)
    elif strategy == "uniform_cell":
        cell_w = 1.0 / (len(unique_cells) * counts)
    else:  # sqrt
        sq = np.sqrt(counts)
        cell_w = sq / (sq.sum() * counts)

    for ci, label in enumerate(unique_cells):
        sp, k = label.split("|", 1)
        ov = cell_overrides.get((sp, k))
        if ov is not None:
            cell_w[ci] *= float(ov)

    weights = cell_w[cell_inverse].astype(np.float64)

    if class_balance:
        # Invert per-task class frequency so each (task, class) bucket carries
        # equal mass within its task.
        cb = np.ones(n, dtype=np.float64)
        for tid in (TASK_ENDPOINT, TASK_JUNCTION):
            for cls in (0, 1):
                mask = (task_ids == tid) & (cls_labels == cls)
                count = int(mask.sum())
                if count > 0:
                    cb[mask] = 1.0 / count
        weights = weights * cb

    return weights


@dataclass(slots=True)
class _Item:  # kept for API compatibility with compute_item_weights callers
    row_idx: int
    view_idx: int
    op_id: str
    task_id: int
    cls_label: int
    has_mask_gt: bool
    species: str
    sample_type: str
    kind: str
    same_neuron: bool
    modality: int
    em_view_suffix: str
    has_synapse_label: bool
    synapse_label: int


def compute_item_weights(
    items: list[_Item],
    strategy: str,
    cell_overrides: dict[tuple[str, str], float] | None = None,
    class_balance: bool = True,
) -> np.ndarray:
    """List-of-_Item wrapper around `compute_row_weights`."""
    return compute_row_weights(
        species=np.array([it.species for it in items]),
        kinds=np.array([it.kind for it in items]),
        task_ids=np.array([it.task_id for it in items]),
        cls_labels=np.array([it.cls_label for it in items]),
        strategy=strategy, cell_overrides=cell_overrides, class_balance=class_balance,
    )


# ───────────────────────────── Collate ────────────────────────────────────

def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors; keep python-side fields as lists."""
    keys_tensor = ("input", "mask_label")
    keys_long = ("modality", "task_id", "cls_label", "synapse_label", "view_idx")
    keys_bool = ("has_mask_gt", "has_synapse_label")
    keys_pass = ("op_id", "species", "kind", "sample_type")

    out: dict[str, Any] = {}
    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    for k in keys_long:
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
    for k in keys_bool:
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.bool)
    for k in keys_pass:
        out[k] = [b[k] for b in batch]
    return out


__all__ = [
    "BlendConfig",
    "NeurIPSDataset",
    "_Item",
    "TASK_ENDPOINT", "TASK_JUNCTION",
    "MODALITY_GEOM", "MODALITY_EM",
    "EM_VIEWS", "EM_COLUMNS_V1", "EM_COLUMNS_V2", "EM_VIEW_NAMES",
    "SAMPLE_TYPE_TO_KIND", "SPLIT_ALIASES",
    "METADATA_COLUMNS", "METADATA_COLUMNS_V1", "METADATA_COLUMNS_V2",
    "VALID_ABLATIONS",
    "detect_schema", "load_blend_config", "compute_item_weights", "compute_row_weights",
    "download_hf_dataset",
    "route_row", "synapse_label_from_row",
    "collate",
]
