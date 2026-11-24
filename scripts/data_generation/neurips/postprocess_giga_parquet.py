"""Idempotent postprocessing of combined.parquet into ConnectomeBench2 **v2** semantics.

Input: `combined.parquet` from build_giga_parquet.py (renderer `questions.parquet`
files concatenated; `images` = relative paths, `metadata` = JSON string with
`image_types` aligned to `images`).

Output (in place, atomic): the v2 row semantics documented on the HF dataset card
(https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2, branch v2):

  identifiers  combined_sample_hash = md5(f"{source_archive}|{source_archive_sample_hash}"),
               source_archive ∈ {unified_<sp>, unified_controls_<sp>}
  identity     sample_type ∈ {merge_edit, split_edit, adjacent_control,
               junction_control, synapse_control}, same_neuron, species
  images       one `geometry` per row — DUAL render for merge_edit / adjacent /
               synapse, SINGLE render for split_edit / junction (the renderer's
               `geometry_single` is renamed to `geometry`); EM renamed to
               `em_{xy,xz,yz,best}_before`; `split_mask_{front,side,top}` kept for
               split_edit only. `images` / `metadata.image_types` are rewritten.
  flags        has_em, has_after_mask, present_slots
  routing      task_routing from sample_type + has_after_mask
  labels       false_split_correction_label / false_merge_identification_label
               (float, NULL where the task does not apply)
  split        v2 80 µm-cube assignment on the per-type coordinate, using the cube
               tables recovered from the release (configs/cube_splits_v2/, built by
               build_v2_cube_splits.py) so regenerated samples match the release.

Rows without a usable geometry flavour or without all four EM views are dropped
(v2 guarantees both), unless --allow-incomplete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "data_generation"))
from connectome.cube_splits_v2 import (  # noqa: E402
    DEFAULT_TABLE_DIR, assign_split_v2, load_cube_tables,
)

GIGA_DEFAULT = Path.home() / "VLM_Proofreading" / "neurips_dataset" / "combined.parquet"

SAMPLE_TYPES_TRUE = {"merge_edit", "junction_control"}
SAMPLE_TYPES_FALSE = {"split_edit", "adjacent_control", "synapse_control"}
DUAL_TYPES = {"merge_edit", "adjacent_control", "synapse_control"}
SINGLE_TYPES = {"split_edit", "junction_control"}
CANONICAL_STRATEGY = {
    "merge_edit": "merge_correction", "split_edit": "split_correction",
    "adjacent_control": "adjacent_control", "junction_control": "junction_control",
    "synapse_control": "synapse_control",
}
META_DROP_FIELDS = ("original_is_merge", "inverted_action", "is_correct")

ROUTE_SPLIT = "false_split_correction"
ROUTE_MERGE_ID = "false_merge_identification"
ROUTE_MASK_GEN = "split_mask_generation"

EM_VIEWS = ("xy", "xz", "yz", "best")
EM_SLOTS = tuple(f"em_{v}_before" for v in EM_VIEWS)
MASK_SLOTS = tuple(f"split_mask_{v}" for v in ("front", "side", "top"))

# Renderer archive (sub-directory) names → v2 `source_archive`.
_ARCHIVE_PREFIXES = (
    ("edits_and_adj_controls_", "unified_"),
    ("junction_controls_", "unified_controls_"),
    ("synapse_controls_", "unified_controls_"),
    ("unified_controls_", "unified_controls_"),
    ("unified_", "unified_"),
)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")


def canonical_archive(name: str) -> tuple[str, str | None]:
    """(v2 source_archive, species) from a renderer/v1/v2 archive name."""
    for prefix, new in _ARCHIVE_PREFIXES:
        if name.startswith(prefix):
            sp = name[len(prefix):]
            return f"{new}{sp}", sp
    return name, None


def derive_sample_type(archive: str, subdir: str, meta: dict[str, Any]) -> str | None:
    strategy = meta.get("strategy")
    s = str(strategy).lower() if strategy is not None else ""
    if s in CANONICAL_STRATEGY.values():                # already v2-canonical
        return {v: k for k, v in CANONICAL_STRATEGY.items()}[s]
    src = f"{archive} {subdir}"
    if "controls_" in archive or "junction" in subdir or "synapse" in subdir:
        if "synapse" in s or meta.get("source_type") == "synapse_pair" or "synapse" in subdir:
            return "synapse_control"
        if "junction" in s or "junction" in subdir:
            return "junction_control"
        return None
    if "unified_" in src or "edits_and_adj" in src:
        if s in ("inversion", "split_correction"):
            return "split_edit"
        if "adj" in s:
            return "adjacent_control"
        if not s or s == "merge_correction":
            # Plain proofreading operation: the edit direction decides the type.
            is_merge = meta.get("is_merge")
            if meta.get("source_type") == "operation" and is_merge is False:
                return "split_edit"
            return "merge_edit"
    return None


def select_images(sample_type: str, images: list[str], image_types: list[str]) -> tuple[list[str], list[str], dict[str, str]]:
    """Pick v2 slots from the renderer outputs. Returns (images, image_types, slot→path)."""
    by_type: dict[str, str] = {}
    for p, t in zip(images, image_types):
        by_type.setdefault(str(t), str(p))
    slots: dict[str, str] = {}

    # Already-v2 input (re-run): `geometry` is the correct flavour by construction.
    already_v2 = "geometry_single" not in by_type and any(t.endswith("_before") for t in by_type)

    # geometry flavour
    if already_v2 and "geometry" in by_type:
        slots["geometry"] = by_type["geometry"]
    elif sample_type in SINGLE_TYPES:
        if "geometry_single" in by_type:
            slots["geometry"] = by_type["geometry_single"]
        elif sample_type == "junction_control" and "geometry" in by_type:
            # Junction renders have one root → the plain render is already single-segment.
            slots["geometry"] = by_type["geometry"]
    elif "geometry" in by_type:
        slots["geometry"] = by_type["geometry"]

    # EM (renderer: em_xy … em_best; v2: em_xy_before …)
    for v in EM_VIEWS:
        for t in (f"em_{v}_before", f"em_{v}"):
            if t in by_type:
                slots[f"em_{v}_before"] = by_type[t]
                break

    if sample_type == "split_edit":
        for m in MASK_SLOTS:
            if m in by_type:
                slots[m] = by_type[m]

    order = ["geometry", *MASK_SLOTS, *EM_SLOTS]
    new_types = [t for t in order if t in slots]
    return [slots[t] for t in new_types], new_types, slots


def derive_task_routing(sample_type: str, has_geometry: bool, has_after_mask: bool) -> list[str]:
    if not has_geometry:
        return []
    if sample_type in DUAL_TYPES:
        return [ROUTE_SPLIT]
    routes = [ROUTE_MERGE_ID]
    if sample_type == "split_edit" and has_after_mask:
        routes.append(ROUTE_MASK_GEN)
    return routes


def mutate_metadata(meta: dict[str, Any], sample_type: str, image_types: list[str],
                    species: str | None) -> dict[str, Any]:
    out = dict(meta)
    for f in META_DROP_FIELDS:
        out.pop(f, None)
    if sample_type == "split_edit":
        before = out.get("before_root_ids") or []
        after = out.get("after_root_ids") or []
        if isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)) and len(before) > len(after):
            out["before_root_ids"], out["after_root_ids"] = list(after), list(before)
    out["strategy"] = CANONICAL_STRATEGY[sample_type]
    out["is_merge"] = sample_type in DUAL_TYPES           # v2: True ⇔ `geometry` is dual
    out["image_types"] = image_types
    if species and not out.get("species"):
        out["species"] = species
    return out


def write_atomic(df: pd.DataFrame, path: Path, log: logging.Logger) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    tmp = path.with_suffix(".parquet.tmp")
    log.info(f"writing parquet -> {path}")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    log.info(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    setup_logging()
    log = logging.getLogger("postprocess")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, default=GIGA_DEFAULT)
    ap.add_argument("--splits-root", type=Path, default=DEFAULT_TABLE_DIR,
                    help="v2 cube tables (build_v2_cube_splits.py output)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="keep rows missing geometry or some EM views (not v2-conformant)")
    ap.add_argument("--keep-hash-fallback", action=argparse.BooleanOptionalAction, default=True,
                    help="assign samples in cubes unseen in the release by deterministic hash "
                         "(otherwise they are dropped)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.parquet.exists():
        log.error(f"not found: {args.parquet}")
        sys.exit(1)
    t0 = time.time()
    df = pq.read_table(args.parquet).to_pandas()
    log.info(f"loaded {len(df)} rows, {len(df.columns)} cols")

    if "sample_hash" in df.columns and "source_archive_sample_hash" not in df.columns:
        df = df.rename(columns={"sample_hash": "source_archive_sample_hash"})
    if "source_archive_sample_hash" not in df.columns:
        raise RuntimeError("no sample_hash / source_archive_sample_hash column")
    if "source_subdir" not in df.columns:
        df["source_subdir"] = df["source_archive"]

    canon = df["source_archive"].map(canonical_archive)
    df["source_archive"] = canon.map(lambda t: t[0])
    archive_species = canon.map(lambda t: t[1])
    df["combined_sample_hash"] = [hashlib.md5(f"{sa}|{sh}".encode()).hexdigest()
                                  for sa, sh in zip(df["source_archive"], df["source_archive_sample_hash"])]
    dup = df["combined_sample_hash"].duplicated(keep="first")
    if dup.any():
        log.warning(f"dropping {int(dup.sum())} rows with duplicate combined_sample_hash")
        df = df.loc[~dup].reset_index(drop=True)
        archive_species = archive_species.loc[~dup.to_numpy()].reset_index(drop=True)

    metas = [json.loads(m) if isinstance(m, str) else (m or {}) for m in df["metadata"]]
    df["sample_type"] = [derive_sample_type(sa, sd, m) for sa, sd, m in
                         zip(df["source_archive"], df["source_subdir"], metas)]
    bad = df["sample_type"].isna()
    if bad.any():
        ex = Counter(str(m.get("strategy")) for m, b in zip(metas, bad) if b)
        log.warning(f"dropping {int(bad.sum())} rows with unknown sample_type (strategies: {dict(ex)})")
    df["species"] = [m.get("species") or sp for m, sp in zip(metas, archive_species)]

    rows: list[dict[str, Any]] = []
    drop_reasons: Counter[str] = Counter()
    split_prov: Counter[str] = Counter()
    tables = load_cube_tables(args.splits_root)
    if not tables:
        log.error(f"no cube tables in {args.splits_root}; run build_v2_cube_splits.py first")
        sys.exit(1)
    log.info(f"cube tables: {sorted(tables)} from {args.splits_root}")

    for i, rec in enumerate(df.to_dict("records")):
        st = rec["sample_type"]
        if st is None:
            drop_reasons["unknown_sample_type"] += 1
            continue
        meta = metas[i]
        raw_images = rec.get("images")
        images = [] if raw_images is None else list(raw_images)  # numpy array from parquet
        itypes = list(meta.get("image_types") or [])
        new_images, new_types, slots = select_images(st, images, itypes)
        has_geometry = "geometry" in slots
        n_em = sum(s in slots for s in EM_SLOTS)
        if not args.allow_incomplete:
            if not has_geometry:
                drop_reasons[f"no_{'single' if st in SINGLE_TYPES else 'dual'}_geometry"] += 1
                continue
            if n_em < len(EM_SLOTS):
                drop_reasons["incomplete_em"] += 1
                continue
        has_after_mask = st == "split_edit" and all(m in slots for m in MASK_SLOTS)
        same = st in SAMPLE_TYPES_TRUE
        new_meta = mutate_metadata(meta, st, new_types, rec["species"])

        split, prov = assign_split_v2(rec["species"], st, new_meta, tables)
        if prov == "hash" and not args.keep_hash_fallback:
            split, prov = "unmatched", "hash_disabled"
        split_prov[prov] += 1
        if split == "unmatched":
            drop_reasons[f"split_{prov}"] += 1
            continue

        rows.append({
            "combined_sample_hash": rec["combined_sample_hash"],
            "source_archive_sample_hash": rec["source_archive_sample_hash"],
            "source_archive": rec["source_archive"],
            "sample_type": st,
            "same_neuron": same,
            "task_routing": derive_task_routing(st, has_geometry, has_after_mask),
            "false_split_correction_label": (1.0 if same else 0.0) if st in DUAL_TYPES else None,
            "false_merge_identification_label": (0.0 if same else 1.0) if st in SINGLE_TYPES else None,
            "split": split,
            "species": rec["species"],
            "has_em": n_em > 0,
            "has_after_mask": has_after_mask,
            "present_slots": sorted(slots),
            "metadata": json.dumps(new_meta, sort_keys=True, default=str),
            "images": new_images,
            "source_subdir": rec["source_subdir"],
        })

    out = pd.DataFrame(rows)
    log.info("=" * 60)
    log.info(f"kept {len(out)} / {len(df)} rows; dropped: {dict(drop_reasons)}")
    log.info(f"split provenance: {dict(split_prov)}  (table = cube seen in the v2 release)")
    if len(out):
        log.info("rows per (species, sample_type, split):")
        for k, v in out.groupby(["species", "sample_type", "split"]).size().items():
            log.info(f"  {k}: {v}")
        log.info(f"split_edit with after-mask: {int(out['has_after_mask'].sum())} / "
                 f"{int((out['sample_type'] == 'split_edit').sum())}")
    if args.dry_run:
        log.info(f"[dry-run] not writing. elapsed {time.time() - t0:.1f}s")
        return
    write_atomic(out, args.parquet, log)
    log.info(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
