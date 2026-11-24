"""cube_splits_v2.py — ConnectomeBench2 v2 spatial train/val/test assignment.

v2 rule (dataset card + paper §3.1.4):
  * the volume is tiled into (80 µm)^3 cubes anchored at the species'
    canonical segmentation-volume bbox (queried from CloudVolume);
  * each cube is assigned to train / val / test (75 / 12.5 / 12.5);
  * a sample takes the split of the cube containing its coordinate:
        merge_edit, split_edit, adjacent_control -> metadata.edit_point_nm
        junction_control                         -> metadata.interface_point_nm
        synapse_control                          -> metadata.synapse_ctr_pt_nm

The authors' exact per-cube hash is not public, so samples are assigned via a
*cube table* recovered from the released v2 data
(scripts/data_generation/neurips/build_v2_cube_splits.py → configs/cube_splits_v2/<species>.json).
Every cube that contains a released v2 sample keeps its released split, so
re-generated / re-fetched samples land in the same split as the release.
Cubes with no released samples fall back to a deterministic hash with the same
75/12.5/12.5 ratios.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

EDGE_NM_V2 = 80_000.0
RATIOS_V2 = (0.75, 0.125, 0.125)
SPLITS = ("train", "val", "test")

# Coordinate used for the split of each sample type (dataset card).
SPLIT_COORD_FIELD: dict[str, tuple[str, ...]] = {
    "merge_edit": ("edit_point_nm", "interface_point_nm"),
    "split_edit": ("edit_point_nm", "interface_point_nm"),
    "adjacent_control": ("edit_point_nm", "interface_point_nm"),
    "junction_control": ("interface_point_nm",),
    "synapse_control": ("synapse_ctr_pt_nm", "interface_point_nm"),
}

DEFAULT_TABLE_DIR = Path(__file__).resolve().parents[3] / "configs" / "cube_splits_v2"


def split_coordinate(sample_type: str, meta: dict[str, Any]) -> list[float] | None:
    """The coordinate (nm) that decides a sample's split, or None if missing.

    Falls back to `interface_point_nm` for edits/adjacent/synapse rows produced
    by older renderers that did not record the v2 field (≤ ~0.5 µm difference).
    """
    for key in SPLIT_COORD_FIELD.get(sample_type, ("interface_point_nm",)):
        v = meta.get(key)
        if v is not None and len(v) == 3:
            try:
                pt = [float(x) for x in v]
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(x) for x in pt):
                return pt
    return None


def cube_index(pt: Iterable[float], origin_nm: Iterable[float], edge_nm: float = EDGE_NM_V2) -> tuple[int, int, int]:
    p = list(pt)
    o = list(origin_nm)
    return (int(math.floor((p[0] - o[0]) / edge_nm)),
            int(math.floor((p[1] - o[1]) / edge_nm)),
            int(math.floor((p[2] - o[2]) / edge_nm)))


def hash_split(species: str, cube: tuple[int, int, int], seed: int = 0,
               ratios: tuple[float, float, float] = RATIOS_V2) -> str:
    """Deterministic fallback split for cubes without released samples."""
    h = hashlib.blake2b(f"{species}|{cube[0]}|{cube[1]}|{cube[2]}|{seed}".encode(), digest_size=8)
    u = int.from_bytes(h.digest(), "big") / 2.0 ** 64
    if u < ratios[0]:
        return "train"
    if u < ratios[0] + ratios[1]:
        return "val"
    return "test"


@dataclass
class CubeTable:
    """Per-species cube → split lookup (see build_v2_cube_splits.py)."""
    species: str
    origin_nm: list[float]
    edge_nm: float = EDGE_NM_V2
    cubes: dict[tuple[int, int, int], str] = field(default_factory=dict)
    seed: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    def split_of_point(self, pt: Iterable[float]) -> tuple[str, bool]:
        """(split, from_table). from_table=False → hash fallback was used."""
        c = cube_index(pt, self.origin_nm, self.edge_nm)
        s = self.cubes.get(c)
        if s is not None:
            return s, True
        return hash_split(self.species, c, self.seed), False

    def to_json(self) -> dict[str, Any]:
        return {
            "species": self.species, "origin_nm": self.origin_nm, "edge_nm": self.edge_nm,
            "seed": self.seed, "ratios": list(RATIOS_V2), "info": self.info,
            "cubes": {f"{c[0]},{c[1]},{c[2]}": s for c, s in sorted(self.cubes.items())},
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "CubeTable":
        cubes = {tuple(int(x) for x in k.split(",")): v for k, v in d.get("cubes", {}).items()}
        return cls(species=d["species"], origin_nm=[float(x) for x in d["origin_nm"]],
                   edge_nm=float(d.get("edge_nm", EDGE_NM_V2)), cubes=cubes,  # type: ignore[arg-type]
                   seed=int(d.get("seed", 0)), info=d.get("info", {}))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1))
        tmp.replace(p)


def load_cube_tables(table_dir: str | Path = DEFAULT_TABLE_DIR) -> dict[str, CubeTable]:
    out: dict[str, CubeTable] = {}
    d = Path(table_dir)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        t = CubeTable.from_json(json.loads(p.read_text()))
        out[t.species] = t
    return out


def assign_split_v2(
    species: str,
    sample_type: str,
    meta: dict[str, Any],
    tables: dict[str, CubeTable],
) -> tuple[str, str]:
    """Return (split, provenance) for one sample.

    provenance ∈ {"table", "hash", "no_table", "no_coord"}; split is
    "unmatched" for the last two (callers drop those rows).
    """
    pt = split_coordinate(sample_type, meta)
    if pt is None:
        return "unmatched", "no_coord"
    table = tables.get(species)
    if table is None:
        return "unmatched", "no_table"
    split, from_table = table.split_of_point(pt)
    return split, ("table" if from_table else "hash")


__all__ = [
    "EDGE_NM_V2", "RATIOS_V2", "SPLIT_COORD_FIELD", "DEFAULT_TABLE_DIR",
    "split_coordinate", "cube_index", "hash_split",
    "CubeTable", "load_cube_tables", "assign_split_v2",
]
