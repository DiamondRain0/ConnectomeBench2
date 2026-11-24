"""build_v2_cube_splits.py — recover the v2 cube → split table from the released dataset.

ConnectomeBench2 v2 assigns splits per (80 µm)^3 cube, but the per-cube hash
used by the authors is not public. This script reconstructs the assignment
from the released rows themselves (each row carries `split` + coordinates in
`metadata`), so that anything re-generated or re-fetched from upstream with this
repo lands in the SAME split as the release.

Grid origin (cube anchor), per species:
  --origin cloudvolume  segmentation-volume bbox min from CloudVolume (needs the
                        CAVE tokens from env.sh; this is what the card describes)
  --origin infer        search the per-axis offset that makes splits constant
                        inside cubes (no credentials needed)
  --origin auto         (default) try cloudvolume, fall back to / compare with infer
  --origin-nm mouse=x,y,z;fly=x,y,z   explicit values

The output reports *purity* = fraction of released samples whose cube's
majority split equals their own split. With the correct origin it should be
~100%; anything well below means the origin (or coordinate field) is wrong.

Usage (dataset downloaded with scripts/training/download_dataset.py):
    python scripts/data_generation/neurips/build_v2_cube_splits.py \
        --dataset /data/ConnectomeBench2 --out configs/cube_splits_v2

    # without a local copy: stream only the needed columns from the HF Hub (slow)
    python scripts/data_generation/neurips/build_v2_cube_splits.py --hf --out configs/cube_splits_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "data_generation"))
from connectome.cube_splits_v2 import (  # noqa: E402
    EDGE_NM_V2, SPLITS, CubeTable, split_coordinate,
)

HF_REPO = "jeffbbrown2/ConnectomeBench2"
READ_COLS = ["species", "sample_type", "split", "metadata"]
SPLIT_CODE = {s: i for i, s in enumerate(SPLITS)}
# Paper §3.1.4: total cube counts of the 80 µm grid (sanity check only).
PAPER_CUBE_COUNTS = {"mouse": 1960, "fly": 264, "human": 5616, "zebrafish": 784}


# ───────────────────────────── load points ─────────────────────────────────

def _shard_list_local(root: Path, max_shards: int | None) -> list[Path]:
    out: list[Path] = []
    for sp in ("train", "val", "test"):
        shards = sorted((root / sp).glob(f"{sp}-*.parquet"))
        out += shards[:max_shards] if max_shards else shards
    if not out:
        raise SystemExit(f"no shards under {root}/{{train,val,test}}/")
    return out


def _read_points(tables) -> pd.DataFrame:
    df = pd.concat(tables, ignore_index=True)
    pts = [split_coordinate(st, json.loads(m) if isinstance(m, str) else (m or {}))
           for st, m in zip(df["sample_type"], df["metadata"])]
    ok = np.array([p is not None for p in pts])
    xyz = np.array([p for p in pts if p is not None], dtype=np.float64).reshape(-1, 3)
    out = df.loc[ok, ["species", "sample_type", "split"]].reset_index(drop=True)
    out[["x", "y", "z"]] = xyz
    out["split"] = out["split"].replace({"validation": "val"})
    print(f"  {len(out)} points ({int((~ok).sum())} rows without a split coordinate)")
    return out


def load_points_local(root: Path, max_shards: int | None) -> pd.DataFrame:
    shards = _shard_list_local(root, max_shards)
    print(f"reading {len(shards)} local shards (columns {READ_COLS})")
    tables = []
    for i, s in enumerate(shards):
        tables.append(pq.read_table(s, columns=READ_COLS).to_pandas())
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(shards)}", flush=True)
    return _read_points(tables)


def load_points_hf(revision: str, max_shards: int | None) -> pd.DataFrame:
    import csv
    from huggingface_hub import HfFileSystem, hf_hub_download
    fs = HfFileSystem()
    csv_path = hf_hub_download(HF_REPO, "shards.csv", repo_type="dataset", revision=revision)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if max_shards:
        rows = [r for sp in ("train", "val", "test")
                for r in sorted((x for x in rows if x["split"] == sp), key=lambda x: int(x["shard_id"]))[:max_shards]]
    print(f"streaming {len(rows)} shards from the HF Hub (metadata columns only; this is slow)")
    tables = []
    for i, r in enumerate(rows):
        url = f"datasets/{HF_REPO}@{revision}/{r['path']}"
        with fs.open(url, "rb", block_size=1 << 16) as fh:  # small blocks: skip image bytes
            tables.append(pq.ParquetFile(fh).read(columns=READ_COLS).to_pandas())
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    return _read_points(tables)


# ───────────────────────────── origin search ───────────────────────────────

def purity(xyz: np.ndarray, codes: np.ndarray, origin: np.ndarray, edge: float) -> tuple[float, int]:
    """Fraction of points agreeing with their cube's majority split; #occupied cubes."""
    c = np.floor((xyz - origin) / edge).astype(np.int64)
    c -= c.min(0)
    dims = c.max(0) + 1
    key = (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
    k_u, k_inv = np.unique(key, return_inverse=True)
    counts = np.zeros((len(k_u), len(SPLITS)), dtype=np.int64)
    np.add.at(counts, (k_inv, codes), 1)
    return float(counts.max(1).sum() / len(codes)), int(len(k_u))


def infer_origin(xyz: np.ndarray, codes: np.ndarray, edge: float, start: np.ndarray | None = None,
                 coarse_nm: float = 2000.0, fine_nm: float = 100.0, rounds: int = 3) -> tuple[np.ndarray, float]:
    """Coordinate-descent over per-axis offsets in [0, edge) maximising purity."""
    base = np.zeros(3) if start is None else np.mod(start, edge)
    best = base.copy()
    best_p, _ = purity(xyz, codes, best, edge)
    for step, span in ((coarse_nm, edge), (fine_nm, 2 * coarse_nm)):
        for _ in range(rounds):
            improved = False
            for ax in range(3):
                cands = np.arange(best[ax] - span / 2, best[ax] + span / 2, step) if span < edge \
                    else np.arange(0.0, edge, step)
                for v in cands:
                    trial = best.copy()
                    trial[ax] = np.mod(v, edge)
                    p, _ = purity(xyz, codes, trial, edge)
                    if p > best_p + 1e-12:
                        best_p, best, improved = p, trial, True
            if not improved:
                break
    return best, best_p


def cloudvolume_origin(species: str) -> np.ndarray | None:
    """Segmentation bbox min (nm, mip 0) from CloudVolume, or None on failure."""
    try:
        import cloudvolume
        from dotenv import load_dotenv
        load_dotenv()
        from connectome.em_data import DATA_PARAMETERS
        token = os.getenv("CAVE_API_TOKEN_DAF") if species in ("human", "zebrafish") else os.getenv("CAVE_API_TOKEN")
        token = token or os.getenv("CAVE_API_TOKEN")
        secrets = {"token": token} if token else None
        cv = cloudvolume.CloudVolume(DATA_PARAMETERS[species]["seg_path"], mip=0, use_https=True,
                                     progress=False, secrets=secrets)
        return np.asarray(cv.bounds.minpt, dtype=float) * np.asarray(cv.resolution, dtype=float)
    except Exception as e:  # missing creds / network / package
        print(f"  [{species}] CloudVolume origin unavailable: {type(e).__name__}: {e}")
        return None


# ───────────────────────────── main ────────────────────────────────────────

def parse_manual(s: str | None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not s:
        return out
    for part in s.split(";"):
        sp, xyz = part.split("=")
        out[sp.strip()] = np.array([float(v) for v in xyz.split(",")])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=Path, help="local v2 root containing train/ val/ test/")
    src.add_argument("--hf", action="store_true", help="stream metadata columns from the HF Hub")
    ap.add_argument("--revision", default="v2")
    ap.add_argument("--max-shards", type=int, default=None, help="per split (quick partial table)")
    ap.add_argument("--origin", default="auto", choices=["auto", "cloudvolume", "infer"])
    ap.add_argument("--origin-nm", default=None, help="manual override, e.g. 'mouse=0,0,0;fly=0,0,0'")
    ap.add_argument("--edge-nm", type=float, default=EDGE_NM_V2)
    ap.add_argument("--seed", type=int, default=0, help="seed of the hash fallback for unseen cubes")
    ap.add_argument("--min-purity", type=float, default=0.99,
                    help="warn (and exit non-zero) if a species' table is less consistent than this")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "configs" / "cube_splits_v2")
    args = ap.parse_args()

    pts = load_points_local(args.dataset, args.max_shards) if args.dataset else \
        load_points_hf(args.revision, args.max_shards)
    manual = parse_manual(args.origin_nm)
    report: dict[str, dict] = {}
    worst = 1.0

    for species, g in pts.groupby("species"):
        xyz = g[["x", "y", "z"]].to_numpy()
        codes = g["split"].map(SPLIT_CODE).to_numpy()
        if np.isnan(codes.astype(float)).any():
            raise SystemExit(f"unexpected split values: {sorted(g['split'].unique())}")
        codes = codes.astype(np.int64)
        print(f"\n== {species}: {len(g)} samples")

        cands: dict[str, np.ndarray] = {}
        if species in manual:
            cands["manual"] = manual[species]
        else:
            if args.origin in ("auto", "cloudvolume"):
                o = cloudvolume_origin(species)
                if o is not None:
                    cands["cloudvolume"] = o
            if args.origin == "infer" or (args.origin == "auto"):
                o, _ = infer_origin(xyz, codes, args.edge_nm, start=cands.get("cloudvolume"))
                cands["infer"] = o
        if not cands:
            raise SystemExit(f"[{species}] no origin available (use --origin infer or --origin-nm)")

        scored = {k: purity(xyz, codes, o, args.edge_nm) for k, o in cands.items()}
        # Baseline: purity at random origins. Purity is only meaningful relative to this
        # (sparse data makes any grid look pure); a correct origin should sit far above it.
        rng = np.random.default_rng(0)
        base = [purity(xyz, codes, rng.uniform(0, args.edge_nm, 3), args.edge_nm)[0] for _ in range(20)]
        base_mean, base_max = float(np.mean(base)), float(np.max(base))
        print(f"  baseline purity at 20 random origins: mean={base_mean:.5f} max={base_max:.5f}")
        for k, (p, n) in scored.items():
            print(f"  origin[{k}] = {np.round(cands[k], 1).tolist()}  purity={p:.5f}  occupied cubes={n}")
        pick = max(scored, key=lambda k: (scored[k][0], k == "cloudvolume"))
        origin = cands[pick]
        p, n_occ = scored[pick]
        worst = min(worst, p)

        cube = np.floor((xyz - origin) / args.edge_nm).astype(np.int64)
        df = pd.DataFrame({"i": cube[:, 0], "j": cube[:, 1], "k": cube[:, 2], "split": g["split"].to_numpy()})
        maj = df.groupby(["i", "j", "k"])["split"].agg(lambda s: s.value_counts().idxmax())
        table = CubeTable(species=species, origin_nm=[float(v) for v in origin], edge_nm=args.edge_nm,
                          cubes={tuple(int(v) for v in idx): s for idx, s in maj.items()},  # type: ignore[misc]
                          seed=args.seed)
        cube_split_counts = maj.value_counts().to_dict()
        table.info = {
            "origin_source": pick, "purity": p, "baseline_purity_random_origin_mean": base_mean,
            "baseline_purity_random_origin_max": base_max, "n_samples": int(len(g)), "occupied_cubes": n_occ,
            "cube_split_counts": cube_split_counts,
            "sample_split_counts": g["split"].value_counts().to_dict(),
            "paper_total_cubes": PAPER_CUBE_COUNTS.get(species),
            "dataset": str(args.dataset) if args.dataset else f"hf:{HF_REPO}@{args.revision}",
            "max_shards": args.max_shards,
        }
        table.save(args.out / f"{species}.json")
        report[species] = table.info
        print(f"  -> {args.out / (species + '.json')}  (origin={pick}, purity={p:.5f}, cubes {cube_split_counts})")
        if p < args.min_purity:
            print(f"  WARNING: purity {p:.4f} < {args.min_purity}: the cube grid does not reproduce the "
                  f"released split well; check the origin / coordinate fields.")

    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {args.out / 'report.json'}")
    if worst < args.min_purity:
        sys.exit(2)


if __name__ == "__main__":
    main()
