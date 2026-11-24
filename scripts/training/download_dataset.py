"""download_dataset.py — fetch ConnectomeBench2 (HF `jeffbbrown2/ConnectomeBench2`).

The full v2 release is ~230 GB (903 shards of ~250 MB: 669 train / 105 val /
129 test). Download only what you need:

    # everything (paper reproduction)
    python scripts/training/download_dataset.py --out /data/ConnectomeBench2

    # metadata sidecars only (identifiers, labels, coords — no images, a few MB)
    python scripts/training/download_dataset.py --out /data/ConnectomeBench2 --splits none

    # test split only, e.g. to evaluate a checkpoint
    python scripts/training/download_dataset.py --out /data/ConnectomeBench2 --splits test

    # a small subset for debugging: first N shards per split
    python scripts/training/download_dataset.py --out /data/cb2_small --max-shards 4

The resulting directory is what `--parquet-root` expects:
    <out>/{train,val,test}/<split>-NNNNN.parquet, <out>/metadata/, <out>/shards.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

REPO = "jeffbbrown2/ConnectomeBench2"


def sha256(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision", default="v2", help="HF branch/tag (v2 = June 2026 release used by the paper)")
    ap.add_argument("--splits", default="train,val,test", help="comma list of train,val,test, or 'none'")
    ap.add_argument("--max-shards", type=int, default=None, help="only the first N shards per split")
    ap.add_argument("--verify", action="store_true", help="check sha256 against shards.csv after download")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    args.out.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPO, repo_type="dataset", revision=args.revision, local_dir=str(args.out),
                      allow_patterns=["README.md", "shards.csv", "metadata/*", "demo.parquet"])

    splits = [] if args.splits == "none" else [s.strip() for s in args.splits.split(",") if s.strip()]
    with open(args.out / "shards.csv") as f:
        shards = [r for r in csv.DictReader(f) if r["split"] in splits]
    if args.max_shards is not None:
        keep = []
        for sp in splits:
            keep += sorted((r for r in shards if r["split"] == sp), key=lambda r: int(r["shard_id"]))[:args.max_shards]
        shards = keep
    total = sum(int(r["size_bytes"]) for r in shards)
    print(f"downloading {len(shards)} shards ({total / 1e9:.1f} GB) to {args.out}")

    if args.max_shards is None:
        snapshot_download(REPO, repo_type="dataset", revision=args.revision, local_dir=str(args.out),
                          allow_patterns=[f"{sp}/*.parquet" for sp in splits], max_workers=args.workers)
    else:
        for r in shards:
            hf_hub_download(REPO, r["path"], repo_type="dataset", revision=args.revision,
                            local_dir=str(args.out))

    if args.verify:
        bad = [r["path"] for r in shards if sha256(args.out / r["path"]) != r["sha256"]]
        print("sha256 OK" if not bad else f"sha256 MISMATCH in {len(bad)} shards: {bad[:5]}")


if __name__ == "__main__":
    main()
