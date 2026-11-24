"""check_env.py — preflight for a compute machine before a ConnectomeBench2 training run.

    python scripts/training/check_env.py --data /shared/ConnectomeBench2 --runs /shared/runs

Checks (FAIL = training will not work, WARN = it will work but slowly / degraded):
  * Python 3.11, torch + torchvision + sklearn importable
  * CUDA GPUs visible, bf16 supported, NCCL available (multi-GPU)
  * dataset layout: {train,val,test}/<split>-*.parquet present, counts vs shards.csv,
    metadata/ sidecars (fast start-up), v2 schema
  * read throughput of one shard from the shared storage
  * ImageNet ViT weights cached under $TORCH_HOME (offline compute nodes)
  * runs/checkpoint directory writable, free disk space, CPU count vs --num-workers
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

RESULTS: list[tuple[str, str, str]] = []


def report(level: str, what: str, detail: str = "") -> None:
    RESULTS.append((level, what, detail))
    print(f"[{level:4s}] {what}" + (f" - {detail}" if detail else ""), flush=True)


def check_python() -> None:
    v = sys.version_info
    report("OK" if (v.major, v.minor) == (3, 11) else "WARN", f"python {v.major}.{v.minor}.{v.micro}",
           "" if (v.major, v.minor) == (3, 11) else "pyproject pins 3.11 (other versions untested)")
    for mod in ("torch", "torchvision", "sklearn", "pyarrow", "pandas", "yaml", "PIL"):
        try:
            m = __import__(mod)
            report("OK", f"import {mod}", getattr(m, "__version__", ""))
        except Exception as e:
            report("FAIL", f"import {mod}", str(e))


def check_gpu(ngpu_wanted: int) -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        report("FAIL", "CUDA", f"not available (torch {torch.__version__}, built for CUDA {torch.version.cuda}); "
               "check the NVIDIA driver vs the torch CUDA build")
        return
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        report("OK", f"GPU {i}", f"{p.name}, {p.total_memory / 1e9:.0f} GB")
    report("OK" if n >= ngpu_wanted else "FAIL", f"{n} GPU(s) visible", f"wanted {ngpu_wanted}")
    report("OK" if torch.cuda.is_bf16_supported() else "WARN", "bf16 autocast",
           "" if torch.cuda.is_bf16_supported() else "not supported: use --amp-dtype float16")
    if ngpu_wanted > 1:
        import torch.distributed as dist
        report("OK" if dist.is_nccl_available() else "FAIL", "NCCL available")
    try:
        x = torch.randn(64, 3, 224, 224, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            y = torch.nn.functional.conv2d(x, torch.randn(768, 3, 16, 16, device="cuda"), stride=16)
        torch.cuda.synchronize()
        report("OK", "GPU compute", f"conv out {tuple(y.shape)}")
    except Exception as e:
        report("FAIL", "GPU compute", str(e))


def check_data(root: Path, full: bool) -> None:
    if not root.is_dir():
        report("FAIL", f"dataset root {root}", "missing (is the shared storage mounted?)")
        return
    expected: dict[str, int] = {}
    csv_path = root / "shards.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                expected[r["split"]] = expected.get(r["split"], 0) + 1
    else:
        report("WARN", "shards.csv", "missing (cannot verify completeness)")
    for sp in ("train", "val", "test"):
        n = len(list((root / sp).glob(f"{sp}-*.parquet")))
        exp = expected.get(sp)
        if n == 0:
            report("FAIL", f"{sp} shards", f"none under {root / sp}")
        elif exp and n < exp:
            report("FAIL" if full else "WARN", f"{sp} shards", f"{n}/{exp} (partial download)")
        else:
            report("OK", f"{sp} shards", f"{n}" + (f"/{exp}" if exp else ""))
        side = root / "metadata" / f"{sp}.parquet"
        report("OK" if side.exists() else "WARN", f"metadata/{sp}.parquet",
               "" if side.exists() else "missing: start-up will scan every shard (slow on network storage)")
    stray = list(root.rglob("*.incomplete"))[:3]
    if stray:
        report("WARN", "incomplete downloads", ", ".join(str(s) for s in stray))
    shards = sorted((root / "test").glob("test-*.parquet")) or sorted((root / "val").glob("val-*.parquet"))
    if not shards:
        return
    try:
        import pyarrow.parquet as pq
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "neurips_training"))
        from data import detect_schema
        schema = detect_schema(shards[0])
        report("OK" if schema == "v2" else "WARN", f"schema {schema}",
               "" if schema == "v2" else "paper numbers use v2 (HF branch v2)")
        t0 = time.time()
        n_bytes = shards[0].stat().st_size
        pf = pq.ParquetFile(str(shards[0]))
        n = sum(len(b) for b in pf.iter_batches(batch_size=64))
        dt = time.time() - t0
        mbps = n_bytes / 1e6 / max(dt, 1e-6)
        # ViT-B on one H100 consumes roughly 1500 views/s ≈ 215 rows/s ≈ 70 MB/s per GPU.
        report("OK" if mbps > 150 else "WARN", f"shard read {shards[0].name}",
               f"{n} rows, {n_bytes / 1e6:.0f} MB in {dt:.1f}s = {mbps:.0f} MB/s (first read; ~70 MB/s per GPU needed)")
    except Exception as e:
        report("FAIL", "shard read", str(e))


def check_weights(models: list[str]) -> None:
    th = os.environ.get("TORCH_HOME")
    base = Path(th) if th else Path.home() / ".cache" / "torch"
    ckdir = base / "hub" / "checkpoints"
    try:
        from torchvision.models import ViT_B_16_Weights, ViT_H_14_Weights, ViT_L_16_Weights
        urls = {"vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1.url, "vit_l_16": ViT_L_16_Weights.IMAGENET1K_V1.url,
                "vit_h_14": ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.url}
    except Exception:
        urls = {}
    for m in models:
        f = ckdir / Path(urls.get(m, f"{m}.pth")).name
        report("OK" if f.exists() else "WARN", f"ImageNet weights {m}",
               str(f) if f.exists() else f"not cached in {ckdir}; needs internet at first start "
               "(run scripts/lab/setup_compute.sh on a machine with internet, TORCH_HOME on shared storage)")


def check_runs(runs: Path, num_workers: int, ngpu: int) -> None:
    try:
        runs.mkdir(parents=True, exist_ok=True)
        probe = runs / f".write_probe_{os.getpid()}"
        probe.write_bytes(b"x" * 1024)
        probe.unlink()
        free = shutil.disk_usage(runs).free / 1e9
        # each checkpoint (ViT-B best+last with optimizer) ~1.5 GB; ViT-L ~5 GB
        report("OK" if free > 20 else "WARN", f"runs dir {runs}", f"writable, {free:.0f} GB free")
    except Exception as e:
        report("FAIL", f"runs dir {runs}", str(e))
    cpus = os.cpu_count() or 1
    report("OK" if cpus >= num_workers else "WARN", f"{cpus} CPUs",
           f"--num-workers {num_workers} total over {ngpu} GPU(s)"
           + ("" if cpus >= num_workers else f"; lower to ~{max(2, cpus - 2)}"))
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        report("OK" if soft >= 4096 else "WARN", f"open-files limit {soft}",
               "" if soft >= 4096 else "raise with `ulimit -n 65536` (many DataLoader workers)")
    except Exception:
        pass
    try:
        shm = shutil.disk_usage("/dev/shm").total / 1e9
        report("OK" if shm >= 8 else "WARN", f"/dev/shm {shm:.0f} GB",
               "" if shm >= 8 else "small shared memory can crash DataLoader workers (Docker: --shm-size=32g)")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=os.environ.get("CB2_DATA"))
    ap.add_argument("--runs", type=Path, default=os.environ.get("CB2_RUNS", "runs"))
    ap.add_argument("--ngpu", type=int, default=int(os.environ.get("NGPU", "1")))
    ap.add_argument("--num-workers", type=int, default=int(os.environ.get("WORKERS", "16")))
    ap.add_argument("--models", default=os.environ.get("MODEL", "vit_b_16"))
    ap.add_argument("--partial-ok", action="store_true", help="a partial (--max-shards) download is expected")
    args = ap.parse_args()

    check_python()
    check_gpu(args.ngpu)
    if args.data:
        check_data(args.data, full=not args.partial_ok)
    else:
        report("FAIL", "dataset", "pass --data or set CB2_DATA")
    check_weights([m.strip() for m in args.models.split(",")])
    check_runs(args.runs, args.num_workers, args.ngpu)

    n_fail = sum(r[0] == "FAIL" for r in RESULTS)
    n_warn = sum(r[0] == "WARN" for r in RESULTS)
    print(f"\n{n_fail} FAIL, {n_warn} WARN")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
