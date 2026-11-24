"""smoke_test.py — end-to-end CPU check of the training stack on tiny data (~1 minute).

Builds a fake {train,val,test} parquet tree from the 20-row `demo.parquet`
(downloaded from HF, 6.6 MB) and runs: data loading for both tasks and both
modalities, 2 training steps of a randomly initialised ViT-B, checkpointing,
resume, test evaluation with prediction + feature export, an ablation, and
the analysis script. It checks wiring, not accuracy.

    python scripts/training/smoke_test.py [--demo path/to/demo.parquet] [--keep]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "neurips_training" / "training.py"
ANALYZE = ROOT / "scripts" / "training" / "analyze_predictions.py"


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=Path, default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    work = Path(tempfile.mkdtemp(prefix="cb2_smoke_"))
    try:
        demo = args.demo
        if demo is None:
            from huggingface_hub import hf_hub_download
            demo = Path(hf_hub_download("jeffbbrown2/ConnectomeBench2", "demo.parquet",
                                        repo_type="dataset", revision="v2"))
        t = pq.read_table(demo)
        data = work / "data"
        for split in ("train", "val", "test"):
            (data / split).mkdir(parents=True)
            pq.write_table(t, data / split / f"{split}-00000.parquet", row_group_size=10)

        cfg = work / "smoke.yaml"
        cfg.write_text(
            f"parquet:\n  path: {data.as_posix()}\n"
            "filter:\n  include_species: [mouse, fly, zebrafish, human]\n"
            "blend:\n  strategy: sqrt\n  cell_axis: kind\n"
            "class_balance: true\n"
        )
        ck = work / "ckpt"
        common = [sys.executable, str(TRAIN), "--blend-config", str(cfg), "--device", "cpu",
                  "--num-workers", "0", "--batch-size", "8", "--init-source", "none",
                  "--checkpoint-dir", str(ck), "--checkpoint-name", "smoke"]
        run(common + ["--epochs", "1", "--max-samples", "4", "--max-val-samples", "6",
                      "--shuffle-buffer", "16", "--warmup-epochs", "0"])
        run(common + ["--epochs", "2", "--max-samples", "4", "--max-val-samples", "6",
                      "--warmup-epochs", "0", "--resume", str(ck / "smoke_last.pt")])
        res = work / "res"
        run(common + ["--mode", "evaluate", "--checkpoint", str(ck / "smoke_best.pt"),
                      "--eval-split", "test", "--export-predictions", str(res / "smoke_test.parquet"),
                      "--export-features", str(res / "smoke_test_feats.npz")])
        run(common + ["--mode", "evaluate", "--checkpoint", str(ck / "smoke_best.pt"),
                      "--eval-split", "test", "--modalities", "geom", "--ablate", "masks",
                      "--export-predictions", str(res / "smoke_test_geom_nomask.parquet")])
        run([sys.executable, str(ANALYZE), str(res / "smoke_test.parquet"),
             str(res / "smoke_test_geom_nomask.parquet"), "--out-dir", str(res / "analysis"),
             "--bootstrap", "50"])
        print(f"\nSMOKE TEST PASSED (workdir {work})")
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
