"""mahalanobis_ood.py — Mahalanobis-distance OOD analysis for LOSO models (paper §5, Fig. 4c).

Lee et al. (2018): fit class-conditional Gaussians with a shared (tied)
covariance on penultimate (CLS) features of the in-distribution TRAIN set;
the M-distance of a test view is the minimum over classes. Classes are the
(task, label) pairs {split-error neg/pos, merge-error neg/pos}; one model per
modality (geom / EM), as in the paper.

Inputs come from `training.py --mode evaluate --export-features`:

    # LOSO model trained with --include-species mouse,fly,human (zebrafish held out)
    python src/neurips_training/training.py --mode evaluate --checkpoint loso_zf_best.pt \
        --blend-config ... --eval-split train --eval-species mouse,fly,human \
        --max-val-samples 50000 --export-features feats/loso_zf_train.npz
    python src/neurips_training/training.py --mode evaluate --checkpoint loso_zf_best.pt \
        --blend-config ... --eval-split test \
        --export-features feats/loso_zf_test.npz
    python scripts/training/mahalanobis_ood.py feats/loso_zf_train.npz feats/loso_zf_test.npz \
        --held-out zebrafish --out results/loso_zf_mahalanobis.csv

Output: CSV with, per (modality, task, in/out-of-distribution, distance bin):
n, mean distance, accuracy, 15-bin adaptive ECE — the curves of Fig. 4c.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "neurips_training"))
from training_utils import compute_adaptive_ece  # noqa: E402


def fit_tied_gaussians(X: np.ndarray, y: np.ndarray, ridge: float = 1e-4):
    classes = np.unique(y)
    means = np.stack([X[y == c].mean(0) for c in classes])
    centered = X - means[np.searchsorted(classes, y)]
    cov = centered.T @ centered / max(1, len(X) - len(classes))
    cov += ridge * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    prec = np.linalg.inv(cov)
    return classes, means, prec


def min_mahalanobis(X: np.ndarray, means: np.ndarray, prec: np.ndarray, chunk: int = 8192) -> np.ndarray:
    out = np.empty(len(X))
    for s in range(0, len(X), chunk):
        xc = X[s:s + chunk]
        d = np.stack([np.einsum("nd,de,ne->n", xc - m, prec, xc - m) for m in means], 1)
        out[s:s + chunk] = np.sqrt(np.maximum(d.min(1), 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train_features", type=Path)
    ap.add_argument("test_features", type=Path)
    ap.add_argument("--held-out", required=True, help="held-out species name (LOSO)")
    ap.add_argument("--n-bins", type=int, default=10, help="equal-mass distance bins")
    ap.add_argument("--out", type=Path, default=Path("results/mahalanobis.csv"))
    args = ap.parse_args()

    tr = np.load(args.train_features)
    te = np.load(args.test_features)
    rows = []
    for mod in ("geom", "em"):
        mtr = tr["modality"] == mod
        mte = te["modality"] == mod
        if not mtr.any() or not mte.any():
            continue
        Xtr = tr["features"][mtr].astype(np.float64)
        ytr = tr["task_id"][mtr] * 2 + tr["label"][mtr]
        classes, means, prec = fit_tied_gaussians(Xtr, ytr)
        Xte = te["features"][mte].astype(np.float64)
        dist = min_mahalanobis(Xte, means, prec)
        df = pd.DataFrame({
            "dist": dist, "species": te["species"][mte], "task_id": te["task_id"][mte],
            "label": te["label"][mte], "p_true": te["p_true"][mte],
        })
        df["ood"] = df["species"] == args.held_out
        edges = np.quantile(df["dist"], np.linspace(0, 1, args.n_bins + 1))
        df["bin"] = np.clip(np.searchsorted(edges, df["dist"], side="right") - 1, 0, args.n_bins - 1)
        print(f"[{mod}] mean M-distance  in-dist={df.loc[~df.ood, 'dist'].mean():.2f}  "
              f"held-out={df.loc[df.ood, 'dist'].mean():.2f}")
        for (tid, ood, b), g in df.groupby(["task_id", "ood", "bin"]):
            p, y = g["p_true"].to_numpy(), g["label"].to_numpy()
            rows.append({
                "modality": mod, "task": "split_error" if tid == 0 else "merge_error",
                "held_out": bool(ood), "bin": int(b), "n": len(g),
                "dist_mean": g["dist"].mean(),
                "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
                "ece_adaptive_15": compute_adaptive_ece(p, y, 15),
            })
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
