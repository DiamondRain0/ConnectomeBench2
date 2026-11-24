"""analyze_predictions.py — paper-style tables from exported per-view predictions.

Input: one or more parquet files written by
    python src/neurips_training/training.py --mode evaluate --eval-split test \
        --checkpoint <ckpt> --export-predictions results/<name>.parquet [--ablate ...] [--modalities ...]

Output (stdout as markdown, plus <out-dir>/summary.json and reliability CSVs):
  * Table 1  — per-sample (per-op, mean view prob @0.5) Split-Error / Merge-Error
               bAcc and per-image mask mIoU, with 95% op-clustered bootstrap CIs
  * Table 6  — the same, per species
  * Table 7  — per-species TPR / TNR
  * Table 8  — high-confidence (max(p,1-p) ≥ 0.9) coverage, TPR, TNR
  * ROC-AUC, 15-bin adaptive ECE, normalised sharpness for per-image (geom),
    per-image (EM) and per-op predictions; 12-equal-mass-bin reliability curves.

"Split Error" = end_corr head (task_id 0, false-split correction);
"Merge Error" = jn_id head (task_id 1, false-merge identification).

Example:
    python scripts/training/analyze_predictions.py \
        results/vitb_test.parquet results/vitb_test_geom_only.parquet --out-dir results/analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "neurips_training"))
from training_utils import (  # noqa: E402
    compute_adaptive_ece,
    compute_roc_auc,
    compute_sharpness,
)

TASKS = {0: "Split Error", 1: "Merge Error"}
SPECIES_ORDER = ["mouse", "fly", "zebrafish", "human"]


# ───────────────────────── bootstrap machinery ────────────────────────────

def _op_counts(df: pd.DataFrame, prob_col: str = "p_true", label_col: str = "label",
               thr: float = 0.5) -> pd.DataFrame:
    """Per-op confusion counts (views of one op form one cluster)."""
    pred = (df[prob_col].to_numpy() >= thr).astype(int)
    lab = df[label_col].to_numpy().astype(int)
    c = pd.DataFrame({
        "op_id": df["op_id"].to_numpy(),
        "tp": (pred == 1) & (lab == 1), "fn": (pred == 0) & (lab == 1),
        "tn": (pred == 0) & (lab == 0), "fp": (pred == 1) & (lab == 0),
    })
    return c.groupby("op_id", sort=False)[["tp", "fn", "tn", "fp"]].sum()


def _bacc_from_counts(tp, fn, tn, fp):
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
    return 0.5 * (tpr + tnr), tpr, tnr


def _boot_idx(n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(B, n))


def bootstrap_rates(counts: pd.DataFrame, B: int, rng: np.random.Generator) -> dict[str, tuple]:
    """Point estimate + 95% CI for bAcc / TPR / TNR, resampling ops (clusters)."""
    arr = counts[["tp", "fn", "tn", "fp"]].to_numpy(dtype=np.float64)
    if len(arr) == 0:
        nan = (np.nan, np.nan, np.nan)
        return {"bal_acc": nan, "tpr": nan, "tnr": nan, "n_ops": 0}
    tot = arr.sum(0)
    b, tpr, tnr = _bacc_from_counts(*tot)
    sums = arr[_boot_idx(len(arr), B, rng)].sum(1)  # (B, 4)
    bb, bt, bn = _bacc_from_counts(sums[:, 0], sums[:, 1], sums[:, 2], sums[:, 3])

    def ci(x):
        return tuple(np.nanpercentile(x, [2.5, 97.5])) if np.isfinite(x).any() else (np.nan, np.nan)
    return {"bal_acc": (b, *ci(bb)), "tpr": (tpr, *ci(bt)), "tnr": (tnr, *ci(bn)), "n_ops": len(arr)}


def bootstrap_mean_clustered(df: pd.DataFrame, col: str, B: int, rng: np.random.Generator) -> tuple:
    d = df[["op_id", col]].dropna()
    if len(d) == 0:
        return (np.nan, np.nan, np.nan)
    g = d.groupby("op_id", sort=False)[col].agg(["sum", "count"]).to_numpy(dtype=np.float64)
    point = g[:, 0].sum() / g[:, 1].sum()
    s = g[_boot_idx(len(g), B, rng)].sum(1)
    boots = s[:, 0] / s[:, 1]
    return (point, *np.percentile(boots, [2.5, 97.5]))


def per_op(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (op, task): mean view probability (the paper's per-sample decision)."""
    return (df.groupby(["op_id", "task_id"], sort=False)
              .agg(p_true=("p_true", "mean"), label=("label", "first"),
                   species=("species", "first"), n_views=("p_true", "size"))
              .reset_index())


# ───────────────────────────── tables ─────────────────────────────────────

def _fmt(t, pct=True):
    if t is None or not np.isfinite(t[0]):
        return "-"
    k = 100.0 if pct else 1.0
    d = 1 if pct else 3
    return f"{t[0]*k:.{d}f} [{t[1]*k:.{d}f}, {t[2]*k:.{d}f}]"


def headline(df: pd.DataFrame, B: int, rng) -> dict:
    ops = per_op(df)
    out = {}
    for tid, name in TASKS.items():
        o = ops[ops.task_id == tid]
        cnt = _op_counts(o)
        out[name] = bootstrap_rates(cnt, B, rng)
        out[name]["roc_auc"] = compute_roc_auc(o.p_true.to_numpy(), o.label.to_numpy())
        hc = o[np.maximum(o.p_true, 1 - o.p_true) >= 0.9]
        out[name]["high_conf"] = {"coverage": len(hc) / max(1, len(o)),
                                  **bootstrap_rates(_op_counts(hc), B, rng)}
    m = df[df.has_mask_gt]
    out["mIoU"] = bootstrap_mean_clustered(m, "miou", B, rng)
    best = m.groupby("op_id")["miou"].max().rename("miou_best").reset_index()
    out["mIoU_per_op_best"] = bootstrap_mean_clustered(best, "miou_best", B, rng)
    return out


def calibration(df: pd.DataFrame, n_rel_bins: int = 12) -> tuple[dict, pd.DataFrame]:
    """Adaptive ECE-15 / sharpness / AUC for per-image (geom, EM) and per-op."""
    rows, rel = {}, []
    groups = {
        "per_image_geom": df[df.modality == "geom"],
        "per_image_em": df[df.modality == "em"],
        "per_op": per_op(df),
    }
    for gname, g in groups.items():
        for tid, tname in TASKS.items():
            s = g[g.task_id == tid]
            p, y = s.p_true.to_numpy(np.float64), s.label.to_numpy(np.int64)
            rows[f"{gname}/{tname}"] = {
                "n": int(len(s)), "ece_adaptive_15": compute_adaptive_ece(p, y, 15),
                "sharpness": compute_sharpness(p), "roc_auc": compute_roc_auc(p, y),
            }
            if len(s) == 0:
                continue
            conf = np.maximum(p, 1 - p)
            correct = ((p >= 0.5).astype(int) == y)
            order = np.argsort(conf, kind="stable")
            for bi, chunk in enumerate(np.array_split(order, min(n_rel_bins, len(order)))):
                rel.append({"group": gname, "task": tname, "bin": bi, "n": len(chunk),
                            "mean_conf": float(conf[chunk].mean()),
                            "accuracy": float(correct[chunk].mean())})
    return rows, pd.DataFrame(rel)


def analyze(path: Path, B: int, seed: int, out_dir: Path) -> dict:
    df = pd.read_parquet(path)
    rng = np.random.default_rng(seed)
    name = path.stem
    res = {"file": str(path), "n_views": int(len(df)),
           "modalities": sorted(df.modality.unique().tolist()),
           "ablate": (df["ablate"].iloc[0] if "ablate" in df and len(df) else "")}
    res["overall"] = headline(df, B, rng)
    res["per_species"] = {sp: headline(df[df.species == sp], B, rng)
                          for sp in SPECIES_ORDER if (df.species == sp).any()}
    res["calibration"], rel = calibration(df)
    rel.to_csv(out_dir / f"{name}_reliability.csv", index=False)
    return res


def print_report(results: dict[str, dict]) -> None:
    print("\n## Table 1 - per-sample bAcc (%) @0.5 and mIoU, 95% op-clustered bootstrap CI\n")
    print("| Run | Split Error bAcc | Merge Error bAcc | mIoU |")
    print("|---|---|---|---|")
    for name, r in results.items():
        o = r["overall"]
        print(f"| {name} | {_fmt(o['Split Error']['bal_acc'])} | {_fmt(o['Merge Error']['bal_acc'])} "
              f"| {_fmt(o['mIoU'], pct=False)} |")
    print("\nReference (paper, test): ViT-B 97.0 / 93.0 / 0.696; ViT-L 97.2 / 92.8 / 0.678; "
          "human experts 93.0 / 84.1.")

    for name, r in results.items():
        print(f"\n## {name}\n")
        print("### Table 6 - per species\n")
        print("| Species | Split Error bAcc | Merge Error bAcc | mIoU | mIoU (per-op best) | n ops split / merge |")
        print("|---|---|---|---|---|---|")
        for sp, h in r["per_species"].items():
            print(f"| {sp} | {_fmt(h['Split Error']['bal_acc'])} | {_fmt(h['Merge Error']['bal_acc'])} "
                  f"| {_fmt(h['mIoU'], False)} | {_fmt(h['mIoU_per_op_best'], False)} "
                  f"| {h['Split Error']['n_ops']} / {h['Merge Error']['n_ops']} |")
        print("\n### Table 7 - TPR / TNR (%)\n")
        print("| Species | Task | TPR | TNR | ROC-AUC |")
        print("|---|---|---|---|---|")
        for sp, h in r["per_species"].items():
            for t in TASKS.values():
                auc = h[t]["roc_auc"]
                print(f"| {sp} | {t} | {_fmt(h[t]['tpr'])} | {_fmt(h[t]['tnr'])} "
                      f"| {auc:.4f} |" if np.isfinite(auc) else
                      f"| {sp} | {t} | {_fmt(h[t]['tpr'])} | {_fmt(h[t]['tnr'])} | - |")
        print("\n### Table 8 - high confidence (max(p,1-p) >= 0.9)\n")
        print("| Species | Task | Coverage (%) | TPR | TNR |")
        print("|---|---|---|---|---|")
        for sp, h in r["per_species"].items():
            for t in TASKS.values():
                hc = h[t]["high_conf"]
                print(f"| {sp} | {t} | {hc['coverage']*100:.1f} | {_fmt(hc['tpr'])} | {_fmt(hc['tnr'])} |")
        print("\n### Calibration (15-bin adaptive ECE, normalised sharpness)\n")
        print("| Granularity / task | n | aECE | sharpness | ROC-AUC |")
        print("|---|---|---|---|---|")
        for k, v in r["calibration"].items():
            print(f"| {k} | {v['n']} | {v['ece_adaptive_15']:.4f} | {v['sharpness']:.4f} | {v['roc_auc']:.4f} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("results/analysis"))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {p.stem: analyze(p, args.bootstrap, args.seed, args.out_dir) for p in args.predictions}
    print_report(results)
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=1, default=lambda x: float(x) if np.isscalar(x) else str(x))
    print(f"\nwrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
