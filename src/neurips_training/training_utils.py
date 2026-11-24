"""training_utils.py — losses, optimizer, scheduler, AMP, metrics, eval, DDP, viz.

All the plumbing between `(model, batch) → loss → step` and
`(model, val_loader) → wandb-ready metrics dict`. No model defs (model.py),
no data defs (data.py), no main loop (training.py).

Vendored from upstream:
  - `binary_softmax_ce_dice_invariant_loss`, `compute_iou_invariant_batched`,
    `_soft_dice_per_sample` — `src/training/split_mask_training.py`
  - `aggregate_mean_prob`, `aggregate_majority_vote` —
    `scripts/model-post-training/resnet/evaluate.py`
  - `compute_ece`, `compute_roc_auc`, `compute_multitask_losses`, `_aggregate_cls`,
    `run_eval`, `_check_pcie_gen5_or_die`, viz helpers —
    `scripts/model-post-training/neurips_finetune_unified.py`

NO LLRD — only 2-group AdamW. Per Tim 2026-05-07.
"""
from __future__ import annotations

import datetime
import math
import os
import random
import subprocess
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, get_worker_info

# Constants — must match data.py / model.py exactly.
TASK_ENDPOINT = 0
TASK_JUNCTION = 1
MODALITY_GEOM = 0
MODALITY_EM = 1
CH_SILH, CH_DEPTH, CH_NX, CH_NY, CH_NZ, CH_MASK_A, CH_MASK_B = range(7)


# ───────────────── vendored: mask loss + IoU helpers ─────────────────────

def _soft_dice_per_sample(p: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample soft Dice. Vendored from split_mask_training.py."""
    inter = (p * t).sum(dim=(1, 2))
    denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    return 1 - (2 * inter + eps) / (denom + eps)


def binary_softmax_ce_dice_invariant_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ce_weight: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Vectorised order-invariant CE + softmax Dice on foreground pixels.

    Per sample: try both channel assignments (orig + swap), pick min combined loss.
    Non-split samples (no target==2) use single-channel dice; split samples
    average two-channel dice. Samples with no foreground contribute 0.

    reduction:
      "mean" — fg-weighted scalar mean (training default).
      "none" — per-sample tensor (0 for no-fg samples, matching old eval behavior;
               used by run_eval / compute_multitask_losses to gate per-sample).

    Vendored verbatim from split_mask_training.py::binary_softmax_ce_dice_invariant_loss.
    """
    eps = 1e-6
    probs = torch.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    p0, p1 = probs[:, 0], probs[:, 1]
    lp0, lp1 = log_probs[:, 0], log_probs[:, 1]

    ta = (targets == 1).to(logits.dtype)
    tb = (targets == 2).to(logits.dtype)
    fg = ta + tb

    fg_count = fg.sum(dim=(-1, -2))
    tb_count = tb.sum(dim=(-1, -2))
    has_fg = fg_count > 0
    has_b = tb_count > 0

    fg_count_safe = fg_count.clamp_min(1)
    ce_orig = (-(lp0 * ta + lp1 * tb)).sum(dim=(-1, -2)) / fg_count_safe
    ce_swap = (-(lp1 * ta + lp0 * tb)).sum(dim=(-1, -2)) / fg_count_safe

    p0_fg = (p0 * fg).sum(dim=(-1, -2))
    p1_fg = (p1 * fg).sum(dim=(-1, -2))
    ta_sum = ta.sum(dim=(-1, -2))
    tb_sum = tb.sum(dim=(-1, -2))
    p0_ta = (p0 * ta).sum(dim=(-1, -2))
    p1_ta = (p1 * ta).sum(dim=(-1, -2))
    p0_tb = (p0 * tb).sum(dim=(-1, -2))
    p1_tb = (p1 * tb).sum(dim=(-1, -2))

    dice_a_orig = (2 * p0_ta + eps) / (p0_fg + ta_sum + eps)
    dice_b_orig = (2 * p1_tb + eps) / (p1_fg + tb_sum + eps)
    dice_a_swap = (2 * p1_ta + eps) / (p1_fg + ta_sum + eps)
    dice_b_swap = (2 * p0_tb + eps) / (p0_fg + tb_sum + eps)

    nonsplit_orig = ce_weight * ce_orig + (1 - ce_weight) * (1 - dice_a_orig)
    nonsplit_swap = ce_weight * ce_swap + (1 - ce_weight) * (1 - dice_a_swap)
    split_orig = ce_weight * ce_orig + (1 - ce_weight) * (1 - (dice_a_orig + dice_b_orig) / 2)
    split_swap = ce_weight * ce_swap + (1 - ce_weight) * (1 - (dice_a_swap + dice_b_swap) / 2)

    per_sample = torch.where(
        has_b,
        torch.minimum(split_orig, split_swap),
        torch.minimum(nonsplit_orig, nonsplit_swap),
    )
    if reduction == "none":
        return torch.where(has_fg, per_sample, torch.zeros_like(per_sample))
    has_fg_f = has_fg.to(per_sample.dtype)
    return (per_sample * has_fg_f).sum() / has_fg_f.sum().clamp_min(1)


def compute_iou_invariant_batched(
    logits: torch.Tensor, targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched, GPU-resident order-invariant IoU.

    logits:  (B, 2, H, W)
    targets: (B, H, W) ∈ {0=BG, 1=A, 2=B}.

    Returns (iou_a, iou_b) — each (B,) — for the channel assignment with
    higher mIoU per sample. NaN where the union under the fg mask is empty.

    Vendored verbatim from split_mask_training.py.
    """
    fg = targets > 0
    argmax = logits.argmax(dim=1)
    preds_orig = argmax + 1
    preds_swap = 2 - argmax

    def _iou(preds: torch.Tensor, c: int) -> torch.Tensor:
        p = (preds == c) & fg
        t = (targets == c) & fg
        inter = (p & t).sum(dim=(-1, -2)).float()
        union = (p | t).sum(dim=(-1, -2)).float()
        nan = torch.full_like(union, float("nan"))
        return torch.where(union > 0, inter / union, nan)

    iou1_a = _iou(preds_orig, 1)
    iou1_b = _iou(preds_orig, 2)
    iou2_a = _iou(preds_swap, 1)
    iou2_b = _iou(preds_swap, 2)
    miou1 = torch.nanmean(torch.stack([iou1_a, iou1_b], dim=-1), dim=-1)
    miou2 = torch.nanmean(torch.stack([iou2_a, iou2_b], dim=-1), dim=-1)
    # Force NaN to -1 before compare so a valid mIoU always wins over a NaN mIoU.
    pick1 = torch.nan_to_num(miou1, nan=-1.0) >= torch.nan_to_num(miou2, nan=-1.0)
    iou_a = torch.where(pick1, iou1_a, iou2_a)
    iou_b = torch.where(pick1, iou1_b, iou2_b)
    return iou_a, iou_b


# ───────────────── vendored: per-op aggregators ──────────────────────────

def aggregate_mean_prob(p_true: np.ndarray) -> bool:
    """Average P(True) across views, threshold at 0.5.

    Vendored from resnet/evaluate.py.
    """
    return bool(float(p_true.mean()) > 0.5)


def aggregate_majority_vote(p_true: np.ndarray) -> bool:
    """Per-view binary vote, majority wins. Tie-break: most confident boundary.

    Vendored from resnet/evaluate.py.
    """
    votes = (p_true > 0.5).astype(int)
    true_count = int(votes.sum())
    if true_count * 2 == len(votes):
        best_idx = int(np.abs(p_true - 0.5).argmax())
        return bool(p_true[best_idx] > 0.5)
    return bool(true_count * 2 > len(votes))


# ───────────────────── per-image metrics ─────────────────────────────────

def _per_class_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    """Mean of per-class recall (= balanced accuracy on binary tasks)."""
    if len(labels) == 0:
        return float("nan")
    accs = []
    for c in np.unique(labels):
        mask = labels == c
        if mask.sum() > 0:
            accs.append(float((preds[mask] == c).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def cls_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """{acc, bal_acc, n} on integer label arrays."""
    if len(labels) == 0:
        return {"acc": float("nan"), "bal_acc": float("nan"), "n": 0}
    return {
        "acc": float((preds == labels).mean()),
        "bal_acc": _per_class_accuracy(labels, preds),
        "n": int(len(labels)),
    }


def compute_ece(probs_true: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width binned Expected Calibration Error (ECE-15 default).

    `probs_true` = P(class==1). Confidence = max(p, 1-p) — the standard
    binary-ECE formulation regardless of which class the model predicted.
    """
    if len(labels) == 0:
        return float("nan")
    probs_true = probs_true.astype(np.float64)
    labels_np = labels.astype(np.int64)
    preds = (probs_true >= 0.5).astype(np.int64)
    confs = np.where(preds == 1, probs_true, 1.0 - probs_true)
    correct = (preds == labels_np).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Last bin is inclusive on the right (catches confidence == 1.0 exactly).
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = float(confs[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += abs(bin_conf - bin_acc) * float(mask.sum()) / float(n)
    return float(ece)


def compute_roc_auc(probs_true: np.ndarray, labels: np.ndarray) -> float:
    """Binary ROC AUC. NaN when only one class is present."""
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels.astype(np.int64), probs_true.astype(np.float64)))


def nanmean(xs: list[float]) -> float:
    """Mean ignoring NaN floats. Returns NaN on empty / all-NaN."""
    cleaned = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(cleaned)) if cleaned else float("nan")


def _aggregate_cls(
    op_ids: list[str], labels: np.ndarray, probs_true: np.ndarray,
) -> dict[str, Any]:
    """Per-op aggregation via mean_prob (headline) and majority_vote.

    Returns scalar metrics + the aggregated arrays so callers can compute
    additional metrics (ECE / ROC AUC) on the per-op level.

    Per-op invariant (preserves unified-script line 1672): if views in an op
    disagree on the LABEL (which should never happen by data construction),
    skip the op rather than corrupt the aggregate.
    """
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for op, lab, pt in zip(op_ids, labels.tolist(), probs_true.tolist()):
        grouped[op].append((int(lab), float(pt)))
    agg_labels: list[int] = []
    agg_preds_mean: list[int] = []
    agg_preds_majority: list[int] = []
    agg_mean_probs: list[float] = []
    for op, pairs in grouped.items():
        labs = [p[0] for p in pairs]
        if len(set(labs)) > 1:
            continue
        agg_labels.append(labs[0])
        pt_arr = np.array([p[1] for p in pairs])
        agg_preds_mean.append(int(aggregate_mean_prob(pt_arr)))
        agg_preds_majority.append(int(aggregate_majority_vote(pt_arr)))
        agg_mean_probs.append(float(pt_arr.mean()))
    labels_np = np.array(agg_labels, dtype=np.int64)
    mean_np = np.array(agg_preds_mean, dtype=np.int64)
    maj_np = np.array(agg_preds_majority, dtype=np.int64)
    mean_probs_np = np.array(agg_mean_probs, dtype=np.float32)
    return {
        "mean_prob_acc": float((mean_np == labels_np).mean()) if len(labels_np) else float("nan"),
        "mean_prob_bal_acc": _per_class_accuracy(labels_np, mean_np),
        "majority_acc": float((maj_np == labels_np).mean()) if len(labels_np) else float("nan"),
        "n_ops": int(len(labels_np)),
        "labels_np": labels_np,
        "mean_probs_np": mean_probs_np,
    }


# ────────────────── compute_multitask_losses ─────────────────────────────

def compute_multitask_losses(
    endpoint_logits: torch.Tensor,
    junction_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    synapse_logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    label_smoothing: float = 0.1,
    mask_ce_weight: float = 0.5,
    loss_scale_endpoint: float = 1.0,
    loss_scale_junction: float = 1.0,
    loss_scale_mask: float = 1.0,
    loss_scale_synapse: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Per-task masked CE + invariant CE+Dice mask loss + weighted total.

    Vendored from unified-script `compute_multitask_losses` (line 1534) with
    the cfg dataclass replaced by explicit kwargs (avoids cross-module coupling
    to training.py's MultiTaskConfig).

    NO `.any()` / no boolean indexing → no GPU↔CPU sync. Per-task loss is
    `(per_sample_CE * task_mask).sum() / task_mask.sum().clamp_min(1)`.
    """
    task_id = batch["task_id"]
    cls_label = batch["cls_label"]
    has_mask_gt = batch["has_mask_gt"]
    mask_label = batch["mask_label"]
    has_synapse = batch["has_synapse_label"]
    synapse_label = batch["synapse_label"]

    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

    is_endpoint = task_id == TASK_ENDPOINT
    is_junction = task_id == TASK_JUNCTION

    losses: dict[str, torch.Tensor] = {}
    cls_label_long = cls_label.long()

    ep_per = ce(endpoint_logits, cls_label_long)
    ep_mask = is_endpoint.to(dtype=ep_per.dtype)
    losses["L_endpoint"] = (ep_per * ep_mask).sum() / ep_mask.sum().clamp_min(1)

    jn_per = ce(junction_logits, cls_label_long)
    jn_mask = is_junction.to(dtype=jn_per.dtype)
    losses["L_junction"] = (jn_per * jn_mask).sum() / jn_mask.sum().clamp_min(1)

    syn_per = ce(synapse_logits, synapse_label.long())
    syn_mask = has_synapse.to(dtype=syn_per.dtype)
    losses["L_synapse"] = (syn_per * syn_mask).sum() / syn_mask.sum().clamp_min(1)

    # Gate L_mask by has_mask_gt: only split_edit rows with both single+dual
    # masks have real GT (built from the dual-mesh ch5,6 in data.py).
    # reduction="none" preserves the zero-fg-sample behavior so EM rows
    # (mask_label all zeros) still contribute 0.
    mask_per = binary_softmax_ce_dice_invariant_loss(
        mask_logits, mask_label, ce_weight=mask_ce_weight, reduction="none",
    )
    mask_gate = has_mask_gt.to(dtype=mask_per.dtype)
    losses["L_mask"] = (mask_per * mask_gate).sum() / mask_gate.sum().clamp_min(1)

    losses["L_total"] = (
        loss_scale_endpoint * losses["L_endpoint"]
        + loss_scale_junction * losses["L_junction"]
        + loss_scale_mask * losses["L_mask"]
        + loss_scale_synapse * losses["L_synapse"]
    )
    return losses


# ─────────────────────── optimizer + scheduler ───────────────────────────

def build_optimizer(
    model: nn.Module,
    *,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    backbone_lr_scale: float = 0.1,
    peak_lr_multiplier: float = 1.0,
) -> torch.optim.AdamW:
    """2-group AdamW (no LLRD).

    Group 0 — backbone: torchvision-ViT params @ lr * peak_lr_multiplier * backbone_lr_scale
    Group 1 — everything else (cls heads, mask decoder, mask view embedding,
              conv_proj_em) @ lr * peak_lr_multiplier

    (The original release enumerated head modules by hand and silently left
    `mask_head.view_emb` out of both groups, so it was never trained.)
    """
    base_lr = lr * peak_lr_multiplier
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    head_params = [p for p in model.parameters()
                   if id(p) not in backbone_ids and p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr * backbone_lr_scale},
            {"params": head_params, "lr": base_lr},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(
    optim: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    warmup_start_factor: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """LinearLR warmup → CosineAnnealingLR. T_max = total - warmup."""
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optim,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, total_steps - warmup_steps),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optim, [warmup, cosine], milestones=[warmup_steps],
    )


# ───────────────────────── AMP step ──────────────────────────────────────

def optimizer_step_amp(
    loss: torch.Tensor,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model: nn.Module,
    max_grad_norm: float,
) -> float:
    """One full AMP step: scale → backward → unscale → clip → step → update → sched.step.

    Works for bf16 (where GradScaler is a no-op) and fp16. Returns the
    PRE-clip gradient norm (post-unscale) for wandb logging.
    """
    scaler.scale(loss).backward()
    scaler.unscale_(optim)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optim)
    scaler.update()
    scheduler.step()
    return float(grad_norm)


# ────────────── seed helpers + dataloader worker init ────────────────────

def set_seeds(seed: int, rank: int = 0) -> None:
    """Set torch/numpy/random seeds. Per-rank offset so DDP ranks diverge."""
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Per-DataLoader-worker RNG seeding. PyTorch sets a base seed for each
    worker; we propagate it to numpy + python's `random`."""
    info = get_worker_info()
    if info is None:
        return
    seed = int(info.seed) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# ──────────────────── PCIe Gen5 guard (Modal H100) ───────────────────────

def check_pcie_gen5_or_die() -> None:
    """Hard-fail if H100 is on PCIe Gen<5 (~2× slower).

    Skipped on non-H100 GPUs. Vendored from unified-script (line 1698).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception as e:
        print(f"[pcie-guard] nvidia-smi failed ({e}); skipping check", flush=True)
        return

    print(f"[pcie-guard] nvidia-smi: {out}", flush=True)
    fields = [f.strip() for f in out.splitlines()[0].split(",")]
    name = fields[0]
    if "H100" not in name:
        print(f"[pcie-guard] skip: GPU={name} (Gen5 check is H100-only)", flush=True)
        return
    try:
        gen_current = int(fields[1])
    except (ValueError, IndexError):
        print(f"[pcie-guard] could not parse PCIe gen from {fields!r}; skipping", flush=True)
        return
    if gen_current < 5:
        msg = (
            f"[pcie-guard] FAIL: GPU={name} on PCIe Gen{gen_current} "
            f"(need Gen5 for full host↔GPU bandwidth). Bailing so Modal reschedules."
        )
        print(msg, flush=True)
        raise RuntimeError(msg)
    print(f"[pcie-guard] OK: PCIe Gen{gen_current}", flush=True)


# ──────────────────────── DDP helpers ────────────────────────────────────

def ddp_active(world_size: int) -> bool:
    """True iff DDP is initialised AND world_size > 1."""
    return (
        int(world_size) > 1
        and dist.is_available()
        and dist.is_initialized()
    )


def ddp_unwrap(model: nn.Module) -> nn.Module:
    """Return underlying module; no-op if not wrapped in DDP."""
    return model.module if hasattr(model, "module") else model


def ddp_all_reduce_metrics(
    d: dict[str, Any], world_size: int, op: str = "avg",
) -> dict[str, Any]:
    """All-reduce scalar metrics across DDP ranks. Non-numeric values pass through.

    op="avg" divides the sum by world_size; op="sum" leaves it.
    No-op (returns input) when DDP isn't active.
    """
    if not ddp_active(world_size):
        return d
    out: dict[str, Any] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for k, v in d.items():
        if not isinstance(v, (int, float)):
            out[k] = v
            continue
        if isinstance(v, float) and math.isnan(v):
            out[k] = v
            continue
        t = torch.tensor(float(v), dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if op == "avg":
            t /= world_size
        out[k] = t.item()
    return out


def setup_ddp(
    rank: int,
    world_size: int,
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
    backend: str = "nccl",
) -> None:
    """Initialise NCCL process group. Idempotent against repeat env-var sets.

    Validation runs on rank 0 only while the other ranks wait at a barrier. On slow
    storage one validation pass can exceed NCCL's
    default 10-minute collective timeout, and the watchdog then aborts the job.
    The timeout is therefore raised (default 120 min, override with CB2_DDP_TIMEOUT_MIN).
    """
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    timeout_min = int(os.environ.get("CB2_DDP_TIMEOUT_MIN", "120"))
    if not dist.is_initialized():
        dist.init_process_group(backend, rank=int(rank), world_size=int(world_size),
                                timeout=datetime.timedelta(minutes=timeout_min))


def teardown_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ───────────────────────────── run_eval ──────────────────────────────────

_MODALITY_TAG = {MODALITY_GEOM: "geom", MODALITY_EM: "em"}
_HEAD_TAG = {TASK_ENDPOINT: "end_corr", TASK_JUNCTION: "jn_id"}


def compute_adaptive_ece(probs_true: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Adaptive (equal-mass) ECE [Nixon et al. 2019], binary, confidence = max(p, 1-p).

    The paper reports 15-bin adaptive ECE (Fig. 4a)."""
    n = len(labels)
    if n == 0:
        return float("nan")
    p = probs_true.astype(np.float64)
    preds = (p >= 0.5).astype(np.int64)
    conf = np.where(preds == 1, p, 1.0 - p)
    correct = (preds == labels.astype(np.int64)).astype(np.float64)
    order = np.argsort(conf, kind="stable")
    ece = 0.0
    for chunk in np.array_split(order, min(n_bins, n)):
        if len(chunk) == 0:
            continue
        ece += abs(conf[chunk].mean() - correct[chunk].mean()) * len(chunk) / n
    return float(ece)


def compute_sharpness(probs_true: np.ndarray) -> float:
    """Normalised sharpness Var(p) / [p̄(1-p̄)] (paper Fig. 4a)."""
    if len(probs_true) == 0:
        return float("nan")
    p = probs_true.astype(np.float64)
    pb = p.mean()
    denom = pb * (1 - pb)
    return float(p.var() / denom) if denom > 0 else float("nan")


def _tpr_tnr(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    pos, neg = labels == 1, labels == 0
    tpr = float((preds[pos] == 1).mean()) if pos.any() else float("nan")
    tnr = float((preds[neg] == 0).mean()) if neg.any() else float("nan")
    return tpr, tnr


def binary_metrics(probs_true: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """bAcc / acc / TPR / TNR / ROC-AUC / ECE-15 / adaptive ECE-15 / sharpness / n."""
    labels = np.asarray(labels, dtype=np.int64)
    probs_true = np.asarray(probs_true, dtype=np.float64)
    preds = (probs_true >= threshold).astype(np.int64)
    tpr, tnr = _tpr_tnr(labels, preds)
    return {
        "acc": float((preds == labels).mean()) if len(labels) else float("nan"),
        "bal_acc": _per_class_accuracy(labels, preds),
        "tpr": tpr,
        "tnr": tnr,
        "roc_auc": compute_roc_auc(probs_true, labels),
        "ece": compute_ece(probs_true, labels),
        "ece_adaptive": compute_adaptive_ece(probs_true, labels),
        "sharpness": compute_sharpness(probs_true),
        "n": float(len(labels)),
    }


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    label_smoothing: float = 0.1,
    mask_ce_weight: float = 0.5,
    return_features: bool = False,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_batches: int | None = None,
    progress: bool = False,
):
    """Run the model over `loader` and return a per-view prediction table.

    Columns: op_id (= combined_sample_hash, one row of the dataset), species,
    kind, sample_type, modality, view_idx, task_id, label, p_true (the routed
    head's P(class 1)), cls_loss, p_endpoint, p_junction, has_synapse_label,
    synapse_label, p_synapse, synapse_loss, has_mask_gt, iou_a, iou_b, miou,
    mask_loss. Optionally also returns (N, D) float16 CLS features (row-aligned).
    """
    import pandas as pd

    model.eval()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")
    cols: dict[str, list] = defaultdict(list)
    feats: list[np.ndarray] = []
    amp_on = use_amp and device.type == "cuda"

    it = loader
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(loader, desc="eval", dynamic_ncols=True)
        except ImportError:
            pass

    with torch.no_grad():
        for b_i, batch in enumerate(it):
            if max_batches is not None and b_i >= max_batches:
                break
            x = batch["input"].to(device, non_blocking=True)
            modality = batch["modality"].to(device, non_blocking=True)
            view_idx = batch["view_idx"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_on):
                out = model(x, modality, view_idx=view_idx, return_features=return_features)
            ep_logits, jn_logits, mask_logits, syn_logits = (o.float() for o in out[:4])
            task_id = batch["task_id"].to(device)
            label = batch["cls_label"].to(device)
            syn_label = batch["synapse_label"].to(device)

            routed_logits = torch.where((task_id == TASK_JUNCTION)[:, None], jn_logits, ep_logits)
            p_ep = torch.softmax(ep_logits, -1)[:, 1]
            p_jn = torch.softmax(jn_logits, -1)[:, 1]
            p_syn = torch.softmax(syn_logits, -1)[:, 1]
            p_true = torch.softmax(routed_logits, -1)[:, 1]
            cls_loss = ce(routed_logits, label)
            syn_loss = ce(syn_logits, syn_label)

            bsz = x.shape[0]
            iou_a = np.full(bsz, np.nan, dtype=np.float32)
            iou_b = np.full(bsz, np.nan, dtype=np.float32)
            mask_loss = np.full(bsz, np.nan, dtype=np.float32)
            has_mask = batch["has_mask_gt"]
            if has_mask.any():
                idx_cpu = has_mask.nonzero(as_tuple=True)[0]
                idx = idx_cpu.to(device)
                sub_logits = mask_logits.index_select(0, idx)
                sub_t = batch["mask_label"].index_select(0, idx_cpu).to(device)
                ml = binary_softmax_ce_dice_invariant_loss(
                    sub_logits, sub_t, ce_weight=mask_ce_weight, reduction="none")
                ia, ib = compute_iou_invariant_batched(sub_logits, sub_t)
                sel = idx_cpu.numpy()
                iou_a[sel] = ia.cpu().numpy()
                iou_b[sel] = ib.cpu().numpy()
                mask_loss[sel] = ml.cpu().numpy()

            stacked = torch.stack([p_true, cls_loss, p_ep, p_jn, p_syn, syn_loss]).cpu().numpy()
            cols["op_id"] += list(batch["op_id"])
            cols["species"] += list(batch["species"])
            cols["kind"] += list(batch["kind"])
            cols["sample_type"] += list(batch["sample_type"])
            cols["modality"] += [_MODALITY_TAG[int(m)] for m in batch["modality"].tolist()]
            cols["view_idx"] += batch["view_idx"].tolist()
            cols["task_id"] += batch["task_id"].tolist()
            cols["label"] += batch["cls_label"].tolist()
            cols["p_true"] += stacked[0].tolist()
            cols["cls_loss"] += stacked[1].tolist()
            cols["p_endpoint"] += stacked[2].tolist()
            cols["p_junction"] += stacked[3].tolist()
            cols["p_synapse"] += stacked[4].tolist()
            cols["synapse_loss"] += stacked[5].tolist()
            cols["has_synapse_label"] += batch["has_synapse_label"].tolist()
            cols["synapse_label"] += batch["synapse_label"].tolist()
            cols["has_mask_gt"] += has_mask.tolist()
            cols["iou_a"] += iou_a.tolist()
            cols["iou_b"] += iou_b.tolist()
            cols["mask_loss"] += mask_loss.tolist()
            if return_features:
                feats.append(out[4].float().cpu().numpy().astype(np.float16))

    df = pd.DataFrame(cols)
    if len(df):
        df["miou"] = df[["iou_a", "iou_b"]].mean(axis=1, skipna=True)
    else:
        df["miou"] = []
    if return_features:
        F_ = np.concatenate(feats, 0) if feats else np.zeros((0, 0), np.float16)
        return df, F_
    return df


def per_op_table(df, prob_col: str = "p_true", label_col: str = "label",
                 extra_keys: tuple[str, ...] = ("species", "kind", "sample_type")):
    """Aggregate per-view rows → one row per (op, task): mean view probability.

    Ops whose views disagree on the label are dropped (should never happen)."""
    import pandas as pd
    if len(df) == 0:
        return pd.DataFrame(columns=["op_id", "task_id", label_col, prob_col, "n_views", *extra_keys])
    g = df.groupby(["op_id", "task_id"], sort=False)
    agg = g.agg(**{prob_col: (prob_col, "mean"),
                   "_lab_min": (label_col, "min"), "_lab_max": (label_col, "max"),
                   "n_views": (prob_col, "size"),
                   **{k: (k, "first") for k in extra_keys}}).reset_index()
    agg = agg[agg["_lab_min"] == agg["_lab_max"]].rename(columns={"_lab_min": label_col})
    return agg.drop(columns=["_lab_max"])


def metrics_from_predictions(df, split_tag: str = "val") -> dict[str, float]:
    """Flat metrics dict from the per-view table (wandb/JSON-ready).

    Keys: `{head}[/{slice}]/{split}_{metric}_{per_image|mean_prob}` where head
    ∈ {end_corr, jn_id, synapse}, slice ∈ {species, species_kind, modality,
    modality/species_kind}; mask: `mask[/{slice}]/{split}_mIoU_inv` etc.
    The headline numbers of the paper are `end_corr/{split}_bal_acc_mean_prob`
    (Split Error bAcc), `jn_id/{split}_bal_acc_mean_prob` (Merge Error bAcc)
    and `mask/{split}_mIoU_inv`.
    """
    out: dict[str, float] = {}
    s = split_tag
    if len(df) == 0:
        return out
    df = df.copy()
    df["species_kind"] = df["species"].astype(str) + "_" + df["kind"].astype(str)

    def _cls_block(prefix: str, sub, prob_col: str, label_col: str):
        slices = [(None, sub)]
        for col in ("species", "species_kind", "modality"):
            for key, part in sub.groupby(col, sort=True):
                slices.append((str(key), part))
        for (mod, sk), part in sub.groupby(["modality", "species_kind"], sort=True):
            slices.append((f"{mod}/{sk}", part))
        for key, part in slices:
            tag = prefix if key is None else f"{prefix}/{key}"
            pi = binary_metrics(part[prob_col].to_numpy(), part[label_col].to_numpy())
            for m, v in pi.items():
                out[f"{tag}/{s}_{m}_per_image"] = v
            op = per_op_table(part.rename(columns={prob_col: "_p", label_col: "_l"}),
                              prob_col="_p", label_col="_l", extra_keys=())
            po = binary_metrics(op["_p"].to_numpy(), op["_l"].to_numpy())
            for m, v in po.items():
                out[f"{tag}/{s}_{m}_mean_prob"] = v
            # majority vote (per-view vote, tie → most confident view)
            maj = part.groupby("op_id")[prob_col].apply(lambda a: aggregate_majority_vote(a.to_numpy()))
            lab = part.groupby("op_id")[label_col].first()
            out[f"{tag}/{s}_acc_majority"] = float((maj.astype(int) == lab.astype(int)).mean())
            out[f"{tag}/{s}_loss"] = float(part["_loss"].mean()) if "_loss" in part else float("nan")
            # legacy aliases used by the training-loop printout / checkpoint selection
            out[f"{tag}/{s}_n_mean_prob"] = po["n"]

    for tid, prefix in _HEAD_TAG.items():
        sub = df[df["task_id"] == tid].assign(_loss=lambda d: d["cls_loss"])
        if len(sub):
            _cls_block(prefix, sub, "p_true", "label")
    syn = df[df["has_synapse_label"]].assign(_loss=lambda d: d["synapse_loss"])
    if len(syn):
        _cls_block("synapse", syn, "p_synapse", "synapse_label")

    m = df[df["has_mask_gt"]]
    if len(m):
        for key, part in [(None, m), *[(str(k), p) for k, p in m.groupby("species")],
                          *[(str(k), p) for k, p in m.groupby("species_kind")]]:
            tag = "mask" if key is None else f"mask/{key}"
            out[f"{tag}/{s}_mIoU_inv"] = float(np.nanmean(part["miou"])) if part["miou"].notna().any() else float("nan")
            out[f"{tag}/{s}_IoU_A"] = float(np.nanmean(part["iou_a"])) if part["iou_a"].notna().any() else float("nan")
            out[f"{tag}/{s}_IoU_B"] = float(np.nanmean(part["iou_b"])) if part["iou_b"].notna().any() else float("nan")
            out[f"{tag}/{s}_loss"] = float(np.nanmean(part["mask_loss"]))
            out[f"{tag}/{s}_n_samples"] = float(len(part))
            # per-op best view (paper Table 6 also reports per-op best mIoU)
            best = part.groupby("op_id")["miou"].max()
            out[f"{tag}/{s}_mIoU_inv_per_op_best"] = float(np.nanmean(best)) if best.notna().any() else float("nan")
    return out


def run_eval(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    label_smoothing: float = 0.1,
    mask_ce_weight: float = 0.5,
    loss_scale_endpoint: float = 1.0,
    loss_scale_junction: float = 1.0,
    loss_scale_mask: float = 1.0,
    loss_scale_synapse: float = 0.25,
    return_raw: bool = False,
    split_tag: str = "val",
    progress: bool = False,
):
    """Full eval → flat metrics dict (and the per-view prediction table if `return_raw`).

    Per-task gating: each cls head is only scored on rows routed to it; the
    mask head on rows with GT; the synapse head on rows with a synapse label.
    Per-image and per-op (mean view probability, threshold 0.5) variants.
    """
    df = collect_predictions(model, val_loader, device, label_smoothing=label_smoothing,
                             mask_ce_weight=mask_ce_weight, progress=progress)
    out = metrics_from_predictions(df, split_tag=split_tag)
    s = split_tag

    def _g(k):
        v = out.get(k, float("nan"))
        return 0.0 if (isinstance(v, float) and math.isnan(v)) else v

    out[f"{s}/total_loss"] = (
        loss_scale_endpoint * _g(f"end_corr/{s}_loss")
        + loss_scale_junction * _g(f"jn_id/{s}_loss")
        + loss_scale_mask * _g(f"mask/{s}_loss")
        + loss_scale_synapse * _g(f"synapse/{s}_loss")
    )
    if return_raw:
        return out, df
    return out


# ─────────────────────── wandb visualisation ─────────────────────────────

# Match split_mask_training.py palette: BG=black, A=blue, B=orange.
_CLASS_COLORS = np.array([[0, 0, 0], [31, 119, 180], [255, 127, 14]], dtype=np.uint8)


def make_viz_row(img: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> np.ndarray:
    """One viz row: [normals | GT | pred], each HxWx3 uint8.

    img: (7, H, W) — geom input. target: (H, W) ∈ {0,1,2}. pred: (2, H, W) logits.
    Vendored from unified script `_make_viz_row`.
    """
    silh = img[CH_SILH].detach().cpu().float().numpy().clip(0, 1)
    normals = img[CH_NX:CH_NZ + 1].detach().cpu().float().numpy()
    nrgb = (np.transpose(normals, (1, 2, 0)) + 1.0) / 2.0
    nrgb = nrgb * silh[:, :, None]
    nrgb_u8 = (nrgb.clip(0, 1) * 255).astype(np.uint8)

    target_np = target.detach().cpu().numpy()
    fg = target_np > 0
    target_rgb = _CLASS_COLORS[target_np.clip(0, 2).astype(np.int64)]

    probs = torch.softmax(pred.detach().cpu().float(), dim=0).numpy()
    ab_colors = _CLASS_COLORS[1:3].astype(np.float32)
    pred_rgb = (np.transpose(probs, (1, 2, 0)) @ ab_colors).clip(0, 255).astype(np.uint8)
    pred_rgb[~fg] = 0
    return np.concatenate([nrgb_u8, target_rgb, pred_rgb], axis=1)


def make_mask_viz_grid(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    n_max: int = 8,
) -> np.ndarray | None:
    """Pull a single batch from val_loader, pick up to `n_max` has_mask_gt geom items,
    stack [normals | GT | pred] rows. Returns (n*H, 3*W, 3) uint8 or None."""
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            mask_gt = batch["has_mask_gt"]
            modality = batch["modality"]
            eligible = (mask_gt & (modality == MODALITY_GEOM)).nonzero(as_tuple=True)[0]
            if len(eligible) == 0:
                continue
            picks = eligible[:n_max]
            x = batch["input"].to(device, non_blocking=True)
            mod_gpu = batch["modality"].to(device, non_blocking=True)
            vidx = batch["view_idx"].to(device, non_blocking=True)
            _, _, mask_logits, _ = model(x, mod_gpu, view_idx=vidx)
            for i in picks.tolist():
                rows.append(make_viz_row(
                    batch["input"][i],
                    batch["mask_label"][i],
                    mask_logits[i],
                ))
            if len(rows) >= n_max:
                break
    if not rows:
        return None
    return np.concatenate(rows[:n_max], axis=0)


__all__ = [
    # losses
    "compute_multitask_losses",
    "binary_softmax_ce_dice_invariant_loss",
    "compute_iou_invariant_batched",
    # optim/sched/AMP
    "build_optimizer", "build_scheduler", "optimizer_step_amp",
    # metrics
    "compute_ece", "compute_roc_auc", "cls_metrics", "nanmean",
    "aggregate_mean_prob", "aggregate_majority_vote", "_aggregate_cls",
    "compute_adaptive_ece", "compute_sharpness", "binary_metrics",
    # eval orchestration
    "run_eval", "collect_predictions", "per_op_table", "metrics_from_predictions",
    # DDP
    "ddp_active", "ddp_unwrap", "ddp_all_reduce_metrics",
    "setup_ddp", "teardown_ddp",
    # misc
    "set_seeds", "dataloader_worker_init_fn", "check_pcie_gen5_or_die",
    # viz
    "make_viz_row", "make_mask_viz_grid",
    # constants
    "TASK_ENDPOINT", "TASK_JUNCTION", "MODALITY_GEOM", "MODALITY_EM",
]
