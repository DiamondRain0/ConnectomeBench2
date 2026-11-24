"""training.py — ConnectomeBench2 multi-task ViT: train / evaluate / export.

Plain argparse + torchrun (or mp.spawn) for DDP.

Usage (from the repo root):
    # single GPU, paper ViT-B recipe (20 epochs, bs 256 → lr 2.83e-4)
    python src/neurips_training/training.py \\
        --blend-config src/neurips_training/configs/full_4sp.yaml \\
        --parquet-root /data/ConnectomeBench2 --batch-size 256

    # multi-GPU (torchrun sets RANK / WORLD_SIZE / LOCAL_RANK)
    torchrun --nproc-per-node=3 src/neurips_training/training.py \\
        --blend-config src/neurips_training/configs/full_4sp.yaml \\
        --model vit_l_16 --batch-size 768

    # held-out TEST evaluation + per-view prediction export (→ analyze_predictions.py)
    python src/neurips_training/training.py --mode evaluate \\
        --blend-config src/neurips_training/configs/full_4sp.yaml \\
        --checkpoint checkpoints_neurips/neurips_run_best.pt \\
        --eval-split test --export-predictions results/vitb_test.parquet

    # inference-time ablations (Table 1): --modalities geom | em, --ablate shape|normals|masks
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Sibling-module imports (this package uses flat names).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (  # noqa: E402
    VALID_ABLATIONS,
    BlendConfig,
    NeurIPSDataset,
    collate,
    load_blend_config,
)
from model import build_model  # noqa: E402
from training_utils import (  # noqa: E402
    build_optimizer,
    build_scheduler,
    collect_predictions,
    compute_multitask_losses,
    dataloader_worker_init_fn,
    ddp_active,
    ddp_all_reduce_metrics,
    ddp_unwrap,
    make_mask_viz_grid,
    metrics_from_predictions,
    optimizer_step_amp,
    run_eval,
    set_seeds,
    setup_ddp,
    teardown_ddp,
)


# ─────────────────────────── Config ──────────────────────────────────────

@dataclass
class MultiTaskConfig:
    """Multi-task training config. Saved into checkpoints via `vars(cfg)`.

    Optimisation hyper-parameters left as None are filled with the paper's
    values by `bridge_vit_defaults` (App. B.2); explicit CLI values win.
    """
    # Data
    parquet_root: str | None = None
    blend_config_path: str | None = None
    image_size: int = 224
    modalities: tuple[str, ...] = ("geom", "em")
    include_species: tuple[str, ...] | None = None       # train-time species override (LOSO)
    eval_include_species: tuple[str, ...] | None = None  # eval-time species override (LOSO)
    epoch_size_mode: str = "full"
    shuffle_buffer: int = 512
    drop_em_best_for_single: bool = True

    # Model
    model_name: str = "vit_b_16"  # vit_b_16 | vit_l_16 | vit_h_14
    use_view_embedding: bool = True

    # Splits / sampling
    seed: int = 42
    data_subsample_seed: int | None = None
    find_unused_parameters: bool = False

    # Training
    epochs: int = 20
    batch_size: int = 256
    global_batch_size: int | None = None
    num_workers: int = 16
    prefetch_factor: int = 4
    max_samples: int | None = None
    max_val_samples: int | None = None
    eval_every: int = 1

    # Optim (None → paper default via bridge_vit_defaults)
    learning_rate: float | None = None
    weight_decay: float | None = None
    warmup_epochs: int | None = None
    label_smoothing: float | None = None
    max_grad_norm: float | None = None
    drop_path_rate: float | None = None
    head_dropout: float | None = None
    backbone_lr_scale: float = 0.1
    warmup_start_factor: float = 0.1
    peak_lr_multiplier: float = 1.0

    # Multi-task losses
    mask_ce_weight: float = 0.5
    loss_scale_endpoint: float = 1.0
    loss_scale_junction: float = 1.0
    loss_scale_mask: float = 1.0
    loss_scale_synapse: float = 0.25
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float16"

    # Class balance
    class_balance: bool = True

    # Init / warmstart / resume
    init_source: str = "imagenet"  # "imagenet" | "none" | "fm_checkpoint"
    fm_checkpoint_path: str | None = None
    warmstart_checkpoint_path: str | None = None
    resume_path: str | None = None

    # Logging / ckpt
    use_wandb: bool = False
    wandb_project: str = "neurips-streamlined"
    wandb_run_name: str | None = None
    checkpoint_dir: str = "./checkpoints_neurips"
    checkpoint_name: str = "neurips_run"
    epoch_offset: int = 0

    # DDP
    world_size: int = 1

    # Device override ("auto" | "cuda" | "mps" | "cpu")
    device: str = "auto"


# Paper App. B.2: reference LR 1e-4 at batch 32, sqrt scaling
# (→ 2.83e-4 at bs 256, 4.90e-4 at bs 768); wd 0.05 / 0.1 and drop-path
# 0.1 / 0.25 for ViT-B / ViT-L; 5 warmup epochs; label smoothing 0.1; clip 1.0.
_PAPER_REF_LR, _PAPER_REF_BS = 1e-4, 32
_PAPER_ARCH_DEFAULTS = {
    "vit_b_16": {"weight_decay": 0.05, "drop_path_rate": 0.10},
    "vit_l_16": {"weight_decay": 0.10, "drop_path_rate": 0.25},
    "vit_h_14": {"weight_decay": 0.10, "drop_path_rate": 0.25},
}


def bridge_vit_defaults(cfg: MultiTaskConfig) -> None:
    """Fill unset optimisation hyper-parameters with the paper's values.

    Uses the GLOBAL batch size for LR scaling. Idempotent; never overrides a
    value the user set explicitly.
    """
    bs = int(cfg.global_batch_size or cfg.batch_size)
    arch = _PAPER_ARCH_DEFAULTS.get(cfg.model_name, _PAPER_ARCH_DEFAULTS["vit_b_16"])
    if cfg.learning_rate is None:
        cfg.learning_rate = _PAPER_REF_LR * math.sqrt(bs / _PAPER_REF_BS)
    if cfg.weight_decay is None:
        cfg.weight_decay = arch["weight_decay"]
    if cfg.drop_path_rate is None:
        cfg.drop_path_rate = arch["drop_path_rate"]
    if cfg.label_smoothing is None:
        cfg.label_smoothing = 0.1
    if cfg.max_grad_norm is None:
        cfg.max_grad_norm = 1.0
    if cfg.head_dropout is None:
        cfg.head_dropout = 0.1
    if cfg.warmup_epochs is None:
        cfg.warmup_epochs = 5


# ─────────────── Yaml + dataset wiring ───────────────────────────────────

def _load_training_yaml(path: str | Path) -> tuple[Path | None, BlendConfig]:
    """Parse the training yaml. Returns (parquet_root or None, blend_cfg)."""
    raw = yaml.safe_load(Path(path).read_text())
    parquet_root_raw = (raw.get("parquet") or {}).get("path")
    blend_cfg = load_blend_config(path)
    return (Path(parquet_root_raw) if parquet_root_raw else None), blend_cfg


def _resolve_data(cfg: MultiTaskConfig) -> tuple[Path, BlendConfig]:
    if not cfg.blend_config_path:
        raise ValueError("cfg.blend_config_path is required")
    root_yaml, blend_cfg = _load_training_yaml(cfg.blend_config_path)
    root = Path(cfg.parquet_root) if cfg.parquet_root else root_yaml
    if root is None:
        raise ValueError("no parquet root: pass --parquet-root or set parquet.path in the yaml")
    blend_cfg.class_balance = bool(cfg.class_balance)  # CLI is authoritative
    return root, blend_cfg


def _eval_blend(cfg: MultiTaskConfig, blend_cfg: BlendConfig) -> BlendConfig:
    b = BlendConfig(**vars(blend_cfg))
    if cfg.eval_include_species is not None:
        b.include_species = list(cfg.eval_include_species)
    return b


def _build_train_loader(cfg: MultiTaskConfig, root: Path, blend_cfg: BlendConfig):
    b = BlendConfig(**vars(blend_cfg))
    if cfg.include_species is not None:
        b.include_species = list(cfg.include_species)
    ds = NeurIPSDataset(
        parquet_root=root, split="train", blend=b, modalities=cfg.modalities,
        augment=True, max_samples=cfg.max_samples,
        data_subsample_seed=cfg.data_subsample_seed, seed=cfg.seed,
        epoch_size_mode=cfg.epoch_size_mode, shuffle_buffer=cfg.shuffle_buffer,
        drop_em_best_for_single=cfg.drop_em_best_for_single,
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True, collate_fn=collate,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        worker_init_fn=dataloader_worker_init_fn if cfg.num_workers > 0 else None,
    )
    return loader, ds


def _build_eval_loader(cfg: MultiTaskConfig, root: Path, blend_cfg: BlendConfig, split: str,
                       ablate: tuple[str, ...] = (), max_samples: int | None = None,
                       persistent: bool = False):
    ds = NeurIPSDataset(
        parquet_root=root, split=split, blend=_eval_blend(cfg, blend_cfg),
        modalities=cfg.modalities, augment=False, max_samples=max_samples,
        data_subsample_seed=cfg.data_subsample_seed if max_samples else None,
        seed=cfg.seed, sample_all=True,
        drop_em_best_for_single=cfg.drop_em_best_for_single, ablate=ablate,
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(), collate_fn=collate,
        persistent_workers=persistent and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        worker_init_fn=dataloader_worker_init_fn if cfg.num_workers > 0 else None,
    )
    return loader, ds


# ─────────────────── checkpoint save / load ───────────────────────────────

def save_checkpoint(
    path: str | Path, model: nn.Module, cfg: MultiTaskConfig, epoch: int,
    metrics: dict[str, float], extra: dict[str, Any] | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": ddp_unwrap(model).state_dict(),
        "cfg": vars(cfg),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    if extra:
        payload.update(extra)
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)  # atomic: a crash mid-save never corrupts the last good ckpt


def load_checkpoint_for_eval(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load `{model, cfg, ...}`. Strips `module.` prefix from DDP saves."""
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    return sd, state.get("cfg", {})


def _resolve_device(cfg: MultiTaskConfig, rank: int = 0) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{rank}")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device(f"cuda:{rank}")
    return torch.device(cfg.device)


def _model_from_cfg(cfg: MultiTaskConfig, *, for_eval: bool = False) -> nn.Module:
    return build_model(
        model_name=cfg.model_name,
        image_size=cfg.image_size,
        drop_path_rate=cfg.drop_path_rate,
        head_dropout=cfg.head_dropout,
        # For eval the weights come from the checkpoint; skip the ImageNet download.
        init_source="none" if for_eval else cfg.init_source,
        fm_checkpoint_path=None if for_eval else cfg.fm_checkpoint_path,
        em_enabled="em" in cfg.modalities,
        warmstart_checkpoint_path=None if for_eval else cfg.warmstart_checkpoint_path,
        use_view_embedding=cfg.use_view_embedding,
    )


# ───────────────────────── train ──────────────────────────────────────────

def _print_val(metrics: dict[str, float], s: str = "val") -> None:
    def _fmt(key: str, label: str) -> str:
        v = metrics.get(key, float("nan"))
        return "" if (isinstance(v, float) and math.isnan(v)) else f"{label}={v:.4f}"
    print(f"  {s}:", flush=True)
    for tag in ("end_corr", "jn_id", "synapse"):
        parts = [p for p in (
            _fmt(f"{tag}/{s}_bal_acc_mean_prob", "bAcc_op"),
            _fmt(f"{tag}/{s}_bal_acc_per_image", "bAcc_img"),
            _fmt(f"{tag}/{s}_roc_auc_mean_prob", "auc_op"),
            _fmt(f"{tag}/{s}_ece_adaptive_mean_prob", "aECE_op"),
            _fmt(f"{tag}/{s}_loss", "loss"),
            _fmt(f"{tag}/{s}_n_mean_prob", "n_ops"),
        ) if p]
        if parts:
            print(f"    {tag:9s}: " + " ".join(parts), flush=True)
    mp_ = [p for p in (_fmt(f"mask/{s}_mIoU_inv", "mIoU"), _fmt(f"mask/{s}_loss", "loss"),
                       _fmt(f"mask/{s}_n_samples", "n")) if p]
    if mp_:
        print("    mask     : " + " ".join(mp_), flush=True)
    for sp in ("mouse", "fly", "zebrafish", "human"):
        parts = [p for p in (_fmt(f"end_corr/{sp}/{s}_bal_acc_mean_prob", "split_err"),
                             _fmt(f"jn_id/{sp}/{s}_bal_acc_mean_prob", "merge_err"),
                             _fmt(f"mask/{sp}/{s}_mIoU_inv", "mIoU")) if p]
        if parts:
            print(f"    {sp:9s}: " + " ".join(parts), flush=True)


def _headline(metrics: dict[str, float], s: str = "val") -> float:
    vals = [metrics.get(f"end_corr/{s}_bal_acc_mean_prob", float("nan")),
            metrics.get(f"jn_id/{s}_bal_acc_mean_prob", float("nan")),
            metrics.get(f"mask/{s}_mIoU_inv", float("nan"))]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else -float("inf")


def train(cfg: MultiTaskConfig, rank: int = 0, world_size: int = 1) -> dict[str, float]:
    """Main train loop. Single-rank or per-rank under DDP.

      - bf16 AMP (fp16 opt-in, GradScaler only for fp16)
      - clip → step → sched.step() per batch; fixed #steps per epoch on every
        rank (ranks stop together when any rank's stream runs dry → no DDP hang)
      - rank-0 eval on the FULL val split (or --max-val-samples rows)
      - headline = nanmean(split-error bAcc, merge-error bAcc, mask mIoU) on val;
        best.pt on improve; last.pt (with optimizer state) every epoch → --resume
      - per-epoch metrics appended to <checkpoint_dir>/<name>_metrics.jsonl
    """
    bridge_vit_defaults(cfg)  # idempotent; DDP callers already bridged with global bs
    set_seeds(cfg.seed, rank=rank)
    is_main = rank == 0
    device = _resolve_device(cfg, rank)
    if is_main:
        print(f"Device: {device} (world_size={world_size}, rank={rank})", flush=True)
        print("Config: " + json.dumps(vars(cfg), default=str), flush=True)

    root, blend_cfg = _resolve_data(cfg)
    train_loader, train_ds = _build_train_loader(cfg, root, blend_cfg)
    val_loader = val_ds = None
    if is_main:
        val_loader, val_ds = _build_eval_loader(cfg, root, blend_cfg, "val",
                                                max_samples=cfg.max_val_samples,
                                                persistent=True)
        print(f"  schema={train_ds.schema} train rows={train_ds.n_rows} "
              f"(~{len(train_ds)} items/epoch/rank), val rows={val_ds.n_rows}, "
              f"shards train/val: {len(train_ds.shards)}/{len(val_ds.shards)}", flush=True)

    model: nn.Module = _model_from_cfg(cfg).to(device)
    optimizer = build_optimizer(
        model, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        backbone_lr_scale=cfg.backbone_lr_scale, peak_lr_multiplier=cfg.peak_lr_multiplier,
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps,
                                warmup_start_factor=cfg.warmup_start_factor)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Resume (model + optimizer + scheduler + scaler + counters)
    start_epoch, global_step, best_headline = 0, 0, -float("inf")
    if cfg.resume_path:
        st = torch.load(cfg.resume_path, map_location="cpu", weights_only=False)
        sd = st["model"]
        if any(k.startswith("module.") for k in sd):
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
        model.load_state_dict(sd)
        if "optimizer" in st:
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            if st.get("scaler"):
                scaler.load_state_dict(st["scaler"])
        start_epoch = int(st.get("epoch", -1)) + 1
        global_step = int(st.get("global_step", 0))
        best_headline = float(st.get("best_headline", -float("inf")))
        if is_main:
            print(f"  resumed from {cfg.resume_path} -> starting at epoch {start_epoch}", flush=True)

    if world_size > 1:
        fup = ("em" in cfg.modalities) or cfg.find_unused_parameters
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=fup)

    if is_main:
        n_params = sum(p.numel() for p in ddp_unwrap(model).parameters())
        print(f"  params={n_params/1e6:.1f}M steps/epoch={steps_per_epoch} total_steps={total_steps} "
              f"lr={cfg.learning_rate:.3g} wd={cfg.weight_decay} drop_path={cfg.drop_path_rate} "
              f"AMP={use_amp}/{cfg.amp_dtype}", flush=True)

    wandb_run = None
    if cfg.use_wandb and is_main:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                               config=vars(cfg), resume="allow")

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = ckpt_dir / f"{cfg.checkpoint_name}_metrics.jsonl"
    last_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, cfg.epochs):
        train_ds.epoch = epoch  # shared-memory → visible to persistent workers
        model.train()
        ep_running = {k: torch.zeros((), device=device)
                      for k in ("L_endpoint", "L_junction", "L_mask", "L_synapse", "L_total")}
        n_batches = 0
        t0 = time.time()
        it = iter(train_loader)

        for batch_idx in range(steps_per_epoch):
            try:
                batch = next(it)
                have = 1
            except StopIteration:
                batch, have = None, 0
            if ddp_active(world_size):
                flag = torch.tensor(have, device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                have = int(flag.item())
            if not have:
                break

            x = batch["input"].to(device, non_blocking=True)
            gpu_batch = {k: batch[k].to(device, non_blocking=True) for k in (
                "cls_label", "task_id", "has_mask_gt", "mask_label", "modality",
                "has_synapse_label", "synapse_label", "view_idx")}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                ep_logits, jn_logits, mask_logits, syn_logits = model(
                    x, gpu_batch["modality"], gpu_batch["view_idx"])
            losses = compute_multitask_losses(
                ep_logits.float(), jn_logits.float(), mask_logits.float(), syn_logits.float(),
                gpu_batch,
                label_smoothing=cfg.label_smoothing, mask_ce_weight=cfg.mask_ce_weight,
                loss_scale_endpoint=cfg.loss_scale_endpoint,
                loss_scale_junction=cfg.loss_scale_junction,
                loss_scale_mask=cfg.loss_scale_mask, loss_scale_synapse=cfg.loss_scale_synapse,
            )
            grad_norm = optimizer_step_amp(losses["L_total"], optimizer, scaler, scheduler,
                                           model, cfg.max_grad_norm)

            for k in ep_running:
                ep_running[k] += losses[k].detach()
            n_batches += 1
            global_step += 1

            if is_main and batch_idx % 100 == 0:
                print(f"    [ep {epoch} step {batch_idx}/{steps_per_epoch}] "
                      f"loss={losses['L_total'].item():.4f} lr={optimizer.param_groups[-1]['lr']:.3g} "
                      f"{(batch_idx + 1) * cfg.batch_size / max(1e-6, time.time() - t0):.0f} img/s/rank",
                      flush=True)
            if cfg.use_wandb and batch_idx % 20 == 0:
                log_losses = {k: float(losses[k].detach().item()) for k in ep_running}
                if ddp_active(world_size):
                    log_losses = ddp_all_reduce_metrics(log_losses, world_size, op="avg")
                if is_main and wandb_run is not None:
                    lrs = [pg["lr"] for pg in optimizer.param_groups]
                    wandb_run.log({
                        "train/total_loss": log_losses["L_total"],
                        "end_corr/train_loss": log_losses["L_endpoint"],
                        "jn_id/train_loss": log_losses["L_junction"],
                        "mask/train_loss": log_losses["L_mask"],
                        "synapse/train_loss": log_losses["L_synapse"],
                        "train/lr_backbone": lrs[0], "train/lr_heads": lrs[-1],
                        "train/grad_norm": float(grad_norm),
                        "train/global_step": global_step,
                        "train/epoch": epoch + cfg.epoch_offset,
                    })

        if ddp_active(world_size):
            for v in ep_running.values():
                dist.all_reduce(v, op=dist.ReduceOp.SUM)
        denom = max(1, n_batches) * max(1, world_size)
        avg_losses = {k: v.item() / denom for k, v in ep_running.items()}
        if is_main:
            print(f"[epoch {epoch:3d}] {n_batches} steps in {time.time() - t0:.0f}s  "
                  + " ".join(f"{k}={v:.4f}" for k, v in avg_losses.items()), flush=True)

        # Eval — rank 0 only; others wait at the barrier.
        metrics: dict[str, float] = {}
        do_eval = ((epoch + 1) % max(1, cfg.eval_every) == 0) or (epoch == cfg.epochs - 1)
        if is_main and do_eval:
            metrics = run_eval(
                ddp_unwrap(model), val_loader, device,
                label_smoothing=cfg.label_smoothing, mask_ce_weight=cfg.mask_ce_weight,
                loss_scale_endpoint=cfg.loss_scale_endpoint,
                loss_scale_junction=cfg.loss_scale_junction,
                loss_scale_mask=cfg.loss_scale_mask, loss_scale_synapse=cfg.loss_scale_synapse,
            )
            _print_val(metrics)
        if ddp_active(world_size):
            dist.barrier()

        if is_main:
            if metrics:
                last_metrics = metrics
            headline = _headline(metrics) if metrics else -float("inf")
            with open(metrics_log, "a") as f:
                f.write(json.dumps({"epoch": epoch, "global_step": global_step,
                                    "headline": headline, **{f"train/avg_{k}": v for k, v in avg_losses.items()},
                                    **metrics}, default=float) + "\n")
            if cfg.use_wandb and wandb_run is not None:
                import wandb
                payload: dict[str, Any] = {**metrics, "train/epoch": epoch + cfg.epoch_offset,
                                           **{f"train/avg_{k}": v for k, v in avg_losses.items()}}
                grid = make_mask_viz_grid(ddp_unwrap(model), val_loader, device, n_max=8) if metrics else None
                if grid is not None:
                    payload["mask/val_predictions"] = wandb.Image(grid, caption="val: normals | GT | pred")
                wandb_run.log(payload)
            if metrics and headline > best_headline:
                best_headline = headline
                save_checkpoint(ckpt_dir / f"{cfg.checkpoint_name}_best.pt", model, cfg, epoch, metrics)
                print(f"  new best headline={headline:.4f} -> {cfg.checkpoint_name}_best.pt", flush=True)
            save_checkpoint(
                ckpt_dir / f"{cfg.checkpoint_name}_last.pt", model, cfg, epoch, metrics,
                extra={"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "scaler": scaler.state_dict() if use_scaler else None,
                       "global_step": global_step, "best_headline": best_headline},
            )
        if ddp_active(world_size):
            dist.barrier()

    if is_main and wandb_run is not None:
        wandb_run.finish()
    return last_metrics


# ──────────────────────── _train_worker (mp.spawn / torchrun) ────────────

def _train_worker(rank: int, world_size: int, cfg: MultiTaskConfig, local_rank: int | None = None) -> None:
    """Per-rank entrypoint: NCCL setup, per-rank batch/workers, train(), teardown."""
    local_rank = rank if local_rank is None else local_rank
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        setup_ddp(rank=rank, world_size=world_size)
    cfg.batch_size = max(1, cfg.batch_size // max(1, world_size))
    cfg.num_workers = max(2, cfg.num_workers // max(1, world_size))
    try:
        train(cfg, rank=rank, world_size=world_size)
    finally:
        teardown_ddp()


# ──────────────────────────── evaluate ────────────────────────────────────

def _cfg_with_checkpoint_arch(cfg: MultiTaskConfig, ckpt_cfg: dict[str, Any]) -> MultiTaskConfig:
    """Architecture-defining fields come from the checkpoint, not the CLI."""
    for k in ("model_name", "image_size", "use_view_embedding"):
        if k in ckpt_cfg and ckpt_cfg[k] is not None:
            setattr(cfg, k, ckpt_cfg[k])
    if "use_view_embedding" not in ckpt_cfg:
        cfg.use_view_embedding = False  # checkpoint from the original release
    return cfg


def evaluate(
    cfg: MultiTaskConfig,
    checkpoint_path: str | Path,
    split: str = "val",
    ablate: tuple[str, ...] = (),
    export_predictions: str | Path | None = None,
    export_features: str | Path | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate a checkpoint on the FULL `split` ("val" | "test" | "train").

    Writes the per-view prediction table (parquet; input of
    scripts/training/analyze_predictions.py) and optionally per-view CLS
    features (npz; input of scripts/training/mahalanobis_ood.py).
    """
    sd, ckpt_cfg = load_checkpoint_for_eval(checkpoint_path)
    cfg = _cfg_with_checkpoint_arch(cfg, ckpt_cfg)
    bridge_vit_defaults(cfg)
    set_seeds(cfg.seed, rank=0)
    device = _resolve_device(cfg, 0)

    root, blend_cfg = _resolve_data(cfg)
    loader, ds = _build_eval_loader(cfg, root, blend_cfg, split, ablate=ablate,
                                    max_samples=cfg.max_val_samples)
    print(f"  eval split={split} schema={ds.schema} rows={ds.n_rows} items~{len(ds)} "
          f"modalities={cfg.modalities} ablate={ablate or '-'} "
          f"species={ds.blend.include_species}", flush=True)

    model = _model_from_cfg(cfg, for_eval=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # conv_proj_em may legitimately be absent from geom-only checkpoints, etc.
    print(f"  loaded {checkpoint_path}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"    missing (first 10): {missing[:10]}", flush=True)
    if unexpected:
        print(f"    unexpected (first 10): {unexpected[:10]}", flush=True)
    model = model.to(device)

    t0 = time.time()
    res = collect_predictions(
        model, loader, device, label_smoothing=cfg.label_smoothing,
        mask_ce_weight=cfg.mask_ce_weight, return_features=export_features is not None,
        max_batches=max_batches, progress=True,
        amp_dtype=torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16,
    )
    df, feats = (res if export_features is not None else (res, None))
    dt = time.time() - t0
    metrics = metrics_from_predictions(df, split_tag=split)
    metrics[f"{split}/n_views"] = float(len(df))
    metrics[f"{split}/seconds"] = dt
    metrics[f"{split}/views_per_second"] = len(df) / max(dt, 1e-9)
    if device.type == "cuda":
        metrics[f"{split}/peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    _print_val(metrics, split)

    if export_predictions:
        p = Path(export_predictions)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.attrs = {}
        df.assign(checkpoint=str(checkpoint_path), split=split,
                  ablate=",".join(ablate), modalities=",".join(cfg.modalities)).to_parquet(p, index=False)
        print(f"  wrote {len(df)} per-view predictions -> {p}", flush=True)
        with open(p.with_suffix(".metrics.json"), "w") as f:
            json.dump(metrics, f, indent=1, default=float)
    if export_features is not None and feats is not None:
        p = Path(export_features)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, features=feats, op_id=df["op_id"].to_numpy(str), species=df["species"].to_numpy(str),
            modality=df["modality"].to_numpy(str), task_id=df["task_id"].to_numpy(),
            label=df["label"].to_numpy(), p_true=df["p_true"].to_numpy(),
        )
        print(f"  wrote features {feats.shape} -> {p}", flush=True)
    return metrics


def evaluate_export(cfg: MultiTaskConfig, checkpoint_path: str | Path, output_path: str | Path,
                    split: str = "val") -> dict[str, float]:
    """Backwards-compatible alias: evaluate + export per-view predictions."""
    return evaluate(cfg, checkpoint_path, split=split, export_predictions=output_path)


# ──────────────────────────── argparse ────────────────────────────────────

def _csv(s: str | None) -> tuple[str, ...] | None:
    if s is None:
        return None
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    p.add_argument("--blend-config", required=True, help="training yaml (parquet + filter + blend)")
    p.add_argument("--parquet-root", default=None, help="override yaml's parquet.path (dir with train/ val/ test/)")
    p.add_argument("--modalities", default="geom,em", help="comma-separated subset of {geom,em}")
    p.add_argument("--include-species", default=None,
                   help="train-time species override, e.g. 'mouse,fly,human' (LOSO: hold one out)")
    p.add_argument("--eval-species", default=None,
                   help="eval-time species override, e.g. 'zebrafish' (LOSO held-out evaluation)")
    p.add_argument("--max-samples", type=int, default=None, help="cap train rows (scaling curves)")
    p.add_argument("--max-val-samples", type=int, default=None, help="cap val/eval rows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-subsample-seed", type=int, default=None)
    p.add_argument("--epoch-size-mode", default="full", choices=["full", "accept"],
                   help="full = paper (N weighted draws w/ replacement per epoch); accept = original release")
    p.add_argument("--shuffle-buffer", type=int, default=512, help="per-worker item shuffle buffer (0=off)")
    p.add_argument("--keep-em-best-for-single", action="store_true",
                   help="keep em_best views for merge-error identification (dataset card advises against)")
    # Model
    p.add_argument("--model", dest="model_name", default="vit_b_16",
                   choices=["vit_b_16", "vit_l_16", "vit_h_14"])
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--init-source", default="imagenet", choices=["imagenet", "none", "fm_checkpoint"])
    p.add_argument("--fm-checkpoint", dest="fm_checkpoint_path", default=None)
    p.add_argument("--warmstart", dest="warmstart_checkpoint_path", default=None)
    p.add_argument("--resume", dest="resume_path", default=None, help="resume from a *_last.pt")
    p.add_argument("--no-view-embedding", action="store_true",
                   help="disable the mask-decoder view embedding (original-release behaviour)")
    # Optim / schedule (None → paper defaults)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256, help="GLOBAL batch size")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--lr", dest="learning_rate", type=float, default=None,
                   help="default: 1e-4 * sqrt(global_bs / 32)")
    p.add_argument("--weight-decay", type=float, default=None, help="default: 0.05 (B) / 0.1 (L)")
    p.add_argument("--drop-path", dest="drop_path_rate", type=float, default=None,
                   help="default: 0.1 (B) / 0.25 (L)")
    p.add_argument("--warmup-epochs", type=int, default=None, help="default: 5")
    p.add_argument("--warmup-start-factor", type=float, default=0.1)
    p.add_argument("--peak-lr-multiplier", type=float, default=1.0)
    p.add_argument("--backbone-lr-scale", type=float, default=0.1)
    p.add_argument("--epoch-offset", type=int, default=0)
    p.add_argument("--max-grad-norm", type=float, default=None, help="default: 1.0")
    p.add_argument("--label-smoothing", type=float, default=None, help="default: 0.1")
    p.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--eval-every", type=int, default=1, help="validate every N epochs")
    # Losses
    p.add_argument("--mask-ce-weight", type=float, default=0.5)
    p.add_argument("--loss-scale-endpoint", type=float, default=1.0)
    p.add_argument("--loss-scale-junction", type=float, default=1.0)
    p.add_argument("--loss-scale-mask", type=float, default=1.0)
    p.add_argument("--loss-scale-synapse", type=float, default=0.25)
    p.add_argument("--class-balance", action=argparse.BooleanOptionalAction, default=True)
    # Logging / checkpoint
    p.add_argument("--wandb", dest="use_wandb", action="store_true")
    p.add_argument("--wandb-project", default="neurips-streamlined")
    p.add_argument("--run-name", dest="wandb_run_name", default=None)
    p.add_argument("--checkpoint-dir", default="./checkpoints_neurips")
    p.add_argument("--checkpoint-name", default="neurips_run")
    # DDP
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--world-size", type=int, default=1, help=">1 with --spawn enables DDP via mp.spawn")
    p.add_argument("--spawn", action="store_true", help="use mp.spawn instead of torchrun env vars")
    p.add_argument("--find-unused-parameters", action="store_true")
    # Mode
    p.add_argument("--mode", default="train", choices=["train", "evaluate", "evaluate_export"])
    p.add_argument("--checkpoint", default=None, help="checkpoint for --mode evaluate*")
    p.add_argument("--eval-split", default="val", choices=["train", "val", "validation", "test"])
    p.add_argument("--ablate", default="", help=f"comma-separated subset of {VALID_ABLATIONS} (eval only)")
    p.add_argument("--export-predictions", "--export-output", dest="export_predictions", default=None,
                   help="per-view predictions .parquet")
    p.add_argument("--export-features", default=None, help="per-view CLS features .npz (Mahalanobis OOD)")
    p.add_argument("--max-eval-batches", type=int, default=None, help="debug: stop eval early")
    return p


def _cfg_from_args(args: argparse.Namespace) -> MultiTaskConfig:
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    world = max(args.world_size, env_world)
    return MultiTaskConfig(
        parquet_root=args.parquet_root,
        blend_config_path=args.blend_config,
        image_size=args.image_size,
        modalities=_csv(args.modalities) or ("geom", "em"),
        include_species=_csv(args.include_species),
        eval_include_species=_csv(args.eval_species),
        epoch_size_mode=args.epoch_size_mode,
        shuffle_buffer=args.shuffle_buffer,
        drop_em_best_for_single=not args.keep_em_best_for_single,
        model_name=args.model_name,
        use_view_embedding=not args.no_view_embedding,
        seed=args.seed,
        data_subsample_seed=args.data_subsample_seed,
        find_unused_parameters=args.find_unused_parameters,
        epochs=args.epochs,
        batch_size=args.batch_size,
        global_batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        max_samples=args.max_samples,
        max_val_samples=args.max_val_samples,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        drop_path_rate=args.drop_path_rate,
        backbone_lr_scale=args.backbone_lr_scale,
        warmup_start_factor=args.warmup_start_factor,
        peak_lr_multiplier=args.peak_lr_multiplier,
        mask_ce_weight=args.mask_ce_weight,
        loss_scale_endpoint=args.loss_scale_endpoint,
        loss_scale_junction=args.loss_scale_junction,
        loss_scale_mask=args.loss_scale_mask,
        loss_scale_synapse=args.loss_scale_synapse,
        amp_dtype=args.amp_dtype,
        class_balance=args.class_balance,
        init_source=args.init_source,
        fm_checkpoint_path=args.fm_checkpoint_path,
        warmstart_checkpoint_path=args.warmstart_checkpoint_path,
        resume_path=args.resume_path,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        epoch_offset=args.epoch_offset,
        world_size=world,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg = _cfg_from_args(args)

    if args.mode in ("evaluate", "evaluate_export"):
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --mode evaluate")
        if args.mode == "evaluate_export" and not args.export_predictions:
            raise SystemExit("--export-predictions is required for --mode evaluate_export")
        ablate = _csv(args.ablate) or ()
        metrics = evaluate(cfg, args.checkpoint, split=args.eval_split, ablate=ablate,
                           export_predictions=args.export_predictions,
                           export_features=args.export_features,
                           max_batches=args.max_eval_batches)
        headline = {k: v for k, v in metrics.items() if k.count("/") == 1}
        print(json.dumps(headline, indent=1, default=float))
        return

    env_world = int(os.environ.get("WORLD_SIZE", "0"))
    if env_world > 1 and not args.spawn:
        # torchrun: this process is one rank. Bridge with the GLOBAL batch size,
        # then _train_worker divides batch size / workers per rank.
        cfg.world_size = env_world
        bridge_vit_defaults(cfg)
        _train_worker(int(os.environ["RANK"]), env_world, cfg,
                      local_rank=int(os.environ.get("LOCAL_RANK", os.environ["RANK"])))
        return

    world_size = max(1, args.world_size)
    if world_size <= 1:
        train(cfg, rank=0, world_size=1)
        return

    bridge_vit_defaults(cfg)
    mp.spawn(_train_worker, args=(world_size, cfg), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
