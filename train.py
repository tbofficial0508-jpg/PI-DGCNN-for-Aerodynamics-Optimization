"""
train.py – Training loop for the DGCNN CFD surrogate.

Features
────────
  - Adam optimiser + CosineAnnealingWarmRestarts (or ReduceLROnPlateau fallback)
  - Mixed-precision training (AMP) to stay within 6 GB VRAM
  - Multi-task loss: MSE scalars (weight 1.0) + MSE fields (weight 0.1)
  - Best checkpoint saved by validation scalar MAE (the key optimisation target)
  - Periodic checkpoint every N epochs (config.checkpoint_interval)
  - Early stopping on val scalar error with patience (config.early_stopping_patience)
  - All losses logged to loss_log.csv per epoch for post-processing
  - Training curves saved as PNG + PDF at the end

Usage
─────
  python train.py
  python train.py --epochs 100 --lr 5e-4 --batch_size 2
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from config import Config, cfg
from dataset import CFDDataset, make_loaders
from losses import HybridPhysicsLoss
from model import DGCNN
from physics import AutogradPhysicsLoss
from physics_nemo import PhysicsNeMoLoss
from plotting_utils import (
    BODY_COLOR,
    EDF_COLOR,
    apply_publication_style,
    panel_label,
    save_figure_png_pdf,
)


apply_publication_style()


# ── EMA weight averaging ──────────────────────────────────────────────────────

class ModelEMA:
    """Bias-corrected exponential moving average of model weights.

    Float tensors are initialised to zero and updated `shadow = decay*shadow +
    (1-decay)*current`. When read via `apply_to`, the shadow is divided by
    `1 - decay**step_count` (Adam β₂ bias correction, Kingma & Ba 2014). This
    removes the early-training lag you get if you initialise shadow to the
    random model weights, which is especially bad here because the dataset
    has only 5 batches/epoch – the live model improves ~100× faster than
    a naive EMA can track.

    Integer buffers (e.g. GroupNorm has none, but safe for BN) are copied
    directly. `apply_to` swaps the model weights to the (debiased) shadow
    for evaluation/checkpointing; `restore` puts the trained weights back
    so training continues on the live copy.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k] = torch.zeros_like(v)
            else:
                self.shadow[k] = v.detach().clone()
        self.step_count = 0
        self._backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step_count += 1
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.is_floating_point():
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def _debiased(self) -> dict[str, torch.Tensor]:
        if self.step_count == 0:
            return self.shadow
        bias_corr = 1.0 - (self.decay ** self.step_count)
        if bias_corr <= 0.0:
            return self.shadow
        corrected: dict[str, torch.Tensor] = {}
        for k, s in self.shadow.items():
            corrected[k] = (s / bias_corr) if s.is_floating_point() else s
        return corrected

    def apply_to(self, model: nn.Module) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self._debiased(), strict=True)

    def restore(self, model: nn.Module) -> None:
        if self._backup is None:
            return
        model.load_state_dict(self._backup, strict=True)
        self._backup = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self._debiased().items()}


# ── Loss helpers ─────────────────────────────────────────────────────────────

def multi_task_loss(
    scalar_p: torch.Tensor,
    scalar_t: torch.Tensor,
    field_p: torch.Tensor,
    field_t: torch.Tensor,
    lambda_scalars: float,
    lambda_fields: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weighted sum of MSE losses.

    Scalars (thrust, drag) carry weight 1.0 – these drive optimisation.
    Fields (p, u, v, w) carry weight 0.1 – auxiliary, improves generalisation.
    """
    mse = nn.MSELoss()
    loss_s = mse(scalar_p, scalar_t)
    loss_f = mse(field_p,  field_t)
    total  = lambda_scalars * loss_s + lambda_fields * loss_f
    return total, loss_s, loss_f


def checkpoint_score(metrics: dict[str, float], c: Config) -> float:
    """Score to minimise when selecting the 'best' checkpoint.

    `combined_mae` uses *normalised* MAE (thrust_mae + drag_mae) so neither
    target dominates due to its physical magnitude. Previous behaviour used
    `_phys` units, which let drag (~3.84 N) dominate thrust (~0.92 N) by ~4×.
    """
    mode = str(c.checkpoint_metric).lower()
    if mode == "thrust_mae":
        return float(metrics["thrust_mae"])
    if mode == "drag_mae":
        return float(metrics["drag_mae"])
    if mode == "loss":
        return float(metrics["loss"])
    if mode == "combined_mae":
        return (
            float(c.checkpoint_thrust_weight) * float(metrics["thrust_mae"])
            + float(c.checkpoint_drag_weight)   * float(metrics["drag_mae"])
        )
    print(f"[train][warn] Unknown checkpoint_metric={c.checkpoint_metric!r}; falling back to combined_mae.")
    return (
        float(c.checkpoint_thrust_weight) * float(metrics["thrust_mae"])
        + float(c.checkpoint_drag_weight)   * float(metrics["drag_mae"])
    )


# ── Evaluation pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: DGCNN,
    loader,
    device: torch.device,
    c: Config,
    scaler_enabled: bool,
    loss_manager: HybridPhysicsLoss,
    scalar_scale: torch.Tensor,
    epoch: int,
) -> dict[str, float]:
    """Run one full validation pass and return averaged metrics."""
    model.eval()
    totals = dict(
        loss_total=0.0,
        loss_data=0.0,
        loss_scalar=0.0,
        loss_field=0.0,
        loss_field_grad=0.0,
        loss_thrust=0.0,
        loss_mass=0.0,
        loss_bc=0.0,
        loss_momentum=0.0,
        physics_ramp=0.0,
        spatial_weight_mean=0.0,
        thrust_mae=0.0,
        drag_mae=0.0,
        thrust_mae_phys=0.0,
        drag_mae_phys=0.0,
    )
    n = 0

    for batch in loader:
        geo       = batch["geometry_points"].to(device, non_blocking=True)
        mid_xyz   = batch["midplane_xyz"].to(device, non_blocking=True)
        mid_flds  = batch["midplane_fields"].to(device, non_blocking=True)
        scalar_t  = batch["scalar_targets"].to(device, non_blocking=True)
        cond      = batch["conditions"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=scaler_enabled):
            scalar_p, field_p = model(geo, mid_xyz, cond)
            _, terms = loss_manager(
                scalar_p=scalar_p,
                scalar_t=scalar_t,
                field_p=field_p,
                field_t=mid_flds,
                mid_xyz=mid_xyz,
                batch=batch,
                epoch=epoch,
            )

        totals["loss_total"] += terms["loss_total"].item()
        totals["loss_data"] += terms["loss_data"].item()
        totals["loss_scalar"] += terms["loss_scalar"].item()
        totals["loss_field"] += terms["loss_field"].item()
        totals["loss_field_grad"] += terms["loss_field_grad"].item()
        totals["loss_thrust"] += terms["loss_thrust"].item()
        totals["loss_mass"] += terms["loss_mass"].item()
        totals["loss_bc"] += terms["loss_bc"].item()
        totals["loss_momentum"] += terms["loss_momentum"].item()
        totals["physics_ramp"] += terms["physics_ramp"].item()
        totals["spatial_weight_mean"] += terms["spatial_weight_mean"].item()
        # drag=index0, thrust=index1 (from targets_scalar_raw in build script)
        thrust_mae_norm = (scalar_p[:, 1] - scalar_t[:, 1]).abs().mean()
        drag_mae_norm = (scalar_p[:, 0] - scalar_t[:, 0]).abs().mean()
        totals["thrust_mae"] += thrust_mae_norm.item()
        totals["drag_mae"] += drag_mae_norm.item()
        totals["thrust_mae_phys"] += (thrust_mae_norm * scalar_scale[1]).item()
        totals["drag_mae_phys"] += (drag_mae_norm * scalar_scale[0]).item()
        n += 1

    if n == 0:
        return totals
    metrics = {k: v / n for k, v in totals.items()}
    # Backward-compatible aliases used by scheduler/history plotting.
    metrics["loss"] = metrics["loss_total"]
    metrics["scalar"] = metrics["loss_scalar"]
    metrics["field"] = metrics["loss_field"]
    return metrics


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: DGCNN,
    optimizer,
    epoch: int,
    metrics: dict,
    norm_stats: dict,
    c: Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()

    # Snapshot model weights onto CPU to avoid any stale/in-flight GPU tensor state.
    model_state_cpu = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    payload = {
        "epoch":           epoch,
        "model_state":     model_state_cpu,
        "optimizer_state": optimizer.state_dict(),
        "metrics":         metrics,
        "norm_stats":      norm_stats,
        "config": {
            "k":             c.k,
            "in_channels":   c.in_channels,
            "cond_dim":      c.cond_dim,
            "scalar_dim":    c.scalar_dim,
            "field_dim":     c.field_dim,
            "edge_channels": c.edge_channels,
            "dropout":       c.dropout,
            "norm_type":     c.norm_type,
            "norm_groups":   c.norm_groups,
            "use_mean_pool": c.use_mean_pool,
            "fourier_levels": int(getattr(c, "fourier_levels", 6)),
        },
    }

    # Write to a sibling temp file then rename so OneDrive / Windows Defender
    # can't hold a lock on the final path while we're writing.
    # Catches both PermissionError (Python) and RuntimeError code 32
    # (Windows ERROR_SHARING_VIOLATION from the C++ zip writer).
    import tempfile, os
    tmp_path = path.with_suffix(".tmp")
    last_err: Exception | None = None
    for _ in range(5):
        try:
            torch.save(payload, tmp_path)
            if path.exists():
                path.unlink()
            tmp_path.rename(path)
            return
        except (PermissionError, RuntimeError, OSError) as err:
            last_err = err
            time.sleep(0.5)
    if last_err is not None:
        raise last_err


def load_checkpoint(path: Path, device: torch.device) -> tuple[DGCNN, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cc = ckpt["config"]
    model = DGCNN(
        k=cc["k"],
        in_channels=cc["in_channels"],
        cond_dim=cc["cond_dim"],
        scalar_dim=cc["scalar_dim"],
        field_dim=cc["field_dim"],
        edge_channels=tuple(cc["edge_channels"]),
        dropout=cc["dropout"],
        norm_type=cc.get("norm_type", "group"),
        norm_groups=int(cc.get("norm_groups", 8)),
        use_mean_pool=bool(cc.get("use_mean_pool", True)),
        fourier_levels=int(cc.get("fourier_levels", 6)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


# ── Training curves ───────────────────────────────────────────────────────────

def plot_history(history: list[dict], save_path: Path) -> None:
    """
    2×2 publication figure:
      (a) Total loss   (b) Scalar loss (thrust+drag)
      (c) Field loss   (d) Physics / grad losses
    Train and val on same axes, log-scale y, legend in each panel.
    """
    epochs = [r["epoch"] for r in history]

    def _safe(key: str) -> list[float]:
        return [max(r.get(key, 1e-12), 1e-12) for r in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_total, ax_scalar, ax_field, ax_phys = axes.flat

    # (a) Total loss
    ax_total.plot(epochs, _safe("train_loss"), color=BODY_COLOR, label="Train")
    ax_total.plot(epochs, _safe("val_loss"), color=EDF_COLOR, linestyle="--", label="Val")
    ax_total.set_yscale("log")
    ax_total.set_xlabel("Epoch [-]")
    ax_total.set_ylabel("Total loss [-]")
    ax_total.legend(loc="best")
    panel_label(ax_total, "a")

    # (b) Scalar loss (thrust + drag)
    ax_scalar.plot(epochs, _safe("train_scalar"), color=BODY_COLOR, label="Train scalar")
    ax_scalar.plot(epochs, _safe("val_scalar"), color=EDF_COLOR, linestyle="--", label="Val scalar")
    # Physical MAE on secondary y if available
    if "val_thrust_mae_phys" in history[0]:
        ax_scalar.plot(epochs, [max(r["val_thrust_mae_phys"], 1e-12) for r in history],
                       color="0.35", linestyle=":", label="Val thrust MAE [N]")
    ax_scalar.set_yscale("log")
    ax_scalar.set_xlabel("Epoch [-]")
    ax_scalar.set_ylabel("Scalar loss [-]")
    ax_scalar.legend(loc="best")
    panel_label(ax_scalar, "b")

    # (c) Field loss
    ax_field.plot(epochs, _safe("train_field"), color=BODY_COLOR, label="Train field")
    ax_field.plot(epochs, _safe("val_field"), color=EDF_COLOR, linestyle="--", label="Val field")
    if "train_field_grad" in history[0]:
        ax_field.plot(epochs, _safe("train_field_grad"), color=BODY_COLOR, linestyle="-.", label="Train field-grad")
        ax_field.plot(epochs, _safe("val_field_grad"), color=EDF_COLOR, linestyle=":", label="Val field-grad")
    ax_field.set_yscale("log")
    ax_field.set_xlabel("Epoch [-]")
    ax_field.set_ylabel("Field loss [-]")
    ax_field.legend(loc="best", ncol=2)
    panel_label(ax_field, "c")

    # (d) Physics losses (mass, BC, thrust physics term)
    has_phys = any(r.get("train_mass", 0) + r.get("train_bc", 0) + r.get("train_thrust", 0) > 1e-12 for r in history)
    if has_phys:
        ax_phys.plot(epochs, _safe("train_thrust"), color=BODY_COLOR, label="Train thrust-PI")
        ax_phys.plot(epochs, _safe("val_thrust"), color=EDF_COLOR, linestyle="--", label="Val thrust-PI")
        ax_phys.plot(epochs, _safe("train_mass"), color="0.30", label="Train mass")
        ax_phys.plot(epochs, _safe("train_bc"), color="purple", linestyle="-.", label="Train BC")
    else:
        # Show LR decay instead when no physics terms active
        lrs = [r["lr"] for r in history]
        ax_phys.plot(epochs, lrs, color=BODY_COLOR, label="LR")
        ax_phys.set_ylabel("Learning rate [-]")
    ax_phys.set_yscale("log")
    ax_phys.set_xlabel("Epoch [-]")
    if has_phys:
        ax_phys.set_ylabel("Physics loss [-]")
        ax_phys.legend(loc="best")
    panel_label(ax_phys, "d")

    png_path, pdf_path = save_figure_png_pdf(fig, save_path)
    plt.close(fig)
    print(f"Training curves saved: {png_path}")
    print(f"Training curves saved: {pdf_path}")


# ── Main training loop ────────────────────────────────────────────────────────

def train(c: Config = cfg) -> None:
    torch.manual_seed(c.seed)

    device       = c.resolve_device()
    use_amp      = c.use_amp and device.type == "cuda"
    out_dir      = c.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Device: {device}  AMP: {use_amp}")
    print(f"[train] Output: {out_dir}")
    print(f"[train] Normalization: type={c.norm_type} groups={c.norm_groups}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, ds = make_loaders(c, seed=c.seed)
    norm_stats = ds.norm_stats()
    scalar_scale = torch.tensor(norm_stats["scalar_scale"], dtype=torch.float32, device=device)
    loss_manager = HybridPhysicsLoss(c, norm_stats).to(device)

    # Physics scale tensors (float32, kept on CPU; moved to device inside physics loss)
    _field_scale   = torch.tensor(norm_stats["field_scale"],   dtype=torch.float32)
    _mid_xyz_scale = torch.tensor(norm_stats["mid_xyz_scale"], dtype=torch.float32)

    use_phys      = bool(getattr(c, "use_autograd_physics", False))
    use_phys_nemo = bool(getattr(c, "use_physicsnemo", False))

    autograd_physics = AutogradPhysicsLoss(
        lambda_continuity=float(getattr(c, "lambda_continuity", 0.0)),
        lambda_wall_bc=float(getattr(c, "lambda_wall_bc", 0.0)),
    ).to(device) if use_phys else None

    physicsnemo_loss = PhysicsNeMoLoss(
        nu=float(getattr(c, "fluid_nu", 1.5e-5)),
        rho=float(getattr(c, "fluid_rho", 1.225)),
        lambda_continuity=float(getattr(c, "lambda_continuity", 0.0)),
        lambda_momentum=float(getattr(c, "lambda_momentum", 0.0)),
        lambda_wall_bc=float(getattr(c, "lambda_wall_bc", 0.0)),
    ).to(device) if use_phys_nemo else None

    if use_phys:
        print(
            f"[train] Tier-1 AutogradPhysics: lambda_continuity={c.lambda_continuity}  "
            f"lambda_wall_bc={c.lambda_wall_bc}  n_wall_pts={c.n_wall_pts}"
        )
    if use_phys_nemo:
        print(
            f"[train] Tier-2 PhysicsNeMo: lambda_continuity={c.lambda_continuity}  "
            f"lambda_momentum={c.lambda_momentum}  lambda_wall_bc={c.lambda_wall_bc}  "
            f"nu={c.fluid_nu}  rho={c.fluid_rho}"
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DGCNN(
        k=c.k,
        in_channels=c.in_channels,
        cond_dim=c.cond_dim,
        scalar_dim=c.scalar_dim,
        field_dim=c.field_dim,
        edge_channels=c.edge_channels,
        dropout=c.dropout,
        norm_type=c.norm_type,
        norm_groups=c.norm_groups,
        use_mean_pool=c.use_mean_pool,
        fourier_levels=int(getattr(c, "fourier_levels", 6)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fl = int(getattr(c, "fourier_levels", 6))
    print(f"[train] Trainable parameters: {n_params:,}")
    print(f"[train] Global pool:          {'max+mean' if c.use_mean_pool else 'max only'}")
    print(f"[train] Fourier levels:       {fl}  (pos_dim={model.pos_encoder.out_dim})")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    opt_kind = str(getattr(c, "optimizer_type", "adam")).lower()
    wd = float(getattr(c, "weight_decay", 0.0))
    if opt_kind == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=wd)
        print(f"[train] Optimizer: AdamW(lr={c.learning_rate:.1e}, weight_decay={wd:.1e})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
        print(f"[train] Optimizer: Adam(lr={c.learning_rate:.1e})")

    # EMA shadow weights – used for evaluation / checkpoint selection when enabled.
    # The live model keeps training on its raw gradient-driven weights; the EMA
    # is a smoothed trajectory-average that typically generalises 10-30% better
    # and is almost free (one clone per step).
    use_ema = bool(getattr(c, "use_ema", False))
    ema = ModelEMA(model, decay=float(getattr(c, "ema_decay", 0.999))) if use_ema else None
    if use_ema:
        print(f"[train] EMA enabled  (decay={c.ema_decay:.4f})")

    sched_type = str(getattr(c, "scheduler_type", "cosine_warm")).lower()
    if sched_type == "cosine_warm":
        T0 = max(1, int(getattr(c, "cosine_T0", 50)))
        eta_min = float(getattr(c, "cosine_eta_min", 1e-6))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T0, T_mult=1, eta_min=eta_min,
        )
        print(f"[train] Scheduler: CosineAnnealingWarmRestarts(T_0={T0}, eta_min={eta_min:.1e})")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=c.lr_factor,
            patience=c.lr_patience, min_lr=c.lr_min,
        )
        print(f"[train] Scheduler: ReduceLROnPlateau(patience={c.lr_patience}, factor={c.lr_factor})")
    scaler = GradScaler(enabled=use_amp)

    # ── CSV loss log setup ────────────────────────────────────────────────────
    csv_path = out_dir / "loss_log.csv"
    csv_fields = [
        "epoch", "lr",
        "train_loss", "train_scalar", "train_field", "train_field_grad",
        "train_thrust", "train_mass", "train_bc",
        "val_loss", "val_scalar", "val_field", "val_field_grad",
        "val_thrust", "val_mass", "val_bc",
        "val_thrust_mae_norm", "val_drag_mae_norm",
        "val_thrust_mae_phys", "val_drag_mae_phys",
    ]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields, extrasaction="ignore")
    csv_writer.writeheader()

    # ── Early stopping state ──────────────────────────────────────────────────
    es_patience = int(getattr(c, "early_stopping_patience", 30))
    es_best = float("inf")
    es_counter = 0
    ckpt_interval = int(getattr(c, "checkpoint_interval", 25))

    # ── Training loop ─────────────────────────────────────────────────────────
    print(
        f"[train] Checkpoint metric: {c.checkpoint_metric} "
        f"(w_thrust={c.checkpoint_thrust_weight:.3f}, w_drag={c.checkpoint_drag_weight:.3f})"
    )
    print(f"[train] Early stopping patience: {es_patience}  Checkpoint interval: {ckpt_interval}")
    best_score = float("inf")
    history: list[dict] = []
    t0 = time.time()
    stopped_early = False

    skip_p1 = bool(getattr(c, "skip_phase1", False))
    if skip_p1:
        print("[train] skip_phase1=True — skipping Phase 1, loading existing best.pt")
        if not c.best_checkpoint.exists():
            raise FileNotFoundError(
                f"skip_phase1 requires an existing checkpoint at {c.best_checkpoint}"
            )

    for epoch in range(1, c.epochs + 1) if not skip_p1 else []:
        model.train()
        totals = dict(
            loss_total=0.0,
            loss_data=0.0,
            loss_scalar=0.0,
            loss_field=0.0,
            loss_field_grad=0.0,
            loss_thrust=0.0,
            loss_mass=0.0,
            loss_bc=0.0,
            loss_momentum=0.0,
            physics_ramp=0.0,
            spatial_weight_mean=0.0,
            loss_continuity=0.0,
            loss_wall_bc=0.0,
        )
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:4d}/{c.epochs}", leave=False, ncols=100)
        for batch in pbar:
            geo      = batch["geometry_points"].to(device, non_blocking=True)
            mid_xyz  = batch["midplane_xyz"].to(device, non_blocking=True)
            mid_flds = batch["midplane_fields"].to(device, non_blocking=True)
            scalar_t = batch["scalar_targets"].to(device, non_blocking=True)
            cond     = batch["conditions"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                if use_phys or use_phys_nemo:
                    # forward_with_embedding returns (scalar, fields, g) so the
                    # encoder runs once; g is then detached for the physics pass.
                    scalar_p, field_p, g_embed = model.forward_with_embedding(
                        geo, mid_xyz, cond
                    )
                else:
                    scalar_p, field_p = model(geo, mid_xyz, cond)
                    g_embed = None

                loss, terms = loss_manager(
                    scalar_p=scalar_p,
                    scalar_t=scalar_t,
                    field_p=field_p,
                    field_t=mid_flds,
                    mid_xyz=mid_xyz,
                    batch=batch,
                    epoch=epoch,
                )

            # Physics residual pass. This block runs inside the same autocast
            # context as the supervised pass above — autograd for the PDE
            # residuals is performed internally in the physics modules with
            # explicit float32 casts where needed.
            # Gradients flow only through the field head (encoder g is detached).
            loss_cont = loss.new_zeros(())
            loss_wall = loss.new_zeros(())
            loss_mom  = loss.new_zeros(())

            if use_phys and g_embed is not None:
                wall_pts = batch.get("wall_pts")
                if wall_pts is not None:
                    wall_pts = wall_pts.to(device, non_blocking=True)
                phys_total, phys_terms = autograd_physics.compute(
                    model=model,
                    g=g_embed,
                    mid_xyz_norm=mid_xyz,
                    field_scale=_field_scale,
                    mid_xyz_scale=_mid_xyz_scale,
                    wall_pts=wall_pts,
                )
                loss = loss + phys_total
                loss_cont = phys_terms["loss_continuity"]
                loss_wall = phys_terms["loss_wall_bc"]

            if use_phys_nemo and g_embed is not None:
                wall_pts = batch.get("wall_pts")
                if wall_pts is not None:
                    wall_pts = wall_pts.to(device, non_blocking=True)
                nemo_total, nemo_terms = physicsnemo_loss.compute(
                    model=model,
                    g=g_embed,
                    mid_xyz_norm=mid_xyz,
                    field_scale=_field_scale,
                    mid_xyz_scale=_mid_xyz_scale,
                    wall_pts=wall_pts,
                )
                loss      = loss + nemo_total
                loss_cont = nemo_terms["loss_continuity"]
                loss_mom  = nemo_terms["loss_momentum"]
                loss_wall = nemo_terms["loss_wall_bc"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), c.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            totals["loss_total"] += terms["loss_total"].item()
            totals["loss_data"] += terms["loss_data"].item()
            totals["loss_scalar"] += terms["loss_scalar"].item()
            totals["loss_field"] += terms["loss_field"].item()
            totals["loss_field_grad"] += terms["loss_field_grad"].item()
            totals["loss_thrust"] += terms["loss_thrust"].item()
            totals["loss_mass"] += terms["loss_mass"].item()
            totals["loss_bc"] += terms["loss_bc"].item()
            totals["loss_momentum"] += terms["loss_momentum"].item()
            totals["physics_ramp"] += terms["physics_ramp"].item()
            totals["spatial_weight_mean"] += terms["spatial_weight_mean"].item()
            totals["loss_continuity"] += loss_cont.item()
            totals["loss_wall_bc"]    += loss_wall.item()
            totals["loss_momentum"]   += loss_mom.item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{terms['loss_total'].item():.4f}",
                data=f"{terms['loss_data'].item():.4f}",
                cont=f"{loss_cont.item():.2e}",
                wall=f"{loss_wall.item():.2e}",
            )

        pbar.close()

        n = max(1, n_batches)
        train_loss = totals["loss_total"] / n
        train_data = totals["loss_data"] / n
        train_scalar = totals["loss_scalar"] / n
        train_field = totals["loss_field"] / n
        train_field_grad = totals["loss_field_grad"] / n
        train_thrust = totals["loss_thrust"] / n
        train_mass = totals["loss_mass"] / n
        train_bc = totals["loss_bc"] / n
        train_momentum = totals["loss_momentum"] / n
        train_ramp = totals["physics_ramp"] / n
        train_spatial_w = totals["spatial_weight_mean"] / n

        if ema is not None:
            ema.apply_to(model)
        try:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                c,
                use_amp,
                loss_manager=loss_manager,
                scalar_scale=scalar_scale,
                epoch=epoch,
            )
        finally:
            if ema is not None:
                ema.restore(model)

        # Scheduler step
        if sched_type == "cosine_warm":
            scheduler.step(epoch)
        else:
            scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch":          epoch,
            "lr":             current_lr,
            "train_loss":     train_loss,
            "train_data":     train_data,
            "train_scalar":   train_scalar,
            "train_field":    train_field,
            "train_field_grad": train_field_grad,
            "train_thrust":   train_thrust,
            "train_mass":     train_mass,
            "train_bc":       train_bc,
            "train_momentum": train_momentum,
            "train_ramp":     train_ramp,
            "train_spatial_weight_mean": train_spatial_w,
            "val_loss":       val_metrics["loss"],
            "val_data":       val_metrics["loss_data"],
            "val_scalar":     val_metrics["scalar"],
            "val_field":      val_metrics["field"],
            "val_field_grad": val_metrics["loss_field_grad"],
            "val_thrust":     val_metrics["loss_thrust"],
            "val_mass":       val_metrics["loss_mass"],
            "val_bc":         val_metrics["loss_bc"],
            "val_momentum":   val_metrics["loss_momentum"],
            "val_ramp":       val_metrics["physics_ramp"],
            "val_spatial_weight_mean": val_metrics["spatial_weight_mean"],
            "val_thrust_mae": val_metrics["thrust_mae"],
            "val_drag_mae":   val_metrics["drag_mae"],
            "val_thrust_mae_phys": val_metrics["thrust_mae_phys"],
            "val_drag_mae_phys": val_metrics["drag_mae_phys"],
        }
        history.append(row)

        # CSV log – written every epoch so partial runs are recoverable
        csv_row = {
            "epoch": epoch, "lr": current_lr,
            "train_loss": train_loss, "train_scalar": train_scalar,
            "train_field": train_field, "train_field_grad": train_field_grad,
            "train_thrust": train_thrust, "train_mass": train_mass, "train_bc": train_bc,
            "val_loss": val_metrics["loss"], "val_scalar": val_metrics["scalar"],
            "val_field": val_metrics["field"], "val_field_grad": val_metrics["loss_field_grad"],
            "val_thrust": val_metrics["loss_thrust"], "val_mass": val_metrics["loss_mass"],
            "val_bc": val_metrics["loss_bc"],
            "val_thrust_mae_norm": val_metrics["thrust_mae"],
            "val_drag_mae_norm": val_metrics["drag_mae"],
            "val_thrust_mae_phys": val_metrics["thrust_mae_phys"],
            "val_drag_mae_phys": val_metrics["drag_mae_phys"],
        }
        csv_writer.writerow(csv_row)
        csv_file.flush()

        score = checkpoint_score(val_metrics, c)

        # Save best model according to checkpoint score policy. When EMA is
        # on, checkpoint the EMA shadow weights (which is what val_metrics
        # above were measured against) – not the live training weights.
        improved = ""
        if score < best_score:
            best_score = score
            if ema is not None:
                ema.apply_to(model)
                try:
                    save_checkpoint(c.best_checkpoint, model, optimizer, epoch,
                                    val_metrics, norm_stats, c)
                finally:
                    ema.restore(model)
            else:
                save_checkpoint(c.best_checkpoint, model, optimizer, epoch,
                                val_metrics, norm_stats, c)
            improved = " [best]"

        # Periodic checkpoint every N epochs (live weights – useful for resuming)
        if ckpt_interval > 0 and epoch % ckpt_interval == 0:
            periodic_path = c.checkpoint_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(periodic_path, model, optimizer, epoch, val_metrics, norm_stats, c)

        # Early stopping on total val loss (scalar + field + physics).
        # Watching only scalar MAE caused premature stopping once scalar saturated,
        # leaving the field head stuck at mean prediction with no more training time.
        es_metric = val_metrics["loss"]
        if es_metric < es_best - 1e-8:
            es_best = es_metric
            es_counter = 0
        else:
            es_counter += 1
            if es_patience > 0 and es_counter >= es_patience:
                print(
                    f"[train] Early stopping at epoch {epoch} "
                    f"(val total loss not improved for {es_patience} epochs)."
                )
                # Print one-liner before breaking so final row is visible
                print(
                    f"Epoch {epoch:4d}  train={train_loss:.5f}  val={val_metrics['loss']:.5f}  "
                    f"thrust_MAE_N={val_metrics['thrust_mae_phys']:.5f}  "
                    f"drag_MAE_N={val_metrics['drag_mae_phys']:.5f}  lr={current_lr:.2e}"
                    f"{improved}"
                )
                stopped_early = True
                break

        print(
            f"Epoch {epoch:4d}  "
            f"train={train_loss:.5f} (s={train_scalar:.5f} f={train_field:.5f} fg={train_field_grad:.5f})  "
            f"phys(tr={train_thrust:.5f} m={train_mass:.5f} bc={train_bc:.5f} r={train_ramp:.2f})  "
            f"val={val_metrics['loss']:.5f} (s={val_metrics['scalar']:.5f} f={val_metrics['field']:.5f})  "
            f"thrust_MAE_N={val_metrics['thrust_mae_phys']:.5f}  "
            f"drag_MAE_N={val_metrics['drag_mae_phys']:.5f}  "
            f"lr={current_lr:.2e}"
            f"{improved}"
        )

    csv_file.close()
    elapsed = time.time() - t0
    stop_reason = "early stopping" if stopped_early else f"{c.epochs} epochs"
    print(f"\n[train] Done ({stop_reason}). Best checkpoint score ({c.checkpoint_metric}): {best_score:.6f}  "
          f"Time: {elapsed/60:.1f} min")
    print(f"[train] Best checkpoint: {c.best_checkpoint}")
    print(f"[train] Loss log CSV:    {csv_path}")

    # Keep explicit global-epoch continuity across post-train phases so
    # history/plots remain interpretable even when earlier phases stop early.
    phase1_last_epoch = int(history[-1]["epoch"]) if history else int(c.epochs)
    phase2_last_epoch = phase1_last_epoch
    best_phase2_val_field = float("inf")

    # ── Phase 2: field fine-tune ──────────────────────────────────────────────
    # The Adam optimizer accumulates large second moments for shared encoder
    # parameters during scalar training.  Once scalar converges, those stale
    # moments give the field head a tiny effective LR, causing it to stall at
    # mean prediction.  Freezing the encoder and resetting Adam lets the field
    # head escape with a fresh, high LR.
    ff_epochs = int(getattr(c, "field_finetune_epochs", 0))
    ff_lr     = float(getattr(c, "field_finetune_lr", 5e-3))
    if ff_epochs > 0:
        print(f"\n[train] -- Phase 2: field fine-tune  ({ff_epochs} epochs, LR={ff_lr:.1e}) --")

        # Important: start phase-2 from the best phase-1 checkpoint, not the
        # final phase-1 epoch state. This preserves the best scalar solution.
        model, _ = load_checkpoint(c.best_checkpoint, device)
        model.to(device)
        print(f"[train] Loaded Phase-1 checkpoint from {c.best_checkpoint}")

        # Phase-2 keeps midplane resampling ON so the SIREN learns over the full
        # ~88K-point pool (not just the first 2048). Disabling it caused the
        # SIREN to memorise a fixed subset; evaluate then queried different
        # points and RMSE blew up 300×. Geometry jitter is still disabled
        # because the SIREN doesn't benefit from encoder-side noise.
        import dataclasses as _dc
        c_no_aug = _dc.replace(c, augment_jitter_std=0.0)
        train_loader_p2, val_loader_p2, _ = make_loaders(c_no_aug, seed=c.seed)
        print("[train] Phase-2 loaders: augmentation disabled (fixed midplane subset).")

        # Re-initialise field head weights from scratch using SIREN-specific init.
        # After joint training, field_head weights are shaped by the scalar-optimised
        # encoder embedding.  Freezing the encoder leaves those weights misaligned
        # for pure coordinate-MLP learning, causing an initial loss spike at phase-2
        # LR before they recover.  A fresh init gives a clean starting point.
        model.reset_field_head()
        print("[train] Field head weights re-initialised (SIREN uniform init).")

        # Freeze everything except the FiLM-SIREN field head layers
        _field_modules = {"fs1", "fs2", "fs3", "fs4", "fs5", "field_out", "pos_encoder"}
        for name, p in model.named_parameters():
            p.requires_grad = any(name.startswith(m + ".") or name == m for m in _field_modules)
        field_params = [p for p in model.parameters() if p.requires_grad]
        n_field_params = sum(p.numel() for p in field_params)
        print(f"[train] Field-head params (trainable): {n_field_params:,}")

        ff_optimizer = torch.optim.Adam(field_params, lr=ff_lr)
        # MultiStepLR with 3 decay milestones at 25%/50%/75% of fine-tune epochs.
        # The standalone SIREN test showed MultiStepLR(milestones=[500,1000,1500],
        # gamma=0.3) converges to RMSE_norm=0.00009 in <600 epochs at LR=5e-5.
        # CosineAnnealing caused oscillations that prevented the SIREN from settling.
        _m1 = max(1, ff_epochs // 4)
        _m2 = max(2, ff_epochs // 2)
        _m3 = max(3, 3 * ff_epochs // 4)
        ff_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            ff_optimizer, milestones=[_m1, _m2, _m3], gamma=0.3
        )
        ff_scaler = GradScaler(enabled=use_amp)

        # Phase-2 EMA: use a faster decay than Phase-1 so the EMA can track
        # through instability spikes without accumulating corrupted history.
        # Default 0.99 (window ~100 steps) vs Phase-1's 0.999 (~1000 steps).
        p2_ema_decay = float(getattr(c, "phase2_ema_decay", 0.99))
        ff_ema = ModelEMA(model, decay=p2_ema_decay) if use_ema else None

        # Rank Phase-2 checkpoints by a combined metric that includes the
        # scalar head (which is frozen but its *val* number can change if
        # val set differs from train; also guards against ever accepting a
        # field-only improvement that silently regresses the scalar). This
        # matches Phase-1's selection criterion (normalised combined MAE).
        phase2_start_epoch = phase1_last_epoch
        ff_best_val_field = float("inf")
        ff_best_combined = float("inf")
        ff_best_live_field = float("inf")   # live (non-EMA) model best
        ff_live_checkpoint = c.checkpoint_dir / "best_live.pt"
        ff_es_counter = 0
        ff_patience   = max(100, ff_epochs // 6)   # SIREN needs patience for LR decay to kick in

        for ff_ep in range(1, ff_epochs + 1):
            model.train()
            ep_field = 0.0
            nb = 0
            for batch in train_loader_p2:
                geo      = batch["geometry_points"].to(device, non_blocking=True)
                mid_xyz  = batch["midplane_xyz"].to(device, non_blocking=True)
                mid_flds = batch["midplane_fields"].to(device, non_blocking=True)
                scalar_t = batch["scalar_targets"].to(device, non_blocking=True)
                cond     = batch["conditions"].to(device, non_blocking=True)

                ff_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    scalar_p, field_p = model(geo, mid_xyz, cond)
                    field_loss = torch.mean((field_p - mid_flds) ** 2)

                ff_scaler.scale(field_loss).backward()
                ff_scaler.unscale_(ff_optimizer)
                clip_grad_norm_(field_params, c.grad_clip)
                ff_scaler.step(ff_optimizer)
                ff_scaler.update()
                if ff_ema is not None:
                    ff_ema.update(model)
                ep_field += field_loss.item()
                nb += 1

            ff_scheduler.step()
            avg_field = ep_field / max(1, nb)
            epoch_global = phase2_start_epoch + ff_ep

            # Evaluate on EMA weights if enabled, then score by combined
            # (field + scalar-MAE) metric so a field drop that silently broke
            # a scalar is not picked as 'best'.
            if ff_ema is not None:
                ff_ema.apply_to(model)
            try:
                val_metrics_ff = evaluate(model, val_loader_p2, device, c,
                                          use_amp, loss_manager, scalar_scale,
                                          epoch=epoch_global)
            finally:
                if ff_ema is not None:
                    ff_ema.restore(model)

            ff_val_field = float(val_metrics_ff["field"])
            ff_scalar_mae = (
                float(c.checkpoint_thrust_weight) * float(val_metrics_ff["thrust_mae"])
                + float(c.checkpoint_drag_weight)   * float(val_metrics_ff["drag_mae"])
            )
            ff_combined = ff_val_field + ff_scalar_mae

            if ff_combined < ff_best_combined - 1e-9:
                ff_best_combined = ff_combined
                ff_best_val_field = ff_val_field
                ff_es_counter = 0
                if ff_ema is not None:
                    ff_ema.apply_to(model)
                    try:
                        save_checkpoint(c.best_checkpoint, model, ff_optimizer,
                                        epoch_global, val_metrics_ff, norm_stats, c)
                    finally:
                        ff_ema.restore(model)
                else:
                    save_checkpoint(c.best_checkpoint, model, ff_optimizer,
                                    epoch_global, val_metrics_ff, norm_stats, c)
                improved_mark = " *"
            else:
                ff_es_counter += 1
                improved_mark = ""
                if ff_es_counter >= ff_patience:
                    print(f"[train] Field fine-tune early stopping at sub-epoch {ff_ep}.")
                    break

            # Also track live model (non-EMA) best using training loss as proxy.
            # Saves us if EMA was corrupted by an instability spike.
            if avg_field < ff_best_live_field - 1e-9:
                ff_best_live_field = avg_field
                save_checkpoint(ff_live_checkpoint, model, ff_optimizer,
                                epoch_global, val_metrics_ff, norm_stats, c)

            if ff_ep % 20 == 0 or ff_ep == 1 or ff_ep == ff_epochs:
                ff_lr_now = ff_optimizer.param_groups[0]["lr"]
                print(f"  [field-ft] sub-ep {ff_ep:4d}/{ff_epochs}  "
                      f"field_loss={avg_field:.6f}  "
                      f"RMSE_norm={avg_field**0.5:.5f}  "
                      f"lr={ff_lr_now:.2e}{improved_mark}")

            # Append to history for plot
            history.append({
                "epoch": epoch_global,
                "lr": ff_optimizer.param_groups[0]["lr"],
                "train_loss": avg_field, "train_data": avg_field,
                "train_scalar": 0.0, "train_field": avg_field,
                "train_field_grad": 0.0, "train_thrust": 0.0,
                "train_mass": 0.0, "train_bc": 0.0,
                "train_momentum": 0.0, "train_ramp": 1.0,
                "train_spatial_weight_mean": 1.0,
                "val_loss": float(val_metrics_ff["loss"]),
                "val_data": float(val_metrics_ff["loss_data"]),
                "val_scalar": float(val_metrics_ff["scalar"]),
                "val_field": ff_val_field,
                "val_field_grad": float(val_metrics_ff["loss_field_grad"]),
                "val_thrust": float(val_metrics_ff["loss_thrust"]),
                "val_mass": float(val_metrics_ff["loss_mass"]),
                "val_bc": float(val_metrics_ff["loss_bc"]),
                "val_momentum": float(val_metrics_ff["loss_momentum"]),
                "val_ramp": float(val_metrics_ff["physics_ramp"]),
                "val_spatial_weight_mean": float(val_metrics_ff["spatial_weight_mean"]),
                "val_thrust_mae": float(val_metrics_ff["thrust_mae"]),
                "val_drag_mae": float(val_metrics_ff["drag_mae"]),
                "val_thrust_mae_phys": float(val_metrics_ff["thrust_mae_phys"]),
                "val_drag_mae_phys": float(val_metrics_ff["drag_mae_phys"]),
            })

        phase2_last_epoch = int(history[-1]["epoch"]) if history else phase2_start_epoch
        if ff_best_val_field < float("inf"):
            best_phase2_val_field = ff_best_val_field

        # Compare EMA best vs live-model best; keep whichever gives lower field loss.
        if ff_live_checkpoint.exists() and ff_best_live_field < ff_best_val_field:
            import shutil as _shutil
            _shutil.copy2(ff_live_checkpoint, c.best_checkpoint)
            print(f"[train] Live model outperformed EMA — using live checkpoint "
                  f"(live={ff_best_live_field**0.5:.5f} vs ema={ff_best_val_field**0.5:.5f})")
            ff_best_val_field = ff_best_live_field

        # Unfreeze all parameters for any downstream use
        for p in model.parameters():
            p.requires_grad = True
        print(f"[train] Phase 2 complete.  Best val_field_loss={ff_best_val_field:.6f}  "
              f"RMSE_norm={ff_best_val_field**0.5:.5f}")

    # ── Phase 3: Physics fine-tune ────────────────────────────────────────────
    # Activates Tier-1 AutogradPhysicsLoss (continuity + wall BC) on top of
    # the converged Phase-2 field head.  The scalar pathway remains frozen; the
    # physics signal is weak (lambda ~0.01) so the field accuracy is preserved
    # while the prediction is steered toward ∇·u = 0 and no-slip BC.
    p3_epochs     = int(getattr(c, "physics_finetune_epochs", 0))
    p3_lr         = float(getattr(c, "physics_finetune_lr", 1e-5))
    p3_lambda_cont = float(getattr(c, "physics_finetune_lambda_cont", 0.01))
    p3_lambda_wall = float(getattr(c, "physics_finetune_lambda_wall", 0.10))

    if p3_epochs > 0:
        phase3_start_epoch = phase2_last_epoch
        print(f"\n[train] -- Phase 3: physics fine-tune  "
              f"({p3_epochs} epochs, LR={p3_lr:.1e}, "
              f"lambda_cont={p3_lambda_cont}, lambda_wall={p3_lambda_wall}) --")

        # Reload best Phase-2 checkpoint so we start from the cleanest field
        model, ckpt_loaded = load_checkpoint(c.best_checkpoint, device)
        model.to(device)
        print(f"[train] Loaded Phase-2 checkpoint from {c.best_checkpoint}")

        # Keep scalar pathway frozen in phase-3 so continuity/wall updates do
        # not regress thrust/drag accuracy.
        _field_modules_p3 = {"fs1", "fs2", "fs3", "fs4", "fs5", "field_out", "pos_encoder"}
        for name, p in model.named_parameters():
            p.requires_grad = any(name.startswith(m + ".") or name == m for m in _field_modules_p3)
        p3_params = [p for p in model.parameters() if p.requires_grad]
        n_p3_params = sum(p.numel() for p in p3_params)
        print(f"[train] Phase-3 trainable params (field head only): {n_p3_params:,}")

        p3_phys = AutogradPhysicsLoss(
            lambda_continuity=p3_lambda_cont,
            lambda_wall_bc=p3_lambda_wall,
        )

        p3_optimizer = torch.optim.Adam(p3_params, lr=p3_lr)
        p3_scaler    = GradScaler(enabled=use_amp)

        # Evaluate once before Phase-3 updates; used as the field-quality guard.
        p3_baseline_val = evaluate(
            model, val_loader, device, c, use_amp, loss_manager, scalar_scale, epoch=phase3_start_epoch
        )
        p3_field_ref = float(p3_baseline_val["field"])
        if best_phase2_val_field < float("inf"):
            p3_field_ref = float(best_phase2_val_field)
        p3_field_guard_ratio = float(getattr(c, "physics_finetune_field_guard_ratio", 1.10))
        p3_field_guard = max(1e-12, p3_field_ref * p3_field_guard_ratio)
        print(
            f"[train] Phase-3 field guard: val_field <= {p3_field_guard:.6f} "
            f"(ref={p3_field_ref:.6f}, ratio={p3_field_guard_ratio:.3f})"
        )

        p3_best   = float("inf")
        p3_es_cnt = 0
        p3_patience = max(30, p3_epochs // 5)

        for p3_ep in range(1, p3_epochs + 1):
            model.train()
            ep_field = 0.0
            ep_cont  = 0.0
            ep_wall  = 0.0
            nb = 0

            for batch in train_loader:
                geo      = batch["geometry_points"].to(device, non_blocking=True)
                mid_xyz_ = batch["midplane_xyz"].to(device, non_blocking=True)
                mid_flds = batch["midplane_fields"].to(device, non_blocking=True)
                cond_    = batch["conditions"].to(device, non_blocking=True)
                wall_pts_ = batch["wall_pts"].to(device, non_blocking=True)

                p3_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    scalar_p, field_p, g = model.forward_with_embedding(geo, mid_xyz_, cond_)
                    field_loss = torch.mean((field_p - mid_flds) ** 2)

                    phys_total, phys_breakdown = p3_phys.compute(
                        model=model,
                        g=g,
                        mid_xyz_norm=mid_xyz_,
                        field_scale=torch.tensor(norm_stats["field_scale"],
                                                 device=device, dtype=torch.float32),
                        mid_xyz_scale=torch.tensor(norm_stats["mid_xyz_scale"],
                                                   device=device, dtype=torch.float32),
                        wall_pts=wall_pts_,
                    )
                    total_loss = field_loss + phys_total

                p3_scaler.scale(total_loss).backward()
                p3_scaler.unscale_(p3_optimizer)
                clip_grad_norm_(p3_params, c.grad_clip)
                p3_scaler.step(p3_optimizer)
                p3_scaler.update()

                ep_field += field_loss.item()
                ep_cont  += phys_breakdown["loss_continuity"].item()
                ep_wall  += phys_breakdown["loss_wall_bc"].item()
                nb += 1

            avg_field = ep_field / max(1, nb)
            avg_cont  = ep_cont  / max(1, nb)
            avg_wall  = ep_wall  / max(1, nb)
            train_combined = avg_field + p3_lambda_cont * avg_cont + p3_lambda_wall * avg_wall
            epoch_global = phase3_start_epoch + p3_ep

            val_m_p3 = evaluate(
                model, val_loader, device, c, use_amp, loss_manager, scalar_scale, epoch=epoch_global
            )
            val_field = float(val_m_p3["field"])
            val_cont = float(val_m_p3["loss_mass"])
            val_wall = float(val_m_p3["loss_bc"])
            val_combined = val_field + p3_lambda_cont * val_cont + p3_lambda_wall * val_wall
            field_guard_ok = val_field <= p3_field_guard

            if field_guard_ok and val_combined < p3_best - 1e-9:
                p3_best = val_combined
                p3_es_cnt = 0
                save_checkpoint(c.best_checkpoint, model, p3_optimizer,
                                epoch_global, val_m_p3, norm_stats, c)
                mark = " *"
            else:
                p3_es_cnt += 1
                mark = ""
                if (not field_guard_ok) and (val_combined < p3_best - 1e-9):
                    mark = " !"
                    if p3_ep == 1 or p3_ep % 10 == 0:
                        print(
                            f"[train][phase3] checkpoint blocked by field guard: "
                            f"val_field={val_field:.6f} > guard={p3_field_guard:.6f}"
                        )
                if p3_es_cnt >= p3_patience:
                    print(f"[train] Phase 3 early stopping at sub-epoch {p3_ep}.")
                    break

            if p3_ep % 10 == 0 or p3_ep == 1 or p3_ep == p3_epochs:
                print(f"  [phys-ft] ep {p3_ep:4d}/{p3_epochs}  "
                      f"field={avg_field:.6f}  cont={avg_cont:.4e}  "
                      f"wall={avg_wall:.4e}  val_field={val_field:.6f}  "
                      f"val_combined={val_combined:.6f}  lr={p3_lr:.1e}{mark}")

            history.append({
                "epoch": epoch_global,
                "lr": p3_lr, "train_loss": train_combined, "train_data": avg_field,
                "train_scalar": 0.0, "train_field": avg_field,
                "train_field_grad": 0.0, "train_thrust": 0.0,
                "train_mass": avg_cont, "train_bc": avg_wall,
                "train_momentum": 0.0, "train_ramp": 1.0,
                "train_spatial_weight_mean": 1.0,
                "val_loss": val_combined,
                "val_data": val_field,
                "val_scalar": float(val_m_p3["scalar"]),
                "val_field": val_field,
                "val_field_grad": float(val_m_p3["loss_field_grad"]),
                "val_thrust": float(val_m_p3["loss_thrust"]),
                "val_mass": val_cont,
                "val_bc": val_wall,
                "val_momentum": float(val_m_p3["loss_momentum"]),
                "val_ramp": float(val_m_p3["physics_ramp"]),
                "val_spatial_weight_mean": float(val_m_p3["spatial_weight_mean"]),
                "val_thrust_mae": float(val_m_p3["thrust_mae"]),
                "val_drag_mae": float(val_m_p3["drag_mae"]),
                "val_thrust_mae_phys": float(val_m_p3["thrust_mae_phys"]),
                "val_drag_mae_phys": float(val_m_p3["drag_mae_phys"]),
            })

        # Restore default trainability for any downstream use.
        for p in model.parameters():
            p.requires_grad = True
        print(f"[train] Phase 3 complete.  Best combined={p3_best:.6f}")

    # Save full history
    history_path = out_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save training curves
    plot_history(history, out_dir / "training_curves.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(defaults: Config = cfg) -> Config:
    p = argparse.ArgumentParser(description="Train DGCNN surrogate")
    p.add_argument("--data_root",   type=str,   default=str(defaults.data_root))
    p.add_argument("--output_dir",  type=str,   default=str(defaults.output_dir))
    p.add_argument("--epochs",      type=int,   default=defaults.epochs)
    p.add_argument("--batch_size",  type=int,   default=defaults.batch_size)
    p.add_argument("--lr",          type=float, default=defaults.learning_rate)
    p.add_argument("--k",           type=int,   default=defaults.k)
    p.add_argument("--dropout",     type=float, default=defaults.dropout)
    p.add_argument("--norm_type",   type=str,   default=defaults.norm_type, choices=["batch", "group", "layer"])
    p.add_argument("--norm_groups", type=int,   default=defaults.norm_groups)
    p.add_argument("--seed",        type=int,   default=defaults.seed)
    p.add_argument("--train_fraction", type=float, default=defaults.train_fraction)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--num_workers", type=int, default=defaults.num_workers)
    p.add_argument(
        "--checkpoint_metric",
        type=str,
        default=defaults.checkpoint_metric,
        choices=["combined_mae", "thrust_mae", "drag_mae", "loss"],
    )
    p.add_argument("--checkpoint_thrust_weight", type=float, default=defaults.checkpoint_thrust_weight)
    p.add_argument("--checkpoint_drag_weight", type=float, default=defaults.checkpoint_drag_weight)
    p.add_argument("--lambda_data",     type=float, default=defaults.lambda_data)
    p.add_argument("--lambda_scalars",  type=float, default=defaults.lambda_scalars)
    p.add_argument("--lambda_fields",   type=float, default=defaults.lambda_fields)
    p.add_argument("--lambda_thrust",   type=float, default=defaults.lambda_thrust)
    p.add_argument("--lambda_mass",     type=float, default=defaults.lambda_mass)
    p.add_argument("--lambda_bc",       type=float, default=defaults.lambda_bc)
    p.add_argument("--lambda_momentum", type=float, default=defaults.lambda_momentum)
    p.add_argument("--lambda_field_grad", type=float, default=defaults.lambda_field_grad)
    p.add_argument(
        "--thrust_loss_type",
        type=str,
        default=defaults.thrust_loss_type,
        choices=["huber", "mse", "l1"],
    )
    p.add_argument("--thrust_huber_delta", type=float, default=defaults.thrust_huber_delta)
    p.add_argument("--field_data_loss_type", type=str, default=defaults.field_data_loss_type, choices=["mse", "huber", "l1"])
    p.add_argument("--field_huber_delta", type=float, default=defaults.field_huber_delta)
    p.add_argument("--field_spatial_weight_alpha", type=float, default=defaults.field_spatial_weight_alpha)
    p.add_argument("--field_spatial_weight_clip", type=float, default=defaults.field_spatial_weight_clip)
    p.add_argument("--field_spatial_knn_k", type=int, default=defaults.field_spatial_knn_k)
    p.add_argument("--field_grad_loss_type", type=str, default=defaults.field_grad_loss_type, choices=["mse", "huber", "l1"])
    p.add_argument("--field_grad_huber_delta", type=float, default=defaults.field_grad_huber_delta)
    p.add_argument("--field_grad_knn_k", type=int, default=defaults.field_grad_knn_k)
    p.add_argument("--field_grad_relative", action="store_true", dest="field_grad_relative")
    p.add_argument("--no_field_grad_relative", action="store_false", dest="field_grad_relative")
    p.set_defaults(field_grad_relative=defaults.field_grad_relative)
    p.add_argument("--field_grad_relative_eps", type=float, default=defaults.field_grad_relative_eps)
    p.add_argument(
        "--slice_mode",
        type=str,
        default=defaults.physics_slice_mode,
        choices=["full3d", "quasi2d", "midplane3d"],
    )
    p.add_argument(
        "--physics_schedule",
        type=str,
        default=defaults.physics_schedule,
        choices=["none", "linear", "cosine"],
    )
    p.add_argument("--physics_warmup_epochs", type=int, default=defaults.physics_warmup_epochs)
    p.add_argument("--midplane3d_weak_factor", type=float, default=defaults.physics_midplane3d_weak_factor)
    p.add_argument("--physics_knn_k", type=int, default=defaults.physics_knn_k)
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument("--device",      type=str,   default=defaults.device)
    p.add_argument("--scheduler",   type=str,   default=getattr(defaults, "scheduler_type", "cosine_warm"),
                   choices=["cosine_warm", "plateau"])
    p.add_argument("--cosine_T0",   type=int,   default=getattr(defaults, "cosine_T0", 50))
    p.add_argument("--early_stopping_patience", type=int,
                   default=getattr(defaults, "early_stopping_patience", 60))
    p.add_argument("--checkpoint_interval", type=int,
                   default=getattr(defaults, "checkpoint_interval", 25))
    p.add_argument("--field_finetune_epochs", type=int,
                   default=getattr(defaults, "field_finetune_epochs", 500))
    p.add_argument("--field_finetune_lr", type=float,
                   default=getattr(defaults, "field_finetune_lr", 5e-3))
    p.add_argument("--no_mean_pool", action="store_true",
                   help="Disable mean-pool (use max-pool only) in global embedding")
    args = p.parse_args()

    c = Config(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        k=args.k,
        dropout=args.dropout,
        norm_type=args.norm_type,
        norm_groups=args.norm_groups,
        seed=args.seed,
        train_fraction=args.train_fraction,
        overfit_mode=args.overfit,
        num_workers=args.num_workers,
        checkpoint_metric=args.checkpoint_metric,
        checkpoint_thrust_weight=args.checkpoint_thrust_weight,
        checkpoint_drag_weight=args.checkpoint_drag_weight,
        lambda_data=args.lambda_data,
        lambda_scalars=args.lambda_scalars,
        lambda_fields=args.lambda_fields,
        lambda_thrust=args.lambda_thrust,
        lambda_mass=args.lambda_mass,
        lambda_bc=args.lambda_bc,
        lambda_momentum=args.lambda_momentum,
        lambda_field_grad=args.lambda_field_grad,
        thrust_loss_type=args.thrust_loss_type,
        thrust_huber_delta=args.thrust_huber_delta,
        field_data_loss_type=args.field_data_loss_type,
        field_huber_delta=args.field_huber_delta,
        field_spatial_weight_alpha=args.field_spatial_weight_alpha,
        field_spatial_weight_clip=args.field_spatial_weight_clip,
        field_spatial_knn_k=args.field_spatial_knn_k,
        field_grad_loss_type=args.field_grad_loss_type,
        field_grad_huber_delta=args.field_grad_huber_delta,
        field_grad_knn_k=args.field_grad_knn_k,
        field_grad_relative=args.field_grad_relative,
        field_grad_relative_eps=args.field_grad_relative_eps,
        physics_slice_mode=args.slice_mode,
        physics_schedule=args.physics_schedule,
        physics_warmup_epochs=args.physics_warmup_epochs,
        physics_midplane3d_weak_factor=args.midplane3d_weak_factor,
        physics_knn_k=args.physics_knn_k,
        use_amp=not args.no_amp,
        device=args.device,
        scheduler_type=args.scheduler,
        cosine_T0=args.cosine_T0,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_interval=args.checkpoint_interval,
        field_finetune_epochs=args.field_finetune_epochs,
        field_finetune_lr=args.field_finetune_lr,
        use_mean_pool=not args.no_mean_pool,
    )
    return c


if __name__ == "__main__":
    train(parse_args())
