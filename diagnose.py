"""
diagnose.py – Data, normalisation, and training isolation diagnostics.

Phases
──────
  1. Data audit:       load samples, print shapes/stats, verify duplicates, check NaN/Inf
  2. Norm audit:       trace normalisation pipeline for one sample
  3. Isolation train:  100 epochs scalar-only with GroupNorm + overfit mode
                       (should converge to near-zero on duplicate data)
  4. Field+scalar:     re-enable field loss, confirm both heads converge
  5. Print summary

Usage
─────
  python diagnose.py
  python diagnose.py --data_root dataset_pointcloud --epochs 100
  python diagnose.py --phase 1          # data audit only
  python diagnose.py --phase 3          # isolation train only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

import matplotlib
matplotlib.use("Agg")

# ── helpers ───────────────────────────────────────────────────────────────────

SEP = "=" * 72
SEP2 = "-" * 72

def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def subsection(title: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


# ── Phase 1: Data audit ───────────────────────────────────────────────────────

def phase1_data_audit(data_root: str) -> None:
    import json
    section("PHASE 1 — DATA AUDIT")
    root = Path(data_root)
    files = sorted(root.glob("sample_*.npz"))
    print(f"Found {len(files)} sample files in {root}")

    # Print detailed info for first 3 samples
    subsection("Sample-level array inspection (samples 0, 1, 2)")
    for idx in range(min(3, len(files))):
        fpath = files[idx]
        print(f"\n  [{fpath.name}]")
        with np.load(fpath, allow_pickle=True) as npz:
            for k in npz.files:
                v = npz[k]
                if v.dtype.kind in ("f", "i", "u"):
                    has_nan = bool(np.any(~np.isfinite(v)))
                    print(
                        f"    {k:35s}  shape={str(tuple(v.shape)):22s}  "
                        f"dtype={str(v.dtype):10s}  "
                        f"min={float(v.min()):+10.4g}  max={float(v.max()):+10.4g}  "
                        f"mean={float(v.mean()):+10.4g}  "
                        f"{'*** NaN/Inf ***' if has_nan else ''}"
                    )
                elif k == "meta":
                    meta = json.loads(v[0])
                    print(
                        f"    meta:  case_dir={meta.get('case_dir')}  "
                        f"u_inf={meta.get('u_inf')}  rpm={meta.get('rpm')}  "
                        f"body_drag={meta.get('body_drag'):.5f}  "
                        f"edf_thrust={meta.get('edf_thrust'):.5f}"
                    )

    # Confirm all samples have identical targets
    subsection("Cross-sample duplicate check")
    all_targets = []
    all_conds = []
    for fpath in files:
        with np.load(fpath, allow_pickle=True) as npz:
            all_targets.append(npz["targets_scalar_raw"].astype(np.float64))
            all_conds.append(npz["conditions"].astype(np.float64))
    T = np.stack(all_targets)
    C = np.stack(all_conds)
    print(f"\n  targets_scalar_raw [drag_N, thrust_N]:")
    print(f"    min across samples : {T.min(axis=0)}")
    print(f"    max across samples : {T.max(axis=0)}")
    print(f"    std across samples : {T.std(axis=0)}")
    print(f"    -> All identical:    {bool(T.std(axis=0).max() < 1e-5)}")
    print(f"\n  conditions [u_inf, rpm]:")
    print(f"    values             : {C[0]}")
    print(f"    std across samples : {C.std(axis=0)}")
    print(f"    -> All identical:    {bool(C.std(axis=0).max() < 1e-5)}")

    # NaN/Inf check across all samples
    subsection("NaN / Inf scan across all samples")
    any_bad = False
    for fpath in files:
        with np.load(fpath, allow_pickle=True) as npz:
            for k in npz.files:
                v = npz[k]
                if v.dtype.kind in ("f",) and not np.all(np.isfinite(v)):
                    print(f"  *** {fpath.name}/{k}: contains NaN or Inf ***")
                    any_bad = True
    if not any_bad:
        print("  All arrays are finite across all 20 samples.")


# ── Phase 2: Normalisation audit ─────────────────────────────────────────────

def phase2_norm_audit(data_root: str) -> None:
    section("PHASE 2 — NORMALISATION AUDIT")
    from config import Config
    from dataset import CFDDataset

    c = Config(data_root=Path(data_root), overfit_mode=True, norm_type="group")
    ds = CFDDataset(data_root=c.data_root, config=c)

    subsection("Normalisation scales")
    ns = ds.norm_stats()
    print(f"  scalar_scale (drag, thrust):  {ns['scalar_scale']}")
    print(f"  cond_scale   (u_inf, rpm):    {ns['cond_scale']}")
    print(f"  field_scale  (p, u, v, w):    {ns['field_scale']}")
    print(f"  mid_xyz_center:               {ns['mid_xyz_center']}")
    print(f"  mid_xyz_scale:                {ns['mid_xyz_scale']}")

    subsection("Sample 0 – normalised values passed to the model")
    item = ds[0]
    for k, v in item.items():
        arr = v.numpy()
        print(
            f"  {k:25s}  shape={str(tuple(arr.shape)):20s}  "
            f"min={arr.min():+9.4f}  max={arr.max():+9.4f}  mean={arr.mean():+9.4f}"
        )

    subsection("Denormalisation round-trip check")
    sn = item["scalar_targets"].numpy()
    sr = item["scalar_raw"].numpy()
    sn_de = sn * np.array(ns["scalar_scale"])
    print(f"  scalar_raw (physical):        drag={sr[0]:.5f} N   thrust={sr[1]:.5f} N")
    print(f"  scalar_norm * scale:          drag={sn_de[0]:.5f} N   thrust={sn_de[1]:.5f} N")
    match = np.allclose(sn_de, sr, atol=1e-3)
    print(f"  Round-trip exact:             {match}  ({'PASS' if match else 'FAIL ← denorm bug'})")


# ── Phase 3+4: Training isolation tests ──────────────────────────────────────

def _train_isolation(
    data_root: str,
    epochs: int,
    label: str,
    lambda_scalars: float = 1.0,
    lambda_fields: float = 0.0,
    print_every: int = 20,
) -> float:
    """
    Train a fresh model in overfit mode and return final normalized scalar MAE.
    """
    from config import Config
    from dataset import make_loaders
    from model import DGCNN
    from losses import HybridPhysicsLoss

    c = Config(
        data_root=Path(data_root),
        overfit_mode=True,
        norm_type="group",
        norm_groups=8,
        batch_size=4,
        learning_rate=1e-3,
        epochs=epochs,
        lambda_scalars=lambda_scalars,
        lambda_fields=lambda_fields,
        lambda_data=1.0,
        lambda_thrust=0.0,
        lambda_mass=0.0,
        lambda_bc=0.0,
        lambda_momentum=0.0,
        dropout=0.0,   # disable dropout for isolation test (easier to converge)
        grad_clip=1.0,
        seed=42,
    )

    device = c.resolve_device()
    use_amp = c.use_amp and device.type == "cuda"
    train_loader, val_loader, ds = make_loaders(c, seed=42)
    norm_stats = ds.norm_stats()
    scalar_scale = torch.tensor(norm_stats["scalar_scale"], dtype=torch.float32, device=device)
    loss_manager = HybridPhysicsLoss(c, norm_stats).to(device)

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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [{label}] model params={n_params:,}  device={device}  AMP={use_amp}")

    optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
    scaler = GradScaler(enabled=use_amp)

    print(
        f"  {'Epoch':>6}  {'train_scalar':>13}  {'thrust_MAE_N':>13}  "
        f"{'drag_MAE_N':>11}  {'pred_thrust':>12}  {'gt_thrust':>10}"
    )

    final_thrust_mae_n = float("inf")
    final_drag_mae_n = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        ep_scalar = ep_field = ep_loss = 0.0
        nb = 0
        for batch in train_loader:
            geo = batch["geometry_points"].to(device, non_blocking=True)
            mid_xyz = batch["midplane_xyz"].to(device, non_blocking=True)
            mid_flds = batch["midplane_fields"].to(device, non_blocking=True)
            scalar_t = batch["scalar_targets"].to(device, non_blocking=True)
            cond = batch["conditions"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                scalar_p, field_p = model(geo, mid_xyz, cond)
                loss, terms = loss_manager(
                    scalar_p=scalar_p, scalar_t=scalar_t,
                    field_p=field_p, field_t=mid_flds,
                    mid_xyz=mid_xyz, batch=batch, epoch=epoch,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), c.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ep_scalar += terms["loss_scalar"].item()
            ep_field += terms["loss_field"].item()
            ep_loss += terms["loss_total"].item()
            nb += 1

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            # Eval pass – collect one batch for printing predictions
            model.eval()
            thrust_mae_n = drag_mae_n = 0.0
            n_eval = 0
            sample_pred_thrust = sample_gt_thrust = float("nan")
            with torch.no_grad():
                for batch in val_loader:
                    geo = batch["geometry_points"].to(device)
                    mid_xyz = batch["midplane_xyz"].to(device)
                    scalar_t = batch["scalar_targets"].to(device)
                    cond = batch["conditions"].to(device)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        scalar_p, _ = model(geo, mid_xyz, cond)
                    t_mae = (scalar_p[:, 1] - scalar_t[:, 1]).abs().mean()
                    d_mae = (scalar_p[:, 0] - scalar_t[:, 0]).abs().mean()
                    thrust_mae_n += (t_mae * scalar_scale[1]).item()
                    drag_mae_n += (d_mae * scalar_scale[0]).item()
                    n_eval += 1
                    if n_eval == 1:
                        sample_pred_thrust = (scalar_p[0, 1] * scalar_scale[1]).item()
                        sample_gt_thrust = (scalar_t[0, 1] * scalar_scale[1]).item()

            thrust_mae_n /= max(1, n_eval)
            drag_mae_n /= max(1, n_eval)
            final_thrust_mae_n = thrust_mae_n
            final_drag_mae_n = drag_mae_n

            print(
                f"  {epoch:>6d}  {ep_scalar/max(1,nb):>13.6f}  {thrust_mae_n:>13.5f}  "
                f"{drag_mae_n:>11.5f}  {sample_pred_thrust:>12.5f}  {sample_gt_thrust:>10.5f}"
            )

    print(f"\n  [{label}] FINAL -> thrust_MAE={final_thrust_mae_n:.5f} N  drag_MAE={final_drag_mae_n:.5f} N")
    target_n = float(norm_stats["scalar_scale"][1])
    print(f"  [{label}] target thrust={target_n:.5f} N  "
          f"normalized_MAE={final_thrust_mae_n/target_n:.5f}")

    return final_thrust_mae_n


def phase3_scalar_only(data_root: str, epochs: int) -> float:
    section("PHASE 3 — SCALAR-ONLY ISOLATION TEST")
    print("  Config: GroupNorm, overfit_mode=True, dropout=0, lambda_fields=0, lambda_physics=0")
    print("  MUST converge to near-zero thrust/drag error on duplicate data.")
    return _train_isolation(
        data_root, epochs,
        label="scalar-only",
        lambda_scalars=1.0,
        lambda_fields=0.0,
        print_every=20,
    )


def phase4_field_plus_scalar(data_root: str, epochs: int) -> float:
    section("PHASE 4 — FIELD + SCALAR")
    print("  Config: GroupNorm, overfit_mode=True, dropout=0, lambda_fields=0.1")
    print("  MUST converge with BOTH heads.")
    return _train_isolation(
        data_root, epochs,
        label="scalar+field",
        lambda_scalars=1.0,
        lambda_fields=0.1,
        print_every=20,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="PI-DGCNN diagnostic suite")
    p.add_argument("--data_root", type=str, default="dataset_pointcloud")
    p.add_argument(
        "--phase",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="0=all phases, 1=data only, 2=norm only, 3=scalar-only train, 4=field+scalar train",
    )
    p.add_argument("--epochs", type=int, default=100,
                   help="Training epochs for phases 3 and 4")
    args = p.parse_args()

    run_all = args.phase == 0

    if run_all or args.phase == 1:
        phase1_data_audit(args.data_root)

    if run_all or args.phase == 2:
        phase2_norm_audit(args.data_root)

    if run_all or args.phase == 3:
        mae3 = phase3_scalar_only(args.data_root, args.epochs)

    if run_all or args.phase == 4:
        mae4 = phase4_field_plus_scalar(args.data_root, args.epochs)

    section("DIAGNOSTIC SUMMARY")
    if run_all or args.phase == 1:
        print("  Phase 1 (data):  See output above – all 20 samples should be identical.")
    if run_all or args.phase == 2:
        print("  Phase 2 (norm):  Round-trip should be PASS.")
    if run_all or args.phase == 3:
        status = "PASS" if mae3 < 0.01 else "FAIL (bug in model or data pipeline)"
        print(f"  Phase 3 (scalar-only):  thrust_MAE={mae3:.5f} N  -> {status}")
    if run_all or args.phase == 4:
        status = "PASS" if mae4 < 0.05 else "WARN (field loss may be interfering)"
        print(f"  Phase 4 (field+scalar): thrust_MAE={mae4:.5f} N  -> {status}")
    print()


if __name__ == "__main__":
    main()
