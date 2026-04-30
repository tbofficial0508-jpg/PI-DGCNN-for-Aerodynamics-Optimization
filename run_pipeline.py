"""
run_pipeline.py – Master script that orchestrates the full PI-DGCNN pipeline.

Stages
──────
  --diagnose  Run data + normalisation + training isolation tests (diagnose.py)
  --train     Load data → train model → save checkpoint + curves
  --evaluate  Load best checkpoint → run inference on all samples → print errors
  --visualise Generate all publication figures (visualise.py)
  --optimise  Load best checkpoint → DE + grid search → save results
  --all       Run diagnose → train → evaluate → visualise → optimise in sequence

Stage flags can be combined:  --train --evaluate --visualise

Output directories
──────────────────
  When --output_dir is not given, a timestamped directory is created automatically:
    runs/<timestamp>_<tag>/
  This avoids overwriting previous runs.

Usage
─────
  python run_pipeline.py --all
  python run_pipeline.py --train --epochs 200 --overfit
  python run_pipeline.py --diagnose
  python run_pipeline.py --evaluate --checkpoint runs/pipeline/checkpoints/best.pt
  python run_pipeline.py --visualise --output_dir runs/my_run
  python run_pipeline.py --optimise --no_grid_search
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

from config import Config, cfg


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_diagnose(c: Config) -> None:
    logging.info("=" * 60)
    logging.info("STAGE: DIAGNOSE")
    logging.info("=" * 60)
    from diagnose import phase1_data_audit, phase2_norm_audit, phase3_scalar_only, phase4_field_plus_scalar
    phase1_data_audit(str(c.data_root))
    phase2_norm_audit(str(c.data_root))
    mae3 = phase3_scalar_only(str(c.data_root), epochs=100)
    mae4 = phase4_field_plus_scalar(str(c.data_root), epochs=100)
    status3 = "PASS" if mae3 < 0.01 else "FAIL"
    status4 = "PASS" if mae4 < 0.05 else "WARN"
    logging.info(f"Diagnose phase 3 (scalar-only): thrust_MAE={mae3:.5f} N  -> {status3}")
    logging.info(f"Diagnose phase 4 (field+scalar): thrust_MAE={mae4:.5f} N  -> {status4}")
    logging.info("Diagnose stage complete.")


def run_train(c: Config) -> None:
    logging.info("=" * 60)
    logging.info("STAGE: TRAIN")
    logging.info("=" * 60)
    from train import train
    train(c)
    logging.info("Train stage complete.")


def run_evaluate(c: Config, checkpoint: str | None = None) -> None:
    logging.info("=" * 60)
    logging.info("STAGE: EVALUATE")
    logging.info("=" * 60)
    import torch
    from dataset import CFDDataset
    from train import load_checkpoint
    from inference import predict_sample, print_scalar_errors, plot_fields, cleanup_sample_plots

    ckpt_path = Path(checkpoint) if checkpoint else c.best_checkpoint
    if not ckpt_path.exists():
        logging.error(f"Checkpoint not found: {ckpt_path}")
        return

    device = c.resolve_device()
    model, ckpt = load_checkpoint(ckpt_path, device)
    norm_stats = ckpt["norm_stats"]

    ds = CFDDataset(data_root=c.data_root, config=c)

    total_thrust_err = 0.0
    total_drag_err   = 0.0

    indices = list(range(len(ds)))
    if c.eval_plot_mode == "first_last" and len(indices) > 1:
        plot_indices = {indices[0], indices[-1]}
    else:
        plot_indices = set(indices)

    if c.eval_plot_mode == "first_last":
        cleanup_sample_plots(c.output_dir, "eval_sample", plot_indices)

    for idx in indices:
        item   = ds[idx]
        result = predict_sample(model, item, device, norm_stats)
        print_scalar_errors(result, idx)
        if idx in plot_indices:
            plot_fields(result, c.output_dir / f"eval_sample{idx:03d}.png", idx,
                        data_root=str(c.data_root))
        total_thrust_err += abs(result["scalar_pred"][1] - result["scalar_true"][1])
        total_drag_err   += abs(result["scalar_pred"][0] - result["scalar_true"][0])

    n = len(ds)
    logging.info(f"Mean thrust MAE over {n} samples: {total_thrust_err/n:.5f} N")
    logging.info(f"Mean drag   MAE over {n} samples: {total_drag_err/n:.5f} N")
    logging.info("Evaluate stage complete.")


def _manifest_rows(data_root: Path) -> list[dict]:
    manifest = data_root / "manifest.json"
    if not manifest.exists():
        return []
    try:
        rows = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning(f"Could not parse manifest.json ({manifest}): {e}")
        return []
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _manifest_case_name(row: dict) -> str:
    case_dir = str(row.get("case_dir", "")).replace("\\", "/")
    return Path(case_dir).name.lower()


def _publication_indices(
    data_root: Path,
    n_total: int,
    publication_geometry: str | None = None,
    publication_case: str | None = None,
) -> list[int]:
    rows = _manifest_rows(data_root)
    if not rows:
        return list(range(n_total))

    gflt = publication_geometry.strip().upper() if publication_geometry else ""
    cflt = publication_case.strip().lower() if publication_case else ""
    n = min(n_total, len(rows))
    sel: list[int] = []
    for i in range(n):
        row = rows[i]
        if gflt and str(row.get("geometry", "")).strip().upper() != gflt:
            continue
        if cflt and _manifest_case_name(row) != cflt:
            continue
        sel.append(i)
    return sel


def run_visualise(
    c: Config,
    checkpoint: str | None = None,
    sample_idx: int = 0,
    publication_geometry: str | None = None,
    publication_case: str | None = None,
) -> None:
    logging.info("=" * 60)
    logging.info("STAGE: VISUALISE")
    logging.info("=" * 60)
    from visualise import (
        fig_training_convergence,
        fig_prediction_scatter,
        fig_field_comparison,
        fig_error_histograms,
        fig_optimisation_landscape,
        fig_optimisation_convergence,
        _collect_all_results,
    )
    import torch
    from dataset import CFDDataset
    from train import load_checkpoint

    ckpt_path = Path(checkpoint) if checkpoint else c.best_checkpoint
    if not ckpt_path.exists():
        logging.warning(f"Checkpoint not found: {ckpt_path}  – skipping visualise stage.")
        return

    device = c.resolve_device()
    model, ckpt = load_checkpoint(ckpt_path, device)
    norm_stats = ckpt["norm_stats"]
    ds = CFDDataset(data_root=c.data_root, config=c)

    out_dir = c.output_dir
    hjson  = out_dir / "history.json"
    ojson  = out_dir / "optimisation_outcome.json"
    lnpz   = out_dir / "thrust_landscape.npz"

    # (A) Training convergence
    if hjson.exists():
        fig_training_convergence(hjson, out_dir / "pub_training_convergence.png")
    else:
        logging.warning(f"history.json not found – skipping convergence figure.")

    # (B–D) Need model predictions
    logging.info("Collecting model predictions for all samples …")
    results_all = _collect_all_results(model, ds, device, norm_stats)

    pub_indices = _publication_indices(
        c.data_root,
        len(results_all),
        publication_geometry=publication_geometry,
        publication_case=publication_case,
    )
    if not pub_indices:
        logging.warning(
            "Publication filters yielded no samples "
            f"(geometry={publication_geometry!r}, case={publication_case!r}); "
            "falling back to all samples."
        )
        pub_indices = list(range(len(results_all)))
    results_pub = [results_all[i] for i in pub_indices]

    subset_label = None
    if publication_geometry:
        subset_label = f"{publication_geometry.strip().upper()} only"
    if publication_case:
        case_lbl = publication_case.strip()
        subset_label = f"{subset_label} / {case_lbl}" if subset_label else case_lbl

    logging.info(
        f"Publication subset: {len(results_pub)}/{len(results_all)} samples "
        f"(geometry={publication_geometry!r}, case={publication_case!r})"
    )

    if sample_idx < 0 or sample_idx >= len(results_all):
        sidx = pub_indices[0]
    elif sample_idx not in pub_indices:
        logging.info(
            f"Requested visualise_sample={sample_idx} is outside publication subset; "
            f"using {pub_indices[0]} instead."
        )
        sidx = pub_indices[0]
    else:
        sidx = sample_idx

    fig_prediction_scatter(results_pub, out_dir / "pub_prediction_scatter.png", subset_label=subset_label)
    fig_field_comparison(results_all[sidx], out_dir / f"pub_field_comparison_s{sidx:03d}.png", sidx,
                         data_root=str(c.data_root))
    fig_error_histograms(results_pub, out_dir / "pub_error_histograms.png", subset_label=subset_label)

    # (E–F) Optimisation figures (only if outcome exists)
    if ojson.exists():
        fig_optimisation_landscape(ojson, lnpz if lnpz.exists() else None,
                                   out_dir / "pub_optimisation_landscape.png")
        fig_optimisation_convergence(ojson, out_dir / "pub_optimisation_convergence.png")
    else:
        logging.info("optimisation_outcome.json not found – optimisation figures skipped.")

    logging.info("Visualise stage complete.")


def run_optimise(
    c: Config,
    checkpoint: str | None = None,
    save_only_outcome: bool | None = None,
    use_grid_search: bool | None = None,
    objective: str = "thrust",
    efficiency_alpha: float = 0.5,
    base_sample_idx: int = 0,
) -> None:
    logging.info("=" * 60)
    logging.info("STAGE: OPTIMISE")
    logging.info("=" * 60)
    from optimise import optimise
    ckpt = Path(checkpoint) if checkpoint else None
    results = optimise(
        c,
        checkpoint_path=ckpt,
        save_only_outcome=save_only_outcome,
        use_grid_search=use_grid_search,
        objective=objective,
        efficiency_alpha=efficiency_alpha,
        base_sample_idx=base_sample_idx,
    )
    logging.info("Optimisation complete.")
    logging.info(f"Best thrust:       {results['de_result']['opt_thrust_N']:.5f} N")
    if "baseline_prediction" in results:
        logging.info(f"Pred baseline:     {results['baseline_prediction']['thrust_N']:.5f} N")
    logging.info(f"Baseline thrust:   {results['baseline']['thrust_N']:.5f} N")
    logging.info(f"Predicted gain:    {results['improvement_pct']:.2f}%")


def print_final_summary(c: Config) -> None:
    """Print model parameter count and best results from the run directory."""
    import json, torch
    from train import load_checkpoint

    logging.info("=" * 60)
    logging.info("FINAL SUMMARY")
    logging.info("=" * 60)

    n_params = "?"
    ckpt_path = c.best_checkpoint
    if ckpt_path.exists():
        device = c.resolve_device()
        try:
            model, ckpt = load_checkpoint(ckpt_path, device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f"Model parameters: {n_params:,}")
            m = ckpt.get("metrics", {})
            if "thrust_mae_phys" in m:
                logging.info(f"Best thrust MAE:  {m['thrust_mae_phys']:.5f} N")
                logging.info(f"Best drag MAE:    {m['drag_mae_phys']:.5f} N")
        except Exception as e:
            logging.warning(f"Could not load checkpoint for summary: {e}")

    ojson = c.output_dir / "optimisation_outcome.json"
    if ojson.exists():
        with open(ojson, encoding="utf-8") as f:
            out = json.load(f)
        logging.info(f"Baseline thrust:  {out['baseline']['thrust_N']:.5f} N")
        logging.info(f"Optimised thrust: {out['de_result']['opt_thrust_N']:.5f} N")
        logging.info(f"Gain:             {out['improvement_pct']:+.2f}%")

    # List publication figures
    pub_figs = sorted(c.output_dir.glob("pub_*.png")) + sorted(c.output_dir.glob("pub_*.pdf"))
    if pub_figs:
        logging.info("Publication figures:")
        for f in pub_figs:
            logging.info(f"  {f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> tuple[argparse.Namespace, Config]:
    p = argparse.ArgumentParser(
        description="PI-DGCNN CFD surrogate – full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage flags
    p.add_argument("--diagnose",  action="store_true", help="Run diagnostic tests")
    p.add_argument("--train",     action="store_true", help="Run training stage")
    p.add_argument("--evaluate",  action="store_true", help="Run evaluation stage")
    p.add_argument("--visualise", action="store_true", help="Generate publication figures")
    p.add_argument("--optimise",  action="store_true", help="Run optimisation stage")
    p.add_argument("--all",       action="store_true", help="Run all stages in sequence")

    # Paths
    p.add_argument("--data_root",   type=str, default=str(cfg.data_root))
    p.add_argument("--output_dir",  type=str, default=None,
                   help="Output directory (auto-timestamped if not given with --all)")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Path to checkpoint (evaluate/visualise/optimise stages)")

    # Training hyperparameters
    p.add_argument("--epochs",      type=int,   default=cfg.epochs)
    p.add_argument("--batch_size",  type=int,   default=cfg.batch_size)
    p.add_argument("--lr",          type=float, default=cfg.learning_rate)
    p.add_argument("--k",           type=int,   default=cfg.k)
    p.add_argument("--dropout",     type=float, default=cfg.dropout)
    p.add_argument("--norm_type",   type=str,   default=cfg.norm_type,
                   choices=["batch", "group", "layer"])
    p.add_argument("--norm_groups", type=int,   default=cfg.norm_groups)
    p.add_argument("--num_workers", type=int,   default=cfg.num_workers)
    p.add_argument("--overfit",     action="store_true",
                   help="Train/evaluate on the full dataset (no held-out split)")
    p.add_argument("--no_overfit",  action="store_true",
                   help="Force held-out split even for duplicate data (for ablation)")
    p.add_argument("--no_mean_pool", action="store_true",
                   help="Disable max+mean pool (use max only)")
    p.add_argument("--fourier_levels", type=int, default=getattr(cfg, "fourier_levels", 6),
                   help="Fourier positional encoding octaves for field head (0=disable)")
    p.add_argument("--scheduler",   type=str, default=getattr(cfg, "scheduler_type", "cosine_warm"),
                   choices=["cosine_warm", "plateau"])
    p.add_argument("--cosine_T0",   type=int, default=getattr(cfg, "cosine_T0", 50))
    p.add_argument("--early_stopping_patience", type=int,
                   default=getattr(cfg, "early_stopping_patience", 60))
    p.add_argument("--checkpoint_interval", type=int,
                   default=getattr(cfg, "checkpoint_interval", 25))
    p.add_argument("--field_finetune_epochs", type=int,
                   default=getattr(cfg, "field_finetune_epochs", 500),
                   help="Phase-2 field fine-tune epochs (0=disabled)")
    p.add_argument("--field_finetune_lr", type=float,
                   default=getattr(cfg, "field_finetune_lr", 5e-3),
                   help="Phase-2 field fine-tune learning rate")

    # Loss lambdas
    p.add_argument("--checkpoint_metric",
                   type=str, default=cfg.checkpoint_metric,
                   choices=["combined_mae", "thrust_mae", "drag_mae", "loss"])
    p.add_argument("--checkpoint_thrust_weight", type=float, default=cfg.checkpoint_thrust_weight)
    p.add_argument("--checkpoint_drag_weight",   type=float, default=cfg.checkpoint_drag_weight)
    p.add_argument("--lambda_data",     type=float, default=cfg.lambda_data)
    p.add_argument("--lambda_scalars",  type=float, default=cfg.lambda_scalars)
    p.add_argument("--lambda_fields",   type=float, default=cfg.lambda_fields)
    p.add_argument("--lambda_field_grad", type=float, default=cfg.lambda_field_grad)
    p.add_argument("--lambda_thrust",   type=float, default=cfg.lambda_thrust)
    p.add_argument("--lambda_mass",     type=float, default=cfg.lambda_mass)
    p.add_argument("--lambda_bc",       type=float, default=cfg.lambda_bc)
    p.add_argument("--field_data_loss_type",  type=str, default=cfg.field_data_loss_type,
                   choices=["mse", "huber", "l1"])
    p.add_argument("--field_huber_delta",     type=float, default=cfg.field_huber_delta)
    p.add_argument("--field_spatial_weight_alpha", type=float, default=cfg.field_spatial_weight_alpha)
    p.add_argument("--field_spatial_weight_clip",  type=float, default=cfg.field_spatial_weight_clip)
    p.add_argument("--field_spatial_knn_k",   type=int,   default=cfg.field_spatial_knn_k)
    p.add_argument("--field_grad_loss_type",  type=str, default=cfg.field_grad_loss_type,
                   choices=["mse", "huber", "l1"])
    p.add_argument("--field_grad_huber_delta", type=float, default=cfg.field_grad_huber_delta)
    p.add_argument("--field_grad_knn_k",  type=int, default=cfg.field_grad_knn_k)
    p.add_argument("--thrust_loss_type",  type=str, default=cfg.thrust_loss_type,
                   choices=["huber", "mse", "l1"])
    p.add_argument("--thrust_huber_delta", type=float, default=cfg.thrust_huber_delta)
    p.add_argument("--slice_mode", type=str, default=cfg.physics_slice_mode,
                   choices=["full3d", "quasi2d", "midplane3d"])
    p.add_argument("--physics_schedule", type=str, default=cfg.physics_schedule,
                   choices=["none", "linear", "cosine"])
    p.add_argument("--physics_warmup_epochs", type=int, default=cfg.physics_warmup_epochs)
    p.add_argument("--midplane3d_weak_factor", type=float, default=cfg.physics_midplane3d_weak_factor)
    p.add_argument("--physics_knn_k", type=int, default=cfg.physics_knn_k)

    # Autograd physics losses (Tier-1 PhysicsNeMo integration)
    p.add_argument("--use_autograd_physics", action="store_true",
                   help="Enable Tier-1 autograd continuity and wall-BC losses")
    p.add_argument("--lambda_continuity", type=float,
                   default=getattr(cfg, "lambda_continuity", 0.0),
                   help="Weight for du/dx+dw/dz=0 continuity residual")
    p.add_argument("--lambda_wall_bc", type=float,
                   default=getattr(cfg, "lambda_wall_bc", 0.0),
                   help="Weight for no-slip wall BC loss (u=v=w=0 at surface)")
    p.add_argument("--n_wall_pts", type=int,
                   default=getattr(cfg, "n_wall_pts", 64),
                   help="Wall surface points sampled per batch item")

    # Tier-2 PhysicsNeMo losses (physics_nemo.py) — full NS equations
    p.add_argument("--use_physicsnemo", action="store_true",
                   help="Enable Tier-2 PhysicsNeMo continuity+momentum+wall losses")
    p.add_argument("--lambda_momentum", type=float,
                   default=getattr(cfg, "lambda_momentum", 0.0),
                   help="Weight for Euler xz-momentum residual (Tier-2)")
    p.add_argument("--fluid_nu", type=float,
                   default=getattr(cfg, "fluid_nu", 1.5e-5),
                   help="Kinematic viscosity [m^2/s] (air STP default)")
    p.add_argument("--fluid_rho", type=float,
                   default=getattr(cfg, "fluid_rho", 1.225),
                   help="Air density [kg/m^3] (air STP default)")

    # Optimisation
    p.add_argument("--grid_res",        type=int, default=cfg.grid_resolution)
    p.add_argument("--grid_batch_size", type=int, default=cfg.grid_batch_size,
                   help="Model batch size for grid sweep (reduce if GPU OOM)")
    p.add_argument("--optimise_uncertainty_samples", type=int,
                   default=cfg.optimise_uncertainty_samples)
    p.add_argument("--optimise_uncertainty_beta", type=float,
                   default=cfg.optimise_uncertainty_beta)
    p.add_argument("--save_only_outcome", action="store_true",
                   help="Optimise stage saves only outcome files (no plots)")
    p.add_argument("--no_grid_search", action="store_true",
                   help="Disable optimise-stage grid search")
    p.add_argument("--objective", type=str, default="thrust",
                   choices=["thrust", "efficiency", "combined"],
                   help="Optimisation objective: thrust | efficiency (T/(D+T)) | combined (T - alpha*D)")
    p.add_argument("--efficiency_alpha", type=float, default=0.5,
                   help="Weight on drag in combined objective: score = thrust - alpha * drag")
    p.add_argument("--optimise_base_sample", type=int, default=0,
                   help="Dataset index to use as the initial/baseline geometry for optimisation (0-based)")
    p.add_argument("--visualise_sample", type=int, default=0,
                   help="Dataset index to use as the focus sample for field comparison figures (0-based)")
    p.add_argument("--publication_geometry", type=str, default=None,
                   help="Optional geometry tag filter for publication scatter/hist figures (e.g. J2)")
    p.add_argument("--publication_case", type=str, default=None,
                   help="Optional case folder filter for publication scatter/hist figures (e.g. case_0020)")

    # Phase 3 physics fine-tune
    p.add_argument("--physics_finetune_epochs", type=int,
                   default=getattr(cfg, "physics_finetune_epochs", 0),
                   help="Phase-3 epochs with continuity+wall physics losses (0=off)")
    p.add_argument("--physics_finetune_lr", type=float,
                   default=getattr(cfg, "physics_finetune_lr", 1e-5))
    p.add_argument("--physics_finetune_lambda_cont", type=float,
                   default=getattr(cfg, "physics_finetune_lambda_cont", 0.01))
    p.add_argument("--physics_finetune_lambda_wall", type=float,
                   default=getattr(cfg, "physics_finetune_lambda_wall", 0.10))

    # Misc
    p.add_argument("--eval_plot_mode", type=str, default=cfg.eval_plot_mode,
                   choices=["all", "first_last"])
    p.add_argument("--device",     type=str, default=cfg.device)
    p.add_argument("--no_amp",     action="store_true")

    # Optimiser + weight averaging (see config.Config)
    p.add_argument("--optimizer_type", type=str,
                   default=getattr(cfg, "optimizer_type", "adamw"),
                   choices=["adam", "adamw"])
    p.add_argument("--weight_decay", type=float,
                   default=getattr(cfg, "weight_decay", 1e-5))
    p.add_argument("--no_ema", action="store_true",
                   help="Disable EMA weight averaging (on by default)")
    p.add_argument("--ema_decay", type=float,
                   default=getattr(cfg, "ema_decay", 0.999))
    p.add_argument("--phase2_ema_decay", type=float,
                   default=getattr(cfg, "phase2_ema_decay", 0.99),
                   help="Faster EMA decay for Phase 2 field fine-tune (default 0.99)")
    p.add_argument("--skip_phase1", action="store_true",
                   help="Skip Phase 1 and resume Phase 2 from existing best.pt")

    args = p.parse_args()

    # Determine output directory.
    # Default layout:
    #   runs/GPMLresults/DGCNN/        – purely supervised runs
    #   runs/GPMLresults/PIDGCNN/      – physics-informed runs (auto-selected
    #                                     when any physics flag is enabled)
    # For --all runs without an explicit --output_dir, we append a timestamped
    # subfolder so repeat runs don't clobber each other.
    physics_on = (
        args.use_autograd_physics
        or args.use_physicsnemo
        or args.physics_finetune_epochs > 0
    )
    default_base = Path("runs/GPMLresults") / ("PIDGCNN" if physics_on else "DGCNN")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.all:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = default_base / f"{ts}_pipeline"
    else:
        out_dir = default_base

    # Determine overfit mode: default True (all data is duplicates)
    overfit = cfg.overfit_mode  # default from config (True)
    if args.overfit:
        overfit = True
    if args.no_overfit:
        overfit = False

    c = Config(
        data_root=Path(args.data_root),
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        k=args.k,
        dropout=args.dropout,
        norm_type=args.norm_type,
        norm_groups=args.norm_groups,
        num_workers=args.num_workers,
        overfit_mode=overfit,
        use_mean_pool=not args.no_mean_pool,
        fourier_levels=args.fourier_levels,
        scheduler_type=args.scheduler,
        cosine_T0=args.cosine_T0,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_interval=args.checkpoint_interval,
        field_finetune_epochs=args.field_finetune_epochs,
        field_finetune_lr=args.field_finetune_lr,
        checkpoint_metric=args.checkpoint_metric,
        checkpoint_thrust_weight=args.checkpoint_thrust_weight,
        checkpoint_drag_weight=args.checkpoint_drag_weight,
        lambda_data=args.lambda_data,
        lambda_scalars=args.lambda_scalars,
        lambda_fields=args.lambda_fields,
        lambda_field_grad=args.lambda_field_grad,
        lambda_thrust=args.lambda_thrust,
        lambda_mass=args.lambda_mass,
        lambda_bc=args.lambda_bc,
        field_data_loss_type=args.field_data_loss_type,
        field_huber_delta=args.field_huber_delta,
        field_spatial_weight_alpha=args.field_spatial_weight_alpha,
        field_spatial_weight_clip=args.field_spatial_weight_clip,
        field_spatial_knn_k=args.field_spatial_knn_k,
        field_grad_loss_type=args.field_grad_loss_type,
        field_grad_huber_delta=args.field_grad_huber_delta,
        field_grad_knn_k=args.field_grad_knn_k,
        thrust_loss_type=args.thrust_loss_type,
        thrust_huber_delta=args.thrust_huber_delta,
        physics_slice_mode=args.slice_mode,
        physics_schedule=args.physics_schedule,
        physics_warmup_epochs=args.physics_warmup_epochs,
        physics_midplane3d_weak_factor=args.midplane3d_weak_factor,
        physics_knn_k=args.physics_knn_k,
        grid_resolution=args.grid_res,
        grid_batch_size=args.grid_batch_size,
        optimise_uncertainty_samples=args.optimise_uncertainty_samples,
        optimise_uncertainty_beta=args.optimise_uncertainty_beta,
        eval_plot_mode=args.eval_plot_mode,
        device=args.device,
        use_amp=not args.no_amp,
        optimise_save_only_outcome=args.save_only_outcome,
        optimise_use_grid_search=not args.no_grid_search,
        use_autograd_physics=args.use_autograd_physics,
        lambda_continuity=args.lambda_continuity,
        lambda_wall_bc=args.lambda_wall_bc,
        n_wall_pts=args.n_wall_pts,
        use_physicsnemo=args.use_physicsnemo,
        lambda_momentum=args.lambda_momentum,
        fluid_nu=args.fluid_nu,
        fluid_rho=args.fluid_rho,
        physics_finetune_epochs=args.physics_finetune_epochs,
        physics_finetune_lr=args.physics_finetune_lr,
        physics_finetune_lambda_cont=args.physics_finetune_lambda_cont,
        physics_finetune_lambda_wall=args.physics_finetune_lambda_wall,
        optimizer_type=args.optimizer_type,
        weight_decay=args.weight_decay,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        phase2_ema_decay=args.phase2_ema_decay,
        skip_phase1=args.skip_phase1,
    )

    return args, c


def main() -> None:
    args, c = parse_args()

    # Default: run all if no specific stage flag given
    run_all = args.all or not any([
        args.diagnose, args.train, args.evaluate, args.visualise, args.optimise
    ])

    c.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(c.output_dir / "pipeline.log")

    logging.info("PI-DGCNN surrogate pipeline starting")
    logging.info(f"  data_root:  {c.data_root}")
    logging.info(f"  output_dir: {c.output_dir}")
    logging.info(f"  device:     {c.resolve_device()}")
    logging.info(f"  epochs:     {c.epochs}  batch_size: {c.batch_size}  k: {c.k}")
    logging.info(f"  norm:       {c.norm_type} (groups={c.norm_groups})")
    logging.info(f"  overfit:    {c.overfit_mode}  mean_pool: {c.use_mean_pool}")
    logging.info(f"  scheduler:  {getattr(c, 'scheduler_type', 'cosine_warm')}")

    if args.diagnose or run_all:
        run_diagnose(c)

    if args.train or run_all:
        run_train(c)

    if args.evaluate or run_all:
        run_evaluate(c, checkpoint=args.checkpoint)

    if args.visualise or run_all:
        run_visualise(
            c,
            checkpoint=args.checkpoint,
            sample_idx=args.visualise_sample,
            publication_geometry=args.publication_geometry,
            publication_case=args.publication_case,
        )

    if args.optimise or run_all:
        run_optimise(
            c,
            checkpoint=args.checkpoint,
            save_only_outcome=args.save_only_outcome,
            use_grid_search=not args.no_grid_search,
            objective=args.objective,
            efficiency_alpha=args.efficiency_alpha,
            base_sample_idx=args.optimise_base_sample,
        )

    if run_all:
        print_final_summary(c)

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
