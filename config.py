"""
config.py – Central configuration for the DGCNN surrogate pipeline.

All hyperparameters, paths, and hardware settings live here.
Import the default instance with:   from config import cfg
Override fields before passing to train/optimise as needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    # ── Architecture ──────────────────────────────────────────────────────────
    k: int = 20                      # k-NN neighbours rebuilt per EdgeConv layer
    num_points: int = 4096           # geometry points sampled from surface mesh
    num_midplane_points: int = 8192  # midplane query points kept per sample
    in_channels: int = 8             # [x, y, z, nx, ny, nz, is_body, is_edf]
    cond_dim: int = 2                # [u_inf, rpm]
    scalar_dim: int = 2              # [drag, thrust]
    field_dim: int = 4               # [pressure, u, v, w] at midplane points
    # Output channels per EdgeConv block (3 blocks as specified)
    edge_channels: tuple = (64, 64, 128)
    dropout: float = 0.3
    norm_type: str = "group"         # "batch" | "group" | "layer"
    # NOTE: "batch" is unstable with batch_size=4 (BatchNorm1d sees only 4 samples
    # in scalar/cond heads → noisy running stats → train/eval mismatch of 10-100×).
    # Always use "group" or "layer" with small batches.
    norm_groups: int = 8             # used when norm_type="group"
    use_mean_pool: bool = True       # concat max+mean global pool (vs max only)
    fourier_levels: int = 10         # Fourier positional encoding octaves for field head
    # NOTE: fourier_levels=10 adds pos_dim=63 (3 + 3*2*10 sinusoidal features).
    # L=10 needed for SIREN field head to achieve RMSE_norm<0.0001; L=6 gives ~0.06.
    # Set to 0 to disable (falls back to raw xyz, field RMSE will be ~100x worse).

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size: int = 4
    epochs: int = 400
    learning_rate: float = 1e-3
    train_fraction: float = 0.8      # fraction of dataset used for training
    overfit_mode: bool = True        # train/eval on the full dataset
    # NOTE: overfit_mode=True is correct when all samples are duplicates of the same
    # simulation. Using a held-out split still works, but the BatchNorm mismatch
    # between subsampled-geometry variants then inflates val error misleadingly.
    # Multi-task loss weights – scalars (thrust/drag) matter most for optimisation
    lambda_scalars: float = 1.0
    lambda_fields: float = 1.0       # equal weight: field head needs as much signal as scalar
    # NOTE: lambda_fields=0.1 caused field head to stall at mean prediction (RMSE >100 m/s).
    # The field head loss >> scalar loss at convergence, so 0.1 weight still dominates
    # total loss — but the *encoder* gradient is weak. Setting to 1.0 ensures the encoder
    # adapts to produce spatially-informative features for the field head.
    lambda_data: float = 1.0         # global multiplier on supervised terms
    lambda_mass: float = 0.0         # continuity / mass-conservation term
    lambda_bc: float = 0.0           # boundary-condition term
    lambda_thrust: float = 0.0       # extra thrust-focused scalar term
    lambda_momentum: float = 0.0     # momentum residual scaffold (disabled by default)
    lambda_field_grad: float = 0.0   # supervised local-structure term on field gradients
    grad_clip: float = 1.0
    seed: int = 42
    checkpoint_metric: str = "combined_mae"  # "combined_mae" | "thrust_mae" | "drag_mae" | "loss"
    checkpoint_thrust_weight: float = 1.0
    checkpoint_drag_weight: float = 1.0

    # Thrust term details (only used when lambda_thrust > 0)
    thrust_loss_type: str = "huber"  # "huber" | "mse" | "l1"
    thrust_huber_delta: float = 0.05
    thrust_index: int = 1            # scalar target index for thrust

    # Field-data weighting details
    field_channel_weights: tuple = (1.0, 1.0, 1.0, 1.0)  # [p, u, v, w]
    field_data_loss_type: str = "mse"  # "mse" | "huber" | "l1"
    field_huber_delta: float = 0.05
    field_spatial_weight_alpha: float = 0.0   # 0 disables spatial reweighting
    field_spatial_weight_clip: float = 4.0    # max multiplicative boost above 1.0
    field_spatial_knn_k: int = 12
    field_spatial_use_denormalized: bool = True

    # Gradient-consistency term (supervised on local pairwise derivatives)
    field_grad_loss_type: str = "huber"   # "mse" | "huber" | "l1"
    field_grad_huber_delta: float = 0.05
    field_grad_channel_weights: tuple = (1.0, 1.0, 1.0, 1.0)  # [p, u, v, w]
    field_grad_knn_k: int = 12
    field_grad_use_denormalized: bool = False
    field_grad_relative: bool = True
    field_grad_relative_eps: float = 1e-4

    # Physics-loss ramp schedule
    physics_schedule: str = "linear"  # "none" | "linear" | "cosine"
    physics_warmup_epochs: int = 20

    # Continuity settings for slice/full-volume data handling
    physics_slice_mode: str = "midplane3d"  # "full3d" | "quasi2d" | "midplane3d"
    physics_midplane3d_weak_factor: float = 0.0
    physics_knn_k: int = 12
    physics_axis_span_tol: float = 1e-5
    physics_lstsq_eps: float = 1e-6
    physics_use_denormalized: bool = True

    # Optional BC settings (used only when boundary masks/tags are present)
    bc_wall_mode: str = "no_slip"     # "no_slip" | "no_penetration"
    eval_plot_mode: str = "first_last"  # "all" | "first_last"

    # ── Autograd physics losses (Tier-1 PhysicsNeMo integration) ─────────────
    # Uses torch.autograd.grad through the FiLM-SIREN field head for exact
    # PDE residuals — no finite-difference approximation.
    # All lambdas default to 0 (disabled). Enable after Phase-1 convergence.
    #
    # Recommended starting values once field loss is below ~1e-3:
    #   lambda_continuity = 0.01  (scale relative to field MSE)
    #   lambda_wall_bc    = 0.10  (hard constraint; stronger weight)
    use_autograd_physics: bool = False  # master switch; set True to activate
    lambda_continuity: float = 0.0     # du/dx + dw/dz = 0 on midplane
    lambda_wall_bc: float = 0.0        # u=v=w=0 at body surface points
    n_wall_pts: int = 64               # wall points sampled per batch item

    # ── Tier-2 PhysicsNeMo losses (physics_nemo.py) ───────────────────────────
    # Full 3D continuity + Euler momentum (x/z) via NavierStokes equations.
    # Requires physicsnemo.sym installed (WSL / Linux recommended).
    # Enable after Tier-1 continuity has converged (field RMSE < 1e-2).
    #
    # Recommended starting values:
    #   lambda_continuity = 0.01   (same as Tier-1)
    #   lambda_momentum   = 1e-4   (pressure gradients are large; start small)
    #   lambda_wall_bc    = 0.10
    use_physicsnemo: bool = False       # master switch for Tier-2 losses
    lambda_momentum: float = 0.0       # Euler x/z momentum residual weight
    fluid_nu: float = 1.5e-5           # kinematic viscosity [m^2/s] (air STP)
    fluid_rho: float = 1.225           # density [kg/m^3] (air STP)

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    scheduler_type: str = "cosine_warm"  # "plateau" | "cosine_warm"
    # CosineAnnealingWarmRestarts (used when scheduler_type="cosine_warm")
    cosine_T0: int = 100             # restart period in epochs (was 50 – too short)
    cosine_eta_min: float = 1e-6    # minimum LR
    # ReduceLROnPlateau (used when scheduler_type="plateau")
    lr_patience: int = 20
    lr_factor: float = 0.5
    lr_min: float = 1e-6
    # Early stopping – watch field loss, not just scalar MAE
    early_stopping_patience: int = 60   # more patience: field head needs longer to escape mean attractor

    # Phase-2 field fine-tune: freeze encoder, fresh Adam optimizer, high LR
    # This breaks the stale-Adam-momentum problem that traps the field head at mean prediction.
    field_finetune_epochs: int = 2000   # 0 = disabled; SIREN needs ~600 to converge, 2000 for safety
    field_finetune_lr: float = 5e-5    # field head only — SIREN optimal LR from standalone test
    # Periodic checkpoint interval (in addition to best-model checkpoint)
    checkpoint_interval: int = 25

    # ── Data Augmentation ─────────────────────────────────────────────────────
    # Augmentations applied at __getitem__ time - free diversity from one sim.
    augment_jitter_std: float = 0.005   # Gaussian xyz noise on geo points (0=off)
    # 0.005 = 0.5% of the normalised [-1,1] range. Large enough to decorrelate
    # the 20 near-identical FPS samples; small enough not to distort surface
    # curvature that the encoder uses to distinguish body from EDF.
    augment_resample_mid: bool = True   # sample fresh midplane query points each epoch
    # The CFD midplane has ~88K points; training always sees the same 2048 if
    # this is False. Setting True stores the full domain and resamples randomly
    # each __getitem__ call, forcing the SIREN to generalise over the whole
    # domain rather than memorising 2048 fixed locations.
    augment_mid_max_pts: int = 16384    # cap on cached midplane points per sample

    # ── Phase 3: Physics fine-tune (post-Phase-2 field convergence) ───────────
    # After Phase 2 field RMSE is ~6e-4. Phase 3 activates Tier-1 physics
    # losses to enforce continuity and no-slip BC without degrading accuracy.
    # Enable only after field_finetune produces RMSE_norm < 1e-3.
    physics_finetune_epochs: int = 0    # 0 = disabled; 100 recommended
    physics_finetune_lr: float = 1e-5   # full-model LR (encoder unfrozen)
    physics_finetune_lambda_cont: float = 0.01   # continuity weight
    physics_finetune_lambda_wall: float = 0.10   # no-slip wall BC weight

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root: Path = Path("dataset_pointcloud")
    fluent_root: Path = Path("fluent_exports")
    # Default lives under runs/GPMLresults/DGCNN; physics runs get routed to
    # runs/GPMLresults/PIDGCNN automatically by run_pipeline.py when any
    # use_autograd_physics / use_physicsnemo flag is active.
    output_dir: Path = Path("runs/GPMLresults/DGCNN")

    # ── Hardware ─────────────────────────────────────────────────────────────
    device: str = "auto"   # "auto" | "cuda" | "cpu"
    use_amp: bool = True   # mixed-precision – halves VRAM for activations
    num_workers: int = 0
    pin_memory: bool = True

    # ── Optimizer & weight averaging ─────────────────────────────────────────
    # AdamW + decoupled weight decay regularises the scalar head (which would
    # otherwise memorise the 20-sample duplicate dataset exactly). EMA averages
    # weights across the training trajectory – consistently gains 10-30% on val
    # accuracy with no extra compute at train time (one extra clone per step).
    optimizer_type: str = "adamw"       # "adam" | "adamw"
    weight_decay: float = 1e-5          # decoupled L2 (only AdamW)
    use_ema: bool = True                # track EMA shadow weights, use for eval/checkpoint
    ema_decay: float = 0.999            # 0.999 = ~1000-step horizon
    phase2_ema_decay: float = 0.99      # faster EMA for Phase 2 to track through instability spikes
    skip_phase1: bool = False           # if True, skip Phase 1 and start from existing best.pt

    # ── Design variable bounds for geometry optimisation ──────────────────────
    # Each entry is [min, max] for one design variable:
    #   DV0 – fuselage length scale:   stretch body along x (1.0 = baseline)
    #   DV1 – duct diameter scale:     scale EDF radially    (1.0 = baseline)
    #   DV2 – EDF axial offset:        shift EDF in x (normalised units)
    dv_bounds: list = field(default_factory=lambda: [
        [0.8, 1.2],
        [0.85, 1.15],
        [-0.1, 0.1],
    ])

    # ── Differential Evolution ────────────────────────────────────────────────
    de_popsize: int = 15
    de_maxiter: int = 100
    de_seed: int = 42
    de_tol: float = 1e-4
    de_mutation: tuple = (0.5, 1.0)
    de_recombination: float = 0.7
    optimise_uncertainty_samples: int = 1    # >1 enables MC-dropout objective
    optimise_uncertainty_beta: float = 0.0   # robust score = mean - beta*std
    optimise_use_grid_search: bool = True
    optimise_save_only_outcome: bool = False

    # ── Grid search over design space ─────────────────────────────────────────
    grid_resolution: int = 20        # points per DV axis (20³ = 8 000 evaluations)
    grid_batch_size: int = 8         # model batch size during grid sweep
    # NOTE: 64 caused CUDA OOM on knn_graph (inner product: 64 * 4096^2 * 4B = 4 GB).
    # 8 is safe on 8 GB VRAM; increase only if model uses N << 4096 points.

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# Default configuration instance – import and modify as needed
cfg = Config()
