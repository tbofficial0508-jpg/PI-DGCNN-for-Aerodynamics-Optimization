from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from utils import knn_graph


class HybridPhysicsLoss(nn.Module):
    """
    Hybrid supervised + physics-informed loss manager.

    Notes on data availability in this repository:
    - The current dataset contains point coordinates and field values on a slice.
    - Mesh/control-volume connectivity is not currently exposed in training batches.
    - Therefore continuity is implemented with a local-neighborhood derivative
      approximation fallback (point-cloud path), while finite-volume and momentum
      residual terms are kept as guarded scaffolds.
    """

    def __init__(self, c: Config, norm_stats: Optional[dict] = None):
        super().__init__()
        self.c = c
        self._warned: set[str] = set()

        norm_stats = norm_stats or {}
        field_scale = norm_stats.get("field_scale", [1.0, 1.0, 1.0, 1.0])
        mid_xyz_scale = norm_stats.get("mid_xyz_scale", [1.0, 1.0, 1.0])

        self.register_buffer(
            "field_scale",
            torch.tensor(field_scale, dtype=torch.float32).view(1, -1, 1),
            persistent=False,
        )
        self.register_buffer(
            "mid_xyz_scale",
            torch.tensor(mid_xyz_scale, dtype=torch.float32).view(1, 3, 1),
            persistent=False,
        )

        self._huber = nn.HuberLoss(delta=float(c.thrust_huber_delta), reduction="mean")

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[loss][warn] {message}")

    def _physics_ramp(self, epoch: int) -> float:
        schedule = str(self.c.physics_schedule).lower()
        warmup = int(self.c.physics_warmup_epochs)
        if schedule == "none" or warmup <= 0:
            return 1.0

        t = min(max((epoch - 1) / max(1, warmup - 1), 0.0), 1.0)
        if schedule == "linear":
            return t
        if schedule == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * t))

        self._warn_once("bad_schedule", f"Unknown physics_schedule={schedule!r}. Using ramp=1.")
        return 1.0

    def _thrust_loss(self, scalar_p: torch.Tensor, scalar_t: torch.Tensor) -> torch.Tensor:
        zero = scalar_p.new_zeros(())
        i_thrust = int(self.c.thrust_index)
        if scalar_p.ndim != 2 or scalar_t.ndim != 2 or scalar_p.shape != scalar_t.shape:
            self._warn_once("bad_scalar_shape", "Scalar prediction/target shapes are incompatible for thrust loss.")
            return zero
        if i_thrust < 0 or i_thrust >= scalar_p.shape[1]:
            self._warn_once("bad_thrust_index", f"thrust_index={i_thrust} is out of range for scalar_dim={scalar_p.shape[1]}.")
            return zero

        pred = scalar_p[:, i_thrust]
        true = scalar_t[:, i_thrust]

        thrust_loss_type = str(self.c.thrust_loss_type).lower()
        if thrust_loss_type == "mse":
            return F.mse_loss(pred, true)
        if thrust_loss_type == "l1":
            return F.l1_loss(pred, true)
        if thrust_loss_type == "huber":
            return self._huber(pred, true)

        self._warn_once("bad_thrust_loss", f"Unknown thrust_loss_type={thrust_loss_type!r}. Falling back to Huber.")
        return self._huber(pred, true)

    def _get_channel_weights(
        self,
        weights_cfg,
        n_channels: int,
        ref_tensor: torch.Tensor,
        warn_key: str,
    ) -> torch.Tensor:
        if isinstance(weights_cfg, (tuple, list)):
            weights = torch.as_tensor(weights_cfg, dtype=ref_tensor.dtype, device=ref_tensor.device).flatten()
        else:
            self._warn_once(warn_key, f"Channel weights must be tuple/list; got {type(weights_cfg).__name__}. Using ones.")
            weights = ref_tensor.new_ones((n_channels,))

        if weights.numel() == n_channels:
            return weights

        self._warn_once(
            warn_key,
            f"Channel weight count ({weights.numel()}) does not match field channels ({n_channels}); using ones.",
        )
        return ref_tensor.new_ones((n_channels,))

    def _elementwise_error(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        loss_type: str,
        huber_delta: float,
        warn_key: str,
    ) -> torch.Tensor:
        mode = str(loss_type).lower()
        if mode == "mse":
            return (pred - true) ** 2
        if mode == "l1":
            return (pred - true).abs()
        if mode == "huber":
            return F.huber_loss(pred, true, delta=float(huber_delta), reduction="none")
        self._warn_once(warn_key, f"Unknown loss type {loss_type!r}; defaulting to MSE.")
        return (pred - true) ** 2

    def _spatial_weight_map(
        self,
        field_t: torch.Tensor,
        mid_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build point-wise weights that emphasize high-gradient regions.

        Returns shape (B, N). If spatial weighting is disabled, returns ones.
        """
        alpha = float(self.c.field_spatial_weight_alpha)
        bsz, _, n_points = field_t.shape
        weights = field_t.new_ones((bsz, n_points))
        if alpha <= 0.0 or n_points < 3:
            return weights

        coords = mid_xyz
        fields = field_t
        if bool(self.c.field_spatial_use_denormalized):
            coords = coords * self.mid_xyz_scale.to(device=coords.device, dtype=coords.dtype)
            fields = fields * self.field_scale.to(device=fields.device, dtype=fields.dtype)

        dims = self._select_plane_dims(coords, n_dims=2)
        if len(dims) < 2:
            self._warn_once(
                "spatial_weight_skip",
                "Spatial weighting skipped: fewer than 2 varying coordinate axes.",
            )
            return weights

        k = min(max(2, int(self.c.field_spatial_knn_k)), n_points - 1)
        if k < 2:
            return weights

        coords_sel = coords[:, dims, :].float()
        idx = knn_graph(coords_sel, k=k)  # (B, N, k)
        coords_t = coords_sel.transpose(1, 2)  # (B, N, d)
        batch_idx = torch.arange(bsz, device=field_t.device).view(bsz, 1, 1).expand(-1, n_points, k)

        nbr_coords = coords_t[batch_idx, idx]        # (B, N, k, d)
        ctr_coords = coords_t.unsqueeze(2)           # (B, N, 1, d)
        dist = torch.linalg.vector_norm(nbr_coords - ctr_coords, dim=-1).clamp_min(1e-8)  # (B, N, k)

        pressure = fields[:, 0, :]                   # (B, N)
        vel_mag = torch.linalg.vector_norm(fields[:, 1:4, :], dim=1)  # (B, N)

        def local_grad_indicator(values_bn: torch.Tensor) -> torch.Tensor:
            nbr = values_bn[batch_idx, idx]          # (B, N, k)
            ctr = values_bn.unsqueeze(-1)            # (B, N, 1)
            g = ((nbr - ctr).abs() / dist).mean(dim=-1)  # (B, N)
            mean_g = g.mean(dim=1, keepdim=True).clamp_min(1e-8)
            return g / mean_g

        gp = local_grad_indicator(pressure)
        gv = local_grad_indicator(vel_mag)
        indicator = 0.5 * (gp + gv)
        extra = torch.clamp(alpha * (indicator - 1.0), min=0.0, max=float(self.c.field_spatial_weight_clip))
        weights = 1.0 + extra
        return torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)

    def _field_data_loss(
        self,
        field_p: torch.Tensor,
        field_t: torch.Tensor,
        spatial_weights: torch.Tensor,
    ) -> torch.Tensor:
        bsz, channels, _ = field_p.shape
        ch_w = self._get_channel_weights(
            self.c.field_channel_weights,
            channels,
            field_p,
            warn_key="bad_field_channel_weights",
        )
        err = self._elementwise_error(
            field_p,
            field_t,
            loss_type=self.c.field_data_loss_type,
            huber_delta=float(self.c.field_huber_delta),
            warn_key="bad_field_data_loss_type",
        )  # (B, C, N)
        point_w = spatial_weights.unsqueeze(1)  # (B, 1, N)
        denom_points = point_w.sum(dim=-1).clamp_min(1e-8)  # (B, 1)
        per_channel = (err * point_w).sum(dim=-1) / denom_points  # (B, C)

        denom_channels = ch_w.sum().clamp_min(1e-8)
        per_sample = (per_channel * ch_w.view(1, channels)).sum(dim=1) / denom_channels
        return per_sample.mean()

    def _field_gradient_loss(
        self,
        field_p: torch.Tensor,
        field_t: torch.Tensor,
        mid_xyz: torch.Tensor,
        spatial_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Supervised local-structure loss on pairwise directional derivatives.
        """
        zero = field_p.new_zeros(())
        bsz, channels, n_points = field_p.shape
        if n_points < 3:
            return zero

        coords = mid_xyz
        pred = field_p
        true = field_t
        if bool(self.c.field_grad_use_denormalized):
            coords = coords * self.mid_xyz_scale.to(device=coords.device, dtype=coords.dtype)
            scale = self.field_scale.to(device=pred.device, dtype=pred.dtype)
            pred = pred * scale
            true = true * scale

        dims = self._select_plane_dims(coords, n_dims=2)
        if len(dims) < 2:
            self._warn_once(
                "field_grad_skip_dims",
                "Field-gradient loss skipped: fewer than 2 varying coordinate axes.",
            )
            return zero

        k = min(max(2, int(self.c.field_grad_knn_k)), n_points - 1)
        if k < 2:
            return zero

        coords_sel = coords[:, dims, :].float()                 # (B, d, N)
        idx = knn_graph(coords_sel, k=k)                        # (B, N, k)
        coords_t = coords_sel.transpose(1, 2)                   # (B, N, d)
        pred_t = pred.transpose(1, 2)                           # (B, N, C)
        true_t = true.transpose(1, 2)                           # (B, N, C)

        batch_idx = torch.arange(bsz, device=field_p.device).view(bsz, 1, 1).expand(-1, n_points, k)
        nbr_coords = coords_t[batch_idx, idx]                   # (B, N, k, d)
        ctr_coords = coords_t.unsqueeze(2)                      # (B, N, 1, d)
        dist = torch.linalg.vector_norm(nbr_coords - ctr_coords, dim=-1, keepdim=True).clamp_min(1e-8)  # (B,N,k,1)

        pred_grad = (pred_t[batch_idx, idx] - pred_t.unsqueeze(2)) / dist  # (B, N, k, C)
        true_grad = (true_t[batch_idx, idx] - true_t.unsqueeze(2)) / dist  # (B, N, k, C)

        if bool(self.c.field_grad_relative):
            eps = float(self.c.field_grad_relative_eps)
            grad_scale = true_grad.abs().mean(dim=2, keepdim=True).detach().clamp_min(eps)  # (B, N, 1, C)
            pred_grad = pred_grad / grad_scale
            true_grad = true_grad / grad_scale

        grad_err = self._elementwise_error(
            pred_grad,
            true_grad,
            loss_type=self.c.field_grad_loss_type,
            huber_delta=float(self.c.field_grad_huber_delta),
            warn_key="bad_field_grad_loss_type",
        )  # (B, N, k, C)
        grad_err = grad_err.mean(dim=2).permute(0, 2, 1).contiguous()  # (B, C, N)

        ch_w = self._get_channel_weights(
            self.c.field_grad_channel_weights,
            channels,
            field_p,
            warn_key="bad_field_grad_channel_weights",
        )
        point_w = spatial_weights.unsqueeze(1)  # (B,1,N)
        denom_points = point_w.sum(dim=-1).clamp_min(1e-8)  # (B,1)
        per_channel = (grad_err * point_w).sum(dim=-1) / denom_points  # (B,C)
        per_sample = (per_channel * ch_w.view(1, channels)).sum(dim=1) / ch_w.sum().clamp_min(1e-8)
        return per_sample.mean()

    def _select_plane_dims(self, coords: torch.Tensor, n_dims: int) -> list[int]:
        spans = (coords.max(dim=-1).values - coords.min(dim=-1).values).mean(dim=0)
        vals, idx = torch.topk(spans, k=min(n_dims, spans.numel()))
        tol = float(self.c.physics_axis_span_tol)
        valid = [int(i) for v, i in zip(vals.tolist(), idx.tolist()) if v > tol]
        return sorted(valid)

    def _local_divergence(self, coords_sel: torch.Tensor, vel_sel: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Estimate divergence with a local least-squares gradient fit.

        coords_sel : (B, d, N)
        vel_sel    : (B, d, N) where velocity components correspond to coords_sel axes
        """
        bsz, dim, n_pts = coords_sel.shape
        if dim < 2 or n_pts < 3:
            return None

        k = min(max(2, int(self.c.physics_knn_k)), n_pts - 1)
        if k < 2:
            return None

        coords_sel = coords_sel.float()
        vel_sel = vel_sel.float()

        idx = knn_graph(coords_sel, k=k)  # (B, N, k)
        coords_t = coords_sel.transpose(1, 2)  # (B, N, d)
        vel_t = vel_sel.transpose(1, 2)        # (B, N, d)

        batch_idx = torch.arange(bsz, device=coords_sel.device).view(bsz, 1, 1).expand(-1, n_pts, k)
        nbr_coords = coords_t[batch_idx, idx]          # (B, N, k, d)
        ctr_coords = coords_t.unsqueeze(2)             # (B, N, 1, d)
        dx = nbr_coords - ctr_coords                   # (B, N, k, d)

        nbr_vel = vel_t[batch_idx, idx]                # (B, N, k, d)
        ctr_vel = vel_t.unsqueeze(2)                   # (B, N, 1, d)
        du = nbr_vel - ctr_vel                         # (B, N, k, d)

        dx_t = dx.transpose(-2, -1)                    # (B, N, d, k)
        a_mat = torch.matmul(dx_t, dx)                 # (B, N, d, d)
        b_mat = torch.matmul(dx_t, du)                 # (B, N, d, d)

        eye = torch.eye(dim, device=coords_sel.device, dtype=coords_sel.dtype).view(1, 1, dim, dim)
        a_mat = a_mat + float(self.c.physics_lstsq_eps) * eye

        try:
            jac = torch.linalg.solve(a_mat, b_mat)     # (B, N, d, d)
        except RuntimeError:
            jac = torch.matmul(torch.linalg.pinv(a_mat), b_mat)

        div = torch.diagonal(jac, dim1=-2, dim2=-1).sum(dim=-1)  # (B, N)
        return torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)

    def _mass_loss_fvm_scaffold(self, batch: Optional[dict]) -> Optional[torch.Tensor]:
        """
        Finite-volume mass residual scaffold.

        The current training batch does not expose mesh cells/faces/areas/normals, so
        this returns None and the caller falls back to point-cloud divergence.
        """
        if batch is None:
            return None

        needed = ("cell_face_index", "face_normals", "face_areas")
        if all(k in batch for k in needed):
            self._warn_once(
                "fv_todo",
                "Finite-volume mass residual metadata detected but interpolation wiring is not implemented yet.",
            )
        return None

    def _mass_loss(self, field_p: torch.Tensor, mid_xyz: torch.Tensor, batch: Optional[dict]) -> torch.Tensor:
        zero = field_p.new_zeros(())

        fv_loss = self._mass_loss_fvm_scaffold(batch)
        if fv_loss is not None:
            return fv_loss

        vel = field_p[:, 1:4, :]
        coords = mid_xyz
        if bool(self.c.physics_use_denormalized):
            vel = vel * self.field_scale[:, 1:4, :].to(device=vel.device, dtype=vel.dtype)
            coords = coords * self.mid_xyz_scale.to(device=coords.device, dtype=coords.dtype)

        mode = str(self.c.physics_slice_mode).lower()
        dims: list[int]
        scale = 1.0

        if mode == "quasi2d":
            dims = self._select_plane_dims(coords, n_dims=2)
            if len(dims) < 2:
                self._warn_once("quasi2d_skip", "quasi2d continuity skipped: fewer than 2 varying coordinate axes.")
                return zero
        elif mode == "full3d":
            dims = [0, 1, 2]
            spans = (coords.max(dim=-1).values - coords.min(dim=-1).values).mean(dim=0)
            tol = float(self.c.physics_axis_span_tol)
            if any(float(spans[d].item()) <= tol for d in dims):
                self._warn_once(
                    "full3d_skip",
                    "full3d continuity skipped: at least one axis is effectively constant in the current representation.",
                )
                return zero
        elif mode == "midplane3d":
            scale = float(self.c.physics_midplane3d_weak_factor)
            if scale <= 0.0:
                self._warn_once(
                    "midplane3d_skip",
                    "slice_mode='midplane3d' with weak factor <= 0: strict continuity disabled for slice-only data.",
                )
                return zero
            dims = self._select_plane_dims(coords, n_dims=2)
            if len(dims) < 2:
                self._warn_once("midplane3d_dims_skip", "midplane3d weak continuity skipped: insufficient varying axes.")
                return zero
            self._warn_once(
                "midplane3d_weak",
                "slice_mode='midplane3d' uses weak in-plane continuity regularization only.",
            )
        else:
            self._warn_once("bad_slice_mode", f"Unknown physics_slice_mode={mode!r}. Mass term skipped.")
            return zero

        coords_sel = coords[:, dims, :]
        vel_sel = vel[:, dims, :]
        div = self._local_divergence(coords_sel, vel_sel)
        if div is None:
            self._warn_once("mass_local_skip", "Mass residual skipped: neighborhood gradient estimation was not feasible.")
            return zero

        return scale * div.pow(2).mean()

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.to(dtype=values.dtype, device=values.device)
        denom = w.sum().clamp_min(1.0)
        return (values * w).sum() / denom

    def _mask_from_batch(
        self,
        batch: Optional[dict],
        key: str,
        n_points: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if batch is None or key not in batch:
            return None
        mask = batch[key]
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        mask = mask.to(device=device)
        if mask.ndim == 3 and mask.shape[1] == 1:
            mask = mask[:, 0, :]
        elif mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if mask.ndim != 2 or mask.shape[1] != n_points:
            self._warn_once(f"bad_mask_{key}", f"Boundary mask {key!r} has unsupported shape {tuple(mask.shape)}.")
            return None
        return (mask > 0.5).to(dtype=torch.float32)

    def _vector_field_from_batch(
        self,
        batch: Optional[dict],
        key: str,
        bsz: int,
        n_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if batch is None or key not in batch:
            return None
        vec = batch[key]
        if not torch.is_tensor(vec):
            vec = torch.as_tensor(vec)
        vec = vec.to(device=device, dtype=dtype)

        if vec.ndim == 3 and vec.shape == (bsz, 3, n_points):
            return vec
        if vec.ndim == 3 and vec.shape == (bsz, n_points, 3):
            return vec.permute(0, 2, 1).contiguous()
        if vec.ndim == 2 and vec.shape == (bsz, 3):
            return vec.unsqueeze(-1).expand(-1, -1, n_points)
        if vec.ndim == 1 and vec.numel() == 3:
            return vec.view(1, 3, 1).expand(bsz, -1, n_points)

        self._warn_once(f"bad_vec_{key}", f"Boundary vector field {key!r} has unsupported shape {tuple(vec.shape)}.")
        return None

    def _scalar_field_from_batch(
        self,
        batch: Optional[dict],
        key: str,
        bsz: int,
        n_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if batch is None or key not in batch:
            return None
        val = batch[key]
        if not torch.is_tensor(val):
            val = torch.as_tensor(val)
        val = val.to(device=device, dtype=dtype)

        if val.ndim == 2 and val.shape == (bsz, n_points):
            return val
        if val.ndim == 2 and val.shape == (bsz, 1):
            return val.expand(-1, n_points)
        if val.ndim == 1 and val.numel() == bsz:
            return val.unsqueeze(1).expand(-1, n_points)
        if val.ndim == 0:
            return val.view(1, 1).expand(bsz, n_points)

        self._warn_once(f"bad_scalar_{key}", f"Boundary scalar field {key!r} has unsupported shape {tuple(val.shape)}.")
        return None

    def _boundary_loss(self, field_p: torch.Tensor, batch: Optional[dict]) -> torch.Tensor:
        zero = field_p.new_zeros(())
        if batch is None:
            return zero

        bsz, _, n_points = field_p.shape
        vel = field_p[:, 1:4, :]
        p = field_p[:, 0, :]
        if bool(self.c.physics_use_denormalized):
            vel = vel * self.field_scale[:, 1:4, :].to(device=vel.device, dtype=vel.dtype)
            p = p * self.field_scale[:, 0:1, :].to(device=p.device, dtype=p.dtype).squeeze(1)

        terms: list[torch.Tensor] = []
        device = field_p.device
        dtype = field_p.dtype

        wall_mask = self._mask_from_batch(batch, "wall_mask", n_points, device)
        if wall_mask is not None and wall_mask.sum() > 0:
            wall_mode = str(self.c.bc_wall_mode).lower()
            if wall_mode == "no_slip":
                terms.append(self._masked_mean((vel ** 2).sum(dim=1), wall_mask))
            elif wall_mode == "no_penetration":
                normals = self._vector_field_from_batch(batch, "boundary_normals", bsz, n_points, device, dtype)
                if normals is None:
                    self._warn_once("wall_normals_missing", "wall no-penetration requested but boundary_normals are missing.")
                else:
                    normal_vel = (vel * normals).sum(dim=1)
                    terms.append(self._masked_mean(normal_vel ** 2, wall_mask))
            else:
                self._warn_once("bad_wall_mode", f"Unknown bc_wall_mode={wall_mode!r}. Wall term skipped.")

        sym_mask = self._mask_from_batch(batch, "symmetry_mask", n_points, device)
        if sym_mask is not None and sym_mask.sum() > 0:
            normals = self._vector_field_from_batch(batch, "boundary_normals", bsz, n_points, device, dtype)
            if normals is None:
                self._warn_once("sym_normals_missing", "symmetry_mask provided but boundary_normals are missing.")
            else:
                normal_vel = (vel * normals).sum(dim=1)
                terms.append(self._masked_mean(normal_vel ** 2, sym_mask))

        inlet_mask = self._mask_from_batch(batch, "inlet_mask", n_points, device)
        if inlet_mask is not None and inlet_mask.sum() > 0:
            inlet_target = self._vector_field_from_batch(batch, "inlet_velocity_target", bsz, n_points, device, dtype)
            if inlet_target is None:
                self._warn_once("inlet_target_missing", "inlet_mask provided but inlet_velocity_target is missing.")
            else:
                inlet_err = ((vel - inlet_target) ** 2).sum(dim=1)
                terms.append(self._masked_mean(inlet_err, inlet_mask))

        outlet_mask = self._mask_from_batch(batch, "outlet_mask", n_points, device)
        if outlet_mask is not None and outlet_mask.sum() > 0:
            outlet_target = self._scalar_field_from_batch(batch, "outlet_pressure_target", bsz, n_points, device, dtype)
            if outlet_target is None:
                self._warn_once("outlet_target_missing", "outlet_mask provided but outlet_pressure_target is missing.")
            else:
                terms.append(self._masked_mean((p - outlet_target) ** 2, outlet_mask))

        if not terms:
            self._warn_once(
                "bc_unavailable",
                "BC loss requested but no compatible boundary tags/targets were found in the training batch.",
            )
            return zero
        return torch.stack(terms).mean()

    def _momentum_loss_scaffold(self, batch: Optional[dict], ref_tensor: torch.Tensor) -> torch.Tensor:
        """
        Momentum residual scaffold.

        A consistent RANS momentum residual needs additional physics metadata
        (density, viscosity/eddy viscosity and discretization support). This
        scaffold intentionally returns zero until those are available.
        """
        zero = ref_tensor.new_zeros(())
        required = ("density", "viscosity")
        if batch is None or any(k not in batch for k in required):
            self._warn_once(
                "momentum_missing",
                "Momentum residual is enabled but required fields (density/viscosity) are not present; term is kept at zero.",
            )
            return zero

        self._warn_once(
            "momentum_todo",
            "Momentum residual scaffold reached but a consistent discretization is not implemented yet; term is zero.",
        )
        return zero

    def forward(
        self,
        scalar_p: torch.Tensor,
        scalar_t: torch.Tensor,
        field_p: torch.Tensor,
        field_t: torch.Tensor,
        mid_xyz: torch.Tensor,
        batch: Optional[dict] = None,
        epoch: int = 1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_scalar = F.mse_loss(scalar_p, scalar_t)
        spatial_weights = self._spatial_weight_map(field_t, mid_xyz)
        loss_field = self._field_data_loss(field_p, field_t, spatial_weights)

        zero = loss_scalar.new_zeros(())
        loss_field_grad = zero
        if float(self.c.lambda_field_grad) > 0.0:
            loss_field_grad = self._field_gradient_loss(field_p, field_t, mid_xyz, spatial_weights)

        loss_data = (
            float(self.c.lambda_scalars) * loss_scalar
            + float(self.c.lambda_fields) * loss_field
            + float(self.c.lambda_field_grad) * loss_field_grad
        )

        loss_thrust = zero
        loss_mass = zero
        loss_bc = zero
        loss_momentum = zero

        if float(self.c.lambda_thrust) > 0.0:
            loss_thrust = self._thrust_loss(scalar_p, scalar_t)
        if float(self.c.lambda_mass) > 0.0:
            loss_mass = self._mass_loss(field_p, mid_xyz, batch)
        if float(self.c.lambda_bc) > 0.0:
            loss_bc = self._boundary_loss(field_p, batch)
        if float(self.c.lambda_momentum) > 0.0:
            loss_momentum = self._momentum_loss_scaffold(batch, loss_data)

        ramp = float(self._physics_ramp(epoch))
        ramp_t = loss_data.new_tensor(ramp)

        weighted_data = float(self.c.lambda_data) * loss_data
        weighted_thrust = ramp_t * float(self.c.lambda_thrust) * loss_thrust
        weighted_mass = ramp_t * float(self.c.lambda_mass) * loss_mass
        weighted_bc = ramp_t * float(self.c.lambda_bc) * loss_bc
        weighted_momentum = ramp_t * float(self.c.lambda_momentum) * loss_momentum

        total = weighted_data + weighted_thrust + weighted_mass + weighted_bc + weighted_momentum

        terms = {
            "loss_total": total.detach(),
            "loss_data": loss_data.detach(),
            "loss_scalar": loss_scalar.detach(),
            "loss_field": loss_field.detach(),
            "loss_field_grad": loss_field_grad.detach(),
            "loss_thrust": loss_thrust.detach(),
            "loss_mass": loss_mass.detach(),
            "loss_bc": loss_bc.detach(),
            "loss_momentum": loss_momentum.detach(),
            "physics_ramp": ramp_t.detach(),
            "spatial_weight_mean": spatial_weights.mean().detach(),
            "weighted_data": weighted_data.detach(),
            "weighted_thrust": weighted_thrust.detach(),
            "weighted_mass": weighted_mass.detach(),
            "weighted_bc": weighted_bc.detach(),
            "weighted_momentum": weighted_momentum.detach(),
        }
        return total, terms
