"""
physics.py - Autograd-based physics-informed losses for the DGCNN surrogate.

Computes exact PDE residuals by differentiating the FiLM-SIREN field head
with respect to midplane query coordinates via torch.autograd.

This is the Tier 1 PhysicsNeMo integration: physics constraints are added
to the existing architecture without requiring PhysicsNeMo as a dependency.
The mathematical approach is identical to physicsnemo.sym - automatic
differentiation through the implicit neural representation to evaluate
PDE residuals at collocation points.

Physics enforced
----------------
Continuity (incompressible RANS):
    du/dx + dv/dy + dw/dz = 0

  On the xz symmetry plane (y=0), dv/dy = 0 by symmetry, so this reduces
  to the quasi-2D form:  du/dx + dw/dz = 0

  The chain rule converts normalised-coordinate autograd outputs to
  physical-unit derivatives:
    d(field_phys)/d(x_phys) = [d(field_norm)/d(x_norm)] * (field_scale / xyz_scale)

No-slip wall BC:
    u = v = w = 0  at body/EDF surface points

  The field head is queried at stored wall point coordinates and the
  squared velocity magnitude is penalised.

Why autograd and not finite differences
-----------------------------------------
  The existing _mass_loss in losses.py estimates divergence from a
  local k-NN neighbourhood, which introduces O(h) truncation error
  and is sensitive to point-cloud sampling density near the wall.
  Autograd through the SIREN gives the exact spatial Jacobian of the
  network's prediction function - zero discretisation error.

Design note
-----------
  The encoder embedding g is always detached before the physics pass.
  Gradients flow only through the FiLM-SIREN field head (5 layers,
  cheap) and not through the EdgeConv backbone (expensive). This is
  consistent with the Phase-2 training philosophy and keeps the
  physics regulariser from destabilising the geometry encoder.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AutogradPhysicsLoss(nn.Module):
    """
    Physics-informed loss using automatic differentiation through the SIREN.

    Parameters
    ----------
    lambda_continuity : float
        Weight for the quasi-2D continuity residual loss.
        Recommended starting value: 0.01 (scale relative to field MSE ~1e-3).
    lambda_wall_bc : float
        Weight for the no-slip wall boundary condition loss.
        Recommended starting value: 0.1 (velocity MSE at wall is naturally small).
    """

    def __init__(
        self,
        lambda_continuity: float = 0.0,
        lambda_wall_bc: float = 0.0,
    ):
        super().__init__()
        self.lambda_continuity = float(lambda_continuity)
        self.lambda_wall_bc = float(lambda_wall_bc)

    @staticmethod
    def _scalar_grad(
        scalar_field: torch.Tensor,   # (B, N)
        xyz_with_grad: torch.Tensor,  # (B, 3, N) with requires_grad=True
        retain: bool = True,
    ) -> torch.Tensor:
        """
        Compute d(scalar_field)/d(xyz_with_grad) via autograd.

        Because the FiLM-SIREN is pointwise (Conv1d kernel=1), element i of
        scalar_field depends only on element i of xyz_with_grad.  Therefore:
            autograd(sum(scalar_field), xyz_with_grad)[i] = d(field_i)/d(xyz_i)
        giving exact per-point spatial gradients.

        Returns : (B, 3, N) - gradient along each coordinate axis.
        """
        return torch.autograd.grad(
            scalar_field.sum(),
            xyz_with_grad,
            create_graph=True,
            retain_graph=retain,
        )[0]  # (B, 3, N)

    def continuity_loss(
        self,
        model: nn.Module,
        g: torch.Tensor,               # (B, D) - encoder embedding
        mid_xyz_norm: torch.Tensor,    # (B, 3, N) - normalised midplane coords
        field_scale: torch.Tensor,     # (4,) physical scales [p, u, v, w]
        mid_xyz_scale: torch.Tensor,   # (3,) coord half-ranges [x, y, z]
    ) -> torch.Tensor:
        """
        Quasi-2D continuity residual: du/dx + dw/dz = 0 on the midplane.

        Uses autograd through the FiLM-SIREN field head to obtain exact
        du_norm/dx_norm and dw_norm/dz_norm, then converts to physical
        derivatives via the chain rule before squaring and averaging.
        """
        g_d = g.detach().float()
        # Enable grad on a fresh leaf so autograd tracks the computation graph
        xyz_g = mid_xyz_norm.detach().float().requires_grad_(True)  # (B, 3, N)

        fields = model.field_from_embedding(g_d, xyz_g)   # (B, 4, N)

        fs = field_scale.to(fields.device).float()         # (4,)
        ms = mid_xyz_scale.to(xyz_g.device).float()        # (3,)

        # Physical velocities (B, N) - derivatives will be in physical units
        u_phys = fields[:, 1, :] * fs[1]
        w_phys = fields[:, 3, :] * fs[3]

        # Chain rule:  d(f_phys)/d(x_phys) = [d(f_phys)/d(x_norm)] / mid_xyz_scale
        # d(f_phys)/d(x_norm) is the autograd output divided by mid_xyz_scale[axis]
        # retain_graph=True for both calls: the forward graph (G) must stay alive
        # until phys_total.backward() fires — the backward-of-G graphs created by
        # create_graph=True hold internal references to G and will segfault if G
        # is freed early.  G is freed naturally when the outer backward() completes.
        grad_u = self._scalar_grad(u_phys, xyz_g, retain=True)   # (B, 3, N)
        grad_w = self._scalar_grad(w_phys, xyz_g, retain=True)   # (B, 3, N)

        du_dx = grad_u[:, 0, :] / ms[0]   # (B, N)  physical du/dx
        dw_dz = grad_w[:, 2, :] / ms[2]   # (B, N)  physical dw/dz

        div = du_dx + dw_dz                # quasi-2D divergence
        return (div ** 2).mean()

    def wall_bc_loss(
        self,
        model: nn.Module,
        g: torch.Tensor,           # (B, D) - encoder embedding
        wall_pts: torch.Tensor,    # (B, 3, N_wall) - normalised wall coords
    ) -> torch.Tensor:
        """
        No-slip wall BC: u = v = w = 0 at body surface points.

        Queries the FiLM-SIREN at wall point coordinates and penalises the
        mean squared normalised velocity magnitude. Gradients flow through
        the field-head weights so the network is trained to predict zero
        velocity at the wall.  Pressure is left unconstrained.
        """
        if wall_pts is None or wall_pts.shape[-1] == 0:
            return g.new_zeros(())

        g_d = g.detach().float()
        wall_f = wall_pts.to(device=g.device).float()  # (B, 3, N_wall)

        # No requires_grad on wall_f — we don't need spatial derivatives here.
        # Gradients flow through wall_fields to the field-head parameters.
        wall_fields = model.field_from_embedding(g_d, wall_f)  # (B, 4, N_wall)

        vel_norm = wall_fields[:, 1:4, :]   # (B, 3, N_wall)  normalised u, v, w
        return (vel_norm ** 2).mean()

    def compute(
        self,
        model: nn.Module,
        g: torch.Tensor,
        mid_xyz_norm: torch.Tensor,
        field_scale: torch.Tensor,
        mid_xyz_scale: torch.Tensor,
        wall_pts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute all active physics losses and return their weighted sum.

        Parameters
        ----------
        model         : DGCNN instance
        g             : (B, D) geometry+condition embedding from model.encode()
        mid_xyz_norm  : (B, 3, N) normalised midplane query coordinates
        field_scale   : (4,) physical scale for [p, u, v, w]
        mid_xyz_scale : (3,) physical half-range for [x, y, z]
        wall_pts      : (B, 3, N_wall) normalised wall point coordinates, or None

        Returns
        -------
        total_physics_loss : scalar tensor
        breakdown          : dict with detached per-term losses for logging
        """
        zero = g.new_zeros(())
        loss_cont = zero
        loss_wall = zero

        if self.lambda_continuity > 0.0:
            loss_cont = self.continuity_loss(
                model, g, mid_xyz_norm, field_scale, mid_xyz_scale
            )

        if self.lambda_wall_bc > 0.0:
            loss_wall = self.wall_bc_loss(model, g, wall_pts)

        total = self.lambda_continuity * loss_cont + self.lambda_wall_bc * loss_wall

        return total, {
            "loss_continuity": loss_cont.detach(),
            "loss_wall_bc":    loss_wall.detach(),
        }
