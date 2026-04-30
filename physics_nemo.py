"""
physics_nemo.py - Tier 2 PhysicsNeMo physics losses for the DGCNN surrogate.

Upgrades over Tier 1 (physics.py):
  - Uses physicsnemo.sym.NavierStokes to define and document PDE equations
  - Full 3D continuity: du/dx + dv/dy + dw/dz = 0
  - Euler momentum on the xz midplane:
      u*du/dx + w*du/dz + (1/rho)*dp/dx = 0   (x-momentum)
      u*dw/dx + w*dw/dz + (1/rho)*dp/dz = 0   (z-momentum)
  - No-slip wall BC: u = v = w = 0 at body/EDF surface

Architecture fit
----------------
PhysicsNeMo's sym framework is designed for INR networks (coordinate -> field).
Our DGCNN is a conditional model (geometry + coords -> field). Rather than
forcing a full Domain/Constraint/Solver rewrite, we use PhysicsNeMo's
NavierStokes class for equation definition and verification, then evaluate
the residuals using torch.autograd through the FiLM-SIREN field head.

This gives:
  - PhysicsNeMo traceability (equations are logged on init)
  - Exact spatial Jacobians via autograd (same as Tier 1)
  - Momentum equations as a genuine physics upgrade over Tier 1 continuity-only

Derivative chain rule (same as Tier 1):
  d(field_phys)/d(x_phys) = [d(field_norm)/d(x_norm)] * (field_scale / mid_xyz_scale)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _try_import_navier_stokes():
    """Import NavierStokes from physicsnemo, return None if not installed."""
    try:
        from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
        return NavierStokes
    except ImportError:
        return None


class PhysicsNeMoLoss(nn.Module):
    """
    Tier 2 physics loss using PhysicsNeMo NavierStokes equations.

    Parameters
    ----------
    nu : float
        Kinematic viscosity [m^2/s]. Air at STP: 1.5e-5.
        Used for documentation/logging only — momentum uses inviscid Euler form.
    rho : float
        Air density [kg/m^3]. Air at STP: 1.225.
    lambda_continuity : float
        Weight for the full 3D continuity residual (du/dx+dv/dy+dw/dz=0).
    lambda_momentum : float
        Weight for the inviscid xz-momentum residual.
        Recommended: start at 1e-4 (pressure gradients are large).
    lambda_wall_bc : float
        Weight for the no-slip wall boundary condition.
    """

    def __init__(
        self,
        nu: float = 1.5e-5,
        rho: float = 1.225,
        lambda_continuity: float = 0.0,
        lambda_momentum: float = 0.0,
        lambda_wall_bc: float = 0.0,
    ):
        super().__init__()
        self.nu = float(nu)
        self.rho = float(rho)
        self.lambda_continuity = float(lambda_continuity)
        self.lambda_momentum   = float(lambda_momentum)
        self.lambda_wall_bc    = float(lambda_wall_bc)

        # Log PhysicsNeMo equation definitions on init for traceability
        NavierStokes = _try_import_navier_stokes()
        if NavierStokes is not None:
            ns = NavierStokes(nu=nu, rho=rho, dim=3, time=False)
            print("[PhysicsNeMoLoss] NavierStokes equations registered:")
            for name, expr in ns.equations.items():
                print(f"  {name}: {expr}")
        else:
            print("[PhysicsNeMoLoss] physicsnemo.sym not available; "
                  "equations evaluated via raw autograd.")

    @staticmethod
    def _grad(
        scalar_field: torch.Tensor,   # (B, N)
        xyz_with_grad: torch.Tensor,  # (B, 3, N) requires_grad=True
        retain: bool = True,
    ) -> torch.Tensor:
        """d(scalar_field)/d(xyz) via autograd. Returns (B, 3, N)."""
        return torch.autograd.grad(
            scalar_field.sum(),
            xyz_with_grad,
            create_graph=True,
            retain_graph=retain,
        )[0]

    def _get_fields_and_derivs(
        self,
        model: nn.Module,
        g: torch.Tensor,
        mid_xyz_norm: torch.Tensor,
        field_scale: torch.Tensor,
        mid_xyz_scale: torch.Tensor,
    ) -> dict:
        """
        Query field head and compute all first-order spatial derivatives needed
        for continuity and Euler momentum residuals.

        Returns dict of physical-unit field values and derivatives (B, N) each.
        """
        g_d  = g.detach().float()
        xyz_g = mid_xyz_norm.detach().float().requires_grad_(True)  # (B, 3, N)

        fields = model.field_from_embedding(g_d, xyz_g)   # (B, 4, N)

        fs = field_scale.to(fields.device).float()         # (4,) [p, u, v, w]
        ms = mid_xyz_scale.to(xyz_g.device).float()        # (3,) [x, y, z]

        # Physical field values (B, N)
        p_phys = fields[:, 0, :] * fs[0]
        u_phys = fields[:, 1, :] * fs[1]
        v_phys = fields[:, 2, :] * fs[2]
        w_phys = fields[:, 3, :] * fs[3]

        # Spatial Jacobians via autograd (B, 3, N) in normalised coords.
        # chain rule: d/dx_phys = (1/ms[0]) * d/dx_norm
        grad_u = self._grad(u_phys, xyz_g, retain=True)   # (B, 3, N)
        grad_v = self._grad(v_phys, xyz_g, retain=True)
        grad_w = self._grad(w_phys, xyz_g, retain=True)
        grad_p = self._grad(p_phys, xyz_g, retain=True)

        def dx(g, axis): return g[:, axis, :] / ms[axis]

        # Non-dimensional reference scales (computed once per batch).
        # For each axis, per-axis scale = field_scale / xyz_scale — its
        # magnitude sets the natural size of the corresponding derivative in
        # physical units. We normalise residuals by the dominant axis scale
        # so the loss magnitude is O(1) at model initialisation regardless
        # of anisotropic domain dimensions (e.g. thin midplane slice).
        ax_u = fs[1] / ms[0]
        ax_v = fs[2] / ms[1]
        ax_w = fs[3] / ms[2]
        div_ref = torch.stack([ax_u, ax_v, ax_w]).abs().max().clamp(min=1e-8)

        # Momentum residual scale: max of convective (U^2/L) and pressure
        # gradient (p/(rho*L)) magnitudes, across x and z axes.
        U_max = fs[1:4].abs().max()
        conv_x = U_max * (fs[1] / ms[0]).abs()
        conv_z = U_max * (fs[3] / ms[2]).abs()
        pres_x = (fs[0] / (self.rho * ms[0])).abs()
        pres_z = (fs[0] / (self.rho * ms[2])).abs()
        mom_ref = torch.stack([conv_x, conv_z, pres_x, pres_z]).max().clamp(min=1e-8)

        return dict(
            u=u_phys, v=v_phys, w=w_phys, p=p_phys,
            # continuity terms
            du_dx=dx(grad_u, 0),
            dv_dy=dx(grad_v, 1),
            dw_dz=dx(grad_w, 2),
            # x-momentum terms
            du_dz=dx(grad_u, 2),
            dp_dx=dx(grad_p, 0),
            # z-momentum terms
            dw_dx=dx(grad_w, 0),
            dp_dz=dx(grad_p, 2),
            # non-dimensionalisation references
            _div_ref=div_ref,
            _mom_ref=mom_ref,
        )

    def continuity_loss(self, d: dict) -> torch.Tensor:
        """
        Full 3D incompressible continuity: du/dx + dv/dy + dw/dz = 0.

        Residual is non-dimensionalised by a characteristic divergence scale
        U_ref / L_ref so the loss magnitude is O(1) at model initialisation
        regardless of field/xyz physical scales.
        """
        div = d['du_dx'] + d['dv_dy'] + d['dw_dz']
        ref = d['_div_ref']                         # scalar, positive
        return ((div / ref) ** 2).mean()

    def momentum_loss(self, d: dict) -> torch.Tensor:
        """
        Inviscid (Euler) momentum projected onto the xz midplane.

        x-momentum:  u*du/dx + w*du/dz + (1/rho)*dp/dx = 0
        z-momentum:  u*dw/dx + w*dw/dz + (1/rho)*dp/dz = 0

        Viscous terms are omitted: at Re~1e6 (drone cruise) the laminar
        viscosity contribution is O(1e-5) of the convective terms.

        Residual is non-dimensionalised by (p_ref / (rho * L_ref)) so the
        loss magnitude is O(1) at initialisation.
        """
        inv_rho = 1.0 / self.rho
        u, w    = d['u'], d['w']

        res_x = u * d['du_dx'] + w * d['du_dz'] + inv_rho * d['dp_dx']
        res_z = u * d['dw_dx'] + w * d['dw_dz'] + inv_rho * d['dp_dz']

        ref = d['_mom_ref']                         # scalar, positive
        return (((res_x / ref) ** 2) + ((res_z / ref) ** 2)).mean()

    def wall_bc_loss(
        self,
        model: nn.Module,
        g: torch.Tensor,
        wall_pts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """No-slip: u = v = w = 0 at body/EDF surface points."""
        if wall_pts is None or wall_pts.shape[-1] == 0:
            return g.new_zeros(())
        g_d      = g.detach().float()
        wall_f   = wall_pts.to(device=g.device).float()
        wall_fld = model.field_from_embedding(g_d, wall_f)   # (B, 4, N_wall)
        return (wall_fld[:, 1:4, :] ** 2).mean()

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
        Compute all active physics losses.

        Returns
        -------
        total  : scalar tensor (weighted sum)
        breakdown : dict with detached per-term losses for logging
        """
        zero       = g.new_zeros(())
        loss_cont  = zero
        loss_mom   = zero
        loss_wall  = zero

        need_derivs = self.lambda_continuity > 0.0 or self.lambda_momentum > 0.0
        if need_derivs:
            d = self._get_fields_and_derivs(
                model, g, mid_xyz_norm, field_scale, mid_xyz_scale
            )
            if self.lambda_continuity > 0.0:
                loss_cont = self.continuity_loss(d)
            if self.lambda_momentum > 0.0:
                loss_mom  = self.momentum_loss(d)

        if self.lambda_wall_bc > 0.0:
            loss_wall = self.wall_bc_loss(model, g, wall_pts)

        total = (self.lambda_continuity * loss_cont
                 + self.lambda_momentum  * loss_mom
                 + self.lambda_wall_bc   * loss_wall)

        return total, {
            "loss_continuity": loss_cont.detach(),
            "loss_momentum":   loss_mom.detach(),
            "loss_wall_bc":    loss_wall.detach(),
        }
