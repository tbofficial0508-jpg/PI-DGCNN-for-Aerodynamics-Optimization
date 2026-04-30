from __future__ import annotations

import unittest

import torch

from config import Config
from losses import HybridPhysicsLoss


def _make_norm_stats() -> dict:
    return {
        "field_scale": [1.0, 1.0, 1.0, 1.0],
        "mid_xyz_scale": [1.0, 1.0, 1.0],
    }


def _make_basic_batch(batch_size: int = 2, n_points: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    scalar_p = torch.randn(batch_size, 2)
    scalar_t = torch.randn(batch_size, 2)
    field_p = torch.randn(batch_size, 4, n_points)
    field_t = torch.randn(batch_size, 4, n_points)
    return scalar_p, scalar_t, field_p, field_t


def _make_quasi2d_coords(n_side: int = 5) -> torch.Tensor:
    x_lin = torch.linspace(-1.0, 1.0, n_side)
    z_lin = torch.linspace(-1.0, 1.0, n_side)
    xg, zg = torch.meshgrid(x_lin, z_lin, indexing="ij")
    x = xg.reshape(1, -1)
    z = zg.reshape(1, -1)
    y = torch.zeros_like(x)
    coords = torch.stack([x, y, z], dim=1)  # (1, 3, N)
    return coords


class HybridPhysicsLossTests(unittest.TestCase):
    def test_total_loss_is_finite(self) -> None:
        c = Config(
            lambda_data=1.0,
            lambda_scalars=1.0,
            lambda_fields=0.1,
            lambda_thrust=0.2,
            lambda_mass=0.2,
            physics_slice_mode="quasi2d",
            physics_schedule="linear",
            physics_warmup_epochs=5,
            physics_knn_k=8,
            physics_use_denormalized=False,
            thrust_loss_type="huber",
        )
        loss_fn = HybridPhysicsLoss(c, _make_norm_stats())

        scalar_p, scalar_t, field_p, field_t = _make_basic_batch(batch_size=2, n_points=32)
        mid_xyz = torch.randn(2, 3, 32)
        total, terms = loss_fn(
            scalar_p=scalar_p,
            scalar_t=scalar_t,
            field_p=field_p,
            field_t=field_t,
            mid_xyz=mid_xyz,
            batch={},
            epoch=3,
        )

        self.assertTrue(torch.isfinite(total).item())
        for key, value in terms.items():
            self.assertTrue(torch.isfinite(value).item(), msg=f"{key} is not finite")

    def test_disabled_physics_terms_are_zero(self) -> None:
        c = Config(
            lambda_data=1.0,
            lambda_scalars=1.0,
            lambda_fields=0.1,
            lambda_thrust=0.0,
            lambda_mass=0.0,
            lambda_bc=0.0,
            lambda_momentum=0.0,
            physics_schedule="none",
        )
        loss_fn = HybridPhysicsLoss(c, _make_norm_stats())

        scalar_p, scalar_t, field_p, field_t = _make_basic_batch(batch_size=2, n_points=16)
        mid_xyz = torch.randn(2, 3, 16)
        _, terms = loss_fn(
            scalar_p=scalar_p,
            scalar_t=scalar_t,
            field_p=field_p,
            field_t=field_t,
            mid_xyz=mid_xyz,
            batch={},
            epoch=1,
        )

        self.assertEqual(float(terms["loss_thrust"].item()), 0.0)
        self.assertEqual(float(terms["loss_mass"].item()), 0.0)
        self.assertEqual(float(terms["loss_bc"].item()), 0.0)
        self.assertEqual(float(terms["loss_momentum"].item()), 0.0)

    def test_enabling_thrust_term_changes_loss(self) -> None:
        scalar_p = torch.tensor([[0.0, 2.0], [0.0, 2.0]], dtype=torch.float32)
        scalar_t = torch.zeros_like(scalar_p)
        field_p = torch.zeros(2, 4, 10)
        field_t = torch.zeros_like(field_p)
        mid_xyz = torch.zeros(2, 3, 10)

        c_off = Config(
            lambda_data=0.0,
            lambda_scalars=0.0,
            lambda_fields=0.0,
            lambda_thrust=0.0,
            physics_schedule="none",
        )
        c_on = Config(
            lambda_data=0.0,
            lambda_scalars=0.0,
            lambda_fields=0.0,
            lambda_thrust=1.0,
            thrust_loss_type="mse",
            physics_schedule="none",
        )
        loss_off = HybridPhysicsLoss(c_off, _make_norm_stats())
        loss_on = HybridPhysicsLoss(c_on, _make_norm_stats())

        total_off, terms_off = loss_off(scalar_p, scalar_t, field_p, field_t, mid_xyz, batch={}, epoch=1)
        total_on, terms_on = loss_on(scalar_p, scalar_t, field_p, field_t, mid_xyz, batch={}, epoch=1)

        self.assertEqual(float(terms_off["loss_thrust"].item()), 0.0)
        self.assertGreater(float(terms_on["loss_thrust"].item()), 0.0)
        self.assertGreater(float(total_on.item()), float(total_off.item()))

    def test_quasi2d_mass_term_on_synthetic_flow(self) -> None:
        coords = _make_quasi2d_coords(n_side=5)
        n_points = coords.shape[-1]

        c = Config(
            lambda_data=0.0,
            lambda_scalars=0.0,
            lambda_fields=0.0,
            lambda_mass=1.0,
            physics_slice_mode="quasi2d",
            physics_schedule="none",
            physics_knn_k=8,
            physics_use_denormalized=False,
        )
        loss_fn = HybridPhysicsLoss(c, _make_norm_stats())

        scalar = torch.zeros(1, 2)
        target_scalar = torch.zeros_like(scalar)
        target_fields = torch.zeros(1, 4, n_points)

        # Divergence-free in x-z plane: u=-z, w=+x.
        field_div_free = torch.zeros(1, 4, n_points)
        x = coords[:, 0, :]
        z = coords[:, 2, :]
        field_div_free[:, 1, :] = -z
        field_div_free[:, 3, :] = x
        _, terms_free = loss_fn(
            scalar,
            target_scalar,
            field_div_free,
            target_fields,
            coords,
            batch={},
            epoch=1,
        )

        # Divergent field in x-z plane: u=x, w=z => div ~= 2.
        field_div = torch.zeros(1, 4, n_points)
        field_div[:, 1, :] = x
        field_div[:, 3, :] = z
        _, terms_div = loss_fn(
            scalar,
            target_scalar,
            field_div,
            target_fields,
            coords,
            batch={},
            epoch=1,
        )

        self.assertLess(float(terms_free["loss_mass"].item()), 1e-3)
        self.assertGreater(float(terms_div["loss_mass"].item()), float(terms_free["loss_mass"].item()) + 1e-2)

    def test_field_gradient_term_is_zero_when_disabled(self) -> None:
        coords = _make_quasi2d_coords(n_side=4)
        n_points = coords.shape[-1]
        scalar = torch.zeros(1, 2)
        field_t = torch.zeros(1, 4, n_points)
        field_p = torch.randn(1, 4, n_points)

        c = Config(
            lambda_data=1.0,
            lambda_scalars=0.0,
            lambda_fields=1.0,
            lambda_field_grad=0.0,
            physics_schedule="none",
        )
        loss_fn = HybridPhysicsLoss(c, _make_norm_stats())
        _, terms = loss_fn(
            scalar_p=scalar,
            scalar_t=scalar,
            field_p=field_p,
            field_t=field_t,
            mid_xyz=coords,
            batch={},
            epoch=1,
        )

        self.assertEqual(float(terms["loss_field_grad"].item()), 0.0)

    def test_field_gradient_term_increases_with_gradient_mismatch(self) -> None:
        coords = _make_quasi2d_coords(n_side=5)
        n_points = coords.shape[-1]
        x = coords[:, 0, :]
        z = coords[:, 2, :]

        scalar = torch.zeros(1, 2)
        field_t = torch.zeros(1, 4, n_points)
        field_t[:, 1, :] = x + z

        field_p_match = field_t.clone()
        field_p_bad = torch.zeros_like(field_t)

        c = Config(
            lambda_data=1.0,
            lambda_scalars=0.0,
            lambda_fields=0.0,
            lambda_field_grad=1.0,
            field_grad_loss_type="mse",
            field_grad_knn_k=8,
            field_grad_use_denormalized=False,
            physics_schedule="none",
        )
        loss_fn = HybridPhysicsLoss(c, _make_norm_stats())

        _, terms_match = loss_fn(
            scalar_p=scalar,
            scalar_t=scalar,
            field_p=field_p_match,
            field_t=field_t,
            mid_xyz=coords,
            batch={},
            epoch=1,
        )
        _, terms_bad = loss_fn(
            scalar_p=scalar,
            scalar_t=scalar,
            field_p=field_p_bad,
            field_t=field_t,
            mid_xyz=coords,
            batch={},
            epoch=1,
        )

        self.assertLess(float(terms_match["loss_field_grad"].item()), 1e-8)
        self.assertGreater(float(terms_bad["loss_field_grad"].item()), 1e-4)

    def test_spatial_weighting_boosts_weight_mean(self) -> None:
        coords = _make_quasi2d_coords(n_side=5)
        n_points = coords.shape[-1]
        scalar = torch.zeros(1, 2)
        field_t = torch.zeros(1, 4, n_points)
        field_p = torch.zeros(1, 4, n_points)
        # Localized spike to mimic sharp CFD feature region.
        field_t[:, 0, n_points // 2] = 10.0

        c = Config(
            lambda_data=1.0,
            lambda_scalars=0.0,
            lambda_fields=1.0,
            field_spatial_weight_alpha=1.0,
            field_spatial_weight_clip=3.0,
            field_spatial_knn_k=8,
            field_spatial_use_denormalized=False,
            physics_schedule="none",
        )
        loss_fn = HybridPhysicsLoss(c, _make_norm_stats())
        _, terms = loss_fn(
            scalar_p=scalar,
            scalar_t=scalar,
            field_p=field_p,
            field_t=field_t,
            mid_xyz=coords,
            batch={},
            epoch=1,
        )

        self.assertGreater(float(terms["spatial_weight_mean"].item()), 1.0)


if __name__ == "__main__":
    unittest.main()
