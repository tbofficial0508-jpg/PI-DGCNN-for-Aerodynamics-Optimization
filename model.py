"""
model.py – DGCNN surrogate model for CFD field and scalar prediction.

Architecture
────────────
  Input:  geometry surface point cloud  (B, 8, N_geo)
          midplane query coordinates     (B, 3, N_mid)
          operating conditions           (B, 2) [u_inf, rpm]

  Backbone: 3 x EdgeConv blocks with dynamic k-NN graph recomputation
            per layer, giving multi-scale local geometry representations.
            Skip connections from all three blocks are concatenated before
            global max-pooling.

  Global head   -> predicts [drag, thrust] scalars  (B, 2)
  Point-wise head -> predicts [p, u, v, w] fields at every midplane query
                    point by broadcasting the global geometry embedding
                    alongside Fourier-encoded query coordinates   (B, 4, N_mid)

  Fourier positional encoding (NeRF-style, fourier_levels octaves) breaks the
  MLP spectral bias that traps the field head at mean prediction.

Run sanity check (matches the spec (2, 3, 4096)):
  python model.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import knn_graph


def _effective_group_count(channels: int, requested_groups: int) -> int:
    groups = max(1, min(int(requested_groups), int(channels)))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def make_norm_1d(channels: int, norm_type: str = "batch", norm_groups: int = 8) -> nn.Module:
    mode = str(norm_type).lower()
    if mode == "batch":
        return nn.BatchNorm1d(channels)
    if mode == "group":
        groups = _effective_group_count(channels, norm_groups)
        return nn.GroupNorm(groups, channels)
    if mode == "layer":
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unsupported norm_type={norm_type!r}. Use 'batch', 'group', or 'layer'.")


def make_norm_2d(channels: int, norm_type: str = "batch", norm_groups: int = 8) -> nn.Module:
    mode = str(norm_type).lower()
    if mode == "batch":
        return nn.BatchNorm2d(channels)
    if mode == "group":
        groups = _effective_group_count(channels, norm_groups)
        return nn.GroupNorm(groups, channels)
    if mode == "layer":
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unsupported norm_type={norm_type!r}. Use 'batch', 'group', or 'layer'.")


# ── Fourier positional encoder ────────────────────────────────────────────────

class FourierEncoder(nn.Module):
    """
    NeRF-style Fourier positional encoding for 3-D coordinates.

    Maps raw (x, y, z) -> [x, y, z, sin(pi*x), cos(pi*x), ...,
                            sin(2^{L-1}*pi*x), cos(2^{L-1}*pi*x), ...]
    where L = fourier_levels (one octave per level).

    This breaks the spectral bias of coordinate MLPs: without positional
    encoding, an MLP naturally fits low-frequency components first and can
    stall at the mean-prediction attractor for hundreds of epochs.

    Parameters
    ----------
    fourier_levels : int
        Number of frequency octaves L.  L=6 adds 36 sinusoidal features.
    include_input  : bool
        Prepend the raw coordinates to the sinusoidal features (recommended).

    out_dim : 3*(1 + 2*L) with include_input, else 3*2*L.
    """

    def __init__(self, fourier_levels: int = 6, include_input: bool = True):
        super().__init__()
        self.fourier_levels = int(fourier_levels)
        self.include_input  = bool(include_input)
        # frequencies: pi, 2pi, 4pi, ..., 2^(L-1)*pi
        freqs = math.pi * (2.0 ** torch.arange(fourier_levels, dtype=torch.float32))
        self.register_buffer("freqs", freqs)          # (L,)

    @property
    def out_dim(self) -> int:
        base = 3 if self.include_input else 0
        return base + 3 * 2 * self.fourier_levels

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz : (B, 3, N)
        out : (B, out_dim, N)
        """
        # Scale each coordinate by each frequency: (B, 3, N, L)
        x_scaled = xyz.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1)
        sins = torch.sin(x_scaled).reshape(xyz.shape[0], 3 * self.fourier_levels, -1)
        coss = torch.cos(x_scaled).reshape(xyz.shape[0], 3 * self.fourier_levels, -1)
        parts: list[torch.Tensor] = [sins, coss]
        if self.include_input:
            parts.insert(0, xyz)
        return torch.cat(parts, dim=1)                # (B, out_dim, N)


# ── SIREN layer ──────────────────────────────────────────────────────────────

class FiLMSirenLayer(nn.Module):
    """
    FiLM-conditioned SIREN layer for conditional implicit neural representations.

    Separates spatial computation (SIREN on Fourier-encoded positions) from
    global conditioning (geometry/condition embedding injected as a learned
    additive shift via Feature-wise Linear Modulation).

    Why separation matters: when global embedding (576-dim) and Fourier features
    (63-dim) are concatenated, the SIREN first-layer init U[-1/639, 1/639] dilutes
    the spatial signal to 10% of the dot product.  FiLM keeps the SIREN input
    at the correct dimensionality (63) so init is calibrated for high-frequency
    position learning, while the global embedding conditions via a separately
    learned bias at every layer.

    Architecture: sin(omega * W*x + film_bias(g))
      where x  = Fourier-encoded position or hidden state (Conv1d, SIREN init)
            g  = global geometry+condition embedding (Linear → zero-init → additive shift)

    Reference: Sitzmann et al. SIREN (NeurIPS 2020); Perez et al. FiLM (AAAI 2018).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        omega_0: float = 30.0,
        is_first: bool = False,
    ):
        super().__init__()
        self.omega = float(omega_0)
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # FiLM additive shift from global condition embedding
        self.film_bias = nn.Linear(cond_dim, out_channels)

        # SIREN weight init (calibrated for in_channels, not the full concat size)
        if is_first:
            nn.init.uniform_(self.linear.weight, -1.0 / in_channels, 1.0 / in_channels)
        else:
            bound = math.sqrt(6.0 / in_channels) / omega_0
            nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)

        # FiLM bias init: zero so Phase 2 starts as pure SIREN, then adapts
        nn.init.zeros_(self.film_bias.weight)
        nn.init.zeros_(self.film_bias.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, in_channels, N)
        cond : (B, cond_dim)
        out  : (B, out_channels, N)
        """
        h    = self.omega * self.linear(x)              # (B, out_channels, N)
        bias = self.film_bias(cond).unsqueeze(-1)       # (B, out_channels, 1)
        return torch.sin(h + bias)                      # broadcast bias over N


# ── Edge feature construction ─────────────────────────────────────────────────

def build_edge_features(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Construct edge features for all points using their k nearest neighbours.

    For each point i and neighbour j, the edge feature is:
        [x_i,  x_j - x_i]   (concatenation of centre and relative offset)

    This lets the MLP reason about both absolute and relative geometry –
    essential for capturing local aerodynamic surface curvature.

    x   : (B, C, N)
    k   : neighbourhood size
    Returns: (B, 2C, N, k)
    """
    B, C, N = x.shape
    device = x.device

    idx = knn_graph(x, k)                  # (B, N, k)

    # Gather neighbour features
    idx_flat = idx.reshape(B, -1)          # (B, N*k)
    # Batch offset so we can index into the flattened feature tensor
    offset = torch.arange(B, device=device).unsqueeze(1) * N  # (B, 1)
    idx_flat = (idx_flat + offset).reshape(-1)                 # (B*N*k,)

    x_t = x.permute(0, 2, 1).reshape(B * N, C)                # (B*N, C)
    neighbours = x_t[idx_flat].reshape(B, N, k, C)            # (B, N, k, C)
    centre = x.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, k, C)

    # Edge feature: [centre, neighbour - centre]
    edge = torch.cat([centre, neighbours - centre], dim=-1)    # (B, N, k, 2C)
    return edge.permute(0, 3, 1, 2).contiguous()              # (B, 2C, N, k)


# ── EdgeConv block ────────────────────────────────────────────────────────────

class EdgeConv(nn.Module):
    """
    One Dynamic Graph CNN (EdgeConv) block.

    Rebuilds the k-NN graph on the *current* feature space at each forward
    pass so the effective neighbourhood adapts to learned representations
    (i.e. the graph is dynamic, not fixed to initial xyz distances).

    Architecture per block:
        edge_feat (2*C_in) → Conv2d(1×1) → BN → LeakyReLU → max-pool over k
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        norm_type: str = "batch",
        norm_groups: int = 8,
    ):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            make_norm_2d(out_channels, norm_type=norm_type, norm_groups=norm_groups),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, N)  →  out: (B, C_out, N)"""
        edge_feat = build_edge_features(x, self.k)  # (B, 2*C_in, N, k)
        out = self.conv(edge_feat)                   # (B, C_out,  N, k)
        return out.max(dim=-1)[0]                    # (B, C_out,  N)


# ── Full DGCNN surrogate ──────────────────────────────────────────────────────

class DGCNN(nn.Module):
    """
    Multi-task DGCNN surrogate for EDF/body aerodynamics.

    Two prediction heads share a common geometry encoder:

    1. Scalar head  – global max-pooled embedding → [drag, thrust]
       Trained with MSE loss (weight λ=1.0 by default).

    2. Field head   – broadcast global embedding to each midplane query
       point → [pressure, u, v, w] per point.
       Trained with MSE loss (weight λ=0.1 – auxiliary task).

    Both heads are conditioned on operating point [u_inf, rpm] via a small
    MLP branch whose output is concatenated to the global geometry embedding.
    """

    def __init__(
        self,
        k: int = 20,
        in_channels: int = 8,
        cond_dim: int = 2,
        scalar_dim: int = 2,
        field_dim: int = 4,
        edge_channels: tuple[int, ...] = (64, 64, 128),
        dropout: float = 0.3,
        norm_type: str = "group",
        norm_groups: int = 8,
        use_mean_pool: bool = True,
        fourier_levels: int = 6,
    ):
        super().__init__()
        c1, c2, c3 = edge_channels
        skip_dim = c1 + c2 + c3     # channels after skip-concat
        self.norm_type = str(norm_type).lower()
        self.norm_groups = int(norm_groups)
        self.use_mean_pool = bool(use_mean_pool)
        self.fourier_levels = int(fourier_levels)

        # ── Backbone: 3 EdgeConv blocks ──────────────────────────────────────
        self.ec1 = EdgeConv(in_channels, c1, k, norm_type=self.norm_type, norm_groups=self.norm_groups)
        self.ec2 = EdgeConv(c1, c2, k, norm_type=self.norm_type, norm_groups=self.norm_groups)
        self.ec3 = EdgeConv(c2, c3, k, norm_type=self.norm_type, norm_groups=self.norm_groups)

        # Point-wise aggregation after skip concat: skip_dim → skip_dim
        self.agg_conv = nn.Sequential(
            nn.Conv1d(skip_dim, skip_dim, kernel_size=1, bias=False),
            make_norm_1d(skip_dim, norm_type=self.norm_type, norm_groups=self.norm_groups),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Global pooling: max-pool always; optionally concat mean-pool.
        # Concatenating max+mean doubles the global feature richness while
        # preserving equivariance. Helps the scalar head distinguish
        # "what is the strongest signal" (max) from "what is the average flow"
        # (mean) — both relevant for thrust/drag prediction.
        geo_dim = 2 * skip_dim if self.use_mean_pool else skip_dim

        # ── Condition branch ─────────────────────────────────────────────────
        # Encodes operating conditions (U∞, RPM) into a compact embedding
        # and appends it to the geometry global feature.
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 64, bias=False),
            make_norm_1d(64, norm_type=self.norm_type, norm_groups=self.norm_groups),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 64, bias=False),
            make_norm_1d(64, norm_type=self.norm_type, norm_groups=self.norm_groups),
            nn.LeakyReLU(0.2, inplace=True),
        )
        combined_dim = geo_dim + 64  # geometry embedding + condition embedding

        # ── Scalar head: [drag, thrust] ───────────────────────────────────────
        # global features + condition → MLP (256→128→64→scalar_dim) with dropout.
        self.scalar_head = nn.Sequential(
            nn.Linear(combined_dim, 256, bias=False),
            make_norm_1d(256, norm_type=self.norm_type, norm_groups=self.norm_groups),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=False),
            make_norm_1d(128, norm_type=self.norm_type, norm_groups=self.norm_groups),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64, bias=False),
            make_norm_1d(64, norm_type=self.norm_type, norm_groups=self.norm_groups),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, scalar_dim),
        )

        # ── Fourier positional encoder for midplane query points ─────────────
        # Breaks the spectral bias that traps the field head at mean prediction.
        # Without this, raw (x,y,z) coordinates cause the MLP to stall at the
        # trivial constant (mean-field) solution for hundreds of epochs.
        self.pos_encoder = FourierEncoder(
            fourier_levels=self.fourier_levels, include_input=True
        )
        pos_dim = self.pos_encoder.out_dim   # 3 + 3*2*L  (default: 39)

        # ── FiLM-conditioned SIREN field head: [p, u, v, w] at midplane query points ─
        # SIREN processes ONLY Fourier-encoded positions (63-dim, correctly init'd).
        # Global geometry+condition embedding injects learned additive shifts (FiLM)
        # at each layer so spatial resolution and global conditioning are decoupled.
        #
        # Why not concat: concatenating 576-dim global embedding with 63-dim Fourier
        # features makes the SIREN first-layer init U[-1/639, 1/639] dilute the spatial
        # signal to <10% of the dot product. FiLM keeps SIREN inputs at 63-dim
        # (correct calibration) while the global embedding conditions independently.
        self.fs1 = FiLMSirenLayer(pos_dim,  512, combined_dim, omega_0=30.0, is_first=True)
        self.fs2 = FiLMSirenLayer(512,      512, combined_dim, omega_0=30.0)
        self.fs3 = FiLMSirenLayer(512,      512, combined_dim, omega_0=30.0)
        self.fs4 = FiLMSirenLayer(512,      256, combined_dim, omega_0=30.0)
        self.fs5 = FiLMSirenLayer(256,      128, combined_dim, omega_0=30.0)
        self.field_out = nn.Conv1d(128, field_dim, kernel_size=1)

    def field_from_embedding(self, g: torch.Tensor, mid_xyz: torch.Tensor) -> torch.Tensor:
        """
        Run the FiLM-SIREN field head from a precomputed geometry embedding.

        Used by AutogradPhysicsLoss to compute PDE residuals via autograd
        through the field head without re-running the expensive DGCNN encoder.
        Since every layer is Conv1d(kernel_size=1) or elementwise, each point's
        output depends only on its own input coordinates – so autograd w.r.t.
        mid_xyz gives exact per-point spatial derivatives.

        g       : (B, combined_dim) – from encode(), typically detached
        mid_xyz : (B, 3, N)         – query coords; may have requires_grad=True
        Returns : (B, field_dim, N) – normalised [p, u, v, w]
        """
        pos_enc = self.pos_encoder(mid_xyz)   # (B, pos_dim, N)
        h = self.fs1(pos_enc, g)              # (B, 512, N)
        h = self.fs2(h, g)                    # (B, 512, N)
        h = self.fs3(h, g)                    # (B, 512, N)
        h = self.fs4(h, g)                    # (B, 256, N)
        h = self.fs5(h, g)                    # (B, 128, N)
        return self.field_out(h)              # (B, field_dim, N)

    def forward_with_embedding(
        self,
        geo_pts: torch.Tensor,
        mid_xyz: torch.Tensor,
        conditions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Like forward() but also returns the global geometry embedding g.

        Used in training when AutogradPhysicsLoss is active so the encoder
        only runs once per step (g is then detached and reused for the physics
        residual pass through field_from_embedding).

        Returns (scalar, fields, g) where g : (B, combined_dim).
        """
        g = self.encode(geo_pts, conditions)              # (B, combined_dim)
        scalar = self.scalar_head(g)                      # (B, 2)
        fields = self.field_from_embedding(g, mid_xyz)    # (B, field_dim, N_mid)
        return scalar, fields, g

    def reset_field_head(self) -> None:
        """
        Re-initialise the FiLM-SIREN field head from scratch.

        Resets SIREN weights to the uniform init calibrated for their input
        dimensionality, and resets FiLM biases to zero (so phase 2 starts
        as a pure SIREN, then adapts the conditioning).
        """
        for i, layer in enumerate([self.fs1, self.fs2, self.fs3, self.fs4, self.fs5]):
            in_c = layer.linear.in_channels
            if i == 0:  # first layer
                nn.init.uniform_(layer.linear.weight, -1.0 / in_c, 1.0 / in_c)
            else:
                bound = math.sqrt(6.0 / in_c) / layer.omega
                nn.init.uniform_(layer.linear.weight, -bound, bound)
            nn.init.zeros_(layer.linear.bias)
            nn.init.zeros_(layer.film_bias.weight)
            nn.init.zeros_(layer.film_bias.bias)
        # Final linear output layer
        nn.init.kaiming_normal_(self.field_out.weight, a=0.0, nonlinearity="linear")
        if self.field_out.bias is not None:
            nn.init.zeros_(self.field_out.bias)

    def encode(self, geo_pts: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Encode geometry + operating conditions into a global feature vector.

        geo_pts    : (B, in_channels, N_geo)
        conditions : (B, cond_dim)
        Returns    : (B, combined_dim) – the shared embedding used by both heads.
        """
        # EdgeConv backbone with skip connections
        x1 = self.ec1(geo_pts)                            # (B, c1, N)
        x2 = self.ec2(x1)                                 # (B, c2, N)
        x3 = self.ec3(x2)                                 # (B, c3, N)

        # Concatenate multi-scale features (skip connections)
        skip = torch.cat([x1, x2, x3], dim=1)            # (B, c1+c2+c3, N)
        feat = self.agg_conv(skip)                        # (B, skip_dim, N)

        # Global pooling: max-pool captures salient peaks (stagnation, suction);
        # mean-pool captures average flow state. Concatenating both gives the
        # scalar head richer information for predicting thrust and drag.
        g_max = feat.max(dim=-1)[0]                       # (B, skip_dim)
        if self.use_mean_pool:
            g_mean = feat.mean(dim=-1)                    # (B, skip_dim)
            g_geo = torch.cat([g_max, g_mean], dim=1)    # (B, 2*skip_dim)
        else:
            g_geo = g_max                                 # (B, skip_dim)

        # Condition embedding
        g_cond = self.cond_mlp(conditions)                # (B, 64)

        return torch.cat([g_geo, g_cond], dim=1)          # (B, combined_dim)

    def forward(
        self,
        geo_pts: torch.Tensor,
        mid_xyz: torch.Tensor,
        conditions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        geo_pts    : (B, in_channels, N_geo) – geometry surface points
        mid_xyz    : (B, 3, N_mid)           – midplane query coordinates
        conditions : (B, cond_dim)           – [u_inf_norm, rpm_norm]

        Returns
        -------
        scalar : (B, scalar_dim)         – [drag, thrust] normalised
        fields : (B, field_dim, N_mid)   – [p, u, v, w] normalised
        """
        N_mid = mid_xyz.shape[2]

        # Shared global embedding
        g = self.encode(geo_pts, conditions)              # (B, combined_dim)

        # ── Scalar head ───────────────────────────────────────────────────────
        scalar = self.scalar_head(g)                      # (B, 2)

        # ── FiLM-SIREN field head ─────────────────────────────────────────────
        # SIREN processes only Fourier-encoded positions (correct 63-dim init).
        # Global embedding g is passed as FiLM condition at each layer (additive shift).
        pos_enc = self.pos_encoder(mid_xyz)               # (B, pos_dim, N_mid)
        h = self.fs1(pos_enc, g)                          # (B, 512, N_mid)
        h = self.fs2(h, g)                                # (B, 512, N_mid)
        h = self.fs3(h, g)                                # (B, 512, N_mid)
        h = self.fs4(h, g)                                # (B, 256, N_mid)
        h = self.fs5(h, g)                                # (B, 128, N_mid)
        fields = self.field_out(h)                        # (B, field_dim, N_mid)

        return scalar, fields


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DGCNN surrogate – sanity check")
    print("=" * 60)

    # Test with in_channels=3 to match the requested (2, 3, 4096) tensor shape
    model = DGCNN(k=10, in_channels=3, cond_dim=2, scalar_dim=2, field_dim=4,
                  norm_type="group", use_mean_pool=True)
    model.eval()

    geo   = torch.randn(2, 3, 4096)   # (batch, channels, points) as requested
    mid   = torch.randn(2, 3, 512)    # midplane query points
    cond  = torch.randn(2, 2)         # [u_inf, rpm]

    with torch.no_grad():
        scalar_out, field_out = model(geo, mid, cond)

    print(f"\nInput geometry:     {tuple(geo.shape)}")
    print(f"Input midplane:     {tuple(mid.shape)}")
    print(f"Input conditions:   {tuple(cond.shape)}")
    print(f"\nScalar output:      {tuple(scalar_out.shape)}  <- expected (2, 2) [drag, thrust]")
    print(f"Field  output:      {tuple(field_out.shape)}  <- expected (2, 4, 512) [p,u,v,w]")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:   {n_params:,}")

    # Full 8-channel model as used in training (with max+mean pool + Fourier L=10 + SIREN)
    print("\n--- Full 8-channel model with max+mean pool + Fourier L=10 + SIREN field head ---")
    model_full = DGCNN(k=20, in_channels=8, norm_type="group", use_mean_pool=True, fourier_levels=10)
    n_full = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    pos_dim = model_full.pos_encoder.out_dim
    print(f"Fourier out_dim:    {pos_dim}  (3 raw + 3x2x10 sinusoidal)")
    print(f"Total parameters:   {n_full:,}")

    # Without Fourier for comparison
    print("\n--- Same model without Fourier encoding (L=0) ---")
    model_nof = DGCNN(k=20, in_channels=8, norm_type="group", use_mean_pool=True, fourier_levels=0)
    n_nof = sum(p.numel() for p in model_nof.parameters() if p.requires_grad)
    print(f"Total parameters:   {n_nof:,}")
    print("=" * 60)
