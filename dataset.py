"""
dataset.py – PyTorch Dataset for the DGCNN CFD surrogate pipeline.

Reads the pre-built point-cloud npz files from dataset_pointcloud/ and pairs
each sample with its corresponding midplane pressure/velocity CSV from
fluent_exports/.  All midplane data is loaded and cached at init time to
avoid repeated CSV reads during training.

Normalisation strategy
─────────────────────
  geometry_points  – used as-is (already in ~[-1,1] from build step)
  conditions       – divided by dataset-max absolute value
  scalar_targets   – divided by dataset-max absolute value  ([drag, thrust])
  midplane xyz     – mapped to same frame as geometry using meta norm params
  midplane fields  – each channel divided by its dataset-max absolute value

Run standalone to inspect a sample:
  python dataset.py --data_root dataset_pointcloud --index 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from config import Config, cfg
from utils import fps_numpy


# ── Column name helpers (mirrors build_pointcloud_dataset.py) ─────────────────

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, candidates: list[str], name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find {name} column; tried {candidates}. Have: {list(df.columns)}")


# ── Midplane loader ───────────────────────────────────────────────────────────

def load_midplane(
    csv_path: Path,
    n_points: int,
    norm_center: np.ndarray,
    norm_scale: float,
    seed: int = 0,
    max_pts: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a Fluent midplane CSV and return spatially subsampled arrays.

    Parameters
    ----------
    n_points : target number of points (subsample down to this if max_pts=0)
    max_pts  : if > 0, keep up to max_pts points without further reduction.
               Used by the dataset to cache a larger pool for per-epoch resampling.

    Returns
    -------
    xyz    : (N, 3)  normalised to geometry coordinate frame
    fields : (N, 4)  [pressure, x_velocity, y_velocity, z_velocity] raw
    """
    df = _norm_cols(pd.read_csv(csv_path))

    x_col = _pick_col(df, ["x_coordinate", "x", "xcoord"], "x")
    y_col = _pick_col(df, ["y_coordinate", "y", "ycoord"], "y")
    z_col = _pick_col(df, ["z_coordinate", "z", "zcoord"], "z")
    p_col = _pick_col(df, ["pressure", "static_pressure", "p"], "pressure")
    u_col = _pick_col(df, ["x_velocity", "velocity[x]", "vx", "u"], "x-velocity")
    v_col = _pick_col(df, ["y_velocity", "velocity[y]", "vy", "v"], "y-velocity")
    w_col = _pick_col(df, ["z_velocity", "velocity[z]", "vz", "w"], "z-velocity")

    arr = df[[x_col, y_col, z_col, p_col, u_col, v_col, w_col]].to_numpy(np.float32)
    arr = arr[np.isfinite(arr).all(axis=1)]

    n = arr.shape[0]
    cap = max_pts if max_pts > 0 else n_points
    rng = np.random.default_rng(seed)
    if n > cap:
        idx = rng.choice(n, cap, replace=False)
        arr = arr[idx]

    # Normalise xyz into the same frame as the geometry point cloud
    raw_xyz = arr[:, :3]
    norm_xyz = (raw_xyz - norm_center) / norm_scale

    fields = arr[:, 3:]  # [p, u, v, w], raw physical units

    return norm_xyz.astype(np.float32), fields.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class CFDDataset(Dataset):
    """
    Point-cloud surrogate dataset.

    Each sample contains:
      geometry_points  – (8, N_geo) surface point cloud with normal + label channels
      midplane_xyz     – (3, N_mid) normalised midplane coordinates
      midplane_fields  – (4, N_mid) normalised [p, u, v, w] at midplane
      scalar_targets   – (2,) normalised [drag, thrust]
      conditions       – (2,) normalised [u_inf, rpm]
      scalar_raw       – (2,) physical [drag (N), thrust (N)] – for error reporting
    """

    def __init__(
        self,
        data_root: str | Path = cfg.data_root,
        config: Optional[Config] = None,
        seed: int = 42,
    ):
        c = config or cfg
        self.data_root = Path(data_root)
        self.project_root = self.data_root.parent
        self.n_mid = c.num_midplane_points
        self.n_wall_pts = int(getattr(c, "n_wall_pts", 64))
        self.seed = seed

        # Augmentation settings
        self.augment_jitter_std   = float(getattr(c, "augment_jitter_std", 0.0))
        self.augment_resample_mid = bool(getattr(c, "augment_resample_mid", False))
        self.augment_mid_max_pts  = int(getattr(c, "augment_mid_max_pts", 0))
        # When resampling, cache up to max_pts per sample so each epoch sees
        # a different n_mid-point subset of the full ~88K CFD domain.
        _cache_pts = (self.augment_mid_max_pts
                      if (self.augment_resample_mid and self.augment_mid_max_pts > 0)
                      else 0)

        # Discover sample files
        self.files = sorted(self.data_root.glob("sample_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No sample_*.npz files found in {self.data_root}")

        print(f"[dataset] Found {len(self.files)} samples. Loading & caching …")

        # First pass: load npz metadata and midplane CSVs into RAM
        self._geo: list[np.ndarray] = []        # (N_geo, 8)
        self._mid_xyz: list[np.ndarray] = []    # (N_mid, 3) normalised
        self._mid_fields: list[np.ndarray] = [] # (N_mid, 4) raw
        self._scalars: list[np.ndarray] = []    # (2,) raw [drag, thrust]
        self._conds: list[np.ndarray] = []      # (2,) raw [u_inf, rpm]
        # Wall points from dense_body_points — in geom-norm coordinates (step-1
        # normalised, same frame as load_midplane output).  Step-2 normalisation
        # (mid_xyz_center / mid_xyz_scale) is applied after the second pass.
        self._wall_pts_geom: list[np.ndarray] = []   # (M, 3) per sample

        for i, fpath in enumerate(self.files):
            with np.load(fpath, allow_pickle=True) as npz:
                pts = npz["points"].astype(np.float32)             # (N, 8)
                scalar_raw = npz["targets_scalar_raw"].astype(np.float32)  # [drag, thrust]
                cond = npz["conditions"].astype(np.float32)        # [u_inf, rpm]
                meta = json.loads(npz["meta"][0])

                # Load wall points (dense_body_points are in the geom-norm
                # coordinate frame — the same frame as load_midplane output).
                wall_geom = np.zeros((0, 3), dtype=np.float32)
                for key in ("dense_body_points", "dense_edf_points"):
                    if key not in npz:
                        continue
                    pts3d = npz[key].astype(np.float32)  # (N, 3) geom-norm
                    # Midplane slice: |y| < 0.025 (geom-norm units)
                    mid_mask = np.abs(pts3d[:, 1]) < 0.025
                    pts_mid = pts3d[mid_mask]            # (M, 3)
                    if len(pts_mid) > 0:
                        wall_geom = np.concatenate([wall_geom, pts_mid], axis=0)
                self._wall_pts_geom.append(wall_geom)

            self._geo.append(pts)
            self._scalars.append(scalar_raw)
            self._conds.append(cond)

            # Midplane
            case_dir = self.project_root / Path(meta["case_dir"].replace("\\", "/"))
            mid_files = sorted(case_dir.glob("midplane_z0_*.csv"))

            if mid_files:
                norm_center = np.array(meta["norm_center"], dtype=np.float32)
                norm_scale  = float(meta["norm_scale"])
                xyz, fields = load_midplane(
                    mid_files[0], self.n_mid, norm_center, norm_scale,
                    seed=seed + i, max_pts=_cache_pts
                )
            else:
                print(f"  [warn] No midplane CSV for {case_dir.name} – using zeros")
                xyz    = np.zeros((self.n_mid, 3), dtype=np.float32)
                fields = np.zeros((self.n_mid, 4), dtype=np.float32)

            self._mid_xyz.append(xyz)
            self._mid_fields.append(fields)

        # Second pass: compute normalisation scales across dataset
        self.scalar_scale = np.maximum(
            np.abs(np.stack(self._scalars, axis=0)).max(axis=0), 1e-8
        ).astype(np.float32)  # (2,)

        self.cond_scale = np.maximum(
            np.abs(np.stack(self._conds, axis=0)).max(axis=0), 1e-8
        ).astype(np.float32)  # (2,)

        # Per-channel max over all points and all samples
        all_fields = np.concatenate(self._mid_fields, axis=0)  # (N_total, 4)
        self.field_scale = np.maximum(
            np.abs(all_fields).max(axis=0), 1e-8
        ).astype(np.float32)  # (4,) – one scale per [p, u, v, w]

        # Midplane xyz: the CFD domain extends far beyond the geometry surface
        # (~10x), so we normalise independently to keep field-head inputs in [-1,1].
        all_mid_xyz = np.concatenate(self._mid_xyz, axis=0)   # (N_total, 3)
        mid_mins = all_mid_xyz.min(axis=0)
        mid_maxs = all_mid_xyz.max(axis=0)
        self.mid_xyz_center = (0.5 * (mid_mins + mid_maxs)).astype(np.float32)
        self.mid_xyz_scale  = np.maximum(
            0.5 * (mid_maxs - mid_mins), 1e-8
        ).astype(np.float32)   # half-range per axis → normalised values in [-1, 1]

        # Apply step-2 normalisation to wall points now that mid_xyz_center/scale
        # are known.  Each sample gets its own (M, 3) array; __getitem__ subsamples.
        self._wall_pts_norm: list[np.ndarray] = []
        for wall_geom in self._wall_pts_geom:
            if len(wall_geom) > 0:
                wall_n = (wall_geom - self.mid_xyz_center) / self.mid_xyz_scale
                self._wall_pts_norm.append(wall_n.astype(np.float32))
            else:
                self._wall_pts_norm.append(np.zeros((0, 3), dtype=np.float32))

        n_wall_example = len(self._wall_pts_norm[0]) if self._wall_pts_norm else 0
        print(f"[dataset] scalar_scale (drag, thrust): {self.scalar_scale}")
        print(f"[dataset] cond_scale   (u_inf, rpm):   {self.cond_scale}")
        print(f"[dataset] field_scale  (p, u, v, w):   {self.field_scale}")
        print(f"[dataset] mid_xyz_center:               {self.mid_xyz_center}")
        print(f"[dataset] mid_xyz_scale:                {self.mid_xyz_scale}")
        print(f"[dataset] wall_pts per sample:          {n_wall_example}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        geo    = self._geo[idx].copy()     # (N_geo, 8) - copy so augmentation is not in-place
        mid_xy = self._mid_xyz[idx]        # (N_pool, 3)
        fields = self._mid_fields[idx]     # (N_pool, 4)
        scalar = self._scalars[idx]        # (2,) [drag, thrust]
        cond   = self._conds[idx]          # (2,) [u_inf, rpm]

        # ── Augmentation ───────────────────────────────────────────────────────
        # 1. Per-epoch midplane re-sampling: draw a fresh n_mid-point subset from
        #    the cached pool so the SIREN must generalise over the whole domain.
        if self.augment_resample_mid and len(mid_xy) > self.n_mid:
            ridx = np.random.choice(len(mid_xy), self.n_mid, replace=False)
            mid_xy = mid_xy[ridx]
            fields = fields[ridx]
        else:
            mid_xy = mid_xy[:self.n_mid]
            fields = fields[:self.n_mid]

        # 2. Geometry xyz jitter: small Gaussian noise on surface point coords.
        #    Decorrelates the 20 near-identical FPS samples each epoch.
        #    Normals (cols 3-5) and flags (cols 6-7) are left unchanged.
        if self.augment_jitter_std > 0.0:
            noise = np.random.randn(geo.shape[0], 3).astype(np.float32)
            geo[:, :3] += noise * self.augment_jitter_std

        # Normalise midplane xyz to [-1, 1] using domain-wide stats
        mid_xy_n = (mid_xy - self.mid_xyz_center) / self.mid_xyz_scale           # (n_mid, 3)

        # Transpose to (C, N) format expected by DGCNN
        geo_t    = torch.from_numpy(geo.T).float()                               # (8, N_geo)
        mid_xyz  = torch.from_numpy(mid_xy_n.T).float()                          # (3, n_mid)
        mid_flds = torch.from_numpy((fields / self.field_scale).T).float()       # (4, n_mid)
        scalar_n = torch.from_numpy(scalar / self.scalar_scale).float()          # (2,)
        scalar_r = torch.from_numpy(scalar).float()                              # (2,) raw
        cond_n   = torch.from_numpy(cond / self.cond_scale).float()              # (2,)

        # Wall points — subsample to a fixed count so batches collate cleanly.
        wall_all = self._wall_pts_norm[idx]  # (M, 3)
        n_wall = self.n_wall_pts
        if len(wall_all) >= n_wall:
            rng = np.random.default_rng(self.seed + idx)
            widx = rng.choice(len(wall_all), n_wall, replace=False)
            wall_sample = wall_all[widx]
        elif len(wall_all) > 0:
            rng = np.random.default_rng(self.seed + idx)
            widx = rng.choice(len(wall_all), n_wall, replace=True)
            wall_sample = wall_all[widx]
        else:
            wall_sample = np.zeros((n_wall, 3), dtype=np.float32)

        wall_pts_t = torch.from_numpy(wall_sample.T).float()  # (3, n_wall)

        return {
            "geometry_points": geo_t,
            "midplane_xyz":    mid_xyz,
            "midplane_fields": mid_flds,
            "scalar_targets":  scalar_n,
            "scalar_raw":      scalar_r,
            "conditions":      cond_n,
            "wall_pts":        wall_pts_t,
        }

    def norm_stats(self) -> dict:
        """Return normalisation statistics – embedded in checkpoint for inference."""
        return {
            "scalar_scale":    self.scalar_scale.tolist(),
            "cond_scale":      self.cond_scale.tolist(),
            "field_scale":     self.field_scale.tolist(),
            "mid_xyz_center":  self.mid_xyz_center.tolist(),
            "mid_xyz_scale":   self.mid_xyz_scale.tolist(),
        }


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_loaders(
    config: Config = cfg,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, CFDDataset]:
    """
    Create train/val DataLoaders from config paths.
    Returns (train_loader, val_loader, full_dataset).
    """
    ds = CFDDataset(data_root=config.data_root, config=config, seed=seed)

    n_total = len(ds)
    if n_total == 1 or config.overfit_mode or config.train_fraction >= 1.0:
        # Overfit path: train and evaluate on the same full dataset.
        train_ds = val_ds = ds
        shuffle = n_total > 1
    else:
        n_train = int(round(config.train_fraction * n_total))
        # Keep one held-out sample when not in overfit mode.
        n_train = max(1, min(n_total - 1, n_train))
        n_val = n_total - n_train
        train_ds, val_ds = random_split(
            ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        shuffle = True

    bs_train = min(config.batch_size, len(train_ds))
    bs_val   = min(config.batch_size, len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=bs_train,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,          # avoid single-sample batches that break BatchNorm
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs_val,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        persistent_workers=config.num_workers > 0,
    )

    print(f"[dataset] train={len(train_ds)}, val={len(val_ds)}, "
          f"train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    return train_loader, val_loader, ds


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset_pointcloud")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    ds = CFDDataset(data_root=args.data_root)
    item = ds[args.index]
    print(f"\nSample {args.index}:")
    for k, v in item.items():
        print(f"  {k:20s}: {tuple(v.shape)}, min={v.min():.4f}, max={v.max():.4f}")
