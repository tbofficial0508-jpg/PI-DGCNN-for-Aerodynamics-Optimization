from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def find_one(case_dir: Path, pattern: str) -> Path:
    matches = sorted(case_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {case_dir}")
    if len(matches) > 1:
        print(f"[warn] Multiple matches for {pattern!r} in {case_dir}, using {matches[0].name}")
    return matches[0]


def find_one_optional(case_dir: Path, pattern: str) -> Path | None:
    """Like find_one but returns None instead of raising when no match."""
    matches = sorted(case_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[warn] Multiple matches for {pattern!r} in {case_dir}, using {matches[0].name}")
    return matches[0]


def read_report_last_value(path: Path) -> float:
    lines = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    numeric_line = None
    for line in reversed(lines):
        s = line.strip()
        if not s or s.startswith(("(", ")", ";", "#")):
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if nums:
            numeric_line = nums
            break
    if numeric_line is None:
        raise ValueError(f"Could not find numeric data in {path}")
    return float(numeric_line[-1])


def read_csv_auto(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=r"\s+")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def infer_xyz_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    def pick(cands, name):
        for c in cands:
            if c in df.columns:
                return c
        raise KeyError(f"Could not infer {name} column from {list(df.columns)}")
    return (
        pick(["x_coordinate", "x", "xcoord"], "x"),
        pick(["y_coordinate", "y", "ycoord"], "y"),
        pick(["z_coordinate", "z", "zcoord"], "z"),
    )


def infer_pressure_column(df: pd.DataFrame) -> str:
    for n in ["pressure", "static_pressure", "p"]:
        if n in df.columns:
            return n
    raise KeyError(f"Could not find pressure column in {list(df.columns)}")


def load_surface_xyzp(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = normalize_columns(read_csv_auto(path))
    xcol, ycol, zcol = infer_xyz_columns(df)
    pcol = infer_pressure_column(df)
    arr = df[[xcol, ycol, zcol, pcol]].to_numpy(np.float32)
    arr = arr[np.isfinite(arr).all(axis=1)]
    return arr[:, :3], arr[:, 3]


def filter_body_wall_points(
    body_pts: np.ndarray,
    body_p: np.ndarray,
    edf_pts: np.ndarray,
    max_dist_m: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Drop body-surface rows that lie farther than ``max_dist_m`` from the EDF
    centroid. Some Fluent exports (observed on J2) include far-field domain-wall
    nodes in the body-surface CSV, which inflates the normalisation bounding
    box and collapses the rendered point cloud to a thin line. The EDF nacelle
    is always tightly localised, so distance-from-EDF-centroid is a reliable
    discriminator between drone-body points (kept) and domain walls (dropped).
    """
    if body_pts.shape[0] == 0 or edf_pts.shape[0] == 0:
        return body_pts, body_p, 0
    edf_center = edf_pts.mean(axis=0)
    d = np.linalg.norm(body_pts - edf_center[None, :], axis=1)
    keep = d <= max_dist_m
    n_removed = int((~keep).sum())
    return body_pts[keep], body_p[keep], n_removed


def normalize_xyz(points: np.ndarray, bounds: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        bounds = np.stack([mins, maxs], axis=0)
    mins, maxs = bounds
    center = 0.5 * (mins + maxs)
    scale = np.max(maxs - mins)
    scale = 1.0 if scale < 1e-12 else scale
    pts = (points - center) / scale
    return pts.astype(np.float32), np.vstack([center, np.array([scale, scale, scale], dtype=np.float32)])


def farthest_point_sampling(points: np.ndarray, n_samples: int, seed: int = 1234) -> np.ndarray:
    n = points.shape[0]
    if n <= n_samples:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    xyz = points[:, :3].astype(np.float32)
    idx = np.empty(n_samples, dtype=np.int64)
    idx[0] = int(rng.integers(0, n))
    dist2 = np.sum((xyz - xyz[idx[0]]) ** 2, axis=1)
    for i in range(1, n_samples):
        idx[i] = int(np.argmax(dist2))
        d2 = np.sum((xyz - xyz[idx[i]]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2)
    return idx


def stratified_sample(body_pts: np.ndarray, edf_pts: np.ndarray, total_points: int, edf_fraction: float, seed: int):
    n_edf = max(1, int(round(total_points * edf_fraction)))
    n_body = max(1, total_points - n_edf)
    if body_pts.shape[0] < n_body:
        n_body = body_pts.shape[0]
        n_edf = min(edf_pts.shape[0], total_points - n_body)
    if edf_pts.shape[0] < n_edf:
        n_edf = edf_pts.shape[0]
        n_body = min(body_pts.shape[0], total_points - n_edf)

    body_idx = farthest_point_sampling(body_pts, n_body, seed=seed)
    edf_idx = farthest_point_sampling(edf_pts, n_edf, seed=seed + 1)
    return body_idx, edf_idx


def make_pressure_descriptor(points_xyz: np.ndarray, pressure: np.ndarray, n_bins: int = 96):
    x = points_xyz[:, 0]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax - xmin < 1e-12:
        xmax = xmin + 1e-12
    edges = np.linspace(xmin, xmax, n_bins + 1, dtype=np.float32)
    inds = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    mean_prof = np.zeros(n_bins, dtype=np.float32)
    std_prof = np.zeros(n_bins, dtype=np.float32)
    q10 = np.zeros(n_bins, dtype=np.float32)
    q90 = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        vals = pressure[inds == i]
        if vals.size:
            mean_prof[i] = np.mean(vals)
            std_prof[i] = np.std(vals)
            q10[i] = np.quantile(vals, 0.10)
            q90[i] = np.quantile(vals, 0.90)
    centers = 0.5 * (edges[:-1] + edges[1:])
    desc = np.stack([mean_prof, std_prof, q10, q90], axis=0)
    return desc.astype(np.float32), centers.astype(np.float32)


def parse_case(case_dir: Path, num_points: int, pressure_bins: int, edf_fraction: float, seed: int):
    # Force report names vary: "body_drag*.out" and "body_dragcoeff*.out" share
    # a prefix, so the plain glob greedily matches both. Use the -coeff files
    # (more specific) first and then pick the non-coeff force file by
    # excluding them explicitly.
    def _exclude_coeff(matches):
        return [p for p in matches if "coeff" not in p.name.lower()]

    body_drag_candidates = _exclude_coeff(sorted(case_dir.glob("body_drag*.out")))
    edf_thrust_candidates = _exclude_coeff(sorted(case_dir.glob("edf_thrust*.out")))
    if not body_drag_candidates:
        raise FileNotFoundError(f"No body_drag*.out (non-coeff) in {case_dir}")
    if not edf_thrust_candidates:
        raise FileNotFoundError(f"No edf_thrust*.out (non-coeff) in {case_dir}")
    body_drag_f  = body_drag_candidates[0]
    edf_thrust_f = edf_thrust_candidates[0]

    body_cd_f  = find_one_optional(case_dir, "body_dragcoeff*.out")
    edf_ct_f   = find_one_optional(case_dir, "edf_thrustcoeff*.out")
    body_surface_f = find_one(case_dir, "body_surface*.csv")
    edf_surface_f  = find_one(case_dir, "edf_surface*.csv")

    body_drag  = read_report_last_value(body_drag_f)
    edf_thrust = read_report_last_value(edf_thrust_f)
    # When coefficient reports are absent (some cases only export raw forces),
    # derive Cd, Ct analytically from u_inf (parsed later) and a fixed ref
    # area/density. Here we defer until u_inf is known; placeholder NaN.
    body_cd  = read_report_last_value(body_cd_f) if body_cd_f is not None else float("nan")
    edf_ct   = read_report_last_value(edf_ct_f)  if edf_ct_f  is not None else float("nan")

    body_pts_raw, body_p_raw = load_surface_xyzp(body_surface_f)
    edf_pts_raw, edf_p_raw = load_surface_xyzp(edf_surface_f)
    body_pts_raw, body_p_raw, n_wall_dropped = filter_body_wall_points(
        body_pts_raw, body_p_raw, edf_pts_raw, max_dist_m=1.5
    )
    if n_wall_dropped > 0:
        print(f"  [wall-filter] Dropped {n_wall_dropped} body pts >1.5 m from EDF "
              f"(kept {body_pts_raw.shape[0]}) in {case_dir.name}")
    all_pts_raw = np.concatenate([body_pts_raw, edf_pts_raw], axis=0)
    all_norm, norm_meta = normalize_xyz(all_pts_raw)
    body_norm = all_norm[: body_pts_raw.shape[0]]
    edf_norm = all_norm[body_pts_raw.shape[0] :]

    body_idx, edf_idx = stratified_sample(body_norm, edf_norm, num_points, edf_fraction, seed)
    body_pts = body_norm[body_idx]
    edf_pts = edf_norm[edf_idx]
    body_p = body_p_raw[body_idx]
    edf_p = edf_p_raw[edf_idx]

    # features: x,y,z, nx,ny,nz, is_body, is_edf
    def estimate_normals(pts: np.ndarray) -> np.ndarray:
        # Lightweight fallback: radial-like normal approximation from PCA-centered geometry.
        c = pts.mean(axis=0, keepdims=True)
        v = pts - c
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n = np.where(n < 1e-12, 1.0, n)
        return (v / n).astype(np.float32)

    body_n = estimate_normals(body_pts)
    edf_n = estimate_normals(edf_pts)
    body_feat = np.concatenate([body_pts, body_n, np.ones((body_pts.shape[0], 1), np.float32), np.zeros((body_pts.shape[0], 1), np.float32)], axis=1)
    edf_feat = np.concatenate([edf_pts, edf_n, np.zeros((edf_pts.shape[0], 1), np.float32), np.ones((edf_pts.shape[0], 1), np.float32)], axis=1)
    points = np.concatenate([body_feat, edf_feat], axis=0).astype(np.float32)

    # shuffle mixed points but keep deterministic
    rng = np.random.default_rng(seed)
    perm = rng.permutation(points.shape[0])
    points = points[perm]

    body_desc, body_x = make_pressure_descriptor(body_pts, body_p, pressure_bins)
    edf_desc, edf_x = make_pressure_descriptor(edf_pts, edf_p, pressure_bins)

    joined_names = " ".join([p.name for p in case_dir.iterdir()])
    m_speed = re.search(r"U(\d+(?:\.\d+)?)", joined_names, flags=re.IGNORECASE)
    m_rpm = re.search(r"RPM(\d+(?:\.\d+)?)", joined_names, flags=re.IGNORECASE)
    u_inf = float(m_speed.group(1)) if m_speed else np.nan
    rpm = float(m_rpm.group(1)) if m_rpm else np.nan

    # Fallback coefficient derivation when the Fluent coefficient reports
    # were not exported. Uses standard-air density and a reference area of
    # 1 m^2 (consistent with Fluent's default when no Aref is set).
    if not np.isfinite(body_cd) or not np.isfinite(edf_ct):
        rho_ref = 1.225
        A_ref = 1.0
        q_inf = 0.5 * rho_ref * (u_inf ** 2) * A_ref if np.isfinite(u_inf) and u_inf > 0 else np.nan
        if np.isfinite(q_inf) and q_inf > 0:
            if not np.isfinite(body_cd):
                body_cd = float(body_drag / q_inf)
            if not np.isfinite(edf_ct):
                edf_ct = float(edf_thrust / q_inf)
        else:
            body_cd = 0.0 if not np.isfinite(body_cd) else body_cd
            edf_ct  = 0.0 if not np.isfinite(edf_ct)  else edf_ct

    # Tag geometry family for downstream analysis (T1, J1, J2, ...).
    m_geom = re.search(r"Geometry([A-Za-z0-9]+?)U\d", joined_names, flags=re.IGNORECASE)
    geom_tag = m_geom.group(1).upper() if m_geom else "UNKNOWN"

    return {
        "points": points,
        "dense_body_points": body_norm.astype(np.float32),
        "dense_edf_points": edf_norm.astype(np.float32),
        "conditions": np.array([u_inf, rpm], dtype=np.float32),
        "targets_scalar": np.array([body_cd, edf_ct], dtype=np.float32),
        "targets_scalar_raw": np.array([body_drag, edf_thrust], dtype=np.float32),
        "body_pressure_desc": body_desc,
        "edf_pressure_desc": edf_desc,
        "body_pressure_x": body_x,
        "edf_pressure_x": edf_x,
        "meta": {
            "case_dir": str(case_dir),
            "u_inf": u_inf,
            "rpm": rpm,
            "body_drag": body_drag,
            "edf_thrust": edf_thrust,
            "body_cd": body_cd,
            "edf_ct": edf_ct,
            "norm_center": norm_meta[0].tolist(),
            "norm_scale": float(norm_meta[1, 0]),
            "n_body_dense": int(body_norm.shape[0]),
            "n_edf_dense": int(edf_norm.shape[0]),
            "geometry": geom_tag,
        },
    }


def save_sample(sample: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        points=sample["points"],
        dense_body_points=sample["dense_body_points"],
        dense_edf_points=sample["dense_edf_points"],
        conditions=sample["conditions"],
        targets_scalar=sample["targets_scalar"],
        targets_scalar_raw=sample["targets_scalar_raw"],
        body_pressure_desc=sample["body_pressure_desc"],
        edf_pressure_desc=sample["edf_pressure_desc"],
        body_pressure_x=sample["body_pressure_x"],
        edf_pressure_x=sample["edf_pressure_x"],
        meta=np.array([json.dumps(sample["meta"])], dtype=object),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--pressure_bins", type=int, default=96)
    parser.add_argument("--edf_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--min_abs_force_n",
        type=float,
        default=0.0,
        help="If >0, skip samples where BOTH |body_drag| and |edf_thrust| are below this threshold [N].",
    )
    parser.add_argument(
        "--skip_bad_cases",
        action="store_true",
        help="Skip malformed/unphysical cases instead of failing the whole build.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stale_files = sorted(output_root.glob("sample_*.npz"))
    if stale_files:
        for stale in stale_files:
            stale.unlink()
        print(f"Removed {len(stale_files)} stale sample_*.npz files from {output_root}")
    # Allow users to park bad/unconverged cases under input_root/excluded/.
    # We only parse direct child case folders, excluding that quarantine dir.
    case_dirs = sorted([
        p for p in input_root.iterdir()
        if p.is_dir() and p.name.lower() != "excluded"
    ])
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found in {input_root}")

    manifest = []
    skipped: list[tuple[str, str]] = []
    for i, cdir in enumerate(case_dirs):
        print(f"[{i+1}/{len(case_dirs)}] Parsing {cdir.name}")
        try:
            sample = parse_case(cdir, args.num_points, args.pressure_bins, args.edf_fraction, args.seed + 17 * i)
        except Exception as e:
            if args.skip_bad_cases:
                reason = f"parse_error: {e}"
                print(f"  [skip] {cdir.name}: {reason}")
                skipped.append((cdir.name, reason))
                continue
            raise

        if args.min_abs_force_n > 0.0:
            d = float(sample["meta"]["body_drag"])
            t = float(sample["meta"]["edf_thrust"])
            if abs(d) < args.min_abs_force_n and abs(t) < args.min_abs_force_n:
                reason = (f"near-zero forces (|drag|={abs(d):.3e} N, "
                          f"|thrust|={abs(t):.3e} N) < {args.min_abs_force_n:.3e} N")
                if args.skip_bad_cases:
                    print(f"  [skip] {cdir.name}: {reason}")
                    skipped.append((cdir.name, reason))
                    continue
                raise ValueError(f"{cdir.name}: {reason}")

        out_file = output_root / f"sample_{len(manifest):05d}.npz"
        save_sample(sample, out_file)
        manifest.append(sample["meta"])

    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest)} samples to {output_root}")
    if skipped:
        print(f"Skipped {len(skipped)} case(s):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
    if not manifest:
        raise RuntimeError("No valid samples were saved. Check Fluent exports / filter thresholds.")


if __name__ == "__main__":
    main()
