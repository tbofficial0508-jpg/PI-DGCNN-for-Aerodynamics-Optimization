from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils import plot_pointcloud


def load_npz(sample_path: Path):
    data = np.load(sample_path, allow_pickle=True)

    if "dense_body_points" not in data or "dense_edf_points" not in data:
        raise KeyError(
            "NPZ must contain 'dense_body_points' and 'dense_edf_points'. "
            "Rebuild the dataset with the geometry-quality builder."
        )

    body = data["dense_body_points"].astype(np.float32)
    edf = data["dense_edf_points"].astype(np.float32)

    meta = {}
    if "meta" in data:
        try:
            meta = json.loads(data["meta"][0])
        except Exception:
            meta = {}

    return body, edf, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--max_body_points", type=int, default=30000)
    parser.add_argument("--max_edf_points", type=int, default=12000)
    parser.add_argument("--body_size", type=float, default=0.55)
    parser.add_argument("--edf_size", type=float, default=0.80)
    parser.add_argument("--body_alpha", type=float, default=0.95)
    parser.add_argument("--edf_alpha", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    sample_path = Path(args.sample)
    save_path = Path(args.save_path)

    body, edf, meta = load_npz(sample_path)
    case_name = Path(meta.get("case_dir", sample_path.stem)).name

    plot_pointcloud(
        body_pts=body,
        edf_pts=edf,
        title=f"Geometry point-cloud visualization: {case_name}",
        save_path=save_path,
        max_body=args.max_body_points,
        max_edf=args.max_edf_points,
        body_size=args.body_size,
        edf_size=args.edf_size,
        body_alpha=args.body_alpha,
        edf_alpha=args.edf_alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
