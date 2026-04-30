from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils import plot_pointcloud


def load_item(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as npz:
        body = npz["dense_body_points"].astype(np.float32)
        edf = npz["dense_edf_points"].astype(np.float32)
        meta = json.loads(npz["meta"][0]) if "meta" in npz else {}
    return body, edf, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--max_body", type=int, default=30000)
    parser.add_argument("--max_edf", type=int, default=12000)
    args = parser.parse_args()

    body, edf, meta = load_item(Path(args.sample))
    case_name = Path(meta.get("case_dir", "sample")).name

    plot_pointcloud(
        body_pts=body,
        edf_pts=edf,
        title=f"Dense geometry point cloud: {case_name}",
        save_path=Path(args.save_path),
        max_body=args.max_body,
        max_edf=args.max_edf,
    )


if __name__ == "__main__":
    main()
