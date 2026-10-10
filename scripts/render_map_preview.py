#!/usr/bin/env python3
"""Render map.pcd and poses.tum as a headless PNG on the board.

Source deploy/rk3588/env.sh, then run this script with the SLAM output directory.
The original map and poses are read only. The PNG shows model coordinates.
"""

import argparse
import json
from pathlib import Path
import time


def render_preview(directory, output, max_points=50000):
    if not 1 <= max_points <= 50000:
        raise ValueError("max_points must be between 1 and 50000")
    if output.suffix.lower() != ".png":
        raise ValueError("Preview output must have a .png extension")
    if output.exists():
        raise FileExistsError(f"Preview already exists: {output}")
    map_path, poses_path = directory / "map.pcd", directory / "poses.tum"
    for path in (map_path, poses_path):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing or empty SLAM output: {path}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Preview output directory does not exist: {output.parent}")

    # Select Agg before importing pyplot or any third-party visualization code.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    import open3d as o3d

    started = time.monotonic()
    poses = np.loadtxt(poses_path, ndmin=2)
    if (poses.ndim != 2 or poses.shape[1] != 8 or len(poses) < 2
            or not np.isfinite(poses).all() or not np.all(np.diff(poses[:, 0]) > 0)
            or np.any(np.linalg.norm(poses[:, 4:8], axis=1) < 1e-10)):
        raise ValueError("poses.tum must contain ordered, unique, finite TUM poses")
    positions = poses[:, 1:4]
    cloud = o3d.io.read_point_cloud(str(map_path))
    points = np.asarray(cloud.points)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("map.pcd must contain nonempty finite XYZ points")
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    if colors is not None and not np.isfinite(colors).all():
        raise ValueError("map.pcd contains non-finite colors")

    # A deterministic stride sample avoids allocating a permutation of every
    # map point. Only the selected points are handed to Matplotlib.
    count = min(len(points), max_points)
    indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    sampled = points[indices]
    sampled_colors = np.clip(colors[indices], 0, 1) if colors is not None else "#7698b3"
    map_min, map_max = points.min(axis=0), points.max(axis=0)
    minimum = np.minimum(map_min, positions.min(axis=0))
    maximum = np.maximum(map_max, positions.max(axis=0))
    span = maximum - minimum
    if not np.isfinite(span).all():
        raise ValueError("Map/trajectory coordinate range is too large to plot")
    padding = np.maximum(span * 0.05, max(float(span.max()), 1e-6) * 0.01)
    limits = [(float(low - pad), float(high + pad))
              for low, high, pad in zip(minimum, maximum, padding)]

    figure = plt.figure(figsize=(13, 10), constrained_layout=True)
    try:
        figure.suptitle(f"{directory.name}: {len(points):,} map points, {count:,} shown; "
                       f"{len(poses)} trajectory poses\nModel coordinates")
        axis = figure.add_subplot(2, 2, 1, projection="3d")
        axis.scatter(*sampled.T, c=sampled_colors, s=1, alpha=0.55,
                     linewidths=0, depthshade=False)
        axis.plot(*positions.T, color="orangered", linewidth=1.5, label="Trajectory")
        axis.scatter(*positions[0], c="limegreen", s=45, edgecolors="black", label="Start")
        axis.scatter(*positions[-1], c="crimson", s=45, edgecolors="black", label="End")
        for name, setter, limit_setter, limit in zip(
                "XYZ", (axis.set_xlabel, axis.set_ylabel, axis.set_zlabel),
                (axis.set_xlim, axis.set_ylim, axis.set_zlim), limits):
            setter(name)
            limit_setter(*limit)
        axis.set_box_aspect([high - low for low, high in limits])
        axis.view_init(elev=25, azim=45)
        axis.set_title("3D map and trajectory")
        axis.legend(fontsize=8, loc="upper right")

        for slot, (horizontal, vertical) in enumerate(((0, 1), (0, 2), (1, 2)), start=2):
            axis = figure.add_subplot(2, 2, slot)
            axis.scatter(sampled[:, horizontal], sampled[:, vertical], c=sampled_colors,
                         s=1, alpha=0.45, linewidths=0)
            axis.plot(positions[:, horizontal], positions[:, vertical],
                      color="orangered", linewidth=1.5, zorder=3)
            axis.scatter(positions[0, horizontal], positions[0, vertical],
                         c="limegreen", s=35, edgecolors="black", zorder=4)
            axis.scatter(positions[-1, horizontal], positions[-1, vertical],
                         c="crimson", s=35, edgecolors="black", zorder=4)
            axis.set_xlabel("XYZ"[horizontal])
            axis.set_ylabel("XYZ"[vertical])
            axis.set_xlim(*limits[horizontal])
            axis.set_ylim(*limits[vertical])
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
            axis.set_title(f"{'XYZ'[horizontal]}{'XYZ'[vertical]} projection")
        # Exclusive creation prevents overwriting any existing image or data.
        with output.open("xb") as handle:
            figure.savefig(handle, format="png", dpi=150)
    finally:
        plt.close(figure)
    if not output.stat().st_size:
        raise RuntimeError("Rendering produced an empty PNG")
    return dict(output=str(output), source_map=str(map_path), source_poses=str(poses_path),
                total_map_points=len(points), preview_points=count, trajectory_poses=len(poses),
                point_colors=colors is not None, map_bounds_min=map_min.tolist(),
                map_bounds_max=map_max.tolist(), sampling="evenly spaced file indices",
                matplotlib_backend=str(matplotlib.get_backend()), seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="SLAM output directory containing map.pcd and poses.tum")
    parser.add_argument("--output", type=Path, help="New PNG path; defaults to DIRECTORY/map_preview.png")
    parser.add_argument("--max-points", type=int, default=50000, help="Maximum drawn map points, between 1 and 50000")
    args = parser.parse_args()
    if not 1 <= args.max_points <= 50000:
        parser.error("--max-points must be between 1 and 50000")
    output = args.output if args.output is not None else args.directory / "map_preview.png"
    result = render_preview(args.directory, output, args.max_points)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
