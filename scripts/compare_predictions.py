#!/usr/bin/env python3
"""Report numerical differences; these are not ground-truth SLAM accuracy metrics."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference, candidate, source_root=None):
    source_root = Path(source_root).resolve() if source_root is not None else None
    if source_root is not None and not source_root.is_dir():
        raise ValueError(f"Source root must be an existing directory: {source_root}")
    reference_files = sorted(reference.glob("predictions_*.npz"))
    candidate_files = sorted(candidate.glob("predictions_*.npz"))
    if not reference_files or [p.name for p in reference_files] != [p.name for p in candidate_files]:
        raise ValueError("Nonempty, matching sets of prediction files are required")
    rows = []
    path_identity_validation_performed = False
    for ref_path, test_path in zip(reference_files, candidate_files):
        with np.load(ref_path, allow_pickle=False) as ref, np.load(test_path, allow_pickle=False) as test:
            reference_names, candidate_names = ref["frame_names"], test["frame_names"]
            if not np.array_equal(reference_names, candidate_names):
                if (source_root is None or reference_names.shape != candidate_names.shape
                        or reference_names.ndim != 1):
                    raise ValueError(f"Input frame mismatch: {ref_path.name}")
                for index, (reference_name, candidate_name) in enumerate(zip(reference_names, candidate_names)):
                    paths = []
                    for name in (reference_name, candidate_name):
                        if not isinstance(name, str):
                            raise ValueError(f"Input frame mismatch: {ref_path.name}, frame {index}: names must be strings")
                        path = Path(name)
                        path = path if path.is_absolute() else source_root / path
                        if not path.is_file():
                            raise ValueError(f"Input frame mismatch: {ref_path.name}, frame {index}: source image missing or not a file: {path}")
                        paths.append(path)
                    try:
                        same = paths[0].samefile(paths[1])
                    except OSError as exc:
                        raise ValueError(f"Input frame mismatch: {ref_path.name}, frame {index}: cannot verify source file identity: {exc}") from exc
                    if not same:
                        raise ValueError(f"Input frame mismatch: {ref_path.name}, frame {index}: different source files: {paths[0]} and {paths[1]}")
                path_identity_validation_performed = True
            row = {"submap": ref_path.name}
            for key in ("depth", "depth_conf", "extrinsic", "intrinsic"):
                x, y = ref[key].astype(np.float64), test[key].astype(np.float64)
                if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError(f"Non-finite or mismatched {key}: {ref_path.name}")
                delta = np.abs(x-y)
                row[key] = {"mean_absolute_error": float(delta.mean()),
                            "max_absolute_error": float(delta.max()),
                            "relative_l2_error": float(np.linalg.norm(x-y) / max(np.linalg.norm(x), 1e-12))}
            depth = ref["depth"].astype(np.float64)
            good = depth > 1e-6
            if not good.any():
                raise ValueError("Reference has no positive depth values")
            row["depth"]["mean_absolute_relative_error"] = float(
                np.mean(np.abs(test["depth"][good]-depth[good]) / depth[good]))
            pose_ref = ref["extrinsic"].astype(np.float64)
            pose_test = test["extrinsic"].astype(np.float64)
            relative_rotation = pose_test[..., :3, :3] @ np.swapaxes(pose_ref[..., :3, :3], -2, -1)
            skew = np.stack((relative_rotation[..., 2, 1]-relative_rotation[..., 1, 2],
                             relative_rotation[..., 0, 2]-relative_rotation[..., 2, 0],
                             relative_rotation[..., 1, 0]-relative_rotation[..., 0, 1]), axis=-1)
            sine = np.linalg.norm(skew, axis=-1) / 2
            cosine = (np.trace(relative_rotation, axis1=-2, axis2=-1)-1) / 2
            angles = np.degrees(np.arctan2(sine, np.clip(cosine, -1, 1)))
            row["max_camera_rotation_difference_degrees"] = float(angles.max())
            row["max_camera_translation_difference_per_median_depth"] = float(
                np.linalg.norm(pose_test[..., :3, 3]-pose_ref[..., :3, 3], axis=-1).max() / np.median(depth[good]))
            rows.append(row)
    return {"comparison": "numerical_only_no_ground_truth", "submaps": rows,
            "source_root": str(source_root) if source_root is not None else None,
            "path_identity_validation_performed": path_identity_validation_performed,
            "accuracy_validated": False, "real_time_validated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path,
                        help="Resolve differing relative frame paths against this root and require samefile identity")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, source_root=args.source_root)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    args.output.write_text(encoded+"\n")
    print(encoded)
