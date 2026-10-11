import os
import glob
import time
import argparse

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.submap import Submap

from vggt_slam.runtime import load_model, resolve_device

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_imgs", action="store_true", help="Show camera images in the viser frustums. By default only the frustums are shown (faster visualization)")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
parser.add_argument("--vis_flow", action="store_true", help="Visualize original keyframe-selection optical flow")
parser.add_argument("--headless", action="store_true", help="Disable local optical-flow windows")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, choices=[1], help="Number of overlapping frames; only 1 is supported")
parser.add_argument("--max_loops", type=int, default=0, choices=[0], help="Only 0 is supported; no model-based retrieval")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--checkpoint", default=None, help="Local VGGT-1B checkpoint")
parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
parser.add_argument("--image_size", type=int, default=518, help="Multiple of 14; changing this requires accuracy validation")
parser.add_argument("--offline", action="store_true", help="Never download model weights")
parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")



def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()
    if args.submap_size < 2 or args.threads < 1:
        parser.error("--submap_size must be >= 2 and --threads must be >= 1")

    # Check input before allocating the model and graph.
    if not os.path.isdir(args.image_folder):
        raise ValueError(f"Image folder does not exist: {args.image_folder}")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*"))
                   if os.path.isfile(f)
                   and os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
                   and "depth" not in os.path.basename(f).lower()]
    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, 1)
    if not image_names:
        raise ValueError(f"No input images found in {args.image_folder}")

    use_optical_flow_downsample = True
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        vis_voxel_size=args.vis_voxel_size,
        vis_imgs=args.vis_imgs,
        device=device,
        image_size=args.image_size,
        max_loops=args.max_loops,
        offline=args.offline,
    )

    print("Initializing and loading VGGT model...")



    model = load_model(
        checkpoint=args.checkpoint, device=device, image_size=args.image_size,
        prune_unused_heads=True, offline=args.offline,
    )

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    new_frames = 0
    count = 0
    image_count = 0
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            with keyframe_time:
                img = cv2.imread(image_name)
                if img is None:
                    raise ValueError(f"Could not decode input image: {image_name}")
                enough_disparity = solver.flow_tracker.compute_disparity(
                    img, args.min_disparity, args.vis_flow and not args.headless
                )
                if enough_disparity:
                    image_names_subset.append(image_name)
                    new_frames += 1
                    image_count += 1
        else:
            image_names_subset.append(image_name)
            new_frames += 1

        # Run submap processing if enough images are collected or if it's the last group of images.
        ready = len(image_names_subset) >= args.submap_size + args.overlapping_window_size
        if (ready or image_name == image_names[-1]) and new_frames > 0 and len(image_names_subset) >= 2:
            count += 1
            print(image_names_subset)
            t1 = time.time()
            predictions = solver.run_predictions(image_names_subset, model, max_loops=args.max_loops)
            print("Solver total time", time.time()-t1)
            print(count, "submaps processed")

            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            
            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]
            new_frames = 0

    if count == 0:
        raise RuntimeError("No submap completed; at least two accepted keyframes are required")
    total_time = time.time() - total_time_start
    average_fps = total_time / image_count
    print(image_count, "frames processed")
    print("Total time:", total_time)
    print(f"Total time for VGGT calls: {solver.vggt_timer.total_time:.4f}s")
    print("Average VGGT time per frame:", solver.vggt_timer.total_time / image_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
    print("Average backend time per frame:", backend_time.total_time / image_count)
    print("Average total time per frame:", total_time / image_count)
    print("Average FPS:", 1 / average_fps)
        
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())



    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)

        if not args.skip_dense_log:
            # Log the full point cloud as one file
            solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))


if __name__ == "__main__":
    main()
