import os
import time
import threading
import argparse
import queue

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.cameras import BACKENDS

from vggt_slam.runtime import load_model, resolve_device

# --- Thread Safety Primitives ---
solver_lock = threading.Lock()  # Ensures only one solver thread runs at a time
data_lock = threading.Lock()    # Protects shared SLAM state (solver)

parser = argparse.ArgumentParser(description="VGGT-SLAM RealSense live demo")
parser.add_argument("--keyframe_folder", type=str, default="keyframes", help="Folder to save captured keyframes")
parser.add_argument("--camera", type=str, default="realsense", choices=list(BACKENDS.keys()), help="Camera backend (default: realsense)")
parser.add_argument("--camera_device", "--camera-device", default=None, help="V4L2 device /dev/videoN or index; omitted selects a USB capture device")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--camera_timeout", "--camera-timeout", type=float, default=5.0, help="V4L2 open/read timeout in seconds")
parser.add_argument("--headless", action="store_true", help="Disable local camera and optical-flow windows")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being built, otherwise only show the final map")
parser.add_argument("--vis_imgs", action="store_true", help="Show camera images in the viser frustums. By default only the frustums are shown (faster visualization)")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, choices=[1], help="Number of overlapping frames; only 1 is supported")
parser.add_argument("--max_loops", type=int, default=1, choices=[0, 1], help="Maximum loop closures per submap; 0 disables them")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")
parser.add_argument("--checkpoint", default=None, help="Local VGGT-1B checkpoint")
parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
parser.add_argument("--image_size", type=int, default=518, help="Multiple of 14; changing this requires accuracy validation")
parser.add_argument("--offline", action="store_true", help="Never download model weights")
parser.add_argument("--salad_checkpoint", default=None, help="Local SALAD checkpoint when loops are enabled")
parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")

parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_keyframe(img: np.ndarray, folder: str, idx: int) -> str:
    filename = os.path.join(folder, f"frame_{idx:06d}.png")
    if not cv2.imwrite(filename, img):
        raise OSError(f"Failed to write keyframe: {filename}")
    return filename


def raise_worker_error(error_queue):
    """Move an inference failure to the main thread, preserving its cause."""
    if error_queue is None:
        return
    try:
        error = error_queue.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError(f"SLAM worker failed: {error}") from error


def append_pending_keyframe(pending, filename, limit):
    """Bound unsent keyframes; the committed overlap is kept separately."""
    pending.append(filename)
    if len(pending) > limit:
        del pending[:len(pending)-limit]


def restart_camera(camera, stop_event=None, retry_delay: float = 2.0, error_queue=None):
    """Stop and re-start the camera, retrying until a device is streaming again.

    Used to recover from a mid-session disconnect (e.g. a RealSense USB drop),
    where ``capture()`` raises. Honors ``stop_event`` so the user can still
    cancel while a camera is unplugged. Returns the working camera object, or
    None if aborted via stop_event.
    """
    try:
        camera.stop()
    except Exception:
        pass  # device may already be gone; ignore teardown errors

    attempt = 0
    while stop_event is None or not stop_event.is_set():
        raise_worker_error(error_queue)
        attempt += 1
        try:
            camera.start()
            print(f"[Camera] Reconnected after {attempt} attempt(s).")
            return camera
        except Exception as e:
            print(f"[Camera] Reconnect attempt {attempt} failed: {e}.")
            print(f"[Camera] Retrying in {retry_delay:.0f}s...")
            time.sleep(retry_delay)
    print("[Camera] Reconnection aborted (session cancelled).")
    return None


def threaded_process_submap(image_names_subset, solver, model, args, clip_model, clip_preprocess, error_queue):
    """Background thread: run VGGT inference + graph optimisation for one submap."""
    try:
        print(f"[SLAM] Processing submap ({len(image_names_subset)} frames)...")
        predictions = solver.run_predictions(
            image_names_subset, model, args.max_loops, clip_model, clip_preprocess
        )
        with data_lock:
            solver.add_points(predictions)
            solver.graph.optimize()
            if args.vis_map:
                if len(predictions.get("detected_loops", [])) > 0:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
        print("[SLAM] Submap done.")
    except BaseException as e:
        error_queue.put(e)
    finally:
        if solver_lock.locked():
            solver_lock.release()


def run_semantic_query_loop(args, solver, clip_model, clip_tokenizer, processor):
    """Interactive open-set semantic query loop, run after capture ends."""
    while True:
        query = input("\nEnter text query or q to quit: ").strip()
        if len(query) == 0:
            print("Empty query. Exiting.")
            return
        if query == "q":
            print("Exiting.")
            return

        text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
        overall_best_score, overall_best_submap_id, overall_best_frame_index = \
            solver.map.retrieve_best_semantic_frame(text_emb)

        found_submap = solver.map.get_submap(overall_best_submap_id)

        best_img = found_submap.get_frame_at_index(overall_best_frame_index)
        print("Score:", overall_best_score)
        with torch.no_grad():
            best_img = to_pil_image(best_img)
            inference_state = processor.set_image(best_img)
            output = processor.set_text_prompt(state=inference_state, prompt=query)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
            print("Scores:", scores.cpu().numpy())

        masked_img = utils.overlay_masks(best_img, masks)
        masked_img.show()

        for i in range(masks.shape[0]):
            mask = masks[i].cpu().numpy()
            obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(
                found_submap.get_points_in_mask(overall_best_frame_index, mask, solver.graph)
            )
            solver.viewer.visualize_obb(
                center=obb_center,
                extent=obb_extent,
                rotation=obb_rotation,
                color=(255, 0, 0),
                line_width=8.0,
            )


def main():
    args = parser.parse_args()
    if args.submap_size < 2 or args.threads < 1:
        parser.error("--submap_size must be >= 2 and --threads must be >= 1")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or not np.isfinite(args.camera_timeout) or args.camera_timeout <= 0:
        parser.error("Camera dimensions, fps and timeout must be positive")
    if args.camera == "realsense" and args.camera_device is not None:
        parser.error("--camera-device applies to the V4L2/UVC backend")

    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    print(f"Using device: {device}")

    # When --run_os is set, SAM3/decord loads its own libxcb which poisons
    # OpenCV's XCB state and makes cv2.waitKey() hang. Skip cv2 display in that
    # case and print periodic status to the console instead.
    use_display = not args.run_os and not args.headless

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        vis_voxel_size=args.vis_voxel_size,
        vis_imgs=args.vis_imgs,
        device=device,
        image_size=args.image_size,
        max_loops=args.max_loops,
        salad_checkpoint=args.salad_checkpoint,
        offline=args.offline,
    )

    print("Initializing and loading VGGT model...")

    if args.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        sam3_model = build_sam3_image_model()
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
        clip_model = clip_model.cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)
    else:
        clip_model, clip_preprocess = None, None
        clip_tokenizer, processor = None, None

    model = load_model(
        checkpoint=args.checkpoint, device=device, image_size=args.image_size,
        prune_unused_heads=True, offline=args.offline,
    )
    print("All models loaded. Starting SLAM loop.")

    # Register the viser object-query panel so the user can search for objects
    # live (and after capture) without using the terminal.
    if args.run_os:
        solver.viewer.add_object_query_gui(solver, clip_model, clip_tokenizer, processor, data_lock)

    # --- Camera setup ---
    camera = None
    frame_count = 0
    anchor = []
    pending = []
    target_size = args.submap_size + args.overlapping_window_size
    submap_count = 0
    last_status_frame = 0  # for console status throttle when display is off
    stop_event = threading.Event()
    error_queue = queue.Queue()
    worker_thread = None
    main_error = None

    try:
        camera_kwargs = dict(width=args.width, height=args.height, fps=args.fps)
        if args.camera in {"v4l2", "uvc", "opencv"}:
            camera_kwargs.update(device=args.camera_device, timeout=args.camera_timeout)
        camera = BACKENDS[args.camera](**camera_kwargs)
        print(f"Initializing {args.camera} camera...")
        os.makedirs(args.keyframe_folder, exist_ok=True)
        camera.start()

        # Warm up the camera — first few frames can be None.
        print("Waiting for first camera frame...")
        first_frame = None
        while first_frame is None:
            try:
                first_frame = camera.capture()
            except Exception as e:
                print(f"[Camera] Error during warm-up ({e}). Reconnecting...")
                camera = restart_camera(camera, stop_event, error_queue=error_queue)
        if use_display:
            cv2.imshow("VGGT-SLAM Live", first_frame)
            cv2.waitKey(1)
        print("Camera ready.")

        while True:
            if stop_event.is_set():
                break
            raise_worker_error(error_queue)

            try:
                img = camera.capture()
            except Exception as e:
                print(f"[Camera] Lost connection ({e}). Attempting to reconnect...")
                camera = restart_camera(camera, stop_event, error_queue=error_queue)
                if camera is None:  # session cancelled while reconnecting
                    break
                continue

            if img is None:
                continue

            frame_count += 1

            if solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow and not args.headless):
                frame_path = save_keyframe(img, args.keyframe_folder, frame_count)
                pending_limit = args.submap_size if anchor else target_size
                append_pending_keyframe(pending, frame_path, pending_limit)

            required_new_frames = args.submap_size if anchor else target_size
            if len(pending) >= required_new_frames:
                raise_worker_error(error_queue)
                if solver_lock.acquire(blocking=False):
                    # The worker can fail just before it releases its lock.
                    try:
                        raise_worker_error(error_queue)
                    except BaseException:
                        solver_lock.release()
                        raise
                    submap_count += 1
                    print(f"[Main] Launching submap {submap_count} (frame {frame_count})...")
                    image_names_subset = anchor + pending
                    worker_thread = threading.Thread(
                        target=threaded_process_submap,
                        args=(list(image_names_subset), solver, model, args, clip_model, clip_preprocess, error_queue),
                        daemon=True,
                    )
                    try:
                        worker_thread.start()
                    except BaseException:
                        solver_lock.release()
                        worker_thread = None
                        raise
                    anchor = image_names_subset[-args.overlapping_window_size:]
                    pending = []

            kf = len(anchor) + len(pending)
            slam_busy = solver_lock.locked()

            if use_display:
                display = img.copy()
                status = f"KFs: {kf}/{target_size}  Submaps: {submap_count}"
                if slam_busy:
                    status += "  [SLAM running]"
                cv2.putText(display, status, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("VGGT-SLAM Live", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if frame_count - last_status_frame >= 30:
                    busy_str = "  [SLAM running]" if slam_busy else ""
                    print(f"[Camera] frame={frame_count}  KFs={kf}/{target_size}"
                          f"  submaps={submap_count}{busy_str}")
                    last_status_frame = frame_count

    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
    except BaseException as e:
        main_error = e
        raise
    finally:
        stop_event.set()
        shutdown_error = None
        try:
            if camera is not None:
                camera.stop()
        except Exception as e:
            shutdown_error = e
        finally:
            if worker_thread is not None:
                worker_thread.join()
            if use_display:
                cv2.destroyAllWindows()
            if main_error is None:
                raise_worker_error(error_queue)
        if shutdown_error is not None:
            if main_error is None:
                raise RuntimeError(f"Camera shutdown failed: {shutdown_error}") from shutdown_error
            print(f"[Camera] Shutdown also failed: {shutdown_error}")

    if solver.map.get_num_submaps() == 0:
        raise RuntimeError("No submap completed; this session did not validate SLAM")

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.run_os:
        run_semantic_query_loop(args, solver, clip_model, clip_tokenizer, processor)

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)
        if not args.skip_dense_log:
            solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))


if __name__ == "__main__":
    main()
