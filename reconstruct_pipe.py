#!/usr/bin/env python3

import os
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import trimesh
from datetime import datetime
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues


class VGGTProcessor:
    def __init__(self, device="cuda", seed=42):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.seed = seed
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Initialize model
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        self.model.eval()
        self.model = self.model.to(self.device)
        
        print(f"VGGT Processor initialized on {self.device}")

    def process_images(self, image_paths, output_dir=None, conf_thres_value=5.0):
        """
        Process a list of image paths with VGGT and return camera poses and point cloud.
        
        Args:
            image_paths: List of paths to images
            output_dir: Directory to save outputs (optional)
            conf_thres_value: Confidence threshold for depth filtering
            
        Returns:
            dict: Contains camera poses, point cloud data, and metadata
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Load and preprocess images
        vggt_resolution = 518
        img_load_resolution = 1024
        
        images, original_coords = load_and_preprocess_images_square(image_paths, img_load_resolution)
        images = images.to(self.device)
        original_coords = original_coords.to(self.device)
        
        print(f"Processing {len(images)} images")
        
        # Run VGGT inference
        extrinsic, intrinsic, depth_map, depth_conf = self._run_vggt(images, vggt_resolution)
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        
        # Generate point cloud with colors
        points_3d, points_rgb = self._generate_point_cloud(
            images, depth_map, depth_conf, extrinsic, intrinsic, 
            vggt_resolution, conf_thres_value
        )
        
        # Check for zero points and raise exception if found
        if len(points_3d) == 0:
            raise ValueError("Scene failed: Zero valid 3D points after unprojection and filtering")
        
        # Rescale camera matrices to original image resolution
        scaled_extrinsic, scaled_intrinsic = self._rescale_camera_matrices(
            extrinsic, intrinsic, original_coords.cpu().numpy(), vggt_resolution
        )
        
        # Prepare results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'image_paths': image_paths,
            'num_images': len(image_paths),
            'camera_poses': {
                'extrinsic': scaled_extrinsic.tolist(),
                'intrinsic': scaled_intrinsic.tolist()
            },
            'point_cloud': None,
            'point_cloud_path': None
        }
        
        # Save outputs if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save camera matrices
            camera_path = os.path.join(output_dir, "camera_matrices.npz")
            np.savez(camera_path, 
                    extrinsic=scaled_extrinsic, 
                    intrinsic=scaled_intrinsic,
                    image_names=[os.path.basename(p) for p in image_paths])
            
            # Save point cloud
            point_cloud_path = os.path.join(output_dir, "points.ply")
            trimesh.PointCloud(points_3d, colors=points_rgb).export(point_cloud_path)
            results['point_cloud_path'] = point_cloud_path
            
            # Save metadata
            metadata_path = os.path.join(output_dir, "processing_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to {output_dir}")
            print(f"Camera matrices saved to: {camera_path}")
            print(f"Point cloud saved to: {point_cloud_path}")
        
        # Add point cloud data to results
        results['point_cloud'] = {
            'points': points_3d.tolist(),
            'colors': points_rgb.tolist() if points_rgb is not None else None,
            'num_points': len(points_3d)
        }
        
        return results

    def _run_vggt(self, images, resolution=518):
        """Run VGGT model inference"""
        images_resized = F.interpolate(images, size=(resolution, resolution), 
                                     mode="bilinear", align_corners=False)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images_batch = images_resized[None]
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_batch)
            
            # Predict cameras and depth
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
        
        return (extrinsic.squeeze(0).cpu().numpy(),
                intrinsic.squeeze(0).cpu().numpy(),
                depth_map.squeeze(0).cpu().numpy(),
                depth_conf.squeeze(0).cpu().numpy())

    def _generate_point_cloud(self, images, depth_map, depth_conf, extrinsic, intrinsic,
                             vggt_resolution, conf_thres_value):
        """Generate point cloud with RGB colors"""
        # Unproject depth to 3D points
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        
        num_frames, height, width, _ = points_3d.shape
        
        # Get RGB colors for points
        points_rgb = F.interpolate(
            images, size=(vggt_resolution, vggt_resolution), 
            mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)
        
        # Flatten points and colors for point cloud export
        points_3d_flat = points_3d.reshape(-1, 3)
        points_rgb_flat = points_rgb.reshape(-1, 3)
        
        # Remove invalid points (NaN or infinite values) and corresponding colors
        valid_mask = np.isfinite(points_3d_flat).all(axis=1)
        points_3d_filtered = points_3d_flat[valid_mask]
        points_rgb_filtered = points_rgb_flat[valid_mask]
        
        return points_3d_filtered, points_rgb_filtered

    def _rescale_camera_matrices(self, extrinsic, intrinsic, original_coords, img_size):
        """Rescale camera matrices to original image coordinates"""
        scaled_intrinsic = intrinsic.copy()
        
        for i in range(len(extrinsic)):
            real_image_size = original_coords[i, -2:]
            resize_ratio = max(real_image_size) / img_size
            
            # Scale focal lengths and principal point
            scaled_intrinsic[i, :2, :] *= resize_ratio
            # Set principal point to image center
            scaled_intrinsic[i, 0, 2] = real_image_size[0] / 2
            scaled_intrinsic[i, 1, 2] = real_image_size[1] / 2
        
        return extrinsic, scaled_intrinsic



def process_jsonl_entry(processor, entry, base_data_path, base_output_path):
    """Process a single JSONL entry"""
    entry_id = entry['id']
    image_paths = entry['images']
    
    # Convert relative paths to absolute paths
    full_image_paths = []
    for img_path in image_paths:
        full_path = os.path.join(base_data_path, img_path)
        if os.path.exists(full_path):
            full_image_paths.append(full_path)
        else:
            print(f"Warning: Image not found: {full_path}")
    
    if not full_image_paths:
        print(f"No valid images found for entry {entry_id}")
        return {'entry_id': entry_id, 'status': 'failed', 'reason': 'No valid images found'}
    
    # Create output directory for this entry
    output_dir = os.path.join(base_output_path, entry_id)
    
    try:
        print(f"Processing entry {entry_id} with {len(full_image_paths)} images")
        results = processor.process_images(full_image_paths, output_dir)
        results['entry_id'] = entry_id
        results['original_entry'] = entry
        results['status'] = 'success'
        return results
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing entry {entry_id}: {error_msg}")
        # Mark as failed with specific reason
        return {
            'entry_id': entry_id,
            'status': 'failed',
            'reason': error_msg,
            'original_entry': entry
        }


def worker_thread(gpu_id, task_queue, result_queue, base_data_path, base_output_path, conf_thres_value, seed):
    """Worker thread function that processes entries on a specific GPU"""
    device = f"cuda:{gpu_id}"
    print(f"Worker thread started on {device}")

    try:
        processor = VGGTProcessor(device=device, seed=seed + gpu_id)
    except Exception as e:
        print(f"Failed to initialize processor on {device}: {e}")
        return

    while True:
        try:
            entry = task_queue.get(timeout=1)
            if entry is None:
                break

            print(f"[GPU {gpu_id}] Processing entry: {entry['id']}")
            result = process_jsonl_entry(processor, entry, base_data_path, base_output_path)
            result['gpu_id'] = gpu_id
            result_queue.put(result)
            task_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing entry: {e}")
            if 'entry' in locals():
                error_result = {
                    'entry_id': entry.get('id', 'unknown'),
                    'status': 'failed',
                    'reason': str(e),
                    'gpu_id': gpu_id,
                    'original_entry': entry
                }
                result_queue.put(error_result)
            task_queue.task_done()


def process_batch_multithreaded(jsonl_path, base_data_path, base_output_path,
                               gpu_ids, conf_thres_value, seed, max_entries=None):
    """Process JSONL entries using multiple threads on different GPUs"""

    if not gpu_ids:
        gpu_ids = [0]

    print(f"Starting multi-threaded processing on GPUs: {gpu_ids}")

    task_queue = queue.Queue()
    result_queue = queue.Queue()

    with open(jsonl_path, 'r') as f:
        entries_loaded = 0
        for line_num, line in enumerate(f, 1):
            if max_entries and entries_loaded >= max_entries:
                break

            try:
                entry = json.loads(line.strip())
                task_queue.put(entry)
                entries_loaded += 1
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_num}")

    total_entries = entries_loaded
    print(f"Loaded {total_entries} entries for processing")

    threads = []
    for gpu_id in gpu_ids:
        for _ in range(gpu_ids.count(gpu_id)):
            thread = threading.Thread(
                target=worker_thread,
                args=(gpu_id, task_queue, result_queue, base_data_path,
                     base_output_path, conf_thres_value, seed)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)

    print(f"Started {len(threads)} worker threads")

    processed_count = 0
    successful_count = 0
    failed_results = []

    # Initialize progress bar
    pbar = tqdm(total=total_entries, desc="Processing entries", unit="entry")

    while processed_count < total_entries:
        try:
            result = result_queue.get(timeout=30)
            processed_count += 1

            if result['status'] == 'success':
                successful_count += 1
                pbar.set_postfix({
                    'Success': successful_count,
                    'Failed': processed_count - successful_count,
                    f'GPU{result["gpu_id"]}': f'{result["point_cloud"]["num_points"]} pts'
                })
                pbar.write(f"[GPU {result['gpu_id']}] ✓ {result['entry_id']} - "
                          f"{result['point_cloud']['num_points']} points")
            else:
                failed_results.append(result)
                pbar.set_postfix({
                    'Success': successful_count,
                    'Failed': processed_count - successful_count,
                    f'GPU{result["gpu_id"]}': 'Failed'
                })
                pbar.write(f"[GPU {result['gpu_id']}] ✗ {result['entry_id']}: {result['reason']}")

            pbar.update(1)

        except queue.Empty:
            pbar.write("Warning: Timeout waiting for results")
            break

    pbar.close()

    for _ in gpu_ids:
        task_queue.put(None)

    for thread in threads:
        thread.join(timeout=5)

    return processed_count, successful_count, failed_results


def main():
    parser = argparse.ArgumentParser(description="VGGT Image Processing Pipeline")
    parser.add_argument("--input_dir", type=str, default="/data/Datasets/MindCube/data/other_all_image/among/bottle_118", 
                       help="Directory containing input images (for single directory mode)")
    parser.add_argument("--jsonl_path", type=str, default="/data/Datasets/MindCube/data/raw/MindCube.jsonl",
                       help="Path to JSONL file for batch processing")
    parser.add_argument("--output_dir", type=str, default="/data/Datasets/MindCube/data/vggt_2",
                       help="Directory to save outputs (auto-generated if not specified)")
    parser.add_argument("--scene_name", type=str, default=None,
                       help="Scene name for auto output directory generation")
    parser.add_argument("--mode", type=str, choices=['single', 'batch'], default='batch',
                       help="Processing mode: single directory or batch from JSONL")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--conf_thres_value", type=float, default=0.0,
                       help="Confidence threshold for depth filtering")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum number of JSONL entries to process (for testing)")
    parser.add_argument("--gpu_ids", type=str, default="6,7",
                       help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')")
    parser.add_argument("--num_threads", type=int, default=1,
                       help="Number of threads per GPU (default: 1)")
    parser.add_argument("--multithreaded", default=False, action="store_true",
                       help="Enable multi-threaded processing across multiple GPUs")

    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]

    # Validate GPU availability
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        for gpu_id in gpu_ids:
            if gpu_id >= available_gpus:
                print(f"Warning: GPU {gpu_id} not available. Available GPUs: 0-{available_gpus-1}")
                gpu_ids = [x for x in gpu_ids if x < available_gpus]
        if not gpu_ids:
            print("No valid GPUs found, falling back to GPU 0")
            gpu_ids = [0]
    else:
        print("CUDA not available, using CPU")
        gpu_ids = [0]

    print(f"Using GPUs: {gpu_ids} with {args.num_threads} threads per GPU")
    
    if args.mode == 'single':
        # Single directory processing mode
        if args.output_dir is None:
            scene_name = args.scene_name
            if scene_name is None:
                input_path = Path(args.input_dir)
                if input_path.name == "images":
                    scene_name = input_path.parent.name
                else:
                    scene_name = input_path.name
            
            base_output_dir = "/data/Datasets/MindCube/data/vggt_processed_preserved_all"
            args.output_dir = os.path.join(base_output_dir, scene_name)
            print(f"Auto-generated output directory: {args.output_dir}")
        
        # Get image paths
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
        
        if not image_paths:
            images_dir = os.path.join(args.input_dir, "images")
            if os.path.exists(images_dir):
                for ext in image_extensions:
                    image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
                    image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        
        if not image_paths:
            raise ValueError(f"No images found in {args.input_dir}")
        
        image_paths.sort()
        print(f"Found {len(image_paths)} images")
        
        # Initialize processor and run
        processor = VGGTProcessor(seed=args.seed)
        results = processor.process_images(
            image_paths, 
            output_dir=args.output_dir,
            conf_thres_value=args.conf_thres_value
        )
        
        print(f"Processing complete!")
        print(f"Processed {results['num_images']} images")
        print(f"Generated {results['point_cloud']['num_points']} 3D points")
    
    elif args.mode == 'batch':
        # Batch processing mode from JSONL
        if not os.path.exists(args.jsonl_path):
            raise ValueError(f"JSONL file not found: {args.jsonl_path}")

        # Set up paths
        base_data_path = "/data/Datasets/MindCube/data"
        if args.output_dir is None:
            base_output_path = "/data/Datasets/MindCube/data/vggt_processed"
        else:
            base_output_path = args.output_dir

        # Create base output directory if it doesn't exist
        os.makedirs(base_output_path, exist_ok=True)

        print(f"Processing JSONL file: {args.jsonl_path}")
        print(f"Base data path: {base_data_path}")
        print(f"Base output path: {base_output_path}")

        if args.multithreaded and (len(gpu_ids) > 1 or args.num_threads > 1):
            # Multi-threaded processing
            total_threads = len(gpu_ids) * args.num_threads
            print(f"Starting multi-threaded processing with {total_threads} threads across {len(gpu_ids)} GPUs")

            # Expand GPU list based on threads per GPU
            expanded_gpu_ids = []
            for gpu_id in gpu_ids:
                expanded_gpu_ids.extend([gpu_id] * args.num_threads)

            processed_count, successful_count, failed_results = process_batch_multithreaded(
                args.jsonl_path, base_data_path, base_output_path,
                expanded_gpu_ids, args.conf_thres_value, args.seed, args.max_entries
            )

        else:
            # Single-threaded processing (original behavior)
            if args.multithreaded:
                print("Multi-threading requested but only one GPU available, using single-threaded mode")

            # Use first GPU from the list
            device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
            processor = VGGTProcessor(device=device, seed=args.seed)

            # Process JSONL entries
            processed_count = 0
            successful_count = 0
            failed_results = []

            # Count total entries first for progress bar
            total_entries = 0
            with open(args.jsonl_path, 'r') as f:
                for line in f:
                    if args.max_entries and total_entries >= args.max_entries:
                        break
                    try:
                        json.loads(line.strip())
                        total_entries += 1
                    except json.JSONDecodeError:
                        continue

            # Initialize progress bar
            pbar = tqdm(total=total_entries, desc="Processing entries (single-threaded)", unit="entry")

            with open(args.jsonl_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if args.max_entries and processed_count >= args.max_entries:
                        pbar.write(f"Reached maximum entries limit: {args.max_entries}")
                        break

                    try:
                        entry = json.loads(line.strip())
                        processed_count += 1

                        pbar.set_description(f"Processing {entry['id']}")

                        result = process_jsonl_entry(
                            processor, entry, base_data_path, base_output_path
                        )

                        if result['status'] == 'success':
                            successful_count += 1
                            pbar.set_postfix({
                                'Success': successful_count,
                                'Failed': processed_count - successful_count,
                                'Points': f'{result["point_cloud"]["num_points"]}'
                            })
                            pbar.write(f"✓ Successfully processed {entry['id']} - "
                                      f"{result['point_cloud']['num_points']} 3D points")
                        else:
                            failed_results.append(result)
                            pbar.set_postfix({
                                'Success': successful_count,
                                'Failed': processed_count - successful_count,
                                'Status': 'Failed'
                            })
                            pbar.write(f"✗ Failed to process {entry['id']}: {result['reason']}")

                        pbar.update(1)

                    except json.JSONDecodeError:
                        pbar.write(f"Error: Invalid JSON on line {line_num}")
                    except Exception as e:
                        pbar.write(f"Error processing line {line_num}: {str(e)}")

            pbar.close()

        print(f"\n=== Batch Processing Complete ===")
        print(f"Total entries processed: {processed_count}")
        print(f"Successfully processed: {successful_count}")
        print(f"Failed: {processed_count - successful_count}")

        # Save failed entries report
        if failed_results:
            failed_report_path = os.path.join(base_output_path, "failed_entries_report.json")
            with open(failed_report_path, 'w') as f:
                json.dump(failed_results, f, indent=2)
            print(f"\nFailed entries report saved to: {failed_report_path}")
            print("\nFailure reasons:")
            for result in failed_results:
                print(f"  {result['entry_id']}: {result['reason']}")


if __name__ == "__main__":
    main()