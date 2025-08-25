#!/usr/bin/env python3
"""
Batch process camera understanding for all scenes in MindCube_tinybench.jsonl
and update their processing_metadata.json files with natural language descriptions.

This script processes each scene and adds a 'camera_in_language' field to the
processing_metadata.json file with human-readable camera motion descriptions.
"""

import os
import json
import math
import argparse
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np


# ---------- New camera motion logic ---------
SMALL_EPS = 1e-3  # 1 mm threshold for "no movement"

def describe_camera_motion(extr1: np.ndarray, extr2: np.ndarray) -> str:
    """
    Describe camera motion between two views using world-space camera centers.
    
    Args:
        extr1, extr2: 3x4 extrinsic matrices [R|t]
        
    Returns:
        Direction string describing the motion
    """
    # 1. Split each 3×4 into rotation R and translation t
    R1, t1 = extr1[:, 0:3], extr1[:, 3]        # shape (3,3) and (3,)
    R2, t2 = extr2[:, 0:3], extr2[:, 3]

    # 2. Recover camera centres in world coords  (Eq. C = –Rᵀ t)
    C1_world = -R1.T @ t1
    C2_world = -R2.T @ t2

    # 3. World-space displacement between the two views
    d_world = C2_world - C1_world           # vector from cam 1 → cam 2

    # 4. Express that displacement in the first-camera's local frame
    #    v_cam = R1 * v_world     (because R1 maps world → cam1)
    d_cam1 = R1 @ d_world

    # 5. Pick the horizontal components (+x right, +z forward)
    dx = d_cam1[0]
    dz = d_cam1[2]

    # 6. Handle "no movement" noise floor
    if np.linalg.norm([dx, dz]) < SMALL_EPS:         # e.g. 1 mm
        return "No significant movement"

    # 7. Convert to a compass angle (forward = 0°, right = +90°)
    angle = math.atan2(dx, dz) * 180.0 / math.pi    # –180° … +180°

    # 8. Map the angle into one of 8 octants (22.5° boundaries)
    if   -22.5  <= angle <  22.5:  direction = "forward"
    elif  22.5  <= angle <  67.5:  direction = "diagonally forward and right"
    elif  67.5  <= angle < 112.5:  direction = "right"
    elif 112.5  <= angle < 157.5:  direction = "diagonally back and right"
    elif angle >= 157.5  or angle < -157.5:  direction = "backward"
    elif -157.5 <= angle < -112.5: direction = "diagonally back and left"
    elif -112.5 <= angle <  -67.5: direction = "left"
    else:                           direction = "diagonally forward and left"

    return direction


# ---------- camera understanding logic ------------------------------------
def generate_camera_descriptions(metadata: Dict[str, Any], entry: Dict[str, Any] = None, is_translation_question: bool = False) -> List[Dict[str, str]]:
    """
    Generate natural language descriptions for camera motion between consecutive frames.
    
    Args:
        metadata: Processing metadata dictionary containing camera poses
        entry: Original tinybench entry 
        is_translation_question: Whether this is a translation question
        
    Returns:
        List of motion descriptions between consecutive camera pairs
    """
    extrinsics = metadata["camera_poses"]["extrinsic"]
    image_paths = metadata["image_paths"]
    
    if len(extrinsics) < 2:
        return []
    
    descriptions = []
    
    # If this is not a translation question, return empty motion descriptions
    if not is_translation_question:
        return []
    
    # For translation questions, we should have exactly 2 images
    if len(extrinsics) == 2:
        extr1 = np.asarray(extrinsics[0], dtype=float)
        extr2 = np.asarray(extrinsics[1], dtype=float)
        
        img1_name = Path(image_paths[0]).name
        img2_name = Path(image_paths[1]).name
        
        motion_desc = describe_camera_motion(extr1, extr2)
        
        descriptions.append({
            "from_image": img1_name,
            "to_image": img2_name,
            "motion_description": motion_desc,
            "camera_pair": "0 -> 1"
        })
    else:
        # For cases with more than 2 images, process consecutive pairs
        for i in range(len(extrinsics) - 1):
            extr1 = np.asarray(extrinsics[i], dtype=float)
            extr2 = np.asarray(extrinsics[i + 1], dtype=float)

            img1_name = Path(image_paths[i]).name
            img2_name = Path(image_paths[i + 1]).name
            
            motion_desc = describe_camera_motion(extr1, extr2)
            
            descriptions.append({
                "from_image": img1_name,
                "to_image": img2_name,
                "motion_description": motion_desc,
                "camera_pair": f"{i} -> {i+1}"
            })
    
    return descriptions


def process_scene_metadata(scene_id: str, base_output_path: str, entry: Dict[str, Any] = None, force_update: bool = False, is_translation_question: bool = False) -> bool:
    """
    Process a single scene's metadata and add camera understanding descriptions.
    
    Args:
        scene_id: Scene identifier
        base_output_path: Base path for processed data
        entry: Original tinybench entry (for ground truth comparison)
        
    Returns:
        True if successful, False otherwise
    """
    metadata_path = Path(base_output_path) / scene_id / "processing_metadata.json"
    
    if not metadata_path.exists():
        print(f"Warning: Metadata file not found for scene {scene_id}: {metadata_path}")
        return False
    
    try:
        # Load existing metadata
        with metadata_path.open('r') as f:
            metadata = json.load(f)
        
        # Check if camera understanding already exists (only skip if not forcing update)
        if "camera_in_language" in metadata and not force_update:
            print(f"Scene {scene_id}: Camera understanding already exists, skipping...")
            return True
        
        # Generate camera descriptions
        camera_descriptions = generate_camera_descriptions(metadata, entry, is_translation_question)
        
        # For translation questions, we need camera descriptions
        if is_translation_question and not camera_descriptions:
            print(f"Scene {scene_id}: Insufficient camera poses for motion analysis")
            return False
        
        # Add camera understanding to metadata
        if is_translation_question:
            summary = f"Camera motion analysis for {len(camera_descriptions)} consecutive frame pairs"
        else:
            summary = "Non-translation question - motion descriptions not applicable"
        
        metadata["camera_in_language"] = {
            "generated_timestamp": Path(__file__).stat().st_mtime,  # Script modification time
            "num_motions": len(camera_descriptions),
            "motion_descriptions": camera_descriptions,
            "summary": summary,
            "is_translation_question": is_translation_question
        }
        
        # Save updated metadata
        with metadata_path.open('w') as f:
            json.dump(metadata, f, indent=2)
        
        # print(f"✓ Scene {scene_id}: Added {len(camera_descriptions)} camera motion descriptions")
        return True
        
    except Exception as e:
        print(f"✗ Scene {scene_id}: Error processing metadata - {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch process camera understanding for MindCube scenes")
    parser.add_argument("--jsonl_path", type=str, 
                       default="/data/Datasets/MindCube/data/raw/MindCube_tinybench.jsonl",
                       help="Path to JSONL file containing scene information")
    parser.add_argument("--base_output_path", type=str,
                       default="/data/Datasets/MindCube/data/vggt_processed",
                       help="Base path for processed data")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum number of entries to process (for testing)")
    parser.add_argument("--force_update",default=True, action="store_true",
                       help="Force update even if camera_in_language already exists")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_path):
        raise ValueError(f"JSONL file not found: {args.jsonl_path}")
    
    print(f"Processing JSONL file: {args.jsonl_path}")
    print(f"Base output path: {args.base_output_path}")
    print(f"Force update: {args.force_update}")
    print("="*60)
    
    # Process JSONL entries
    processed_count = 0
    successful_count = 0
    skipped_count = 0
    translation_count = 0
    non_translation_count = 0
    
    with open(args.jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if args.max_entries and processed_count >= args.max_entries:
                print(f"Reached maximum entries limit: {args.max_entries}")
                break
            
            try:
                entry = json.loads(line.strip())
                
                # Check if this is a translation question
                question = entry.get('question', '')
                is_translation_question = question.startswith("Based on these two views showing the same scene: in which direction did I move from the first view to the second view")
                
                scene_id = entry['id']
                processed_count += 1
                
                if is_translation_question:
                    translation_count += 1
                else:
                    non_translation_count += 1
                
                # print(f"\n--- Processing entry {processed_count}: {scene_id} ---")
                
                # Check if we should skip (when camera_in_language exists and not forcing update)
                metadata_path = Path(args.base_output_path) / scene_id / "processing_metadata.json"
                if metadata_path.exists() and not args.force_update:
                    with metadata_path.open('r') as f:
                        existing_metadata = json.load(f)
                    if "camera_in_language" in existing_metadata:
                        print(f"Scene {scene_id}: Camera understanding already exists, skipping...")
                        skipped_count += 1
                        continue
                
                success = process_scene_metadata(scene_id, args.base_output_path, entry, args.force_update, is_translation_question)
                
                if success:
                    successful_count += 1
                else:
                    print(f"✗ Failed to process {scene_id}")
                    
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_num}")
            except Exception as e:
                print(f"Error processing line {line_num}: {str(e)}")
    
    print(f"\n=== Batch Camera Understanding Complete ===")
    print(f"Total entries processed: {processed_count}")
    print(f"Translation questions: {translation_count}")
    print(f"Non-translation questions: {non_translation_count}")
    print(f"Successfully processed: {successful_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Failed: {processed_count - successful_count - skipped_count}")


if __name__ == "__main__":
    main()