import json
import math
import numpy as np
import yaml
import sys
import os
from typing import Tuple, List, Dict, Union, Any
from pathlib import Path

# Get absolute paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


class CameraUnderstandingTool:
    def __init__(self, config_path='configs/main.yaml'):
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        # Handle relative config path from project root
        if not os.path.isabs(self.config_path):
            self.config_path = os.path.join(PROJECT_ROOT, self.config_path)
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    def split_extrinsic(self, ext: Union[list, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a 3×4 extrinsic list/array into (R, t).

        Args:
            ext: 3x4 extrinsic matrix as list or numpy array

        Returns:
            R: 3×3 rotation matrix
            t: translation column vector (3,)
        """
        ext = np.asarray(ext, dtype=float)
        R = ext[:, :3]
        t = ext[:, 3]
        return R, t

    def relative_pose(self, R1: np.ndarray, t1: np.ndarray,
                      R2: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate pose of camera 2 in camera 1's coordinate frame.

        Args:
            R1, t1: Rotation and translation of camera 1
            R2, t2: Rotation and translation of camera 2

        Returns:
            R_rel: Relative rotation matrix
            t_rel: Relative translation vector
        """
        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1
        return R_rel, t_rel

    def describe_camera_motion_from_extrinsics(self, extrinsics: List[List[float]], 
                                             image_paths: List[str] = None) -> List[str]:
        """
        Describe camera motion between consecutive views from extrinsic matrices.

        Args:
            extrinsics: List of 3x4 extrinsic matrices
            image_paths: Optional list of image paths for descriptive output

        Returns:
            List of motion descriptions between consecutive camera poses
        """
        if len(extrinsics) < 2:
            raise ValueError(f"Need at least 2 camera poses, got {len(extrinsics)}")

        descriptions = ""
        # we use a string to store the descrition because list will be used to store image paths
        
        for i in range(len(extrinsics) - 1):
            extr1 = np.asarray(extrinsics[i], dtype=float)
            extr2 = np.asarray(extrinsics[i + 1], dtype=float)
            
            motion_desc = describe_camera_motion(extr1, extr2)
            full_desc = f"From pose {i} to pose {i+1}: {motion_desc}"
            descriptions += full_desc

        return descriptions


    def analyze_camera_trajectory(self, extrinsics: List[List[float]]) -> Dict[str, float]:
        """
        Analyze overall camera trajectory statistics.

        Args:
            extrinsics: List of 3x4 extrinsic matrices

        Returns:
            Dictionary containing trajectory statistics
        """
        if len(extrinsics) < 2:
            return {'error': 'Need at least 2 poses for trajectory analysis'}

        total_distance = 0.0
        total_rotation = 0.0
        max_distance = 0.0
        min_distance = float('inf')

        for i in range(len(extrinsics) - 1):
            R1, t1 = self.split_extrinsic(extrinsics[i])
            R2, t2 = self.split_extrinsic(extrinsics[i + 1])
            R_rel, t_rel = self.relative_pose(R1, t1, R2, t2)

            # Distance calculation
            distance = np.linalg.norm(t_rel)
            total_distance += distance
            max_distance = max(max_distance, distance)
            min_distance = min(min_distance, distance)

            # Rotation calculation (approximate)
            yaw_rad = math.atan2(R_rel[0, 2], R_rel[2, 2])
            total_rotation += abs(math.degrees(yaw_rad))

        return {
            'total_distance': total_distance,
            'average_distance': total_distance / (len(extrinsics) - 1),
            'max_distance': max_distance,
            'min_distance': min_distance,
            'total_rotation': total_rotation,
            'average_rotation': total_rotation / (len(extrinsics) - 1),
            'num_segments': len(extrinsics) - 1
        }

def describe_camera_motion_from_extrinsics(extrinsics: List[List[float]], 
                                         image_paths: List[str] = None,
                                         config_path: str = 'configs/main.yaml') -> List[str]:
    """
    Simple function interface for describing camera motion from extrinsics.

    Args:
        extrinsics: List of 3x4 extrinsic matrices
        image_paths: Optional list of image paths
        config_path: Path to config file

    Returns:
        List of natural language motion descriptions
    """
    tool = CameraUnderstandingTool(config_path)
    return tool.describe_camera_motion_from_extrinsics(extrinsics, image_paths)


def analyze_camera_trajectory(extrinsics: List[List[float]],
                            config_path: str = 'configs/main.yaml') -> Dict[str, float]:
    """
    Simple function interface for analyzing camera trajectory.

    Args:
        extrinsics: List of 3x4 extrinsic matrices
        config_path: Path to config file

    Returns:
        Dictionary containing trajectory statistics
    """
    return describe_camera_motion_from_extrinsics(extrinsics, config_path)

