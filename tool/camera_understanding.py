import json
import math
import numpy as np
import yaml
import sys
import os
from typing import Tuple, List, Dict, Union
from pathlib import Path

# Get absolute paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CameraUnderstandingTool:
    def __init__(self, config_path='configs/main.yaml'):
        self.config_path = config_path
        self._load_config()
        
        # Compass directions for motion description
        self.COMPASS = [
            "FORWARD",          # 0° … 45°
            "FORWARD-RIGHT",    # 45° … 90°
            "RIGHT",            # 90° … 135°
            "BACK-RIGHT",       # 135° … 180°
            "BACKWARD",         # 180° … 225°
            "BACK-LEFT",        # 225° … 270°
            "LEFT",             # 270° … 315°
            "FORWARD-LEFT"      # 315° … 360°
        ]
    
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

    def bucket_direction(self, t_rel: np.ndarray,
                        mag_eps: float = 1e-3) -> Tuple[str, float, float]:
        """
        Map the horizontal component of t_rel into one of eight compass buckets.

        Args:
            t_rel: Relative translation vector
            mag_eps: Minimum magnitude threshold

        Returns:
            bucket_name: Direction bucket name
            horizontal_distance: Horizontal movement distance
            heading_deg: Heading angle in degrees
        """
        x, z = float(t_rel[0]), float(t_rel[2])  # camera 1 frame: +x right, +z forward
        horiz_dist = math.hypot(x, z)

        if horiz_dist < mag_eps:  # almost no motion
            return "STATIONARY", horiz_dist, 0.0

        heading_rad = math.atan2(x, z)  # atan2(x, z) → 0° = forward
        heading_deg = math.degrees(heading_rad) % 360
        bucket = self.COMPASS[int((heading_deg + 22.5) // 45) % 8]
        return bucket, horiz_dist, heading_deg

    def describe_motion(self, t_rel: np.ndarray,
                       R_rel: np.ndarray,
                       dist_round: int = 2) -> str:
        """
        Convert relative translation + rotation into natural language description.

        Args:
            t_rel: Relative translation vector
            R_rel: Relative rotation matrix
            dist_round: Number of decimal places for distances

        Returns:
            Natural language description of camera motion
        """
        bucket, horiz_dist, heading_deg = self.bucket_direction(t_rel)

        # vertical shift
        y_up = float(t_rel[1])
        updown = ""
        if abs(y_up) > 0.02:  # > 2 cm
            updown = f", {abs(y_up):.{dist_round}f} m {'up' if y_up > 0 else 'down'}"

        # approximate yaw from R_rel (proj onto x-z plane)
        yaw_rad = math.atan2(R_rel[0, 2], R_rel[2, 2])
        yaw_deg = math.degrees(yaw_rad)
        yaw_txt = ""
        if abs(yaw_deg) > 1.0:
            direction = "clockwise" if yaw_deg < 0 else "counter-clockwise"
            yaw_txt = f", with a {abs(yaw_deg):.0f}° {direction} yaw"

        if bucket == "STATIONARY":
            return f"The second view is essentially at the same horizontal position{updown}{yaw_txt}."

        bucket_readable = (bucket
                          .replace("FORWARD", "forward")
                          .replace("BACKWARD", "backward")
                          .replace("RIGHT", "right")
                          .replace("LEFT", "left")
                          .replace("-", "-and-"))

        return (f"The second view is roughly {horiz_dist:.{dist_round}f} m "
               f"{bucket_readable}{updown}{yaw_txt}.")

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

        descriptions = []
        
        for i in range(len(extrinsics) - 1):
            R1, t1 = self.split_extrinsic(extrinsics[i])
            R2, t2 = self.split_extrinsic(extrinsics[i + 1])
            R_rel, t_rel = self.relative_pose(R1, t1, R2, t2)
            
            motion_desc = self.describe_motion(t_rel, R_rel)
            
            if image_paths and len(image_paths) > i + 1:
                img1_name = Path(image_paths[i]).name
                img2_name = Path(image_paths[i + 1]).name
                full_desc = f"From '{img1_name}' to '{img2_name}': {motion_desc}"
            else:
                full_desc = f"From pose {i} to pose {i+1}: {motion_desc}"
                
            descriptions.append(full_desc)

        return descriptions

    def describe_camera_motion_from_file(self, metadata_path: str) -> Dict[str, Union[List[str], int, List[str]]]:
        """
        Describe camera motion from a processing metadata JSON file.

        Args:
            metadata_path: Path to processing_metadata.json file

        Returns:
            Dictionary containing motion descriptions and metadata
        """
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with metadata_path.open() as f:
            meta = json.load(f)

        if 'camera_poses' not in meta or 'extrinsic' not in meta['camera_poses']:
            raise KeyError("Metadata file must contain 'camera_poses.extrinsic' field")

        extrinsics = meta["camera_poses"]["extrinsic"]
        image_paths = meta.get('image_paths', [])

        descriptions = self.describe_camera_motion_from_extrinsics(extrinsics, image_paths)

        return {
            'motion_descriptions': descriptions,
            'num_poses': len(extrinsics),
            'image_paths': image_paths,
            'metadata_path': str(metadata_path)
        }

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

def describe_camera_motion(extrinsics: List[List[float]], 
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

def describe_camera_motion_from_file(metadata_path: str,
                                   config_path: str = 'configs/main.yaml') -> Dict[str, Union[List[str], int, List[str]]]:
    """
    Simple function interface for describing camera motion from metadata file.

    Args:
        metadata_path: Path to processing_metadata.json file
        config_path: Path to config file

    Returns:
        Dictionary containing motion descriptions and metadata
    """
    tool = CameraUnderstandingTool(config_path)
    return tool.describe_camera_motion_from_file(metadata_path)

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
    tool = CameraUnderstandingTool(config_path)
    return tool.analyze_camera_trajectory(extrinsics)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Translate camera motion matrices into natural language")
    parser.add_argument("--metadata_path", type=str, required=True,
                       help="Path to processing_metadata.json file")
    
    args = parser.parse_args()
    
    try:
        result = describe_camera_motion_from_file(args.metadata_path)
        
        print(f"Processing {result['num_poses']} camera poses from {result['metadata_path']}")
        if result['image_paths']:
            print(f"Images: {', '.join([Path(p).name for p in result['image_paths']])}")
        print("=" * 60)
        
        for description in result['motion_descriptions']:
            print(description)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)