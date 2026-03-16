import os
import glob
import json
import numpy as np
from typing import List, Union

from tool.recontruct import reconstruct_3d
# from tool.segment import segment_image, segment_automatic
# from tool.estimate_depth import estimate_depth
from tool.camera_understanding import analyze_camera_trajectory
from tool.novel_view_synthesis import (
    novel_view_synthesis, rotate_right, rotate_left,
    move_forward, move_backward, turn_around,
    average_look_at_directions,
)
import re


class Reconstruction:
    def __init__(self, point_cloud, extrinsics, intrinsics):
        self.point_cloud = point_cloud
        self.extrinsics = extrinsics # list of 4 *4 numpy array
        self.intrinsics = intrinsics


class Scene:
    """Simple scene class that holds image data."""

    def __init__(self, path_to_images: Union[str, List[str]], question: str = "", scene_id: str = None):
        self.question = question
        self.scene_id = scene_id
        self.images = self._load_images(path_to_images)
        self.reconstruction : Reconstruction = None
        self.code : str = None
        self.visual_clue = None

    def _load_images(self, path_to_images: Union[str, List[str]]) -> List[str]:
        """Load image paths from directory or list."""
        if isinstance(path_to_images, str):
            if os.path.isdir(path_to_images):
                # Load all images from directory
                image_extensions = ['*.png', '*.jpg', '*.jpeg']
                images = []
                for ext in image_extensions:
                    images.extend(glob.glob(os.path.join(path_to_images, ext)))
                return sorted(images)
            else:
                # Single image file
                return [path_to_images]
        else:
            # List of image paths
            return list(path_to_images)


def _load_processed_scene(processed_dir):
    """Load a previously processed scene from disk.

    Supports two layouts:
      1. reconstruct_pipe.py output: camera_matrices.npz + points.ply + processing_metadata.json
      2. ReconstructionTool output: cameras.npy + points3d.npy + metadata.json

    Returns a Reconstruction object, or None if the directory doesn't contain valid data.
    """
    if not os.path.isdir(processed_dir):
        return None

    point_cloud = None
    extrinsics = None
    intrinsics = None

    # --- Layout 1: reconstruct_pipe.py ---
    npz_path = os.path.join(processed_dir, 'camera_matrices.npz')
    ply_path = os.path.join(processed_dir, 'points.ply')
    meta_path = os.path.join(processed_dir, 'processing_metadata.json')

    if os.path.exists(ply_path) and (os.path.exists(npz_path) or os.path.exists(meta_path)):
        try:
            import trimesh
            pc = trimesh.load(ply_path)
            point_cloud = np.asarray(pc.vertices)
        except Exception:
            return None

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            extrinsics = data.get('extrinsic', None)
            intrinsics = data.get('intrinsic', None)
        elif os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            camera_poses = metadata.get('camera_poses', {})
            if 'extrinsic' in camera_poses:
                extrinsics = np.array(camera_poses['extrinsic'])
            if 'intrinsic' in camera_poses:
                intrinsics = np.array(camera_poses['intrinsic'])

        return Reconstruction(point_cloud, extrinsics, intrinsics)

    # --- Layout 2: ReconstructionTool._save_results ---
    cameras_path = os.path.join(processed_dir, 'cameras.npy')
    points_path = os.path.join(processed_dir, 'points3d.npy')

    if os.path.exists(points_path):
        point_cloud = np.load(points_path)
        if os.path.exists(cameras_path):
            extrinsics = np.load(cameras_path)
        return Reconstruction(point_cloud, extrinsics, intrinsics)

    return None


class pySpatial:
    """Simple interface for 3D vision tools."""

    # Base directory where reconstruct_pipe.py saves processed scenes
    PROCESSED_BASE_DIR = "/data/Datasets/MindCube/data/pySpatial_preprocessed"

    @staticmethod
    def reconstruct(scene: Scene, processed_dir: str = None):
        """3D reconstruction from scene images.

        If a previously processed result exists, load it instead of re-running
        reconstruction. The lookup order is:
          1. An explicit `processed_dir` argument
          2. PROCESSED_BASE_DIR / scene.scene_id  (if scene_id is set)
          3. Fall back to running reconstruct_3d()
        """
        # --- try to load cached reconstruction ---
        recon = None

        if processed_dir:
            recon = _load_processed_scene(processed_dir)
            if recon:
                print(f"Loaded processed scene from: {processed_dir}")

        if recon is None and scene.scene_id:
            candidate = os.path.join(pySpatial.PROCESSED_BASE_DIR, scene.scene_id)
            recon = _load_processed_scene(candidate)
            if recon:
                print(f"Loaded processed scene for scene_id '{scene.scene_id}' from: {candidate}")

        if recon is not None:
            scene.reconstruction = recon
            return recon

        # --- no cached result found, run reconstruction ---
        result = reconstruct_3d(scene.images, scene_id=scene.scene_id)

        # Convert the raw result dictionary to a Reconstruction object
        point_cloud = result.get('points', None)
        cameras = result.get('cameras', None)

        # Convert point cloud to numpy if it's a tensor
        if point_cloud is not None:
            if hasattr(point_cloud, 'cpu'):  # PyTorch tensor
                point_cloud = point_cloud.cpu().numpy()
            elif hasattr(point_cloud, 'numpy'):  # Other tensor types
                point_cloud = point_cloud.numpy()

        # Extract extrinsics and intrinsics from cameras if available
        extrinsics = None
        intrinsics = None

        if cameras is not None:
            extrinsics = cameras.cpu().numpy() if hasattr(cameras, 'cpu') else cameras

        # Also check for intrinsics in the result metadata
        metadata = result.get('metadata', {})
        if metadata and isinstance(metadata, dict):
            camera_poses = metadata.get('camera_poses', {})
            if isinstance(camera_poses, dict) and 'intrinsic' in camera_poses:
                intrinsics = np.array(camera_poses['intrinsic'])

        # Create and return Reconstruction object
        reconstruction = Reconstruction(point_cloud, extrinsics, intrinsics)

        # Store the raw result for debugging
        reconstruction._raw_result = result

        scene.reconstruction = reconstruction
        return reconstruction
    
    @staticmethod
    def describe_camera_motion(recon: Reconstruction):
        """Describe camera motion from reconstruction results.
        Args:
        """
        extrinsics = recon.extrinsics
        return analyze_camera_trajectory(extrinsics)

    @staticmethod
    def synthesize_novel_view(recon: Reconstruction, new_camera_pose, width=512, height=512, out_path=None):
        """Generate novel view synthesis from reconstruction results.
        Args:
            recon: Reconstruction object with point_cloud, extrinsics, intrinsics
            new_camera_pose: 3x4 or 4x4 extrinsic matrix for the new viewpoint
            width: output image width (default: 512)
            height: output image height (default: 512)  
            out_path: output image path (default: None, returns image object if not provided)
        Returns:
            str or image: path to the rendered image if out_path provided, otherwise image object
        """
        return novel_view_synthesis(recon, new_camera_pose, width, height, out_path)
    
    
    @staticmethod
    def _get_rotation_axis(recon):
        """Compute rotation axis from reconstruction extrinsics."""
        if recon is not None and recon.extrinsics is not None:
            extrinsics = recon.extrinsics
            # Handle (N, 3, 4) or (N, 4, 4) arrays as list of matrices
            if extrinsics.ndim == 3:
                return average_look_at_directions(extrinsics)
            # Single extrinsic — can't average, fall back
        return None

    @staticmethod
    def rotate_right(extrinsic, angle=None, recon=None):
        """Rotate camera pose to the right. Uses recon extrinsics to compute rotation axis."""
        axis = pySpatial._get_rotation_axis(recon)
        if angle is None:
            return rotate_right(extrinsic, axis=axis)
        else:
            return rotate_right(extrinsic, angle, axis=axis)

    @staticmethod
    def rotate_left(extrinsic, angle=None, recon=None):
        """Rotate camera pose to the left. Uses recon extrinsics to compute rotation axis."""
        axis = pySpatial._get_rotation_axis(recon)
        if angle is None:
            return rotate_left(extrinsic, axis=axis)
        else:
            return rotate_left(extrinsic, angle, axis=axis)

    @staticmethod
    def move_forward(extrinsic, distance=None):
        """Move camera pose forward, Noted that a default small step is provided"""
        if distance is None:
            return move_forward(extrinsic)
        else:
            return move_forward(extrinsic, distance)

    @staticmethod
    def move_backward(extrinsic, distance=None):
        """Move camera pose backward"""
        if distance is None:
            return move_backward(extrinsic)
        else:
            return move_backward(extrinsic, distance)

    @staticmethod
    def turn_around(extrinsic, recon=None):
        """Turn camera pose around 180 degrees. Uses recon extrinsics to compute rotation axis."""
        axis = pySpatial._get_rotation_axis(recon)
        return turn_around(extrinsic, axis=axis)


class Agent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
    def generate_code(self, scene: Scene):
        from agent.codeAgent.query import generate_code_from_query
        return generate_code_from_query(scene, self.api_key)
        
    def parse_LLM_response(self, scene: Scene, response: str):
        """
        Extracts the first python code block (```python ... ```) from text.
        Returns the code as a string, or "" if not found.
        """
        from agent.codeAgent.execute import parse_LLM_response
        code = parse_LLM_response(response)
        scene.code = code
        return code
        
    def execute(self, scene: Scene):
        """
        Execute a code string with a scene and return the visual clue result.
        """
        # try:
        #     from agent.codeAgent.execute import execute_code
        #     program = execute_code(scene.code)
            
        #     visual_clue = program(scene)
        #     return visual_clue
        # except Exception as e:
        #     import traceback
        #     error_details = f"Execution failed: {str(e)}\nTraceback: {traceback.format_exc()}"
        #     # Store the error for detailed reporting
        #     self.last_execution_error = error_details
        #     return f"there is an error during code generation, no visual clue provided. Error: {str(e)}"
        
        from agent.codeAgent.execute import execute_code
        program = execute_code(scene.code)
        
        visual_clue = program(scene)
        return visual_clue
    
    def answer(self, scene: Scene, visual_clue):
        # answer the question with visual clue
        from agent.anwer import answer

        # Set the visual clue in the scene
        scene.visual_clue = visual_clue

        # Call the answer function with API key
        return answer(scene, self.api_key)

    def basic_qa(self, scene: Scene):
        """Fallback: answer using only images + question, no pySpatial framework."""
        from agent.anwer import answer_without_visual_clue
        return answer_without_visual_clue(scene, self.api_key)