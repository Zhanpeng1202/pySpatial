import os
import glob
from typing import List, Union

from tool.recontruct import reconstruct_3d
# from tool.segment import segment_image, segment_automatic  
# from tool.estimate_depth import estimate_depth
from tool.camera_understanding import analyze_camera_trajectory
from tool.novel_view_synthesis import novel_view_synthesis, rotate_right, rotate_left, move_forward, move_backward, turn_around
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


class pySpatial:
    """Simple interface for 3D vision tools."""
    
    @staticmethod
    def reconstruct(scene: Scene):
        """3D reconstruction from scene images."""
        
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
            # Assume cameras contains extrinsic matrices
            extrinsics = cameras.cpu().numpy() if hasattr(cameras, 'cpu') else cameras
        
        # Create and return Reconstruction object
        reconstruction = Reconstruction(point_cloud, extrinsics, intrinsics)
        
        # Store the raw result for debugging
        reconstruction._raw_result = result
        
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
    def rotate_right(extrinsic, angle=None):
        """Rotate camera pose to the right"""
        if angle is None:
            return rotate_right(extrinsic)
        else:
            return rotate_right(extrinsic, angle)
    
    @staticmethod
    def rotate_left(extrinsic, angle=None):
        """Rotate camera pose to the left"""
        if angle is None:
            return rotate_left(extrinsic)
        else:
            return rotate_left(extrinsic, angle)
    
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
    def turn_around(extrinsic):
        """Turn camera pose around 180 degrees"""
        return turn_around(extrinsic)


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