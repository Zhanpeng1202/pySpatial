import os
import glob
from typing import List, Union

from tool.recontruct import reconstruct_3d
from tool.segment import segment_image, segment_automatic  
from tool.estimate_depth import estimate_depth
from tool.camera_understanding import describe_camera_motion
from tool.novel_view_synthesis import novel_view_synthesis
import re


class Reconstruction:
    def __init__(self, point_cloud, extrinsics, intrinsics):
        self.point_cloud = point_cloud
        self.extrinsics = extrinsics # list of 4 *4 numpy array
        self.intrinsics = intrinsics
        

class Scene:
    """Simple scene class that holds image data."""
    
    def __init__(self, path_to_images: Union[str, List[str]], question: str = ""):
        self.question = question
        self.images = self._load_images(path_to_images)
        self.reconstruction : Reconstruction = None
        
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
    # we disable other function for now
    
    @staticmethod
    def reconstruct(scene: Scene):
        """3D reconstruction from scene images."""
        
        return reconstruct_3d(scene.images)
    
    @staticmethod
    def describe_camera_motion(recon: Reconstruction):
        """Describe camera motion from reconstruction results.
        Args:
        """
        extrinsics = recon.extrinsics
        return describe_camera_motion(extrinsics)

    @staticmethod
    def synthesize_novel_view(recon: Reconstruction, new_camera_pose, width=512, height=512, out_path="novel_view.png"):
        """Generate novel view synthesis from reconstruction results.
        Args:
            recon: Reconstruction object with point_cloud, extrinsics, intrinsics
            new_camera_pose: 3x4 or 4x4 extrinsic matrix for the new viewpoint
            width: output image width (default: 512)
            height: output image height (default: 512)  
            out_path: output image path (default: "novel_view.png")
        Returns:
            str: path to the rendered image
        """
        return novel_view_synthesis(recon, new_camera_pose, width, height, out_path)
    
    @staticmethod
    def estimate_depth(image):
        return estimate_depth(image)


class Agent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
    def generate_code(self, scene: Scene):
        from VLMs.codeAgent.query import generate_code_from_query
        return generate_code_from_query(scene, self.api_key)
        
    def parse_LLM_response(self, response: str):
        """
        Extracts the first python code block (```python ... ```) from text.
        Returns the code as a string, or "" if not found.
        """
        match = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
        
    def execute(self, code: str):
        # we should implement this later
        pass
    
    