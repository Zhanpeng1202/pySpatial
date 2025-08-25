import os
import glob
from typing import List, Union

from tool.recontruct import reconstruct_3d
from tool.segment import segment_image, segment_automatic  
from tool.estimate_depth import estimate_depth
from tool.camera_understanding import describe_camera_motion


class Scene:
    """Simple scene class that holds image data."""
    
    def __init__(self, path_to_images: Union[str, List[str]], question: str = ""):
        self.question = question
        self.images = self._load_images(path_to_images)
        
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
    def describe_camera_motion(scene: Scene, reconstruction_result):
        """Describe camera motion from reconstruction results."""
        if 'cameras' in reconstruction_result and reconstruction_result['cameras'] is not None:
            # Extract extrinsics from reconstruction
            cameras = reconstruction_result['cameras'].cpu().numpy()
            extrinsics = cameras.tolist()
            return describe_camera_motion(extrinsics, scene.images)
        else:
            return ["Camera motion analysis not available"]

    # def novel_view_synthesis():
    #     pass
    
    # def generate_cogMap():
    #     pass

class Agent:
    def __init__():
        pass
    def generate_code():
        pass
    def parse_LLM_response():
        pass
    def execute():
        pass
    
    