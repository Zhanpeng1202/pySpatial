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
    
    @staticmethod
    def reconstruct(scene: Scene):
        """3D reconstruction from scene images."""
        return reconstruct_3d(scene.images)
    
    @staticmethod
    def segment(scene: Scene, image_index: int, prompts=None):
        """Segment specific image in scene."""
        return segment_image(scene.images[image_index], prompts)
    
    @staticmethod
    def segment_all(scene: Scene, prompts=None):
        """Segment all images in scene."""
        results = []
        for image_path in scene.images:
            masks, scores, logits = segment_image(image_path, prompts)
            results.append((masks, scores, logits))
        return results
    
    @staticmethod
    def estimate_depth(scene: Scene, image_index: int):
        """Estimate depth for specific image in scene."""
        return estimate_depth(scene.images[image_index])
    
    @staticmethod
    def estimate_all_depths(scene: Scene):
        """Estimate depth for all images in scene."""
        results = []
        for image_path in scene.images:
            depth = estimate_depth(image_path)
            results.append(depth)
        return results
    
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


class Agent:
    """AI agent interface - to be implemented."""
    pass
    