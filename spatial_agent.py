"""
Spatial Agent Interface - Unified API for 3D Vision Models

This module provides a simple, unified interface for VLM agents to access:
- Depth estimation (Depth-Anything-V2)
- Image segmentation (SAM2) 
- 3D reconstruction (VGGT)

Usage for VLM agents:
```python
from spatial_agent import SpatialAgent

# Initialize the agent
agent = SpatialAgent()

# Estimate depth
depth_map = agent.estimate_depth("image.jpg")

# Segment image
masks = agent.segment_image("image.jpg")

# Reconstruct 3D scene
reconstruction = agent.reconstruct_3d(["img1.jpg", "img2.jpg", "img3.jpg"])
```
"""

import os
import sys
import numpy as np
from typing import Union, List, Dict, Any, Optional

# Get absolute paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add tool directory to Python path
TOOL_PATH = os.path.join(PROJECT_ROOT, 'tool')
if TOOL_PATH not in sys.path:
    sys.path.insert(0, TOOL_PATH)

from estimate_depth import estimate_depth, DepthEstimator
from segment import segment_image, segment_automatic, SegmentationTool
from recontruct import reconstruct_3d, reconstruct_single_view, ReconstructionTool

class SpatialAgent:
    """
    Unified interface for 3D spatial understanding models.
    
    This class provides simple methods for VLM agents to perform:
    - Depth estimation
    - Image segmentation
    - 3D reconstruction
    """
    
    def __init__(self, config_path: str = 'configs/main.yaml'):
        """
        Initialize the Spatial Agent.
        
        Args:
            config_path (str): Path to configuration file containing model weights
        """
        self.config_path = config_path
        self._depth_estimator = None
        self._segmentation_tool = None
        self._reconstruction_tool = None
    
    @property
    def depth_estimator(self):
        """Lazy loading of depth estimation model."""
        if self._depth_estimator is None:
            self._depth_estimator = DepthEstimator(self.config_path)
        return self._depth_estimator
    
    @property
    def segmentation_tool(self):
        """Lazy loading of segmentation model."""
        if self._segmentation_tool is None:
            self._segmentation_tool = SegmentationTool(self.config_path)
        return self._segmentation_tool
    
    @property
    def reconstruction_tool(self):
        """Lazy loading of reconstruction model."""
        if self._reconstruction_tool is None:
            self._reconstruction_tool = ReconstructionTool(self.config_path)
        return self._reconstruction_tool
    
    # DEPTH ESTIMATION METHODS
    def estimate_depth(self, image_path: Union[str, np.ndarray]) -> np.ndarray:
        """
        Estimate depth from a single image.
        
        Args:
            image_path: Path to image or loaded image array
            
        Returns:
            np.ndarray: Depth map in meters (H x W)
        """
        return self.depth_estimator.estimate_depth(image_path)
    
    # SEGMENTATION METHODS
    def segment_image(self, 
                     image_path: Union[str, np.ndarray], 
                     prompts: Optional[Dict] = None) -> tuple:
        """
        Segment image using prompts (points, boxes, etc.).
        
        Args:
            image_path: Path to image or loaded image array
            prompts: Dictionary with 'point_coords', 'point_labels', 'box'
            
        Returns:
            tuple: (masks, scores, logits)
        """
        return self.segmentation_tool.segment_image(image_path, prompts)
    
    def segment_automatic(self, image_path: Union[str, np.ndarray]) -> List[Dict]:
        """
        Perform automatic segmentation of entire image.
        
        Args:
            image_path: Path to image or loaded image array
            
        Returns:
            List[Dict]: List of mask dictionaries with segmentation, score, etc.
        """
        return self.segmentation_tool.segment_automatic(image_path)
    
    def segment_object_at_point(self, 
                               image_path: Union[str, np.ndarray], 
                               x: int, y: int) -> np.ndarray:
        """
        Segment object at specific point coordinates.
        
        Args:
            image_path: Path to image or loaded image array
            x, y: Pixel coordinates of point to segment
            
        Returns:
            np.ndarray: Binary mask of segmented object
        """
        prompts = {
            'point_coords': np.array([[x, y]]),
            'point_labels': np.array([1])
        }
        masks, scores, _ = self.segment_image(image_path, prompts)
        return masks[0] if len(masks) > 0 else None
    
    def segment_object_in_box(self, 
                             image_path: Union[str, np.ndarray], 
                             x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """
        Segment object within bounding box.
        
        Args:
            image_path: Path to image or loaded image array
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            np.ndarray: Binary mask of segmented object
        """
        prompts = {
            'box': np.array([x1, y1, x2, y2])
        }
        masks, scores, _ = self.segment_image(image_path, prompts)
        return masks[0] if len(masks) > 0 else None
    
    # 3D RECONSTRUCTION METHODS
    def reconstruct_3d(self, 
                      image_paths: Union[str, List[str]], 
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform 3D reconstruction from multiple images.
        
        Args:
            image_paths: List of image paths or directory containing images
            output_dir: Optional directory to save reconstruction results
            
        Returns:
            Dict: Reconstruction results with cameras, depths, points, tracks
        """
        return self.reconstruction_tool.reconstruct_3d(image_paths, output_dir)
    
    def reconstruct_single_view(self, image_path: str) -> Dict[str, Any]:
        """
        Perform single-view depth-based reconstruction.
        
        Args:
            image_path: Path to single image
            
        Returns:
            Dict: Single-view reconstruction with depth and camera info
        """
        return self.reconstruction_tool.reconstruct_single_view(image_path)
    
    # COMBINED ANALYSIS METHODS
    def analyze_scene_3d(self, 
                        image_paths: Union[str, List[str]], 
                        segment_objects: bool = True,
                        estimate_depth: bool = True) -> Dict[str, Any]:
        """
        Complete 3D scene analysis combining all models.
        
        Args:
            image_paths: Image path(s) for analysis
            segment_objects: Whether to perform object segmentation
            estimate_depth: Whether to estimate depth maps
            
        Returns:
            Dict: Complete scene analysis results
        """
        results = {}
        
        # Handle single image vs multiple images
        if isinstance(image_paths, str):
            if os.path.isdir(image_paths):
                import glob
                img_list = sorted(glob.glob(os.path.join(image_paths, "*.png")) + 
                                glob.glob(os.path.join(image_paths, "*.jpg")))
            else:
                img_list = [image_paths]
        else:
            img_list = image_paths
        
        results['image_paths'] = img_list
        results['num_images'] = len(img_list)
        
        # 3D Reconstruction (if multiple images)
        if len(img_list) >= 2:
            print(f"Performing 3D reconstruction with {len(img_list)} images...")
            results['reconstruction'] = self.reconstruct_3d(img_list)
        elif len(img_list) == 1:
            print("Performing single-view reconstruction...")
            results['reconstruction'] = self.reconstruct_single_view(img_list[0])
        
        # Depth estimation for each image
        if estimate_depth:
            print("Estimating depth for each image...")
            results['depths'] = []
            for img_path in img_list:
                depth = self.estimate_depth(img_path)
                results['depths'].append(depth)
        
        # Object segmentation for each image
        if segment_objects:
            print("Segmenting objects in each image...")
            results['segments'] = []
            for img_path in img_list:
                segments = self.segment_automatic(img_path)
                results['segments'].append(segments)
        
        return results
    
    def get_depth_at_point(self, image_path: str, x: int, y: int) -> float:
        """
        Get depth value at specific pixel coordinates.
        
        Args:
            image_path: Path to image
            x, y: Pixel coordinates
            
        Returns:
            float: Depth value in meters
        """
        depth_map = self.estimate_depth(image_path)
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return float(depth_map[y, x])
        else:
            raise ValueError(f"Coordinates ({x}, {y}) out of image bounds")
    
    def get_object_depth(self, image_path: str, x: int, y: int) -> Dict[str, Any]:
        """
        Get segmented object and its depth information.
        
        Args:
            image_path: Path to image
            x, y: Point coordinates within object
            
        Returns:
            Dict: Object mask, depth map, and depth statistics
        """
        # Segment object at point
        mask = self.segment_object_at_point(image_path, x, y)
        
        # Get depth map
        depth_map = self.estimate_depth(image_path)
        
        # Extract depth for segmented object
        object_depths = depth_map[mask > 0] if mask is not None else []
        
        return {
            'mask': mask,
            'depth_map': depth_map,
            'object_depths': object_depths,
            'mean_depth': np.mean(object_depths) if len(object_depths) > 0 else None,
            'min_depth': np.min(object_depths) if len(object_depths) > 0 else None,
            'max_depth': np.max(object_depths) if len(object_depths) > 0 else None,
            'point_depth': self.get_depth_at_point(image_path, x, y)
        }

# Convenience functions for direct use
def quick_depth(image_path: str) -> np.ndarray:
    """Quick depth estimation."""
    agent = SpatialAgent()
    return agent.estimate_depth(image_path)

def quick_segment(image_path: str, x: int = None, y: int = None) -> np.ndarray:
    """Quick segmentation at point or automatic."""
    agent = SpatialAgent()
    if x is not None and y is not None:
        return agent.segment_object_at_point(image_path, x, y)
    else:
        return agent.segment_automatic(image_path)

def quick_reconstruct(image_paths: Union[str, List[str]]) -> Dict[str, Any]:
    """Quick 3D reconstruction."""
    agent = SpatialAgent()
    return agent.reconstruct_3d(image_paths)

def quick_analyze(image_paths: Union[str, List[str]]) -> Dict[str, Any]:
    """Quick complete scene analysis."""
    agent = SpatialAgent()
    return agent.analyze_scene_3d(image_paths)

if __name__ == "__main__":
    # Example usage
    agent = SpatialAgent()
    
    print("Spatial Agent initialized successfully!")
    print("Available methods:")
    print("- agent.estimate_depth(image_path)")
    print("- agent.segment_image(image_path, prompts)")
    print("- agent.segment_automatic(image_path)")
    print("- agent.reconstruct_3d(image_paths)")
    print("- agent.analyze_scene_3d(image_paths)")
    
    # Test with example images if available
    import glob
    examples = glob.glob("example/**/*.png", recursive=True) + glob.glob("example/**/*.jpg", recursive=True)
    
    if examples:
        print(f"\nTesting with example image: {examples[0]}")
        
        try:
            # Test depth estimation
            depth = agent.estimate_depth(examples[0])
            print(f"✓ Depth estimation: {depth.shape}")
            
            # Test segmentation
            segments = agent.segment_automatic(examples[0])
            print(f"✓ Segmentation: {len(segments)} objects found")
            
            # Test single-view reconstruction
            recon = agent.reconstruct_single_view(examples[0])
            print(f"✓ Single-view reconstruction completed")
            
        except Exception as e:
            print(f"Error during testing: {e}")
    else:
        print("No example images found for testing")