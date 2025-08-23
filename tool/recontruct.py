import torch
import numpy as np
import yaml
import sys
import os
import glob

# Get absolute paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# For VGGT, add the model directory to path
VGGT_PATH = os.path.join(PROJECT_ROOT, 'base_models', 'vggt')
if VGGT_PATH not in sys.path:
    sys.path.insert(0, VGGT_PATH)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

class ReconstructionTool:
    def __init__(self, config_path='configs/main.yaml'):
        self.config_path = config_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self._load_config()
        self._init_model()
    
    def _load_config(self):
        # Handle relative config path from project root
        if not os.path.isabs(self.config_path):
            self.config_path = os.path.join(PROJECT_ROOT, self.config_path)
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.checkpoint_path = config['tool']['vggt']['checkpoint']
    
    def _init_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"VGGT checkpoint not found: {self.checkpoint_path}")
            
        # Initialize VGGT model and load checkpoint
        self.model = VGGT()
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
    
    def reconstruct_3d(self, image_paths, output_dir=None):
        """
        Perform 3D reconstruction from multiple images.
        
        Args:
            image_paths (list or str): List of image paths or directory containing images
            output_dir (str): Optional output directory for saving results
            
        Returns:
            dict: Dictionary containing reconstruction results
                - cameras: Camera parameters
                - depths: Depth maps for each image
                - points: 3D point clouds
                - tracks: Feature tracks between images
        """
        if isinstance(image_paths, str):
            # If string provided, treat as directory and load all images
            if os.path.isdir(image_paths):
                image_paths = sorted(glob.glob(os.path.join(image_paths, "*.png")) + 
                                   glob.glob(os.path.join(image_paths, "*.jpg")) +
                                   glob.glob(os.path.join(image_paths, "*.jpeg")))
            else:
                # Single image path
                image_paths = [image_paths]
        
        if len(image_paths) < 2:
            raise ValueError("Need at least 2 images for 3D reconstruction")
        
        # Load and preprocess images
        images = load_and_preprocess_images(image_paths).to(self.device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                # Predict 3D reconstruction
                predictions = self.model(images)
        
        # Extract results
        results = {
            'cameras': predictions.get('cameras', None),
            'depths': predictions.get('depths', None),
            'points': predictions.get('points3d', None),
            'tracks': predictions.get('tracks', None),
            'image_paths': image_paths,
            'num_images': len(image_paths)
        }
        
        # Save results if output directory specified
        if output_dir:
            self._save_results(results, output_dir)
        
        return results
    
    def reconstruct_single_view(self, image_path):
        """
        Perform single-view 3D reconstruction (depth estimation + simple point cloud).
        
        Args:
            image_path (str): Path to single image
            
        Returns:
            dict: Single-view reconstruction results
        """
        images = load_and_preprocess_images([image_path]).to(self.device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions = self.model(images)
        
        results = {
            'depth': predictions.get('depths', None),
            'camera': predictions.get('cameras', None),
            'image_path': image_path
        }
        
        return results
    
    def _save_results(self, results, output_dir):
        """Save reconstruction results to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as numpy arrays for easy loading
        if results.get('cameras') is not None:
            np.save(os.path.join(output_dir, 'cameras.npy'), results['cameras'].cpu().numpy())
        
        if results.get('depths') is not None:
            np.save(os.path.join(output_dir, 'depths.npy'), results['depths'].cpu().numpy())
        
        if results.get('points') is not None:
            np.save(os.path.join(output_dir, 'points3d.npy'), results['points'].cpu().numpy())
        
        if results.get('tracks') is not None:
            np.save(os.path.join(output_dir, 'tracks.npy'), results['tracks'].cpu().numpy())
        
        # Save metadata
        metadata = {
            'image_paths': results['image_paths'],
            'num_images': results['num_images']
        }
        import json
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

def reconstruct_3d(image_paths, output_dir=None, config_path='configs/main.yaml'):
    """
    Simple function interface for VLM agent to perform 3D reconstruction.
    
    Args:
        image_paths (list or str): List of image paths or directory
        output_dir (str): Optional output directory
        config_path (str): Path to config file
        
    Returns:
        dict: Reconstruction results
    """
    reconstructor = ReconstructionTool(config_path)
    return reconstructor.reconstruct_3d(image_paths, output_dir)

def reconstruct_single_view(image_path, config_path='configs/main.yaml'):
    """
    Simple function interface for single-view reconstruction.
    
    Args:
        image_path (str): Path to image
        config_path (str): Path to config file
        
    Returns:
        dict: Single-view reconstruction results
    """
    reconstructor = ReconstructionTool(config_path)
    return reconstructor.reconstruct_single_view(image_path)

if __name__ == "__main__":
    # Test with example images
    example_dirs = glob.glob("example/*/")
    
    if example_dirs:
        print(f"Testing 3D reconstruction with {example_dirs[0]}")
        
        # Test multi-view reconstruction
        results = reconstruct_3d(example_dirs[0])
        print(f"Reconstruction completed for {results['num_images']} images")
        
        if results.get('cameras') is not None:
            print(f"Camera parameters shape: {results['cameras'].shape}")
        if results.get('depths') is not None:
            print(f"Depth maps shape: {results['depths'].shape}")
        if results.get('points') is not None:
            print(f"3D points shape: {results['points'].shape}")
    
    # Test single image reconstruction
    single_images = glob.glob("example/**/*.png", recursive=True) + glob.glob("example/**/*.jpg", recursive=True)
    if single_images:
        print(f"\nTesting single-view reconstruction with {single_images[0]}")
        single_result = reconstruct_single_view(single_images[0])
        if single_result.get('depth') is not None:
            print(f"Single-view depth shape: {single_result['depth'].shape}")
    
    if not example_dirs and not single_images:
        print("No example images found")