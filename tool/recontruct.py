import torch
import numpy as np
import yaml
import sys
import os
import glob
import json
import open3d as o3d

# Get absolute paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# For VGGT, add the model directory to path
VGGT_PATH = os.path.join(PROJECT_ROOT, 'base_models', 'vggt')
if VGGT_PATH not in sys.path:
    sys.path.insert(0, VGGT_PATH)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

class ReconstructionTool:
    def __init__(self, config_path='configs/main.yaml', use_precomputed=True):
        self.config_path = config_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.use_precomputed = use_precomputed
        self.precomputed_base_dir = "/data/Datasets/MindCube/data/vggt_processed_all"
        self._load_config()
        if not self.use_precomputed:
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
    
    def _extract_scene_name_from_path(self, image_paths):
        """Extract scene name from image paths."""
        if isinstance(image_paths, str):
            if os.path.isdir(image_paths):
                # If it's a directory, use the directory name as scene name
                return os.path.basename(image_paths.rstrip('/'))
            else:
                # Single image, extract from path
                image_paths = [image_paths]
        
        # Try to extract scene name from the first image path
        first_path = image_paths[0]
        path_parts = first_path.split(os.sep)
        
        # Look for the MindCube scene name pattern (long hash-like string)
        # The scene name is typically a very long string with underscores
        for part in path_parts:
            # Check if this part looks like a MindCube scene name
            # (contains long hash and underscores, and is reasonably long)
            if len(part) > 50 and '_' in part and 'q' in part:
                return part
        
        # Fallback: look for any part that's longer than 20 characters with underscores
        for part in path_parts:
            if len(part) > 20 and '_' in part:
                return part
        
        # Last resort: use parent directory name
        return os.path.basename(os.path.dirname(first_path))
    
    def _load_precomputed_data(self, scene_name):
        """Load pre-computed reconstruction data from MindCube processed directory."""
        scene_dir = os.path.join(self.precomputed_base_dir, scene_name)
        
        if not os.path.exists(scene_dir):
            raise FileNotFoundError(f"Pre-computed data not found for scene: {scene_name} at {scene_dir}")
        
        # Load processing metadata (contains extrinsics)
        metadata_path = os.path.join(scene_dir, 'processing_metadata.json')
        point_cloud_path = os.path.join(scene_dir, 'points.ply')  # Updated to correct filename
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Processing metadata not found: {metadata_path}")
        
        if not os.path.exists(point_cloud_path):
            raise FileNotFoundError(f"Point cloud not found: {point_cloud_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load point cloud
        point_cloud = o3d.io.read_point_cloud(point_cloud_path)
        points = np.asarray(point_cloud.points)
        
        # Convert to torch tensors
        points_tensor = torch.from_numpy(points).float()
        
        # Extract camera parameters from metadata only
        cameras = None
        
        # Load camera data from JSON metadata
        if 'cameras' in metadata or 'extrinsics' in metadata or 'camera_poses' in metadata:
            camera_data = metadata.get('cameras', metadata.get('extrinsics', metadata.get('camera_poses', None)))
            if camera_data is not None:
                # Handle nested structure like camera_poses.extrinsic
                if isinstance(camera_data, dict) and 'extrinsic' in camera_data:
                    camera_data = camera_data['extrinsic']
                cameras = torch.tensor(camera_data).float()
        
        results = {
            'cameras': cameras,
            'depths': None,  # Not typically saved in pre-computed data
            'points': points_tensor,
            'tracks': None,  # Not typically saved in pre-computed data
            'metadata': metadata,
            'scene_name': scene_name,
            'loaded_from_precomputed': True
        }
        
        return results
    
    def reconstruct_3d(self, image_paths, output_dir=None, scene_id=None):
        """
        Perform 3D reconstruction from multiple images.
        
        Args:
            image_paths (list or str): List of image paths or directory containing images
            output_dir (str): Optional output directory for saving results
            scene_id (str): Optional scene ID to use instead of extracting from paths
            
        Returns:
            dict: Dictionary containing reconstruction results
                - cameras: Camera parameters
                - depths: Depth maps for each image
                - points: 3D point clouds
                - tracks: Feature tracks between images
        """
        # Handle input paths
        original_input = image_paths
        if isinstance(image_paths, str):
            # If string provided, treat as directory and load all images
            if os.path.isdir(image_paths):
                image_paths = sorted(glob.glob(os.path.join(image_paths, "*.png")) + 
                                   glob.glob(os.path.join(image_paths, "*.jpg")) +
                                   glob.glob(os.path.join(image_paths, "*.jpeg")))
            else:
                # Single image path
                image_paths = [image_paths]
        
        # Try to use pre-computed data if enabled
        if self.use_precomputed:
            try:
                # Use provided scene_id or extract from paths
                if scene_id:
                    scene_name = scene_id
                    print(f"Using provided scene ID: {scene_name}")
                else:
                    scene_name = self._extract_scene_name_from_path(original_input)
                    print(f"Extracted scene name from path: {scene_name}")
                
                print(f"Attempting to load pre-computed data for scene: {scene_name}")
                results = self._load_precomputed_data(scene_name)
                
                # Add image paths and count to results
                results['image_paths'] = image_paths
                results['num_images'] = len(image_paths)
                
                print(f"Successfully loaded pre-computed data for scene: {scene_name}")
                
                # Save results if output directory specified
                if output_dir:
                    self._save_results(results, output_dir)
                
                return results
                
            except (FileNotFoundError, KeyError) as e:
                print(f"Pre-computed data not available: {e}")
                if not self.use_precomputed:
                    print("Falling back to VGGT reconstruction...")
                    if self.model is None:
                        print("Initializing VGGT model for fallback reconstruction...")
                        self._init_model()
                else:
                    # If pre-computed mode is enabled and data not found, raise error
                    raise FileNotFoundError(f"Pre-computed data not found for scene and VGGT fallback is disabled.")
        
        # Only run VGGT reconstruction if pre-computed is disabled
        if not self.use_precomputed:
            # Fallback to original VGGT reconstruction
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
                'num_images': len(image_paths),
                'loaded_from_precomputed': False
            }
            
            # Save results if output directory specified
            if output_dir:
                self._save_results(results, output_dir)
            
            return results
        else:
            # If we reach here, it means pre-computed was enabled but data not found
            raise FileNotFoundError("Pre-computed reconstruction data not available")
    
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

def reconstruct_3d(image_paths, output_dir=None, config_path='configs/main.yaml', use_precomputed=True, scene_id=None):
    """
    Simple function interface for VLM agent to perform 3D reconstruction.
    
    Args:
        image_paths (list or str): List of image paths or directory
        output_dir (str): Optional output directory
        config_path (str): Path to config file
        use_precomputed (bool): Whether to use pre-computed data from MindCube dataset
        scene_id (str): Optional scene ID to use for pre-computed data lookup
        
    Returns:
        dict: Reconstruction results
    """
    reconstructor = ReconstructionTool(config_path, use_precomputed=use_precomputed)
    return reconstructor.reconstruct_3d(image_paths, output_dir, scene_id=scene_id)

def reconstruct_single_view(image_path, config_path='configs/main.yaml', use_precomputed=True):
    """
    Simple function interface for single-view reconstruction.
    
    Args:
        image_path (str): Path to image
        config_path (str): Path to config file
        use_precomputed (bool): Whether to use pre-computed data from MindCube dataset
        
    Returns:
        dict: Single-view reconstruction results
    """
    reconstructor = ReconstructionTool(config_path, use_precomputed=use_precomputed)
    return reconstructor.reconstruct_single_view(image_path)

def set_reconstruction_mode(use_precomputed=True):
    """
    Set the default reconstruction mode for the module.
    
    Args:
        use_precomputed (bool): If True, use pre-computed MindCube data; if False, use VGGT reconstruction
    """
    global _DEFAULT_USE_PRECOMPUTED
    _DEFAULT_USE_PRECOMPUTED = use_precomputed
    print(f"Reconstruction mode set to: {'pre-computed data' if use_precomputed else 'VGGT reconstruction'}")

# Global setting for default behavior
_DEFAULT_USE_PRECOMPUTED = True

if __name__ == "__main__":
    print("=== Reconstruction Tool Test ===")
    print(f"Pre-computed data directory: {ReconstructionTool().precomputed_base_dir}")
    
    # Test with example images
    example_dirs = glob.glob("example/*/")
    
    if example_dirs:
        print(f"\nTesting with example directory: {example_dirs[0]}")
        
        # Test with pre-computed data (default mode)
        print("\n--- Testing with pre-computed data mode ---")
        try:
            results = reconstruct_3d(example_dirs[0], use_precomputed=True)
            print(f"Reconstruction completed for {results['num_images']} images")
            print(f"Loaded from pre-computed: {results.get('loaded_from_precomputed', False)}")
            
            if results.get('cameras') is not None:
                print(f"Camera parameters shape: {results['cameras'].shape}")
            if results.get('depths') is not None:
                print(f"Depth maps shape: {results['depths'].shape}")
            if results.get('points') is not None:
                print(f"3D points shape: {results['points'].shape}")
                
        except Exception as e:
            print(f"Pre-computed reconstruction failed: {e}")
        
        # Test with VGGT reconstruction (disabled mode)
        print("\n--- Testing with VGGT reconstruction mode ---")
        try:
            results_vggt = reconstruct_3d(example_dirs[0], use_precomputed=False)
            print(f"VGGT reconstruction completed for {results_vggt['num_images']} images")
            print(f"Loaded from pre-computed: {results_vggt.get('loaded_from_precomputed', False)}")
            
        except Exception as e:
            print(f"VGGT reconstruction failed (this is expected if VGGT model is not available): {e}")
    
    # Test single image reconstruction
    single_images = glob.glob("example/**/*.png", recursive=True) + glob.glob("example/**/*.jpg", recursive=True)
    if single_images:
        print(f"\nTesting single-view reconstruction with {single_images[0]}")
        try:
            single_result = reconstruct_single_view(single_images[0], use_precomputed=True)
            if single_result.get('depth') is not None:
                print(f"Single-view depth shape: {single_result['depth'].shape}")
        except Exception as e:
            print(f"Single-view reconstruction failed: {e}")
    
    if not example_dirs and not single_images:
        print("No example images found")
        print("\nTo test with MindCube data, use:")
        print("  results = reconstruct_3d('/path/to/scene/images')")
        print("  # Will automatically load from /data/Datasets/MindCube/data/vggt_processed_all/{scene_name}/")
    
    # Demonstrate mode switching
    print("\n--- Mode switching demonstration ---")
    print("Default mode: pre-computed data enabled")
    set_reconstruction_mode(False)  # Disable pre-computed data
    set_reconstruction_mode(True)   # Re-enable pre-computed data