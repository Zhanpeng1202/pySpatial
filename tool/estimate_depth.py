import cv2
import torch
import numpy as np
import yaml
import sys
import os

# Get absolute paths for config resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# For Depth Anything V2, add the model directory to path
DEPTH_ANYTHING_PATH = os.path.join(PROJECT_ROOT, 'base_models', 'Depth_Anything_V2')
if DEPTH_ANYTHING_PATH not in sys.path:
    sys.path.insert(0, DEPTH_ANYTHING_PATH)

from depth_anything_v2.dpt import DepthAnythingV2

class DepthEstimator:
    def __init__(self, config_path='configs/main.yaml'):
        self.config_path = config_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_config()
        self._init_model()
    
    def _load_config(self):
        # Handle relative config path from project root
        if not os.path.isabs(self.config_path):
            self.config_path = os.path.join(PROJECT_ROOT, self.config_path)
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.checkpoint_path = config['tool']['depthAnything2']['checkpoint']
    
    def _init_model(self):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        
        encoder = 'vitl'  # Using large model
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Depth Anything V2 checkpoint not found: {self.checkpoint_path}")
        
        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()
    
    def estimate_depth(self, image_path):
        """
        Estimate depth from an image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            np.ndarray: Depth map in meters (HxW)
        """
        if isinstance(image_path, str):
            raw_img = cv2.imread(image_path)
        else:
            raw_img = image_path  # Already loaded image
            
        if raw_img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        print(f"Input image shape: {raw_img.shape}")
        print(f"Model device: {next(self.model.parameters()).device}")
        
        with torch.no_grad():
            # Use the model's forward pass manually to debug
            image, (h, w) = self.model.image2tensor(raw_img, input_size=518)
            print(f"Processed image tensor shape: {image.shape}")
            
            # Move to device
            image = image.to(self.device)
            print(f"Image tensor device: {image.device}")
            
            # Forward pass
            depth = self.model.forward(image)
            print(f"Raw depth output shape: {depth.shape}")
            print(f"Raw depth range: {depth.min().item():.6f} - {depth.max().item():.6f}")
            
            # Resize to original dimensions
            depth = torch.nn.functional.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
            
            depth_np = depth.cpu().numpy()
            print(f"Final depth range: {depth_np.min():.6f} - {depth_np.max():.6f}")
        
        return depth_np

def estimate_depth(image_path, config_path='configs/main.yaml'):
    """
    Simple function interface for VLM agent to estimate depth.
    
    Args:
        image_path (str): Path to the input image
        config_path (str): Path to config file
        
    Returns:
        np.ndarray: Depth map in meters
    """
    estimator = DepthEstimator(config_path)
    return estimator.estimate_depth(image_path)

def visualize_depth(image_path, depth_map, output_path=None):
    """
    Visualize depth estimation results.
    
    Args:
        image_path (str): Path to original image
        depth_map (np.ndarray): Depth map
        output_path (str): Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Load original image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show depth map
    depth_vis = axes[1].imshow(depth_map, cmap='plasma')
    axes[1].set_title(f"Depth Map\nRange: {depth_map.min():.2f} - {depth_map.max():.2f}m")
    axes[1].axis('off')
    plt.colorbar(depth_vis, ax=axes[1], label='Depth (m)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Depth visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # Test with example images
    import glob
    import matplotlib.pyplot as plt
    
    example_images = glob.glob("example/**/*.png", recursive=True) + glob.glob("example/**/*.jpg", recursive=True)
    
    if example_images:
        print(f"Testing depth estimation with {example_images[0]}")
        depth = estimate_depth(example_images[0])
        print(f"Depth map shape: {depth.shape}")
        print(f"Depth range: {depth.min():.2f} - {depth.max():.2f} meters")
        
        # Check if depth values are meaningful
        if depth.max() > 0:
            print("Depth estimation appears successful")
        else:
            print("Warning: Depth map contains all zeros - there may be an issue")
            # Check raw depth values before normalization
            print(f"Unique depth values: {np.unique(depth)[:10]}")  # Show first 10 unique values
        
        # Visualize depth estimation
        visualize_depth(example_images[0], depth, output_path="depth_estimation_result.png")
    else:
        print("No example images found")