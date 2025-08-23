import torch
import cv2
import numpy as np
import yaml
import sys
import os

# Get absolute paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# For SAM2, we need to add the sam2 directory to path so it can find its internal sam2 module
SAM2_PATH = os.path.join(PROJECT_ROOT, 'base_models', 'sam2')
if SAM2_PATH not in sys.path:
    sys.path.insert(0, SAM2_PATH)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SegmentationTool:
    def __init__(self, config_path='configs/main.yaml'):
        self.config_path = config_path
        self.predictor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_config()
        self._init_model()
    
    def _load_config(self):
        # Handle relative config path from project root
        if not os.path.isabs(self.config_path):
            self.config_path = os.path.join(PROJECT_ROOT, self.config_path)
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.checkpoint_path = config['tool']['sam2']['checkpoint']
    
    def _init_model(self):
        # Use relative path for model config (relative to SAM2 package)
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint_path}")
            
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, self.checkpoint_path))
    
    def segment_image(self, image_path, prompts=None, point_coords=None, point_labels=None, box=None):
        """
        Segment image using SAM2.
        
        Args:
            image_path (str or np.ndarray): Path to image or loaded image
            prompts (dict): Dictionary containing prompt information
            point_coords (np.ndarray): Point coordinates for prompting (Nx2)
            point_labels (np.ndarray): Point labels (N,) - 1 for foreground, 0 for background
            box (np.ndarray): Bounding box [x1, y1, x2, y2]
            
        Returns:
            tuple: (masks, scores, logits) from SAM2
        """
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
            
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)
            
            # Use provided prompts or default to center point
            if prompts:
                point_coords = prompts.get('point_coords', point_coords)
                point_labels = prompts.get('point_labels', point_labels)
                box = prompts.get('box', box)
            
            # Default to center point if no prompts provided
            if point_coords is None and box is None:
                h, w = image.shape[:2]
                point_coords = np.array([[w//2, h//2]])
                point_labels = np.array([1])
            
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True
            )
        
        return masks, scores, logits
    
    def segment_automatic(self, image_path):
        """
        Perform automatic mask generation on the entire image.
        
        Args:
            image_path (str or np.ndarray): Path to image or loaded image
            
        Returns:
            list: List of mask dictionaries
        """
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
            
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # For automatic segmentation, we'd need SAM2AutomaticMaskGenerator
        # For now, return multiple segments using different points
        masks_list = []
        h, w = image.shape[:2]
        
        # Sample points across the image
        points = [
            [w//4, h//4], [3*w//4, h//4], 
            [w//4, 3*h//4], [3*w//4, 3*h//4],
            [w//2, h//2]
        ]
        
        for point in points:
            masks, scores, _ = self.segment_image(image, point_coords=np.array([point]), point_labels=np.array([1]))
            if len(masks) > 0:
                masks_list.append({
                    'segmentation': masks[0],  # Best mask
                    'score': scores[0],
                    'point_coords': point
                })
        
        return masks_list

def segment_image(image_path, prompts=None, config_path='configs/main.yaml'):
    """
    Simple function interface for VLM agent to segment images.
    
    Args:
        image_path (str): Path to the input image
        prompts (dict): Optional prompts (point_coords, point_labels, box)
        config_path (str): Path to config file
        
    Returns:
        tuple: (masks, scores, logits)
    """
    segmenter = SegmentationTool(config_path)
    return segmenter.segment_image(image_path, prompts)

def segment_automatic(image_path, config_path='configs/main.yaml'):
    """
    Simple function interface for automatic segmentation.
    
    Args:
        image_path (str): Path to the input image
        config_path (str): Path to config file
        
    Returns:
        list: List of mask dictionaries
    """
    segmenter = SegmentationTool(config_path)
    return segmenter.segment_automatic(image_path)

def visualize_masks(image_path, masks, scores, output_path=None, point_coords=None, point_labels=None):
    """
    Visualize segmentation masks overlaid on the original image.
    
    Args:
        image_path (str): Path to original image
        masks (np.ndarray): Array of masks
        scores (np.ndarray): Mask scores
        output_path (str): Optional path to save visualization
        point_coords (np.ndarray): Optional point coordinates used for segmentation
        point_labels (np.ndarray): Optional point labels
    """
    import matplotlib.pyplot as plt
    
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path
    
    # Create subplots
    n_masks = len(masks)
    fig, axes = plt.subplots(1, n_masks + 1, figsize=(5 * (n_masks + 1), 5))
    
    if n_masks == 0:
        axes = [axes]
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Add points if provided
    if point_coords is not None and point_labels is not None:
        pos_points = point_coords[point_labels == 1]
        neg_points = point_coords[point_labels == 0]
        if len(pos_points) > 0:
            axes[0].scatter(pos_points[:, 0], pos_points[:, 1], 
                          color='green', marker='*', s=200, 
                          edgecolor='white', linewidth=2, label='Positive')
        if len(neg_points) > 0:
            axes[0].scatter(neg_points[:, 0], neg_points[:, 1], 
                          color='red', marker='*', s=200, 
                          edgecolor='white', linewidth=2, label='Negative')
        if len(pos_points) > 0 or len(neg_points) > 0:
            axes[0].legend()
    
    # Show each mask
    for i, (mask, score) in enumerate(zip(masks, scores)):
        ax = axes[i + 1]
        
        # Show original image
        ax.imshow(image)
        
        # Overlay mask with transparency
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[:, :, 0] = 30/255   # Red
        colored_mask[:, :, 1] = 144/255  # Green  
        colored_mask[:, :, 2] = 255/255  # Blue
        colored_mask[:, :, 3] = 0.6 * mask  # Alpha based on mask
        
        ax.imshow(colored_mask)
        ax.set_title(f"Mask {i+1} (Score: {score:.3f})")
        ax.axis('off')
        
        # Add points if provided
        if point_coords is not None and point_labels is not None:
            pos_points = point_coords[point_labels == 1]
            neg_points = point_coords[point_labels == 0]
            if len(pos_points) > 0:
                ax.scatter(pos_points[:, 0], pos_points[:, 1], 
                          color='green', marker='*', s=200, 
                          edgecolor='white', linewidth=2)
            if len(neg_points) > 0:
                ax.scatter(neg_points[:, 0], neg_points[:, 1], 
                          color='red', marker='*', s=200, 
                          edgecolor='white', linewidth=2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt
    
    example_images = glob.glob("example/**/*.png", recursive=True) + glob.glob("example/**/*.jpg", recursive=True)
    
    if example_images:
        print(f"Testing segmentation with {example_images[0]}")
        
        # Test point-based segmentation
        masks, scores, _ = segment_image(example_images[0])
        print(f"Generated {len(masks)} masks with scores: {scores}")
        
        # Visualize the results
        visualize_masks(example_images[0], masks, scores, 
                       output_path="segmentation_result.png",
                       point_coords=np.array([[masks.shape[2]//2, masks.shape[1]//2]]),
                       point_labels=np.array([1]))
        
        # Test automatic segmentation
        auto_masks = segment_automatic(example_images[0])
        print(f"Automatic segmentation generated {len(auto_masks)} segments")
        
        # Visualize automatic segmentation (show first 3 masks)
        if len(auto_masks) > 0:
            auto_masks_array = np.array([m['segmentation'] for m in auto_masks[:3]])
            auto_scores = np.array([m['score'] for m in auto_masks[:3]])
            visualize_masks(example_images[0], auto_masks_array, auto_scores, 
                           output_path="auto_segmentation_result.png")
    else:
        print("No example images found")