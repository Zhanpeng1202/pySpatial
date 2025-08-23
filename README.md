# pySpatial: 3D-Aware VLM Agent Toolkit

A unified interface for integrating 3D spatial understanding capabilities into Vision Language Models (VLMs), addressing current limitations in spatial reasoning for real-world problems.

## 🎯 Project Motivation

Current Vision Language Models (VLMs) struggle with spatial-related real-world problems due to lack of 3D understanding. This project aims to add 3D inductive bias into VLMs by creating an agentic system that seamlessly integrates multiple state-of-the-art 3D vision models.

## 🏗️ Architecture

### Core Models
- **Depth-Anything-V2**: Robust monocular depth estimation
- **SAM2**: Advanced image and video segmentation 
- **VGGT**: Large-scale 3D reconstruction model

### Design Philosophy
- **Simple Interface**: VLM agents need only write minimal code to access powerful 3D capabilities
- **Unified API**: Single entry point for all spatial understanding tasks
- **Modular Design**: Each model can be used independently or combined
- **Easy Integration**: Drop-in solution for existing VLM workflows

## 📁 Project Structure

```
pySpatial/
├── base_models/              # Downloaded model repositories
│   ├── Depth-Anything-V2/    # Depth estimation model
│   ├── sam2/                 # Segmentation model
│   └── vggt/                 # 3D reconstruction model
├── configs/
│   └── main.yaml            # Model weights configuration
├── tool/                    # Individual model interfaces
│   ├── estimate_depth.py    # Depth estimation tool
│   ├── segment.py           # Segmentation tool
│   └── recontruct.py        # 3D reconstruction tool
├── example/                 # Test images
│   └── computer/            # Sample images for testing
├── spatial_agent.py         # 🚀 Main unified interface
└── README.md               # This file
```

## 🚀 Quick Start

### For VLM Agents (Recommended)

The simplest way to use the toolkit:

```python
from spatial_agent import SpatialAgent

# Initialize the agent
agent = SpatialAgent()

# Estimate depth from image
depth_map = agent.estimate_depth("image.jpg")

# Segment objects in image
masks = agent.segment_automatic("image.jpg")

# Segment specific object at point
object_mask = agent.segment_object_at_point("image.jpg", x=100, y=200)

# 3D reconstruction from multiple images
reconstruction = agent.reconstruct_3d(["img1.jpg", "img2.jpg", "img3.jpg"])

# Complete scene analysis
scene_analysis = agent.analyze_scene_3d("path/to/images/")
```

### Individual Model Usage

```python
# Depth estimation
from tool.estimate_depth import estimate_depth
depth = estimate_depth("image.jpg")

# Segmentation
from tool.segment import segment_image, segment_automatic
masks, scores, _ = segment_image("image.jpg")
auto_segments = segment_automatic("image.jpg")

# 3D reconstruction
from tool.recontruct import reconstruct_3d
results = reconstruct_3d(["img1.jpg", "img2.jpg"])
```

## 🔧 Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda environment named `vggt`

### Model Weights
Model weights are configured in `configs/main.yaml`:

```yaml
tool:
  sam2:
    checkpoint: /path/to/sam2/weights.pt
  depthAnything2:
    checkpoint: /path/to/depth_anything_v2_weights.pth
  vggt:
    checkpoint: /path/to/vggt/model.pt
```

### Installation
1. Clone the repository with submodules for base models
2. Set up the conda environment: `conda activate vggt`
3. Install dependencies from each base model
4. Download model weights and update `configs/main.yaml`

## 🎯 Use Cases for VLM Agents

### Spatial Reasoning Tasks
```python
# Get depth of object at specific location
depth_info = agent.get_object_depth("image.jpg", x=150, y=200)
print(f"Object is {depth_info['mean_depth']:.2f} meters away")

# Analyze spatial relationships
scene = agent.analyze_scene_3d("room_images/")
```

### Object Analysis
```python
# Segment and analyze specific objects
mask = agent.segment_object_in_box("image.jpg", x1=100, y1=100, x2=300, y2=300)
depth = agent.get_depth_at_point("image.jpg", x=200, y=200)
```

### 3D Scene Understanding
```python
# Reconstruct 3D scene from multiple viewpoints
reconstruction = agent.reconstruct_3d("scene_images/")
camera_poses = reconstruction['cameras']
point_cloud = reconstruction['points']
```

## 🔍 Model Capabilities

### Depth-Anything-V2
- **Input**: Single RGB image
- **Output**: Dense depth map in meters
- **Use Case**: Understanding object distances, spatial layout

### SAM2
- **Input**: RGB image + prompts (points, boxes, or automatic)
- **Output**: Precise object masks
- **Use Case**: Object isolation, instance segmentation

### VGGT
- **Input**: Multiple RGB images
- **Output**: 3D reconstruction (cameras, depth, point clouds)
- **Use Case**: 3D scene reconstruction, novel view synthesis

## 🧠 Integration with VLMs

The toolkit is designed to be called by VLM agents with minimal code:

```python
# VLM agent code example
def analyze_spatial_scene(image_paths):
    """VLM agent function for spatial analysis"""
    agent = SpatialAgent()
    
    # Get complete 3D understanding
    results = agent.analyze_scene_3d(image_paths)
    
    # Extract key information for reasoning
    depth_info = results['depths']
    object_segments = results['segments'] 
    reconstruction_3d = results['reconstruction']
    
    return {
        'spatial_layout': depth_info,
        'objects': object_segments,
        '3d_structure': reconstruction_3d
    }
```

## 📊 Example Outputs

### Depth Estimation
- **Input**: RGB image (H×W×3)
- **Output**: Depth map (H×W) in meters

### Segmentation
- **Input**: RGB image + optional prompts
- **Output**: Binary masks, confidence scores

### 3D Reconstruction
- **Input**: Multiple images
- **Output**: Camera parameters, 3D point cloud, depth maps

## 🔬 Testing

Test the models with provided examples:

```bash
# Test individual models
python tool/estimate_depth.py
python tool/segment.py  
python tool/recontruct.py

# Test unified interface
python spatial_agent.py
```

Example images are provided in `example/computer/` for testing.

## 🤝 Contributing

This toolkit is designed to be extended and improved:

1. **Add new models**: Extend the base_models directory
2. **Improve interfaces**: Enhance the tool scripts
3. **Optimize performance**: Add caching, batching, etc.
4. **Add utilities**: Visualization, export functions, etc.

## 📝 Notes for Future Development

### Current Status
- ✅ Individual model interfaces cleaned and organized
- ✅ Unified SpatialAgent interface created
- ✅ Configuration system implemented
- ✅ Example usage documented
- 🔄 Testing with model weights (in progress)

### Planned Improvements
- [ ] Add visualization utilities
- [ ] Implement result caching
- [ ] Add batch processing support
- [ ] Create web interface demo
- [ ] Optimize memory usage
- [ ] Add more robust error handling

### VLM Integration Points
The toolkit provides multiple integration levels:
1. **Function-level**: Direct function calls for specific tasks
2. **Class-level**: SpatialAgent for persistent model loading
3. **Pipeline-level**: Complete scene analysis workflows

## 🔗 Model Sources

- **Depth-Anything-V2**: [GitHub Repository](https://github.com/DepthAnything/Depth-Anything-V2)
- **SAM2**: [GitHub Repository](https://github.com/facebookresearch/sam2)
- **VGGT**: [GitHub Repository](https://github.com/facebookresearch/vggt)

---

*This project enables VLM agents to understand and reason about 3D spatial relationships in real-world scenarios by providing simple, unified access to state-of-the-art 3D vision models.*