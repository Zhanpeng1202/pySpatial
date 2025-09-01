#!/usr/bin/env python3
"""
Test script for novel_view_synthesis tool on scene among_group693_q1_5_2
"""

import json
import numpy as np
import open3d as o3d
import os
from pySpatial_Interface import Scene, Reconstruction, pySpatial
from tool.novel_view_synthesis import rotate_right

def load_scene_data(scene_name):
    """Load scene data from processing_metadata.json"""
    metadata_path = f"/data/Datasets/MindCube/data/vggt_processed_all/{scene_name}/processing_metadata.json"
    
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def create_reconstruction_from_metadata(metadata):
    """Create a Reconstruction object from metadata"""
    
    # Load point cloud
    point_cloud_path = metadata["point_cloud_path"]
    print(f"Loading point cloud from: {point_cloud_path}")
    
    if not os.path.exists(point_cloud_path):
        raise FileNotFoundError(f"Point cloud file not found: {point_cloud_path}")
    
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    print(f"Loaded point cloud with {len(point_cloud.points)} points")
    
    # Convert extrinsics to numpy arrays (they are 3x4 in the JSON)
    extrinsics = []
    for ext in metadata["camera_poses"]["extrinsic"]:
        ext_array = np.array(ext, dtype=np.float32)
        # Convert to 4x4 matrix
        ext_4x4 = np.vstack([ext_array, [0, 0, 0, 1]])
        extrinsics.append(ext_4x4)
    
    # Convert intrinsics to numpy arrays  
    intrinsics = []
    for intr in metadata["camera_poses"]["intrinsic"]:
        intr_array = np.array(intr, dtype=np.float32)
        intrinsics.append(intr_array)
    
    print(f"Loaded {len(extrinsics)} camera poses")
    print(f"Loaded {len(intrinsics)} intrinsic matrices")
    
    # Create reconstruction object
    reconstruction = Reconstruction(point_cloud, extrinsics, intrinsics)
    
    return reconstruction

def test_novel_view_synthesis(scene_name="among_group693_q1_5_2"):
    """
    Test the novel view synthesis tool on the specified scene
    This implements the generated code:
    ```python
    def program(input_scene: Scene):
        reconstruction3D = pySpatial.reconstruct(input_scene)
        novel_viewpoint = pySpatial.rotate_right(reconstruction3D.extrinsics[1])  # rotate right from viewpoint 2
        novel_view = pySpatial.synthesize_novel_view(reconstruction3D, novel_viewpoint)
        return novel_view
    ```
    """
    
    print(f"Testing novel view synthesis on scene: {scene_name}")
    print("="*50)
    
    try:
        # Load scene data
        metadata = load_scene_data(scene_name)
        
        # Create reconstruction from metadata
        reconstruction3D = create_reconstruction_from_metadata(metadata)
        
        print("\nScene setup complete. Now testing novel view synthesis...")
        
        # Get viewpoint 2 (index 1) extrinsics - this corresponds to image 2 (left view)
        viewpoint_2_extrinsics = reconstruction3D.extrinsics[1]  # 4x4 matrix
        print(f"Original viewpoint 2 extrinsics shape: {viewpoint_2_extrinsics.shape}")
        print("Original viewpoint 2 extrinsics:")
        print(viewpoint_2_extrinsics)
        
        # Rotate right from viewpoint 2 (as per the generated code)
        novel_viewpoint = rotate_right(viewpoint_2_extrinsics)  
        print(f"\nNovel viewpoint after rotating right:")
        print(novel_viewpoint)
        
        # Use the first camera's intrinsics for rendering
        K = reconstruction3D.intrinsics[1]  # Use viewpoint 2's intrinsics
        print(f"\nUsing intrinsics from camera 2:")
        print(K)
        
        # Set output parameters
        width, height = 512, 512
        output_path = f"test_novel_view_{scene_name}.png"
        
        print(f"\nSynthesizing novel view...")
        print(f"Output resolution: {width}x{height}")
        print(f"Output path: {output_path}")
        
        # Test the novel view synthesis
        novel_view_path = pySpatial.synthesize_novel_view(
            reconstruction3D, 
            novel_viewpoint, 
            width=width, 
            height=height, 
            out_path=output_path
        )
        
        print(f"\n✓ Novel view synthesis completed!")
        print(f"Generated image saved to: {novel_view_path}")
        
        # Verify the output file exists
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ Output file verified - Size: {file_size} bytes")
        else:
            print("✗ Warning: Output file not found")
        
        return novel_view_path
        
    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def print_scene_info(scene_name="among_group693_q1_5_2"):
    """Print detailed information about the scene"""
    metadata = load_scene_data(scene_name)
    
    print(f"\n{'='*60}")
    print(f"SCENE INFORMATION: {scene_name}")
    print(f"{'='*60}")
    
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Number of images: {metadata['num_images']}")
    print(f"Point cloud path: {metadata['point_cloud_path']}")
    
    print(f"\nImage paths:")
    for i, img_path in enumerate(metadata['image_paths']):
        print(f"  Image {i+1}: {os.path.basename(img_path)}")
    
    print(f"\nCamera poses (extrinsics):")
    for i, ext in enumerate(metadata['camera_poses']['extrinsic']):
        ext_array = np.array(ext)
        print(f"  Camera {i+1} extrinsics shape: {ext_array.shape}")
    
    print(f"\nCamera intrinsics:")
    for i, intr in enumerate(metadata['camera_poses']['intrinsic']):
        intr_array = np.array(intr)
        fx, fy = intr_array[0,0], intr_array[1,1]
        cx, cy = intr_array[0,2], intr_array[1,2]
        print(f"  Camera {i+1}: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

if __name__ == "__main__":
    scene_name = "among_group693_q1_5_2"
    
    # Print scene information
    print_scene_info(scene_name)
    
    # Run the test
    result = test_novel_view_synthesis(scene_name)
    
    if result:
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print("✓ Test completed successfully")
        print(f"✓ Novel view rendered and saved to: {result}")
        print("\nThe test simulates the following generated code:")
        print("```python")
        print("def program(input_scene: Scene):")
        print("    reconstruction3D = pySpatial.reconstruct(input_scene)")
        print("    novel_viewpoint = pySpatial.rotate_right(reconstruction3D.extrinsics[1])  # rotate right from viewpoint 2")
        print("    novel_view = pySpatial.synthesize_novel_view(reconstruction3D, novel_viewpoint)")
        print("    return novel_view")
        print("```")
    else:
        print("\n✗ Test failed - check error messages above")