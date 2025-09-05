#!/usr/bin/env python3
"""
Simple test for reconstruction tool with specific MindCube scene.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tool.recontruct import reconstruct_3d

def main():
    print("=== Testing Reconstruction with Specific Scene ===")
    
    # Use the specific scene name
    test_scene = "among_07a60a723ebf3a478a81f413345b5d6da80aebd75c67e5e4e40836c09831b338_q0_1_1"
    print(f"Scene: {test_scene}")
    
    # Create fake image paths that contain the scene name
    # The tool should extract the scene name from these paths
    fake_images = [
        f"/fake/dataset/scenes/{test_scene}/images/img001.jpg",
        f"/fake/dataset/scenes/{test_scene}/images/img002.jpg"
    ]
    
    print(f"Input images: {fake_images}")
    
    try:
        print("\n--- Testing with pre-computed data (default) ---")
        result = reconstruct_3d(fake_images, use_precomputed=True)
        
        print("✓ Reconstruction successful!")
        print(f"Loaded from pre-computed: {result.get('loaded_from_precomputed', False)}")
        print(f"Detected scene name: {result.get('scene_name', 'N/A')}")
        print(f"Number of input images: {result.get('num_images', 'N/A')}")
        
        # Check camera data
        if result.get('cameras') is not None:
            print(f"Camera data shape: {result['cameras'].shape}")
            print(f"Camera data type: {type(result['cameras'])}")
        else:
            print("No camera data found")
        
        # Check point cloud
        if result.get('points') is not None:
            print(f"Point cloud shape: {result['points'].shape}")
            print(f"Point cloud type: {type(result['points'])}")
            
            # Show first few points
            points = result['points']
            num_points_to_show = min(5, len(points))
            print(f"First {num_points_to_show} points:")
            for i in range(num_points_to_show):
                point = points[i]
                print(f"  Point {i}: [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]")
        else:
            print("No point cloud found")
        
        # Check metadata
        metadata = result.get('metadata', {})
        if metadata:
            print(f"Metadata keys: {list(metadata.keys())}")
            # Show some metadata info if available
            for key in ['cameras', 'extrinsics', 'intrinsics']:
                if key in metadata:
                    print(f"  {key}: {type(metadata[key])}")
        else:
            print("No metadata found")
            
        return True
        
    except Exception as e:
        print(f"✗ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ Test completed successfully!")
        print("\nThe reconstruction tool is ready to use with MindCube data.")
        print("It will automatically load pre-computed data when available.")
    else:
        print("\n✗ Test failed!")