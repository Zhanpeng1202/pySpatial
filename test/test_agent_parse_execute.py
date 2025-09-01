#!/usr/bin/env python3
"""
Test script for Agent's parse_LLM_response and execute functionality
Tests using the first example from agent_code_generation_results_v3.json
"""

import json
import numpy as np
import open3d as o3d
import os
import sys
sys.path.append('..')
from pySpatial_Interface import Scene, Reconstruction, pySpatial, Agent
from tool.novel_view_synthesis import rotate_right


def load_test_data():
    """Load the first test case from agent_code_generation_results_v3.json"""
    json_path = "output/agent_code_generation_results_v3.json"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get the first test case
    first_result = data["results"][0]
    
    print(f"Loaded test case: {first_result['scene_id']}")
    print(f"Question: {first_result['question']}")
    print(f"Generated response length: {len(first_result['generated_response'])} chars")
    
    return first_result


def load_scene_data(scene_name):
    """Load scene data from processing_metadata.json (from test_novel_view_synthesis.py)"""
    metadata_path = f"/data/Datasets/MindCube/data/vggt_processed_all/{scene_name}/processing_metadata.json"
    
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def create_reconstruction_from_metadata(metadata):
    """Create a Reconstruction object from metadata (from test_novel_view_synthesis.py)"""
    
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


def test_agent_parse_response():
    """Test Agent's parse_LLM_response functionality"""
    print("Testing Agent.parse_LLM_response()...")
    print("="*50)
    
    # Load test data
    test_case = load_test_data()
    
    # Create agent instance
    agent = Agent()
    
    # Test parsing the generated response
    generated_response = test_case["generated_response"]
    print(f"Original response:\n{generated_response}")
    
    # Parse using agent method
    parsed_code = agent.parse_LLM_response(generated_response)
    print(f"\nParsed code:\n{parsed_code}")
    
    # Compare with expected result
    expected_parsed = test_case["parsed_code"]
    print(f"\nExpected parsed code:\n{expected_parsed}")
    
    # Check if parsing was successful
    if parsed_code == expected_parsed:
        print("\n✓ Parse test PASSED - parsed code matches expected result")
        return True, parsed_code
    else:
        print("\n✗ Parse test FAILED - parsed code doesn't match expected result")
        print("\nDifferences:")
        print(f"Actual length: {len(parsed_code)}")
        print(f"Expected length: {len(expected_parsed)}")
        return False, parsed_code


def test_agent_execute():
    """Test Agent's execute functionality"""
    print("\n\nTesting Agent.execute()...")
    print("="*50)
    
    # Load test data and get parsed code
    test_case = load_test_data()
    scene_name = test_case["scene_id"]
    
    # Create agent instance
    agent = Agent()
    
    # Parse the response first
    parsed_code = agent.parse_LLM_response(test_case["generated_response"])
    print(f"Using parsed code:\n{parsed_code}")
    
    try:
        # Execute the code to get the program function
        print("\nExecuting code to get program function...")
        program_func = agent.execute(parsed_code)
        print(f"✓ Successfully executed code and got program function: {program_func}")
        
        # Create a Scene object for testing
        print(f"\nCreating Scene object for {scene_name}...")
        
        # Load actual scene data
        metadata = load_scene_data(scene_name)
        reconstruction3D = create_reconstruction_from_metadata(metadata)
        
        # Create Scene object (we'll use the image paths from metadata)
        scene = Scene(metadata["image_paths"])
        # Manually set the reconstruction since we loaded it from metadata
        scene.reconstruction = reconstruction3D
        
        print(f"✓ Scene created with {len(scene.images)} images")
        
        # Test running the program function
        print("\nRunning the program function...")
        
        # Mock the pySpatial.reconstruct to return our loaded reconstruction
        # Since the program expects to call pySpatial.reconstruct(input_scene)
        original_reconstruct = pySpatial.reconstruct
        pySpatial.reconstruct = lambda scene: reconstruction3D
        
        try:
            result = program_func(scene)
            print(f"✓ Program function executed successfully!")
            print(f"Result: {result}")
            
            # Verify the result is a path to an image file
            if isinstance(result, str) and result.endswith('.png'):
                if os.path.exists(result):
                    file_size = os.path.getsize(result)
                    print(f"✓ Generated image file verified - Size: {file_size} bytes")
                else:
                    print(f"✗ Warning: Generated image file not found: {result}")
            
            return True, result
            
        finally:
            # Restore original reconstruct method
            pySpatial.reconstruct = original_reconstruct
            
    except Exception as e:
        print(f"✗ Execute test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Run all tests"""
    print("="*60)
    print("AGENT PARSE AND EXECUTE TESTING")
    print("="*60)
    
    # Test parsing
    parse_success, parsed_code = test_agent_parse_response()
    
    # Test execution
    execute_success, result = test_agent_execute()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if parse_success:
        print("✓ Parse LLM Response: PASSED")
    else:
        print("✗ Parse LLM Response: FAILED")
    
    if execute_success:
        print("✓ Code Execution: PASSED")
        print(f"✓ Program result: {result}")
    else:
        print("✗ Code Execution: FAILED")
    
    if parse_success and execute_success:
        print("\n🎉 All tests PASSED! Agent parsing and execution working correctly.")
        return True
    else:
        print("\n❌ Some tests FAILED - check error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)