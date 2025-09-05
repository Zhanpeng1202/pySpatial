#!/usr/bin/env python3
"""
Test scene ID functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pySpatial_Interface import Scene, pySpatial

# Test the scene ID functionality
test_scene_id = "among_group693_q1_5_2"
dummy_images = ["other_all_image/among/shoe_216/front_007.jpg", "other_all_image/among/shoe_216/left_084.jpg"]

print(f"Creating scene with ID: {test_scene_id}")
scene = Scene(dummy_images, "Test question", scene_id=test_scene_id)

print(f"Scene ID: {scene.scene_id}")
print(f"Images: {scene.images}")

# Test reconstruction call
try:
    print("Calling pySpatial.reconstruct()...")
    result = pySpatial.reconstruct(scene)
    print("✓ Reconstruction successful!")
    print(f"Loaded from pre-computed: {result.get('loaded_from_precomputed', False)}")
    print(f"Scene name: {result.get('scene_name', 'N/A')}")
except Exception as e:
    print(f"✗ Reconstruction failed: {e}")
    import traceback
    traceback.print_exc()