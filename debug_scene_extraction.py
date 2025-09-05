#!/usr/bin/env python3
"""
Debug script to test scene name extraction
"""

test_scene = "among_07a60a723ebf3a478a81f413345b5d6da80aebd75c67e5e4e40836c09831b338_q0_1_1"
fake_images = [
    f"/fake/dataset/scenes/{test_scene}/images/img001.jpg",
    f"/fake/dataset/scenes/{test_scene}/images/img002.jpg"
]

print(f"Expected scene name: {test_scene}")
print(f"Test image paths: {fake_images}")

# Test the extraction logic
def extract_scene_name_from_path(image_paths):
    """Extract scene name from image paths."""
    if isinstance(image_paths, str):
        if image_paths.endswith('/'):
            return image_paths.rstrip('/').split('/')[-1]
        else:
            image_paths = [image_paths]
    
    # Try to extract scene name from the first image path
    first_path = image_paths[0]
    path_parts = first_path.split('/')
    
    print(f"Path parts: {path_parts}")
    
    # Look for the MindCube scene name pattern (long hash-like string)
    for i, part in enumerate(path_parts):
        print(f"Part {i}: '{part}' (len={len(part)}, has_underscore={'_' in part}, has_q={'q' in part})")
        # Check if this part looks like a MindCube scene name
        if len(part) > 50 and '_' in part and 'q' in part:
            print(f"Found scene name by pattern: {part}")
            return part
    
    # Fallback: look for any part that's longer than 20 characters with underscores
    for i, part in enumerate(path_parts):
        if len(part) > 20 and '_' in part:
            print(f"Found scene name by fallback: {part}")
            return part
    
    # Last resort: use parent directory name
    result = path_parts[-2] if len(path_parts) > 1 else path_parts[0]
    print(f"Using parent directory: {result}")
    return result

extracted = extract_scene_name_from_path(fake_images)
print(f"\nExtracted scene name: '{extracted}'")
print(f"Match: {extracted == test_scene}")