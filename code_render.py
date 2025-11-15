
def program(input_scene: Scene):
    reconstruction3D = pySpatial.reconstruct(input_scene)
    camera_motion = pySpatial.describe_camera_motion(reconstruction3D)
    return camera_motion

def program(input_scene: Scene):
    reconstructed_scene = pySpatial.reconstruct(input_scene)
    base_viewpoint = reconstructed_scene.extrinsics[1]  # image 2 corresponds to index 1
    
    # To see what is behind, we turn the camera around (180 degrees)
    viewpoint_behind = pySpatial.turn_around(base_viewpoint)
    
    # Generate the novel view from the behind viewpoint
    image_behind = pySpatial.synthesize_novel_view(reconstructed_scene, viewpoint_behind)
    return image_behind

def program(input_scene: Scene):
    # Step 1: reconstruct the 3D scene from the input images
    reconstruction3D = pySpatial.reconstruct(input_scene)
    
    # Step 2: describe the camera motion between the two views using extrinsics
    camera_motion_description = pySpatial.describe_camera_motion(reconstruction3D)
    
    # Return the description which should match one of the answer choices
    return camera_motion_description