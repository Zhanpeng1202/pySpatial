
def program(input_scene: Scene):
    reconstruction3D = pySpatial.reconstruct(input_scene)
    camera_motion = pySpatial.describe_camera_motion(reconstruction3D)
    return camera_motion