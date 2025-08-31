import numpy as np
import open3d as o3d

def render_pcd_with_extrinsics(points_xyz, colors_rgb, K, E_world2cam, width, height,
                               point_size=2.0, out_path="render.png"):
    """
    points_xyz: (N,3) float32 in world coords
    colors_rgb: (N,3) float32 in [0,1] or None
    K: 3x3 intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]]
    E_world2cam: 4x4 extrinsic (world -> camera). If you have cam->world (pose), invert it.
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_xyz))
    if colors_rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_rgb, 0, 1))

    # Set up camera intrinsics
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fx, fy, cx, cy)

    # Use OffscreenRenderer for headless rendering
    renderer = o3d.visualization.rendering.OffscreenRenderer(int(width), int(height))
    
    # Add the point cloud to the scene
    renderer.scene.add_geometry("pointcloud", pcd, o3d.visualization.rendering.MaterialRecord())
    
    # Set material properties including point size
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.point_size = float(point_size)
    renderer.scene.modify_geometry_material("pointcloud", mat)
    
    # Set camera parameters
    renderer.setup_camera(intrinsic, E_world2cam)
    
    # Set background to white
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    
    # Render the image
    image = renderer.render_to_image()
    
    # Save the rendered image
    o3d.io.write_image(out_path, image)
    
    print(f"Rendered image saved to: {out_path}")
    return out_path

# Example usage:
# points_xyz = np.load("points.npy").astype(np.float32)   # (N,3)
# colors_rgb = None                                       # or (N,3) in [0,1]
# K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
# E_world2cam = np.load("E.npy").astype(np.float32)       # 4x4 world->camera
# render_pcd_with_extrinsics(points_xyz, colors_rgb, K, E_world2cam, 1024, 768, 2.5, "view.png")



# Noted that the camera pose follows opencv convention.
# we want 4 API to manipulate the camera pose.

def rotate_right(extrinsic, angle=np.pi/2):
    """
    Rotate camera right (yaw rotation around Y-axis in camera coordinates)
    Args:
        extrinsic: 3x4 or 4x4 extrinsic matrix [R|t]
        angle: rotation angle in radians (default: 90 degrees)
    """
    # Ensure we have 4x4 matrix
    if extrinsic.shape == (3, 4):
        extr_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
    else:
        extr_4x4 = extrinsic.copy()
    
    c, s = np.cos(-angle), np.sin(-angle)  # right turn (negative angle)
    # Y-axis rotation matrix (yaw)
    rot_y = np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])
    
    # Apply rotation to the extrinsic matrix
    new_extr = extr_4x4 @ rot_y
    return new_extr[:3, :] if extrinsic.shape == (3, 4) else new_extr

def rotate_left(extrinsic, angle=np.pi/2):
    """
    Rotate camera left (yaw rotation around Y-axis in camera coordinates)
    Args:
        extrinsic: 3x4 or 4x4 extrinsic matrix [R|t]
        angle: rotation angle in radians (default: 90 degrees)
    """
    # Left rotation is just negative right rotation
    return rotate_right(extrinsic, -angle)

def move_forward(extrinsic, distance=0.1):
    """
    Move camera forward in its viewing direction (negative Z-axis)
    Args:
        extrinsic: 3x4 or 4x4 extrinsic matrix [R|t]
        distance: step size (default: 0.1)
    """
    if extrinsic.shape == (3, 4):
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        # Move forward along negative Z-axis in camera coordinates
        new_t = t - distance * R[:, 2]  # Forward is -Z in camera coordinates
        return np.hstack([R, new_t.reshape(-1, 1)])
    else:
        new_extr = extrinsic.copy()
        R = new_extr[:3, :3]
        t = new_extr[:3, 3]
        new_extr[:3, 3] = t - distance * R[:, 2]
        return new_extr

def move_backward(extrinsic, distance=0.1):
    """
    Move camera backward (opposite of viewing direction)
    Args:
        extrinsic: 3x4 or 4x4 extrinsic matrix [R|t]
        distance: step size (default: 0.1)
    """
    if extrinsic.shape == (3, 4):
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        # Move backward along positive Z-axis in camera coordinates
        new_t = t + distance * R[:, 2]
        return np.hstack([R, new_t.reshape(-1, 1)])
    else:
        new_extr = extrinsic.copy()
        R = new_extr[:3, :3]
        t = new_extr[:3, 3]
        new_extr[:3, 3] = t + distance * R[:, 2]
        return new_extr

def turn_around(extrinsic):
    """
    Turn camera 180 degrees (around Y-axis)
    """
    return rotate_right(extrinsic, np.pi)

def novel_view_synthesis(reconstruction, new_camera_pose, width=512, height=512, out_path="novel_view.png"):
    """
    Main novel view synthesis function that works with pySpatial Reconstruction objects.
    
    Args:
        reconstruction: Reconstruction object with point_cloud, extrinsics, intrinsics
        new_camera_pose: 3x4 or 4x4 extrinsic matrix for the new viewpoint
        width: output image width
        height: output image height  
        out_path: output image path
        
    Returns:
        str: path to the rendered image
    """
    # Extract data from reconstruction
    point_cloud = reconstruction.point_cloud
    intrinsics = reconstruction.intrinsics
    
    # Handle different point cloud formats
    if hasattr(point_cloud, 'points') and hasattr(point_cloud, 'colors'):
        # Open3D point cloud format
        points_xyz = np.asarray(point_cloud.points)
        if len(point_cloud.colors) > 0:
            colors_rgb = np.asarray(point_cloud.colors)
        else:
            colors_rgb = None
    elif isinstance(point_cloud, dict):
        # Dictionary format
        points_xyz = point_cloud.get('points', point_cloud.get('xyz'))
        colors_rgb = point_cloud.get('colors', point_cloud.get('rgb'))
    elif isinstance(point_cloud, np.ndarray):
        # Raw numpy array (assume xyz only)
        if point_cloud.shape[1] == 3:
            points_xyz = point_cloud
            colors_rgb = None
        elif point_cloud.shape[1] == 6:
            points_xyz = point_cloud[:, :3]
            colors_rgb = point_cloud[:, 3:6]
        else:
            raise ValueError(f"Unsupported point cloud shape: {point_cloud.shape}")
    else:
        raise ValueError(f"Unsupported point cloud type: {type(point_cloud)}")
    
    # Handle intrinsics format
    if isinstance(intrinsics, dict):
        K = intrinsics.get('K', intrinsics.get('intrinsic_matrix'))
    elif isinstance(intrinsics, np.ndarray):
        K = intrinsics
    else:
        # Default intrinsics if not provided
        K = np.array([[400, 0, width/2], [0, 400, height/2], [0, 0, 1]], dtype=np.float32)
    
    # Ensure new_camera_pose is 4x4
    if new_camera_pose.shape == (3, 4):
        E_world2cam = np.vstack([new_camera_pose, [0, 0, 0, 1]])
    else:
        E_world2cam = new_camera_pose
    
    # Render the novel view
    return render_pcd_with_extrinsics(
        points_xyz, colors_rgb, K, E_world2cam, 
        width, height, out_path=out_path
    )