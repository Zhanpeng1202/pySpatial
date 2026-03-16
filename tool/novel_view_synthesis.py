import os
import numpy as np

# Set up headless rendering environment before importing Open3D
if "XDG_RUNTIME_DIR" not in os.environ:
    xdg_dir = f"/tmp/runtime-{os.getenv('USER', 'default')}"
    os.makedirs(xdg_dir, mode=0o700, exist_ok=True)
    os.environ["XDG_RUNTIME_DIR"] = xdg_dir
os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")

import open3d as o3d

def zoom_out_K(K, scale=0.5):
    """Scale focal lengths by `scale` (e.g., 0.5 zooms out)."""
    K_new = K.copy().astype(np.float32)
    K_new[0, 0] *= float(scale)  # fx
    K_new[1, 1] *= float(scale)  # fy
    return K_new

def render_pcd_with_extrinsics(points_xyz, colors_rgb, K, E_world2cam, width, height,
                               point_size=2.0, out_path=None, zoom_out_scale=0.5):
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

    # Apply zoom out to enlarge field of view
    K_zoomed = zoom_out_K(K, zoom_out_scale)
    
    # Set up camera intrinsics (cast to float for Open3D compatibility)
    fx, fy = float(K_zoomed[0,0]), float(K_zoomed[1,1])
    cx, cy = float(K_zoomed[0,2]), float(K_zoomed[1,2])
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
    
    # Save the rendered image only if out_path is provided
    if out_path is not None:
        o3d.io.write_image(out_path, image)
        print(f"Rendered image saved to: {out_path}")
        return out_path
    else:
        return image

# Example usage:
# points_xyz = np.load("points.npy").astype(np.float32)   # (N,3)
# colors_rgb = None                                       # or (N,3) in [0,1]
# K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
# E_world2cam = np.load("E.npy").astype(np.float32)       # 4x4 world->camera
# render_pcd_with_extrinsics(points_xyz, colors_rgb, K, E_world2cam, 1024, 768, 2.5, "view.png")



# Noted that the camera pose follows opencv convention.
# We use the average look-at direction of input frames as the rotation axis,
# which approximates a "gravity" axis without needing explicit gravity info.

# ---- helpers to compute rotation axis from camera extrinsics ----

def extract_look_at_direction(extrinsic):
    """Extract the look-at direction (forward vector) from a camera extrinsic matrix."""
    M = np.vstack([extrinsic, [0, 0, 0, 1]]) if extrinsic.shape == (3, 4) else extrinsic
    R = M[:3, :3]
    look_at = R[:, 2]  # camera looks along +Z in camera coords
    return look_at / np.linalg.norm(look_at)


def average_look_at_directions(extrinsics):
    """Compute the average look-at direction from multiple camera extrinsics."""
    directions = np.array([extract_look_at_direction(e) for e in extrinsics])
    avg = np.mean(directions, axis=0)
    avg = -avg / np.linalg.norm(avg)
    return avg


def _rotation_matrix_around_axis(axis, angle):
    """Rodrigues' rotation: return a 4x4 rotation matrix around an arbitrary axis."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    M = np.eye(4)
    M[:3, :3] = R
    return M


# ---- camera manipulation functions ----

def rotate_right(extrinsic, angle=np.pi/2, axis=None):
    """
    Rotate camera right around the given axis.
    Args:
        extrinsic: 3x4 or 4x4 extrinsic matrix [R|t]
        angle: rotation angle in radians (default: 90 degrees)
        axis: 3D rotation axis vector. If None, falls back to Y-axis [0,1,0].
    """
    if axis is None:
        axis = np.array([0.0, 1.0, 0.0])

    if extrinsic.shape == (3, 4):
        extr_4x4 = np.vstack([extrinsic, [0, 0, 0, 1]])
    else:
        extr_4x4 = extrinsic.copy()

    rot = _rotation_matrix_around_axis(axis, -angle)  # right = negative angle
    new_extr = rot @ extr_4x4
    return new_extr[:3, :] if extrinsic.shape == (3, 4) else new_extr


def rotate_left(extrinsic, angle=np.pi/2, axis=None):
    """
    Rotate camera left around the given axis.
    Args:
        extrinsic: 3x4 or 4x4 extrinsic matrix [R|t]
        angle: rotation angle in radians (default: 90 degrees)
        axis: 3D rotation axis vector. If None, falls back to Y-axis [0,1,0].
    """
    return rotate_right(extrinsic, -angle, axis=axis)


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
        new_t = t - distance * R[:, 2]
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
        new_t = t + distance * R[:, 2]
        return np.hstack([R, new_t.reshape(-1, 1)])
    else:
        new_extr = extrinsic.copy()
        R = new_extr[:3, :3]
        t = new_extr[:3, 3]
        new_extr[:3, 3] = t + distance * R[:, 2]
        return new_extr


def turn_around(extrinsic, axis=None):
    """Turn camera 180 degrees around the given axis."""
    return rotate_right(extrinsic, np.pi, axis=axis)

def novel_view_synthesis(reconstruction, new_camera_pose, width=512, height=512, out_path=None):
    """
    Main novel view synthesis function that works with pySpatial Reconstruction objects.

    Args:
        reconstruction: Reconstruction object with point_cloud, extrinsics, intrinsics
        new_camera_pose: 3x4 or 4x4 extrinsic matrix for the new viewpoint
        width: output image width
        height: output image height
        out_path: output image path (optional, if None returns image object directly)

    Returns:
        str or image: path to the rendered image if out_path provided, otherwise image object
    """
    # Extract data from reconstruction
    point_cloud = reconstruction.point_cloud
    intrinsics = reconstruction.intrinsics

    # Check if reconstruction has colors attribute
    colors_rgb = None
    if hasattr(reconstruction, 'colors') and reconstruction.colors is not None:
        colors_rgb = reconstruction.colors
        if colors_rgb.max() > 1.0:
            colors_rgb = colors_rgb / 255.0

    # Handle different point cloud formats
    if hasattr(point_cloud, 'points') and hasattr(point_cloud, 'colors'):
        # Open3D point cloud format
        points_xyz = np.asarray(point_cloud.points)
        if colors_rgb is None and len(point_cloud.colors) > 0:
            colors_rgb = np.asarray(point_cloud.colors)
    elif isinstance(point_cloud, dict):
        points_xyz = point_cloud.get('points', point_cloud.get('xyz'))
        if colors_rgb is None:
            colors_rgb = point_cloud.get('colors', point_cloud.get('rgb'))
    elif isinstance(point_cloud, np.ndarray):
        if point_cloud.shape[1] == 3:
            points_xyz = point_cloud
        elif point_cloud.shape[1] == 6:
            points_xyz = point_cloud[:, :3]
            if colors_rgb is None:
                colors_rgb = point_cloud[:, 3:6]
        else:
            raise ValueError(f"Unsupported point cloud shape: {point_cloud.shape}")
    elif hasattr(point_cloud, 'cpu') or hasattr(point_cloud, 'numpy'):
        point_cloud_np = point_cloud.cpu().numpy() if hasattr(point_cloud, 'cpu') else point_cloud.numpy()
        if point_cloud_np.shape[1] == 3:
            points_xyz = point_cloud_np
        elif point_cloud_np.shape[1] == 6:
            points_xyz = point_cloud_np[:, :3]
            if colors_rgb is None:
                colors_rgb = point_cloud_np[:, 3:6]
        else:
            raise ValueError(f"Unsupported point cloud shape: {point_cloud_np.shape}")
    else:
        raise ValueError(f"Unsupported point cloud type: {type(point_cloud)}")

    # Handle intrinsics format
    if isinstance(intrinsics, dict):
        K = intrinsics.get('K', intrinsics.get('intrinsic_matrix'))
    elif isinstance(intrinsics, np.ndarray):
        K = intrinsics
    else:
        K = np.array([[400, 0, width/2], [0, 400, height/2], [0, 0, 1]], dtype=np.float32)

    # If K is a batch (N, 3, 3), use the first one
    if isinstance(K, np.ndarray) and K.ndim == 3:
        K = K[0]

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
    
    
