import os
import numpy as np
import yaml

# Simplified load_files function with debugging
def load_files(folder):
    """Load all files from a folder with debugging."""
    if not os.path.exists(folder):
        print(f"Error: Folder {folder} does not exist!")
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.bin')]
    print(f"Found {len(files)} .bin files in {folder}: {files}")
    return files

def load_calibration(filepath):
    """Load KITTI calibration data dynamically."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    calib_dict = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 1:
            key = parts[0].strip(":")
            values = np.array(parts[1:], dtype=np.float32)
            calib_dict[key] = values
    
    P2 = calib_dict["P2"].reshape(3, 4)  # Camera projection matrix
    T_velo_cam = calib_dict["Tr"].reshape(3, 4)  # LiDAR to camera transform
    T_velo_cam = np.vstack((T_velo_cam, [0, 0, 0, 1]))
    T_cam_velo = np.linalg.inv(T_velo_cam)  # Camera to LiDAR transform
    
    fx = P2[0, 0]
    fy = P2[1, 1]
    cx = P2[0, 2]
    cy = P2[1, 2]
    
    print(f"P2: {P2}")
    print(f"T_velo_cam: {T_velo_cam}")
    
    return P2, T_velo_cam, T_cam_velo, fx, fy, cx, cy

def range_projection(current_vertex, P2, T_velo_cam, proj_H=64, proj_W=900, max_range=50, img_w=1242, img_h=375):
    """ Project a pointcloud into a camera-frame projection, range image, aligned with image dimensions.
        Args:
        current_vertex: raw point clouds
        P2: camera projection matrix
        T_velo_cam: LiDAR to camera transform
        proj_H: projection height
        proj_W: projection width
        max_range: maximum range for depth
        img_w: RGB image width (for scaling)
        img_h: RGB image height (for scaling)
        Returns: 
        proj_range: projected range image with depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_idx: each pixel contains the corresponding index
    """
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    initial_mask = (depth > 0) & (depth < max_range)
    current_vertex = current_vertex[initial_mask]
    depth = depth[initial_mask]
    print(f"Initial points after depth filtering: {current_vertex.shape[0]}")

    # Transform LiDAR points to camera frame
    homogeneous = np.hstack((current_vertex[:, :3], np.ones((current_vertex.shape[0], 1))))
    cam_points = (T_velo_cam @ homogeneous.T).T[:, :3]

    # Filter points in front of the camera (z > 0)
    valid_mask = cam_points[:, 2] > 0
    cam_points = cam_points[valid_mask]
    depth = depth[valid_mask]  # Update depth with the same mask
    print(f"Camera points after z > 0 filtering: {cam_points.shape[0]}")

    # Project to 2D camera coordinates
    homogeneous = np.hstack((cam_points, np.ones((cam_points.shape[0], 1))))
    uv = (P2 @ homogeneous.T).T
    uv = uv[:, :2] / uv[:, 2:3]  # Normalize by z

    # Map uv coordinates to 64x900 grid, ensuring principal point (cx, cy) maps to (proj_H/2, proj_W/2)
    cx, cy = P2[0, 2], P2[1, 2]
    proj_x = (uv[:, 0] / img_w) * proj_W  # Scale to [0, proj_W]
    proj_y = (uv[:, 1] / img_h) * proj_H  # Scale to [0, proj_H]

    # Adjust to center on (proj_H/2, proj_W/2)
    proj_x = proj_x - (cx / img_w) * proj_W + proj_W / 2
    proj_y = proj_y - (cy / img_h) * proj_H + proj_H / 2

    # Round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)

    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]
    scan_x = cam_points[:, 0][order]
    scan_y = cam_points[:, 1][order]
    scan_z = cam_points[:, 2][order]
    indices = np.arange(depth.shape[0])[order]

    proj_range = np.full((proj_H, proj_W), -1, dtype=np.float32)
    proj_vertex = np.full((proj_H, proj_W, 4), -1, dtype=np.float32)
    proj_idx = np.full((proj_H, proj_W), -1, dtype=np.int32)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices

    return proj_range, proj_vertex, proj_idx

def gen_depth_data(scan_folder, dst_folder, P2, T_velo_cam, proj_H=64, proj_W=900, max_range=50, img_w=1242, img_h=375):
    """ Generate projected range data in the camera frame in the shape of (64, 900, 1).
        The input raw data are in the shape of (Num_points, 4).
    """
    # Specify the goal folder
    dst_folder = os.path.join(dst_folder, 'depth')
    try:
        os.stat(dst_folder)
        print('Generating depth data in:', dst_folder)
    except:
        print('Creating new depth folder:', dst_folder)
        os.mkdir(dst_folder)

    # Load LiDAR scan files
    scan_paths = load_files(scan_folder)

    depths = []

    # Iterate over all scan files
    for idx in range(len(scan_paths)):
        # Load a point cloud
        current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32)
        remains = current_vertex.shape[0] % 4
        if remains != 0:
            print(f"Warning: Truncating {remains} points from {scan_paths[idx]}")
            current_vertex = current_vertex[:current_vertex.shape[0] - remains].reshape((-1, 4))
        else:
            current_vertex = current_vertex.reshape((-1, 4))
        print(f"Loaded {current_vertex.shape[0]} points from {scan_paths[idx]}")

        # Generate depth image in camera frame
        proj_range, proj_vertex, proj_idx = range_projection(current_vertex, P2, T_velo_cam, proj_H, proj_W, max_range, img_w=img_w, img_h=img_h)

        # Generate the destination path
        dst_path = os.path.join(dst_folder, str(idx).zfill(6) + ".npy")

        # Save the depth image as .npy
        np.save(dst_path, proj_range)
        depths.append(proj_range)
        print('Finished generating depth data at:', dst_path)

    return depths

if __name__ == "__main__":
    # Load config file
    config_filename = '/home/kavi/Fusion/lorcon_lo/config/config.yml'
    if not os.path.exists(config_filename):
        print(f"Error: Config file {config_filename} not found!")
        exit(1)
    config = yaml.load(open(config_filename), yaml.Loader)

    data_seqs = ["04"]  # Process only sequence 04

    for seq in data_seqs:
        # Set the related parameters
        scan_folder = os.path.join(config["scan_folder"], seq, "velodyne")  # Adjusted for KITTI structure
        dst_folder = os.path.join(config["depth_preprocessed_folder"], seq)
        
        os.makedirs(scan_folder, exist_ok=True)
        os.makedirs(dst_folder, exist_ok=True)

        # Load calibration for sequence 04
        calib_file = os.path.join(config["calib_folder"], seq + ".txt")
        if not os.path.exists(calib_file):
            print(f"Error: Calibration file {calib_file} not found!")
            exit(1)
        P2, T_velo_cam, T_cam_velo, fx, fy, cx, cy = load_calibration(calib_file)

        proj_H = config["proj_H"]
        proj_W = config["proj_W"]
        max_range = config["max_range"]
        
        dataset = config["dataset"]
        
        # Generate depth data in camera frame for sequence 04
        gen_depth_data(scan_folder, dst_folder, P2, T_velo_cam, proj_H, proj_W, max_range)