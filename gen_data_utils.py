import numpy as np
import os
from tqdm import tqdm
from scipy.spatial import KDTree
import cv2

def load_files(folder):
    """Load all .bin files from the specified folder."""
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.bin')])

def gen_spherical_depth_data(scan_folder, dst_folder, dataset, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50, pbar=None):
    """Generate high-resolution spherical depth data with progress."""
    dst_folder = os.path.join(dst_folder, 'depth')
    os.makedirs(dst_folder, exist_ok=True)
    print(f"Starting depth generation for {scan_folder} → {dst_folder}")

    scan_paths = load_files(scan_folder)
    if not scan_paths:
        print(f"Error: No .bin files found in {scan_folder}")
        return

    for idx in tqdm(range(len(scan_paths)), desc="Depth Files", leave=False, unit="file"):
        # Load LiDAR point cloud
        lidar_data = np.fromfile(scan_paths[idx], dtype=np.float32)
        if lidar_data.size % 4 != 0:
            print(f"Warning: {scan_paths[idx]} size {lidar_data.size} not divisible by 4, trimming excess")
            lidar_data = lidar_data[:-(lidar_data.size % 4)]
        lidar_data = lidar_data.reshape(-1, 4)
        print(f"Processing {scan_paths[idx]}: Reshaped to {lidar_data.shape} points")

        X, Y, Z = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2]

        # Compute depth (range) and spherical coordinates
        R_lidar = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arctan2(Y, X)
        phi = np.arcsin(Z / R_lidar)

        # Define FOV and resolution
        theta_min, theta_max = -np.pi, np.pi
        phi_min, phi_max = np.radians(fov_down), np.radians(fov_up)

        # Normalize for projection (with horizontal flip)
        u = ((1 - (theta - theta_min) / (theta_max - theta_min)) * proj_W).astype(np.int32)
        v = ((phi_max - phi) / (phi_max - phi_min) * proj_H).astype(np.int32)

        # Create depth map
        depth_map = np.zeros((proj_H, proj_W), dtype=np.float32)
        valid_indices = (u >= 0) & (u < proj_W) & (v >= 0) & (v < proj_H) & (R_lidar < max_range)
        depth_map[v[valid_indices], u[valid_indices]] = R_lidar[valid_indices]

        # Save as .npy
        npy_path = os.path.join(dst_folder, f"{idx:06d}.npy")
        np.save(npy_path, depth_map)
        print(f"✓ Depth file (npy) {idx+1}/{len(scan_paths)} generated at {npy_path} (Shape: {depth_map.shape})")

        # Save as .png (normalized for visualization)
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        png_path = os.path.join(dst_folder, f"{idx:06d}.png")
        cv2.imwrite(png_path, depth_map_normalized)
        print(f"✓ Depth file (png) {idx+1}/{len(scan_paths)} generated at {png_path}")

def gen_spherical_intensity_data(scan_folder, dst_folder, dataset, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50, pbar=None):
    """Generate high-resolution spherical intensity data with progress."""
    dst_folder = os.path.join(dst_folder, 'intensity')
    os.makedirs(dst_folder, exist_ok=True)
    print(f"Starting intensity generation for {scan_folder} → {dst_folder}")

    scan_paths = load_files(scan_folder)
    if not scan_paths:
        print(f"Error: No .bin files found in {scan_folder}")
        return

    for idx in tqdm(range(len(scan_paths)), desc="Intensity Files", leave=False, unit="file"):
        # Load LiDAR point cloud
        lidar_data = np.fromfile(scan_paths[idx], dtype=np.float32)
        if lidar_data.size % 4 != 0:
            print(f"Warning: {scan_paths[idx]} size {lidar_data.size} not divisible by 4, trimming excess")
            lidar_data = lidar_data[:-(lidar_data.size % 4)]
        lidar_data = lidar_data.reshape(-1, 4)
        print(f"Processing {scan_paths[idx]}: Reshaped to {lidar_data.shape} points")

        X, Y, Z, intensity = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2], lidar_data[:, 3]

        # Compute spherical coordinates
        R_lidar = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arctan2(Y, X)
        phi = np.arcsin(Z / R_lidar)

        # Define FOV and resolution
        theta_min, theta_max = -np.pi, np.pi
        phi_min, phi_max = np.radians(fov_down), np.radians(fov_up)

        # Normalize for projection (with horizontal flip)
        u = ((1 - (theta - theta_min) / (theta_max - theta_min)) * proj_W).astype(np.int32)
        v = ((phi_max - phi) / (phi_max - phi_min) * proj_H).astype(np.int32)

        # Create intensity map
        intensity_map = np.zeros((proj_H, proj_W), dtype=np.float32)
        valid_indices = (u >= 0) & (u < proj_W) & (v >= 0) & (v < proj_H) & (R_lidar < max_range)
        intensity_map[v[valid_indices], u[valid_indices]] = intensity[valid_indices]

        # Save as .npy
        npy_path = os.path.join(dst_folder, f"{idx:06d}.npy")
        np.save(npy_path, intensity_map)
        print(f"✓ Intensity file (npy) {idx+1}/{len(scan_paths)} generated at {npy_path} (Shape: {intensity_map.shape}, Type: {intensity_map.dtype})")

        # Save as .png (normalized for visualization)
        intensity_map_normalized = cv2.normalize(intensity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        png_path = os.path.join(dst_folder, f"{idx:06d}.png")
        cv2.imwrite(png_path, intensity_map_normalized)
        print(f"✓ Intensity file (png) {idx+1}/{len(scan_paths)} generated at {png_path}")

def gen_spherical_normal_data(scan_folder, dst_folder, dataset, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50, pbar=None):
    """Generate high-resolution spherical normal data with progress."""
    dst_folder = os.path.join(dst_folder, 'normal')
    os.makedirs(dst_folder, exist_ok=True)
    print(f"Starting normal generation for {scan_folder} → {dst_folder}")

    scan_paths = load_files(scan_folder)
    if not scan_paths:
        print(f"Error: No .bin files found in {scan_folder}")
        return

    for idx in tqdm(range(len(scan_paths)), desc="Normal Files", leave=False, unit="file"):
        # Load LiDAR point cloud
        lidar_data = np.fromfile(scan_paths[idx], dtype=np.float32)
        if lidar_data.size % 4 != 0:
            print(f"Warning: {scan_paths[idx]} size {lidar_data.size} not divisible by 4, trimming excess")
            lidar_data = lidar_data[:-(lidar_data.size % 4)]
        lidar_data = lidar_data.reshape(-1, 4)
        print(f"Processing {scan_paths[idx]}: Reshaped to {lidar_data.shape} points")

        X, Y, Z = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2]

        # Compute spherical coordinates
        R_lidar = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arctan2(Y, X)
        phi = np.arcsin(Z / R_lidar)

        # Define FOV and resolution
        theta_min, theta_max = -np.pi, np.pi
        phi_min, phi_max = np.radians(fov_down), np.radians(fov_up)

        # Normalize for projection (with horizontal flip)
        u = ((1 - (theta - theta_min) / (theta_max - theta_min)) * proj_W).astype(np.int32)
        v = ((phi_max - phi) / (phi_max - phi_min) * proj_H).astype(np.int32)

        # Create normal map
        normals = np.zeros((proj_H, proj_W, 3), dtype=np.float32)
        valid_indices = (u >= 0) & (u < proj_W) & (v >= 0) & (v < proj_H) & (R_lidar < max_range)

        # Compute normals using KD-Tree and PCA
        xyz_lidar = np.vstack((X, Y, Z)).T
        print(f"Building KD-Tree for {xyz_lidar.shape[0]} points...")
        kdtree = KDTree(xyz_lidar)

        def compute_normal(i):
            """Finds nearest neighbors and computes the normal vector."""
            _, neighbor_idx = kdtree.query(xyz_lidar[i], k=10)  # Get 10 nearest neighbors
            neighbors = xyz_lidar[neighbor_idx]

            # Compute surface normal using PCA
            mean = np.mean(neighbors, axis=0)
            cov_matrix = np.cov((neighbors - mean).T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # Use eigh for symmetric matrix
            normal_idx = np.argmin(eigenvalues)  # Index of smallest eigenvalue
            normal = eigenvectors[:, normal_idx]  # Smallest eigenvector as normal
            normal /= np.linalg.norm(normal) + 1e-6  # Normalize to unit length
            return i, (normal + 1) / 2  # Normalize to [0, 1] range

        print(f"Computing surface normals for {xyz_lidar.shape[0]} points...")
        results = [compute_normal(i) for i in tqdm(range(len(xyz_lidar)), total=len(xyz_lidar), desc="Generating Normals", unit="points", dynamic_ncols=True)]

        # Store results in normal map
        for i, normal in results:
            if valid_indices[i]:
                normals[v[i], u[i], :] = normal

        # Save as .npy
        npy_path = os.path.join(dst_folder, f"{idx:06d}.npy")
        np.save(npy_path, normals.astype(np.float32))
        print(f"✓ Normal file (npy) {idx+1}/{len(scan_paths)} generated at {npy_path} (Shape: {normals.shape})")

        # Save as .png (normalized for visualization)
        normal_map_normalized = (normals * 255).astype(np.uint8)  # Scale [0, 1] to [0, 255]
        png_path = os.path.join(dst_folder, f"{idx:06d}.png")
        cv2.imwrite(png_path, cv2.cvtColor(normal_map_normalized, cv2.COLOR_RGB2BGR))
        print(f"✓ Normal file (png) {idx+1}/{len(scan_paths)} generated at {png_path}")