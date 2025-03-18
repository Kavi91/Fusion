import numpy as np
import os
from tqdm import tqdm
from scipy.spatial import KDTree
import cv2

def load_files(folder, extensions=('.bin', '.png')):
    """Load files from the specified folder with given extensions."""
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(extensions)]
    #print(f"Found {len(files)} files with {extensions} in {folder}")
    return sorted(files)

def gen_spherical_depth_data(scan_folder, rgb_folder, dst_folder, dataset, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50, pbar=None):
    """Generate spherical depth data with progress."""
    dst_folder = os.path.join(dst_folder, 'depth')
    os.makedirs(dst_folder, exist_ok=True)
    #print(f"Starting depth generation for {scan_folder} → {dst_folder}")

    scan_paths = [p for p in load_files(scan_folder, ('.bin',)) if p.endswith('.bin')]  # Only .bin files
    rgb_paths = load_files(rgb_folder, ('.png',))  # Only .png files
    if not scan_paths:
        print(f"Error: No .bin files found in {scan_folder}")
        return
    if not rgb_paths:
        print(f"Warning: No .png files found in {rgb_folder}, RGB generation may be affected")

    for idx in tqdm(range(min(len(scan_paths), len(rgb_paths))), desc="Depth Files", leave=False, unit="file"):
        # Load LiDAR point cloud
        lidar_data = np.fromfile(scan_paths[idx], dtype=np.float32)
        if lidar_data.size % 4 != 0:
            print(f"Warning: {scan_paths[idx]} size {lidar_data.size} not divisible by 4, trimming excess")
            lidar_data = lidar_data[:-(lidar_data.size % 4)]
        lidar_data = lidar_data.reshape(-1, 4)
        #print(f"Processing {scan_paths[idx]}: Reshaped to {lidar_data.shape} points")

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
        #print(f"✓ Depth file (npy) {idx+1}/{min(len(scan_paths), len(rgb_paths))} generated at {npy_path} (Shape: {depth_map.shape})")

        # Save as .png with error handling
        try:
            depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if np.any(np.isnan(depth_map_normalized)) or np.any(np.isinf(depth_map_normalized)):
                print(f"Warning: NaN or Inf values detected in depth_map_normalized for {idx}")
            png_path = os.path.join(dst_folder, f"{idx:06d}.png")
            success = cv2.imwrite(png_path, depth_map_normalized)
            if success:
                print(f"✓ Depth file (png) {idx+1}/{min(len(scan_paths), len(rgb_paths))} generated at {png_path}")
            else:
                print(f"Failed to save depth.png for {idx}")
        except Exception as e:
            #print(f"Error generating depth.png for {idx}: {str(e)}")

def gen_spherical_intensity_data(scan_folder, rgb_folder, dst_folder, dataset, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50, pbar=None):
    """Generate spherical intensity data with progress."""
    dst_folder = os.path.join(dst_folder, 'intensity')
    os.makedirs(dst_folder, exist_ok=True)
    #print(f"Starting intensity generation for {scan_folder} → {dst_folder}")

    scan_paths = [p for p in load_files(scan_folder, ('.bin',)) if p.endswith('.bin')]  # Only .bin files
    rgb_paths = load_files(rgb_folder, ('.png',))  # Only .png files
    if not scan_paths:
        print(f"Error: No .bin files found in {scan_folder}")
        return
    if not rgb_paths:
        print(f"Warning: No .png files found in {rgb_folder}, RGB generation may be affected")

    for idx in tqdm(range(min(len(scan_paths), len(rgb_paths))), desc="Intensity Files", leave=False, unit="file"):
        # Load LiDAR point cloud
        lidar_data = np.fromfile(scan_paths[idx], dtype=np.float32)
        if lidar_data.size % 4 != 0:
            print(f"Warning: {scan_paths[idx]} size {lidar_data.size} not divisible by 4, trimming excess")
            lidar_data = lidar_data[:-(lidar_data.size % 4)]
        lidar_data = lidar_data.reshape(-1, 4)
        #print(f"Processing {scan_paths[idx]}: Reshaped to {lidar_data.shape} points")

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
        #print(f"✓ Intensity file (npy) {idx+1}/{min(len(scan_paths), len(rgb_paths))} generated at {npy_path} (Shape: {intensity_map.shape}, Type: {intensity_map.dtype})")

        # Save as .png with error handling
        try:
            intensity_map_normalized = cv2.normalize(intensity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if np.any(np.isnan(intensity_map_normalized)) or np.any(np.isinf(intensity_map_normalized)):
                print(f"Warning: NaN or Inf values detected in intensity_map_normalized for {idx}")
            png_path = os.path.join(dst_folder, f"{idx:06d}.png")
            success = cv2.imwrite(png_path, intensity_map_normalized)
            if success:
                print(f"✓ Intensity file (png) {idx+1}/{min(len(scan_paths), len(rgb_paths))} generated at {png_path}")
            else:
                #print(f"Failed to save intensity.png for {idx}")
        except Exception as e:
            #print(f"Error generating intensity.png for {idx}: {str(e)}")

def gen_spherical_normal_data(scan_folder, rgb_folder, dst_folder, dataset, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50, pbar=None):
    """Generate spherical normal data with progress."""
    dst_folder = os.path.join(dst_folder, 'normal')
    os.makedirs(dst_folder, exist_ok=True)
    #print(f"Starting normal generation for {scan_folder} → {dst_folder}")

    scan_paths = [p for p in load_files(scan_folder, ('.bin',)) if p.endswith('.bin')]  # Only .bin files
    rgb_paths = load_files(rgb_folder, ('.png',))  # Only .png files
    if not scan_paths:
        print(f"Error: No .bin files found in {scan_folder}")
        return
    if not rgb_paths:
        #print(f"Warning: No .png files found in {rgb_folder}, RGB generation may be affected")

    for idx in tqdm(range(min(len(scan_paths), len(rgb_paths))), desc="Normal Files", leave=False, unit="file"):
        # Load LiDAR point cloud
        lidar_data = np.fromfile(scan_paths[idx], dtype=np.float32)
        if lidar_data.size % 4 != 0:
            print(f"Warning: {scan_paths[idx]} size {lidar_data.size} not divisible by 4, trimming excess")
            lidar_data = lidar_data[:-(lidar_data.size % 4)]
        lidar_data = lidar_data.reshape(-1, 4)
        #print(f"Processing {scan_paths[idx]}: Reshaped to {lidar_data.shape} points")

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
        #print(f"Building KD-Tree for {xyz_lidar.shape[0]} points...")
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

        #print(f"Computing surface normals for {xyz_lidar.shape[0]} points...")
        results = [compute_normal(i) for i in tqdm(range(len(xyz_lidar)), total=len(xyz_lidar), desc="Generating Normals", unit="points", dynamic_ncols=True)]

        # Store results in normal map
        for i, normal in results:
            if valid_indices[i]:
                normals[v[i], u[i], :] = normal

        # Save as .npy
        npy_path = os.path.join(dst_folder, f"{idx:06d}.npy")
        np.save(npy_path, normals.astype(np.float32))
        #print(f"✓ Normal file (npy) {idx+1}/{min(len(scan_paths), len(rgb_paths))} generated at {npy_path} (Shape: {normals.shape})")

        # Save as .png with error handling
        try:
            normal_map_normalized = (normals * 255).astype(np.uint8)  # Scale [0, 1] to [0, 255]
            if np.any(np.isnan(normal_map_normalized)) or np.any(np.isinf(normal_map_normalized)):
                print(f"Warning: NaN or Inf values detected in normal_map_normalized for {idx}")
            png_path = os.path.join(dst_folder, f"{idx:06d}.png")
            success = cv2.imwrite(png_path, cv2.cvtColor(normal_map_normalized, cv2.COLOR_RGB2BGR))
            if success:
                print(f"✓ Normal file (png) {idx+1}/{min(len(scan_paths), len(rgb_paths))} generated at {png_path}")
            else:
                #print(f"Failed to save normal.png for {idx}")
        except Exception as e:
            #print(f"Error generating normal.png for {idx}: {str(e)}")

def gen_spherical_rgb_data(scan_folder, rgb_folder, dst_folder, dataset, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50, pbar=None):
    """Generate spherical RGB data with progress."""
    dst_folder = os.path.join(dst_folder, 'rgb')
    os.makedirs(dst_folder, exist_ok=True)
    #print(f"Starting RGB generation for {scan_folder} → {dst_folder}")

    scan_paths = [p for p in load_files(scan_folder, ('.bin',)) if p.endswith('.bin')]  # Only .bin files
    rgb_paths = load_files(rgb_folder, ('.png',))  # Only .png files from rgb_folder
    if not scan_paths:
        print(f"Error: No .bin files found in {scan_folder}")
        return
    if not rgb_paths:
        print(f"Error: No .png files found in {rgb_folder}")
        return

    for idx in tqdm(range(min(len(scan_paths), len(rgb_paths))), desc="RGB Files", leave=False, unit="file"):
        # Load LiDAR point cloud for projection reference
        lidar_data = np.fromfile(scan_paths[idx], dtype=np.float32)
        if lidar_data.size % 4 != 0:
            print(f"Warning: {scan_paths[idx]} size {lidar_data.size} not divisible by 4, trimming excess")
            lidar_data = lidar_data[:-(lidar_data.size % 4)]
        lidar_data = lidar_data.reshape(-1, 4)
        #print(f"Processing {scan_paths[idx]}: Reshaped to {lidar_data.shape} points")

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

        # Load and process RGB image
        rgb_image = cv2.imread(rgb_paths[idx])
        if rgb_image is None:
            print(f"Error: Failed to load RGB image at {rgb_paths[idx]}")
            continue
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_image.shape

        # Create RGB spherical map
        rgb_map = np.zeros((proj_H, proj_W, 3), dtype=np.float32)
        valid_indices = (u >= 0) & (u < proj_W) & (v >= 0) & (v < proj_H) & (R_lidar < max_range)

        # Project RGB to spherical grid with bilinear interpolation
        for i in range(len(u)):
            if valid_indices[i]:
                x = (u[i] / proj_W) * w
                y = (v[i] / proj_H) * h
                x0, y0 = int(x), int(y)
                x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
                dx, dy = x - x0, y - y0

                if x0 >= 0 and y0 >= 0 and x1 < w and y1 < h:
                    rgb_val = (rgb_image[y0, x0] * (1 - dx) * (1 - dy) +
                              rgb_image[y0, x1] * dx * (1 - dy) +
                              rgb_image[y1, x0] * (1 - dx) * dy +
                              rgb_image[y1, x1] * dx * dy)
                    rgb_map[v[i], u[i], :] = rgb_val / 255.0  # Normalize to [0, 1]

        # Save as .npy
        npy_path = os.path.join(dst_folder, f"{idx:06d}.npy")
        np.save(npy_path, rgb_map)
        #print(f"✓ RGB file (npy) {idx+1}/{min(len(scan_paths), len(rgb_paths))} generated at {npy_path} (Shape: {rgb_map.shape})")

        # Save as .png with error handling
        try:
            rgb_map_normalized = (rgb_map * 255).astype(np.uint8)
            if np.any(np.isnan(rgb_map_normalized)) or np.any(np.isinf(rgb_map_normalized)):
                print(f"Warning: NaN or Inf values detected in rgb_map_normalized for {idx}")
            png_path = os.path.join(dst_folder, f"{idx:06d}.png")
            success = cv2.imwrite(png_path, cv2.cvtColor(rgb_map_normalized, cv2.COLOR_RGB2BGR))
            if success:
                print(f"✓ RGB file (png) {idx+1}/{min(len(scan_paths), len(rgb_paths))} generated at {png_path}")
            else:
                #print(f"Failed to save RGB.png for {idx}")
        except Exception as e:
            #print(f"Error generating RGB.png for {idx}: {str(e)}")