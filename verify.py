import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Paths
calib_file = "/home/kavi/Datasets/KITTI_raw/kitti_data/calib/04.txt"
lidar_file = "/home/kavi/Datasets/KITTI_raw/kitti_data/scan/04/velodyne/0000000000.bin"
rgb_file = "/home/kavi/Datasets/KITTI_raw/kitti_data/sequences/04/image_02/0000000000.png"
output_dir =  "/home/kavi/Fusion/Projections" # Save images in the same folder

# **1. Load KITTI Calibration File**
def load_calib(filepath):
    """Parse KITTI calibration file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    calib = {}
    for line in lines:
        key, values = line.split(":")
        calib[key] = np.array([float(v) for v in values.split()]).reshape(3, 4)
    
    return calib

calib = load_calib(calib_file)
P2 = calib["P2"]  # Projection matrix for KITTI RGB camera
Tr = calib["Tr"]  # LiDAR to camera transformation

# Extract rotation (R) and translation (t) from `Tr`
R = Tr[:, :3]  # 3x3 rotation
t = Tr[:, 3:].reshape(3, 1)  # 3x1 translation

# **2. Load LiDAR Point Cloud**
lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)  # (X, Y, Z, Reflectance)

# **3. Convert LiDAR Points to Spherical Coordinates (θ, φ, R)**
X, Y, Z = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]
R_lidar = np.sqrt(X**2 + Y**2 + Z**2)  # Depth (Range)
theta = np.arctan2(Y, X)  # Azimuth angle
phi = np.arcsin(Z / R_lidar)  # Elevation angle

# **4. Define KITTI RGB Resolution**
width = 1226  # Match KITTI RGB width
height = 370  # Match KITTI RGB height

# **5. Normalize Spherical Depth Map for Projection**
theta_min, theta_max = -np.pi, np.pi  # Full 360° azimuth range
phi_min, phi_max = np.radians(-25), np.radians(3)  # LiDAR vertical FOV

u = ((1 - (theta - theta_min) / (theta_max - theta_min)) * width).astype(np.int32)  # Fixed Horizontal Flip
v = ((phi_max - phi) / (phi_max - phi_min) * height).astype(np.int32)  # Fixed Vertical Mapping

# **6. Create Spherical Depth Map**
spherical_depth_map = np.zeros((height, width), dtype=np.float32)
valid_indices = (u >= 0) & (u < width) & (v >= 0) & (v < height)
spherical_depth_map[v[valid_indices], u[valid_indices]] = R_lidar[valid_indices]  # Assign depth values

# **7. Transform Spherical Depth Points to RGB Camera Frame**
xyz_lidar = np.vstack((X, Y, Z))  # (3, N)
xyz_cam = np.dot(R, xyz_lidar) + t  # Apply LiDAR-to-Camera transformation

valid_idx = xyz_cam[2, :] > 0  # Only keep points in front of the camera
xyz_cam = xyz_cam[:, valid_idx]
depth_values = xyz_cam[2, :]

# **8. Project Depth Points onto KITTI RGB Camera Plane**
projected_2d = P2 @ np.vstack((xyz_cam, np.ones((1, xyz_cam.shape[1]))))  # Homogeneous coordinates
projected_2d /= projected_2d[2]  # Normalize by depth

u_proj = projected_2d[0].astype(int)
v_proj = projected_2d[1].astype(int)

# **9. Load RGB Image**
rgb_image = cv2.imread(rgb_file)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

h, w, _ = rgb_image.shape

# **10. Filter Points Inside Image Bounds**
valid_idx = (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h)
u_proj, v_proj = u_proj[valid_idx], v_proj[valid_idx]
depth_values = depth_values[valid_idx]

# **11. Create Depth Map Aligned to RGB Camera**
aligned_depth_map = np.zeros((h, w), dtype=np.float32)
aligned_depth_map[v_proj, u_proj] = depth_values

# Normalize depth map for visualization
aligned_depth_map_normalized = cv2.normalize(aligned_depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# **12. Overlay Depth Map on RGB Image**
overlay_depth = rgb_image.copy()

for i in range(len(u_proj)):
    color = (0, 255, 0)  # Green for depth points
    cv2.circle(overlay_depth, (u_proj[i], v_proj[i]), 2, color, -1)

# **13. Save All High-Resolution Images**
cv2.imwrite(os.path.join(output_dir, "01_raw_rgb.png"), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "02_spherical_depth_map.png"), cv2.normalize(spherical_depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "03_aligned_depth_map.png"), aligned_depth_map_normalized)
cv2.imwrite(os.path.join(output_dir, "04_rgb_depth_overlay.png"), cv2.cvtColor(overlay_depth, cv2.COLOR_RGB2BGR))

# **14. Visualize All Images**
fig, ax = plt.subplots(1, 4, figsize=(28, 8))

ax[0].imshow(rgb_image)
ax[0].set_title("Raw RGB Image")
ax[0].axis("off")

ax[1].imshow(spherical_depth_map, cmap='plasma')
ax[1].set_title("Raw Spherical Depth Map")
ax[1].axis("off")

ax[2].imshow(aligned_depth_map_normalized, cmap='plasma')
ax[2].set_title("Depth Map Aligned to RGB")
ax[2].axis("off")

ax[3].imshow(overlay_depth)
ax[3].set_title("Aligned RGB + Depth Overlay")
ax[3].axis("off")

plt.show()

print(f"✅ All images saved in: {output_dir}")
