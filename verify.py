import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set paths
base_dir = "/home/kavi/Datasets/KITTI_raw/kitti_data/old_preprocessed_data/04/"
depth_file = os.path.join(base_dir, "depth/000000.npy")
intensity_file = os.path.join(base_dir, "intensity/000000.npy")
normal_file = os.path.join(base_dir, "normal/000000.npy")
calib_file = "/home/kavi/Datasets/KITTI_raw/kitti_data/calib/04.txt"
rgb_file = "/home/kavi/Datasets/KITTI_raw/kitti_data/sequences/04/image_02/0000000000.png"
output_dir = "/home/kavi/Fusion/Projections"
os.makedirs(output_dir, exist_ok=True)

# **1. Load Calibration Data**
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
P2 = calib["P2"]  # Camera projection matrix
Tr = calib["Tr"]  # LiDAR-to-camera transformation

# **2. Load RGB Image**
rgb_image = cv2.imread(rgb_file)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
rgb_H, rgb_W, _ = rgb_image.shape  # 370x1226

# **3. Load Preprocessed Depth, Intensity, and Normal Maps**
depth_map = np.load(depth_file)
intensity_map = np.load(intensity_file)
normal_map = np.load(normal_file)

# Get spherical map resolution (expected 64x900)
H, W = depth_map.shape

# **4. Upsample Maps to Match RGB Resolution**
depth_map_resized = cv2.resize(depth_map, (rgb_W, rgb_H), interpolation=cv2.INTER_LINEAR)
intensity_map_resized = cv2.resize(intensity_map, (rgb_W, rgb_H), interpolation=cv2.INTER_LINEAR)
normal_map_resized = cv2.resize(normal_map, (rgb_W, rgb_H), interpolation=cv2.INTER_LINEAR)

# **5. Compute Projection from Spherical to Camera Frame**
theta_min, theta_max = -np.pi, np.pi
phi_min, phi_max = np.radians(-25), np.radians(3)

# Generate spherical coordinates
theta = np.linspace(theta_min, theta_max, W)
phi = np.linspace(phi_max, phi_min, H)
theta, phi = np.meshgrid(theta, phi)

# Convert to Cartesian (LiDAR frame)
X = depth_map * np.cos(phi) * np.cos(theta)
Y = depth_map * np.cos(phi) * np.sin(theta)
Z = depth_map * np.sin(phi)

xyz_lidar = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T  # (N,3)
xyz_camera = (Tr[:, :3] @ xyz_lidar.T + Tr[:, 3:].reshape(3, 1)).T  # Transform to Camera Frame

# **6. Project to Image Plane**
xyz_camera_hom = np.hstack((xyz_camera, np.ones((xyz_camera.shape[0], 1))))  # Convert to homogeneous coordinates
uv_camera = (P2 @ xyz_camera_hom.T).T  # Apply projection matrix

# Convert homogeneous coordinates (u, v, w) → (u/w, v/w)
u_proj = (uv_camera[:, 0] / uv_camera[:, 2]).astype(int)
v_proj = (uv_camera[:, 1] / uv_camera[:, 2]).astype(int)

# **7. Ensure Projection Validity**
valid = (u_proj >= 0) & (u_proj < rgb_W) & (v_proj >= 0) & (v_proj < rgb_H)
u_proj_valid = u_proj[valid]
v_proj_valid = v_proj[valid]

# **8. Overlay Depth, Intensity & Normal Maps on RGB**
def overlay_colormap(rgb, u, v, values, colormap="rainbow"):
    """Overlay depth/intensity/normal points with a colormap, ensuring valid indices."""
    # Normalize values
    values_normalized = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_map = cv2.applyColorMap(values_normalized, cv2.COLORMAP_RAINBOW)

    # Overlay valid points
    for i in range(len(u)):
        rgb[v[i], u[i]] = color_map[v[i], u[i]]

    return rgb

overlayed_depth = overlay_colormap(rgb_image.copy(), u_proj_valid, v_proj_valid, depth_map_resized)
overlayed_intensity = overlay_colormap(rgb_image.copy(), u_proj_valid, v_proj_valid, intensity_map_resized)
overlayed_normals = overlay_colormap(rgb_image.copy(), u_proj_valid, v_proj_valid, np.linalg.norm(normal_map_resized, axis=2))

# **9. Save Overlayed Images**
cv2.imwrite(os.path.join(output_dir, "projected_src_depth.png"), cv2.cvtColor(overlayed_depth, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "projected_src_intensity.png"), cv2.cvtColor(overlayed_intensity, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "projected_src_normal.png"), cv2.cvtColor(overlayed_normals, cv2.COLOR_RGB2BGR))

# **10. Display Results**
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

titles = [
    "Original RGB", "Depth Map Overlay",
    "Original RGB", "Intensity Map Overlay",
    "Original RGB", "Normal Map Overlay"
]
images = [rgb_image, overlayed_depth, rgb_image, overlayed_intensity, rgb_image, overlayed_normals]

for ax, img, title in zip(axes.flat, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()

print(f"\n✅ All projected maps saved in: {output_dir}")
