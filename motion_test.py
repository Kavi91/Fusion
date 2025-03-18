import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Paths
calib_file = "/home/kavi/Datasets/KITTI_raw/kitti_data/calib/04.txt"
output_dir = "/home/kavi/Fusion/Projections"
rgb_file = "/home/kavi/Datasets/KITTI_raw/kitti_data/sequences/04/image_02/0000000000.png"  # For vector field overlay

# **1. Load Calibration File**
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
P2 = calib["P2"]  # Camera projection matrix for image_02
Tr = calib["Tr"]  # LiDAR to camera transformation

# Extract camera parameters
fx = P2[0, 0]
fy = P2[1, 1]
cx = P2[0, 2]
cy = P2[1, 2]

# **2. Motion Estimation**
# Load two consecutive frames with a larger gap for noticeable motion
frame1_idx = 0
frame2_idx = 10  # Increased frame gap to detect motion

rgb1 = cv2.imread(f"/home/kavi/Datasets/KITTI_raw/kitti_data/sequences/04/image_02/{str(frame1_idx).zfill(10)}.png", 0)  # Grayscale
rgb2 = cv2.imread(f"/home/kavi/Datasets/KITTI_raw/kitti_data/sequences/04/image_02/{str(frame2_idx).zfill(10)}.png", 0)  # Grayscale
depth1 = np.load(f"/home/kavi/Fusion/Projections/depth_map_{str(frame1_idx).zfill(6)}.npy")
depth2 = np.load(f"/home/kavi/Fusion/Projections/depth_map_{str(frame2_idx).zfill(6)}.npy")

# Check if images and depth maps are loaded correctly
if rgb1 is None or rgb2 is None:
    print("Error: One or both RGB images failed to load.")
    exit(1)
if depth1.size == 0 or depth2.size == 0:
    print("Error: One or both depth maps failed to load.")
    exit(1)

# Verify depth map content
print("Depth1 valid pixels (non-zero):", np.sum(depth1 > 0))
print("Depth2 valid pixels (non-zero):", np.sum(depth2 > 0))

# Compute optical flow with adjusted parameters
flow = cv2.calcOpticalFlowFarneback(rgb1, rgb2, None, 0.5, 5, 25, 5, 7, 1.5, 0)  # Adjusted: levels=5, winsize=25, iterations=5, poly_sigma=1.5

# Align flow shape with depth maps (resize if necessary)
flow = cv2.resize(flow, (depth1.shape[1], depth1.shape[0]), interpolation=cv2.INTER_LINEAR)

# Convert 2D flow to 3D motion using depth
h, w = depth1.shape
flow_3d = np.zeros((h, w, 3), dtype=np.float32)

for i in range(h):
    for j in range(w):
        if depth1[i, j] > 0 and depth2[i, j] > 0 and not np.isnan(flow[i, j, 0]) and not np.isnan(flow[i, j, 1]):
            z1 = depth1[i, j]
            z2 = depth2[i, j]
            # Convert pixel coordinates to camera coordinates
            x1 = (j - cx) * z1 / fx
            y1 = (i - cy) * z1 / fy
            # Apply 2D flow to get new pixel coordinates
            u2 = j + flow[i, j, 0]
            v2 = i + flow[i, j, 1]
            # Ensure new coordinates are within bounds
            if 0 <= u2 < w and 0 <= v2 < h:
                if 0 <= int(v2) < h and 0 <= int(u2) < w and depth2[int(v2), int(u2)] > 0:
                    x2 = (u2 - cx) * depth2[int(v2), int(u2)] / fx
                    y2 = (v2 - cy) * depth2[int(v2), int(u2)] / fy
                    # Compute 3D motion
                    flow_3d[i, j, 0] = x2 - x1
                    flow_3d[i, j, 1] = y2 - y1
                    flow_3d[i, j, 2] = z2 - z1

# Save the 3D flow
output_flow_path = os.path.join(output_dir, f"3d_flow_{str(frame1_idx).zfill(6)}_{str(frame2_idx).zfill(6)}.npy")
np.save(output_flow_path, flow_3d)
print(f"3D flow saved to {output_flow_path}")

# **3. Check and Visualize the Motion File**

# Load the .npy file
flow_3d = np.load(output_flow_path)

# Print basic information about the array
print("Shape of the 3D flow array:", flow_3d.shape)
print("Data type:", flow_3d.dtype)
print("Array contents (first few elements):", flow_3d[:5, :5, :])  # Display a small subset

# **4. Visualize the Motion Field**

# Method 1: Color-Coded Visualization (Magnitude of Motion)
motion_magnitude = np.sqrt(flow_3d[:, :, 0]**2 + flow_3d[:, :, 1]**2 + flow_3d[:, :, 2]**2)
motion_magnitude_normalized = cv2.normalize(motion_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display the motion magnitude
plt.figure(figsize=(12, 6))
plt.imshow(motion_magnitude_normalized, cmap='viridis')
plt.title("3D Motion Magnitude")
plt.colorbar(label="Motion Magnitude (meters)")
plt.axis("off")
plt.show()

# Save the visualization
output_magnitude_vis_path = os.path.join(output_dir, "3d_flow_magnitude_visualization.png")
plt.figure(figsize=(12, 6))
plt.imshow(motion_magnitude_normalized, cmap='viridis')
plt.title("3D Motion Magnitude")
plt.colorbar(label="Motion Magnitude (meters)")
plt.axis("off")
plt.savefig(output_magnitude_vis_path)
plt.close()
print(f"Motion magnitude visualization saved to {output_magnitude_vis_path}")

# Method 2: Vector Field Visualization
# Extract 2D flow components (ignoring z for visualization)
flow_2d = flow_3d[:, :, :2]  # (dx, dy)

# Downsample for visualization (to reduce clutter)
step = 10
y, x = np.mgrid[step//2:h:step, step//2:w:step]
u = flow_2d[step//2:h:step, step//2:w:step, 0]
v = flow_2d[step//2:h:step, step//2:w:step, 1]

# Create a figure
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB), alpha=0.5)
plt.quiver(x, y, u, -v, color='r', scale=50, width=0.002)  # -v for correct orientation
plt.title("2D Motion Vector Field")
plt.axis("off")
plt.show()

# Save the visualization
output_vector_vis_path = os.path.join(output_dir, "3d_flow_vector_field.png")
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB), alpha=0.5)
plt.quiver(x, y, u, -v, color='r', scale=50, width=0.002)  # -v for correct orientation
plt.title("2D Motion Vector Field")
plt.axis("off")
plt.savefig(output_vector_vis_path)
plt.close()
print(f"Motion vector field visualization saved to {output_vector_vis_path}")

# **5. Analyze Specific Values**
# Check motion at a specific region (e.g., center of the road)
center_x, center_y = w//2, h//2
print(f"Motion at center ({center_x}, {center_y}): {flow_3d[center_y, center_x, :]} meters")

# Check motion near a car (adjust coordinates based on your image)
car_x, car_y = 300, 200  # Approximate car location (e.g., left side of the image)
print(f"Motion near car ({car_x}, {car_y}): {flow_3d[car_y, car_x, :]} meters")