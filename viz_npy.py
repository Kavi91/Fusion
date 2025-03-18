import numpy as np
import os
import matplotlib.pyplot as plt

# Define base directory and sequence
base_dir = "/home/kavi/Datasets/KITTI_raw/kitti_data/preprocessed_data"
sequence = "04"  # Change this if you want a different sequence

# Paths to the first files
depth_file = os.path.join(base_dir, sequence, "depth", "000000.npy")
intensity_file = os.path.join(base_dir, sequence, "intensity", "000000.npy")
normal_file = os.path.join(base_dir, sequence, "normal", "000000.npy")

# Function to load and print array info
def load_and_print(file_path, data_type):
    if os.path.exists(file_path):
        data = np.load(file_path)
        print(f"{data_type} file: {file_path}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Min value: {np.min(data)}")
        print(f"Max value: {np.max(data)}")
        print(f"Sample data (first 5 elements): {data.flatten()[:5]}")
        print("-" * 50)
        return data
    else:
        print(f"Error: {file_path} does not exist")
        return None

# Load data
depth_data = load_and_print(depth_file, "Depth")
intensity_data = load_and_print(intensity_file, "Intensity")
normal_data = load_and_print(normal_file, "Normals")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(24, 6))  # 3 subplots side by side

# Depth visualization
if depth_data is not None:
    axes[0].imshow(depth_data, cmap='plasma', interpolation='nearest')
    axes[0].set_title("Spherical Depth Map")
    axes[0].axis('off')
    plt.colorbar(axes[0].imshow(depth_data, cmap='plasma'), ax=axes[0], label='Depth (m)')

# Intensity visualization
if intensity_data is not None:
    axes[1].imshow(intensity_data, cmap='gray', interpolation='nearest')
    axes[1].set_title("Spherical Intensity Map")
    axes[1].axis('off')
    plt.colorbar(axes[1].imshow(intensity_data, cmap='gray'), ax=axes[1], label='Intensity')

# Normals visualization
if normal_data is not None:
    # Normalize normals from [-1, 1] to [0, 1] for RGB display
    normal_vis = (normal_data + 1) / 2  # Shift from [-1, 1] to [0, 1]
    normal_vis = np.where(normal_data == -1, 0, normal_vis)  # Set unset pixels to black (0,0,0)
    axes[2].imshow(normal_vis, interpolation='nearest')
    axes[2].set_title("Spherical Normal Map (RGB)")
    axes[2].axis('off')

plt.tight_layout()
plt.show()