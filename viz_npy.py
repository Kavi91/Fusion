import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Paths
output_dir = "/home/kavi/Fusion/Projections"

# Ensure output directory exists
if not os.path.exists(output_dir):
    print(f"Error: Directory {output_dir} does not exist!")
    exit(1)

# Load all .npy files from the directory
npy_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
if not npy_files:
    print(f"No .npy files found in {output_dir}")
    exit(1)

print(f"Found .npy files: {npy_files}")

# Function to visualize a single .npy file
def visualize_npy_file(file_path, title_prefix="Spherical Depth Map"):
    # Load the .npy file
    depth_map = np.load(file_path)
    
    # Check the shape and print basic info
    print(f"Loaded {os.path.basename(file_path)} with shape: {depth_map.shape}, dtype: {depth_map.dtype}")
    print(f"Depth range (before normalization): {np.min(depth_map)} to {np.max(depth_map)}")

    # Check for invalid values (NaNs or Infs)
    if np.any(np.isnan(depth_map)) or np.any(np.isinf(depth_map)):
        print("Warning: Depth map contains NaN or Inf values. Replacing with 0.")
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize depth map for visualization (0 to 255)
    if np.max(depth_map) == np.min(depth_map):
        print("Warning: Depth map has no variation (all values are the same). Setting to zeros.")
        depth_map_normalized = np.zeros_like(depth_map, dtype=np.uint8)
    else:
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    print(f"Normalized depth range: {np.min(depth_map_normalized)} to {np.max(depth_map_normalized)}")
    print(f"Number of non-zero values in normalized depth map: {np.sum(depth_map_normalized > 0)}")

    # Create figure with two subplots: grayscale and color-coded
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Grayscale visualization
    ax1.imshow(depth_map_normalized, cmap='gray')
    ax1.set_title(f"{title_prefix} (Grayscale)")
    ax1.axis("off")

    # Color-coded visualization
    if np.max(depth_map_normalized) > np.min(depth_map_normalized):
        im = ax2.imshow(depth_map_normalized, cmap='plasma')
    else:
        # Fallback to contourf if imshow fails to create a mappable
        print("Falling back to contourf for color-coded visualization due to lack of variation.")
        levels = np.linspace(0, 255, 10)  # Create some levels for contour
        im = ax2.contourf(depth_map_normalized, levels=levels, cmap='plasma')

    ax2.set_title(f"{title_prefix} (Color-Coded)")
    ax2.axis("off")

    # Add colorbar only if a valid mappable exists
    try:
        plt.colorbar(im, ax=ax2, label="Depth (normalized)")
    except Exception as e:
        print(f"Warning: Failed to create colorbar - {str(e)}")

    plt.tight_layout()
    plt.show()

    # Save the visualizations
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_grayscale.png"), depth_map_normalized)
    plt.imsave(os.path.join(output_dir, f"{base_name}_colorcoded.png"), depth_map_normalized, cmap='plasma')
    print(f"Visualizations saved as {base_name}_grayscale.png and {base_name}_colorcoded.png in {output_dir}")

# Visualize a specific file or all files
def main():
    # Option 1: Visualize a specific file (e.g., frame 0)
    specific_file = "spherical_depth_map_000000.npy"
    if specific_file in npy_files:
        file_path = os.path.join(output_dir, specific_file)
        visualize_npy_file(file_path, "Spherical Depth Map - Frame 000000")
    else:
        print(f"Specified file {specific_file} not found!")

    # Option 2: Visualize all .npy files (commented out to avoid clutter, uncomment if needed)
    """
    for npy_file in npy_files:
        file_path = os.path.join(output_dir, npy_file)
        frame_idx = npy_file.split('_')[-1].split('.')[0]  # Extract frame index (e.g., "000000")
        visualize_npy_file(file_path, f"Spherical Depth Map - Frame {frame_idx}")
    """

if __name__ == "__main__":
    main()