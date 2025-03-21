# visualize_lidar_rgb.py
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

def load_lidar_point_cloud(velodyne_folder, sequence="04", frame="000000"):
    """Load the LiDAR point cloud from .bin file."""
    # Update frame to 10-digit format to match KITTI naming
    frame_10digit = frame.zfill(10)
    bin_file = os.path.join(velodyne_folder, "scan", sequence, "velodyne", f"{frame_10digit}.bin")
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"LiDAR point cloud {bin_file} not found. Check the path or dataset structure.")
    
    # Read binary file
    scan = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)  # [x, y, z, intensity]
    points = scan[:, :3]  # Use only x, y, z coordinates
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def load_rgb_image(image_folder, sequence="04", frame="000000"):
    """Load the corresponding RGB image, converting 6-digit to 10-digit format."""
    frame_10digit = frame.zfill(10)  # Pad with zeros to 10 digits
    image_file = os.path.join(image_folder, "sequences", sequence, "image_02", f"{frame_10digit}.png")
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"RGB image {image_file} not found. Check your dataset structure.")
    
    rgb_image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    if rgb_image is None:
        raise ValueError(f"Failed to load RGB image {image_file}")
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    return rgb_image

def visualize_point_cloud(pcd, output_file, title):
    """Visualize the point cloud using Open3D and save a screenshot."""
    # Set visualization options
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=800, height=600)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2
    vis.run()  # Interactive visualization
    vis.capture_screen_image(output_file)
    vis.destroy_window()
    print(f"Point cloud visualization saved to {output_file}")

def visualize_rgb_image(rgb_image, output_file, title):
    """Visualize the RGB image and save it."""
    plt.figure(figsize=(10, 5))
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis('off')
    plt.savefig(output_file)
    plt.close()
    print(f"RGB image visualization saved to {output_file}")

def main():
    velodyne_folder = "/home/kavi/Datasets/KITTI_raw/kitti_data"  # Root of KITTI dataset
    image_folder = "/home/kavi/Datasets/KITTI_raw/kitti_data"  # Root of KITTI dataset
    output_dir = "/home/kavi/Fusion/New_Poses/"  # Directory to save visualizations
    sequence = "04"  # Focus on sequence 04
    frames = ["000000", "000001"]  # First two frames

    for frame in frames:
        # Load and visualize LiDAR point cloud
        pcd = load_lidar_point_cloud(velodyne_folder, sequence, frame)
        print(f"Point Cloud Frame {frame} - Number of Points: {len(pcd.points)}")
        visualize_point_cloud(pcd, os.path.join(output_dir, f"pcd_{sequence}_{frame}.png"),
                             f"Point Cloud - Sequence {sequence}, Frame {frame}")

        # Load and visualize RGB image
        rgb_image = load_rgb_image(image_folder, sequence, frame)
        print(f"RGB Image Frame {frame} - Shape: {rgb_image.shape}")
        visualize_rgb_image(rgb_image, os.path.join(output_dir, f"rgb_{sequence}_{frame}.png"),
                           f"RGB Image - Sequence {sequence}, Frame {frame}")

if __name__ == "__main__":
    # Check if Open3D is installed
    try:
        import open3d
    except ImportError:
        print("Open3D is not installed. Please install it using: pip install open3d")
        exit(1)
    
    main()