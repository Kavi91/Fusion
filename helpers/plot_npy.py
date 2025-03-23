#!/home/kavi/LO-env/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to load GT poses from .npy file (FusionLIVO style)
def load_gt_poses_from_npy(pose_path):
    poses = np.load(pose_path)  # Shape: [N, 7] (x, y, z, w, x, y, z)
    
    # Convert to 4x4 transformation matrices
    absolute_poses = []
    for pose in poses:
        trans = pose[:3]  # Translation (x, y, z)
        quat = pose[3:]   # Quaternion (w, x, y, z)
        
        # Convert quaternion to rotation matrix
        w, x, y, z = quat
        rot = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = trans
        absolute_poses.append(transform)
    
    return np.array(absolute_poses)

# Plot GT trajectory
def plot_gt_poses(poses, seq, save_path=None):
    translations = poses[:, :3, 3]  # Extract translations [x, y, z]
    
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(translations[:, 0], translations[:, 2], c=translations[:, 2], s=20, alpha=0.5, cmap='viridis', label='Ground Truth')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title(f'Ground Truth Trajectory - Sequence {seq} (from poses_7dof/{seq}.npy)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Main
if __name__ == "__main__":
    seq = "01"
    pose_path = f"/home/kavi/Datasets/KITTI_raw/kitti_data/poses_7dof/{seq}.npy"  # Path to poses_7dof/01.npy
    save_path = f'gt_trajectory_seq_{seq}_from_npy.png'
    
    # Load GT poses
    gt_poses = load_gt_poses_from_npy(pose_path)
    print(f"Loaded {len(gt_poses)} GT poses from {pose_path}")
    
    # Plot GT trajectory
    plot_gt_poses(gt_poses, seq, save_path)