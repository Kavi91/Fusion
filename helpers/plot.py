#!/home/kavi/LO-env/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to load absolute GT poses from .txt file (LoRCoN-LO style)
def load_gt_poses_from_txt(pose_path):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            pose = np.array(list(map(float, line.strip().split())), dtype=np.float64)
            pose = pose.reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))  # Add [0, 0, 0, 1] to make 4x4
            poses.append(pose)
    return np.array(poses)

# Plot GT trajectory
def plot_gt_poses(poses, seq, save_path=None):
    translations = poses[:, :3, 3]  # Extract translations [x, y, z]
    
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(translations[:, 0], translations[:, 2], c=translations[:, 2], s=20, alpha=0.5, cmap='viridis', label='Ground Truth')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title(f'Ground Truth Trajectory - Sequence {seq} (from poses/{seq}.txt)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Main
if __name__ == "__main__":
    seq = "01"
    pose_path = f"/home/kavi/Datasets/KITTI_raw/kitti_data/poses/{seq}.txt"  # Path to poses/01.txt
    save_path = f'gt_trajectory_seq_{seq}_from_txt.png'
    
    # Load GT poses
    gt_poses = load_gt_poses_from_txt(pose_path)
    print(f"Loaded {len(gt_poses)} GT poses from {pose_path}")
    
    # Plot GT trajectory
    plot_gt_poses(gt_poses, seq, save_path)