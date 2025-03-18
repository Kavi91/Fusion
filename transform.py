# transform_lorconlo_poses_to_camera.py
import yaml
import numpy as np
import os
from scipy.spatial.transform import Rotation

def load_config(config_path="config.yaml"):
    """Load configuration from config.yaml."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_absolute_poses(pose_file):
    """Load absolute poses from KITTI's pose file (camera frame)."""
    with open(pose_file, "r") as f:
        lines = f.readlines()
        poses = []
        for line in lines:
            pose = np.array([float(x) for x in line.strip().split()])
            pose_mat = np.eye(4)
            pose_mat[:3, :] = pose.reshape(3, 4)
            poses.append(pose_mat)
    return np.array(poses)

def absolute_to_relative_poses(poses, sequence):
    """Compute relative poses from absolute poses in the camera frame."""
    relative_poses = []
    for i in range(len(poses) - 1):
        pose_pre = poses[i]
        pose_next = poses[i + 1]
        relative_transform = np.linalg.inv(pose_pre) @ pose_next  # Relative transform in camera frame
        t = relative_transform[:3, 3]  # Translation in camera frame
        R = relative_transform[:3, :3]
        # Convert rotation matrix to Euler angles (ZYX convention, matching DeepVO)
        angles = Rotation.from_matrix(R).as_euler('ZYX', degrees=False)
        # Adjust signs to match DeepVO's observed convention
        angles = -angles  # Flip signs based on observed mismatch
        relative_pose = np.concatenate([angles, t])  # [roll, pitch, yaw, tx, ty, tz]
        print(f"Sequence {sequence} - Frame {i} to {i+1} - Relative Transform:\n{relative_transform}")
        print(f"Translation: {t}, Angles: {angles}")
        relative_poses.append(relative_pose)
    return np.array(relative_poses)

def save_poses(poses, output_file, is_relative=False):
    """Save poses to a file."""
    with open(output_file, "w") as f:
        for i, pose in enumerate(poses):
            if is_relative:
                # Save 6-DoF pose (roll, pitch, yaw, tx, ty, tz)
                line = " ".join(map(str, pose))
            else:
                # Save 4x4 pose matrix as 12 elements (3x4)
                line = " ".join(map(str, pose[:3, :4].flatten()))
            if i < len(poses) - 1:
                line += "\n"
            f.write(line)

def main():
    config = load_config()
    new_pose_dir = "/home/kavi/Fusion/New_Poses/"  # Specified directory
    os.makedirs(new_pose_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    # List of sequences from 00 to 10
    sequences = [f"{i:02d}" for i in range(11)]  # '00' to '10'
    
    for sequence in sequences:
        print(f"\n=== Processing Sequence {sequence} ===")
        # Load KITTI absolute poses (in camera frame)
        pose_file = os.path.join(new_pose_dir, f"{sequence}.txt")
        if not os.path.exists(pose_file):
            print(f"Pose file {pose_file} not found. Skipping sequence {sequence}.")
            continue
        
        absolute_poses = load_absolute_poses(pose_file)
        
        # Save absolute poses (already in camera frame)
        output_abs_file = os.path.join(new_pose_dir, f"{sequence}_camera_frame.txt")
        save_poses(absolute_poses, output_abs_file)
        print(f"Sequence {sequence} - Absolute poses (camera frame) saved to {output_abs_file}")
        
        # Compute relative poses in camera frame
        relative_poses_camera = absolute_to_relative_poses(absolute_poses, sequence)
        
        # Save relative poses
        output_rel_file = os.path.join(new_pose_dir, f"{sequence}_relative_camera_frame.txt")
        save_poses(relative_poses_camera, output_rel_file, is_relative=True)
        print(f"Sequence {sequence} - Transformed relative poses (camera frame) saved to {output_rel_file}")

if __name__ == "__main__":
    main()