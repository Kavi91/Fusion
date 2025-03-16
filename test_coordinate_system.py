# FUSION/test_coordinate_system.py
import os
import numpy as np
import transforms3d as t3d
import yaml

def transform_absolute_pose(absolute_pose, calib_matrix):
    """
    Transform a 6D absolute pose from velodyne to camera frame.
    Convert degrees to radians for input angles.
    """
    rot_vec = np.deg2rad(absolute_pose[:3])  # Convert degrees to radians
    trans_vec = absolute_pose[3:6]
    rot_matrix = t3d.euler.euler2mat(rot_vec[0], rot_vec[1], rot_vec[2], 'sxyz')
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rot_matrix
    pose_matrix[:3, 3] = trans_vec
    print("Original pose (degrees):", absolute_pose)
    print("Rotation vector (radians):", rot_vec)
    print("Rotation matrix:", rot_matrix)
    print("Pose matrix:", pose_matrix)
    # Correct transformation: velodyne to camera
    transformed_matrix = calib_matrix @ pose_matrix
    print("Transformed matrix:", transformed_matrix)
    new_rot_matrix = transformed_matrix[:3, :3]
    new_trans = transformed_matrix[:3, 3]
    new_rot_vec = t3d.euler.mat2euler(new_rot_matrix, 'sxyz')
    print("New rotation matrix:", new_rot_matrix)
    print("New translation:", new_trans)
    print("Transformed pose:", np.concatenate([new_rot_vec, new_trans]))
    return np.concatenate([new_rot_vec, new_trans])

def compute_relative_pose(absolute_poses):
    """
    Compute relative poses from consecutive absolute poses (15D to 6D).
    """
    relative_poses = []
    for i in range(1, len(absolute_poses)):
        rot_matrix = absolute_poses[i][:9].reshape(3, 3)
        trans_vec = absolute_poses[i][[3, 7, 11]]  # Correct indices for [tx, ty, tz]
        rot_vec = t3d.euler.mat2euler(rot_matrix, 'sxyz')
        pose = np.concatenate([rot_vec, trans_vec])
        prev_rot_matrix = absolute_poses[i-1][:9].reshape(3, 3)
        prev_trans_vec = absolute_poses[i-1][[3, 7, 11]]
        prev_rot_vec = t3d.euler.mat2euler(prev_rot_matrix, 'sxyz')
        prev_pose = np.concatenate([prev_rot_vec, prev_trans_vec])
        rel_pose = pose - prev_pose
        rel_pose[:3] = (rel_pose[:3] + np.pi) % (2 * np.pi) - np.pi
        relative_poses.append(rel_pose)
    return np.array(relative_poses)

def compute_relative_pose_from_matrices(absolute_poses, calib_matrix):
    """
    Compute relative poses directly from consecutive transformation matrices and transform to camera frame.
    """
    relative_poses = []
    for i in range(1, len(absolute_poses)):
        # Extract transformation matrices
        rot_matrix = absolute_poses[i][:9].reshape(3, 3)
        trans_vec = absolute_poses[i][[3, 7, 11]]
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rot_matrix
        pose_matrix[:3, 3] = trans_vec

        prev_rot_matrix = absolute_poses[i-1][:9].reshape(3, 3)
        prev_trans_vec = absolute_poses[i-1][[3, 7, 11]]
        prev_pose_matrix = np.eye(4)
        prev_pose_matrix[:3, :3] = prev_rot_matrix
        prev_pose_matrix[:3, 3] = prev_trans_vec

        # Compute relative pose in the original frame
        relative_matrix = np.linalg.inv(prev_pose_matrix) @ pose_matrix
        relative_rot_matrix = relative_matrix[:3, :3]
        relative_trans = relative_matrix[:3, 3]
        relative_rot_vec = t3d.euler.mat2euler(relative_rot_matrix, 'sxyz')

        # Transform the relative pose to the camera frame
        transformed_relative_matrix = calib_matrix @ relative_matrix @ np.linalg.inv(calib_matrix)
        transformed_rot_matrix = transformed_relative_matrix[:3, :3]
        transformed_trans = transformed_relative_matrix[:3, 3]
        transformed_rot_vec = t3d.euler.mat2euler(transformed_rot_matrix, 'sxyz')
        transformed_pose = np.concatenate([transformed_rot_vec, transformed_trans])
        transformed_pose[:3] = (transformed_pose[:3] + np.pi) % (2 * np.pi) - np.pi
        relative_poses.append(transformed_pose)
    return np.array(relative_poses)

def test_coordinate_system():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load DeepVO absolute poses (camera frame)
    video_id = "00"
    deepvo_pose_file = os.path.join(config["deepvo"]["pose_dir"], f"{video_id}.npy")
    deepvo_poses = np.load(deepvo_pose_file)
    deepvo_poses_6d = np.zeros((len(deepvo_poses), 6))
    for i in range(len(deepvo_poses)):
        rot_matrix = deepvo_poses[i][:9].reshape(3, 3)
        trans_vec = deepvo_poses[i][[3, 7, 11]]  # Correct indices for [tx, ty, tz]
        rot_vec = t3d.euler.mat2euler(rot_matrix, 'sxyz')
        deepvo_poses_6d[i] = np.concatenate([rot_vec, trans_vec])
    print("DeepVO first 5 absolute poses (camera frame, 6D extracted):")
    for i in range(5):
        print(f"Frame {i}: {deepvo_poses_6d[i]}")

    # Compute DeepVO relative poses
    deepvo_relative_poses = compute_relative_pose(deepvo_poses)
    print("\nDeepVO first 4 relative poses (camera frame):")
    for i in range(4):
        print(f"Frame {i+1}: {deepvo_relative_poses[i]}")

    # Load KITTI poses for validation
    kitti_pose_file = os.path.join(config["deepvo"]["pose_dir"], f"{video_id}.txt")
    kitti_poses = []
    if os.path.exists(kitti_pose_file):
        with open(kitti_pose_file, "r") as f:
            kitti_poses = [np.array(list(map(float, line.strip().split()))) for line in f.readlines()]
        print("\nKITTI first 5 poses (12D matrix flattened):")
        for i in range(5):
            print(f"Frame {i}: {kitti_poses[i]}")

    # Load LoRCoN-LO relative poses (velodyne frame)
    lorcon_pose_file = os.path.join(config["lorcon_lo"]["relative_pose_folder"], f"{video_id}.txt")
    lorcon_poses = []
    with open(lorcon_pose_file, "r") as f:
        for line in f:
            pose = np.array(list(map(float, line.strip().split())), dtype=np.float64)
            lorcon_poses.append(pose)
    lorcon_poses = np.array(lorcon_poses)
    print("\nLoRCoN first 5 relative poses (velodyne frame):")
    for i in range(5):
        print(f"Frame {i}: {lorcon_poses[i]}")

    # Load calibration data
    calib_file = os.path.join(config["deepvo"]["calib_folder"], f"{video_id}.txt")
    calib_matrix = None
    if os.path.exists(calib_file):
        with open(calib_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Tr:" in line:
                    vals = [float(x) for x in line.strip().split()[1:]]
                    calib_matrix = np.array(vals).reshape(3, 4)
                    calib_matrix = np.vstack([calib_matrix, [0, 0, 0, 1]])  # 4x4 matrix
                    break
    if calib_matrix is None:
        print(f"Calibration file {calib_file} not found or missing Tr")
        return

    print("\nCalibration matrix (Tr_velo_to_cam):")
    print(calib_matrix)

    # Compute LoRCoN-LO relative poses in velodyne frame
    lorcon_relative_poses = []
    for i in range(1, len(lorcon_poses)):
        rot_vec = np.deg2rad(lorcon_poses[i][:3])
        trans_vec = lorcon_poses[i][3:6]
        rot_matrix = t3d.euler.euler2mat(rot_vec[0], rot_vec[1], rot_vec[2], 'sxyz')
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rot_matrix
        pose_matrix[:3, 3] = trans_vec

        prev_rot_vec = np.deg2rad(lorcon_poses[i-1][:3])
        prev_trans_vec = lorcon_poses[i-1][3:6]
        prev_rot_matrix = t3d.euler.euler2mat(prev_rot_vec[0], prev_rot_vec[1], prev_rot_vec[2], 'sxyz')
        prev_pose_matrix = np.eye(4)
        prev_pose_matrix[:3, :3] = prev_rot_matrix
        prev_pose_matrix[:3, 3] = prev_trans_vec

        relative_matrix = np.linalg.inv(prev_pose_matrix) @ pose_matrix
        relative_rot_matrix = relative_matrix[:3, :3]
        relative_trans = relative_matrix[:3, 3]
        relative_rot_vec = t3d.euler.mat2euler(relative_rot_matrix, 'sxyz')
        lorcon_relative_poses.append(np.concatenate([relative_rot_vec, relative_trans]))
    lorcon_relative_poses = np.array(lorcon_relative_poses)
    print("\nLoRCoN first 4 relative poses (velodyne frame, computed):")
    for i in range(4):
        print(f"Frame {i+1}: {lorcon_relative_poses[i]}")

    # Transform LoRCoN-LO relative poses to camera frame
    transformed_lorcon_relative_poses = []
    for pose in lorcon_relative_poses:
        rot_vec = pose[:3]
        trans_vec = pose[3:6]
        rot_matrix = t3d.euler.euler2mat(rot_vec[0], rot_vec[1], rot_vec[2], 'sxyz')
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rot_matrix
        pose_matrix[:3, 3] = trans_vec
        transformed_matrix = calib_matrix @ pose_matrix
        new_rot_matrix = transformed_matrix[:3, :3]
        new_trans = transformed_matrix[:3, 3]
        new_rot_vec = t3d.euler.mat2euler(new_rot_matrix, 'sxyz')
        transformed_pose = np.concatenate([new_rot_vec, new_trans])
        transformed_pose[:3] = (transformed_pose[:3] + np.pi) % (2 * np.pi) - np.pi
        transformed_lorcon_relative_poses.append(transformed_pose)
    transformed_lorcon_relative_poses = np.array(transformed_lorcon_relative_poses)
    print("\nTransformed LoRCoN first 4 relative poses (camera frame, computed):")
    for i in range(4):
        print(f"Frame {i+1}: {transformed_lorcon_relative_poses[i]}")

    # Compare DeepVO and transformed LoRCoN-LO relative poses
    print("\nComparison of DeepVO and transformed LoRCoN-LO relative poses:")
    for i in range(min(4, len(deepvo_relative_poses), len(transformed_lorcon_relative_poses))):
        print(f"Frame {i+1}: DeepVO {deepvo_relative_poses[i]}, Transformed LoRCoN {transformed_lorcon_relative_poses[i]}")

if __name__ == "__main__":
    test_coordinate_system()