import os
import numpy as np
import torch
import torch.nn.functional as F  # Add this import for F.normalize

# Directories
pose_dir = "/home/kavi/Datasets/KITTI_raw/kitti_data/poses/"
calib_dir = "/home/kavi/Datasets/KITTI_raw/kitti_data/calib/"
output_dir = "/home/kavi/Datasets/KITTI_raw/kitti_data/poses_7dof/"
os.makedirs(output_dir, exist_ok=True)

# Sequences to process
sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

for seq in sequences:
    # Load poses
    poses_path = os.path.join(pose_dir, f"{seq}.npy")
    poses = np.load(poses_path)
    if np.isnan(poses).any() or np.isinf(poses).any():
        raise ValueError(f"NaN or Inf detected in poses for sequence {seq}")
    
    # Load calibration
    calib_path = os.path.join(calib_dir, f"{seq}.txt")
    Tr_velo_to_cam0 = None
    P0 = None
    P2 = None
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('P0'):
                P0 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            elif line.startswith('P2'):
                P2 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            elif line.startswith('Tr'):
                Tr_velo_to_cam0 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
                Tr_velo_to_cam0 = np.vstack([Tr_velo_to_cam0, np.array([0, 0, 0, 1])])
    
    K = P0[:, :3]
    t0 = P0[:, 3]
    t2 = P2[:, 3]
    baseline = t2 - t0
    Tr_cam0_to_cam2 = np.eye(4)
    Tr_cam0_to_cam2[:3, 3] = baseline
    
    Tr_velo_to_cam0 = torch.from_numpy(Tr_velo_to_cam0).float()
    Tr_cam0_to_cam2 = torch.from_numpy(Tr_cam0_to_cam2).float()
    Tr_velo_to_cam2 = Tr_cam0_to_cam2 @ Tr_velo_to_cam0
    
    poses = torch.from_numpy(poses).float()
    poses_4x4 = []
    for i in range(poses.shape[0]):
        t = poses[i, 0:3]
        roll, pitch, yaw = poses[i, 3:6]
        R_x = torch.tensor([[1, 0, 0],
                            [0, torch.cos(roll), -torch.sin(roll)],
                            [0, torch.sin(roll), torch.cos(roll)]])
        R_y = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                            [0, 1, 0],
                            [-torch.sin(pitch), 0, torch.cos(pitch)]])
        R_z = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                            [torch.sin(yaw), torch.cos(yaw), 0],
                            [0, 0, 1]])
        R = R_z @ R_y @ R_x
        T = torch.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        poses_4x4.append(T)
    poses_4x4 = torch.stack(poses_4x4)
    
    poses_4x4_cam = Tr_velo_to_cam2 @ poses_4x4
    if torch.isnan(poses_4x4_cam).any() or torch.isinf(poses_4x4_cam).any():
        raise ValueError(f"NaN or Inf detected in poses_4x4_cam for sequence {seq}")
    
    translation = poses_4x4_cam[:, :3, 3]
    def rotation_matrix_to_quaternion(R):
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        q = torch.zeros((R.shape[0], 4), device=R.device)
        mask = trace > 0
        s = torch.sqrt(trace[mask] + 1.0) * 2
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
        mask = ~mask
        mask_i = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        mask_j = (R[:, 1, 1] > R[:, 0, 0]) & (R[:, 1, 1] > R[:, 2, 2])
        mask_k = (R[:, 2, 2] > R[:, 0, 0]) & (R[:, 2, 2] > R[:, 1, 1])
        s = torch.sqrt(1.0 + R[mask_i, 0, 0] - R[mask_i, 1, 1] - R[mask_i, 2, 2]) * 2
        q[mask_i, 0] = (R[mask_i, 2, 1] - R[mask_i, 1, 2]) / s
        q[mask_i, 1] = 0.25 * s
        q[mask_i, 2] = (R[mask_i, 1, 0] + R[mask_i, 0, 1]) / s
        q[mask_i, 3] = (R[mask_i, 0, 2] + R[mask_i, 2, 0]) / s
        s = torch.sqrt(1.0 + R[mask_j, 1, 1] - R[mask_j, 0, 0] - R[mask_j, 2, 2]) * 2
        q[mask_j, 0] = (R[mask_j, 0, 2] - R[mask_j, 2, 0]) / s
        q[mask_j, 1] = (R[mask_j, 1, 0] + R[mask_j, 0, 1]) / s
        q[mask_j, 2] = 0.25 * s
        q[mask_j, 3] = (R[mask_j, 2, 1] + R[mask_j, 1, 2]) / s
        s = torch.sqrt(1.0 + R[mask_k, 2, 2] - R[mask_k, 0, 0] - R[mask_k, 1, 1]) * 2
        q[mask_k, 0] = (R[mask_k, 1, 0] - R[mask_k, 0, 1]) / s
        q[mask_k, 1] = (R[mask_k, 0, 2] + R[mask_k, 2, 0]) / s
        q[mask_k, 2] = (R[mask_k, 2, 1] + R[mask_k, 1, 2]) / s
        q[mask_k, 3] = 0.25 * s
        return q

    rotation = rotation_matrix_to_quaternion(poses_4x4_cam[:, :3, :3])
    max_translation = 10.0
    translation = torch.clamp(translation / max_translation, -1.0, 1.0) * max_translation
    rotation = F.normalize(rotation, p=2, dim=-1)
    poses_7dof = torch.cat([translation, rotation], dim=1)
    if torch.isnan(poses_7dof).any() or torch.isinf(poses_7dof).any():
        raise ValueError(f"NaN or Inf detected in transformed poses_7dof for sequence {seq}")
    
    # Save the precomputed 7-DoF poses
    output_path = os.path.join(output_dir, f"{seq}.npy")
    np.save(output_path, poses_7dof.numpy())
    print(f"Saved precomputed 7-DoF poses for sequence {seq} to {output_path}")