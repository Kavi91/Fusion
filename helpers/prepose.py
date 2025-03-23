#!/home/krkavinda/LO-env/bin/python3
import os
import numpy as np
import torch
import torch.nn.functional as F

# Directories
pose_dir = "/home/krkavinda/Datasets/KITTI_raw/kitti_data/poses/"
output_dir = "/home/krkavinda/Datasets/KITTI_raw/kitti_data/poses_7dof/"
os.makedirs(output_dir, exist_ok=True)

# Sequences to process
sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

for seq in sequences:
    # Load poses from .txt file (absolute poses in left color camera coordinate system, relative to frame 0)
    pose_path = os.path.join(pose_dir, f"{seq}.txt")
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            pose = np.array(list(map(float, line.strip().split())), dtype=np.float64)
            pose = pose.reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses.append(pose)
    poses = np.array(poses)
    
    # Convert to 7-DoF format (x, y, z, w, x, y, z)
    poses_7dof = []
    for pose in poses:
        translation = torch.from_numpy(pose[:3, 3]).float()  # Translation (x, y, z)
        rotation = torch.from_numpy(pose[:3, :3]).float()    # Rotation matrix
        
        # Convert rotation matrix to quaternion (w, x, y, z)
        def rotation_matrix_to_quaternion(R):
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            q = torch.zeros(4, device=R.device)
            if trace > 0:
                s = torch.sqrt(trace + 1.0) * 2
                q[0] = 0.25 * s
                q[1] = (R[2, 1] - R[1, 2]) / s
                q[2] = (R[0, 2] - R[2, 0]) / s
                q[3] = (R[1, 0] - R[0, 1]) / s
            else:
                if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                    s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                    q[0] = (R[2, 1] - R[1, 2]) / s
                    q[1] = 0.25 * s
                    q[2] = (R[1, 0] + R[0, 1]) / s
                    q[3] = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                    q[0] = (R[0, 2] - R[2, 0]) / s
                    q[1] = (R[1, 0] + R[0, 1]) / s
                    q[2] = 0.25 * s
                    q[3] = (R[2, 1] + R[1, 2]) / s
                else:
                    s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                    q[0] = (R[1, 0] - R[0, 1]) / s
                    q[1] = (R[0, 2] + R[2, 0]) / s
                    q[2] = (R[2, 1] + R[1, 2]) / s
                    q[3] = 0.25 * s
            return q
        
        rotation = rotation_matrix_to_quaternion(rotation)
        rotation = F.normalize(rotation, p=2, dim=-1)
        pose_7dof = torch.cat([translation, rotation])
        poses_7dof.append(pose_7dof)
    
    poses_7dof = torch.stack(poses_7dof).numpy()
    if np.isnan(poses_7dof).any() or np.isinf(poses_7dof).any():
        raise ValueError(f"NaN or Inf detected in poses_7dof for sequence {seq}")
    
    # Save the precomputed 7-DoF poses
    output_path = os.path.join(output_dir, f"{seq}.npy")
    np.save(output_path, poses_7dof)
    print(f"Saved precomputed 7-DoF poses for sequence {seq} to {output_path}")