#!/home/krkavinda/LO-env/bin/python3
import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import matplotlib.pyplot as plt
from models import FusionLIVO
from fusion_dataset import FusionDataset
from tqdm import tqdm

# Function to compute ATE and RPE
def compute_trajectory_metrics(pred_poses, gt_poses):
    pred_poses = pred_poses.cpu().numpy()
    gt_poses = gt_poses.cpu().numpy()
    
    pred_trans = pred_poses[:, :, :3]
    gt_trans = gt_poses[:, :, :3]
    pred_rot = pred_poses[:, :, 3:]
    gt_rot = gt_poses[:, :, 3:]
    
    # ATE: Compute RMSE of translation differences
    trans_diff = pred_trans - gt_trans
    trans_diff = trans_diff.reshape(-1, 3)
    ate = np.sqrt(np.mean(np.sum(trans_diff**2, axis=1)))
    
    # RPE: Compute relative pose errors between consecutive frames
    rpe_trans = []
    rpe_rot = []
    for b in range(pred_poses.shape[0]):
        for t in range(pred_poses.shape[1] - 1):
            # Predicted relative pose
            pred_trans1 = pred_trans[b, t]
            pred_trans2 = pred_trans[b, t+1]
            pred_rot1 = pred_rot[b, t]
            pred_rot2 = pred_rot[b, t+1]
            pred_rel_trans = pred_trans2 - pred_trans1
            pred_rel_rot = pred_rot2 - pred_rot1
            
            # Ground truth relative pose
            gt_trans1 = gt_trans[b, t]
            gt_trans2 = gt_trans[b, t+1]
            gt_rot1 = gt_rot[b, t]
            gt_rot2 = gt_rot[b, t+1]
            gt_rel_trans = gt_trans2 - gt_trans1
            gt_rel_rot = gt_rot2 - gt_rot1
            
            # RPE translation
            rpe_trans.append(np.linalg.norm(pred_rel_trans - gt_rel_trans))
            
            # RPE rotation: Compute angular difference
            rot_diff = np.linalg.norm(gt_rel_rot - pred_rel_rot, axis=-1)
            angle = rot_diff * 180 / np.pi
            rpe_rot.append(angle)
    
    rpe_trans = np.sqrt(np.mean(np.array(rpe_trans)**2))
    rpe_rot = np.sqrt(np.mean(np.array(rpe_rot)**2))
    
    return ate, rpe_trans, rpe_rot

# Function to convert relative poses to absolute poses
def relative_to_absolute_poses(relative_poses, translation_scale=100.0):
    # relative_poses: [N, 6] (translation [x, y, z], rotation [pitch, yaw, roll])
    absolute_poses = []
    current_pose = np.eye(4)  # Start at identity matrix (origin)
    
    for rel_pose in relative_poses:
        trans = rel_pose[:3] * translation_scale  # Scale back translations
        pitch, yaw, roll = rel_pose[3:]
        
        # Convert Euler angles to rotation matrix
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cr, sr = np.cos(roll), np.sin(roll)
        Rx = np.array([
            [1, 0, 0],
            [0, cp, -sp],
            [0, sp, cp]
        ])
        Ry = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])
        Rz = np.array([
            [cr, -sr, 0],
            [sr, cr, 0],
            [0, 0, 1]
        ])
        rot = Rz @ Ry @ Rx
        
        # Create transformation matrix
        rel_transform = np.eye(4)
        rel_transform[:3, :3] = rot
        rel_transform[:3, 3] = trans
        
        # Update current pose
        current_pose = current_pose @ rel_transform
        absolute_pose = current_pose[:3, 3]  # Extract translation
        absolute_poses.append(absolute_pose)
    
    return np.array(absolute_poses)

# Function to load absolute ground truth poses directly from .npy file
def load_absolute_gt_poses(pose_path, translation_scale=100.0):
    poses = np.load(pose_path)  # Shape: [N, 7] (x, y, z, w, x, y, z)
    
    # Convert to 4x4 transformation matrices
    absolute_poses = []
    for pose in poses:
        trans = pose[:3]  # Translation (x, y, z)
        trans = trans * translation_scale  # Scale back translations
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

# Custom collate function to handle None values
def custom_collate_fn(batch):
    batch = list(zip(*batch))
    rgb_left = torch.stack(batch[0]) if batch[0][0] is not None else None
    rgb_right = torch.stack(batch[1]) if batch[1][0] is not None else None
    lidar_combined = torch.stack(batch[2]) if batch[2][0] is not None else None
    targets = torch.stack(batch[3])
    return rgb_left, rgb_right, lidar_combined, targets

# Load configuration
with open("/home/krkavinda/Fusion/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Device configuration
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# Initialize model
model = FusionLIVO(config).to(device)
model.load_state_dict(torch.load(config["fusion"]["model_path"], map_location=device))
model.eval()

# Test sequences (same as DeepVO)
test_sequences_with_gt = ["03", "04", "05", "06", "07", "10"]
test_sequences_without_gt = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]

# Test on sequences with ground truth
results_with_gt = {}
for seq in test_sequences_with_gt:
    print(f"\nTesting on sequence {seq}...")
    test_dataset = FusionDataset(
        config,
        seqs=[seq],
        seq_len=config["fusion"]["rnn_size"],
        use_augmentation=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["fusion"]["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    # Load absolute ground truth poses
    pose_path = os.path.join(config["deepvo"]["pose_dir"], f"{seq}.npy")
    gt_absolute_poses = load_absolute_gt_poses(pose_path, translation_scale=100.0)  # Shape: [N, 4, 4]

    # Lists to store all predictions and ground truth (relative poses)
    all_pred_poses = []
    all_gt_poses = []

    # Test loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing Sequence {seq}"):
            rgb_left, rgb_right, lidar_combined, targets = batch
            rgb_left = rgb_left.to(device) if rgb_left is not None else None
            rgb_right = rgb_right.to(device) if rgb_right is not None else None
            lidar_combined = lidar_combined.to(device) if lidar_combined is not None else None
            targets = targets.to(device)

            outputs = model(rgb_left, rgb_right, lidar_combined)
            
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                raise ValueError("NaN or Inf detected in model outputs during testing")

            all_pred_poses.append(outputs)
            all_gt_poses.append(targets)

    # Concatenate all predictions and ground truth
    all_pred_poses = torch.cat(all_pred_poses, dim=0)  # Shape: [M, seq_len-1, 6]
    all_gt_poses = torch.cat(all_gt_poses, dim=0)      # Shape: [M, seq_len-1, 6]

    # Reshape to [N, 6] for relative_to_absolute_poses
    num_samples, seq_len_minus_1, pose_dim = all_pred_poses.shape
    all_pred_poses = all_pred_poses.view(-1, pose_dim)  # [M * (seq_len-1), 6]
    all_gt_poses = all_gt_poses.view(-1, pose_dim)      # [M * (seq_len-1), 6]

    # Compute metrics
    ate, rpe_trans, rpe_rot = compute_trajectory_metrics(
        all_pred_poses.view(num_samples, seq_len_minus_1, pose_dim),
        all_gt_poses.view(num_samples, seq_len_minus_1, pose_dim)
    )
    print(f"Sequence {seq} - Test ATE: {ate:.4f} m, RPE Trans: {rpe_trans:.4f} m, RPE Rot: {rpe_rot:.4f} deg")
    results_with_gt[seq] = {"ATE": ate, "RPE Trans": rpe_trans, "RPE Rot": rpe_rot}

    # Convert predicted relative poses to absolute poses
    pred_absolute_poses = relative_to_absolute_poses(all_pred_poses.cpu().numpy(), translation_scale=100.0)  # Shape: [M * (seq_len-1), 3]

    # Extract translations from absolute GT poses
    gt_translations = gt_absolute_poses[:, :3, 3]  # Shape: [N, 3]

    # Plot both trajectories on the same plot (x-z plane, top-down view)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(gt_translations[:, 0], gt_translations[:, 2], c=gt_translations[:, 2], s=20, alpha=0.5, cmap='viridis', label='Ground Truth')
    plt.scatter(pred_absolute_poses[:, 0], pred_absolute_poses[:, 2], c=pred_absolute_poses[:, 2], s=20, alpha=0.5, cmap='magma', label='Predicted')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title(f'Trajectory Comparison - Sequence {seq}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f'trajectory_seq_{seq}.png')
    plt.close()

# Test on sequences without ground truth (qualitative evaluation)
for seq in test_sequences_without_gt:
    print(f"\nTesting on sequence {seq} (no ground truth)...")
    test_dataset = FusionDataset(
        config,
        seqs=[seq],
        seq_len=config["fusion"]["rnn_size"],
        use_augmentation=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["fusion"]["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    # Lists to store all predictions
    all_pred_poses = []

    # Test loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing Sequence {seq}"):
            rgb_left, rgb_right, lidar_combined, targets = batch
            rgb_left = rgb_left.to(device) if rgb_left is not None else None
            rgb_right = rgb_right.to(device) if rgb_right is not None else None
            lidar_combined = lidar_combined.to(device) if lidar_combined is not None else None
            targets = targets.to(device)

            outputs = model(rgb_left, rgb_right, lidar_combined)
            
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                raise ValueError("NaN or Inf detected in model outputs during testing")

            all_pred_poses.append(outputs)

    # Concatenate all predictions
    all_pred_poses = torch.cat(all_pred_poses, dim=0)  # Shape: [M, seq_len-1, 6]

    # Reshape to [N, 6] for relative_to_absolute_poses
    num_samples, seq_len_minus_1, pose_dim = all_pred_poses.shape
    all_pred_poses = all_pred_poses.view(-1, pose_dim)  # [M * (seq_len-1), 6]

    # Convert predicted relative poses to absolute poses
    pred_absolute_poses = relative_to_absolute_poses(all_pred_poses.cpu().numpy(), translation_scale=100.0)  # Shape: [M * (seq_len-1), 3]

    # Plot predicted trajectory (no GT available)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(pred_absolute_poses[:, 0], pred_absolute_poses[:, 2], c=pred_absolute_poses[:, 2], s=20, alpha=0.5, cmap='magma', label='Predicted')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title(f'Predicted Trajectory - Sequence {seq} (No Ground Truth)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f'trajectory_seq_{seq}.png')
    plt.close()

# Print summary of results for sequences with ground truth
print("\nSummary of Results (Sequences with Ground Truth):")
for seq, metrics in results_with_gt.items():
    print(f"Sequence {seq}: ATE: {metrics['ATE']:.4f} m, RPE Trans: {metrics['RPE Trans']:.4f} m, RPE Rot: {metrics['RPE Rot']:.4f} deg")