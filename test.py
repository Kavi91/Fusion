#!/home/kavi/LO-env/bin/python3
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
    pred_yaw = pred_poses[:, :, 3]
    gt_yaw = gt_poses[:, :, 3]
    
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
            pred_yaw1 = pred_yaw[b, t]
            pred_yaw2 = pred_yaw[b, t+1]
            pred_rel_trans = pred_trans2 - pred_trans1
            pred_rel_yaw = pred_yaw2 - pred_yaw1
            
            # Ground truth relative pose
            gt_trans1 = gt_trans[b, t]
            gt_trans2 = gt_trans[b, t+1]
            gt_yaw1 = gt_yaw[b, t]
            gt_yaw2 = gt_yaw[b, t+1]
            gt_rel_trans = gt_trans2 - gt_trans1
            gt_rel_yaw = gt_yaw2 - gt_yaw1
            
            # RPE translation
            rpe_trans.append(np.linalg.norm(pred_rel_trans - gt_rel_trans))
            
            # RPE rotation: Compute angular difference (handle wrap-around)
            yaw_diff = np.arctan2(np.sin(pred_rel_yaw - gt_rel_yaw), np.cos(pred_rel_yaw - gt_rel_yaw))
            angle = np.abs(yaw_diff) * 180 / np.pi
            rpe_rot.append(angle)
    
    rpe_trans = np.sqrt(np.mean(np.array(rpe_trans)**2))
    rpe_rot = np.sqrt(np.mean(np.array(rpe_rot)**2))
    
    return ate, rpe_trans, rpe_rot

# Function to convert relative poses to absolute poses (using yaw)
def relative_to_absolute_poses(relative_poses):
    # relative_poses: [N, 4] (translation [x, y, z], yaw [delta phi])
    absolute_poses = []
    current_pose = np.eye(4)  # Start at identity matrix (origin)
    
    for rel_pose in relative_poses:
        trans = rel_pose[:3]
        delta_phi = rel_pose[3]  # Yaw angle in radians
        
        # Convert yaw to rotation matrix (2D rotation around y-axis)
        cos_phi = np.cos(delta_phi)
        sin_phi = np.sin(delta_phi)
        rot = np.array([
            [cos_phi, 0, sin_phi],
            [0, 1, 0],
            [-sin_phi, 0, cos_phi]
        ])
        
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
def load_absolute_gt_poses(pose_path):
    # pose_path: Path to the pose file (e.g., poses_7dof/01.npy)
    # Returns: A numpy array of size [N, 4, 4] with N poses as 4x4 transformation matrices
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

# Custom collate function to handle None values
def custom_collate_fn(batch):
    batch = list(zip(*batch))
    rgb_left = torch.stack(batch[0]) if batch[0][0] is not None else None
    rgb_right = torch.stack(batch[1]) if batch[1][0] is not None else None
    lidar_combined = torch.stack(batch[2]) if batch[2][0] is not None else None
    targets = torch.stack(batch[3])
    return rgb_left, rgb_right, lidar_combined, targets

# Load configuration
with open("/home/kavi/Fusion/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Device configuration
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# Initialize model
model = FusionLIVO(config).to(device)
model.load_state_dict(torch.load(config["fusion"]["model_path"], map_location=device))
model.eval()

# Test dataset (e.g., Sequence 01)
test_sequence = ["01"]
test_dataset = FusionDataset(
    config,
    seqs=test_sequence,
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

# Load absolute ground truth poses directly from file
pose_path = os.path.join(config["deepvo"]["pose_dir"], f"{test_sequence[0]}.npy")
gt_absolute_poses = load_absolute_gt_poses(pose_path)  # Shape: [N, 4, 4]

# Lists to store all predictions and ground truth (relative poses)
all_pred_poses = []
all_gt_poses = []

# Test loop
print(f"Testing on sequence {test_sequence[0]}...")
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
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
all_pred_poses = torch.cat(all_pred_poses, dim=0)  # Shape: [M, seq_len-1, 4]
all_gt_poses = torch.cat(all_gt_poses, dim=0)      # Shape: [M, seq_len-1, 4]

# Reshape to [N, 4] for relative_to_absolute_poses
num_samples, seq_len_minus_1, pose_dim = all_pred_poses.shape
all_pred_poses = all_pred_poses.view(-1, pose_dim)  # [M * (seq_len-1), 4]
all_gt_poses = all_gt_poses.view(-1, pose_dim)      # [M * (seq_len-1), 4]

# Compute metrics
ate, rpe_trans, rpe_rot = compute_trajectory_metrics(
    all_pred_poses.view(num_samples, seq_len_minus_1, pose_dim),
    all_gt_poses.view(num_samples, seq_len_minus_1, pose_dim)
)
print(f"Test ATE: {ate:.4f} m, RPE Trans: {rpe_trans:.4f} m, RPE Rot: {rpe_rot:.4f} deg")

# Convert predicted relative poses to absolute poses
pred_absolute_poses = relative_to_absolute_poses(all_pred_poses.cpu().numpy())  # Shape: [M * (seq_len-1), 3]

# Extract translations from absolute GT poses
gt_translations = gt_absolute_poses[:, :3, 3]  # Shape: [N, 3]

# Plot both trajectories on the same plot (x-z plane, top-down view)
fig = plt.figure(figsize=(10, 10))
plt.scatter(gt_translations[:, 0], gt_translations[:, 2], c=gt_translations[:, 2], s=20, alpha=0.5, cmap='viridis', label='Ground Truth')
plt.scatter(pred_absolute_poses[:, 0], pred_absolute_poses[:, 2], c=pred_absolute_poses[:, 2], s=20, alpha=0.5, cmap='magma', label='Predicted')
plt.xlabel('X (m)')
plt.ylabel('Z (m)')
plt.title(f'Trajectory Comparison - Sequence {test_sequence[0]}')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig(f'trajectory_seq_{test_sequence[0]}.png')
plt.show()