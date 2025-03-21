#!/home/kavi/LO-env/bin/python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import wandb
from models import FusionLIVO, WeightedLoss
from fusion_dataset import FusionDataset
from tqdm import tqdm
import time
import numpy as np

# Function to compute ATE and RPE
def compute_trajectory_metrics(pred_poses, gt_poses):
    """
    Compute ATE and RPE for a batch of predicted and ground truth poses.
    pred_poses: [batch_size, seq_len-1, 7] (translation + quaternion)
    gt_poses: [batch_size, seq_len-1, 7] (translation + quaternion)
    Returns: ATE (m), RPE translation (m), RPE rotation (degrees)
    """
    pred_poses = pred_poses.cpu().numpy()
    gt_poses = gt_poses.cpu().numpy()
    
    # Extract translations and quaternions
    pred_trans = pred_poses[:, :, :3]  # [batch_size, seq_len-1, 3]
    gt_trans = gt_poses[:, :, :3]  # [batch_size, seq_len-1, 3]
    pred_quat = pred_poses[:, :, 3:]  # [batch_size, seq_len-1, 4]
    gt_quat = gt_poses[:, :, 3:]  # [batch_size, seq_len-1, 4]
    
    # ATE: Compute RMSE of translation differences
    trans_diff = pred_trans - gt_trans  # [batch_size, seq_len-1, 3]
    trans_diff = trans_diff.reshape(-1, 3)  # [batch_size*(seq_len-1), 3]
    ate = np.sqrt(np.mean(np.sum(trans_diff**2, axis=1)))  # RMSE in meters
    
    # RPE: Compute relative pose errors between consecutive frames
    rpe_trans = []
    rpe_rot = []
    for b in range(pred_poses.shape[0]):
        for t in range(pred_poses.shape[1] - 1):  # seq_len-2 pairs
            # Predicted relative pose
            pred_trans1 = pred_trans[b, t]
            pred_trans2 = pred_trans[b, t+1]
            pred_quat1 = pred_quat[b, t]
            pred_quat2 = pred_quat[b, t+1]
            pred_rel_trans = pred_trans2 - pred_trans1
            
            # Ground truth relative pose
            gt_trans1 = gt_trans[b, t]
            gt_trans2 = gt_trans[b, t+1]
            gt_quat1 = gt_quat[b, t]
            gt_quat2 = gt_quat[b, t+1]
            gt_rel_trans = gt_trans2 - gt_trans1
            
            # RPE translation
            rpe_trans.append(np.linalg.norm(pred_rel_trans - gt_rel_trans))
            
            # RPE rotation (angle between quaternions in degrees)
            dot_product = np.abs(np.dot(pred_quat1, pred_quat2) * np.dot(gt_quat1, gt_quat2))
            dot_product = np.clip(dot_product, -1.0, 1.0)
            rpe_rot.append(2 * np.arccos(dot_product) * 180 / np.pi)
    
    rpe_trans = np.sqrt(np.mean(np.array(rpe_trans)**2))  # RMSE in meters
    rpe_rot = np.sqrt(np.mean(np.array(rpe_rot)**2))  # RMSE in degrees
    
    return ate, rpe_trans, rpe_rot

# Load configuration
with open("/home/kavi/Fusion/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize WandB
wandb.init(project=config["wandb_project"], config=config)

# Device configuration
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# Initialize model, criterion, optimizer
model = FusionLIVO(config).to(device)
criterion = WeightedLoss(w_rot=config["fusion"]["w_rot"]).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=float(config["fusion"]["optim"]["lr"]),
    weight_decay=float(config["fusion"]["optim"]["weight_decay"])
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config["fusion"]["scheduler"]["step_size"],
    gamma=config["fusion"]["scheduler"]["gamma"]
)

# Data loaders
train_dataset = FusionDataset(
    config,
    seqs=config["deepvo"]["train_video"],
    seq_len=config["fusion"]["rnn_size"],
    use_augmentation=True
)
val_dataset = FusionDataset(
    config,
    seqs=config["deepvo"]["valid_video"],
    seq_len=config["fusion"]["rnn_size"],
    use_augmentation=False
)
train_loader = DataLoader(
    train_dataset,
    batch_size=config["fusion"]["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["fusion"]["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    drop_last=True
)

# Training loop
best_val_loss = float('inf')
patience = config["fusion"]["early_stopping"]["patience"]
min_delta = config["fusion"]["early_stopping"]["min_delta"]
counter = 0
total_epochs = 10  # 5 frozen + 5 fine-tuning
epoch_times = []  # To track time per epoch for ETA

# First 5 epochs with frozen FlowNet
for param in model.flownet_rgb.parameters():
    param.requires_grad = False
for param in model.flownet_lidar.parameters():
    param.requires_grad = False

for epoch in range(5):
    start_time = time.time()
    model.train()
    train_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{5} (Frozen FlowNet) [Train]")
    for batch in train_bar:
        rgb_high, lidar_combined, targets = batch
        rgb_high, lidar_combined, targets = rgb_high.to(device), lidar_combined.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(rgb_high, lidar_combined)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["fusion"]["grad_clip"]["max_norm"])
        optimizer.step()
        train_loss += loss.item()
        train_bar.set_postfix({"train_loss": loss.item()})
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    val_ate, val_rpe_trans, val_rpe_rot = 0, 0, 0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{5} (Frozen FlowNet) [Val]")
    with torch.no_grad():
        for batch in val_bar:
            rgb_high, lidar_combined, targets = batch
            rgb_high, lidar_combined, targets = rgb_high.to(device), lidar_combined.to(device), targets.to(device)
            outputs = model(rgb_high, lidar_combined)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            # Compute ATE, RPE for the batch
            ate, rpe_trans, rpe_rot = compute_trajectory_metrics(outputs, targets)
            val_ate += ate
            val_rpe_trans += rpe_trans
            val_rpe_rot += rpe_rot
            
            val_bar.set_postfix({"val_loss": loss.item()})
    
    val_loss /= len(val_loader)
    val_ate /= len(val_loader)
    val_rpe_trans /= len(val_loader)
    val_rpe_rot /= len(val_loader)
    
    # Log to WandB
    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_ate": val_ate,
        "val_rpe_trans": val_rpe_trans,
        "val_rpe_rot": val_rpe_rot
    })
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Val ATE: {val_ate:.4f} m, Val RPE Trans: {val_rpe_trans:.4f} m, Val RPE Rot: {val_rpe_rot:.4f} deg")
    
    # Save best model
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), config["fusion"]["model_path"])
        print(f"Saved best model at epoch {epoch+1} with val_loss: {val_loss:.4f}")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break
    
    # Calculate ETA
    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = total_epochs - (epoch + 1)
    eta_seconds = remaining_epochs * avg_epoch_time
    eta_minutes = int(eta_seconds // 60)
    eta_secs = int(eta_seconds % 60)
    print(f"ETA: {eta_minutes:02d}:{eta_secs:02d} (mm:ss)")

# Unfreeze FlowNet for fine-tuning
for param in model.flownet_rgb.parameters():
    param.requires_grad = True
for param in model.flownet_lidar.parameters():
    param.requires_grad = True

# Fine-tune for remaining epochs
for epoch in range(5, 10):
    start_time = time.time()
    model.train()
    train_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{10} (Fine-Tuning) [Train]")
    for batch in train_bar:
        rgb_high, lidar_combined, targets = batch
        rgb_high, lidar_combined, targets = rgb_high.to(device), lidar_combined.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(rgb_high, lidar_combined)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["fusion"]["grad_clip"]["max_norm"])
        optimizer.step()
        train_loss += loss.item()
        train_bar.set_postfix({"train_loss": loss.item()})
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    val_ate, val_rpe_trans, val_rpe_rot = 0, 0, 0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{10} (Fine-Tuning) [Val]")
    with torch.no_grad():
        for batch in val_bar:
            rgb_high, lidar_combined, targets = batch
            rgb_high, lidar_combined, targets = rgb_high.to(device), lidar_combined.to(device), targets.to(device)
            outputs = model(rgb_high, lidar_combined)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            # Compute ATE, RPE for the batch
            ate, rpe_trans, rpe_rot = compute_trajectory_metrics(outputs, targets)
            val_ate += ate
            val_rpe_trans += rpe_trans
            val_rpe_rot += rpe_rot
            
            val_bar.set_postfix({"val_loss": loss.item()})
    
    val_loss /= len(val_loader)
    val_ate /= len(val_loader)
    val_rpe_trans /= len(val_loader)
    val_rpe_rot /= len(val_loader)
    
    # Log to WandB
    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_ate": val_ate,
        "val_rpe_trans": val_rpe_trans,
        "val_rpe_rot": val_rpe_rot
    })
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Val ATE: {val_ate:.4f} m, Val RPE Trans: {val_rpe_trans:.4f} m, Val RPE Rot: {val_rpe_rot:.4f} deg")
    
    # Save best model
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), config["fusion"]["model_path"])
        print(f"Saved best model at epoch {epoch+1} with val_loss: {val_loss:.4f}")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break
    
    # Calculate ETA
    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = total_epochs - (epoch + 1)
    eta_seconds = remaining_epochs * avg_epoch_time
    eta_minutes = int(eta_seconds // 60)
    eta_secs = int(eta_seconds % 60)
    print(f"ETA: {eta_minutes:02d}:{eta_secs:02d} (mm:ss)")

    scheduler.step()

wandb.finish()