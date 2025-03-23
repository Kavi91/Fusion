#!/home/kavi/LO-env/bin/python3
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml
import wandb
from models import FusionLIVO, WeightedLoss
from fusion_dataset import FusionDataset
from tqdm import tqdm
import time
import numpy as np

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

# Initialize WandB
wandb.init(project=config["wandb_project"], config=config)

# Device configuration
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# Initialize model, criterion, optimizer
model = FusionLIVO(config).to(device)
criterion = WeightedLoss(device=device).to(device)

# Create parameter groups for different learning rates
flownet_params = []
if config["fusion"]["modalities"]["use_rgb_left"]:
    flownet_params.extend(list(model.flownet_rgb_left.parameters()))
if config["fusion"]["modalities"]["use_rgb_right"]:
    flownet_params.extend(list(model.flownet_rgb_right.parameters()))
if config["fusion"]["modalities"]["use_lidar"]:
    flownet_params.extend(list(model.flownet_lidar.parameters()))
other_params = [
    param for name, param in model.named_parameters()
    if not any(name.startswith(prefix) for prefix in ["flownet_rgb_left", "flownet_rgb_right", "flownet_lidar"])
]
optimizer = torch.optim.Adagrad([
    {"params": flownet_params, "lr": float(config["fusion"]["optim"]["lr"])},
    {"params": other_params, "lr": float(config["fusion"]["optim"]["lr"])},
], weight_decay=float(config["fusion"]["optim"]["weight_decay"]))

# Data loaders with custom collate function
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
    drop_last=True,
    collate_fn=custom_collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["fusion"]["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    drop_last=True,
    collate_fn=custom_collate_fn
)

# Training loop
best_val_loss = float('inf')
patience = config["fusion"]["early_stopping"]["patience"]
min_delta = config["fusion"]["early_stopping"]["min_delta"]
counter = 0
total_epochs = config["fusion"]["epochs"]
epoch_times = []

# First 2 epochs with frozen FlowNet
for param in flownet_params:
    param.requires_grad = False

for epoch in range(2):
    start_time = time.time()
    model.train()
    train_loss = 0
    train_loss_t = 0
    train_loss_yaw = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/2 (Frozen FlowNet) [Train]")
    for batch_idx, batch in enumerate(train_bar):
        rgb_left, rgb_right, lidar_combined, targets = batch
        rgb_left = rgb_left.to(device) if rgb_left is not None else None
        rgb_right = rgb_right.to(device) if rgb_right is not None else None
        lidar_combined = lidar_combined.to(device) if lidar_combined is not None else None
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(rgb_left, rgb_right, lidar_combined)
        
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            raise ValueError(f"NaN or Inf detected in model outputs at batch {batch_idx}")
        
        loss, L_t, L_yaw = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["fusion"]["grad_clip"]["max_norm"])
        optimizer.step()
        train_loss += loss.item()
        train_loss_t += L_t.item()
        train_loss_yaw += L_yaw.item()
        train_bar.set_postfix({"train_loss": loss.item(), "train_loss_t": L_t.item(), "train_loss_yaw": L_yaw.item()})
    train_loss /= len(train_loader)
    train_loss_t /= len(train_loader)
    train_loss_yaw /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    val_loss_t = 0
    val_loss_yaw = 0
    val_ate, val_rpe_trans, val_rpe_rot = 0, 0, 0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/2 (Frozen FlowNet) [Val]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_bar):
            rgb_left, rgb_right, lidar_combined, targets = batch
            rgb_left = rgb_left.to(device) if rgb_left is not None else None
            rgb_right = rgb_right.to(device) if rgb_right is not None else None
            lidar_combined = lidar_combined.to(device) if lidar_combined is not None else None
            targets = targets.to(device)
            outputs = model(rgb_left, rgb_right, lidar_combined)
            
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                raise ValueError(f"NaN or Inf detected in model outputs at batch {batch_idx} (validation)")
            
            loss, L_t, L_yaw = criterion(outputs, targets)
            val_loss += loss.item()
            val_loss_t += L_t.item()
            val_loss_yaw += L_yaw.item()
            
            ate, rpe_trans, rpe_rot = compute_trajectory_metrics(outputs, targets)
            val_ate += ate
            val_rpe_trans += rpe_trans
            val_rpe_rot += rpe_rot
            
            val_bar.set_postfix({"val_loss": loss.item(), "val_loss_t": L_t.item(), "val_loss_yaw": L_yaw.item()})
    
    val_loss /= len(val_loader)
    val_loss_t /= len(val_loader)
    val_loss_yaw /= len(val_loader)
    val_ate /= len(val_loader)
    val_rpe_trans /= len(val_loader)
    val_rpe_rot /= len(val_loader)
    
    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "train_loss_t": train_loss_t,
        "train_loss_yaw": train_loss_yaw,
        "val_loss": val_loss,
        "val_loss_t": val_loss_t,
        "val_loss_yaw": val_loss_yaw,
        "val_ate": val_ate,
        "val_rpe_trans": val_rpe_trans,
        "val_rpe_rot": val_rpe_rot,
    })
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} (T: {train_loss_t:.4f}, Yaw: {train_loss_yaw:.4f}), "
          f"Val Loss: {val_loss:.4f} (T: {val_loss_t:.4f}, Yaw: {val_loss_yaw:.4f}), "
          f"Val ATE: {val_ate:.4f} m, Val RPE Trans: {val_rpe_trans:.4f} m, Val RPE Rot: {val_rpe_rot:.4f} deg")
    
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
    
    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = total_epochs - (epoch + 1)
    eta_seconds = remaining_epochs * avg_epoch_time
    eta_minutes = int(eta_seconds // 60)
    eta_secs = int(eta_seconds % 60)
    print(f"ETA: {eta_minutes:02d}:{eta_secs:02d} (mm:ss)")

# Unfreeze FlowNet for fine-tuning
for param in flownet_params:
    param.requires_grad = True

# Fine-tune for remaining epochs
for epoch in range(2, total_epochs):
    start_time = time.time()
    model.train()
    train_loss = 0
    train_loss_t = 0
    train_loss_yaw = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} (Fine-Tuning) [Train]")
    for batch_idx, batch in enumerate(train_bar):
        rgb_left, rgb_right, lidar_combined, targets = batch
        rgb_left = rgb_left.to(device) if rgb_left is not None else None
        rgb_right = rgb_right.to(device) if rgb_right is not None else None
        lidar_combined = lidar_combined.to(device) if lidar_combined is not None else None
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(rgb_left, rgb_right, lidar_combined)
        
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            raise ValueError(f"NaN or Inf detected in model outputs at batch {batch_idx}")
        
        loss, L_t, L_yaw = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["fusion"]["grad_clip"]["max_norm"])
        optimizer.step()
        train_loss += loss.item()
        train_loss_t += L_t.item()
        train_loss_yaw += L_yaw.item()
        train_bar.set_postfix({"train_loss": loss.item(), "train_loss_t": L_t.item(), "train_loss_yaw": L_yaw.item()})
    train_loss /= len(train_loader)
    train_loss_t /= len(train_loader)
    train_loss_yaw /= len(train_loader)
    
    model.eval()
    val_loss = 0
    val_loss_t = 0
    val_loss_yaw = 0
    val_ate, val_rpe_trans, val_rpe_rot = 0, 0, 0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} (Fine-Tuning) [Val]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_bar):
            rgb_left, rgb_right, lidar_combined, targets = batch
            rgb_left = rgb_left.to(device) if rgb_left is not None else None
            rgb_right = rgb_right.to(device) if rgb_right is not None else None
            lidar_combined = lidar_combined.to(device) if lidar_combined is not None else None
            targets = targets.to(device)
            outputs = model(rgb_left, rgb_right, lidar_combined)
            
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                raise ValueError(f"NaN or Inf detected in model outputs at batch {batch_idx} (validation)")
            
            loss, L_t, L_yaw = criterion(outputs, targets)
            val_loss += loss.item()
            val_loss_t += L_t.item()
            val_loss_yaw += L_yaw.item()
            
            ate, rpe_trans, rpe_rot = compute_trajectory_metrics(outputs, targets)
            val_ate += ate
            val_rpe_trans += rpe_trans
            val_rpe_rot += rpe_rot
            
            val_bar.set_postfix({"val_loss": loss.item(), "val_loss_t": L_t.item(), "val_loss_yaw": L_yaw.item()})
    
    val_loss /= len(val_loader)
    val_loss_t /= len(val_loader)
    val_loss_yaw /= len(val_loader)
    val_ate /= len(val_loader)
    val_rpe_trans /= len(val_loader)
    val_rpe_rot /= len(val_loader)
    
    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "train_loss_t": train_loss_t,
        "train_loss_yaw": train_loss_yaw,
        "val_loss": val_loss,
        "val_loss_t": val_loss_t,
        "val_loss_yaw": val_loss_yaw,
        "val_ate": val_ate,
        "val_rpe_trans": val_rpe_trans,
        "val_rpe_rot": val_rpe_rot,
    })
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} (T: {train_loss_t:.4f}, Yaw: {train_loss_yaw:.4f}), "
          f"Val Loss: {val_loss:.4f} (T: {val_loss_t:.4f}, Yaw: {val_loss_yaw:.4f}), "
          f"Val ATE: {val_ate:.4f} m, Val RPE Trans: {val_rpe_trans:.4f} m, Val RPE Rot: {val_rpe_rot:.4f} deg")
    
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
    
    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = total_epochs - (epoch + 1)
    eta_seconds = remaining_epochs * avg_epoch_time
    eta_minutes = int(eta_seconds // 60)
    eta_secs = int(eta_seconds % 60)
    print(f"ETA: {eta_minutes:02d}:{eta_secs:02d} (mm:ss)")

wandb.finish()