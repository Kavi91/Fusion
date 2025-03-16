import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import pandas as pd
import wandb
from tqdm import tqdm
from params import par  # Updated import
from model import DeepVO  # Updated import
from data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset, get_partition_data_info  # Updated import

def compute_ate(gt_poses, pred_poses):
    gt_poses = np.array(gt_poses)
    pred_poses = np.array(pred_poses)
    if gt_poses.shape != pred_poses.shape:
        print(f"ATE Shape Mismatch: gt_poses {gt_poses.shape}, pred_poses {pred_poses.shape}")
        return np.nan
    return np.sqrt(np.mean(np.linalg.norm(gt_poses - pred_poses, axis=1) ** 2))

def compute_rpe(gt_poses, pred_poses):
    gt_poses = np.array(gt_poses)
    pred_poses = np.array(pred_poses)
    if gt_poses.shape != pred_poses.shape or gt_poses.shape[0] < 2:
        print(f"RPE Shape Mismatch: gt_poses {gt_poses.shape}, pred_poses {pred_poses.shape}")
        return np.nan
    gt_rel = np.diff(gt_poses, axis=0)
    pred_rel = np.diff(pred_poses, axis=0)
    return np.sqrt(np.mean(np.linalg.norm(gt_rel - pred_rel, axis=1) ** 2))

# Construct config dictionary from par
config = {
    "deepvo": {
        "pose_dir": par.pose_dir,
        "image_dir": par.image_dir,
        "calib_folder": getattr(par, "calib_folder", None),
        "timestamp_folder": getattr(par, "timestamp_folder", None),
        "train_video": par.train_video,
        "valid_video": par.valid_video,
        "resize_mode": par.resize_mode,
        "img_h": par.img_h,
        "img_w": par.img_w,
        "img_means": par.img_means,
        "img_stds": par.img_stds,
        "minus_point_5": par.minus_point_5,
        "seq_len": par.seq_len,
        "sample_times": par.sample_times,
        "batch_size": par.batch_size,
        "epochs": par.epochs,
        "rnn_hidden_size": par.rnn_hidden_size,
        "conv_dropout": par.conv_dropout,
        "rnn_dropout_out": par.rnn_dropout_out,
        "rnn_dropout_between": par.rnn_dropout_between,
        "clip": par.clip,
        "batch_norm": par.batch_norm,
        "optim": par.optim,
        "pin_mem": par.pin_mem,
        "model_path": par.save_model_path + "_best.pth",
        "train_data_info_path": par.train_data_info_path,
        "valid_data_info_path": par.valid_data_info_path
    }
}

wandb.init(project="Fusion", name="DeepVO-Training-0", config=vars(par))

# Fix sequence length to a single value (average of range)
seq_len = int((par.seq_len[0] + par.seq_len[1]) / 2)
print(f"Using fixed sequence length: {seq_len}")

# Delete cached dataframes to ensure fresh computation
if os.path.exists(par.train_data_info_path):
    os.remove(par.train_data_info_path)
    print(f"Deleted cached train_data_info_path: {par.train_data_info_path}")
if os.path.exists(par.valid_data_info_path):
    os.remove(par.valid_data_info_path)
    print(f"Deleted cached valid_data_info_path: {par.valid_data_info_path}")

# Pass the full config dictionary
train_df = pd.read_pickle(par.train_data_info_path) if os.path.isfile(par.train_data_info_path) else get_data_info(folder_list=par.train_video, seq_len_range=[seq_len, seq_len], overlap=1, sample_times=par.sample_times, config=config)
print(f"train_df shape: {train_df.shape}, columns: {train_df.columns}")
# Debug: Verify sequence lengths in train_df
seq_lengths = train_df['seq_len'].unique()
print(f"Unique sequence lengths in train_df (before filtering): {seq_lengths}")
# Filter train_df to include only sequences of length seq_len
train_df = train_df[train_df['seq_len'] == seq_len]
print(f"train_df shape after filtering: {train_df.shape}")
seq_lengths = train_df['seq_len'].unique()
print(f"Unique sequence lengths in train_df (after filtering): {seq_lengths}")
if len(seq_lengths) != 1 or seq_lengths[0] != seq_len:
    raise ValueError(f"Expected sequence length {seq_len}, but found lengths {seq_lengths}")

if train_df.empty:
    raise ValueError("train_df is empty after filtering, check image and pose file paths")

valid_df = pd.read_pickle(par.valid_data_info_path) if os.path.isfile(par.valid_data_info_path) else get_data_info(folder_list=par.valid_video, seq_len_range=[seq_len, seq_len], overlap=1, sample_times=par.sample_times, config=config)
print(f"valid_df shape: {valid_df.shape}, columns: {valid_df.columns}")
# Debug: Verify sequence lengths in valid_df
seq_lengths = valid_df['seq_len'].unique()
print(f"Unique sequence lengths in valid_df (before filtering): {seq_lengths}")
# Filter valid_df to include only sequences of length seq_len
valid_df = valid_df[valid_df['seq_len'] == seq_len]
print(f"valid_df shape after filtering: {valid_df.shape}")
seq_lengths = valid_df['seq_len'].unique()
print(f"Unique sequence lengths in valid_df (after filtering): {seq_lengths}")
if len(seq_lengths) != 1 or seq_lengths[0] != seq_len:
    raise ValueError(f"Expected sequence length {seq_len}, but found lengths {seq_lengths}")

if valid_df.empty:
    raise ValueError("valid_df is empty after filtering, check image and pose file paths")

train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
valid_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)

# Set num_workers=0 to avoid multi-worker issues (temporary)
train_dl = DataLoader(train_dataset, batch_sampler=SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True), num_workers=0, pin_memory=par.pin_mem)
valid_dl = DataLoader(valid_dataset, batch_sampler=SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True), num_workers=0, pin_memory=par.pin_mem)
print(f"Number of training batches: {len(train_dl)}")
print(f"Number of validation batches: {len(valid_dl)}")

M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=par.optim['lr']) if par.optim['opt'] == 'Adam' else torch.optim.Adagrad(M_deepvo.parameters(), lr=par.optim['lr'])

min_loss_v, total_train_time, start_time = float("inf"), 0, time.time()

for epoch in range(par.epochs):
    epoch_start_time = time.time()
    print('=' * 50)

    M_deepvo.train()
    train_loss, grad_norm_total, train_bar = 0.0, 0.0, tqdm(train_dl, desc=f"Epoch {epoch+1}/{par.epochs} [Training]", leave=True)
    
    for _, inputs, targets in train_bar:
        print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
        inputs, targets = inputs.to("cuda"), targets.to("cuda")
        optimizer.zero_grad()
        loss = M_deepvo.step(inputs, targets, optimizer)
        
        grad_norm = 0
        for p in M_deepvo.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        grad_norm_total += grad_norm
        
        loss_value = loss.data.cpu().numpy()
        train_loss += float(loss_value)
        train_bar.set_postfix({'loss': f"{loss_value:.6f}", 'grad_norm': f"{grad_norm:.4f}"})

    avg_train_loss = train_loss / len(train_dl)
    avg_grad_norm = grad_norm_total / len(train_dl)

    M_deepvo.eval()
    val_loss, gt_poses, pred_poses, val_bar = 0.0, [], [], tqdm(valid_dl, desc=f"Epoch {epoch+1}/{par.epochs} [Validation]", leave=True)
    
    with torch.no_grad():
        for _, inputs, targets in val_bar:
            print(f"Validation Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs = M_deepvo(inputs)
            gt_poses.append(targets.cpu().numpy())
            pred_poses.append(outputs.cpu().numpy())
            val_loss += float(M_deepvo.get_loss(inputs, targets).data.cpu().numpy())
            val_bar.set_postfix(loss=f"{val_loss:.6f}")

    avg_val_loss = val_loss / len(valid_dl)

    min_length = min(min(poses.shape[1] for poses in gt_poses), min(poses.shape[1] for poses in pred_poses))
    gt_poses_trimmed = [poses[:, :min_length] for poses in gt_poses if poses.shape[1] >= min_length]
    pred_poses_trimmed = [poses[:, :min_length] for poses in pred_poses if poses.shape[1] >= min_length]

    if len(gt_poses_trimmed) == 0 or len(pred_poses_trimmed) == 0:
        ate_result, rpe_result = np.nan, np.nan
    else:
        gt_poses_array = np.vstack(gt_poses_trimmed)
        pred_poses_array = np.vstack(pred_poses_trimmed)
        ate_result = compute_ate(gt_poses_array, pred_poses_array)
        rpe_result = compute_rpe(gt_poses_array, pred_poses_array)

    epoch_duration = time.time() - epoch_start_time
    total_train_time += epoch_duration
    avg_epoch_time = total_train_time / (epoch + 1)
    eta_seconds = avg_epoch_time * (par.epochs - epoch - 1)
    eta_formatted = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

    wandb.log({
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "ATE": ate_result,
        "RPE": rpe_result,
        "grad_norm": avg_grad_norm,
        "epoch_time": epoch_duration,
        "ETA": eta_seconds,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "GPU_usage": torch.cuda.memory_allocated() / 1e9
    })

    print(f"Epoch {epoch+1}/{par.epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
          f"Grad Norm: {avg_grad_norm:.4f}, ATE: {ate_result:.4f}, RPE: {rpe_result:.4f}, ETA: {eta_formatted}")

    if avg_val_loss < min_loss_v:
        min_loss_v = avg_val_loss
        torch.save(M_deepvo.state_dict(), f"{par.save_model_path}_best.pth")

    torch.save(M_deepvo.state_dict(), f"{par.save_model_path}_epoch{epoch+1}.pth")

wandb.finish()