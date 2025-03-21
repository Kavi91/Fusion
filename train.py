import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb

from fusion_dataset import FusionDataset
from models import FusionLIVO, WeightedLoss

def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)

def get_dataset_and_model(model_name, config, seq_len, train=True):
    section = config[model_name]
    if "train_video" in section and "valid_video" in section:
        seqs = section["train_video"] if train else section["valid_video"]
    elif model_name == "lorcon_lo":
        seqs = section["data_seqs"].split(",") if train else section["test_seqs"].split(",")
    elif model_name == "deepvo" or model_name == "fusion":
        deepvo_section = config["deepvo"]
        seqs = deepvo_section["train_video"] if train else deepvo_section["valid_video"]
    else:
        raise KeyError(f"Unknown sequence definition for mode {model_name}")
    
    use_augmentation = section.get("use_augmentation", False) and train
    dataset = FusionDataset(config, seqs, seq_len, use_augmentation=use_augmentation)
    
    model = FusionLIVO(config, rgb_height=256, rgb_width=832, lidar_height=64, lidar_width=900, rnn_hidden_size=config["fusion"]["rnn_hidden_size"])
    return dataset, model

def train_model(mode, config, device):
    section = config[mode]
    seq_len = section.get("rnn_size", 2) if mode != "deepvo" else int((section["seq_len"][0] + section["seq_len"][1]) / 2)
    
    deepvo_section = config["deepvo"]
    num_workers = deepvo_section.get("num_workers", config.get("num_workers", 0))
    pin_mem = deepvo_section.get("pin_mem", config.get("pin_mem", False))
    
    train_dataset, model = get_dataset_and_model(mode, config, seq_len, train=True)
    val_dataset, _ = get_dataset_and_model(mode, config, seq_len, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=section["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=section["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    
    model = model.to(device)
    criterion = WeightedLoss(w_rot=section.get("w_rot", 1.0)).to(device)
    
    optim_config = section["optim"]
    lr = float(optim_config.get("lr", 0.0005))
    weight_decay = float(optim_config.get("weight_decay", 5e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler_config = section["scheduler"]
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config.get("step_size", 5), gamma=scheduler_config.get("gamma", 0.1))
    
    grad_clip_config = section["grad_clip"]
    grad_clip_max_norm = grad_clip_config.get("max_norm", 1.0)
    
    early_stopping_config = section["early_stopping"]
    patience = early_stopping_config.get("patience", 5)
    min_delta = early_stopping_config.get("min_delta", 0.01)
    
    scaler = GradScaler() if section.get("use_grad_scaler", False) else None
    use_autocast = to_bool(section.get("use_autocast", False))
    
    torch.autograd.set_detect_anomaly(True)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    epochs = section["epochs"]
    
    wandb.init(project=config["wandb_project"], config=config)
    
    for epoch in tqdm(range(epochs), desc="fusion Epochs"):
        model.train()
        train_loss = 0
        train_rmse = 0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            try:
                rgb_high, lidar_combined, targets = [b.to(device) for b in batch]
                
                optimizer.zero_grad()
                if use_autocast:
                    with autocast(enabled=True):
                        outputs = model(rgb_high, lidar_combined)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(rgb_high, lidar_combined)
                    loss = criterion(outputs, targets)
                
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                    optimizer.step()
                
                train_loss += loss.item()
                train_rmse += criterion.RMSEError(outputs, targets).item()
                train_batches += 1
            except Exception as e:
                print(f"Error in training loop: {str(e)}")
                raise
        
        train_loss /= train_batches
        train_rmse /= train_batches
        train_unweighted_loss = train_loss / (section.get("w_rot", 1.0) + 1)
        
        model.eval()
        val_loss = 0
        val_rmse = 0
        val_batches = 0
        ate = 0
        rpe = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation"):
                rgb_high, lidar_combined, targets = [b.to(device) for b in batch]
                
                if use_autocast:
                    with autocast(enabled=True):
                        outputs = model(rgb_high, lidar_combined)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(rgb_high, lidar_combined)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_rmse += criterion.RMSEError(outputs, targets).item()
                ate += torch.mean(torch.abs(outputs - targets)).item()
                rpe += torch.mean(torch.abs(outputs[:, 1:] - outputs[:, :-1] - (targets[:, 1:] - targets[:, :-1]))).item()
                val_batches += 1
        
        val_loss /= val_batches
        val_rmse /= val_batches
        val_unweighted_loss = val_loss / (section.get("w_rot", 1.0) + 1)
        ate /= val_batches
        rpe /= val_batches
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f} (Unweighted: {train_unweighted_loss:.6f}), "
              f"Val Loss: {val_loss:.6f} (Unweighted: {val_unweighted_loss:.6f}), "
              f"Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, "
              f"ATE: {ate:.4f}, RPE: {rpe:.4f}, Grad Norm: {grad_norm}, "
              f"Time: {time.time() - start_time:.2f}s, "
              f"ETA: {((time.time() - start_time) * (epochs - epoch - 1) / (epoch + 1)) / 3600:.2f}h, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "ate": ate,
            "rpe": rpe,
            "grad_norm": grad_norm,
            "lr": scheduler.get_last_lr()[0]
        })
        
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), section["model_path"])
            print(f"Saved best model to {section['model_path']}")
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")
        
        torch.save(model.state_dict(), f"models/fusion/fusion_model_epoch{epoch+1}.pth")
        print(f"Saved epoch {epoch+1} model to models/fusion/fusion_model_epoch{epoch+1}.pth")
        
        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break

def main():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    mode = "fusion"
    train_model(mode, config, device)

if __name__ == "__main__":
    start_time = time.time()
    main()