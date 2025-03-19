import yaml
import torch
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
import os
import torch.nn.functional as F
from deepvo.data_helper import get_data_info, ImageSequenceDataset, SortedRandomBatchSampler
from lorcon_lo.utils.process_data import LoRCoNLODataset, count_seq_sizes, process_input_data
from models import DeepVO, LoRCoNLO, WeightedLoss, FusionLIVO
from fusion_dataset import FusionDataset
from torch.cuda.amp import GradScaler, autocast

def compute_ate(gt_poses, pred_poses):
    gt_poses, pred_poses = np.array(gt_poses), np.array(pred_poses)
    if gt_poses.shape != pred_poses.shape:
        print(f"ATE Shape Mismatch: gt_poses {gt_poses.shape}, pred_poses {pred_poses.shape}")
        return np.nan
    return np.sqrt(np.mean(np.linalg.norm(gt_poses - pred_poses, axis=1) ** 2))

def compute_rpe(gt_poses, pred_poses):
    gt_poses, pred_poses = np.array(gt_poses), np.array(pred_poses)
    if gt_poses.shape != pred_poses.shape or gt_poses.shape[0] < 2:
        print(f"RPE Shape Mismatch or Too Short: gt_poses {gt_poses.shape}, pred_poses {pred_poses.shape}")
        return np.nan
    gt_rel, pred_rel = np.diff(gt_poses, axis=0), np.diff(pred_poses, axis=0)
    return np.sqrt(np.mean(np.linalg.norm(gt_rel - pred_rel, axis=1) ** 2))

def compute_unweighted_mse(pred, target):
    return F.mse_loss(pred, target).item()

def get_dataset_and_model(model_name, config, seq_len, train=True):
    section = config[model_name]
    if model_name == "deepvo":
        seq_len = int((section["seq_len"][0] + section["seq_len"][1]) / 2)
        folder_list = [os.path.join(f, "image_02") for f in section["train_video" if train else "valid_video"]]
        df = get_data_info(folder_list=folder_list, seq_len_range=[seq_len, seq_len], 
                           overlap=1, sample_times=section["sample_times"], config=config)
        df = df[df['seq_len'] == seq_len]
        dataset = ImageSequenceDataset(df, resize_mode=section["resize_mode"], new_size=(section["img_w"], section["img_h"]), 
                                       img_mean=section["img_means"], img_std=section["img_stds"], minus_point_5=section["minus_point_5"], config=config)
        model = DeepVO(section["img_h"], section["img_w"], batchNorm=section["batch_norm"], conv_dropout=section["conv_dropout"], 
                       rnn_hidden_size=section["rnn_hidden_size"], rnn_dropout_out=section["rnn_dropout_out"], 
                       rnn_dropout_between=section["rnn_dropout_between"], clip=section["clip"])
    elif model_name == "lorcon_lo":
        seq_sizes = count_seq_sizes(section["preprocessed_folder"], section["data_seqs"].split(","), {})
        Y_data = process_input_data(section["preprocessed_folder"], section["relative_pose_folder"], section["data_seqs"].split(","), seq_sizes)
        train_idx, test_idx = np.array([], dtype=int), np.array([], dtype=int)
        test_seqs = section["test_seqs"].split(",")
        start_idx = 0
        for seq in section["data_seqs"].split(","):
            seq_len = seq_sizes.get(seq, 0)
            end_idx = start_idx + seq_len - 1
            max_idx = end_idx - (section["rnn_size"] - 1)
            idx_range = np.arange(start_idx, min(max_idx, start_idx + seq_len - section["rnn_size"]), dtype=int)
            if train and seq not in test_seqs:
                train_idx = np.append(train_idx, idx_range)
            elif not train and seq in test_seqs:
                test_idx = np.append(test_idx, idx_range)
            start_idx += seq_len - 1
        idx = train_idx if train else test_idx
        print(f"{model_name} {'train' if train else 'valid'} idx: {idx.min()} to {idx.max()}, len={len(idx)}, Y_data len={len(Y_data)}")
        modalities = ["depth", "intensity", "normals"]
        print(f"{model_name} modalities enabled: {', '.join(modalities)}")
        dataset = LoRCoNLODataset(section["preprocessed_folder"], Y_data, idx, seq_sizes, section["rnn_size"], 
                                  section["image_width"], section["image_height"], section["depth_name"], 
                                  section["intensity_name"], section["normal_name"], section["dni_size"], section["normal_size"])
        model = LoRCoNLO(batch_size=section["batch_size"], batchNorm=False)
    elif model_name == "fusion":
        deepvo_seqs = config["deepvo"]["train_video" if train else "valid_video"]
        lorcon_seqs = config["lorcon_lo"]["data_seqs"]
        print(f"DeepVO seqs (raw): {deepvo_seqs}")
        print(f"LoRCoN seqs (raw): {lorcon_seqs}")
        lorcon_seqs_split = lorcon_seqs.split(",")
        print(f"LoRCoN seqs (split): {lorcon_seqs_split}")
        seqs = sorted(list(set(deepvo_seqs) & set(lorcon_seqs_split)))
        print(f"Fusion sequences: {seqs}")
        dataset = FusionDataset(config, seqs, seq_len)
        modalities = [m for m, enabled in config["fusion"]["modalities"].items() if enabled]
        print(f"{model_name} modalities enabled: {', '.join(modalities)}")
        model = FusionLIVO(config, rgb_height=184, rgb_width=608, lidar_height=64, lidar_width=900)
    return dataset, model

def train_model(model_name, config, device):
    wandb_project = config.get("wandb_project", "FusionLIVO")
    wandb.init(project=wandb_project, name=f"{model_name}-Training-0", config=config[model_name])
    section = config[model_name]
    seq_len = section.get("rnn_size", 2) if model_name != "deepvo" else int((section["seq_len"][0] + section["seq_len"][1]) / 2)
    
    train_dataset, model = get_dataset_and_model(model_name, config, seq_len, train=True)
    valid_dataset, _ = get_dataset_and_model(model_name, config, seq_len, train=False)
    model = model.to(device)
    
    train_dl = DataLoader(train_dataset, batch_size=section["batch_size"], shuffle=True, num_workers=config["num_workers"])
    valid_dl = DataLoader(valid_dataset, batch_size=section["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    optimizer = optim.Adam(model.parameters(), lr=section.get("optim", {}).get("lr", 0.0005))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    criterion = WeightedLoss().to(device)
    scaler = GradScaler() if section.get("use_grad_scaler", False) else None
    
    epochs = section["epochs"]
    min_loss_v = float("inf")
    
    if model_name == "deepvo":
        model_dir = "models/deepvo"
    elif model_name == "fusion":
        model_dir = "models/fusion"
    else:  # lorcon_lo
        model_dir = os.path.join(section.get("cp_folder", "checkpoints"), config["dataset"], str(len(next(os.walk(section.get("cp_folder", "checkpoints")))[1])).zfill(4))
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in tqdm(range(epochs), desc=f"{model_name} Epochs"):
        t_start = time.time()
        model.train()
        train_loss, train_loss_unweighted, rmse_train, grad_norm = 0.0, 0.0, 0.0, 0.0
        
        for batch in train_dl:
            inputs = batch[1] if model_name == "deepvo" else batch[0] if model_name == "lorcon_lo" else (batch[0], batch[1])
            targets = batch[2]
            if model_name == "fusion":
                rgb_high = inputs[0].float().to(device)
                lidar_combined = inputs[1].float().to(device)
            else:
                inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            optimizer.zero_grad()
            
            if section.get("use_autocast", False):
                with autocast():
                    if model_name == "fusion":
                        outputs = model(rgb_high, lidar_combined)  # Unpack explicitly
                    else:
                        outputs = model(inputs)
                    loss = criterion(outputs, targets[:, 1:, :] if model_name == "deepvo" else targets)
            else:
                if model_name == "fusion":
                    outputs = model(rgb_high, lidar_combined)  # Unpack explicitly
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, targets[:, 1:, :] if model_name == "deepvo" else targets)
            
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            train_loss += loss.item()
            train_loss_unweighted += compute_unweighted_mse(outputs, targets[:, 1:, :] if model_name == "deepvo" else targets)
            rmse_train += WeightedLoss.RMSEError(outputs, targets[:, 1:, :] if model_name == "deepvo" else targets).item()
            grad_norm += sum(p.grad.norm(2).item() for p in model.parameters() if p.grad is not None)
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_dl)
        avg_train_loss_unweighted = train_loss_unweighted / len(train_dl)
        avg_rmse_train = rmse_train / len(train_dl)
        avg_grad_norm = grad_norm / len(train_dl)
        
        model.eval()
        val_loss, val_loss_unweighted, rmse_val, gt_poses, pred_poses = 0.0, 0.0, 0.0, [], []
        with torch.no_grad():
            for batch in valid_dl:
                inputs = batch[1] if model_name == "deepvo" else batch[0] if model_name == "lorcon_lo" else (batch[0], batch[1])
                targets = batch[2]
                if model_name == "fusion":
                    rgb_high = inputs[0].float().to(device)
                    lidar_combined = inputs[1].float().to(device)
                else:
                    inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                if section.get("use_autocast", False):
                    with autocast():
                        if model_name == "fusion":
                            outputs = model(rgb_high, lidar_combined)  # Unpack explicitly
                        else:
                            outputs = model(inputs)
                        loss = criterion(outputs, targets[:, 1:, :] if model_name == "deepvo" else targets)
                else:
                    if model_name == "fusion":
                        outputs = model(rgb_high, lidar_combined)  # Unpack explicitly
                    else:
                        outputs = model(inputs)
                    loss = criterion(outputs, targets[:, 1:, :] if model_name == "deepvo" else targets)
                val_loss += loss.item()
                val_loss_unweighted += compute_unweighted_mse(outputs, targets[:, 1:, :] if model_name == "deepvo" else targets)
                rmse_val += WeightedLoss.RMSEError(outputs, targets[:, 1:, :] if model_name == "deepvo" else targets).item()
                gt = targets[:, 1:, :] if model_name == "deepvo" else targets
                gt_poses.append(gt.cpu().numpy())
                pred_poses.append(outputs.cpu().numpy())
        
        avg_val_loss = val_loss / len(valid_dl)
        avg_val_loss_unweighted = val_loss_unweighted / len(valid_dl)
        avg_rmse_val = rmse_val / len(valid_dl)
        ate = compute_ate(np.vstack(gt_poses), np.vstack(pred_poses))
        rpe = compute_rpe(np.vstack(gt_poses), np.vstack(pred_poses))
        
        epoch_duration = time.time() - t_start
        eta = epoch_duration * (epochs - epoch - 1)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        
        wandb.log({
            "train_loss": avg_train_loss_unweighted, "val_loss": avg_val_loss_unweighted,
            "train_loss_weighted": avg_train_loss, "val_loss_weighted": avg_val_loss,
            "train_rmse": avg_rmse_train, "val_rmse": avg_rmse_val,
            "ate": ate, "rpe": rpe, "grad_norm": avg_grad_norm,
            "epoch_time": epoch_duration, "eta": eta,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "gpu_usage": torch.cuda.memory_allocated() / 1e9
        })
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f} (Unweighted: {avg_train_loss_unweighted:.6f}), "
              f"Val Loss: {avg_val_loss:.6f} (Unweighted: {avg_val_loss_unweighted:.6f}), Train RMSE: {avg_rmse_train:.4f}, "
              f"Val RMSE: {avg_rmse_val:.4f}, ATE: {ate:.4f}, RPE: {rpe:.4f}, Grad Norm: {avg_grad_norm:.4f}, "
              f"Time: {epoch_duration:.2f}s, ETA: {eta_str}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        try:
            if avg_val_loss < min_loss_v:
                min_loss_v = avg_val_loss
                best_path = os.path.join(model_dir, f"{model_name}_model_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"Saved best model to {best_path}")
            if model_name == "deepvo" or model_name == "fusion":
                epoch_path = os.path.join(model_dir, f"{model_name}_model_epoch{epoch+1}.pth")
                torch.save(model.state_dict(), epoch_path)
                print(f"Saved epoch {epoch+1} model to {epoch_path}")
            elif model_name == "lorcon_lo":
                cp_path = os.path.join(model_dir, f"cp-{epoch:04d}.pt")
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                            'loss': avg_train_loss}, cp_path)
                print(f"Saved checkpoint to {cp_path}")
                if epoch % section["checkpoint_epoch"] == 0 and epoch > 0:
                    special_path = os.path.join(model_dir, f"cp-special-{epoch:04d}.pt")
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                                'loss': avg_train_loss}, special_path)
                    print(f"Saved special checkpoint to {special_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
    
    print(f"{model_name} training completed successfully!")
    wandb.finish()

def main():
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config.yaml: {e}")
        return
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
    mode_map = {"camera": "deepvo", "lidar": "lorcon_lo", "fusion": "fusion"}
    
    for config_key, mode in mode_map.items():
        if config.get(f"use_{config_key}", False):
            train_model(mode, config, device)

if __name__ == "__main__":
    main()