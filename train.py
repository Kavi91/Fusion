import yaml
import torch
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
from torch.utils.tensorboard import SummaryWriter
import os
from deepvo.data_helper import get_data_info, ImageSequenceDataset, SortedRandomBatchSampler
from lorcon_lo.utils.process_data import LoRCoNLODataset, count_seq_sizes, process_input_data
from models import DeepVO, LoRCoNLO, WeightedLoss, FusionLIVO
from fusion_dataset import FusionDataset
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

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

def compute_unweighted_mse(pred, target):
    """Compute unweighted MSE for common comparison across models."""
    return F.mse_loss(pred, target).item()

def train_deepvo(config, device):
    wandb.init(project="FusionLIVO", name="DeepVO-Training-0", config=config["deepvo"])
    seq_len = int((config["deepvo"]["seq_len"][0] + config["deepvo"]["seq_len"][1]) / 2)
    print(f"Using fixed sequence length: {seq_len}")

    for path in [config["deepvo"]["train_data_info_path"], config["deepvo"]["valid_data_info_path"]]:
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted cached path: {path}")

    train_df = get_data_info(folder_list=config["deepvo"]["train_video"], seq_len_range=[seq_len, seq_len], overlap=1, sample_times=config["deepvo"]["sample_times"], config=config)
    train_df = train_df[train_df['seq_len'] == seq_len]
    valid_df = get_data_info(folder_list=config["deepvo"]["valid_video"], seq_len_range=[seq_len, seq_len], overlap=1, sample_times=config["deepvo"]["sample_times"], config=config)
    valid_df = valid_df[valid_df['seq_len'] == seq_len]

    if train_df.empty or valid_df.empty:
        raise ValueError("Train or valid dataframe is empty; check image and pose file paths")

    train_dataset = ImageSequenceDataset(train_df, resize_mode=config["deepvo"]["resize_mode"], new_size=(config["deepvo"]["img_w"], config["deepvo"]["img_h"]), img_mean=config["deepvo"]["img_means"], img_std=config["deepvo"]["img_stds"], minus_point_5=config["deepvo"]["minus_point_5"], config=config)
    valid_dataset = ImageSequenceDataset(valid_df, resize_mode=config["deepvo"]["resize_mode"], new_size=(config["deepvo"]["img_w"], config["deepvo"]["img_h"]), img_mean=config["deepvo"]["img_means"], img_std=config["deepvo"]["img_stds"], minus_point_5=config["deepvo"]["minus_point_5"], config=config)

    train_dl = DataLoader(train_dataset, batch_sampler=SortedRandomBatchSampler(train_df, config["deepvo"]["batch_size"], drop_last=True), num_workers=config["num_workers"], pin_memory=config["deepvo"]["pin_mem"])
    valid_dl = DataLoader(valid_dataset, batch_sampler=SortedRandomBatchSampler(valid_df, config["deepvo"]["batch_size"], drop_last=True), num_workers=config["num_workers"], pin_memory=config["deepvo"]["pin_mem"])
    print(f"Number of training batches: {len(train_dl)}, validation batches: {len(valid_dl)}")

    model = DeepVO(config["deepvo"]["img_h"], config["deepvo"]["img_w"], batchNorm=config["deepvo"]["batch_norm"], conv_dropout=config["deepvo"]["conv_dropout"], rnn_hidden_size=config["deepvo"]["rnn_hidden_size"], rnn_dropout_out=config["deepvo"]["rnn_dropout_out"], rnn_dropout_between=config["deepvo"]["rnn_dropout_between"], clip=config["deepvo"]["clip"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["deepvo"]["optim"]["lr"]) if config["deepvo"]["optim"]["opt"] == "Adam" else optim.Adagrad(model.parameters(), lr=config["deepvo"]["optim"]["lr"])
    if os.path.exists(config["deepvo"]["model_path"]):
        model.load_state_dict(torch.load(config["deepvo"]["model_path"]))

    min_loss_v, total_train_time = float("inf"), 0
    for epoch in range(config["deepvo"]["epochs"]):
        epoch_start_time = time.time()
        print('=' * 50)

        model.train()
        train_loss, train_loss_unweighted, grad_norm_total = 0.0, 0.0, 0.0
        for batch_idx, (_, inputs, targets) in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{config['deepvo']['epochs']} [Training]", leave=True)):
            if batch_idx == 0:
                print(f"Train Batch 0 - Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            loss = model.step(inputs, targets, optimizer)
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            grad_norm_total += grad_norm
            train_loss += float(loss.data.cpu().numpy())
            with torch.no_grad():
                outputs = model(inputs)
                train_loss_unweighted += compute_unweighted_mse(outputs, targets[:, 1:, :])

        avg_train_loss = train_loss / len(train_dl)
        avg_train_loss_unweighted = train_loss_unweighted / len(train_dl)
        avg_grad_norm = grad_norm_total / len(train_dl)

        model.eval()
        val_loss, val_loss_unweighted, gt_poses, pred_poses = 0.0, 0.0, [], []
        for batch_idx, (_, inputs, targets) in enumerate(tqdm(valid_dl, desc=f"Epoch {epoch+1}/{config['deepvo']['epochs']} [Validation]", leave=True)):
            if batch_idx == 0:
                print(f"Valid Batch 0 - Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            with torch.no_grad():
                outputs = model(inputs)
                gt_poses.append(targets.cpu().numpy())
                pred_poses.append(outputs.cpu().numpy())
                val_loss += float(model.get_loss(inputs, targets).data.cpu().numpy())
                val_loss_unweighted += compute_unweighted_mse(outputs, targets[:, 1:, :])

        avg_val_loss = val_loss / len(valid_dl)
        avg_val_loss_unweighted = val_loss_unweighted / len(valid_dl)
        ate_result = compute_ate(np.vstack(gt_poses), np.vstack(pred_poses))
        rpe_result = compute_rpe(np.vstack(gt_poses), np.vstack(pred_poses))

        epoch_duration = time.time() - epoch_start_time
        total_train_time += epoch_duration
        eta_seconds = (total_train_time / (epoch + 1)) * (config["deepvo"]["epochs"] - epoch - 1)
        eta_formatted = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        wandb.log({
            "train_loss": avg_train_loss_unweighted,
            "val_loss": avg_val_loss_unweighted,
            "train_loss_weighted": avg_train_loss,
            "val_loss_weighted": avg_val_loss,
            "ATE": ate_result,
            "RPE": rpe_result,
            "grad_norm": avg_grad_norm,
            "epoch_time": epoch_duration,
            "ETA": eta_seconds,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "GPU_usage": torch.cuda.memory_allocated() / 1e9
        })

        print(f"Epoch {epoch+1}/{config['deepvo']['epochs']}: Train Loss: {avg_train_loss:.6f} (Unweighted: {avg_train_loss_unweighted:.6f}), Val Loss: {avg_val_loss:.6f} (Unweighted: {avg_val_loss_unweighted:.6f}), "
              f"Grad Norm: {avg_grad_norm:.4f}, ATE: {ate_result:.4f}, RPE: {rpe_result:.4f}, Epoch Time: {epoch_duration:.2f}s, ETA: {eta_formatted}")

        os.makedirs(os.path.dirname(config["deepvo"]["model_path"]), exist_ok=True)
        if avg_val_loss < min_loss_v:
            min_loss_v = avg_val_loss
            torch.save(model.state_dict(), config["deepvo"]["model_path"])
        torch.save(model.state_dict(), config["deepvo"]["model_path"].replace("_best.pth", f"_epoch{epoch+1}.pth"))

    wandb.finish()

def train_lorcon_lo(config, device):
    wandb.init(project="FusionLIVO", name="LoRCoNLO-Training-0", config=config["lorcon_lo"])
    seq_sizes = {}
    batch_size = config["lorcon_lo"]["batch_size"]
    num_workers = config["num_workers"]
    torch.set_num_threads(num_workers)

    preprocessed_folder = config["lorcon_lo"]["preprocessed_folder"]
    relative_pose_folder = config["lorcon_lo"]["relative_pose_folder"]
    dataset = config["dataset"]
    data_seqs = config["lorcon_lo"]["data_seqs"].split(",")
    test_seqs = config["lorcon_lo"]["test_seqs"].split(",")
    rnn_size = config["lorcon_lo"]["rnn_size"]
    image_width = config["lorcon_lo"]["image_width"]
    image_height = config["lorcon_lo"]["image_height"]
    depth_name = config["lorcon_lo"]["depth_name"]
    intensity_name = config["lorcon_lo"]["intensity_name"]
    normal_name = config["lorcon_lo"]["normal_name"]
    dni_size = config["lorcon_lo"]["dni_size"]
    normal_size = config["lorcon_lo"]["normal_size"]

    print(f"Data sequences: {data_seqs}, Test sequences: {test_seqs}")
    seq_sizes = count_seq_sizes(preprocessed_folder, data_seqs, seq_sizes)
    print(f"Sequence sizes: {seq_sizes}")

    for seq in data_seqs:
        seq_path = os.path.join(relative_pose_folder, f"{seq}.txt")
        if os.path.exists(seq_path):
            with open(seq_path, "r") as f:
                num_frames = len(f.readlines())
            print(f"Number of frames in sequence {seq} (from {seq_path}): {num_frames}")
        else:
            print(f"Relative pose file for sequence {seq} not found at {seq_path}")

    Y_data = process_input_data(preprocessed_folder, relative_pose_folder, data_seqs, seq_sizes)
    print(f"Y_data shape: {Y_data.shape}")

    start_idx, end_idx = 0, 0
    train_idx, test_idx = np.array([], dtype=int), np.array([], dtype=int)
    for seq in data_seqs:
        end_idx += seq_sizes.get(seq, 0) - 1
        if seq in test_seqs:
            if end_idx - (rnn_size - 1) >= start_idx:
                test_idx = np.append(test_idx, np.arange(start_idx, end_idx - (rnn_size - 1), dtype=int))
        else:
            train_idx = np.append(train_idx, np.arange(start_idx, end_idx - (rnn_size - 1), dtype=int))
        start_idx += seq_sizes.get(seq, 0) - 1

    print(f"Train indices: {len(train_idx)}, Test indices: {len(test_idx)}")
    training_data = LoRCoNLODataset(preprocessed_folder, Y_data, train_idx, seq_sizes, rnn_size, image_width, image_height, depth_name, intensity_name, normal_name, dni_size, normal_size)
    test_data = LoRCoNLODataset(preprocessed_folder, Y_data, test_idx, seq_sizes, rnn_size, image_width, image_height, depth_name, intensity_name, normal_name, dni_size, normal_size)
    print(f"Training data length: {len(training_data)}, Test data length: {len(test_data)}")

    train_dataloader = DataLoader(training_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    model = LoRCoNLO(batch_size=batch_size, batchNorm=False).to(device)
    criterion = WeightedLoss().to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=0.0005)

    from_checkpoint = config["lorcon_lo"].get("from_checkpoint", False)
    is_same_dataset = config["lorcon_lo"].get("is_same_dataset", True)
    start_epoch = 1
    if from_checkpoint:
        cp_folder = os.path.join(config["lorcon_lo"].get("cp_folder", "checkpoints"), dataset)
        checkpoint_path = os.path.join(cp_folder, config["lorcon_lo"]["checkpoint_path"])
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if is_same_dataset:
            start_epoch = checkpoint['epoch'] + 1
        print(f"Train from checkpoint {checkpoint_path}, start_epoch: {start_epoch}")
    else:
        print("Train from scratch")

    log_folder = os.path.join(config["lorcon_lo"].get("log_folder", "runs"), dataset)
    os.makedirs(log_folder, exist_ok=True)
    new_log_dir = os.path.join(log_folder, str(len(next(os.walk(log_folder))[1])).zfill(4))
    os.makedirs(new_log_dir, exist_ok=True)
    writer = SummaryWriter(new_log_dir)

    cp_folder = os.path.join(config["lorcon_lo"].get("cp_folder", "checkpoints"), dataset)
    os.makedirs(cp_folder, exist_ok=True)
    new_cp_dir = os.path.join(cp_folder, str(len(next(os.walk(cp_folder))[1])).zfill(4))
    os.makedirs(new_cp_dir, exist_ok=True)
    model_path = os.path.join(new_cp_dir, "cp-{epoch:04d}.pt")

    data_loader_len = len(train_dataloader)
    test_data_loader_len = len(test_dataloader)
    print(f"Training data loader length: {data_loader_len}, Test data loader length: {test_data_loader_len}")

    epochs = config["lorcon_lo"]["epochs"]
    log_epoch = config["lorcon_lo"]["log_epoch"]
    cp_epoch = config["lorcon_lo"]["checkpoint_epoch"]
    total_train_time = 0

    for epoch in tqdm(range(start_epoch, epochs+1)):
        epoch_start_time = time.time()
        model.train()
        criterion.train()
        running_loss, train_loss_unweighted, rmse_error_train, rmse_t_error_train, rmse_r_error_train = 0.0, 0.0, 0.0, 0.0, 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loss_unweighted += compute_unweighted_mse(outputs, labels)
            rmse_error_train += WeightedLoss.RMSEError(outputs, labels).item()
            rmse_t_error_train += WeightedLoss.RMSEError(outputs[:, :, :3], labels[:, :, :3]).item()
            rmse_r_error_train += WeightedLoss.RMSEError(outputs[:, :, 3:], labels[:, :, 3:]).item()

        if epoch % log_epoch == 0:
            avg_train_loss = running_loss / data_loader_len
            avg_train_loss_unweighted = train_loss_unweighted / data_loader_len
            avg_rmse_error_train = rmse_error_train / data_loader_len
            avg_rmse_t_error_train = rmse_t_error_train / data_loader_len
            avg_rmse_r_error_train = rmse_r_error_train / data_loader_len
            print(f'[{epoch + 1}, {i + 1}] training loss: {avg_train_loss:.10f}')

            model.eval()
            criterion.eval()
            test_loss, test_loss_unweighted, rmse_error_test, rmse_t_error_test, rmse_r_error_test = 0.0, 0.0, 0.0, 0.0, 0.0
            with torch.no_grad():
                for t_i, (t_inputs, t_labels) in enumerate(test_dataloader):
                    t_inputs, t_labels = t_inputs.float().to(device), t_labels.float().to(device)
                    t_outputs = model(t_inputs)
                    t_loss = criterion(t_outputs, t_labels)
                    test_loss += t_loss.item()
                    test_loss_unweighted += compute_unweighted_mse(t_outputs, t_labels)
                    rmse_error_test += WeightedLoss.RMSEError(t_outputs, t_labels).item()
                    rmse_t_error_test += WeightedLoss.RMSEError(t_outputs[:, :, :3], t_labels[:, :, :3]).item()
                    rmse_r_error_test += WeightedLoss.RMSEError(t_outputs[:, :, 3:], t_labels[:, :, 3:]).item()
            avg_val_loss = test_loss / test_data_loader_len if test_data_loader_len > 0 else 0
            avg_val_loss_unweighted = test_loss_unweighted / test_data_loader_len if test_data_loader_len > 0 else 0
            avg_rmse_error_test = rmse_error_test / test_data_loader_len if test_data_loader_len > 0 else 0
            avg_rmse_t_error_test = rmse_t_error_test / test_data_loader_len if test_data_loader_len > 0 else 0
            avg_rmse_r_error_test = rmse_r_error_test / test_data_loader_len if test_data_loader_len > 0 else 0
            print(f'[{epoch + 1}, {t_i + 1}] validation loss: {avg_val_loss:.10f}')

            epoch_duration = time.time() - epoch_start_time
            total_train_time += epoch_duration
            eta_seconds = (total_train_time / (epoch + 1 - start_epoch + 1)) * (epochs - epoch)
            eta_formatted = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            wandb.log({
                "train_loss": avg_train_loss_unweighted,
                "val_loss": avg_val_loss_unweighted,
                "train_loss_weighted": avg_train_loss,
                "val_loss_weighted": avg_val_loss,
                "rmse/train": avg_rmse_error_train,
                "rmse_t/train": avg_rmse_t_error_train,
                "rmse_r/train": avg_rmse_r_error_train,
                "rmse/val": avg_rmse_error_test,
                "rmse_t/val": avg_rmse_t_error_test,
                "rmse_r/val": avg_rmse_r_error_test,
                "epoch_time": epoch_duration,
                "ETA": eta_seconds,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "GPU_usage": torch.cuda.memory_allocated() / 1e9
            })

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': running_loss / data_loader_len}, model_path.format(epoch=epoch))
        print(f"Model saved after epoch {epoch} in {model_path.format(epoch=epoch)}")
        if epoch % cp_epoch == 0 and epoch != start_epoch:
            checkpoint_path_special = os.path.join(new_cp_dir, f"cp-special-{epoch:04d}.pt")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': running_loss / data_loader_len}, checkpoint_path_special)
            print(f"Special checkpoint saved at epoch {epoch} in {checkpoint_path_special}")

    print('Finished Training')
    writer.close()
    wandb.finish()

# ... (Fusion Block)

def train_fusion(config, device):
    wandb.init(project="FusionLIVO", name="FusionLIVO-Training-0", config=config["fusion"])
    
    seq_len = 2
    train_seqs = sorted(list(set(config["deepvo"]["train_video"]) & set(config["lorcon_lo"]["data_seqs"])))
    valid_seqs = sorted(list(set(config["deepvo"]["valid_video"]) & set(config["lorcon_lo"]["test_seqs"])))
    
    # Calculate input channels from config flags
    use_depth = config["fusion"]["modalities"]["use_depth"]
    use_intensity = config["fusion"]["modalities"]["use_intensity"]
    use_normals = config["fusion"]["modalities"]["use_normals"]
    use_rgb_low = config["fusion"]["modalities"]["use_rgb_low"]
    input_channels = (3 if use_rgb_low else 0) + (1 if use_depth else 0) + (1 if use_intensity else 0) + (3 if use_normals else 0)
    
    train_dataset = FusionDataset(config, train_seqs, seq_len)  # Flags now read from config inside FusionDataset
    valid_dataset = FusionDataset(config, valid_seqs, seq_len)
    
    train_dl = DataLoader(train_dataset, batch_size=config["fusion"]["batch_size"], shuffle=True, num_workers=config["num_workers"])
    valid_dl = DataLoader(valid_dataset, batch_size=config["fusion"]["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    model = FusionLIVO(rgb_height=184, rgb_width=608, lidar_height=64, lidar_width=900, input_channels=input_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = WeightedLoss().to(device)
    scaler = GradScaler() if config["fusion"]["use_grad_scaler"] else None
    
    epochs = config["fusion"]["epochs"]
    first_epoch_time = None
    
    for epoch in tqdm(range(epochs), desc="FusionLIVO Training Epochs"):
        epoch_start_time = time.time()
        
        model.train()
        train_loss, train_loss_unweighted, rmse_error_train, grad_norm_total = 0.0, 0.0, 0.0, 0.0
        for rgb_high, lidar_combined, targets in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=True):
            rgb_high, lidar_combined, targets = rgb_high.to(device), lidar_combined.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if config["fusion"]["use_autocast"]:
                with autocast():
                    outputs = model(rgb_high, lidar_combined)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(rgb_high, lidar_combined)
                loss = criterion(outputs, targets)
            
            if config["fusion"]["use_grad_scaler"] and scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_loss_unweighted += compute_unweighted_mse(outputs, targets)
            rmse_error_train += WeightedLoss.RMSEError(outputs, targets).item()
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            grad_norm_total += grad_norm
        
        avg_train_loss = train_loss / len(train_dl)
        avg_train_loss_unweighted = train_loss_unweighted / len(train_dl)
        avg_rmse_error_train = rmse_error_train / len(train_dl)
        avg_grad_norm = grad_norm_total / len(train_dl)
        
        model.eval()
        val_loss, val_loss_unweighted, rmse_error_val = 0.0, 0.0, 0.0
        gt_poses, pred_poses = [], []
        with torch.no_grad():
            for rgb_high, lidar_combined, targets in tqdm(valid_dl, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=True):
                rgb_high, lidar_combined, targets = rgb_high.to(device), lidar_combined.to(device), targets.to(device)
                if config["fusion"]["use_autocast"]:
                    with autocast():
                        outputs = model(rgb_high, lidar_combined)
                        val_loss += criterion(outputs, targets).item()
                else:
                    outputs = model(rgb_high, lidar_combined)
                    val_loss += criterion(outputs, targets).item()
                val_loss_unweighted += compute_unweighted_mse(outputs, targets)
                rmse_error_val += WeightedLoss.RMSEError(outputs, targets).item()
                gt_poses.append(targets.cpu().numpy())
                pred_poses.append(outputs.cpu().numpy())
        
        avg_val_loss = val_loss / len(valid_dl)
        avg_val_loss_unweighted = val_loss_unweighted / len(valid_dl)
        avg_rmse_error_val = rmse_error_val / len(valid_dl)
        ate_result = compute_ate(np.vstack(gt_poses), np.vstack(pred_poses)) if gt_poses and pred_poses else np.nan
        rpe_result = compute_rpe(np.vstack(gt_poses), np.vstack(pred_poses)) if gt_poses and pred_poses else np.nan
        
        epoch_duration = time.time() - epoch_start_time
        if epoch == 0:
            first_epoch_time = epoch_duration
        eta_seconds = first_epoch_time * (epochs - epoch - 1) if first_epoch_time else 0
        eta_formatted = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f} (Unweighted: {avg_train_loss_unweighted:.6f}), Val Loss: {avg_val_loss:.6f} (Unweighted: {avg_val_loss_unweighted:.6f})")
        print(f"  Train RMSE: {avg_rmse_error_train:.6f}, Val RMSE: {avg_rmse_error_val:.6f}")
        print(f"  ATE: {ate_result:.6f}, RPE: {rpe_result:.6f}")
        print(f"  Grad Norm: {avg_grad_norm:.6f}, Epoch Time: {epoch_duration:.2f}s, ETA: {eta_formatted}")
        
        wandb.log({
            "train_loss": avg_train_loss_unweighted,
            "val_loss": avg_val_loss_unweighted,
            "train_loss_weighted": avg_train_loss,
            "val_loss_weighted": avg_val_loss,
            "fusion/train_rmse": avg_rmse_error_train,
            "fusion/val_rmse": avg_rmse_error_val,
            "fusion/ate": ate_result,
            "fusion/rpe": rpe_result,
            "fusion/grad_norm": avg_grad_norm,
            "fusion/epoch_time": epoch_duration,
            "fusion/eta": eta_seconds,
            "fusion/learning_rate": optimizer.param_groups[0]['lr'],
            "fusion/gpu_usage": torch.cuda.memory_allocated() / 1e9
        })
        
        torch.save(model.state_dict(), config["fusion"]["model_path"])
    
    wandb.finish()

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    if config["use_camera"]:
        train_deepvo(config, device)
    if config["use_lidar"]:
        train_lorcon_lo(config, device)
    if config["use_fusion"]:
        train_fusion(config, device)

if __name__ == "__main__":
    main()