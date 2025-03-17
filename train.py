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
from models import DeepVO, LoRCoNLO, WeightedLoss

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

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    if config["use_camera"]:
        # DeepVO Training (Matches fixed deepvo/main.py)
        wandb.init(project="Fusion", name="DeepVO-Training-0", config=config["deepvo"])

        # Use a fixed sequence length (average of seq_len range)
        seq_len = int((config["deepvo"]["seq_len"][0] + config["deepvo"]["seq_len"][1]) / 2)
        print(f"Using fixed sequence length: {seq_len}")

        # Delete cached dataframes to ensure fresh computation
        if os.path.exists(config["deepvo"]["train_data_info_path"]):
            os.remove(config["deepvo"]["train_data_info_path"])
            print(f"Deleted cached train_data_info_path: {config['deepvo']['train_data_info_path']}")
        if os.path.exists(config["deepvo"]["valid_data_info_path"]):
            os.remove(config["deepvo"]["valid_data_info_path"])
            print(f"Deleted cached valid_data_info_path: {config['deepvo']['valid_data_info_path']}")

        # Ensure all sequences in train_df and valid_df have the same length
        train_df = get_data_info(
            folder_list=config["deepvo"]["train_video"],
            seq_len_range=[seq_len, seq_len],
            overlap=1,
            sample_times=config["deepvo"]["sample_times"],
            config=config
        )
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

        valid_df = get_data_info(
            folder_list=config["deepvo"]["valid_video"],
            seq_len_range=[seq_len, seq_len],
            overlap=1,
            sample_times=config["deepvo"]["sample_times"],
            config=config
        )
        print(f"valid_df shape: {valid_df.shape}, columns: {train_df.columns}")
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

        train_dataset = ImageSequenceDataset(
            train_df,
            resize_mode=config["deepvo"]["resize_mode"],
            new_size=(config["deepvo"]["img_w"], config["deepvo"]["img_h"]),
            img_mean=config["deepvo"]["img_means"],
            img_std=config["deepvo"]["img_stds"],
            minus_point_5=config["deepvo"]["minus_point_5"],
            config=config
        )
        valid_dataset = ImageSequenceDataset(
            valid_df,
            resize_mode=config["deepvo"]["resize_mode"],
            new_size=(config["deepvo"]["img_w"], config["deepvo"]["img_h"]),
            img_mean=config["deepvo"]["img_means"],
            img_std=config["deepvo"]["img_stds"],
            minus_point_5=config["deepvo"]["minus_point_5"],
            config=config
        )

        # Re-enable multi-worker data loading
        train_dl = DataLoader(
            train_dataset,
            batch_sampler=SortedRandomBatchSampler(train_df, config["deepvo"]["batch_size"], drop_last=True),
            num_workers=config["num_workers"],
            pin_memory=config["deepvo"]["pin_mem"]
        )
        valid_dl = DataLoader(
            valid_dataset,
            batch_sampler=SortedRandomBatchSampler(valid_df, config["deepvo"]["batch_size"], drop_last=True),
            num_workers=config["num_workers"],
            pin_memory=config["deepvo"]["pin_mem"]
        )
        print(f"Number of training batches: {len(train_dl)}")
        print(f"Number of validation batches: {len(valid_dl)}")

        model = DeepVO(
            config["deepvo"]["img_h"], config["deepvo"]["img_w"],
            batchNorm=config["deepvo"]["batch_norm"],
            conv_dropout=config["deepvo"]["conv_dropout"],
            rnn_hidden_size=config["deepvo"]["rnn_hidden_size"],
            rnn_dropout_out=config["deepvo"]["rnn_dropout_out"],
            rnn_dropout_between=config["deepvo"]["rnn_dropout_between"],
            clip=config["deepvo"]["clip"]
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config["deepvo"]["optim"]["lr"]) if config["deepvo"]["optim"]["opt"] == "Adam" else optim.Adagrad(model.parameters(), lr=config["deepvo"]["optim"]["lr"])

        if os.path.exists(config["deepvo"]["model_path"]):
            model.load_state_dict(torch.load(config["deepvo"]["model_path"]))

        min_loss_v, total_train_time, start_time = float("inf"), 0, time.time()

        for epoch in range(config["deepvo"]["epochs"]):
            epoch_start_time = time.time()
            print('=' * 50)

            model.train()
            train_loss, grad_norm_total, train_bar = 0.0, 0.0, tqdm(train_dl, desc=f"Epoch {epoch+1}/{config['deepvo']['epochs']} [Training]", leave=True)
            
            for batch_idx, (_, inputs, targets) in enumerate(train_bar):
                # Print shapes for the first few batches to confirm consistency
                if batch_idx < 3:
                    print(f"Batch {batch_idx}: Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
                inputs, targets = inputs.float().to(device), targets.float().to(device)
                optimizer.zero_grad()
                loss = model.step(inputs, targets, optimizer)
                
                grad_norm = 0
                for p in model.parameters():
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

            model.eval()
            val_loss, gt_poses, pred_poses, val_bar = 0.0, [], [], tqdm(valid_dl, desc=f"Epoch {epoch+1}/{config['deepvo']['epochs']} [Validation]", leave=True)
            
            with torch.no_grad():
                for batch_idx, (_, inputs, targets) in enumerate(val_bar):
                    if batch_idx < 3:
                        print(f"Validation Batch {batch_idx}: Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
                    inputs, targets = inputs.float().to(device), targets.float().to(device)
                    outputs = model(inputs)
                    gt_poses.append(targets.cpu().numpy())
                    pred_poses.append(outputs.cpu().numpy())
                    val_loss += float(model.get_loss(inputs, targets).data.cpu().numpy())
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
            eta_seconds = avg_epoch_time * (config["deepvo"]["epochs"] - epoch - 1)
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

            print(f"Epoch {epoch+1}/{config['deepvo']['epochs']}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                  f"Grad Norm: {avg_grad_norm:.4f}, ATE: {ate_result:.4f}, RPE: {rpe_result:.4f}, Epoch Time: {epoch_duration:.2f}s, ETA: {eta_formatted}")

            os.makedirs(os.path.dirname(config["deepvo"]["model_path"]), exist_ok=True)
            if avg_val_loss < min_loss_v:
                min_loss_v = avg_val_loss
                torch.save(model.state_dict(), config["deepvo"]["model_path"])

            os.makedirs(os.path.dirname(config["deepvo"]["model_path"].replace("_best.pth", f"_epoch{epoch+1}.pth")), exist_ok=True)
            torch.save(model.state_dict(), config["deepvo"]["model_path"].replace("_best.pth", f"_epoch{epoch+1}.pth"))

        wandb.finish()

    if config["use_lidar"]:
        # LoRCoN-LO Training (Matches lorcon_lo/train.py)
        # Initialize WandB for LoRCoN-LO
        wandb.init(project="Fusion", name="LoRCoNLO-Training-0", config=config["lorcon_lo"])

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

        print(f"Data sequences: {data_seqs}")
        print(f"Test sequences: {test_seqs}")
        seq_sizes = count_seq_sizes(preprocessed_folder, data_seqs, seq_sizes)
        print(f"Sequence sizes: {seq_sizes}")

        # Enhanced debug: Check frames for all sequences
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

        start_idx = 0
        end_idx = 0
        train_idx = np.array([], dtype=int)
        test_idx = np.array([], dtype=int)  # Initialized outside the loop
        for seq in data_seqs:
            end_idx += seq_sizes.get(seq, 0) - 1
            if seq in test_seqs:
                if end_idx - (rnn_size - 1) < start_idx:
                    print(f"Warning: Skipping test sequence {seq} due to insufficient frames for rnn_size {rnn_size}")
                else:
                    idx_range = np.arange(start_idx, end_idx - (rnn_size - 1), dtype=int)
                    print(f"Test sequence {seq}: start_idx={start_idx}, end_idx={end_idx}, rnn_size-1={rnn_size-1}, idx_range={idx_range}")
                    if len(idx_range) == 0:
                        print(f"Error: No valid indices generated for test sequence {seq}")
                    test_idx = np.append(test_idx, idx_range)  # Append to the outer test_idx
            else:
                idx_range = np.arange(start_idx, end_idx - (rnn_size - 1), dtype=int)
                print(f"Train sequence {seq}: start_idx={start_idx}, end_idx={end_idx}, rnn_size-1={rnn_size-1}, idx_range={idx_range}")
                train_idx = np.append(train_idx, idx_range)
            start_idx += seq_sizes.get(seq, 0) - 1

        print(f"Train indices: {train_idx}")
        print(f"Test indices: {test_idx}")
        print(f"Number of train indices: {len(train_idx)}")
        print(f"Number of test indices: {len(test_idx)}")

        training_data = LoRCoNLODataset(preprocessed_folder, Y_data, train_idx, seq_sizes, rnn_size, image_width, image_height, depth_name, intensity_name, normal_name, dni_size, normal_size)
        test_data = LoRCoNLODataset(preprocessed_folder, Y_data, test_idx, seq_sizes, rnn_size, image_width, image_height, depth_name, intensity_name, normal_name, dni_size, normal_size)

        print(f"Training data length: {len(training_data)}")
        print(f"Test data length: {len(test_data)}")

        train_dataloader = DataLoader(training_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)
        
        model = LoRCoNLO(batch_size=batch_size, batchNorm=False).to(device)
        criterion = WeightedLoss().to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=0.0005)

        from_checkpoint = config["lorcon_lo"].get("from_checkpoint", False)
        is_same_dataset = config["lorcon_lo"].get("is_same_dataset", True)
        start_epoch = 1
        if from_checkpoint:
            cp_folder = config["lorcon_lo"].get("cp_folder", "checkpoints")
            cp_folder = os.path.join(cp_folder, dataset)
            checkpoint_path = os.path.join(cp_folder, config["lorcon_lo"]["checkpoint_path"])
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if is_same_dataset:
                start_epoch = checkpoint['epoch'] + 1
            print(f"Train from checkpoint {checkpoint_path}, start_epoch: {start_epoch}")
        else:
            print("Train from scratch")

        log_folder = config["lorcon_lo"].get("log_folder", "runs")
        log_folder = os.path.join(log_folder, dataset)
        os.makedirs(log_folder, exist_ok=True)
        _, prev_log_dirs, _ = next(os.walk(log_folder))
        new_log_dir = os.path.join(log_folder, str(len(prev_log_dirs)).zfill(4))
        os.makedirs(new_log_dir, exist_ok=True)
        writer = SummaryWriter(new_log_dir)

        cp_folder = config["lorcon_lo"].get("cp_folder", "checkpoints")
        cp_folder = os.path.join(cp_folder, dataset)
        os.makedirs(cp_folder, exist_ok=True)
        _, prev_cp_dirs, _ = next(os.walk(cp_folder))
        new_cp_dir = os.path.join(cp_folder, str(len(prev_cp_dirs)).zfill(4))
        os.makedirs(new_cp_dir, exist_ok=True)
        model_path = os.path.join(new_cp_dir, "cp-{epoch:04d}.pt")

        data_loader_len = len(train_dataloader)
        test_data_loader_len = len(test_dataloader)
        print(f"Training data loader length: {data_loader_len}")
        print(f"Test data loader length: {test_data_loader_len}")

        epochs = config["lorcon_lo"]["epochs"]
        log_epoch = config["lorcon_lo"]["log_epoch"]
        cp_epoch = config["lorcon_lo"]["checkpoint_epoch"]
        total_train_time = 0  # Track total training time for ETA
        start_time = time.time()

        for epoch in tqdm(range(start_epoch, epochs+1)):
            epoch_start_time = time.time()
            model.train()
            criterion.train()
            running_loss = 0.0
            rmse_error_train = 0.0
            rmse_t_error_train = 0.0
            rmse_r_error_train = 0.0
            for i, data in tqdm(enumerate(train_dataloader, 0)):
                inputs, labels = data
                inputs, labels = inputs.float().to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                rmse_error_train += WeightedLoss.RMSEError(outputs, labels).item()
                rmse_t_error_train += WeightedLoss.RMSEError(outputs[:, :, :3], labels[:, :, :3]).item()
                rmse_r_error_train += WeightedLoss.RMSEError(outputs[:, :, 3:], labels[:, :, 3:]).item()
            if epoch % log_epoch == 0:
                avg_train_loss = running_loss / data_loader_len if data_loader_len > 0 else 0
                avg_rmse_error_train = rmse_error_train / data_loader_len if data_loader_len > 0 else 0
                avg_rmse_t_error_train = rmse_t_error_train / data_loader_len if data_loader_len > 0 else 0
                avg_rmse_r_error_train = rmse_r_error_train / data_loader_len if data_loader_len > 0 else 0
                print('[%d, %5d] training loss: %.10f' %
                      (epoch + 1, i + 1, avg_train_loss))
                writer.add_scalar('Loss/train', avg_train_loss, epoch)
                writer.add_scalar('RMSE/train', avg_rmse_error_train, epoch)
                writer.add_scalar('RMSE_t/train', avg_rmse_t_error_train, epoch)
                writer.add_scalar('RMSE_r/train', avg_rmse_r_error_train, epoch)

                model.eval()
                criterion.eval()
                with torch.no_grad():
                    if test_data_loader_len == 0:
                        print(f"Warning: Test dataloader is empty at epoch {epoch+1}. Skipping validation.")
                        print(f"Debug: test_idx length = {len(test_idx)}, test_data length = {len(test_data)}")
                        for seq in test_seqs:
                            if seq_sizes.get(seq, 0) > 0:
                                print(f"Debug: Sequence {seq} has {seq_sizes[seq]} frames, should support validation")
                            else:
                                print(f"Debug: Sequence {seq} has 0 frames in seq_sizes")
                        test_loss = float('inf')
                        rmse_error_test = 0
                        rmse_t_error_test = 0
                        rmse_r_error_test = 0
                    else:
                        test_loss = 0
                        rmse_error_test = 0
                        rmse_t_error_test = 0
                        rmse_r_error_test = 0
                        t_i = 0
                        for t_i, t_data in enumerate(test_dataloader):
                            t_inputs, t_labels = t_data
                            t_inputs, t_labels = t_inputs.float().to(device), t_labels.float().to(device)
                            t_outputs = model(t_inputs)
                            t_loss = criterion(t_outputs, t_labels)
                            test_loss += t_loss.item()
                            rmse_error_test += WeightedLoss.RMSEError(t_outputs, t_labels).item()
                            rmse_t_error_test += WeightedLoss.RMSEError(t_outputs[:, :, :3], t_labels[:, :, :3]).item()
                            rmse_r_error_test += WeightedLoss.RMSEError(t_outputs[:, :, 3:], t_labels[:, :, 3:]).item()
                        avg_val_loss = test_loss / test_data_loader_len if test_data_loader_len > 0 else 0
                        avg_rmse_error_test = rmse_error_test / test_data_loader_len if test_data_loader_len > 0 else 0
                        avg_rmse_t_error_test = rmse_t_error_test / test_data_loader_len if test_data_loader_len > 0 else 0
                        avg_rmse_r_error_test = rmse_r_error_test / test_data_loader_len if test_data_loader_len > 0 else 0
                        print('[%d, %5d] validation loss: %.10f' %
                              (epoch + 1, t_i + 1, avg_val_loss))
                        writer.add_scalar('Loss/val', avg_val_loss, epoch)
                        writer.add_scalar('RMSE/val', avg_rmse_error_test, epoch)
                        writer.add_scalar('RMSE_t/val', avg_rmse_t_error_test, epoch)
                        writer.add_scalar('RMSE_r/val', avg_rmse_r_error_test, epoch)

                # Log metrics to WandB
                epoch_duration = time.time() - epoch_start_time
                total_train_time += epoch_duration
                avg_epoch_time = total_train_time / (epoch + 1)
                eta_seconds = avg_epoch_time * (config["lorcon_lo"]["epochs"] - epoch - 1)
                eta_formatted = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                wandb.log({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
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

            if epoch % cp_epoch == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss / data_loader_len if data_loader_len > 0 else 0,
                }, model_path.format(epoch=epoch))
                print("Model saved in ", model_path.format(epoch=epoch))

        print('Finished Training')
        writer.close()
        wandb.finish()  # Ensure WandB session is properly closed

if __name__ == "__main__":
    main()