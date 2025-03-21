import yaml
import torch
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import wandb
import glob
from deepvo.data_helper import get_data_info, ImageSequenceDataset
from deepvo.helper import eulerAnglesToRotationMatrix
from lorcon_lo.utils.process_data import LoRCoNLODataset, count_seq_sizes, process_input_data
from lorcon_lo.utils.common import get_original_poses, save_poses
from lorcon_lo.utils.plot_utils import plot_gt, plot_results
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

    device = torch.device(config["device"] if torch.cuda.is_available() else "cuda:1" if config["use_lidar"] else "cuda:0")
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    if config["use_camera"]:
        wandb.init(project="Fusion", name="DeepVO-Testing-0", config=config["deepvo"])

        use_cuda = torch.cuda.is_available()
        save_dir = 'result/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model = DeepVO(config["deepvo"]["img_h"], config["deepvo"]["img_w"], config["deepvo"]["batch_norm"])
        if use_cuda:
            model = model.cuda()
            model.load_state_dict(torch.load(config["deepvo"]["model_path"]))
        else:
            model.load_state_dict(torch.load(config["deepvo"]["model_path"], map_location={'cuda:0': 'cpu'}))
        print('Load model from: ', config["deepvo"]["model_path"])

        n_workers = 1
        seq_len = int((config["deepvo"]["seq_len"][0] + config["deepvo"]["seq_len"][1]) / 2)
        overlap = seq_len - 1
        print(f"seq_len = {seq_len}, overlap = {overlap}")
        batch_size = config["deepvo"]["batch_size"]

        print(f"Loaded deepvo config: {config['deepvo']}")

        with open('test_dump.txt', 'w') as fd:
            fd.write('\n' + '=' * 50 + '\n')

            for test_video in config["deepvo"]["valid_video"]:
                image_path = os.path.join(config["deepvo"]["image_dir"], f"{test_video}", "image_02")
                print(f"Checking image path: {image_path}")
                png_files = glob.glob(os.path.join(image_path, "*.png"))
                jpg_files = glob.glob(os.path.join(image_path, "*.jpg"))
                print(f"Found {len(png_files)} PNG files, {len(jpg_files)} JPG files")

                gt_pose_raw = np.load(os.path.join(config["deepvo"]["pose_dir"], f"{test_video}.npy"))
                print(f"Raw ground truth pose [0]: {gt_pose_raw[0]}")
                gt_pose = gt_pose_raw[:, :6]  # [x, y, z, tx, ty, tz]
                gt_pose_rel = np.diff(gt_pose, axis=0)
                gt_pose = np.vstack(([0, 0, 0, 0, 0, 0], gt_pose_rel))  # 271 poses
                print(f"Video {test_video}: {len(png_files)} frames, {len(gt_pose)} poses")

                df = get_data_info(
                    folder_list=[test_video],
                    seq_len_range=[seq_len, seq_len],
                    overlap=overlap,
                    sample_times=1,
                    shuffle=False,
                    sort=False,
                    config=config
                )
                print(f"Video {test_video}: Added {len(df)} sequences of length {seq_len}")
                df = df.loc[df.seq_len == seq_len]
                df.to_csv('test_df.csv')
                dataset = ImageSequenceDataset(
                    df, config["deepvo"]["resize_mode"],
                    (config["deepvo"]["img_w"], config["deepvo"]["img_h"]),
                    config["deepvo"]["img_means"], config["deepvo"]["img_stds"],
                    config["deepvo"]["minus_point_5"],
                    config=config
                )
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, drop_last=False)

                model.eval()
                answer = [[0.0] * 6]  # Start with initial zero pose
                st_t = time.time()
                n_batch = len(dataloader)

                print(f"First ground truth pose: {gt_pose[0]}")

                for i, batch in enumerate(dataloader):
                    print(f'Folder {test_video} batch {i}/{n_batch}', end='\r', flush=True)
                    _, x, y = batch
                    if use_cuda:
                        x = x.cuda()
                        y = y.cuda()
                    batch_predict_pose = model.forward(x)

                    if i == 0:
                        print(f"Raw prediction for batch 0: {batch_predict_pose[0][0]}")
                        print(f"Input shape to model: {x.shape}")
                        print(f"Output shape from model: {batch_predict_pose.shape}")

                    fd.write(f'Batch: {i}\n')
                    for seq, predict_pose_seq in enumerate(batch_predict_pose):
                        for pose_idx, pose in enumerate(predict_pose_seq):
                            fd.write(f' {seq} {pose_idx} {pose}\n')
                        answer.append(predict_pose_seq[0].data.cpu().numpy().tolist())  # First pose per sequence

                if len(answer) != len(gt_pose):
                    print(f"Adjusting answer length from {len(answer)} to {len(gt_pose)}")
                    if len(answer) > len(gt_pose):
                        answer = answer[:len(gt_pose)]
                    else:
                        answer.extend([[0.0] * 6] * (len(gt_pose) - len(answer)))

                print(f'Folder {test_video} finish in {time.time() - st_t} sec')
                print('len(answer): ', len(answer))
                expected_len = len(glob.glob(os.path.join(config["deepvo"]["image_dir"], f"{test_video}/image_02/*.png")))
                print('expect len: ', expected_len)
                predict_time = time.time() - st_t
                print('Predict use {} sec'.format(predict_time))
                print(f"First predicted pose: {answer[1]}")  # First transition after initial zero

                with open(f'{save_dir}/out_{test_video}.txt', 'w') as f:
                    for pose in answer:
                        if isinstance(pose, list):
                            f.write(', '.join([str(p) for p in pose]))
                        else:
                            f.write(str(pose))
                        f.write('\n')

                loss = 0
                for t in range(len(gt_pose)):
                    angle_loss = np.sum((np.array(answer[t])[:3] - gt_pose[t, :3]) ** 2)
                    translation_loss = np.sum((np.array(answer[t])[3:] - gt_pose[t, 3:]) ** 2)
                    loss += (100 * angle_loss + translation_loss)
                loss /= len(gt_pose)
                print('Loss = ', loss)
                print('=' * 50)

                wandb.log({
                    f"deepvo/test_loss_{test_video}": loss,
                    f"deepvo/predict_time_{test_video}": predict_time,
                    f"deepvo/predicted_pose_length_{test_video}": len(answer),
                    f"deepvo/expected_pose_length_{test_video}": expected_len,
                    f"deepvo/pose_length_mismatch_{test_video}": len(answer) != len(gt_pose)
                })

        fd.close()
        wandb.finish()
    if config["use_lidar"]:
        # LoRCoN-LO Testing (from lorcon_lo/test.py)
        wandb.init(project="Fusion", name="LoRCoNLO-Testing-0", config=config["lorcon_lo"])  # Initialize WandB for LoRCoN-LO testing

        cuda = torch.device('cuda')
        seq_sizes = {}
        batch_size = config["lorcon_lo"]["batch_size"]
        num_workers = config["num_workers"]

        preprocessed_folder = config["lorcon_lo"]["preprocessed_folder"]
        pose_folder = config["lorcon_lo"]["pose_folder"]
        relative_pose_folder = config["lorcon_lo"]["relative_pose_folder"]
        dataset = config["dataset"]
        cp_folder = config["lorcon_lo"].get("cp_folder", "checkpoints")

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

        # Use checkpoint_test_path for testing
        checkpoint_test_path = config["lorcon_lo"]["checkpoint_test_path"]
        checkpoint_path = os.path.join(cp_folder, dataset, checkpoint_test_path)

        seq_sizes = count_seq_sizes(preprocessed_folder, data_seqs, seq_sizes)
        Y_data = process_input_data(preprocessed_folder, relative_pose_folder, data_seqs, seq_sizes)
        
        start_idx = 0
        end_idx = 0
        test_idx = np.array([], dtype=int)
        for seq in data_seqs:
            end_idx += seq_sizes[seq] - 1
            test_idx = np.append(test_idx, np.arange(start_idx, end_idx - (rnn_size - 1), dtype=int))
            start_idx += seq_sizes[seq] - 1

        test_data = LoRCoNLODataset(preprocessed_folder, Y_data, test_idx, seq_sizes, rnn_size, image_width, image_height, depth_name, intensity_name, normal_name, dni_size, normal_size)
        test_dataloader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)

        model = LoRCoNLO(batch_size=batch_size, batchNorm=False).to(device)
        criterion = WeightedLoss(learn_hyper_params=False)
        optimizer = optim.Adagrad(model.parameters(), lr=0.0005)

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        Y_estimated_data = np.empty((0, 6), dtype=np.float64)
        test_data_loader_len = len(test_dataloader)
        test_loss = 0.0
        rmse_error_test = 0.0
        rmse_t_error_test = 0.0
        rmse_r_error_test = 0.0
        st_t = time.time()

        with torch.no_grad():
            for idx, data in tqdm(enumerate(test_dataloader), total=test_data_loader_len):
                inputs, labels = data
                inputs, labels = inputs.float().to(device), labels.float().to(device)
                outputs = model(inputs)
                Y_estimated_data = np.vstack((Y_estimated_data, outputs[:, -1, :].cpu().numpy()))
                test_loss += criterion(outputs, labels).item()
                rmse_error_test += WeightedLoss.RMSEError(outputs, labels).item()
                rmse_t_error_test += WeightedLoss.RMSEError(outputs[:, :, :3], labels[:, :, :3]).item()
                rmse_r_error_test += WeightedLoss.RMSEError(outputs[:, :, 3:], labels[:, :, 3:]).item()

        avg_test_loss = test_loss / test_data_loader_len
        avg_rmse_error_test = rmse_error_test / test_data_loader_len
        avg_rmse_t_error_test = rmse_t_error_test / test_data_loader_len
        avg_rmse_r_error_test = rmse_r_error_test / test_data_loader_len
        predict_time = time.time() - st_t

        print(f"Test loss is {avg_test_loss}")

        Y_origin_data = get_original_poses(pose_folder, preprocessed_folder, data_seqs)
        seq_sizes = {}
        seq_sizes = count_seq_sizes(preprocessed_folder, data_seqs, seq_sizes)
        
        plot_gt(Y_origin_data, pose_folder, preprocessed_folder, data_seqs, seq_sizes, dataset=dataset)
        plot_results(Y_origin_data, Y_estimated_data, data_seqs, rnn_size, seq_sizes, dataset=dataset)
        save_poses(Y_origin_data, Y_estimated_data, data_seqs, rnn_size, seq_sizes, dataset=dataset)

        # Log metrics to WandB for LoRCoNLO
        wandb.log({
            "lorcon_lo/test_loss": avg_test_loss,
            "lorcon_lo/test_rmse": avg_rmse_error_test,
            "lorcon_lo/test_rmse_t": avg_rmse_t_error_test,
            "lorcon_lo/test_rmse_r": avg_rmse_r_error_test,
            "lorcon_lo/predict_time": predict_time,
            "lorcon_lo/test_data_length": len(Y_estimated_data),
            "lorcon_lo/GPU_usage": torch.cuda.memory_allocated() / 1e9
        })

        wandb.finish()  # Close WandB session for LoRCoNLO

if __name__ == "__main__":
    main()