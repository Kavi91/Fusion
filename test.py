# FUSION/test.py
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from deepvo.data_helper import get_data_info, ImageSequenceDataset
from deepvo.helper import eulerAnglesToRotationMatrix
from lorcon_lo.process_data import LoRCoNLODataset, count_seq_sizes, process_input_data
from lorcon_lo.common import get_original_poses, save_poses
from lorcon_lo.plot import plot_gt, plot_results
from models import DeepVO, LoRCoNLO, WeightedLoss

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cuda:1" if config["use_lidar"] else "cuda:0")
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    if config["use_camera"]:
        # DeepVO Testing (from deepvo/test.py)
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
        print('seq_len = {}, overlap = {}'.format(seq_len, overlap))
        batch_size = config["deepvo"]["batch_size"]

        with open('test_dump.txt', 'w') as fd:
            fd.write('\n' + '=' * 50 + '\n')

            for test_video in config["deepvo"]["valid_video"]:
                df = get_data_info(
                    folder_list=[test_video],
                    seq_len_range=[seq_len, seq_len],
                    overlap=overlap,
                    sample_times=1,
                    shuffle=False,
                    sort=False,
                    config=config
                )
                df = df.loc[df.seq_len == seq_len]  # drop last
                df.to_csv('test_df.csv')
                dataset = ImageSequenceDataset(
                    df, config["deepvo"]["resize_mode"],
                    (config["deepvo"]["img_w"], config["deepvo"]["img_h"]),
                    config["deepvo"]["img_means"], config["deepvo"]["img_stds"],
                    config["deepvo"]["minus_point_5"],
                    config=config
                )
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

                gt_pose = np.load(os.path.join(config["deepvo"]["pose_dir"], f"{test_video}.npy"))  # (n_images, 6)

                model.eval()
                has_predict = False
                answer = [[0.0] * 6, ]
                st_t = time.time()
                n_batch = len(dataloader)

                for i, batch in enumerate(dataloader):
                    print('{} / {}'.format(i, n_batch), end='\r', flush=True)
                    _, x, y = batch
                    if use_cuda:
                        x = x.cuda()
                        y = y.cuda()
                    batch_predict_pose = model.forward(x)

                    fd.write('Batch: {}\n'.format(i))
                    for seq, predict_pose_seq in enumerate(batch_predict_pose):
                        for pose_idx, pose in enumerate(predict_pose_seq):
                            fd.write(' {} {} {}\n'.format(seq, pose_idx, pose))

                    batch_predict_pose = batch_predict_pose.data.cpu().numpy()
                    if i == 0:
                        for pose in batch_predict_pose[0]:
                            for j in range(len(pose)):
                                pose[j] += answer[-1][j]
                            answer.append(pose.tolist())
                        batch_predict_pose = batch_predict_pose[1:]

                    for predict_pose_seq in batch_predict_pose:
                        ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0])
                        location = ang.dot(predict_pose_seq[-1][3:])
                        predict_pose_seq[-1][3:] = location[:]
                        last_pose = predict_pose_seq[-1]
                        for j in range(len(last_pose)):
                            last_pose[j] += answer[-1][j]
                        last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
                        answer.append(last_pose.tolist())

                print('len(answer): ', len(answer))
                print('expect len: ', len(glob.glob(os.path.join(config["deepvo"]["image_dir"], f"{test_video}/*.png"))))
                print('Predict use {} sec'.format(time.time() - st_t))

                with open(f'{save_dir}/out_{test_video}.txt', 'w') as f:
                    for pose in answer:
                        if type(pose) == list:
                            f.write(', '.join([str(p) for p in pose]))
                        else:
                            f.write(str(pose))
                        f.write('\n')

                loss = 0
                for t in range(len(gt_pose)):
                    angle_loss = np.sum((answer[t][:3] - gt_pose[t, :3]) ** 2)
                    translation_loss = np.sum((answer[t][3:] - gt_pose[t, 3:6]) ** 2)
                    loss += (100 * angle_loss + translation_loss)
                loss /= len(gt_pose)
                print('Loss = ', loss)
                print('=' * 50)

        fd.close()

    if config["use_lidar"]:
        # LoRCoN-LO Testing (from lorcon_lo/test.py)
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

        checkpoint_path = os.path.join(cp_folder, config["lorcon_lo"]["model_path"])

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
        with torch.no_grad():
            for idx, data in tqdm(enumerate(test_dataloader), total=test_data_loader_len):
                inputs, labels = data
                inputs, labels = Variable(inputs.float().to(device)), Variable(labels.float().to(device))
                outputs = model(inputs)
                Y_estimated_data = np.vstack((Y_estimated_data, outputs[:, -1, :].cpu().numpy()))
                test_loss += WeightedLoss.RMSEError(outputs, labels).item()
        print(f"Test loss is {test_loss / test_data_loader_len}")

        Y_origin_data = get_original_poses(pose_folder, preprocessed_folder, data_seqs)
        seq_sizes = {}
        seq_sizes = count_seq_sizes(preprocessed_folder, data_seqs, seq_sizes)
        
        plot_gt(Y_origin_data, pose_folder, preprocessed_folder, data_seqs, seq_sizes, dataset=dataset)
        plot_results(Y_origin_data, Y_estimated_data, data_seqs, rnn_size, seq_sizes, dataset=dataset)
        save_poses(Y_origin_data, Y_estimated_data, data_seqs, rnn_size, seq_sizes, dataset=dataset)

if __name__ == "__main__":
    main()