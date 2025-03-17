# FUSION/fusion_dataset.py
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from deepvo.data_helper import get_data_info, ImageSequenceDataset
from lorcon_lo.process_data import LoRCoNLODataset, count_seq_sizes, process_input_data
import transforms3d as t3d
from Fusion_utils import create_frame_mapping

class FusionDataset(Dataset):
    def __init__(self, deepvo_df, lorcon_data, config):
        """
        Custom dataset for fusion, pairing camera and LiDAR sequences with pose alignment.
        
        Args:
            deepvo_df (pd.DataFrame): DeepVO DataFrame from get_data_info.
            lorcon_data (LoRCoNLODataset): LoRCoN-LO dataset.
            config (dict): Configuration dictionary from config.yaml.
        """
        self.deepvo_df = deepvo_df
        self.lorcon_data = lorcon_data
        self.config = config
        self.seq_len = int((config["deepvo"]["seq_len"][0] + config["deepvo"]["seq_len"][1]) / 2)
        self.rnn_size = config["lorcon_lo"]["rnn_size"]
        self.calib_folder = config["deepvo"]["calib_folder"]

        # Create frame mapping based on timestamps
        self.frame_pairs = create_frame_mapping(deepvo_df, lorcon_data, config)
        self.length = len(self.frame_pairs)

        # Load calibration data
        self.calib_data = {}
        for seq in config["deepvo"]["train_video"] + config["deepvo"]["valid_video"]:
            calib_file = os.path.join(self.calib_folder, f"{seq}.txt")
            if os.path.exists(calib_file):
                with open(calib_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if "Tr_velo_to_cam" in line:
                            vals = [float(x) for x in line.strip().split()[1:]]
                            self.calib_data[seq] = np.array(vals).reshape(3, 4)
                            break
        self.calib_matrices = {seq: np.vstack([self.calib_data[seq], [0, 0, 0, 1]]) for seq in self.calib_data}

    def transform_absolute_pose(self, absolute_pose, calib_matrix):
        """
        Transform a 6D absolute pose from velodyne to camera frame.
        """
        rot_vec = absolute_pose[:3]
        trans_vec = absolute_pose[3:]
        rot_matrix = t3d.euler.euler2mat(rot_vec[0], rot_vec[1], rot_vec[2], 'sxyz')
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rot_matrix
        pose_matrix[:3, 3] = trans_vec
        transformed_matrix = np.linalg.inv(calib_matrix) @ pose_matrix @ calib_matrix
        new_rot_matrix = transformed_matrix[:3, :3]
        new_trans = transformed_matrix[:3, 3]
        new_rot_vec = t3d.euler.mat2euler(new_rot_matrix, 'sxyz')
        return np.concatenate([new_rot_vec, new_trans])

    def compute_relative_pose(self, poses):
        """
        Compute relative poses from absolute poses.
        """
        relative_poses = []
        for i in range(1, len(poses)):
            # Simplified: Subtract poses (adjust based on DeepVO's logic)
            relative_pose = poses[i] - poses[i-1]
            # Normalize angles
            relative_pose[:3] = (relative_pose[:3] + np.pi) % (2 * np.pi) - np.pi
            relative_poses.append(relative_pose)
        return torch.FloatTensor(np.array(relative_poses))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        deepvo_idx, lorcon_idx = self.frame_pairs[index]

        # Load DeepVO data
        deepvo_item = ImageSequenceDataset(self.deepvo_df, self.config["deepvo"]["resize_mode"],
                                          (self.config["deepvo"]["img_w"], self.config["deepvo"]["img_h"]),
                                          self.config["deepvo"]["img_means"], self.config["deepvo"]["img_stds"],
                                          self.config["deepvo"]["minus_point_5"], config=self.config).__getitem__(deepvo_idx)
        seq_len_deepvo, camera_input, camera_target = deepvo_item

        # Load LoRCoN-LO data
        lidar_input, lidar_target = self.lorcon_data[lorcon_idx]

        # Transform LoRCoN-LO poses to camera frame
        video_id = self.deepvo_df.iloc[deepvo_idx]["image_path"][0].split("/")[0]
        if video_id in self.calib_matrices:
            calib_matrix = self.calib_matrices[video_id]
            transformed_targets = []
            for pose in lidar_target:
                transformed_pose = self.transform_absolute_pose(pose.numpy(), calib_matrix)
                transformed_targets.append(transformed_pose)
            lidar_target = torch.FloatTensor(np.array(transformed_targets))
            # Compute relative poses consistently
            lidar_target = self.compute_relative_pose(lidar_target)

        # Ensure DeepVO target is also relative (already handled in ImageSequenceDataset)
        min_length = min(seq_len_deepvo.item(), self.rnn_size)
        camera_input = camera_input[:min_length]
        camera_target = camera_target[:min_length]
        lidar_input = lidar_input[:min_length]
        lidar_target = lidar_target[:min_length]
        seq_len = torch.tensor(min_length)

        return seq_len, camera_input, lidar_input, camera_target