import os
import numpy as np
import torch
from torch.utils.data import Dataset

def count_seq_sizes(preprocessed_folder, data_seqs, seq_sizes, depth_name="depth"):
    for seq in data_seqs:
        parent_directory = os.path.join(preprocessed_folder, seq)
        npy_directory = os.path.join(parent_directory, depth_name)
        npy_names = os.listdir(npy_directory)
        seq_sizes[seq] = len(npy_names)
    return seq_sizes

def process_input_data(preprocessed_folder, relative_pose_folder, data_seqs, seq_sizes, depth_name="depth"):
    Y_data = np.empty((0, 2), dtype=object)
    for seq in data_seqs:
        with open(os.path.join(relative_pose_folder, seq + ".txt"), "r") as f:
            parent_directory = os.path.join(preprocessed_folder, seq)
            npy_directory = os.path.join(parent_directory, depth_name)
            npy_names = sorted(os.listdir(npy_directory))
            lines = f.readlines()
            # Use minimum of poses and files to avoid mismatch
            min_len = min(len(lines), len(npy_names))
            if len(lines) != len(npy_names):
                print(f"Warning: Mismatch for {seq}: {len(lines)} poses vs {len(npy_names)} files, using {min_len}")
            Y_row = np.zeros((min_len, 2), dtype=object)
            for i in range(min_len):
                Y_row[i, 0] = f"{seq} {npy_names[i]}"
                Y_row[i, 1] = lines[i]
            Y_data = np.vstack((Y_data, Y_row))
    return Y_data

class LoRCoNLODataset(Dataset):
    def __init__(self, img_dir, Y_data, data_idx, seq_sizes, rnn_size, width, height, depth_name, intensity_name, normal_name, dni_size, normal_size):
        self.img_dir = img_dir
        self.Y_data = Y_data
        self.seq_sizes = seq_sizes
        self.rnn_size = rnn_size
        self.data_idx = data_idx
        self.width = width
        self.height = height
        self.depth_name = depth_name
        self.intensity_name = intensity_name
        self.normal_name = normal_name
        self.dni_size = dni_size
        self.normal_size = normal_size

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        start_id = self.data_idx[idx]
        image_pre_path_args = self.Y_data[start_id, 0].split(" ")
        current_seq = image_pre_path_args[0]
        current_npy_id = int(image_pre_path_args[1].split(".")[0])
        
        images = torch.zeros(self.rnn_size, 10, self.height, self.width, dtype=torch.float32)
        labels = torch.zeros(self.rnn_size, 6, dtype=torch.float32)
        
        for i in range(self.rnn_size):
            current_npy_filename = f"{current_npy_id + i:06d}.npy"
            nxt_npy_filename = f"{current_npy_id + i + 1:06d}.npy"
            images_wrapper = torch.zeros(10, self.height, self.width, dtype=torch.float32)
            
            for j, name in enumerate([self.depth_name, self.intensity_name, self.normal_name]):
                pre_path = os.path.join(self.img_dir, current_seq, name, current_npy_filename)
                image_pre = torch.from_numpy(np.load(pre_path)).float()
                if j == 2:
                    image_pre = image_pre.permute(2, 0, 1)
                
                nxt_path = os.path.join(self.img_dir, current_seq, name, nxt_npy_filename)
                try:
                    image_nxt = torch.from_numpy(np.load(nxt_path)).float()
                    if j == 2:
                        image_nxt = image_nxt.permute(2, 0, 1)
                except FileNotFoundError:
                    image_nxt = torch.zeros_like(image_pre)
                
                if j < 2:
                    image_pre /= 255.0
                    image_nxt /= 255.0
                else:
                    image_pre = (image_pre + 1.0) / 2.0
                    image_nxt = (image_nxt + 1.0) / 2.0
                
                if j < 2:
                    images_wrapper[j] = image_pre
                    images_wrapper[j + self.dni_size] = image_nxt
                else:
                    images_wrapper[j:j + self.normal_size] = image_pre
                    images_wrapper[j + self.dni_size:j + self.dni_size + self.normal_size] = image_nxt
            
            label = np.array(list(map(float, self.Y_data[start_id + i, 1].strip().split(" "))), dtype=np.float32)
            labels[i] = torch.from_numpy(label)
            images[i] = images_wrapper
        
        return images, labels