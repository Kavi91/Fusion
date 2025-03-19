import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Visualization Code
base_dir = "/home/kavi/Datasets/KITTI_raw/kitti_data/preprocessed_data"
sequence = "04"

depth_file = os.path.join(base_dir, sequence, "depth", "000000.npy")
intensity_file = os.path.join(base_dir, sequence, "intensity", "000000.npy")
normal_file = os.path.join(base_dir, sequence, "normal", "000000.npy")
rgb_file = os.path.join(base_dir, sequence, "rgb", "000000.npy")

def load_and_print(file_path, data_type):
    if os.path.exists(file_path):
        data = np.load(file_path)
        print(f"{data_type} file: {file_path}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Min value: {np.min(data)}")
        print(f"Max value: {np.max(data)}")
        print(f"Sample data (first 5 elements): {data.flatten()[:5]}")
        print("-" * 50)
        return data
    else:
        print(f"Error: {file_path} does not exist")
        return None

depth_data = load_and_print(depth_file, "Depth")
intensity_data = load_and_print(intensity_file, "Intensity")
normal_data = load_and_print(normal_file, "Normals")
rgb_data = load_and_print(rgb_file, "RGB")

fig, axes = plt.subplots(4, 1, figsize=(24, 24))

if depth_data is not None:
    axes[0].imshow(depth_data, cmap='plasma', interpolation='nearest')
    axes[0].set_title("Spherical Depth Map")
    axes[0].axis('off')

if intensity_data is not None:
    axes[1].imshow(intensity_data, cmap='gray', interpolation='nearest')
    axes[1].set_title("Spherical Intensity Map")
    axes[1].axis('off')

if normal_data is not None:
    normal_vis = (normal_data + 1) / 2
    normal_vis = np.where(normal_data == -1, 0, normal_vis)
    axes[2].imshow(normal_vis, interpolation='nearest')
    axes[2].set_title("Spherical Normal Map (RGB)")
    axes[2].axis('off')

if rgb_data is not None:
    axes[3].imshow(rgb_data, interpolation='nearest')
    axes[3].set_title("Spherical RGB Map")
    axes[3].axis('off')

plt.tight_layout()
plt.show()

# Dataset Class
class KITTIFusionDataset(Dataset):
    def __init__(self, data_dir, rgb_orig_dir, seq_length=2):
        self.data_dir = data_dir
        self.rgb_orig_dir = rgb_orig_dir
        self.seq_length = seq_length
        self.subdirs = ['depth', 'intensity', 'normal', 'rgb']
        self.files = self._load_files()

    def _load_files(self):
        files = {}
        for subdir in self.subdirs:
            subdir_path = os.path.join(self.data_dir, subdir)
            files[subdir] = sorted([f for f in os.listdir(subdir_path) if f.endswith('.npy')])
        return files

    def __len__(self):
        return min([len(self.files[subdir]) - self.seq_length + 1 for subdir in self.subdirs])

    def __getitem__(self, idx):
        data = {}
        for subdir in self.subdirs:
            sequence = [np.load(os.path.join(self.data_dir, subdir, self.files[subdir][idx + i])) for i in range(self.seq_length)]
            data[subdir] = np.stack(sequence, axis=0)  # Shape: (seq_length, H, W, C)
        # Load original RGB (1226x370)
        rgb_orig_paths = [os.path.join(self.rgb_orig_dir, f"{int(self.files['rgb'][idx + i].replace('.npy', '')):06d}.png") for i in range(self.seq_length)]
        rgb_orig = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) / 255.0 for p in rgb_orig_paths]
        data['rgb_original'] = np.stack(rgb_orig, axis=0)  # Shape: (seq_length, 370, 1226, 3)
        return {k: torch.from_numpy(v).permute(0, 3, 1, 2).float() for k, v in data.items()}

# Usage
dataset = KITTIFusionDataset(
    data_dir='/home/kavi/Datasets/KITTI_raw/kitti_data/preprocessed_data/04',
    rgb_orig_dir='/home/kavi/Datasets/KITTI_raw/kitti_data/sequences/04/image_02'
)
print(f"Dataset size: {len(dataset)}")