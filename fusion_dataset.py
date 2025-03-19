import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import glob
from torchvision import transforms

class FusionDataset(Dataset):
    def __init__(self, config, seqs, seq_len):
        print("Initializing FusionDataset...")
        print(f"Config keys: {list(config.keys())}")
        self.rgb_high_dir = config["deepvo"]["image_dir"]
        print(f"RGB High Dir: {self.rgb_high_dir}")
        self.lidar_dir = config["lorcon_lo"]["preprocessed_folder"]
        print(f"LiDAR Dir: {self.lidar_dir}")
        self.pose_dir = config["deepvo"]["pose_dir"]
        print(f"Pose Dir: {self.pose_dir}")
        self.seqs = seqs
        print(f"Sequences: {self.seqs}")
        self.seq_len = seq_len
        print(f"Sequence Length: {self.seq_len}")
        
        try:
            print(f"Fusion modalities: {config['fusion']['modalities']}")
            self.use_depth = config["fusion"]["modalities"]["use_depth"]
            self.use_intensity = config["fusion"]["modalities"]["use_intensity"]
            self.use_normals = config["fusion"]["modalities"]["use_normals"]
            self.use_rgb_low = config["fusion"]["modalities"]["use_rgb_low"]
            print(f"Modality Flags - Depth: {self.use_depth}, Intensity: {self.use_intensity}, "
                  f"Normals: {self.use_normals}, RGB Low: {self.use_rgb_low}")
        except KeyError as e:
            print(f"Config KeyError: {e}")
            raise
        
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((184, 608)),
            transforms.ToTensor()
        ])
        print("RGB Transform initialized")
        
        self.data = []
        for seq in seqs:
            print(f"Processing sequence {seq}...")
            rgb_high_files = sorted(glob.glob(os.path.join(self.rgb_high_dir, seq, "image_02", "*.png")))
            print(f"RGB High files found: {len(rgb_high_files)}")
            
            poses_path = os.path.join(self.pose_dir, f"{seq}.npy")
            print(f"Loading poses from {poses_path}")
            poses = np.load(poses_path)
            print(f"Poses loaded: {len(poses)}, shape: {poses.shape}")
            
            depth_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "depth", "*.npy"))) if self.use_depth else rgb_high_files
            intensity_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "intensity", "*.npy"))) if self.use_intensity else rgb_high_files
            normal_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "normal", "*.npy"))) if self.use_normals else rgb_high_files
            rgb_low_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "rgb", "*.npy"))) if self.use_rgb_low else rgb_high_files
            
            print(f"Sequence {seq}: RGB High: {len(rgb_high_files)}, Depth: {len(depth_files)}, "
                  f"Intensity: {len(intensity_files)}, Normals: {len(normal_files)}, RGB Low: {len(rgb_low_files)}, Poses: {len(poses)}")
            
            min_len = min(len(rgb_high_files), len(depth_files), len(intensity_files), len(normal_files), len(rgb_low_files), len(poses))
            print(f"min_len for {seq}: {min_len}")
            if min_len <= seq_len:
                print(f"Warning: Sequence {seq} has min_len={min_len} <= seq_len={seq_len}, skipping...")
                continue
            
            print(f"Adding {min_len - seq_len} samples for {seq}")
            for i in range(min_len - seq_len):
                self.data.append((
                    rgb_high_files[i:i+seq_len],
                    rgb_low_files[i:i+seq_len],
                    depth_files[i:i+seq_len],
                    intensity_files[i:i+seq_len],
                    normal_files[i:i+seq_len],
                    poses[i:i+seq_len]
                ))
            print(f"Sequence {seq} processed, current data length: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        rgb_high_paths, rgb_low_paths, depth_paths, intensity_paths, normal_paths, poses = self.data[idx]
        
        rgb_high = torch.stack([self.rgb_transform(cv2.imread(p, cv2.COLOR_BGR2RGB)) for p in rgb_high_paths])
        
        channels = []
        if self.use_rgb_low:
            rgb_low = torch.stack([torch.from_numpy(np.load(p)) for p in rgb_low_paths])
            channels.append(rgb_low)
        if self.use_depth:
            depth = torch.stack([torch.from_numpy(np.load(p)).unsqueeze(0) for p in depth_paths])
            channels.append(depth)
        if self.use_intensity:
            intensity = torch.stack([torch.from_numpy(np.load(p)).unsqueeze(0) for p in intensity_paths])
            channels.append(intensity)
        if self.use_normals:
            normals = torch.stack([torch.from_numpy(np.load(p)) for p in normal_paths])
            channels.append(normals)
        
        if not channels:
            raise ValueError("No modalities selected for lidar_combined")
        lidar_combined = torch.cat(channels, dim=1)
        
        # Convert 15-element absolute poses to 6-DoF relative poses
        poses = torch.from_numpy(poses).float()  # (seq_len, 15)
        print(f"Pose shape in __getitem__: {poses.shape}")
        
        # Extract 6-DoF: Euler angles (0:3) and translation (3:6)
        poses_6dof = poses[:, :6]  # (seq_len, 6) - [x, y, z, tx, ty, tz]
        
        # Compute relative poses
        relative_poses = poses_6dof[1:] - poses_6dof[:-1]  # (seq_len-1, 6)
        relative_poses = torch.cat([torch.zeros(1, 6), relative_poses], dim=0)  # (seq_len, 6)
        
        print(f"Relative poses shape: {relative_poses.shape}")
        return rgb_high, lidar_combined, relative_poses

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset = FusionDataset(config, ["00"], 2)
    rgb_high, lidar_combined, poses = dataset[0]
    print(f"RGB High shape: {rgb_high.shape}, LiDAR Combined shape: {lidar_combined.shape}, Poses shape: {poses.shape}")