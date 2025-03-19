import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import glob
from torchvision import transforms

class FusionDataset(Dataset):
    def __init__(self, config, seqs, seq_len):
        self.rgb_high_dir = config["deepvo"]["image_dir"]
        self.lidar_dir = config["lorcon_lo"]["preprocessed_folder"]
        self.pose_dir = config["deepvo"]["pose_dir"]
        self.seqs = seqs
        self.seq_len = seq_len
        
        # Read modality selection flags from config
        self.use_depth = config["fusion"]["modalities"]["use_depth"]
        self.use_intensity = config["fusion"]["modalities"]["use_intensity"]
        self.use_normals = config["fusion"]["modalities"]["use_normals"]
        self.use_rgb_low = config["fusion"]["modalities"]["use_rgb_low"]
        
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((184, 608)),
            transforms.ToTensor()
        ])
        
        self.data = []
        for seq in seqs:
            rgb_high_files = sorted(glob.glob(os.path.join(self.rgb_high_dir, seq, "image_02", "*.png")))
            rgb_low_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "rgb", "*.npy")))
            depth_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "depth", "*.npy")))
            intensity_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "intensity", "*.npy")))
            normal_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "normal", "*.npy")))
            poses = np.load(os.path.join(self.pose_dir, f"{seq}.npy"))
            min_len = min(len(rgb_high_files), len(rgb_low_files), len(depth_files), len(intensity_files), len(normal_files), len(poses))
            for i in range(min_len - seq_len):
                self.data.append((
                    rgb_high_files[i:i+seq_len],
                    rgb_low_files[i:i+seq_len],
                    depth_files[i:i+seq_len],
                    intensity_files[i:i+seq_len],
                    normal_files[i:i+seq_len],
                    poses[i:i+seq_len]
                ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        rgb_high_paths, rgb_low_paths, depth_paths, intensity_paths, normal_paths, poses = self.data[idx]
        
        # High-res RGB (184x608)
        rgb_high = torch.stack([self.rgb_transform(cv2.imread(p, cv2.COLOR_BGR2RGB)) for p in rgb_high_paths])
        
        # Build lidar_combined based on selected modalities
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
        
        # Convert absolute poses to relative poses
        poses = torch.from_numpy(poses).float()
        relative_poses = poses[1:] - poses[:-1]
        relative_poses = torch.cat([torch.zeros(1, 6), relative_poses], dim=0)
        
        return rgb_high, lidar_combined, relative_poses

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    dataset = FusionDataset(config, ["00"], 2)
    rgb_high, lidar_combined, poses = dataset[0]
    print(f"RGB High shape: {rgb_high.shape}, LiDAR Combined shape: {lidar_combined.shape}, Poses shape: {poses.shape}")