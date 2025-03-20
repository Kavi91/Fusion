import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import glob
from torchvision import transforms
import torch.nn.functional as F

class FusionDataset(Dataset):
    def __init__(self, config, seqs, seq_len):
        #print("Initializing FusionDataset...")
        #print(f"Config keys: {list(config.keys())}")
        self.rgb_high_dir = config["deepvo"]["image_dir"]
        #print(f"RGB High Dir: {self.rgb_high_dir}")
        self.lidar_dir = config["lorcon_lo"]["preprocessed_folder"]
        #print(f"LiDAR Dir: {self.lidar_dir}")
        self.pose_dir = config["deepvo"]["pose_dir"]
        #print(f"Pose Dir: {self.pose_dir}")
        self.seqs = seqs
        #print(f"Sequences: {self.seqs}")
        self.seq_len = seq_len
        #print(f"Sequence Length: {self.seq_len}")
        
        try:
            #print(f"Fusion modalities: {config['fusion']['modalities']}")
            self.use_depth = config["fusion"]["modalities"]["use_depth"]
            self.use_intensity = config["fusion"]["modalities"]["use_intensity"]
            self.use_normals = config["fusion"]["modalities"]["use_normals"]
            self.use_rgb_low = config["fusion"]["modalities"]["use_rgb_low"]
            print(f"Modality Flags - Depth: {self.use_depth}, Intensity: {self.use_intensity}, "
                  f"Normals: {self.use_normals}, RGB Low: {self.use_rgb_low}")
        except KeyError as e:
            print(f"Config KeyError: {e}")
            raise
        
        # Add normalization to match DeepVO
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((184, 608)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["deepvo"]["img_means"], std=config["deepvo"]["img_stds"])
        ])
        self.rgb_low_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 900)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["deepvo"]["img_means"], std=config["deepvo"]["img_stds"])
        ])
        #print("RGB Transforms initialized")
        
        self.data = []
        for seq in seqs:
            #print(f"Processing sequence {seq}...")
            rgb_high_files = sorted(glob.glob(os.path.join(self.rgb_high_dir, seq, "image_02", "*.png")))
            #print(f"RGB High files found: {len(rgb_high_files)}")
            
            poses_path = os.path.join(self.pose_dir, f"{seq}.npy")
            #print(f"Loading poses from {poses_path}")
            poses = np.load(poses_path)
            #print(f"Poses loaded: {len(poses)}, shape: {poses.shape}")
            
            depth_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "depth", "*.npy"))) if self.use_depth else rgb_high_files
            intensity_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "intensity", "*.npy"))) if self.use_intensity else rgb_high_files
            normal_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "normal", "*.npy"))) if self.use_normals else rgb_high_files
            rgb_low_files = rgb_high_files if self.use_rgb_low else rgb_high_files  # Use same files as rgb_high
            
            print(f"Sequence {seq}: RGB High: {len(rgb_high_files)}, Depth: {len(depth_files)}, "
                  f"Intensity: {len(intensity_files)}, Normals: {len(normal_files)}, RGB Low: {len(rgb_low_files)}, Poses: {len(poses)}")
            
            min_len = min(len(rgb_high_files), len(depth_files), len(intensity_files), len(normal_files), len(rgb_low_files), len(poses))
            #print(f"min_len for {seq}: {min_len}")
            if min_len <= seq_len:
                #print(f"Warning: Sequence {seq} has min_len={min_len} <= seq_len={seq_len}, skipping...")
                continue
            
            #print(f"Adding {min_len - seq_len} samples for {seq}")
            for i in range(min_len - seq_len):
                self.data.append((
                    rgb_high_files[i:i+seq_len],
                    rgb_low_files[i:i+seq_len],
                    depth_files[i:i+seq_len],
                    intensity_files[i:i+seq_len],
                    normal_files[i:i+seq_len],
                    poses[i:i+seq_len]
                ))
            #print(f"Sequence {seq} processed, current data length: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        rgb_high_paths, rgb_low_paths, depth_paths, intensity_paths, normal_paths, poses = self.data[idx]
        
        rgb_high = torch.stack([self.rgb_transform(cv2.imread(p, cv2.COLOR_BGR2RGB)) for p in rgb_high_paths])
        
        channels = []
        if self.use_rgb_low:
            rgb_low = torch.stack([self.rgb_low_transform(cv2.imread(p, cv2.COLOR_BGR2RGB)) for p in rgb_low_paths])
            #print(f"RGB Low shape: {rgb_low.shape}")
            channels.append(rgb_low)
        if self.use_depth:
            depth_list = [torch.from_numpy(np.load(p)) for p in depth_paths]
            #print(f"Depth shapes: {[d.shape for d in depth_list]}")
            depth_list = [d.transpose(0, 1) if d.shape[0] == 900 else d for d in depth_list]
            depth = torch.stack(depth_list)
            #print(f"Depth stacked shape: {depth.shape}")
            if depth.shape[-1] != 900 or depth.shape[-2] != 64:
                depth = F.interpolate(depth.unsqueeze(0), size=(64, 900), mode='nearest').squeeze(0)
            depth = depth.unsqueeze(1)
            #print(f"Depth final shape: {depth.shape}")
            channels.append(depth)
        if self.use_intensity:
            intensity_list = [torch.from_numpy(np.load(p)) for p in intensity_paths]
            #print(f"Intensity shapes: {[i.shape for i in intensity_list]}")
            intensity_list = [i.transpose(0, 1) if i.shape[0] == 900 else i for i in intensity_list]
            intensity = torch.stack(intensity_list)
            #print(f"Intensity stacked shape: {intensity.shape}")
            if intensity.shape[-1] != 900 or intensity.shape[-2] != 64:
                intensity = F.interpolate(intensity.unsqueeze(0), size=(64, 900), mode='nearest').squeeze(0)
            intensity = intensity.unsqueeze(1)
            #print(f"Intensity final shape: {intensity.shape}")
            channels.append(intensity)
        if self.use_normals:
            normals_list = [torch.from_numpy(np.load(p)) for p in normal_paths]
            #print(f"Normals shapes: {[n.shape for n in normals_list]}")
            normals_list = [n.permute(2, 0, 1) for n in normals_list]
            normals = torch.stack(normals_list)
            #print(f"Normals stacked shape: {normals.shape}")
            if normals.shape[-1] != 900 or normals.shape[-2] != 64:
                normals = F.interpolate(normals, size=(64, 900), mode='nearest')
            #print(f"Normals final shape: {normals.shape}")
            channels.append(normals)
        
        if not channels:
            raise ValueError("No modalities selected for lidar_combined")
        #print(f"Channels shapes before concat: {[c.shape for c in channels]}")
        lidar_combined = torch.cat(channels, dim=1)
        #print(f"LiDAR Combined shape: {lidar_combined.shape}")
        
        poses = torch.from_numpy(poses).float()
        poses_6dof = poses[:, :6]
        relative_poses = poses_6dof[1:] - poses_6dof[:-1]
        relative_poses = torch.cat([torch.zeros(1, 6), relative_poses], dim=0)
        
        return rgb_high, lidar_combined, relative_poses

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset = FusionDataset(config, ["00"], 2)
    rgb_high, lidar_combined, poses = dataset[0]
    #print(f"RGB High shape: {rgb_high.shape}, LiDAR Combined shape: {lidar_combined.shape}, Poses shape: {poses.shape}")