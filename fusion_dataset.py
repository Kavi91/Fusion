import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import glob
from torchvision import transforms
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter  # Move import to top

class FusionDataset(Dataset):
    def __init__(self, config, seqs, seq_len, use_augmentation=False):
        self.rgb_high_dir = config["deepvo"]["image_dir"]
        self.lidar_dir = config["lorcon_lo"]["preprocessed_folder"]
        self.pose_dir = config["deepvo"]["pose_dir"]  # Use config value
        self.calib_dir = config["deepvo"]["calib_folder"]
        self.seqs = seqs
        self.seq_len = seq_len
        self.use_augmentation = use_augmentation
        
        try:
            self.use_depth = config["fusion"]["modalities"]["use_depth"]
            self.use_intensity = config["fusion"]["modalities"]["use_intensity"]
            self.use_normals = config["fusion"]["modalities"]["use_normals"]
            self.use_rgb_low = config["fusion"]["modalities"]["use_rgb_low"]
        except KeyError as e:
            raise KeyError(f"Config KeyError: {e}")
        
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((64, 900)),  # Resize RGB to match LiDAR resolution
        ]
        if self.use_augmentation:
            transform_list.extend([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ])
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=config["deepvo"]["img_means"], std=config["deepvo"]["img_stds"])
        ])
        self.rgb_transform = transforms.Compose(transform_list)

        transform_list_low = [
            transforms.ToPILImage(),
            transforms.Resize((64, 900)),
        ]
        if self.use_augmentation:
            transform_list_low.extend([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ])
        transform_list_low.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=config["deepvo"]["img_means"], std=config["deepvo"]["img_stds"])
        ])
        self.rgb_low_transform = transforms.Compose(transform_list_low)
        
        self.sequence_lengths = {}
        self.total_length = 0
        for seq in seqs:
            rgb_high_files = sorted(glob.glob(os.path.join(self.rgb_high_dir, seq, "image_02", "*.png")))
            depth_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "depth", "*.npy"))) if self.use_depth else rgb_high_files
            intensity_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "intensity", "*.npy"))) if self.use_intensity else rgb_high_files
            normal_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "normal", "*.npy"))) if self.use_normals else rgb_high_files
            poses_path = os.path.join(self.pose_dir, f"{seq}.npy")
            poses_7dof = torch.from_numpy(np.load(poses_path)).float()
            
            min_len = min(len(rgb_high_files), len(depth_files), len(intensity_files), len(normal_files), len(poses_7dof))
            if min_len <= seq_len:
                continue
            
            self.sequence_lengths[seq] = min_len - seq_len
            self.total_length += min_len - seq_len
        
        self.Tr_velo_to_cam2 = {}
        for seq in seqs:
            calib_path = os.path.join(self.calib_dir, f"{seq}.txt")
            Tr_velo_to_cam0 = None
            P0 = None
            P2 = None
            try:
                with open(calib_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('P0'):
                            P0 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
                        elif line.startswith('P2'):
                            P2 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
                        elif line.startswith('Tr'):
                            Tr_velo_to_cam0 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
                            Tr_velo_to_cam0 = np.vstack([Tr_velo_to_cam0, np.array([0, 0, 0, 1])])
                if Tr_velo_to_cam0 is None:
                    raise ValueError(f"Tr not found in calibration file {calib_path}")
                if P0 is None:
                    raise ValueError(f"P0 not found in calibration file {calib_path}")
                if P2 is None:
                    raise ValueError(f"P2 not found in calibration file {calib_path}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Calibration file {calib_path} not found")
            except Exception as e:
                raise ValueError(f"Error reading calibration file {calib_path}: {str(e)}")
            
            K = P0[:, :3]
            t0 = P0[:, 3]
            t2 = P2[:, 3]
            baseline = t2 - t0
            Tr_cam0_to_cam2 = np.eye(4)
            Tr_cam0_to_cam2[:3, 3] = baseline
            
            Tr_velo_to_cam0 = torch.from_numpy(Tr_velo_to_cam0).float()
            Tr_cam0_to_cam2 = torch.from_numpy(Tr_cam0_to_cam2).float()
            Tr_velo_to_cam2 = Tr_cam0_to_cam2 @ Tr_velo_to_cam0
            self.Tr_velo_to_cam2[seq] = Tr_velo_to_cam2
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        cumsum = 0
        seq = None
        local_idx = None
        for s, length in self.sequence_lengths.items():
            if idx < cumsum + length:
                seq = s
                local_idx = idx - cumsum
                break
            cumsum += length
        
        if seq is None:
            raise IndexError(f"Index {idx} out of range for dataset length {self.total_length}")
        
        rgb_high_files = sorted(glob.glob(os.path.join(self.rgb_high_dir, seq, "image_02", "*.png")))
        depth_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "depth", "*.npy"))) if self.use_depth else rgb_high_files
        intensity_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "intensity", "*.npy"))) if self.use_intensity else rgb_high_files
        normal_files = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "normal", "*.npy"))) if self.use_normals else rgb_high_files
        rgb_low_files = rgb_high_files if self.use_rgb_low else rgb_high_files
        poses_path = os.path.join(self.pose_dir, f"{seq}.npy")
        poses_7dof = torch.from_numpy(np.load(poses_path)).float()
        
        min_len = min(len(rgb_high_files), len(depth_files), len(intensity_files), len(normal_files), len(poses_7dof))
        if min_len <= self.seq_len:
            raise ValueError(f"Sequence {seq} has insufficient length: {min_len}")
        
        for i in range(min_len):
            rgb_idx = int(os.path.basename(rgb_high_files[i]).split('.')[0])
            depth_idx = int(os.path.basename(depth_files[i]).split('.')[0]) if self.use_depth else rgb_idx
            pose_idx = i
            if rgb_idx != depth_idx or rgb_idx != pose_idx:
                raise ValueError(f"Mismatch in indices at position {i} in sequence {seq}: RGB {rgb_idx}, Depth {depth_idx}, Pose {pose_idx}")
        
        start_idx = local_idx
        end_idx = start_idx + self.seq_len
        if end_idx > min_len:
            raise IndexError(f"Index range {start_idx}:{end_idx} out of bounds for sequence {seq} with length {min_len}")
        
        rgb_high_paths = rgb_high_files[start_idx:end_idx]
        rgb_low_paths = rgb_low_files[start_idx:end_idx]
        depth_paths = depth_files[start_idx:end_idx]
        intensity_paths = intensity_files[start_idx:end_idx]
        normal_paths = normal_files[start_idx:end_idx]
        poses = poses_7dof[start_idx:end_idx]
        
        rgb_high_list = []
        for p in rgb_high_paths:
            img = cv2.imread(p, cv2.COLOR_BGR2RGB)
            if img is None:
                raise ValueError(f"Failed to load RGB image at {p}")
            img = self.rgb_transform(img)
            if torch.isnan(img).any() or torch.isinf(img).any():
                raise ValueError(f"NaN or Inf detected in RGB image at {p}")
            rgb_high_list.append(img)
        rgb_high = torch.stack(rgb_high_list)
        if torch.isnan(rgb_high).any() or torch.isinf(rgb_high).any():
            raise ValueError(f"NaN or Inf detected in rgb_high at index {idx}")
        
        channels = []
        if self.use_rgb_low:
            rgb_low_list = []
            for p in rgb_low_paths:
                img = cv2.imread(p, cv2.COLOR_BGR2RGB)
                if img is None:
                    raise ValueError(f"Failed to load RGB low-res image at {p}")
                img = self.rgb_low_transform(img)
                if torch.isnan(img).any() or torch.isinf(img).any():
                    raise ValueError(f"NaN or Inf detected in RGB low-res image at {p}")
                rgb_low_list.append(img)
            rgb_low = torch.stack(rgb_low_list)
            if torch.isnan(rgb_low).any() or torch.isinf(rgb_low).any():
                raise ValueError(f"NaN or Inf detected in rgb_low at index {idx}")
            channels.append(rgb_low)
        
        if self.use_depth:
            depth_list = []
            for p in depth_paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Depth file {p} does not exist")
                depth = torch.from_numpy(np.load(p))
                if torch.isnan(depth).any() or torch.isinf(depth).any():
                    raise ValueError(f"NaN or Inf detected in depth map at {p}")
                # Fill missing values using interpolation
                mask = (depth == 0) | torch.isnan(depth)
                if mask.any():
                    depth[mask] = float('nan')
                    depth_np = depth.numpy()
                    depth_np = gaussian_filter(depth_np, sigma=1)
                    depth = torch.from_numpy(depth_np)
                depth = depth.transpose(0, 1) if depth.shape[0] == 900 else depth
                if self.use_augmentation:
                    depth = depth * torch.FloatTensor(depth.shape).uniform_(0.8, 1.2)
                    depth += torch.randn_like(depth) * 0.05
                depth_list.append(depth)
            depth = torch.stack(depth_list)
            if depth.shape[-1] != 900 or depth.shape[-2] != 64:
                depth = F.interpolate(depth.unsqueeze(0), size=(64, 900), mode='nearest').squeeze(0)
                if torch.isnan(depth).any() or torch.isinf(depth).any():
                    raise ValueError(f"NaN or Inf detected in depth map after interpolation at index {idx}")
            depth = depth.unsqueeze(1)  # Shape: [seq_len, 1, 64, 900]
            channels.append(depth)
        
        if self.use_intensity:
            intensity_list = []
            for p in intensity_paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Intensity file {p} does not exist")
                intensity = torch.from_numpy(np.load(p))
                if torch.isnan(intensity).any() or torch.isinf(intensity).any():
                    raise ValueError(f"NaN or Inf detected in intensity map at {p}")
                intensity = intensity.transpose(0, 1) if intensity.shape[0] == 900 else intensity
                intensity_list.append(intensity)
            intensity = torch.stack(intensity_list)
            if intensity.shape[-1] != 900 or intensity.shape[-2] != 64:
                intensity = F.interpolate(intensity.unsqueeze(0), size=(64, 900), mode='nearest').squeeze(0)
                if torch.isnan(intensity).any() or torch.isinf(intensity).any():
                    raise ValueError(f"NaN or Inf detected in intensity map after interpolation at index {idx}")
            intensity = intensity.unsqueeze(1)
            channels.append(intensity)
        
        if self.use_normals:
            normals_list = []
            for p in normal_paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Normal file {p} does not exist")
                normals = torch.from_numpy(np.load(p))
                if torch.isnan(normals).any() or torch.isinf(normals).any():
                    raise ValueError(f"NaN or Inf detected in normals map at {p}")
                normals = normals.permute(2, 0, 1) if normals.shape[-1] == 3 else normals.transpose(0, 1)
                normals_list.append(normals)
            normals = torch.stack(normals_list)
            if normals.shape[-1] != 900 or normals.shape[-2] != 64:
                normals = F.interpolate(normals, size=(64, 900), mode='nearest')
                if torch.isnan(normals).any() or torch.isinf(normals).any():
                    raise ValueError(f"NaN or Inf detected in normals map after interpolation at index {idx}")
            channels.append(normals)
        
        if not channels:
            raise ValueError("No modalities selected for lidar_combined")
        lidar_combined = torch.cat(channels, dim=1)
        if torch.isnan(lidar_combined).any() or torch.isinf(lidar_combined).any():
            raise ValueError(f"NaN or Inf detected in lidar_combined at index {idx}")
        
        relative_poses = poses[1:] - poses[:-1]  # Shape: [seq_len-1, 7]
        # Do not add the initial zero pose, so the target matches the model's output
        if torch.isnan(relative_poses).any() or torch.isinf(relative_poses).any():
            raise ValueError(f"NaN or Inf detected in relative_poses at index {idx}")
        
        return rgb_high, lidar_combined, relative_poses

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset = FusionDataset(config, ["00"], 2)
    rgb_high, lidar_combined, poses = dataset[0]