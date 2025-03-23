import torch
import torch.nn as nn
import torch.nn.functional as F
from flownet_models.FlowNetS import FlowNetS
from flownet_models.util import conv

class QuadBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(QuadBranch, self).__init__()
        self.gap = nn.AvgPool2d(kernel_size=2, stride=2)  # Reduced downsampling
        self.fe1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
        )
        self.fe2 = nn.Sequential(
            nn.Conv2d(64, out_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.gap(x)
        fe1_out = self.fe1(x)
        fe2_out = self.fe2(fe1_out)
        return fe2_out

class FusionLIVO(nn.Module):
    def __init__(self, config, lidar_height=64, lidar_width=900, rnn_hidden_size=1000):
        super(FusionLIVO, self).__init__()
        
        self.use_rgb_left = config["fusion"]["modalities"]["use_rgb_left"]
        self.use_rgb_right = config["fusion"]["modalities"]["use_rgb_right"]
        self.use_lidar = config["fusion"]["modalities"]["use_lidar"]
        self.use_depth = config["fusion"]["modalities"]["use_depth"]
        self.use_intensity = config["fusion"]["modalities"]["use_intensity"]
        self.use_normals = config["fusion"]["modalities"]["use_normals"]
        self.use_rgb_low = config["fusion"]["modalities"]["use_rgb_low"]

        if not (self.use_rgb_left or self.use_rgb_right or self.use_lidar):
            raise ValueError("At least one modality (RGB-L, RGB-R, or LiDAR) must be enabled")

        lidar_input_channels = (1 if self.use_depth else 0) + (1 if self.use_intensity else 0) + (3 if self.use_normals else 0) + (3 if self.use_rgb_low else 0)
        if self.use_lidar and lidar_input_channels == 0:
            raise ValueError("No LiDAR modalities selected when use_lidar is True")

        self.rgb_height = config["deepvo"]["img_h"]
        self.rgb_width = config["deepvo"]["img_w"]

        # FlowNetS branches
        pretrained_path = "/home/kavi/Fusion/flownet_models/pytorch/flownets_bn_EPE2.459.pth"
        self.flownet_rgb_left = FlowNetS(batchNorm=True) if self.use_rgb_left else None
        self.flownet_rgb_right = FlowNetS(batchNorm=True) if self.use_rgb_right else None
        self.flownet_lidar = FlowNetS(batchNorm=True) if self.use_lidar else None
        
        if self.use_lidar:
            self.flownet_lidar.conv1 = conv(
                self.flownet_lidar.batchNorm,
                lidar_input_channels * 2,
                64,
                kernel_size=7,
                stride=2
            )

        loaded_data = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        if isinstance(loaded_data, dict) and 'state_dict' in loaded_data:
            state_dict = loaded_data['state_dict']
        else:
            state_dict = loaded_data
        
        conv1_key = 'conv1.0.weight' if 'conv1.0.weight' in state_dict else None
        if conv1_key is None:
            conv1_key = next((key for key in state_dict.keys() if 'conv1' in key.lower()), None)
            if conv1_key is None:
                raise KeyError("Could not find 'conv1' weight in pretrained state_dict. Available keys: " + str(list(state_dict.keys())))
        
        if self.use_rgb_left:
            if state_dict[conv1_key].shape[1] != 6:
                state_dict[conv1_key] = torch.randn(64, 6, 7, 7)
            self.flownet_rgb_left.load_state_dict(state_dict, strict=False)

        if self.use_rgb_right:
            if state_dict[conv1_key].shape[1] != 6:
                state_dict[conv1_key] = torch.randn(64, 6, 7, 7)
            self.flownet_rgb_right.load_state_dict(state_dict, strict=False)

        if self.use_lidar:
            if state_dict[conv1_key].shape[1] != lidar_input_channels * 2:
                state_dict[conv1_key] = torch.randn(64, lidar_input_channels * 2, 7, 7)
            self.flownet_lidar.load_state_dict(state_dict, strict=False)

        # Four-branch architecture for each modality
        self.common_dim = 256
        self.quad_branches_rgb_left = nn.ModuleList([
            QuadBranch(2, self.common_dim) for _ in range(4)
        ]) if self.use_rgb_left else None
        self.quad_branches_rgb_right = nn.ModuleList([
            QuadBranch(2, self.common_dim) for _ in range(4)
        ]) if self.use_rgb_right else None
        self.quad_branches_lidar = nn.ModuleList([
            QuadBranch(2, self.common_dim) for _ in range(4)
        ]) if self.use_lidar else None

        # Fusion dimension after concatenation
        self.fusion_dim = self.common_dim * (int(self.use_rgb_left) + int(self.use_rgb_right) + int(self.use_lidar))

        # LSTM and dense layers for pose prediction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.rnn = nn.LSTM(self.fusion_dim, rnn_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.dense1 = nn.Linear(rnn_hidden_size * 2, 1000)
        self.dense2 = nn.Linear(1000, 512)
        self.fc = nn.Linear(512, 4)  # 3 for translation, 1 for yaw (delta phi)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, rgb_left, rgb_right, lidar_combined):
        batch_size, seq_len, _, h, w = rgb_left.shape if rgb_left is not None else rgb_right.shape if rgb_right is not None else lidar_combined.shape
        num_pairs = batch_size * (seq_len - 1)

        # Process RGB-L (image_02)
        rgb_left_features = None
        if self.use_rgb_left:
            rgb_left_flat = rgb_left.view(batch_size * seq_len, 3, h, w)
            rgb_left_pairs = torch.cat((rgb_left_flat[:-1], rgb_left_flat[1:]), dim=1)
            rgb_left_flow_output = self.flownet_rgb_left(rgb_left_pairs)
            rgb_left_flow = rgb_left_flow_output[0] if isinstance(rgb_left_flow_output, tuple) else rgb_left_flow_output
            if rgb_left_flow.size(0) > num_pairs:
                rgb_left_flow = rgb_left_flow[:num_pairs]
            elif rgb_left_flow.size(0) < num_pairs:
                padding = torch.zeros(num_pairs - rgb_left_flow.size(0), *rgb_left_flow.shape[1:], device=rgb_left_flow.device)
                rgb_left_flow = torch.cat([rgb_left_flow, padding], dim=0)
            
            # Split into four quadrants
            H, W = rgb_left_flow.shape[2], rgb_left_flow.shape[3]
            h_half, w_half = H // 2, W // 2
            rgb_left_q1 = rgb_left_flow[:, :, :h_half, :w_half]
            rgb_left_q2 = rgb_left_flow[:, :, :h_half, w_half:]
            rgb_left_q3 = rgb_left_flow[:, :, h_half:, :w_half]
            rgb_left_q4 = rgb_left_flow[:, :, h_half:, w_half:]
            
            # Process each quadrant
            rgb_left_features = torch.cat([
                self.quad_branches_rgb_left[0](rgb_left_q1),
                self.quad_branches_rgb_left[1](rgb_left_q2),
                self.quad_branches_rgb_left[2](rgb_left_q3),
                self.quad_branches_rgb_left[3](rgb_left_q4)
            ], dim=1)

        # Process RGB-R (image_03)
        rgb_right_features = None
        if self.use_rgb_right:
            rgb_right_flat = rgb_right.view(batch_size * seq_len, 3, h, w)
            rgb_right_pairs = torch.cat((rgb_right_flat[:-1], rgb_right_flat[1:]), dim=1)
            rgb_right_flow_output = self.flownet_rgb_right(rgb_right_pairs)
            rgb_right_flow = rgb_right_flow_output[0] if isinstance(rgb_right_flow_output, tuple) else rgb_right_flow_output
            if rgb_right_flow.size(0) > num_pairs:
                rgb_right_flow = rgb_right_flow[:num_pairs]
            elif rgb_right_flow.size(0) < num_pairs:
                padding = torch.zeros(num_pairs - rgb_right_flow.size(0), *rgb_right_flow.shape[1:], device=rgb_right_flow.device)
                rgb_right_flow = torch.cat([rgb_right_flow, padding], dim=0)
            
            # Split into four quadrants
            H, W = rgb_right_flow.shape[2], rgb_right_flow.shape[3]
            h_half, w_half = H // 2, W // 2
            rgb_right_q1 = rgb_right_flow[:, :, :h_half, :w_half]
            rgb_right_q2 = rgb_right_flow[:, :, :h_half, w_half:]
            rgb_right_q3 = rgb_right_flow[:, :, h_half:, :w_half]
            rgb_right_q4 = rgb_right_flow[:, :, h_half:, w_half:]
            
            # Process each quadrant
            rgb_right_features = torch.cat([
                self.quad_branches_rgb_right[0](rgb_right_q1),
                self.quad_branches_rgb_right[1](rgb_right_q2),
                self.quad_branches_rgb_right[2](rgb_right_q3),
                self.quad_branches_rgb_right[3](rgb_right_q4)
            ], dim=1)

        # Process LiDAR (if enabled)
        lidar_features = None
        if self.use_lidar:
            lidar_combined_flat = lidar_combined.view(batch_size * seq_len, -1, h, w)
            lidar_pairs = torch.cat((lidar_combined_flat[:-1], lidar_combined_flat[1:]), dim=1)
            lidar_flow_output = self.flownet_lidar(lidar_pairs)
            lidar_flow = lidar_flow_output[0] if isinstance(lidar_flow_output, tuple) else lidar_flow_output
            if lidar_flow.size(0) > num_pairs:
                lidar_flow = lidar_flow[:num_pairs]
            elif lidar_flow.size(0) < num_pairs:
                padding = torch.zeros(num_pairs - lidar_flow.size(0), *lidar_flow.shape[1:], device=lidar_flow.device)
                lidar_flow = torch.cat([lidar_flow, padding], dim=0)
            
            # Split into four quadrants
            H, W = lidar_flow.shape[2], lidar_flow.shape[3]
            h_half, w_half = H // 2, W // 2
            lidar_q1 = lidar_flow[:, :, :h_half, :w_half]
            lidar_q2 = lidar_flow[:, :, :h_half, w_half:]
            lidar_q3 = lidar_flow[:, :, h_half:, :w_half]
            lidar_q4 = lidar_flow[:, :, h_half:, w_half:]
            
            # Process each quadrant
            lidar_features = torch.cat([
                self.quad_branches_lidar[0](lidar_q1),
                self.quad_branches_lidar[1](lidar_q2),
                self.quad_branches_lidar[2](lidar_q3),
                self.quad_branches_lidar[3](lidar_q4)
            ], dim=1)

        # Simple fusion via concatenation
        modality_features = []
        if self.use_rgb_left:
            modality_features.append(rgb_left_features)
        if self.use_rgb_right:
            modality_features.append(rgb_right_features)
        if self.use_lidar:
            modality_features.append(lidar_features)
        if not modality_features:
            raise ValueError("No features available for fusion")
        fused = torch.cat(modality_features, dim=1)  # Concatenate along the channel dimension

        # Pool and reshape for LSTM
        fused = self.pool(fused)  # [num_pairs, fusion_dim, 1, 1]
        fused = fused.view(batch_size, seq_len-1, self.fusion_dim)  # [batch_size, seq_len-1, fusion_dim]

        # LSTM and dense layers for pose prediction
        out, _ = self.rnn(fused)
        out = F.relu(self.dense1(out))
        out = F.relu(self.dense2(out))
        out = self.fc(out)  # [batch_size, seq_len-1, 4] (3 for translation, 1 for yaw)

        translation = out[:, :, :3]
        yaw = out[:, :, 3]  # Delta phi (yaw angle in radians)
        # No clamping on translation to preserve true scale
        out = torch.cat([translation, yaw.unsqueeze(-1)], dim=-1)
        return out

class WeightedLoss(nn.Module):
    def __init__(self, device="cpu"):
        super(WeightedLoss, self).__init__()
        self.w_rot = 100.0  # Fixed weighting factor as in DeepVO and DeepAVO

    def forward(self, pred, target, rgb_high=None):
        # Translation loss
        L_t = F.mse_loss(pred[:, :, :3], target[:, :, :3])
        
        # Yaw loss
        yaw_pred = pred[:, :, 3]
        yaw_target = target[:, :, 3]
        # Compute angular difference (handle wrap-around)
        yaw_diff = torch.atan2(torch.sin(yaw_pred - yaw_target), torch.cos(yaw_pred - yaw_target))
        L_yaw = torch.abs(yaw_diff).mean()
        
        # Weighted loss with fixed w_rot
        loss = L_t + self.w_rot * L_yaw
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError("NaN or Inf detected in loss")
        
        return loss, L_t, L_yaw
    
    @staticmethod
    def RMSEError(pred, label):
        return torch.sqrt(torch.mean((pred - label) ** 2))