import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from flownet_models.FlowNetS import FlowNetS  # Import the class directly
from flownet_models.util import conv  # Import conv function

class FusionLIVO(nn.Module):
    def __init__(self, config, rgb_height=64, rgb_width=900, lidar_height=64, lidar_width=900, rnn_hidden_size=256):
        super(FusionLIVO, self).__init__()
        
        use_depth = config["fusion"]["modalities"]["use_depth"]
        use_intensity = config["fusion"]["modalities"]["use_intensity"]
        use_normals = config["fusion"]["modalities"]["use_normals"]
        use_rgb_low = config["fusion"]["modalities"]["use_rgb_low"]
        input_channels = (3 if use_rgb_low else 0) + (1 if use_depth else 0) + (1 if use_intensity else 0) + (3 if use_normals else 0)
        if input_channels == 0:
            raise ValueError("No modalities selected for FusionLIVO input")
        #print(f"FusionLIVO input channels: {input_channels}")
        
        # Use FlowNetS for both RGB and LiDAR with pretrained weights
        pretrained_path = "/home/kavi/Fusion/flownet_models/pytorch/flownets_bn_EPE2.459.pth"
        self.flownet_rgb = FlowNetS(batchNorm=True)  # 2 RGB frames (3 channels each, total 6)
        self.flownet_lidar = FlowNetS(batchNorm=True)  # 2 LiDAR frames (input_channels * 2)
        
        # Adjust input channels for flownet_lidar
        self.flownet_lidar.conv1 = conv(
            self.flownet_lidar.batchNorm,
            input_channels * 2,
            64,
            kernel_size=7,
            stride=2
        )
        
        # Load pretrained weights
        loaded_data = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        if isinstance(loaded_data, dict) and 'state_dict' in loaded_data:
            state_dict = loaded_data['state_dict']
        else:
            state_dict = loaded_data
        #print("State dict keys:", list(state_dict.keys()))  # Debug print to inspect keys
        
        # Find the conv1 weight key
        conv1_key = 'conv1.0.weight' if 'conv1.0.weight' in state_dict else None
        if conv1_key is None:
            conv1_key = next((key for key in state_dict.keys() if 'conv1' in key.lower()), None)
            if conv1_key is None:
                raise KeyError("Could not find 'conv1' weight in pretrained state_dict. Available keys: " + str(list(state_dict.keys())))
        
        # Check input channels and reinitialize if necessary
        if state_dict[conv1_key].shape[1] != 6:
            state_dict[conv1_key] = torch.randn(64, 6, 7, 7)  # Reinitialize for RGB
        self.flownet_rgb.load_state_dict(state_dict, strict=False)
        if state_dict[conv1_key].shape[1] != input_channels * 2:
            state_dict[conv1_key] = torch.randn(64, input_channels * 2, 7, 7)  # Reinitialize for LiDAR
        self.flownet_lidar.load_state_dict(state_dict, strict=False)
        
        # Simplified FPN for multi-scale fusion
        self.fpn_rgb = nn.ModuleList([
            nn.Conv2d(2, 256, 1),  # FlowNet outputs 2 channels (flow)
            nn.Conv2d(256, 256, 1),
        ])
        self.fpn_rgb_upsample = nn.ModuleList([
            nn.Upsample(size=(16, 225), mode='bilinear', align_corners=False),  # Match rgb_fpn[0] dimensions
            nn.Upsample(size=(lidar_height, lidar_width), mode='bilinear', align_corners=False)
        ])
        
        self.fpn_lidar = nn.ModuleList([
            nn.Conv2d(2, 256, 1),
            nn.Conv2d(256, 256, 1),
        ])
        self.fpn_lidar_upsample = nn.ModuleList([
            nn.Upsample(size=(16, 225), mode='bilinear', align_corners=False),  # Match lidar_fpn[0] dimensions
            nn.Upsample(size=(lidar_height, lidar_width), mode='bilinear', align_corners=False)
        ])
        
        self.fusion_conv = nn.Conv2d(512, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.rnn = nn.LSTM(256, rnn_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden_size * 2, 7)
        
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
    
    def forward(self, rgb_high, lidar_combined):
        batch_size, seq_len, c, h, w = rgb_high.shape
        #print("rgb_high shape:", rgb_high.shape)  # [8, 5, 3, 64, 900]
        #print("lidar_combined shape:", lidar_combined.shape)  # [8, 5, 4, 64, 900]
        
        # Process RGB with FlowNet
        rgb_high_flat = rgb_high.view(batch_size * seq_len, c, h, w)  # Shape: [40, 3, 64, 900]
        rgb_pairs = torch.cat((rgb_high_flat[:-1], rgb_high_flat[1:]), dim=1)  # Shape: [39, 6, 64, 900]
        #print("rgb_pairs shape:", rgb_pairs.shape)
        rgb_flow_output = self.flownet_rgb(rgb_pairs)
        rgb_flow = rgb_flow_output[0] if isinstance(rgb_flow_output, tuple) else rgb_flow_output  # Shape: [39, 2, 16, 225]
        #print("rgb_flow shape:", rgb_flow.shape)
        num_pairs = batch_size * (seq_len - 1)  # 8 * (5-1) = 32
        if rgb_flow.size(0) > num_pairs:
            rgb_flow = rgb_flow[:num_pairs]  # Shape: [32, 2, 16, 225]
        elif rgb_flow.size(0) < num_pairs:
            padding = torch.zeros(num_pairs - rgb_flow.size(0), *rgb_flow.shape[1:], device=rgb_flow.device)
            rgb_flow = torch.cat([rgb_flow, padding], dim=0)
        rgb_flow_flat = rgb_flow  # Shape: [32, 2, 16, 225]
        #print("rgb_flow_flat for FPN:", rgb_flow_flat.shape)
        
        # Process LiDAR with FlowNet
        lidar_combined_flat = lidar_combined.view(batch_size * seq_len, -1, h, w)  # Shape: [40, 4, 64, 900]
        lidar_pairs = torch.cat((lidar_combined_flat[:-1], lidar_combined_flat[1:]), dim=1)  # Shape: [39, 8, 64, 900]
        #print("lidar_pairs shape:", lidar_pairs.shape)
        lidar_flow_output = self.flownet_lidar(lidar_pairs)
        lidar_flow = lidar_flow_output[0] if isinstance(lidar_flow_output, tuple) else lidar_flow_output  # Shape: [39, 2, 16, 225]
        #print("lidar_flow shape:", lidar_flow.shape)
        if lidar_flow.size(0) > num_pairs:
            lidar_flow = lidar_flow[:num_pairs]  # Shape: [32, 2, 16, 225]
        elif lidar_flow.size(0) < num_pairs:
            padding = torch.zeros(num_pairs - lidar_flow.size(0), *lidar_flow.shape[1:], device=lidar_flow.device)
            lidar_flow = torch.cat([lidar_flow, padding], dim=0)
        lidar_flow_flat = lidar_flow  # Shape: [32, 2, 16, 225]
        #print("lidar_flow_flat for FPN:", lidar_flow_flat.shape)
        
        # FPN for RGB
        rgb_features = [rgb_flow_flat]  # Shape: [32, 2, 16, 225]
        rgb_fpn = [self.fpn_rgb[0](rgb_features[0])]  # Shape: [32, 256, 16, 225]
        rgb_fpn.append(self.fpn_rgb[1](rgb_fpn[0]))  # Shape: [32, 256, 16, 225]
        rgb_fpn[0] = rgb_fpn[0] + self.fpn_rgb_upsample[0](rgb_fpn[1])  # Shape: [32, 256, 16, 225]
        rgb_fused = self.fpn_rgb_upsample[1](rgb_fpn[0])  # Shape: [32, 256, 64, 900]
        rgb_fused = rgb_fused.view(batch_size, seq_len-1, rgb_fused.size(1), rgb_fused.size(2), rgb_fused.size(3))  # Shape: [8, 4, 256, 64, 900]
        #print("rgb_fused shape:", rgb_fused.shape)
        
        # FPN for LiDAR
        lidar_features = [lidar_flow_flat]  # Shape: [32, 2, 16, 225]
        lidar_fpn = [self.fpn_lidar[0](lidar_features[0])]  # Shape: [32, 256, 16, 225]
        lidar_fpn.append(self.fpn_lidar[1](lidar_fpn[0]))  # Shape: [32, 256, 16, 225]
        lidar_fpn[0] = lidar_fpn[0] + self.fpn_lidar_upsample[0](lidar_fpn[1])  # Shape: [32, 256, 16, 225]
        lidar_fused = self.fpn_lidar_upsample[1](lidar_fpn[0])  # Shape: [32, 256, 64, 900]
        lidar_fused = lidar_fused.view(batch_size, seq_len-1, lidar_fused.size(1), lidar_fused.size(2), lidar_fused.size(3))  # Shape: [8, 4, 256, 64, 900]
        #print("lidar_fused shape:", lidar_fused.shape)
        
        # Fusion and LSTM
        fused = torch.cat([rgb_fused, lidar_fused], dim=2)  # Shape: [8, 4, 512, 64, 900]
        fused = fused.view(batch_size * (seq_len-1), fused.size(2), fused.size(3), fused.size(4))  # Shape: [32, 512, 64, 900]
        fused = self.fusion_conv(fused)  # Shape: [32, 256, 64, 900]
        fused = fused.view(batch_size, seq_len-1, fused.size(1), fused.size(2), fused.size(3))  # Shape: [8, 4, 256, 64, 900]
        fused = self.pool(fused)  # Shape: [8, 4, 256, 1, 1]
        fused = fused.view(batch_size, seq_len-1, 256)  # Shape: [8, 4, 256]
        out, _ = self.rnn(fused)  # Shape: [8, 4, rnn_hidden_size*2]
        out = self.fc(out)  # Shape: [8, 4, 7]
        max_translation = 10.0
        translation = torch.clamp(out[:, :, :3] / max_translation, -1.0, 1.0) * max_translation
        quaternion = F.normalize(out[:, :, 3:] + 1e-8, p=2, dim=-1)
        out = torch.cat([translation, quaternion], dim=-1)  # Shape: [8, 4, 7]
        return out

class WeightedLoss(nn.Module):
    def __init__(self, w_rot=1.0, learn_hyper_params=True, device="cpu"):
        super(WeightedLoss, self).__init__()
        self.w_rot = w_rot

    def forward(self, pred, target, rgb_high=None):
        L_t = F.mse_loss(pred[:, :, :3], target[:, :, :3])
        q_pred = pred[:, :, 3:]
        q_target = target[:, :, 3:]
        q_pred = F.normalize(q_pred + 1e-8, p=2, dim=-1)
        q_target = F.normalize(q_target + 1e-8, p=2, dim=-1)
        L_r = 1.0 - torch.abs(torch.sum(q_pred * q_target, dim=-1)).mean()
        loss = L_t + L_r * self.w_rot
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError("NaN or Inf detected in loss")
        return loss
    
    @staticmethod
    def RMSEError(pred, label):
        return torch.sqrt(torch.mean((pred - label) ** 2))