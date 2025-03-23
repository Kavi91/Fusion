import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, batch_norm=True, conv_dropout=(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)):
        super(FeatureExtractor, self).__init__()
        self.batch_norm = batch_norm
        self.conv_dropout = conv_dropout
        self.conv1 = conv(self.batch_norm, in_channels, 64, kernel_size=7, stride=2, dropout=self.conv_dropout[0])
        self.conv2 = conv(self.batch_norm, 64, 128, kernel_size=5, stride=2, dropout=self.conv_dropout[1])
        self.conv3 = conv(self.batch_norm, 128, 256, kernel_size=5, stride=2, dropout=self.conv_dropout[2])
        self.conv3_1 = conv(self.batch_norm, 256, 256, kernel_size=3, stride=1, dropout=self.conv_dropout[3])
        self.conv4 = conv(self.batch_norm, 256, 512, kernel_size=3, stride=2, dropout=self.conv_dropout[4])
        self.conv4_1 = conv(self.batch_norm, 512, 512, kernel_size=3, stride=1, dropout=self.conv_dropout[5])
        self.conv5 = conv(self.batch_norm, 512, 512, kernel_size=3, stride=2, dropout=self.conv_dropout[6])
        self.conv5_1 = conv(self.batch_norm, 512, 512, kernel_size=3, stride=1, dropout=self.conv_dropout[7])
        self.conv6 = conv(self.batch_norm, 512, 1024, kernel_size=3, stride=2, dropout=self.conv_dropout[8])

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

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
        self.batch_norm = True
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)

        # Separate feature extractors for each modality
        self.rgb_left_extractor = FeatureExtractor(in_channels=6, batch_norm=self.batch_norm, conv_dropout=self.conv_dropout) if self.use_rgb_left else None
        self.rgb_right_extractor = FeatureExtractor(in_channels=6, batch_norm=self.batch_norm, conv_dropout=self.conv_dropout) if self.use_rgb_right else None
        self.lidar_extractor = FeatureExtractor(in_channels=lidar_input_channels * 2, batch_norm=self.batch_norm, conv_dropout=self.conv_dropout) if self.use_lidar else None

        # Compute the shape of the CNN output for a single pair
        __tmp = torch.zeros(1, 6, self.rgb_height, self.rgb_width)
        __tmp = self.rgb_left_extractor(__tmp) if self.use_rgb_left else self.rgb_right_extractor(__tmp)
        self.cnn_output_dim_per_pair = int(torch.prod(torch.tensor(__tmp.size()[1:])))  # Exclude batch dimension

        # Fusion dimension per pair
        self.num_modalities = int(self.use_rgb_left) + int(self.use_rgb_right) + int(self.use_lidar)
        self.fusion_dim_per_pair = self.cnn_output_dim_per_pair * self.num_modalities

        # RNN (LSTM)
        self.rnn = nn.LSTM(
            input_size=self.fusion_dim_per_pair,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            dropout=0.5,
            batch_first=True,
            bidirectional=True
        )
        self.rnn_drop_out = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=rnn_hidden_size * 2, out_features=6)  # 6-DoF: 3 translation, 3 rotation (pitch, yaw, roll)

        # Initialization
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
                        n = param.size(0)
                        start, end = n//4, n//2
                        param.data[start:end].fill_(1.)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, rgb_left, rgb_right, lidar_combined):
        batch_size, seq_len, _, h, w = rgb_left.shape if rgb_left is not None else rgb_right.shape if rgb_right is not None else lidar_combined.shape
        num_pairs = batch_size * (seq_len - 1)

        # Process RGB-L (image_02)
        rgb_left_features = None
        if self.use_rgb_left:
            rgb_left = rgb_left.view(batch_size, seq_len, 3, h, w)
            rgb_left_pairs_list = []
            for b in range(batch_size):
                batch_frames = rgb_left[b]  # [seq_len, 3, h, w]
                batch_pairs = torch.cat((batch_frames[:-1], batch_frames[1:]), dim=1)  # [seq_len-1, 6, h, w]
                rgb_left_pairs_list.append(batch_pairs)
            rgb_left_pairs = torch.cat(rgb_left_pairs_list, dim=0)  # [batch_size * (seq_len-1), 6, h, w]
            rgb_left_features = self.rgb_left_extractor(rgb_left_pairs)

        # Process RGB-R (image_03)
        rgb_right_features = None
        if self.use_rgb_right:
            rgb_right = rgb_right.view(batch_size, seq_len, 3, h, w)
            rgb_right_pairs_list = []
            for b in range(batch_size):
                batch_frames = rgb_right[b]
                batch_pairs = torch.cat((batch_frames[:-1], batch_frames[1:]), dim=1)
                rgb_right_pairs_list.append(batch_pairs)
            rgb_right_pairs = torch.cat(rgb_right_pairs_list, dim=0)
            rgb_right_features = self.rgb_right_extractor(rgb_right_pairs)

        # Process LiDAR (if enabled)
        lidar_features = None
        if self.use_lidar:
            lidar_combined = lidar_combined.view(batch_size, seq_len, -1, h, w)
            lidar_pairs_list = []
            for b in range(batch_size):
                batch_frames = lidar_combined[b]
                batch_pairs = torch.cat((batch_frames[:-1], batch_frames[1:]), dim=1)
                lidar_pairs_list.append(batch_pairs)
            lidar_pairs = torch.cat(lidar_pairs_list, dim=0)
            lidar_features = self.lidar_extractor(lidar_pairs)

        # Simple fusion via concatenation
        modality_features = []
        if self.use_rgb_left:
            modality_features.append(rgb_left_features.view(-1, self.cnn_output_dim_per_pair))
        if self.use_rgb_right:
            modality_features.append(rgb_right_features.view(-1, self.cnn_output_dim_per_pair))
        if self.use_lidar:
            modality_features.append(lidar_features.view(-1, self.cnn_output_dim_per_pair))
        if not modality_features:
            raise ValueError("No features available for fusion")
        fused = torch.cat(modality_features, dim=1)  # [num_pairs, fusion_dim_per_pair]

        # Reshape for LSTM
        fused = fused.view(batch_size, seq_len-1, self.fusion_dim_per_pair)  # [batch_size, seq_len-1, fusion_dim_per_pair]

        # LSTM and dense layers for pose prediction
        out, _ = self.rnn(fused)
        out = self.rnn_drop_out(out)
        out = self.linear(out)  # [batch_size, seq_len-1, 6] (3 translation, 3 rotation)

        return out

class WeightedLoss(nn.Module):
    def __init__(self, device="cpu"):
        super(WeightedLoss, self).__init__()
        self.w_trans = 1.0
        self.w_rot = 10.0  # Reduced from 100 to 10

    def forward(self, pred, target, rgb_high=None):
        # Translation loss
        L_t = F.mse_loss(pred[:, :, :3], target[:, :, :3])
        
        # Rotation loss (pitch, yaw, roll)
        L_rot = F.mse_loss(pred[:, :, 3:], target[:, :, 3:])
        
        # Total loss
        loss = self.w_trans * L_t + self.w_rot * L_rot
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError("NaN or Inf detected in loss")
        
        return loss, L_t, L_rot
    
    @staticmethod
    def RMSEError(pred, label):
        return torch.sqrt(torch.mean((pred - label) ** 2))