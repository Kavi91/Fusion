import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np

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

class DeepVO(nn.Module):
    def __init__(self, imsize1, imsize2, batchNorm=True, conv_dropout=None, rnn_hidden_size=1000, 
                 rnn_dropout_out=0.5, rnn_dropout_between=0, clip=None):
        super(DeepVO, self).__init__()
        self.batchNorm = batchNorm
        self.clip = clip
        self.conv_dropout = conv_dropout if conv_dropout else [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5]
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2, dropout=self.conv_dropout[0])
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2, dropout=self.conv_dropout[1])
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2, dropout=self.conv_dropout[2])
        self.conv3_1 = conv(self.batchNorm, 256, 256, kernel_size=3, stride=1, dropout=self.conv_dropout[3])
        self.conv4 = conv(self.batchNorm, 256, 512, kernel_size=3, stride=2, dropout=self.conv_dropout[4])
        self.conv4_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=self.conv_dropout[5])
        self.conv5 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=2, dropout=self.conv_dropout[6])
        self.conv5_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=self.conv_dropout[7])
        self.conv6 = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=self.conv_dropout[8])

        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self.encode_image(__tmp)

        self.rnn = nn.LSTM(
            input_size=int(np.prod(__tmp.size())),
            hidden_size=rnn_hidden_size,
            num_layers=2,
            dropout=rnn_dropout_between,
            batch_first=True
        )
        self.rnn_drop_out = nn.Dropout(rnn_dropout_out)
        self.linear = nn.Linear(in_features=rnn_hidden_size, out_features=6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                kaiming_normal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)
                kaiming_normal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = torch.cat((x[:, :-1], x[:, 1:]), dim=2)
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)
        out, hc = self.rnn(x)
        out = self.rnn_drop_out(out)
        out = self.linear(out)
        return out

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def get_loss(self, x, y):
        predicted = self.forward(x)
        y = y[:, 1:, :]
        angle_loss = torch.nn.functional.mse_loss(predicted[:,:,:3], y[:,:,:3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:,:,3:], y[:,:,3:])
        loss = (100 * angle_loss + translation_loss)
        return loss

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), self.clip)
        optimizer.step()
        return loss

class LoRCoNLO(nn.Module):
    def __init__(self, batch_size, batchNorm=True):
        super(LoRCoNLO, self).__init__()
        self.batch_size = batch_size
        
        self.simple_conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, stride=(1, 2), padding=(1, 0))
        self.simple_conv2 = nn.Conv2d(32, 64, 3, (1, 2), (1, 0))
        self.simple_conv3 = nn.Conv2d(64, 128, 3, (1, 2), (1, 0))
        self.simple_conv4 = nn.Conv2d(128, 256, 3, (2, 2), (1, 0))
        self.simple_conv5 = nn.Conv2d(256, 512, 3, (2, 2), (1, 0))
        self.simple_conv6 = nn.Conv2d(512, 128, 1, 1, (1, 0))
        
        self.rnn = nn.LSTM(
            input_size=128 * 306,
            hidden_size=1024,
            num_layers=4,
            dropout=0,
            batch_first=True,
            bidirectional=True
        )
        self.rnn_drop_out = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(2048, 6)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0))
        
        self.conv_bn1 = nn.BatchNorm2d(32)
        self.conv_bn2 = nn.BatchNorm2d(64)
        self.conv_bn3 = nn.BatchNorm2d(128)
        self.conv_bn4 = nn.BatchNorm2d(256)
        self.conv_bn5 = nn.BatchNorm2d(512)
        self.conv_bn6 = nn.BatchNorm2d(128)

    def forward(self, x):
        batch_size = x.size(0)
        rnn_size = x.size(1)
        
        x = x.view(batch_size * rnn_size, x.size(2), x.size(3), x.size(4))
        
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular')
        x = self.maxpool(x)
        
        x = self.encode_image(x)
        
        x = x.view(batch_size, rnn_size, -1)
        
        x, hc = self.rnn(x)

        x = self.rnn_drop_out(x)
        
        x = x.reshape(batch_size * rnn_size, -1)
        
        output = self.fc_part(x)
        
        output = output.reshape(batch_size, rnn_size, -1)

        return output
    
    def encode_image(self, x):
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular')
        x = self.simple_conv1(x)
        x = self.conv_bn1(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular')
        x = self.simple_conv2(x)
        x = self.conv_bn2(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular')
        x = self.simple_conv3(x)
        x = self.conv_bn3(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular')
        x = self.simple_conv4(x)
        x = self.conv_bn4(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular')
        x = self.simple_conv5(x)
        x = self.conv_bn5(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular')
        x = self.simple_conv6(x)
        x = self.conv_bn6(x)
        x = F.leaky_relu(x, 0.1)
        return x
    
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

class FusionLIVO(nn.Module):
    def __init__(self, config, rgb_height=256, rgb_width=832, lidar_height=64, lidar_width=900, rnn_hidden_size=256):
        super(FusionLIVO, self).__init__()
        
        use_depth = config["fusion"]["modalities"]["use_depth"]
        use_intensity = config["fusion"]["modalities"]["use_intensity"]
        use_normals = config["fusion"]["modalities"]["use_normals"]
        use_rgb_low = config["fusion"]["modalities"]["use_rgb_low"]
        input_channels = (3 if use_rgb_low else 0) + (1 if use_depth else 0) + (1 if use_intensity else 0) + (3 if use_normals else 0)
        if input_channels == 0:
            raise ValueError("No modalities selected for FusionLIVO input")
        print(f"FusionLIVO input channels: {input_channels} (Depth: {use_depth}, Intensity: {use_intensity}, Normals: {use_normals}, RGB Low: {use_rgb_low})")
        
        self.deepvo = DeepVO(rgb_height, rgb_width, batchNorm=True)
        self.deepvo.conv1 = conv(self.deepvo.batchNorm, 3, 64, kernel_size=7, stride=2, dropout=self.deepvo.conv_dropout[0])
        self.lorconlo = LoRCoNLO(batch_size=config["fusion"]["batch_size"], batchNorm=False)
        self.lorconlo.simple_conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=(1, 2), padding=(1, 0))
        
        # Add dropout to conv layers
        self.deepvo.conv_dropout = [config["fusion"]["dropout"]] * len(self.deepvo.conv_dropout)
        self.lorconlo.conv_dropout = [config["fusion"]["dropout"]] * 6
        
        self.fpn_rgb = nn.ModuleList([
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(1024, 256, 1)
        ])
        self.fpn_rgb_upsample = nn.ModuleList([
            nn.Upsample(size=(16, 52), mode='bilinear', align_corners=False),
            nn.Upsample(size=(32, 104), mode='bilinear', align_corners=False),
            nn.Upsample(size=(lidar_height, lidar_width), mode='bilinear', align_corners=False)
        ])
        
        self.fpn_lidar = nn.ModuleList([
            nn.Conv2d(128, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(128, 256, 1)
        ])
        self.fpn_lidar_upsample = nn.ModuleList([
            nn.Upsample(size=(64, 111), mode='bilinear', align_corners=False),
            nn.Upsample(size=(32, 55), mode='bilinear', align_corners=False),
            nn.Upsample(size=(lidar_height, lidar_width), mode='bilinear', align_corners=False)
        ])
        
        self.fusion_conv = nn.Conv2d(512, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.rnn = nn.LSTM(256, rnn_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden_size * 2, 7)  # 3 for translation, 4 for quaternion
        
        # Initialize weights to smaller values to reduce numerical instability
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
        rgb_high = rgb_high.view(batch_size * seq_len, c, h, w)
        
        conv1_out = self.deepvo.conv1(rgb_high)
        conv2_out = self.deepvo.conv2(conv1_out)
        conv3_out = self.deepvo.conv3(conv2_out)
        conv3_1_out = self.deepvo.conv3_1(conv3_out)
        conv4_out = self.deepvo.conv4(conv3_1_out)
        conv4_1_out = self.deepvo.conv4_1(conv4_out)
        conv5_out = self.deepvo.conv5(conv4_1_out)
        conv5_1_out = self.deepvo.conv5_1(conv5_out)
        conv6_out = self.deepvo.conv6(conv5_1_out)
        if torch.isnan(conv6_out).any() or torch.isinf(conv6_out).any():
            raise ValueError("NaN or Inf detected in conv6_out")
        
        rgb_features = [conv3_1_out, conv4_1_out, conv6_out]
        
        rgb_fpn = [self.fpn_rgb[i](rgb_features[i]) for i in range(3)]
        rgb_fpn[1] = rgb_fpn[1] + self.fpn_rgb_upsample[0](rgb_fpn[2])
        if torch.isnan(rgb_fpn[1]).any() or torch.isinf(rgb_fpn[1]).any():
            raise ValueError("NaN or Inf detected in rgb_fpn[1]")
        rgb_fpn[0] = rgb_fpn[0] + self.fpn_rgb_upsample[1](rgb_fpn[1])
        if torch.isnan(rgb_fpn[0]).any() or torch.isinf(rgb_fpn[0]).any():
            raise ValueError("NaN or Inf detected in rgb_fpn[0]")
        rgb_fused = self.fpn_rgb_upsample[2](rgb_fpn[0])
        if torch.isnan(rgb_fused).any() or torch.isinf(rgb_fused).any():
            raise ValueError("NaN or Inf detected in rgb_fused")
        rgb_fused = rgb_fused.view(batch_size, seq_len, rgb_fused.size(1), rgb_fused.size(2), rgb_fused.size(3))
        
        batch_size, seq_len, c_lidar, h_lidar, w_lidar = lidar_combined.shape
        lidar_combined = lidar_combined.view(batch_size * seq_len, c_lidar, h_lidar, w_lidar)
        
        lidar_conv1 = self.lorconlo.simple_conv1(lidar_combined)
        lidar_conv2 = self.lorconlo.simple_conv2(self.lorconlo.conv_bn1(lidar_conv1))
        lidar_conv3 = self.lorconlo.simple_conv3(self.lorconlo.conv_bn2(lidar_conv2))
        lidar_conv4 = self.lorconlo.simple_conv4(self.lorconlo.conv_bn3(lidar_conv3))
        lidar_conv5 = self.lorconlo.simple_conv5(self.lorconlo.conv_bn4(lidar_conv4))
        lidar_conv6 = self.lorconlo.simple_conv6(self.lorconlo.conv_bn5(lidar_conv5))
        if torch.isnan(lidar_conv6).any() or torch.isinf(lidar_conv6).any():
            raise ValueError("NaN or Inf detected in lidar_conv6")
        
        lidar_features = [lidar_conv3, lidar_conv4, lidar_conv6]
        
        lidar_fpn = [self.fpn_lidar[i](lidar_features[i]) for i in range(3)]
        lidar_fpn[1] = lidar_fpn[1] + self.fpn_lidar_upsample[1](lidar_fpn[2])
        if torch.isnan(lidar_fpn[1]).any() or torch.isinf(lidar_fpn[1]).any():
            raise ValueError("NaN or Inf detected in lidar_fpn[1]")
        lidar_fpn[0] = lidar_fpn[0] + self.fpn_lidar_upsample[0](lidar_fpn[1])
        if torch.isnan(lidar_fpn[0]).any() or torch.isinf(lidar_fpn[0]).any():
            raise ValueError("NaN or Inf detected in lidar_fpn[0]")
        lidar_fused = self.fpn_lidar_upsample[2](lidar_fpn[0])
        if torch.isnan(lidar_fused).any() or torch.isinf(lidar_fused).any():
            raise ValueError("NaN or Inf detected in lidar_fused")
        lidar_fused = lidar_fused.view(batch_size, seq_len, lidar_fused.size(1), lidar_fused.size(2), lidar_fused.size(3))
        
        fused = torch.cat([rgb_fused, lidar_fused], dim=2)
        fused = fused.view(batch_size * seq_len, fused.size(2), fused.size(3), fused.size(4))
        fused = self.fusion_conv(fused)
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            raise ValueError("NaN or Inf detected in fused after fusion_conv")
        fused = fused.view(batch_size, seq_len, fused.size(1), fused.size(2), fused.size(3))
        
        fused = self.pool(fused)
        fused = fused.view(batch_size, seq_len, 256)
        
        out, _ = self.rnn(fused)
        out = self.fc(out)
        max_translation = 10.0
        translation = torch.clamp(out[:, :, :3] / max_translation, -1.0, 1.0) * max_translation
        quaternion = F.normalize(out[:, :, 3:] + 1e-8, p=2, dim=-1)
        out = torch.cat([translation, quaternion], dim=-1)
        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError("NaN or Inf detected in model output")
        return out

class WeightedLoss(nn.Module):
    def __init__(self, w_rot=1.0, learn_hyper_params=True, device="cpu"):
        super(WeightedLoss, self).__init__()
        self.w_rot = w_rot

    def forward(self, pred, target):
        L_t = F.mse_loss(pred[:, :, :3], target[:, :, :3])
        q_pred = pred[:, :, 3:]
        q_target = target[:, :, 3:]
        q_pred = F.normalize(q_pred + 1e-8, p=2, dim=-1)
        q_target = F.normalize(q_target + 1e-8, p=2, dim=-1)
        if torch.isnan(q_pred).any() or torch.isinf(q_pred).any():
            raise ValueError("NaN or Inf detected in q_pred")
        if torch.isnan(q_target).any() or torch.isinf(q_target).any():
            raise ValueError("NaN or Inf detected in q_target")
        L_r = 1.0 - torch.abs(torch.sum(q_pred * q_target, dim=-1)).mean()
        loss = L_t + L_r * self.w_rot
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError("NaN or Inf detected in loss")
        return loss
    
    @staticmethod
    def RMSEError(pred, label):
        return torch.sqrt(torch.mean((pred - label) ** 2))