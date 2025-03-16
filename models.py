import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np

# DeepVO Helper Function and Model (from deepvo/model.py)
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
        y = y[:, 1:, :]  # (batch, seq, dim_pose)
        angle_loss = torch.nn.functional.mse_loss(predicted[:,:,:3], y[:,:,:3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:,:,3:], y[:,:,3:])
        loss = (100 * angle_loss + translation_loss)
        return loss

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), self.clip)  # Updated to clip_grad_norm_
        optimizer.step()
        return loss

# LoRCoN-LO Model and Loss (from standalone lorcon_lo/model.py)
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
    
    def fc_part(self, x):
        x = F.leaky_relu(x, 0.2)
        x = self.fc1(x)
        return x
    
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

class WeightedLoss(nn.Module):
    def __init__(self, learn_hyper_params=True, device="cpu"):
        super(WeightedLoss, self).__init__()
        self.w_rot = 100

    def forward(self, pred, target):
        L_t = F.mse_loss(pred[:, :, :3], target[:, :, :3])
        L_r = F.mse_loss(pred[:, :, 3:], target[:, :, 3:])
        loss = L_t + L_r * self.w_rot
        return loss
    
    @staticmethod
    def RMSEError(pred, label):
        return torch.sqrt(torch.mean((pred - label) ** 2))