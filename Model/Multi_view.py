import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

class SpatialAttention(nn.Module):
    def __init__(self, W, H):
        super().__init__()
        self.fc = nn.Linear(W * H, W * H)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, L, W, H, C = x.shape
        F_s = x.mean(dim=1)  # (B, W, H, C)
        F_s = F_s.view(B, -1)

        A_spa = self.fc(F_s)
        A_spa = self.softmax(A_spa)

        A_spa = A_spa.view(B, 1, W, H, C)
        A_spa = A_spa.expand(B, L, W, H, C)

        x = x * (1 + A_spa) # torch.Size([16, 64, 32, 32, 1])
        return x

class TemporalAttention(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.fc = nn.Linear(L, L)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, L, W, H, C = x.shape

        F_t = x.mean(dim=[2, 3])

        F_t = F_t.view(B, L * C)
        A_tem = self.fc(F_t)
        A_tem = self.softmax(A_tem)
        A_tem = A_tem.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        A_tem = A_tem.expand(-1, -1, W, H, C)

        x = x * (1 + A_tem) # ([16, 64, 32, 32, 1])
        return x

class SpectralAttention(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.fc = nn.Linear(L, L)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, L, W, H, C = x.shape

        F_t = x.mean(dim=[2, 3])

        F_t = F_t.view(B, L * C)
        A_tem = self.fc(F_t)
        A_tem = self.softmax(A_tem)
        A_tem = A_tem.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        A_tem = A_tem.expand(-1, -1, W, H, C)

        x = x * (1 + A_tem)
        return x

# ResNet3D
class ResNet3DWrapper(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.resnet3d = r3d_18(pretrained=True)
        self.resnet3d.stem[0] = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7),
                                          stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.resnet3d.fc = nn.Identity()

    def forward(self, x):
        return self.resnet3d(x)

class SpaceTimeStream(nn.Module):
    def __init__(self, in_channels, L, W, H):
        super().__init__()
        self.spatial_att = SpatialAttention(W, H)
        self.temporal_att = TemporalAttention(L)
        self.resnet3d = ResNet3DWrapper(in_channels)

    def forward(self, x):
        x = self.spatial_att(x)
        x = self.temporal_att(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, L, W, H)
        x = self.resnet3d(x)
        return x

class SpaceFrequencyStream(nn.Module):
    def __init__(self, in_channels, L, W, H):
        super().__init__()
        self.spatial_att = SpatialAttention(W, H)
        self.spectral_att = SpectralAttention(L)
        self.resnet3d = ResNet3DWrapper(in_channels)

    def forward(self, x):
        x = self.spatial_att(x)
        x = self.spectral_att(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.resnet3d(x)
        return x

class MA3D(nn.Module):
    def __init__(self, in_channels, L, W, H, num_classes=2):
        super().__init__()
        self.space_time_stream = SpaceTimeStream(in_channels, L, W, H)
        self.space_freq_stream = SpaceFrequencyStream(in_channels, L, W, H)
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.fc1 = nn.Linear(512*2, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc_out = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x_time, x_freq):
        x_time = self.space_time_stream(x_time)
        x_freq = self.space_freq_stream(x_freq)

        x = torch.cat((x_time, x_freq), dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x