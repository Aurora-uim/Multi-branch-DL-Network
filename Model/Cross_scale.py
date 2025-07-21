import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, sampling_rate):
        super(MultiScaleConv, self).__init__()

        kernel_size1 = (1, int(sampling_rate * 0.5) - 1) # 63  S：128
        kernel_size2 = (1, (kernel_size1[1] - 1) // 2)  # 31
        kernel_size3 = (1, (kernel_size2[1] - 1) // 2)  # 15

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size1, stride=(1, 1),
                               padding=(0, kernel_size1[1] // 2))
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size2, stride=(1, 1),
                               padding=(0, kernel_size2[1] // 2))
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size3, stride=(1, 1),
                               padding=(0, kernel_size3[1] // 2))

        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))

        self.conv1_bn = nn.BatchNorm2d(out_channels)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        self.conv3_bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(2)  # (batch, channels, time) -> (batch, channels, 1, time)

        x1 = self.relu(self.conv1_bn(self.conv1(x)))
        x2 = self.relu(self.conv2_bn(self.conv2(x)))
        x3 = self.relu(self.conv3_bn(self.conv3(x)))

        s1 = torch.sigmoid(self.conv1x1(x1))
        s2 = torch.sigmoid(self.conv1x1(x2))
        s3 = torch.sigmoid(self.conv1x1(x3))

        # Element-wise Multiplication
        ms1 = x1 * s1
        ms2 = x2 * s2
        ms3 = x3 * s3

        ms1 = ms1.squeeze(2)
        ms2 = ms2.squeeze(2)
        ms3 = ms3.squeeze(2)
        ms1 = ms1.permute(0, 2, 1)  # (batch, time, channels)
        ms2 = ms2.permute(0, 2, 1)
        ms3 = ms3.permute(0, 2, 1)

        return ms1, ms2, ms3

class CrossScaleAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.5):
        super(CrossScaleAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        # Q, K, V
        self.key1 = nn.Linear(emb_size, emb_size)
        self.query1 = nn.Linear(emb_size, emb_size)
        self.value1 = nn.Linear(emb_size, emb_size)

        self.key2 = nn.Linear(emb_size, emb_size)
        self.query2 = nn.Linear(emb_size, emb_size)
        self.value2 = nn.Linear(emb_size, emb_size)

        self.key3 = nn.Linear(emb_size, emb_size)
        self.query3 = nn.Linear(emb_size, emb_size)
        self.value3 = nn.Linear(emb_size, emb_size)

        # Dropout layers for attention scores
        self.att_drop1 = nn.Dropout(dropout)
        self.att_drop2 = nn.Dropout(dropout)
        self.att_drop3 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(0.5)

        self.projection1 = nn.Linear(emb_size, emb_size)
        self.projection2 = nn.Linear(emb_size, emb_size)
        self.projection3 = nn.Linear(emb_size, emb_size)

    def forward(self, ms1, ms2, ms3):
        # （B，T，C）
        query1 = rearrange(self.query1(ms1), "b n (h d) -> b h n d", h=self.num_heads)
        key1 = rearrange(self.key1(ms1), "b n (h d) -> b h n d", h=self.num_heads)
        value1 = rearrange(self.value1(ms1), "b n (h d) -> b h n d", h=self.num_heads)

        query2 = rearrange(self.query2(ms2), "b n (h d) -> b h n d", h=self.num_heads)
        key2 = rearrange(self.key2(ms2), "b n (h d) -> b h n d", h=self.num_heads)
        value2 = rearrange(self.value2(ms2), "b n (h d) -> b h n d", h=self.num_heads)

        query3 = rearrange(self.query3(ms3), "b n (h d) -> b h n d", h=self.num_heads)
        key3 = rearrange(self.key3(ms3), "b n (h d) -> b h n d", h=self.num_heads)
        value3 = rearrange(self.value3(ms3), "b n (h d) -> b h n d", h=self.num_heads)

        # cross-attention
        energy12 = torch.einsum('bhqd, bhkd -> bhqk', query1, key2)
        energy23 = torch.einsum('bhqd, bhkd -> bhqk', query2, key3)
        energy31 = torch.einsum('bhqd, bhkd -> bhqk', query3, key1)

        scaling = self.emb_size ** (1 / 2)

        # Calculate attention scores
        att1 = F.softmax(energy23 / scaling, dim=-1)
        att1 = self.att_drop1(att1)
        out1 = torch.einsum('bhal, bhlv -> bhav ', att1, value3)
        out1 = rearrange(out1, "b h n d -> b n (h d)")
        out1 = self.projection1(out1)

        att2 = F.softmax(energy31 / scaling, dim=-1)
        att2 = self.att_drop2(att2)
        out2 = torch.einsum('bhal, bhlv -> bhav ', att2, value1)
        out2 = rearrange(out2, "b h n d -> b n (h d)")
        out2 = self.projection2(out2)

        att3 = F.softmax(energy12 / scaling, dim=-1)
        att3 = self.att_drop3(att3)
        out3 = torch.einsum('bhal, bhlv -> bhav ', att3, value2)
        out3 = rearrange(out3, "b h n d -> b n (h d)")
        out3 = self.projection3(out3)

        return torch.cat((out1, out2, out3), dim=2)

class FeedForward(nn.Module):
    def __init__(self, emb_size, dropout=0.5):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(emb_size, emb_size * 2)
        self.fc2 = nn.Linear(emb_size * 2, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x + residual

class CSA(nn.Module):
    def __init__(self, in_channels, emb_size, num_classes, num_heads):
        super(CSA, self).__init__()

        self.ms_conv = MultiScaleConv(in_channels, emb_size, 128)

        self.cs_attention = CrossScaleAttention(emb_size, num_heads, 0.5)

        self.ffn = FeedForward(emb_size * 3, 0.5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(emb_size * 3, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        ms1, ms2, ms3 = self.ms_conv(x)

        att_output = self.cs_attention(ms1, ms2, ms3)

        x = self.ffn(att_output)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.fc_out(x)
        return x