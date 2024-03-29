import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        # F.bmm 시에 (B, W^2, H^2) 의 매우 큰 매트릭스가 만들어지므로 주의 필요!
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(batch_size, -1, width * height)
        # print(query.shape, key.shape, value.shape)
        # print(attention.shape)
        # print("***")
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma * out + x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, n_head):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.n_head = n_head
        self.in_channels_sub = self.in_channels // n_head
        assert in_channels % n_head == 0
        self.attentions = nn.ModuleList(
            [SelfAttention(self.in_channels_sub) for _ in range(n_head)]
        )

    def forward(self, x):
        # x (B, C, W, H)
        # x[:, 0:in_channels_sub]
        ret = []
        for i, att in enumerate(self.attentions):
            att_sub = att(
                x[:, i * self.in_channels_sub : (i + 1) * self.in_channels_sub]
            )
            ret.append(att_sub)

        ret = torch.concat(ret, dim=1)
        # print(ret.shape)
        # print("----")
        return ret


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upsample=True,
        use_self_attention=False,
        n_head=1,
    ):
        super(ResidualBlock, self).__init__()

        self.upsample = upsample
        self.use_self_attention = use_self_attention
        self.n_head = n_head

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        if self.upsample:
            self.conv2 = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        else:
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv_shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

        if use_self_attention:
            self.self_attention = (
                MultiHeadSelfAttention(out_channels, n_head)
                if self.n_head == 1
                else SelfAttention(out_channels)
            )

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn1(self.conv1(x)))
        if self.use_self_attention:
            x = self.self_attention(x)
        x = self.bn2(self.conv2(x))

        shortcut = self.conv_shortcut(shortcut)
        if self.upsample:
            # Upsampling the shortcut connection to match dimensions
            shortcut = F.interpolate(shortcut, scale_factor=2)

        return x + shortcut


class NewGenerator(nn.Module):
    def __init__(self, z_dim):
        super(NewGenerator, self).__init__()
        self.hdim = 128
        dropout = 0.2
        n_block = 3
        self.factor = 2**n_block
        self.side = 88 // self.factor  # 8월 샘플: 80, full-data: 88
        assert 88 % self.factor == 0

        # Initial dense layer
        self.fc1 = nn.Linear(z_dim, 256)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, self.hdim)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(self.hdim, self.side * self.side * 128)
        self.dropout3 = nn.Dropout(p=dropout)

        # Residual blocks
        # Residual blocks 지날때마다 W, H 각 x2
        # self-attention F.bmm() 메모리 폭발 때문에 G는 W, H 가 작은 초기부분에 어텐션 사용하고 D는 후반 부분에 사용하길 권장
        self.block1 = ResidualBlock(
            128, 64, upsample=True, use_self_attention=True, n_head=4
        )
        self.block2 = ResidualBlock(
            64, 32, upsample=True, use_self_attention=True, n_head=2
        )
        # self.block3 = ResidualBlock(
        #     32, 32, upsample=True, use_self_attention=True, n_head=2
        # )
        self.block4 = ResidualBlock(
            32, 16, upsample=True, use_self_attention=False, n_head=1
        )

        # Final output layer
        # self.conv_out = nn.Conv2d(16, 6, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # Initial dense layer
        x = self.fc1(z.view(z.shape[0], -1))  # 4d -> 2d
        x = self.dropout1(F.relu(x))
        x = self.fc2((x))
        x = self.dropout2(F.relu(x))
        x = self.fc3(x)
        x = self.dropout3(F.relu(x))
        x = x.view(x.size(0), 128, self.side, self.side)

        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        # x = self.block3(x)
        x = self.block4(x)
        # Final output layer
        # return self.conv_out(x)
        return torch.tanh(self.conv_out(x))


class NewDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super(NewDiscriminator, self).__init__()

        # Residual blocks
        # Max Pooling 지날때마다 W, H 각 /2
        # self-attention F.bmm() 메모리 폭발 때문에 G는 W, H 가 작은 초기부분에 어텐션 사용하고 D는 후반 부분에 사용하길 권장
        self.block1 = ResidualBlock(
            in_channel, 64, upsample=False, use_self_attention=False
        )
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.block2 = ResidualBlock(64, 128, upsample=False, use_self_attention=False)
        self.max_pool2 = nn.MaxPool2d(4, 4)
        self.block3 = ResidualBlock(
            128, 256, upsample=False, use_self_attention=True, n_head=1
        )
        self.max_pool3 = nn.MaxPool2d(2, 2)
        self.block4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        # Final output layer
        self.conv_out = nn.AdaptiveAvgPool2d(1)  # (B, 1, 1, 1)

    def forward(self, x):
        # Residual blocks
        x = self.block1(x)
        x = self.max_pool1(x)
        x = self.block2(x)
        x = self.max_pool2(x)
        x = self.block3(x)
        x = self.max_pool3(x)
        x = self.block4(x)  # before the last layer

        # Final output layer
        return self.conv_out(x), x
