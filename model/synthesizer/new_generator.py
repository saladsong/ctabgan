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
        self.attentions = [SelfAttention(self.in_channels_sub) for _ in range(n_head)]

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
            self.self_attention = MultiHeadSelfAttention(out_channels, n_head)

    #             self.self_attention = SelfAttention(out_channels)

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn1(self.conv1(x)))
        if self.use_self_attention:
            x = self.self_attention(x)
        x = self.bn2(self.conv2(x))

        if self.upsample:
            shortcut = self.conv_shortcut(shortcut)
            # Upsampling the shortcut connection to match dimensions
            shortcut = F.interpolate(shortcut, scale_factor=2)

        return x + shortcut


class NewGenerator(nn.Module):
    def __init__(self, z_dim):
        super(NewGenerator, self).__init__()
        self.side = 5

        # Initial dense layer
        self.fc = nn.Linear(z_dim, self.side * self.side * 256)

        # Residual blocks
        # Residual blocks 지날때마다 W, H 각 x2
        self.block1 = ResidualBlock(
            256, 128, upsample=True, use_self_attention=True, n_head=4
        )
        self.block2 = ResidualBlock(
            128, 64, upsample=True, use_self_attention=True, n_head=4
        )
        self.block3 = ResidualBlock(
            64, 32, upsample=True, use_self_attention=True, n_head=2
        )
        self.block4 = ResidualBlock(
            32, 16, upsample=True, use_self_attention=True, n_head=1
        )
        #         self.block1 = ResidualBlock(256, 128, upsample=True, use_self_attention=True,)
        #         self.block2 = ResidualBlock(128, 64, upsample=True, use_self_attention=False,)
        #         self.block3 = ResidualBlock(64, 32, upsample=True, use_self_attention=False,)
        #         self.block4 = ResidualBlock(32, 16, upsample=True, use_self_attention=False,)

        # Final output layer
        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # Initial dense layer
        x = self.fc(z.view(z.shape[0], -1))  # 4d -> 2d
        x = x.view(x.size(0), 256, self.side, self.side)

        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Final output layer
        return torch.tanh(self.conv_out(x))
