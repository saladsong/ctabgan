import torch
from torch.nn import (
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Tanh,
    Sequential,
    Conv2d,
    ConvTranspose2d,
    Parameter,
    LayerNorm,
    BatchNorm2d
)
import torch.nn.functional as F
from typing import List


class SelfAttention(Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = Conv2d(in_channels, in_channels // 4, 1)
        self.key = Conv2d(in_channels, in_channels // 4, 1)
        self.value = Conv2d(in_channels, in_channels, 1)
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        return self.gamma * out + x


# class MultiHeadSelfAttention(Module):
#     def __init__(self, in_channels, n_head):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.in_channels = in_channels
#         self.n_head = n_head
#         self.in_channels_sub = self.in_channels // n_head
#         assert in_channels % n_head == 0
#         self.attentions = ModuleList(
#             [SelfAttention(self.in_channels_sub) for _ in range(n_head)]
#         )

#     def forward(self, x):
#         # x (B, C, W, H)
#         # x[:, 0:in_channels_sub]
#         ret = []
#         for i, att in enumerate(self.attentions):
#             att_sub = att(
#                 x[:, i * self.in_channels_sub : (i + 1) * self.in_channels_sub]
#             )
#             ret.append(att_sub)

#         ret = torch.concat(ret, dim=1)
#         # print(ret.shape)
#         # print("----")
#         return ret


# class ResidualBlock(Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         upsample=True,
#         use_self_attention=False,
#         n_head=1,
#     ):
#         super(ResidualBlock, self).__init__()

#         self.upsample = upsample
#         self.use_self_attention = use_self_attention
#         self.n_head = n_head

#         self.conv1 = Conv2d(
#             in_channels, out_channels, kernel_size=3, stride=1, padding=1
#         )
#         self.bn1 = BatchNorm2d(out_channels)

#         if self.upsample:
#             self.conv2 = ConvTranspose2d(
#                 out_channels, out_channels, kernel_size=4, stride=2, padding=1
#             )
#         else:
#             self.conv2 = Conv2d(
#                 out_channels, out_channels, kernel_size=3, stride=1, padding=1
#             )
#         self.bn2 = BatchNorm2d(out_channels)

#         self.conv_shortcut = Conv2d(
#             in_channels, out_channels, kernel_size=1, stride=1, padding=0
#         )

#         if use_self_attention:
#             self.self_attention = SelfAttention(out_channels, n_head)  # originally MultiHeadAttn

#     #             self.self_attention = SelfAttention(out_channels)

#     def forward(self, x):
#         shortcut = x
#         x = F.relu(self.bn1(self.conv1(x)))
#         if self.use_self_attention:
#             x = self.self_attention(x)
#         x = self.bn2(self.conv2(x))

#         if self.upsample:
#             shortcut = self.conv_shortcut(shortcut)
#             # Upsampling the shortcut connection to match dimensions
#             shortcut = F.interpolate(shortcut, scale_factor=2)

#         return x + shortcut


# class NeoGenerator(Module):
#     def __init__(self, z_dim):
#         super(NeoGenerator, self).__init__()
#         self.side = 5

#         # Initial dense layer
#         self.fc = Linear(z_dim, self.side * self.side * 256)

#         # Residual blocks
#         # Residual blocks 지날때마다 W, H 각 x2
#         self.block1 = ResidualBlock(
#             256, 128, upsample=True, use_self_attention=True, n_head=4
#         )
#         self.block2 = ResidualBlock(
#             128, 64, upsample=True, use_self_attention=True, n_head=4
#         )
#         self.block3 = ResidualBlock(
#             64, 32, upsample=True, use_self_attention=True, n_head=2
#         )
#         self.block4 = ResidualBlock(
#             32, 16, upsample=True, use_self_attention=True, n_head=1
#         )

#         # Final output layer
#         self.conv_out = Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
#         # self.conv_out = Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

#     def forward(self, z):
#         # Initial dense layer
#         x = self.fc(z.view(z.shape[0], -1))  # 4d -> 2d
#         x = x.view(x.size(0), 256, self.side, self.side)

#         # Residual blocks
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)

#         # Final output layer
#         # return torch.tanh(self.conv_out(x))
#         return self.conv_out(x)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class NeoGenerator(Module):
    def __init__(
        self,
        side: int,
        random_dim: int,
        num_channels: int,
        output_channel: int,
        n_conv_layers: int = 4,
    ):
        """build generator

        Args:
            side (int): 정사각 2d로 변환한 피처의 변의 길이
            random_dim (int): 입력 디멘전, (랜덤 분포 차원 + contional vector) 차원
            num_channels (int): CNN 출력 직전 채널 수
            output_channel (int): 마지막 출력 채널 크기
            n_conv_layers (int): conv layer 개수
        """
        super(NeoGenerator, self).__init__()
        self.side = side
        layers_G1, layers_G2, layers_G3 = self._determine_layers_gen(
            side, random_dim, num_channels, output_channel, n_conv_layers
        )
        self.seq1 = Sequential(*layers_G1)
        self.seq2 = Sequential(*layers_G2)
        self.last = Sequential(*layers_G3)

        self.attn1 = SelfAttention(256)
        self.attn2 = SelfAttention(128)

    def forward(self, input_):
        out = self.seq1(input_)
        out = self.attn1(out)
        out = self.seq2(out)
        out = self.attn2(out)

        return self.last(out)

    def _determine_layers_gen(
        self,
        side: int,
        random_dim: int,
        num_channels: int,
        output_channel: int,
        n_conv_layers: int = 4,
    ) -> List[Module]:
        """GAN generator를 구성할 torch.nn 레이어들 생성

        Args:
            side (int): 맨 처음 CNN 레이어의 정사각 이미지의 한 면 크기
            random_dim (int): 입력 디멘전, (랜덤 분포 차원 + contional vector) 차원
            num_channels (int): CNN 출력 직전 채널 수
            output_channel (int): 마지막 출력 채널 크기
            n_conv_layers (int): conv layer 개수
        Return:
            List[Module]: torch.nn 레이어 리스트
        """

        layer_dims = [
            (output_channel, side),
            (num_channels, side // 2),
        ]  # (channel, side)

        while (
            layer_dims[-1][1] > (n_conv_layers - 1) and len(layer_dims) < n_conv_layers
        ):
            layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))  # #채널 *2, side //2

        # layerNorms = []

        # num_c = num_channels * (2 ** (len(layer_dims) - 2))
        # num_s = int(side / (2 ** (len(layer_dims) - 1)))
        # for _ in range(len(layer_dims) - 1):
        #     layerNorms.append([int(num_c), int(num_s), int(num_s)])
        #     num_c = num_c / 2
        #     num_s = num_s * 2

        layers_G1 = [
            SpectralNorm(
                ConvTranspose2d(
                    random_dim,
                    layer_dims[-1][0],
                    layer_dims[-1][1],
                    1,
                    0,
                    output_padding=0,
                    bias=False)
            ),
            BatchNorm2d(layer_dims[-1][0]),
            ReLU()
        ]

        # get 1st nn.seq
        for prev, curr in zip(
            reversed(layer_dims[3:]), reversed(layer_dims[2:-1])
        ):
            layers_G1 += [
                SpectralNorm(
                    ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)
                ),
                BatchNorm2d(curr[0]),
                ReLU(),
            ]

        # get 2nd nn.seq
        layers_G2 = [
            SpectralNorm(
                ConvTranspose2d(layer_dims[2][0], layer_dims[1][0], 4, 2, 1, output_padding=0, bias=False)
            ),
            BatchNorm2d(layer_dims[-4][0]),
            ReLU()
        ]

        # get 3rd(last) nn.seq
        layers_G3 = [
            ConvTranspose2d(layer_dims[1][0], layer_dims[0][0], 4, 2, 1, output_padding=0, bias=False),
            Tanh()
        ]

        return layers_G1, layers_G2, layers_G3


class NeoDiscriminator(Module):
    def __init__(self, side: int, num_channels: int, input_channel: int):
        """build discriminator

        Args:
            side (int): 정사각 2d로 변환한 피처의 변의 길이
            num_channels (int): CNN 입력 직후 채널 수
            input_channel (int): 입력 채널 크기
        """
        super(NeoDiscriminator, self).__init__()
        self.side = side
        layers_D1, layers_D2, layers_D3 = self._determine_layers_disc(
            side, num_channels, input_channel)

        self.seq1 = Sequential(*layers_D1)
        self.seq2 = Sequential(*layers_D2)
        self.last = Sequential(*layers_D3)

        self.attn1 = SelfAttention(256)
        self.attn2 = SelfAttention(512)

        # info = len(layers) - 1  # 맨 마지막 conv, relu 제외
        # # info = len(layers) - 2  # 맨 마지막 conv, relu 제외
        # self.seq = Sequential(*layers)
        # # 이부분 수정 필요
        # self.seq_info = Sequential(*layers[:info])

    def forward(self, input):
        out = self.seq1(input)
        out = self.attn1(out)
        out = self.seq2(out)
        out_info = self.attn2(out)
        out_fin = self.last(out_info)
        return (out_fin.squeeze()), out_info

    def _determine_layers_disc(
        self, side: int, num_channels: int, input_channel: int
    ) -> List[Module]:
        """GAN discriminator를 구성할 torch.nn 레이어들 생성

        Args:
            side (int): 맨 처음 CNN 레이어의 정사각 이미지의 한 면 크기
            num_channels (int): CNN 입력 직후 채널 수
            input_channel (int): 입력 채널 크기
        Return:
            List[Module]: torch.nn 레이어 리스트
        """

        layer_dims = [
            (input_channel, side),
            (num_channels, side // 2),
        ]  # (channel, side)

        while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
            layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))  # #채널은 *2 & side는 //2 

        # layerNorms = []
        # num_c = num_channels
        # num_s = side / 2
        # for _ in range(len(layer_dims) - 1):
        #     layerNorms.append([int(num_c), int(num_s), int(num_s)])
        #     num_c = num_c * 2
        #     num_s = num_s / 2

        # get 1st nn.seq
        layers_D1 = []
        for prev, curr in zip(layer_dims[:-2], layer_dims[1:-1]):
            layers_D1 += [
                SpectralNorm(Conv2d(prev[0], curr[0], 4, 2, 1, bias=False)),
                LeakyReLU(0.2, inplace=True),
            ]
        
        # get 2nd nn.seq
        layers_D2 = [
            SpectralNorm(Conv2d(layer_dims[-2][0], layer_dims[-1][0], 4, 2, 1, bias=False)),
            LeakyReLU(0.2, inplace=True),
        ]

        # get 3rd(last) nn.seq
        layers_D3 = [
            Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
            # AdaptiveAvgPool2d(1),
        ]
        # layers_D += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), ReLU(True)]

        return layers_D1, layers_D2, layers_D3