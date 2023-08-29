import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    Conv2d,
    ConvTranspose2d,
    AdaptiveAvgPool2d,
    Sigmoid,
    init,
    BCELoss,
    CrossEntropyLoss,
    SmoothL1Loss,
    LayerNorm,
)

# from model.privacy_utils.rdp_accountant import compute_rdp, get_privacy_spent
from model.synthesizer.transformer import ImageTransformer, DataTransformer
from tqdm.auto import tqdm
import logging
import wandb
import os
from typing import List, Tuple, Union


class Classifier(Module):
    """auxiliary classifier
    Simple MLP
    """

    def __init__(self, input_dim, dis_dims, tcol_idx_st_ed_tuple):
        super(Classifier, self).__init__()
        # Calculate the input dimension after excluding the range of tcol_idx_st_ed_tuple
        st, end = tcol_idx_st_ed_tuple
        self.st_end = tcol_idx_st_ed_tuple
        dim = input_dim - (end - st)  # dims for feature(X) only
        seq = []
        # Building the sequential model layers
        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        # Deciding the final layer based on the range of tcol_idx_st_ed_tuple
        if (end - st) == 1:  # target 컬럼이 continuous (reg.)
            seq += [Linear(dim, 1)]

        elif (end - st) == 2:  # target 컬럼이 categorical (binary clf.)
            seq += [Linear(dim, 1), Sigmoid()]
        else:  # target 컬럼이 categorical (multi class clf.)
            seq += [Linear(dim, (end - st))]

        self.seq = Sequential(*seq)

    def forward(self, input):
        label = None
        st, end = self.st_end
        # Extracting label based on the range of st_end
        if (end - st) == 1:
            label = input[:, st:end]
        else:
            label = torch.argmax(input[:, st:end], axis=-1)  # 뒤의것을 1로 간주

        # Concatenate the input by excluding the range of str_end
        new_imp = torch.cat((input[:, :st], input[:, end:]), 1)

        # Return the model's output and the label
        if ((end - st) == 2) | ((end - st) == 1):
            return self.seq(new_imp).view(-1), label
        else:
            return self.seq(new_imp), label


def apply_activate(data: torch.Tensor, output_info: list) -> torch.Tensor:
    """Apply activation functions to data based on output_info
    CNN 통과시 정사각 모양 만드느라 zero-padding 넣어준 것도 여기서 잘라냄
    Args:
        data: (B, M, gside^2) shape tensor
        output_info: DataTransformer.output_info
    Returns:
        torch.Tensor: (B, M, #encode)
    """
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == "tanh":
            end = st + item[0]
            # data_t.append(torch.tanh(data[:, st:end]))
            data_t.append(torch.tanh(data[:, :, st:end]))
            st = end
        elif item[1] == "softmax":
            end = st + item[0]
            # data_t.append(F.gumbel_softmax(data[:, st:end], tau=0.2))
            data_t.append(F.gumbel_softmax(data[:, :, st:end], tau=0.2))
            st = end
    # return torch.cat(data_t, dim=1)
    return torch.cat(data_t, dim=2)


def get_tcol_idx_st_ed_tuple(
    target_col_index: int, output_info: List[tuple]
) -> Tuple[int, int]:
    """classifier 의 타겟 컬럼 인코딩 뒤의 (start_idx, end_idx) 찾기 (df transform 으로 인코딩 후 idx 가 변화)

    Args:
        target_col_index (int): 인코딩전 타겟 컬럼 인덱스
        output_info (List[tuple]): 컬럼 변환 정보. [(len(alpha_i), activaton_fn, col_name, GT_indicator), (len(beta_i), activaton_fn, col_name) ...]
    Return:
        Tuple[int, int]: 타겟 컬럼 인코딩 뒤의 (start_idx, end_idx). msn의 경우 뒤의 beta 부분 제외하고 alpha 부분만 리턴, auxiliary classifier 에서 (end-st) 개수로 clf/reg 판단하므로
    """
    # Retrieve start and end indices for a target column
    st = 0
    col_cnt = 0  # 확인한 original 컬럼 수
    tc = 0  # 확인한 transformed 컬럼 수 (alpha_i, beta_i, gamma_i, etc.)

    for item in output_info:
        if col_cnt == target_col_index:
            break
        if item[1] == "tanh":
            st += item[0]
            if item[3] == "gt":
                col_cnt += 1
        elif item[1] == "softmax":
            st += item[0]
            col_cnt += 1
        tc += 1

    end = st + output_info[tc][0]

    return (st, end)


def random_choice_prob_index_sampling(
    probs: List[np.ndarray], col_idx: np.ndarray
) -> np.ndarray:
    """Sample beta||gamma mode indices based on given probabilities
    학습 후 샘플링 용도로 사용
    Args:
        probs: ( n_col, n_beta||n_gamma ). 학습 후 샘플링 용도로 쓰기 위해 컬럼별 모드 확률 log 취하지 않고 원본 저장한것
        col_idx: (n_batch). 배치 크기의 col 인덱스 배열
    Returns:
        (n_batch,) 의 선택된 모드 인덱스
    """
    option_list = []
    for i in col_idx:
        pp = probs[i]
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))

    return np.array(option_list)  # .reshape(col_idx.shape)


def random_choice_prob_index(probs: np.ndarray, axis=1) -> np.ndarray:
    """Sample beta||gamma mode indices based on given probabilities
    학습 중 샘플링 용도로 사용
    Args:
        probs: (n_col, maximum_interval(n_beta||n_gamma)), 학습 중 샘플링 용도로 쓰기 위해 컬럼별 모드 확률 log 취해 저장한것
    Returns:
        (n_cols,) 의 선택된 모드 인덱스
    """
    # 모드 선택 위한 주사위 굴리기
    r = np.expand_dims(np.random.rand(probs.shape[1 - axis]), axis=axis)
    # 각 모드별 확률 누적 합 계산해서 위의 'r'값이 경계에 있는 모드 선택
    return (probs.cumsum(axis=axis) > r).argmax(axis=axis)


def maximum_interval(output_info):
    # Get the maximum interval from output_info
    max_interval = 0
    for item in output_info:
        max_interval = max(max_interval, item[0])
    return max_interval


class Cond(object):
    def __init__(self, data: np.ndarray, output_info: list):
        """conditional vector 생성기, Training by sampling 기법을 수행
        Args:
            data:
                orig (66590, 6097)
                reshape to (6, 11065, 6097) // (#month, #cust, #encode_col)
                transpose to (11065, 6, 6097) // (#cust, #month, #encode_col) = (N, M, encode_col)
            output_info: DataTransformer.output_info // len(output_info) == n_col
        """
        # self.model = []
        # st = 0
        # counter = 0
        # for item in output_info:
        #     if item[1] == "tanh":
        #         st += item[0]
        #         continue
        #     elif item[1] == "softmax":
        #         # beta, gamma 중 1인 부분 인덱스 저장,
        #         # lsw: Sampler와 같으 로직인데 sampler는 np.nonzero 사용
        #         # 어차피 beta, gamma 가 하나의 col 내에 1 하나만 있어야 하므로 동일하긴 한데
        #         end = st + item[0]
        #         counter += 1
        #         # self.model.append(np.argmax(data[:, st:end], axis=-1))
        #         self.model.append(np.argmax(data[:, 0, st:end], axis=-1))
        #         st = end

        counter = list(
            filter(lambda x: x[1] == "softmax", output_info)
        ).__len__()  # == n_col
        # interval: List[Tuple[int, int]], (n_cols, 2). 각 컬럼별 n_beta||n_gamma (st, end) 인덱스 저장
        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        st = 0
        # p: np.array, (n_col, maximum_interval(n_beta||n_gamma)), 학습 중 샘플링 용도로 쓰기 위해 컬럼별 모드 확률 log 취해 저장한것
        self.p = np.zeros((counter, maximum_interval(output_info)))
        # p_sampling: List[np.ndarray], ( n_col, n_beta||n_gamma ). 학습 후 샘플링 용도로 쓰기 위해 컬럼별 모드 확률 log 취하지 않고 원본 저장한것
        self.p_sampling = []
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
                continue
            elif item[1] == "softmax":
                end = st + item[0]
                # mmmm
                # tmp = np.sum(data[:, st:end], axis=0)
                # tmp_sampling = np.sum(data[:, st:end], axis=0)
                tmp = np.sum(data[:, 0, st:end], axis=0)
                tmp_sampling = np.sum(data[:, 0, st:end], axis=0)

                tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                tmp_sampling = tmp_sampling / np.sum(tmp_sampling)
                self.p_sampling.append(tmp_sampling)
                self.p[self.n_col, : item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = end
        assert self.n_col == counter
        self.interval = np.asarray(self.interval)

    def sample_train(
        self, batch: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """모델 학습 위한 컨디션 벡터 및 마스크 등 샘플링
        Args:
            batch: 배치 사이즈
        Returns:
            cond_vec(#batch, #opt), mask(#batch, #cols), col_idx(#batch,), opt_indicater(#batch,)
        """
        if self.n_col == 0:
            return None
        # 원본 피처 에서 하나씩 랜덤 선택
        idx = np.random.choice(np.arange(self.n_col), batch)  # (#batch,)
        # self.n_opt = encode 이후 컨디션 벡터 길이 (beta + gamma)
        vec = np.zeros((batch, self.n_opt), dtype="float32")  # (#batch, #opt)
        # self.n_col = encode 이전 원본 피처 수
        mask = np.zeros((batch, self.n_col), dtype="float32")  # (#batch, #cols)
        mask[np.arange(batch), idx] = 1
        # opt 각 모드에서 선택된 곳을 1로 인디케이팅
        opt1prime = random_choice_prob_index(self.p[idx])  # (#batch,)
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec, mask, idx, opt1prime

    def sample(self, batch: int) -> torch.Tensor:
        """데이터 생성 위한 컨디션 벡터 샘플링
        Args:
            batch: 배치 사이즈
        Returns:
            cond_vec(#batch, #opt)
        """
        if self.n_col == 0:
            return None
        idx = np.random.choice(np.arange(self.n_col), batch)
        vec = np.zeros((batch, self.n_opt), dtype="float32")
        opt1prime = random_choice_prob_index_sampling(self.p_sampling, idx)

        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec


def cond_loss(data, output_info, condvec, mask) -> torch.Tensor:
    """generator loss 계산
    Args:
        data: generated encoded vector + zero pad  (B, gside^2)
        output_info: DataTransformer 내의 alpha, beta, gamma 정보 담긴 리스트
        condvec: given cond vector. 1개만 1이고 나머지는 0  (B, n_opt)
        mask: mask (B, n_feature)
    Returns:
        torch.Tensor: loss
    """
    loss = []
    st = 0  # for data
    st_c = 0  # for cond vec
    for item in output_info:  # encode vec 에서 컨디션 벡터가 가리키는 위치 찾기
        # data 는 (alpha, beta, gamma) 모두 담고 있고 cond_vec는 (beta, gamma) 맨 담음
        if item[1] == "tanh":  # for alpha
            st += item[0]
            continue

        elif item[1] == "softmax":  # for beta, gamma
            end = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
                # mmmm
                data[:, 0, st:end],  # logits
                torch.argmax(condvec[:, st_c:ed_c], dim=1),  # labels
                reduction="none",
            )
            loss.append(tmp)
            st = end
            st_c = ed_c

    loss = torch.stack(loss, dim=1)
    return (loss * mask).sum() / data.size()[0]


class Sampler(object):
    def __init__(self, data: np.ndarray, output_info: list):
        """실제 데이터 샘플러
        학습 중 원본 인코딩 데이터 샘플링을 위해 사용됨
        Args:
            data:
                orig (66590, 6097)
                reshape to (6, 11065, 6097) // (#month, #cust, #encode_col)
                transpose to (11065, 6, 6097) // (#cust, #month, #encode_col) = (N, M, encode_col)
            output_info: DataTransformer.output_info // len(output_info) == n_col
        """
        super(Sampler, self).__init__()
        self.data = data
        # self.model: List[List[np.ndarray]] 모든 고객의 모드 인디케이팅 정보를 담고 있음. ( n_col, n_beta||n_gamma, #indicated_N )
        self.model = []
        self.data_len = len(data)
        st = 0
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
                continue
            elif item[1] == "softmax":
                end = st + item[0]
                tmp = []
                for j in range(item[0]):
                    # beta, gamma 중 1인 부분 인덱스 저장
                    # tmp.append(np.nonzero(data[:, st + j])[0])  # mmmm
                    tmp.append(np.nonzero(data[:, 0, st + j])[0])  # M의 첫월만 관여
                self.model.append(tmp)
                st = end

    def sample(self, n: int, col: torch.Tensor, opt: torch.Tensor) -> np.ndarray:
        """real data sampling
        Args:
            n: #samples
            col: 샘플링할 원본 피처컬럼 인덱스 col_idx(#batch,)
            opt: 샘플링할 컨디션 벡터 인덱스 opt_indicater(#batch,)
        Retrurns:
            np.daarray: sample data
        """
        if col is None:
            idx = np.random.choice(np.arange(self.data_len), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            candidates_idx = self.model[c][o]  # 입력 컨디션에 해당하는 데이터 후보 인덱스들
            idx.append(np.random.choice(candidates_idx))
        return self.data[idx]


class Discriminator(Module):
    def __init__(self, side: int, num_channels: int, input_channel: int):
        """build discriminator

        Args:
            side (int): 정사각 2d로 변환한 피처의 변의 길이
            num_channels (int): CNN 입력 직후 채널 수
            input_channel (int): 입력 채널 크기
        """
        super(Discriminator, self).__init__()
        self.side = side
        layers = self._determine_layers_disc(side, num_channels, input_channel)
        info = len(layers) - 2  # 맨 마지막 conv, relu 제외
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:info])

    def forward(self, input):
        return (self.seq(input)), self.seq_info(input)

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
            layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

        layerNorms = []
        num_c = num_channels
        num_s = side / 2
        for _ in range(len(layer_dims) - 1):
            layerNorms.append([int(num_c), int(num_s), int(num_s)])
            num_c = num_c * 2
            num_s = num_s / 2

        layers_D = []

        for prev, curr, ln in zip(layer_dims, layer_dims[1:], layerNorms):
            layers_D += [
                Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
                LayerNorm(ln),
                LeakyReLU(0.2, inplace=True),
            ]

        layers_D += [
            Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
            # AdaptiveAvgPool2d(1),
        ]
        # layers_D += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), ReLU(True)]

        return layers_D


class Generator(Module):
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
        super(Generator, self).__init__()
        self.side = side
        layers = self._determine_layers_gen(
            side, random_dim, num_channels, output_channel, n_conv_layers
        )
        self.seq = Sequential(*layers)

    def forward(self, input_):
        return self.seq(input_)

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
            layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

        layerNorms = []

        num_c = num_channels * (2 ** (len(layer_dims) - 2))
        num_s = int(side / (2 ** (len(layer_dims) - 1)))
        for _ in range(len(layer_dims) - 1):
            layerNorms.append([int(num_c), int(num_s), int(num_s)])
            num_c = num_c / 2
            num_s = num_s * 2

        layers_G = [
            ConvTranspose2d(
                random_dim,
                layer_dims[-1][0],
                layer_dims[-1][1],
                1,
                0,
                output_padding=0,
                bias=False,
            )
        ]

        for prev, curr, ln in zip(
            reversed(layer_dims), reversed(layer_dims[:-1]), layerNorms
        ):
            layers_G += [
                LayerNorm(ln),
                ReLU(True),
                ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True),
            ]
        return layers_G


class ForeseeNN(Module):
    def __init__(
        self, input_dim: int, output_channels: int = 6, dropout_prob: float = 0.5
    ):
        """첫달 컨디션 벡터를 6개월 것으로 늘리는 네트워크
        월별 변동 관련 로스 추가 필요(ex. sd, 월별 변동률 등) -> generator loss는 6개월 모두 반영 가능
        Args:
            input_dim: 컨디션 벡터 길이
            output_channels: 늘릴 개월 수
        """
        super(ForeseeNN, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels

        # Define the model layers using Sequential
        self.model = Sequential(
            Linear(input_dim, input_dim),
            ReLU(),
            Dropout(dropout_prob),
            Linear(input_dim, input_dim),
            ReLU(),
            Dropout(dropout_prob),
            Linear(input_dim, output_channels * input_dim),
        )

    def forward(self, x):
        x = self.model(x)

        # Reshape the output to the desired shape
        x = x.view(-1, self.output_channels, self.input_dim)
        return x


# gradient_penalty 계산위한 real, fake data의 interpolation 계산
# 선형보간이 아니라 다른 방법 SLERP(Spherical Linear Interpolation) 사용하네??
# SLERP는 특히 두 벡터가 크게 다를 때 유용하다고 함
def slerp(alpha: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Compute the spherical linear interpolation between two tensors.
    SLERP (Spherical Linear Interpolation) is especially useful when the angle between the two vectors is large.
    Unlike linear interpolation, SLERP ensures that the interpolated vectors are normalized and lie on the spherical surface.
    Args:
        alpha (torch.Tensor): The interpolation coefficient. Should be between 0 and 1.
        low (torch.Tensor): The starting tensor (at alpha=0).
        high (torch.Tensor): The ending tensor (at alpha=1).
    Returns:
        torch.Tensor: Interpolated tensor between `low` and `high` based on `alpha`.
    """
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1)).view(alpha.size(0), 1)
    so = torch.sin(omega)
    res = (torch.sin((1.0 - alpha) * omega) / so) * low + (
        torch.sin(alpha * omega) / so
    ) * high

    return res


def calc_gradient_penalty_slerp(
    netD: Discriminator,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    transformer: ImageTransformer,
    device: str = "cpu",
    lambda_: int = 10,
) -> torch.Tensor:
    """
    Agrs:
        real_data: (B, M, #encode+#condvec)
        fake_data: real_data와 동일
    Returns:
        gradient penalty tensor
    """
    # slerp 계산전에 3lank 텐서를 다시 2lank 로 변환 (B*M, #encode+#condvec)
    b, m, enc_cond = real_data.shape
    real_data = real_data.contiguous().view(-1, enc_cond)
    fake_data = fake_data.contiguous().view(-1, enc_cond)
    new_batchsize = real_data.shape[0]
    alpha = torch.rand(new_batchsize, 1, device=device)
    interpolates = slerp(alpha, real_data, fake_data)
    interpolates = interpolates.to(device)
    # slerp 계산후에 2lank 텐서를 다시 3lank 로 변환 (B, M, #encode+#condvec)
    interpolates = interpolates.view(b, m, enc_cond)
    interpolates = transformer.transform(interpolates)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * lambda_

    return gradient_penalty


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class CTABGANSynthesizer:
    def __init__(
        self,
        *,
        class_dim: Tuple[int] = None,
        random_dim: int = 100,  # generartor 에 입력될 랜덤 분포 샘플 차원
        batch_size: int = 500,
        epochs: int = 50,
        ci: int = 1,  # D(critic) 학습 - ci: 반복 횟수
        lr: float = 2e-4,  # adam optimizer learning rate
        betas: Tuple[float, float] = (0.9, 0.999),  # adam optimizer betas
        weight_decay: float = 1e-5,  # adam optimizer betas weight_decay
        device: str = None,
        generator_device: str = None,
        discriminator_device: str = None,
        accumulation_steps: int = 1,  # Gradient accumulation 설정
        # for info loss
        info_loss_wgt: float = 1,
        info_loss_inc_st_epoch: int = None,
        info_loss_inc_rate: float = 2,
        # for generator
        num_channels: int = 64,  # CNN 출력 직전 채널 수
        n_conv_layers: int = 4,  # conv layer 개수
    ):
        if class_dim is None:
            class_dim = (256, 256, 256, 256)

        # for logger
        self.logger = logging.getLogger()

        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.ci = ci
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.generator_device = generator_device
        self.discriminator_device = discriminator_device
        assert isinstance(accumulation_steps, int) and accumulation_steps >= 1
        self.accumulation_steps = accumulation_steps
        # for info loss
        self.info_loss_wgt = info_loss_wgt
        self.info_loss_inc_st_epoch = info_loss_inc_st_epoch
        self.info_loss_inc_rate = info_loss_inc_rate
        # for generator
        self.n_conv_layers = n_conv_layers

        self.is_fit_ = False

    def fit(
        self,
        *,
        encoded_data: np.ndarray,
        data_transformer: DataTransformer,
        target_index: int = None,
    ):
        """GAN 모델 학습
        orig (66590, 6097)
        reshape to (6, 11065, 6097) // (#month, #cust, #encode_col)
        transpose to (11065, 6, 6097) // (#cust, #month, #encode_col) = (N, M, encode_col)

        Args:
            encoded_data: 학습할 인코딩 데이터
            data_transformer: 입력 데이터 인코딩에 사용한 DataTransformer
            target_index: auxiliary classifier 타겟 컬럼 인덱스
        """
        # 데이터 차원 정보
        n_month = encoded_data.shape[1]
        len_encoded = encoded_data.shape[2]
        assert data_transformer.output_dim == len_encoded
        # 데이터 샘플링 객체
        data_sampler = Sampler(encoded_data, data_transformer.output_info)
        data_dim = data_transformer.output_dim  # 전처리 완료된 데이터 차원 수
        # 컨디션 벡터 생성기
        self.cond_generator = Cond(encoded_data, data_transformer.output_info)
        n_opt = self.cond_generator.n_opt

        # 컬럼 수 많아지는 경우 여기 늘려야함
        # n_opt: 가용 conditioning 컬럼 개수
        # col_size_d = data_dim + n_opt
        col_size_d = data_dim  # mmmm 첫달만 넣는 용으로 일단 컨디션벡터 제거
        col_size_g = data_dim
        # 1d -> 2d sqaure matrix 변환 위한 side: H(W) 계산. side는 반드시 2의 배수여야 함
        self.dside = int(np.ceil(col_size_d**0.5))
        if self.dside % 2 == 1:
            self.dside = self.dside + 1
        self.gside = int(np.ceil(col_size_g**0.5))
        gside_factor = 2 ** (self.n_conv_layers - 1)
        if self.gside % gside_factor != 0:
            # self.gside = self.gside + 1
            # n_conv_layers가 4이면 generator 레이어가 2배로 늘리는게 3번 들어감, 때문에 gside = 2^3 의 배수가 되어야 shape 가 맞음
            # 추후 generator 구조 변경되면 함께 수정 필요
            self.gside = ((self.gside // gside_factor) + 1) * gside_factor
        self.logger.info(f"[gside]: {self.gside}, [dside]: {self.dside}")

        # build generator
        col_size_d = self.random_dim + n_opt
        self.generator = Generator(
            self.gside, col_size_d, self.num_channels, n_month, self.n_conv_layers
        ).to(self.device)

        # build discriminator
        self.discriminator = Discriminator(self.dside, self.num_channels, n_month).to(
            self.device
        )

        # # build foreseeNN
        # self.fsn = ForeseeNN(input_dim=n_opt, output_channels=n_month).to(self.device)

        # set optimizer
        # 이부분 설정으로 빼기
        optimizer_params = dict(
            lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        # optimizer_params = dict(
        #     lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.weight_decay
        # )
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(self.discriminator.parameters(), **optimizer_params)

        # build auxiliary classifier
        tcol_idx_st_ed_tuple = (
            None  # classifier 의 타겟 컬럼 idx 찾기 (df transform 인코딩 전후로 idx 가 변화)
        )
        self.classifier = None
        optimizerC = None
        if target_index is not None:
            tcol_idx_st_ed_tuple = get_tcol_idx_st_ed_tuple(
                target_index, data_transformer.output_info
            )
            self.classifier = Classifier(
                data_dim, self.class_dim, tcol_idx_st_ed_tuple
            ).to(self.device)
            optimizerC = optim.Adam(self.classifier.parameters(), **optimizer_params)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside, n_month=n_month)
        self.Dtransformer = ImageTransformer(self.dside, n_month=n_month)

        steps_per_epoch = max(1, len(encoded_data) // self.batch_size)
        # set learning rate scheduler (cosine annealing)
        schedulerG = optim.lr_scheduler.CosineAnnealingLR(
            optimizerG, T_max=steps_per_epoch * self.epochs, eta_min=self.lr * 0.01
        )
        schedulerD = optim.lr_scheduler.CosineAnnealingLR(
            optimizerD, T_max=steps_per_epoch * self.epochs, eta_min=self.lr * 0.01
        )
        schedulerC = optim.lr_scheduler.CosineAnnealingLR(
            optimizerC, T_max=steps_per_epoch * self.epochs, eta_min=self.lr * 0.01
        )
        for epoch in tqdm(range(self.epochs)):
            # lsw: info loss 기중치 에폭 중반부터 증가 실험용
            if (
                self.info_loss_inc_st_epoch is not None
                and epoch >= self.info_loss_inc_st_epoch
            ):
                self.info_loss_wgt *= self.info_loss_inc_rate

            for i_g in tqdm(range(steps_per_epoch)):
                # G(generator), D(critic), C(auxiliary classifier) 학습
                # G: loss_g = loss_g_default + loss_g_info + loss_g_dstream + loss_g_gen
                # D: loss_d = loss_d_was_gp (Was+GP)
                # C: loss_c = loss_c_dstream

                ### ------ D(critic) 학습 ------ ci: 반복 횟수
                for i_ci in range(self.ci * self.accumulation_steps):
                    # 노이즈(z) 및 컨디션 벡터(condvec) 샘플링
                    noisez = torch.randn(
                        self.batch_size, self.random_dim, device=self.device
                    )
                    condvec, mask, col, opt = self.cond_generator.sample_train(
                        self.batch_size
                    )
                    condvec = torch.from_numpy(condvec).to(self.device)
                    mask = torch.from_numpy(mask).to(self.device)
                    # 컨디션 벡터 차원늘리기: (B, n_opt) -> (B, n_month, n_opt) -> (B, n_month * n_opt)
                    # condvec = self.fsn(condvec).view(self.batch_size, -1)

                    noisez = torch.cat([noisez, condvec], dim=1)
                    noisez = noisez.view(
                        self.batch_size,
                        self.random_dim + n_opt,
                        1,
                        1,
                    )

                    # 배치 사이즈만큼 real data 샘플링
                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)  # lsw: 굳이 셔플이 필요한가???
                    # fake와 같은 컨디션 벡터에서 샘플링
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c_perm = condvec[perm]

                    real = torch.from_numpy(real.astype("float32")).to(self.device)

                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = apply_activate(faket, data_transformer.output_info)

                    # mmmm
                    # fake_cat = torch.cat([fakeact, condvec], dim=1)
                    # real_cat = torch.cat([real, c_perm], dim=1)
                    fake_cat = fakeact
                    real_cat = real

                    real_cat_d = self.Dtransformer.transform(real_cat)
                    fake_cat_d = self.Dtransformer.transform(fake_cat)

                    optimizerD.zero_grad()

                    # Was loss 최적화 (loss_d_was_gp)
                    d_real, _ = self.discriminator(real_cat_d)
                    d_real = -torch.mean(d_real)

                    d_fake, _ = self.discriminator(fake_cat_d)
                    d_fake = torch.mean(d_fake)

                    # GP 적용
                    pen = calc_gradient_penalty_slerp(
                        self.discriminator,
                        real_cat,
                        fake_cat,
                        self.Dtransformer,
                        self.device,
                    )
                    loss_d_was_gp = d_real + d_fake + pen
                    loss_d_was_gp.backward()

                    # accumulation_steps 마다 update
                    if (i_ci + 1) % self.accumulation_steps == 0:
                        optimizerD.step()  # 파라미터 업데이트

                    # ci 이터레이션 중 맨 마지막 값만 step에 기록위헤 아래에서 한번에 등록
                    # wandb.log({"gp": pen, "loss_d_was_gp": loss_d_was_gp})

                ### ------ G(generator) 학습 ------

                # 노이즈(z) 샘플링
                noisez = torch.randn(
                    self.batch_size, self.random_dim, device=self.device
                )
                # 컨디션 벡터(condvec) 샘플링
                condvec, mask, col, opt = self.cond_generator.sample_train(
                    self.batch_size
                )
                condvec = torch.from_numpy(condvec).to(self.device)
                mask = torch.from_numpy(mask).to(self.device)
                noisez = torch.cat([noisez, condvec], dim=1)
                noisez = noisez.view(self.batch_size, self.random_dim + n_opt, 1, 1)

                optimizerG.zero_grad()

                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, data_transformer.output_info)  # encoded

                # discriminator에 입력위해 encoded + condvec  concat
                # mmmm
                # fake_cat = torch.cat([fakeact, condvec], dim=1)
                fake_cat = fakeact
                fake_cat = self.Dtransformer.transform(fake_cat)

                y_fake, info_fake = self.discriminator(fake_cat)

                # loss_g_gen (cross_entropy)
                # 왜 activation 전 값을 넣지??? -> logit을 입력으로 받음, 함수 내부에서 softmax 자동 계산
                loss_g_gen = cond_loss(
                    faket, data_transformer.output_info, condvec, mask
                )

                # lsw: real_cat_d 얘는 위의 discrminator것 재활용 하네?
                _, info_real = self.discriminator(real_cat_d)

                # loss_g_default
                loss_g_default = -torch.mean(y_fake)

                # loss_g_info
                # lsw: 1. 논문은 L2놈인데 왜 L1놈 사용중임?
                # lsw: 2. real_cat_d, fake_cat 은 다른 condvec 으로부터 만들어짐. info_fake, info_real은 D의 마지막 전 레이어 값 (약 13,13)
                #           둘을 맞추는게 맞나...? 너무 불안정하지 않을까 싶기도 하고 피처니까 굳이 상관없을듯도 하고
                loss_mean = torch.norm(
                    torch.mean(info_fake.view(self.batch_size, -1), dim=0)
                    - torch.mean(info_real.view(self.batch_size, -1), dim=0),
                    # 1,
                    2,
                )
                loss_std = torch.norm(
                    torch.std(info_fake.view(self.batch_size, -1), dim=0)
                    - torch.std(info_real.view(self.batch_size, -1), dim=0),
                    # 1,
                    2,
                )
                loss_g_info = self.info_loss_wgt * (loss_mean + loss_std)

                loss_g = loss_g_default + loss_g_info + loss_g_gen
                loss_g.backward()

                wandblog = {
                    # for loss_d
                    "gp": pen,
                    "loss_d_was_gp": loss_d_was_gp,
                    # for loss_g
                    "loss_g_default": loss_g_default,
                    "loss_g_info": loss_g_info,
                    "loss_g_gen": loss_g_gen,
                    "info_loss_wgt": self.info_loss_wgt,
                }
                if (i_g + 1) % self.accumulation_steps == 0:
                    optimizerG.step()

                # loss_g_dstream
                if target_index is not None:
                    # lsw: 그냥 위의 fake 그대로 쓰면 안됨? 왜 다시 계산하지? 나중에 빼보자
                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = apply_activate(faket, data_transformer.output_info)
                    # classifier 입력전에 3lank 텐서를 2lank 로 변환 (B*M, #encode)
                    real = real.contiguous().view(-1, len_encoded)
                    fakeact = fakeact.contiguous().view(-1, len_encoded)
                    # classifier 에 입력
                    real_pre, real_label = self.classifier(real)
                    fake_pre, fake_label = self.classifier(fakeact)

                    c_loss = CrossEntropyLoss()

                    ## target 컬럼이 continuous (reg.)
                    if (tcol_idx_st_ed_tuple[1] - tcol_idx_st_ed_tuple[0]) == 1:
                        c_loss = SmoothL1Loss()
                        real_label = real_label.type_as(real_pre)
                        fake_label = fake_label.type_as(fake_pre)
                        real_label = torch.reshape(real_label, real_pre.size())
                        fake_label = torch.reshape(fake_label, fake_pre.size())

                    ## target 컬럼이 categorical (binary clf.)
                    elif (tcol_idx_st_ed_tuple[1] - tcol_idx_st_ed_tuple[0]) == 2:
                        c_loss = BCELoss()
                        real_label = real_label.type_as(real_pre)
                        fake_label = fake_label.type_as(fake_pre)

                    loss_c_dstream = c_loss(real_pre, real_label)
                    loss_g_dstream = c_loss(fake_pre, fake_label)

                    # loss_g_dstream for Generator
                    optimizerG.zero_grad()
                    loss_g_dstream.backward()

                    # loss_g_dstream for Classifier
                    optimizerC.zero_grad()
                    loss_c_dstream.backward()

                    wandblog.update(
                        {
                            "loss_g_dstream": loss_g_dstream,
                            "loss_g": loss_g + loss_g_dstream,
                            "loss_c_dstream": loss_c_dstream,
                            "epoch": epoch,
                            "lr": optimizerG.param_groups[0]["lr"],
                        }
                    )

                    if (i_g + 1) % self.accumulation_steps == 0:
                        optimizerG.step()
                        optimizerC.step()

                else:
                    wandblog.update(
                        {
                            "loss_g": loss_g,
                            "epoch": epoch,
                            "lr": optimizerG.param_groups[0]["lr"],
                        }
                    )

                # update learning rate
                schedulerG.step()
                schedulerD.step()
                schedulerC.step()
                wandb.log(wandblog)  # 시각화 데이터 등록

        self.is_fit_ = True

    def sample(
        self,
        n: int,
        data_transformer: DataTransformer,
    ) -> np.ndarray:
        """sample encode tensor
        Args:
            n: sample number
            data_transformer: DataTransformer
        Returns:
            np.ndarray: (n, M, #encode)
        """
        self.logger.info("[CTAB-SYN]: data sampling start")
        self.generator.eval()
        n_opt = self.cond_generator.n_opt

        output_info = data_transformer.output_info
        steps = n // self.batch_size + 1

        data = []

        self.logger.info("[CTAB-SYN]: generate raw encoding vectors start")
        for i in tqdm(range(steps)):
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample(self.batch_size)
            condvec = torch.from_numpy(condvec).to(self.device)
            noisez = torch.cat([noisez, condvec], dim=1)
            noisez = noisez.view(self.batch_size, self.random_dim + n_opt, 1, 1)

            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket, output_info)  # (B, M, #encode)
            data.append(
                fakeact.detach().cpu().numpy()
            )  # List[torch.Tensor] (#steps, B, #encode)
        self.logger.info("[CTAB-SYN]: generate raw encoding vectors end")

        data = np.concatenate(data, axis=0)
        return data[:n]

    def save_generator(self, mpath: str) -> None:
        """확장자는 *.pth 로"""

        assert self.is_fit_, "only fitted model could be saved, fit first please..."
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        # mpath = os.path.join(mpath, "generator.pth")
        torch.save(self.generator, mpath)
        self.logger.info(f"[CTAB-SYN]: Generator saved at {mpath}")
        return

    def load_generator(self, mpath: str) -> None:
        if not os.path.exists(mpath):
            raise FileNotFoundError(
                f"[CTAB-SYN]: Generator model not exists at {mpath}"
            )

        self.logger.info(f"[CTAB-SYN]: Generator is loaded at {mpath}")
        generator = torch.load(mpath, map_location=torch.device(self.device))
        generator.eval()
        # generator.to(self.device)
        self.generator = generator
        self.is_fit_ = True

        return
