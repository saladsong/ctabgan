import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

# class ForeseeNN(Module):
#     def __init__(
#         self, input_dim: int, output_channels: int = 6, dropout_prob: float = 0.5
#     ):
#         """첫달 컨디션 벡터를 6개월 것으로 늘리는 네트워크
#         월별 변동 관련 로스 추가 필요(ex. sd, 월별 변동률 등) -> generator loss는 6개월 모두 반영 가능
#         Args:
#             input_dim: 컨디션 벡터 길이
#             output_channels: 늘릴 개월 수
#         """
#         super(ForeseeNN, self).__init__()
#         self.input_dim = input_dim
#         self.output_channels = output_channels

#         # Define the model layers using Sequential
#         self.model = Sequential(
#             Linear(input_dim, input_dim),
#             ReLU(),
#             Dropout(dropout_prob),
#             Linear(input_dim, input_dim),
#             ReLU(),
#             Dropout(dropout_prob),
#             Linear(input_dim, output_channels * input_dim),
#         )

#     def forward(self, x):
#         x = self.model(x)

#         # Reshape the output to the desired shape
#         x = x.view(-1, self.output_channels, self.input_dim)
#         return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# class TransformerModel(nn.Module):
class ForeseeNN(nn.Module):
    def __init__(
        self,
        # ntoken: int,
        d_input: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.encoder = nn.Linear(d_input, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, d_input)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = torch.tanh(output)  # activation fn. encoded 데이터와 레인지 통일 [-1~1]
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


# seq_len = 6  # 시퀀스 길이
# ntokens = len(vocab)  # 단어 사전(어휘집)의 크기 -> a,b,r 크기
# emsize = 256  # 임베딩 차원
# d_hid = 256  # ``nn.TransformerEncoder`` 에서 피드포워드 네트워크(feedforward network) 모델의 차원
# nlayers = 4  # ``nn.TransformerEncoder`` 내부의 nn.TransformerEncoderLayer 개수
# nhead = 4  # ``nn.MultiheadAttention`` 의 헤드 개수
# dropout = 0.2  # 드랍아웃(dropout) 확률
# model = ForeseeNN(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


# criterion = nn.CrossEntropyLoss()
# lr = 5.0  # 학습률(learning rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


# model.train()  # 학습 모드 시작
# total_loss = 0.0
# log_interval = 200
# src_mask = generate_square_subsequent_mask(seq_len).to(device)

# num_batches = len(train_data) // seq_len
# for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_len)):
#     data, targets = get_batch(train_data, i)
#     seq_len = data.size(0)
#     if seq_len != seq_len:  # 마지막 배치에만 적용
#         src_mask = src_mask[:seq_len, :seq_len]
#     output = model(data, src_mask)
#     loss = criterion(output.view(-1, ntokens), targets)

#     optimizer.zero_grad()
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#     optimizer.step()

#     # total_loss += loss.item()
#     # if batch % log_interval == 0 and batch > 0:
#     #     lr = scheduler.get_last_lr()[0]
#     #     ms_per_batch = (time.time() - start_time) * 1000 / log_interval
#     #     cur_loss = total_loss / log_interval
#     #     ppl = math.exp(cur_loss)
#     #     print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
#     #             f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
#     #             f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
#     #     total_loss = 0
#     #     start_time = time.time()


def foresee(
    input: torch.Tensor,
    fsn: ForeseeNN,
    n_month: int,
    optimizerF: torch.optim.Optimizer = None,
) -> torch.Tensor:
    """트랜스포머 이용해 첫 달 데이터로 이후 n_month 예측
    IN: (B, M(S)(1), encode) -> (S(1), B, encode) -> (S(6), B, encode) ->
    OUT: (B, S(6), encode)
    """
    # fsn.train()  # 학습 모드 시작
    # optimizerF.zero_grad()
    fsn.eval()
    with torch.no_grad():
        # input: fakeact  # (B, M(S)(1), encode) -> (S(1), B, encode)
        # output: (S(6), B, encode)
        data = input.permute(1, 0, 2)  # (S, B, encode)
        output = [data]
        for i in range(n_month - 1):
            output.append(fsn(torch.concat(output), None)[-1].unsqueeze(0))

        # (S(6), B, encode) -> (B, S(6), encode)
        ret = torch.concat(output).permute(1, 0, 2)
    return ret
