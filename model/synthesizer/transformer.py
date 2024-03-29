import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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


class TransformerModel(nn.Module):
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


def foresee(
    input: torch.Tensor,
    trsfm: TransformerModel,
    n_month: int,
    train: bool = False,
    optimizerF: torch.optim.Optimizer = None,
) -> torch.Tensor:
    """트랜스포머 이용해 첫 달 데이터로 이후 n_month 예측
    IN: (B, M(S)(1), encode) -> (S(1), B, encode) -> (S(6), B, encode) ->
    OUT: (B, S(6), encode)
    """

    def _run(input):
        # input: fakeact  # (B, M(S)(1), encode) -> (S(1), B, encode)
        # output: (S(6), B, encode)
        data = input.permute(1, 0, 2)  # (S, B, encode)
        output = [data]
        for i in range(n_month - 1):
            output.append(trsfm(torch.concat(output), None)[-1].unsqueeze(0))

        # (S(6), B, encode) -> (B, S(6), encode)
        ret = torch.concat(output).permute(1, 0, 2)
        return ret

    if train:
        assert optimizerF is not None
        trsfm.train()  # 학습 모드 시작
        optimizerF.zero_grad()
        ret = _run(input)
    else:
        trsfm.eval()
        with torch.no_grad():
            ret = _run(input)
    return ret


def foresee_dropout(
    input: torch.Tensor,
    trsfm: TransformerModel,
    n_month: int,
    dropout: float = 0.01,
) -> torch.Tensor:
    """트랜스포머 이용해 첫 달 데이터로 이후 n_month 예측
    IN: (B, M(S)(1), encode) -> (S(1), B, encode) -> (S(6), B, encode) ->
    OUT: (B, S(6), encode)
    """

    def _run(input):
        # input: fakeact  # (B, M(S)(1), encode) -> (S(1), B, encode)
        # output: (S(6), B, encode)
        data = input.permute(1, 0, 2)  # (S, B, encode)
        output = [data]
        for i in range(n_month - 1):
            output.append(trsfm(torch.concat(output), None)[-1].unsqueeze(0))

        # (S(6), B, encode) -> (B, S(6), encode)
        ret = torch.concat(output).permute(1, 0, 2)
        return ret

    # model train/eval 상태확인
    is_training_orig = trsfm.training
    if not is_training_orig:
        trsfm.train()

    # 원래 dropout p 저장
    orig_dropout_list = []
    orig_dropout1_list = []
    orig_dropout2_list = []
    for layer in trsfm.transformer_encoder.layers:
        orig_dropout_list.append(layer.dropout.p)
        orig_dropout1_list.append(layer.dropout1.p)
        orig_dropout2_list.append(layer.dropout2.p)

    # 입력 dropout p 변경
    for layer in trsfm.transformer_encoder.layers:
        layer.dropout.p = dropout
        layer.dropout1.p = dropout
        layer.dropout2.p = dropout

    # trsfm.eval()  # 이건 dropout 완전 작동 안하므로 사용하면 안됨
    with torch.no_grad():
        ret = _run(input)

    # 원래 dropout p 복구
    for i, layer in enumerate(trsfm.transformer_encoder.layers):
        layer.dropout.p = orig_dropout_list[i]
        layer.dropout1.p = orig_dropout1_list[i]
        layer.dropout2.p = orig_dropout2_list[i]

    # model train/eval 상태복구
    if not is_training_orig:
        trsfm.eval()

    return ret
