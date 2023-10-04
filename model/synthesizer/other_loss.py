import torch
import numpy as np
import os

realpath = os.path.dirname(os.path.realpath(__file__))


def _js_divergence(p, q):
    """
    Args:
        p, q: (encoded(n_cols), bins)
    Return: (encoded,)
    """
    eps = 1e-8
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)
    return 0.5 * (p * ((p / m).log()) + q * ((q / m).log())).sum(dim=1)


def get_distribution(data: torch.Tensor, bins: int = 20):
    """
    Args:
        data: (B, encoded)
    Return: (encoded, bins(20))
    """
    # hist = torch.histc(data, bins=bins, min=0, max=1)
    hists = [
        torch.histc(row, bins=bins, min=-1, max=1)
        for row in data.permute(1, 0)  # (encoded, B)
    ]
    hists = torch.stack(hists)  # (encoded, bins(20))
    return hists / hists.sum(dim=1).unsqueeze(1)  # (encoded, bins(20))


# def get_jsd_old(data_p, data_q):  # (B, encoded)
#     p = get_distribution(data_p)
#     q = get_distribution(data_q)
#     return _js_divergence(p, q)


def get_jsd(data_q: torch.Tensor, dists: torch.Tensor, m: int, n: int = None):
    """월별 컬럼별 jsd 계산
    Args:
        data_q: (B, encoded)
        m: month idx [0~5]
    Return: scala
    """
    assert data_q.shape[1] == dists.shape[1]
    if n is None:
        p = dists[m]  # (encoded, bins(20))
        q = get_distribution(data_q)
    else:
        _idxs = np.random.choice(range(dists.shape[1]), size=int(n), replace=False)
        p = dists[m][_idxs]  # (#_idxs, 20)
        q = get_distribution(data_q[:, _idxs])
    return _js_divergence(p, q).mean()


def pearson_correlation(x, y):
    # 각 변수의 평균을 계산
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # 각 변수에서 평균을 뺀 값을 계산
    xm = x - mean_x
    ym = y - mean_y

    # 공분산을 계산
    cov = torch.mean(xm * ym)

    # 각 변수의 표준편차를 계산
    sx = torch.std(x, correction=0)
    sy = torch.std(y, correction=0)

    # 피어슨 상관계수를 계산
    corr = cov / (sx * sy)

    return corr


def batch_pearson_correlation(x, y):
    #  input: (B, N)
    # 각 변수의 평균을 계산
    mean_x = torch.mean(x, dim=1).unsqueeze(1)
    mean_y = torch.mean(y, dim=1).unsqueeze(1)

    #  input: (B, 1)
    # 각 변수에서 평균을 뺀 값 (편차)을 계산
    xm = x - mean_x
    ym = y - mean_y

    # 공분산을 계산
    cov = torch.mean(xm * ym, dim=1)  # (B, 1)

    # 각 변수의 표준편차를 계산
    eps = 1e-8
    sx = torch.std(x, correction=1, dim=1) + eps
    sy = torch.std(y, correction=1, dim=1) + eps

    # 피어슨 상관계수를 계산
    corr = cov / (sx * sy)

    return corr


corrs_idx_pairs = np.load(os.path.join(realpath, "../../corrs-pairs.npy"))  # (2, 약1e6)


def get_cdiff_loss(bb: torch.Tensor, corrs: torch.Tensor, n=1000):
    """
    bb: (B, M, #encoded)
    """
    global corrs_idx_pairs
    # global corrs
    bb_nrow = bb.shape[0]
    bb2 = bb.permute((1, 2, 0)).reshape(-1, bb_nrow)

    if n >= corrs_idx_pairs.shape[1]:
        pairs = corrs_idx_pairs
    else:
        _idxs = np.random.choice(
            range(corrs_idx_pairs.shape[1]), size=int(n), replace=False
        )
        pairs = corrs_idx_pairs[:, _idxs]

    aa_corr = corrs[pairs[0], pairs[1]]
    # print(bb2.shape, pairs)
    bb_corr = batch_pearson_correlation(bb2[pairs[0]], bb2[pairs[1]])
    cdiff = aa_corr - bb_corr
    cdiff = cdiff[~(cdiff.isnan() | cdiff.isinf())]  # 비정상치 (nan, inf, -inf) 제거
    cdiff_mse = (cdiff**2).mean()
    # print(cdiff_mse)
    return cdiff_mse


def get_cdiff_loss_old(aa, bb, n=1000):
    aa_nrow = aa.shape[0]
    bb_nrow = bb.shape[0]
    aa2 = aa.permute((1, 2, 0)).reshape(-1, aa_nrow)
    bb2 = bb.permute((1, 2, 0)).reshape(-1, bb_nrow)

    pairs = torch.randint(low=0, high=aa2.shape[0], size=(2, int(n)))  # 중복 컬럼 허용
    # pairs = np.random.choice(
    #     range(aa2.shape[0]), size=(2, int(n)), replace=False
    # )  # 중복 컬럼 제거
    # print(aa2.shape, pairs)

    aa_corr = batch_pearson_correlation(aa2[pairs[0]], aa2[pairs[1]])
    bb_corr = batch_pearson_correlation(bb2[pairs[0]], bb2[pairs[1]])
    cdiff = aa_corr - bb_corr
    cdiff = cdiff[~(cdiff.isnan() | cdiff.isinf())]  # 비정상치 (nan, inf, -inf) 제거
    cdiff_mse = (cdiff**2).mean()
    # print(cdiff_mse)
    return cdiff_mse
