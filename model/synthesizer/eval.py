import torch
import numpy as np


def _js_divergence(p, q):
    eps = 1e-8
    m = 0.5 * (p + q)
    return 0.5 * (p * ((p / m + eps).log()) + q * ((q / m + eps).log())).sum()


def get_distribution(data, bins=20):
    hist = torch.histc(data, bins=bins, min=0, max=1)
    return hist / hist.sum()


def get_jsd(data_p, data_q):
    p = get_distribution(data_p)
    q = get_distribution(data_q)
    return _js_divergence(p, q)


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


def get_cdiff_loss(ee, ss, n=1000):
    ee_nrow = ee.shape[0]
    ss_nrow = ss.shape[0]
    ee2 = ee.permute((1, 2, 0)).reshape(-1, ee_nrow)
    ss2 = ss.permute((1, 2, 0)).reshape(-1, ss_nrow)

    # pairs = torch.randint(low=0, high=ee2.shape[0], size=(2, n))  # 중복 컬럼 허용
    pairs = np.random.choice(
        range(ee2.shape[0]), size=(2, n), replace=False
    )  # 중복 컬럼 제거
    # print(ee2.shape, pairs)

    ee_corr = batch_pearson_correlation(ee2[pairs[0]], ee2[pairs[1]])
    ss_corr = batch_pearson_correlation(ss2[pairs[0]], ss2[pairs[1]])
    cdiff = ee_corr - ss_corr
    cdiff = cdiff[~(cdiff.isnan() | cdiff.isinf())]  # 비정상치 (nan, inf, -inf) 제거
    cdiff_mse = (cdiff**2).mean()
    print(cdiff_mse)
    return cdiff_mse
