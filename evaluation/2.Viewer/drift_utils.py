#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================================================
# 파일명:    drift_metrics.py
# 설명:      단일 변수 분포 드리프트 지표 산출
# =======================================================================

import copy
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.special import rel_entr
from scipy.stats import chi2, kstwo


def _scale_points(points, min_range, max_range):
    """지점들을 지정된 최소와 최대값에 따라 스케일링

    Args:
        points (ndarray): 1D array containing data with numerical type
        min_range (float): 스케일링 기준이 될 최소값
        max_range (float): 스케일링 기준이 될 최대값

    Returns:
        ndarray: 1D array containing data with numerical type
    """
    points += -(np.min(points))
    points /= np.max(points) / (max_range - min_range)
    points += min_range

    return points


def _trim_minor_bins(
    df_cate_frequencies, max_bin_cum_pct=0.95, min_bin_pct=0.01, max_n_bins=50
):
    """소수 비율을 차지하는 구간을 지정된 조건에 따라 그룹화

    비율 기준으로 정렬했을 때 누적 비율이 상위 `max_bin_cum_pct`를 초과하며
    구간의 비율이 `min_bin_pct` 미만이거나, 비율 기준 상위 `max_n_bins`위
    안에 들지 않는 구간들을 하나의 구간으로 묶음

    Args:
        df_cate_frequencies (pd.DataFrame): 구간별 빈도 정보
        max_bin_cum_pct (float): 최대 구간 누적 비율
        min_bin_pct (float): 최소 구간 비율
        max_n_bins (int): 최대 구간 개수

    Returns:
        pd.DataFrame: 소수 비율 구간 그룹화가 적용된 구간별 빈도 정보
    """
    df_cate_frequencies["bin_pct"] = (
        df_cate_frequencies["frequency"] / df_cate_frequencies["frequency"].sum()
    )
    df_cate_frequencies["bin_cum_pct"] = (
        df_cate_frequencies["bin_pct"].expanding().sum()
    )
    df_cate_frequencies["rank"] = df_cate_frequencies["bin_pct"].rank(
        method="min", ascending=False
    )

    df_cate_frequencies["bin_index"] = np.where(
        (
            (df_cate_frequencies["bin_cum_pct"] > max_bin_cum_pct)
            & (df_cate_frequencies["bin_pct"] < min_bin_pct)
        )
        | (df_cate_frequencies["rank"] > max_n_bins),
        999,
        df_cate_frequencies["bin_index"],
    )

    return df_cate_frequencies.drop(columns=["bin_pct", "bin_cum_pct", "rank"])


def _edge_points_to_bin_ranges(points):
    """구간 분할 지점들로 구간별 좌측/우측 값 산출

    Args:
        points (ndarray): 1D array containing data with numerical type

    Returns:
        pd.DataFrame: 구간별 범위 정보 (좌측/우측)
    """
    df_bin_ranges = (
        pd.concat(
            [
                pd.Series(points[:-1], name="bin_left"),
                pd.Series(points[1:], name="bin_right"),
            ],
            axis=1,
        )
        .reset_index()
        .rename(columns={"index": "bin_index"})
    )
    df_bin_ranges.loc[:, "bin_value"] = None

    return df_bin_ranges


def _bin_ranges_to_edge_points(bin_left_vals, bin_right_vals):
    """구간의 좌측/우측 값으로 구간 분할 지점들을 산출

    Args:
        bin_left_vals (ndarray): 1D array containing data with numerical type
        bin_right_vals (ndarray): 1D array containing data with numerical type

    Returns:
        ndarray: 1D array containing data with numerical type
    """
    return np.append(bin_left_vals, bin_right_vals[-1])


def get_cate_frequencies(arr, trim=True):
    """범주형 데이터에 대해 데이터 구간별 빈도 산출

    [To-Do] 순서형 변수의 경우 순서 정보를 반영해 범주 정렬 필요
    (현재 범주는 빈도 내림차 순으로 정렬됨)

    Args:
        arr (ndarray): 1D array containing data with categorical type
        trim (bool): 소수 비율 구간 그룹화 적용 여부

    Returns:
        tuple: 구간별 빈도, 구간 정보
    """
    df_cate_frequencies = (
        pd.value_counts(arr, dropna=False).rename("frequency").reset_index()
    )
    df_cate_frequencies.columns = ["bin_value", "frequency"]
    #     .rename(columns={'index': 'bin_value'})
    #     print('@@@@@@@')
    #     print(pd.value_counts(arr, dropna=False))
    #     print(df_cate_frequencies)
    # df_cate_frequencies = pd.value_counts(arr, dropna=False).rename('frequency').reset_index().rename(columns={'index': 'bin_value'})
    df_cate_frequencies.loc[:, "bin_left"] = np.nan
    df_cate_frequencies.loc[:, "bin_right"] = np.nan
    df_cate_frequencies = df_cate_frequencies.reset_index().rename(
        columns={"index": "bin_index"}
    )

    if trim:
        df_cate_frequencies = _trim_minor_bins(df_cate_frequencies)

    return df_cate_frequencies["frequency"].values, df_cate_frequencies[
        ["bin_index", "bin_left", "bin_right", "bin_value"]
    ].reset_index(drop=True)


def get_bin_frequencies(arr, bin_edges=None, bin_method="equal_width", n_bins=10):
    """수치형 데이터에 대해 데이터 구간별 빈도 산출

    구간 분할 지점이 제공된 경우 이를 사용해 분할하고, 제공되지 않은 경우
    다음 중 지정된 구간화 방식을 사용해 데이터를 `n_bins`개로 구간화함:
    - `equal_width` (등간격):
        각 구간이 균등한 너비를 가지도록 구간화
    - `equal_freq` (등빈도):
        각 구간이 균등한 빈도를 가지도록 구간화. 완전히 동일하진 않을 수 있음
    (데이터가 제공된 구간 범위를 벗어날 경우 가장 끝의 구간으로 분류됨)

    [To-Do] 결측값을 별도 구간으로 추가 (현재 구간에서 제외됨)
    [To-Do] 0을 별도 구간으로 설정하는 옵션 추가

    구간 개수를 지정하는 대신, `bin_method`를 통해 numpy.histogram_bin_edges에서
    사용할 구간 너비 최적화 방식을 다음 중에서 지정할 수 있음:
    - `auto`:
        Maximum of the `sturges` and `fd` estimators. Provides good
        all around performance.
    - `fd` (Freedman Diaconis Estimator):
        Robust (resilient to outliers) estimator that takes into account
        data variability and data size.
    - `doane`:
        An improved version of Sturges' estimator that works better with
        non-normal datasets.
    - `scott`:
        Less robust estimator that that takes into account data variability
        and data size.
    - `stone`:
        Estimator based on leave-one-out cross-validation estimate of the
        integrated squared error. Can be regarded as a generalization of
        Scott's rule.
    - `rice`:
        Estimator does not take variability into account, only data size.
        Commonly overestimates number of bins required.
    - `sturges`:
        R's default method, only accounts for data size. Only optimal for
        gaussian data and underestimates number of bins for large
        non-gaussian datasets.
    - `sqrt`:
        Square root (of data size) estimator, used by Excel and other
        programs for its speed and simplicity.

    Args:
        arr (ndarray): 1D array containing data with numerical type
        bin_edges (ndarray): 1D array containing data with numerical type
        bin_method (str): 구간화 방식 {`equal_width`, `equal_freq`}
        n_bins (int): 구간 개수

    Returns:
        tuple:
            ndarray (1D array containing data with numerical type),
            ndarray (1D array containing data with numerical type)
    """
    supported_bin_width_opt_methods = [  # 지원되는 최적화 방식
        "auto",
        "fd",
        "doane",
        "scott",
        "stone",
        "rice",
        "sturges",
        "sqrt",
    ]
    if bin_method in supported_bin_width_opt_methods:  # 구간 너비 최적화 사용
        # 지정된 구간 너비 최적화 방식으로 히스토그램 생성
        frequencies, bin_edges = np.histogram(
            a=arr, bins=bin_method, range=(np.nanmin(arr), np.nanmax(arr))
        )

    else:  # 구간 너비 최적화 사용하지 않음
        if bin_edges is None:  # 구간 분할 지점 지정되지 않음
            # 고유값 수가 구간 개수보다 적은 경우 구간 개수를 고유값 수로 설정
            n_unique_vals = arr.nunique()
            if n_unique_vals < n_bins:
                n_bins = n_unique_vals

            bin_edges = np.arange(0, n_bins + 1) / (n_bins) * 100

            if bin_method == "equal_width":  # 등간격
                bin_edges = _scale_points(bin_edges, np.nanmin(arr), np.nanmax(arr))

            elif bin_method == "equal_freq":  # 등빈도
                bin_edges = np.stack([np.nanpercentile(arr, p) for p in bin_edges])

        else:  # 구간 분할 지점 지정됨
            if min(arr) < bin_edges[0]:  # 최소값이 최소 구간을 벗어남
                bin_edges[0] = min(arr)

            if max(arr) > bin_edges[-1]:  # 최대값이 최대 구간을 벗어남
                bin_edges[-1] = max(arr)

        # 지정된 구간 분할 지점으로 히스토그램 생성
        frequencies, bin_edges = np.histogram(a=arr, bins=bin_edges)

    return frequencies, bin_edges


def generate_freq_by_feat(
    data, bin_method="equal_width", n_bins=10, feat_index_col="feat_index"
):
    """변수별 데이터 구간 및 분포 정보 생성

    Args:
        data (ndarray): 2D array containing data with numerical type
        bin_method (str): 구간화 방식 {`equal_width`, `equal_freq`}
        n_bins (int): 구간 개수
        feat_index_col (str): 변수 인덱스 컬럼명
    Returns:
        pd.DataFrame: 변수별 데이터 구간 및 분포 정보
    """
    df_freq_by_feat = []
    for i in range(data.shape[1]):
        if is_numeric_dtype(data.iloc[:, i]):
            frequencies, bin_edges = get_bin_frequencies(
                data.iloc[:, i], bin_method=bin_method, n_bins=n_bins
            )
            df_bin_info = _edge_points_to_bin_ranges(bin_edges)
        else:
            frequencies, df_bin_info = get_cate_frequencies(data.iloc[:, i])

        df_freq_by_feat.append(
            pd.concat(
                [
                    pd.Series([i] * len(frequencies), name="feat_index"),
                    df_bin_info,
                    pd.Series(frequencies, name="base_freq"),
                ],
                axis=1,
            )
        )

    df_freq_by_feat = pd.concat(df_freq_by_feat).reset_index(drop=True)

    # 구간별 비율 산출
    df_freq_by_feat["base_pct"] = df_freq_by_feat[
        "base_freq"
    ] / df_freq_by_feat.groupby(feat_index_col)["base_freq"].transform("sum")

    return df_freq_by_feat


def calc_target_freq_by_feat(
    df_base_freq_by_feat, target_data, feat_index_col="feat_index"
):
    """기준 데이터 구간 사용해 계산한 대상 분포 정보 추가

    [To-Do] 현재 범주 값이 NaN인 경우 Join이 이루어지나, 이는 pandas의 버그임
    NaN의 범주 값을 다른 값으로 대체하는 방법 고려 필요

    Args:
       df_base_freq_by_feat (pd.DataFrame): 변수별 기준 데이터 구간 및 분포
       target_data (ndarray): 2D array containing data with numerical type
       feat_index_col (str): 변수 인덱스 컬럼명

    Returns:
       pd.DataFrame: 기준 및 대상 분포 정보
    """
    # 변수 Index 목록
    feat_indices = sorted(df_base_freq_by_feat[feat_index_col].unique())

    df_all_freq_by_feat = []
    for i, feat_index in enumerate(feat_indices):
        if is_numeric_dtype(target_data.iloc[:, i]):  # 수치형 변수
            # 해당 변수의 기준 데이터 구간별 빈도
            df_base_freqs = df_base_freq_by_feat[
                df_base_freq_by_feat[feat_index_col] == feat_index
            ].reset_index(drop=True)
            base_frequencies = df_base_freqs["base_freq"].values

            # 기준 데이터의 구간 분할 지점 가져오기
            base_bin_edges = _bin_ranges_to_edge_points(
                df_base_freq_by_feat[
                    df_base_freq_by_feat[feat_index_col] == feat_index
                ]["bin_left"].values,
                df_base_freq_by_feat[
                    df_base_freq_by_feat[feat_index_col] == feat_index
                ]["bin_right"].values,
            )
            # 기준 데이터의 구간 분할 지점으로 대상 데이터 빈도 산출
            target_frequencies, target_bin_edges = get_bin_frequencies(
                target_data.iloc[:, i], bin_edges=base_bin_edges
            )

            # 기준 데이터와 대상 데이터 구간별 빈도 정보 결합
            df_full = pd.concat(
                [df_base_freqs, pd.Series(target_frequencies, name="target_freq")],
                axis=1,
            )

        else:  # 범주형 변수
            # 해당 변수의 기준 데이터 구간별 빈도
            df_base_freqs = df_base_freq_by_feat[
                df_base_freq_by_feat[feat_index_col] == feat_index
            ].reset_index(drop=True)

            # 대상 데이터 빈도 산출
            target_frequencies, df_bin_info = get_cate_frequencies(
                target_data.iloc[:, i]
            )
            df_target_freqs = pd.concat(
                [df_bin_info, pd.Series(target_frequencies, name="target_freq")], axis=1
            )

            # 기준 데이터와 대상 데이터 구간별 빈도 정보를 범주 값으로 join
            df_full = df_base_freqs.merge(
                df_target_freqs[["bin_value", "target_freq"]],
                how="outer",
                on=["bin_value"],
            )
            df_full[feat_index_col] = feat_index

            # 구간 빈도 결측값 대체 (기준 데이터 또는 대상 데이터에 존재하지 않는 범주)
            df_full["base_freq"] = df_full["base_freq"].fillna(0)
            df_full["target_freq"] = df_full["target_freq"].fillna(0)

            # 구간 Index 결측값을 알수없는 범주로 대체 (기준 데이터에는 존재하나 대상 데이터에 존재하지 않는 범주)
            UNKNOWN_CATE_FLAG = -1
            df_full["bin_index"] = df_full["bin_index"].fillna(UNKNOWN_CATE_FLAG)

            # 알수없는 범주 빈도 총합 집계
            df_unknown = df_full[df_full["bin_index"] == UNKNOWN_CATE_FLAG]
            df_unknown = (
                df_unknown.groupby(by=[feat_index_col, "bin_index"])[
                    ["base_freq", "target_freq"]
                ]
                .sum()
                .reset_index()
            )

            # 알수없는 범주의 집계된 빈도로 업데이트
            df_full = df_full[df_full["bin_index"] != UNKNOWN_CATE_FLAG]
            df_full = df_full.append(df_unknown, sort=False).reset_index(drop=True)

            # base_frequencies = df_full['base_freq'].values
            # target_frequencies = df_full['target_freq'].values

        df_all_freq_by_feat.append(df_full)

    # 변수별 구간별 빈도 정보 결합
    df_all_freq_by_feat = pd.concat(df_all_freq_by_feat).reset_index(drop=True)

    # 구간별 비율 산출
    df_all_freq_by_feat["base_pct"] = df_all_freq_by_feat[
        "base_freq"
    ] / df_all_freq_by_feat.groupby(feat_index_col)["base_freq"].transform("sum")
    df_all_freq_by_feat["target_pct"] = df_all_freq_by_feat[
        "target_freq"
    ] / df_all_freq_by_feat.groupby(feat_index_col)["target_freq"].transform("sum")

    # 범주 값 컬럼의 NaN을 None으로 대체
    df_all_freq_by_feat["bin_value"] = np.where(
        pd.isnull(df_all_freq_by_feat["bin_value"]),
        None,
        df_all_freq_by_feat["bin_value"],
    )

    # 최종 데이터 타입으로 변환
    df_all_freq_by_feat[feat_index_col] = df_all_freq_by_feat[feat_index_col].astype(
        int
    )
    df_all_freq_by_feat["bin_index"] = df_all_freq_by_feat["bin_index"].astype(int)
    df_all_freq_by_feat["base_freq"] = df_all_freq_by_feat["base_freq"].astype(int)
    df_all_freq_by_feat["target_freq"] = df_all_freq_by_feat["target_freq"].astype(int)

    return df_all_freq_by_feat


def get_feat_psi_per_bin(base_frequencies, target_frequencies):
    """기준 및 대상 데이터 분포 정보를 사용해 단일 변수에 대한 구간별 PSI 계산

    [Deprecated] 변수별 계산 결과를 바로 컬럼으로 추가하도록 변경함 (`calc_psi_by_feat`)

    Args:
       base_frequencies (ndarray): 1D array containing data with numerical type
       target_frequencies (ndarray): 1D array containing data with numerical type

    Returns:
       pd.Series: 구간별 PSI 값
    """
    base_percents = base_frequencies / sum(base_frequencies)
    target_percents = target_frequencies / sum(target_frequencies)

    def psi_i(base_pct_i, target_pct_i, eps=1e-8):
        """특정 구간의 PSI 값 계산
        비율이 0인 경우 오류 방지를 위해 작은 수치 (eps)로 대체

        Args:
            base_pct_i (float): 기준 데이터 내 구간 비율
            target_pct_i (float): 대상 데이터 내 구간 비율
            eps (float): 0의 비율을 대체할 작은 수치

        Returns:
            float: PSI 값
        """
        if base_pct_i == 0:
            base_pct_i = eps
        if target_pct_i == 0:
            target_pct_i = eps

        psi_i = (target_pct_i - base_pct_i) * np.log(target_pct_i / base_pct_i)

        return psi_i

    return pd.Series(
        [
            psi_i(base_percents[i], target_percents[i])
            for i in range(len(base_percents))
        ],
        name="psi_i",
    )


def calc_psi_by_feat(df_all_freq_by_feat, eps=1e-8, feat_index_col="feat_index"):
    """기준 및 대상 데이터 분포 정보 사용해 변수별 PSI 산출

    Args:
       df_all_freq_by_feat (pd.DataFrame): 변수별 기준 및 대상 데이터 구간 및 분포
       feat_index_col (str): 변수 인덱스 컬럼명

    Returns:
       pd.DataFrame: PSI 계산 테이블
    """
    df_psi_by_feat = copy.deepcopy(df_all_freq_by_feat)

    # 구간별 비율 차이 계산
    df_psi_by_feat["pct_diff"] = (
        df_psi_by_feat["target_pct"] - df_psi_by_feat["base_pct"]
    )

    # 구간별 PSI 계산
    df_psi_by_feat["woe_i"] = np.log(
        np.where(df_psi_by_feat["target_pct"] == 0, eps, df_psi_by_feat["target_pct"])
        / np.where(df_psi_by_feat["base_pct"] == 0, eps, df_psi_by_feat["base_pct"])
    )
    df_psi_by_feat["psi_i"] = df_psi_by_feat["pct_diff"] * df_psi_by_feat["woe_i"]

    return df_psi_by_feat


def calc_wasserstein_by_feat(df_all_freq_by_feat, feat_index_col="feat_index"):
    """기준 및 대상 데이터 분포 정보 사용해 변수별 Wasserstein 거리 산출

    Note:
    - 각 구간의 너비가 동일하다는 가정 하에 (i.e. `equal_width` 방식으로
      생성한 구간), 구간별 좌측 값을 사용해 구간 간 거리를 계산함
    - 범주형 변수의 경우 변수가 순서형이라고 가정하며, 모든 구간의 너비를
      동일한 값 (1)으로 설정해 계산함

    Args:
       df_all_freq_by_feat (pd.DataFrame): 변수별 기준 및 대상 데이터 구간 및 분포
       feat_index_col (str): 변수 인덱스 컬럼명

    Returns:
       pd.DataFrame: Wasserstein 계산 테이블
    """
    df_wass_by_feat = copy.deepcopy(df_all_freq_by_feat)

    # 구간별 누적 비율 계산
    df_wass_by_feat["base_cum_pct"] = (
        df_wass_by_feat.groupby(feat_index_col)["base_pct"]
        .expanding()
        .sum()
        .reset_index(drop=True)
    )
    df_wass_by_feat["target_cum_pct"] = (
        df_wass_by_feat.groupby(feat_index_col)["target_pct"]
        .expanding()
        .sum()
        .reset_index(drop=True)
    )
    df_wass_by_feat["abs_cum_pct_diff"] = np.abs(
        df_wass_by_feat["target_cum_pct"] - df_wass_by_feat["base_cum_pct"]
    )

    # 구간별 Wasserstein 산출
    df_wass_by_feat["bin_width"] = (
        df_wass_by_feat["bin_right"] - df_wass_by_feat["bin_left"]
    ).fillna(1)
    df_wass_by_feat["wass_i"] = (
        df_wass_by_feat["abs_cum_pct_diff"] * df_wass_by_feat["bin_width"]
    )

    # Normalize Wasserstein by maximum distance
    df_wass_by_feat = df_wass_by_feat.merge(
        df_wass_by_feat.groupby(feat_index_col)["bin_index"]
        .nunique()
        .rename("n_bins")
        .reset_index(),
        how="left",
        on=feat_index_col,
    )
    df_wass_by_feat["norm_wass_i"] = (
        df_wass_by_feat["abs_cum_pct_diff"] / df_wass_by_feat["n_bins"]
    )

    return df_wass_by_feat


def calc_lp_dist_by_feat(df_all_freq_by_feat, feat_index_col="feat_index"):
    """기준 및 대상 데이터 분포 정보 사용해 변수별 L-p Norm 거리 산출

    Args:
       df_all_freq_by_feat (pd.DataFrame): 변수별 기준 및 대상 데이터 구간 및 분포
       feat_index_col (str): 변수 인덱스 컬럼명

    Returns:
       pd.DataFrame: L-p Norm 거리 계산 테이블
    """
    df_lp_dist_by_feat = copy.deepcopy(df_all_freq_by_feat)

    # 기준 분포와 대상 분포 간 차
    df_lp_dist_by_feat["freq_diff"] = (
        df_lp_dist_by_feat["target_pct"] - df_lp_dist_by_feat["base_pct"]
    )

    # 변수별 L-p norm distance 계산 (p=1, p=2, p=inf)
    df_lp_dist_by_feat = (
        df_lp_dist_by_feat.groupby(feat_index_col)["freq_diff"]
        .apply(
            lambda x: {
                "l1_dist": np.linalg.norm(x, ord=1),
                "l2_dist": np.linalg.norm(x, ord=2),
                "l_inf_dist": np.linalg.norm(x, ord=np.inf),
            }
        )
        .reset_index()
        .pivot(index=feat_index_col, columns="level_1", values="freq_diff")
        .reset_index()
    )

    df_lp_dist_by_feat.columns.name = None

    return df_lp_dist_by_feat


def calc_kld_by_feat(df_all_freq_by_feat, feat_index_col="feat_index"):
    """기준 및 대상 데이터 분포 정보 사용해 변수별 KL Divergence 산출

    대상 데이터를 P(x), 기준 데이터를 Q(x)로 설정하고 P(x) 대신 Q(x)를
    사용함으로서 일어나는 정보 손실인 D_KL(P||Q)를 계산함

    Args:
       df_all_freq_by_feat (pd.DataFrame): 변수별 기준 및 대상 데이터 구간 및 분포
       feat_index_col (str): 변수 인덱스 컬럼명

    Returns:
       pd.DataFrame: KL Divergence 계산 테이블
    """
    df_kld_by_feat = copy.deepcopy(df_all_freq_by_feat)

    # 구간별 KL divergence 산출
    df_kld_by_feat["kld_i"] = (
        df_kld_by_feat.groupby(feat_index_col)
        .apply(lambda x: rel_entr(x["target_pct"], x["base_pct"]))  # P(x)  # Q(x)
        .reset_index(drop=True)
    )

    return df_kld_by_feat


def calc_jsd_by_feat(df_all_freq_by_feat, feat_index_col="feat_index"):
    """기준 및 대상 데이터 분포 정보 사용해 변수별 JS Divergence 산출

    Args:
       df_all_freq_by_feat (pd.DataFrame): 변수별 기준 및 대상 데이터 구간 및 분포
       feat_index_col (str): 변수 인덱스 컬럼명

    Returns:
       pd.DataFrame: JS Divergence 계산 테이블
    """
    df_jsd_by_feat = copy.deepcopy(df_all_freq_by_feat)

    # P(x)와 Q(x)의 평균
    df_jsd_by_feat["mean_pct"] = df_jsd_by_feat[["base_pct", "target_pct"]].mean(axis=1)

    # 구간별 JS divergence 산출
    df_jsd_by_feat["jsd_i"] = (
        df_jsd_by_feat.groupby(feat_index_col)
        .apply(
            lambda x: (
                0.5
                * rel_entr(x["target_pct"], x["mean_pct"])  # D_KL(P||M)  # P(x)  # M
            )
            + (0.5 * rel_entr(x["base_pct"], x["mean_pct"]))  # D_KL(Q||M)  # Q(x)  # M
        )
        .reset_index(drop=True)
    )

    return df_jsd_by_feat


def perform_ks_test_by_feat(df_all_freq_by_feat, feat_index_col="feat_index"):
    """기준 및 대상 데이터 분포 정보 사용해 변수별 Two-Sample Kolmogorov-Smirnov
    검정 결과 생성

    구간별 누적 비율 차를 보여주는 DataFrame과 변수별 검정 결과를 요약해 보여주는
    DataFrame을 제공함

    Args:
       df_all_freq_by_feat (pd.DataFrame): 변수별 기준 및 대상 데이터 구간 및 분포
       feat_index_col (str): 변수 인덱스 컬럼명

    Returns:
       pd.DataFrame: 변수별 KS 계산 테이블
       pd.DataFrame: 변수별 KS 검정 결과 및 효과 크기
    """
    df_ks_by_feat = copy.deepcopy(df_all_freq_by_feat)

    # 구간별 누적 비율 계산
    df_ks_by_feat["base_cum_pct"] = (
        df_ks_by_feat.groupby(feat_index_col)["base_pct"]
        .expanding()
        .sum()
        .reset_index(drop=True)
    )
    df_ks_by_feat["target_cum_pct"] = (
        df_ks_by_feat.groupby(feat_index_col)["target_pct"]
        .expanding()
        .sum()
        .reset_index(drop=True)
    )
    df_ks_by_feat["cum_pct_diff"] = (
        df_ks_by_feat["target_cum_pct"] - df_ks_by_feat["base_cum_pct"]
    )

    # minS, maxS와 그 최대값인 D Statistic 계산
    df_ks_test_result = (
        df_ks_by_feat.groupby(feat_index_col)
        .apply(
            lambda x: pd.DataFrame(
                (np.clip(-np.min(x["cum_pct_diff"]), 0, 1), np.max(x["cum_pct_diff"]))
            ).T.rename(columns={0: "min_s", 1: "max_s"})
        )
        .reset_index()
        .drop(columns="level_1")
    )
    df_ks_test_result["d_statistic"] = df_ks_test_result[["min_s", "max_s"]].max(axis=1)

    # 표본 크기 관련 정보
    df_sample_info = (
        df_ks_by_feat.groupby(feat_index_col)[["base_freq", "target_freq"]]
        .sum()
        .rename(
            columns={"base_freq": "base_n_samples", "target_freq": "target_n_samples"}
        )
        .reset_index()
    )
    df_sample_info["en"] = (
        df_sample_info["base_n_samples"]
        * df_sample_info["target_n_samples"]
        / (df_sample_info["base_n_samples"] + df_sample_info["target_n_samples"])
    )
    df_ks_test_result = df_ks_test_result.merge(
        df_sample_info, how="left", on="feat_index"
    )

    # p-value 계산 (Smirnov's asymptoptic formula 사용)
    # - sf: survival function, 1-cdf와 거의 동일하나 값이 더 정확하게 계산되는 경우 있음
    # - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
    df_ks_test_result["p_value"] = df_ks_test_result.apply(
        lambda x: np.clip(kstwo.sf(x["d_statistic"], np.round(x["en"])), 0, 1), axis=1
    )
    STAT_TEST_SIG_LEVEL = 0.05  # significance level for statistical test
    df_ks_test_result["drift_status"] = np.where(
        df_ks_test_result["p_value"] < STAT_TEST_SIG_LEVEL, 1, 0
    )

    return df_ks_by_feat, df_ks_test_result


def perform_chi2_test_by_feat(df_all_freq_by_feat, feat_index_col="feat_index"):
    """기준 및 대상 데이터 분포 정보 사용해 변수별 Chi-Square 동질성 검정 결과 생성

    구간별 Chi-square 기여도 계산 과정을 상세하게 보여주는 DataFrame과,
    변수별 검정 결과를 요약해 보여주는 DataFrame을 모두 제공함

    Args:
       df_all_freq_by_feat (pd.DataFrame): 변수별 기준 및 대상 데이터 구간 및 분포
       feat_index_col (str): 변수 인덱스 컬럼명

    Returns:
       pd.DataFrame: 변수별 Chi-Square 계산 테이블
       pd.DataFrame: 변수별 Chi-Square 검정 결과 및 효과 크기
    """
    df_contingency = copy.deepcopy(df_all_freq_by_feat)

    # 변수별 contingency table 생성
    df_contingency = (
        df_contingency.groupby([feat_index_col, "bin_index"])[
            ["base_freq", "target_freq"]
        ]
        .sum()
        .reset_index()
    )

    ## [Deprecated]변수별 contingency table로 Chi-Square 검정 수행
    ## - 구간별 기여도를 계산하는 방식으로 변경함
    # df_contingency = df_contingency.groupby(feat_index_col).apply(
    #    lambda x: pd.DataFrame(
    #        chi2_contingency(
    #            x[['base_freq', 'target_freq']],
    #            correction=False
    #        )[:3],
    #        index=['chi2','p_value','dof']
    #    ).T
    # ).reset_index()[[feat_index_col, 'chi2', 'p_value', 'dof']]

    # 변수별로 구간별 Chi-square 기여도 계산
    df_chi2_by_feat = []
    for feat_index in df_contingency[feat_index_col].unique():
        df_feat_contingency = df_contingency[
            df_contingency[feat_index_col] == feat_index
        ].reset_index(drop=True)

        # observed frequencies
        O_ij = df_feat_contingency[["base_freq", "target_freq"]].values

        # sum by row and column
        N_i = O_ij.sum(axis=1)
        N_j = O_ij.sum(axis=0)
        N = N_j.sum()

        # 셀별 기대 빈도
        E_ij = np.array([N_i * N_j[i] / N for i in [0, 1]]).T

        # 셀별 Chi-square 기여도
        df_chi2 = pd.DataFrame(
            np.divide(np.subtract(O_ij, E_ij) ** 2, E_ij),
            columns=["base_chi2_i", "target_chi2_i"],
        )
        df_chi2["chi2_i"] = df_chi2["base_chi2_i"] + df_chi2["target_chi2_i"]

        df_chi2_by_feat.append(
            pd.concat(
                [
                    df_feat_contingency,
                    pd.DataFrame(
                        E_ij, columns=["base_exp_freq_i", "target_exp_freq_i"]
                    ),
                    df_chi2,
                ],
                axis=1,
            )
        )
    df_chi2_by_feat = pd.concat(df_chi2_by_feat).reset_index(drop=True)

    # 표본 크기 관련 정보
    df_sample_info = (
        df_contingency.groupby(feat_index_col)
        .agg(
            base_n_samples=("base_freq", "sum"), target_n_samples=("target_freq", "sum")
        )
        .reset_index()
        .merge(
            (
                df_contingency[
                    (df_contingency["base_freq"] != 0)
                    | (df_contingency["target_freq"] != 0)
                ]
                .groupby(feat_index_col)["bin_index"]
                .nunique()
                - 1
            )
            .rename("dof")
            .reset_index(),
            how="left",
            on=feat_index_col,
        )
    )
    df_sample_info["n_samples"] = (
        df_sample_info["base_n_samples"] + df_sample_info["target_n_samples"]
    )
    df_sample_info = df_sample_info.drop(columns=["base_n_samples", "target_n_samples"])

    # Chi-square 검정 결과 산출
    df_chi2_test_result = (
        df_chi2_by_feat.groupby(feat_index_col)["chi2_i"]
        .sum()
        .rename("chi2")
        .reset_index()
        .merge(df_sample_info, how="left", on=feat_index_col)
    )

    # p-value 계산
    # - sf: survival function, 1-cdf와 거의 동일하나 값이 더 정확하게 계산되는 경우 있음
    # - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
    df_chi2_test_result["p_value"] = df_chi2_test_result.apply(
        lambda x: chi2.sf(x["chi2"], x["dof"]), axis=1
    )
    STAT_TEST_SIG_LEVEL = 0.05  # significance level for statistical test
    df_chi2_test_result["drift_status"] = np.where(
        df_chi2_test_result["p_value"] < STAT_TEST_SIG_LEVEL, 1, 0
    )

    # 효과 크기 (Cramer's V) 산출
    df_chi2_test_result["cramers_v"] = np.sqrt(
        df_chi2_test_result["chi2"] / df_chi2_test_result["n_samples"]
    )

    return df_chi2_by_feat, df_chi2_test_result
