import pandas as pd
import numpy as np
from typing import List, Union
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from dython.nominal import associations

from drift_utils import (
    generate_freq_by_feat,
    calc_jsd_by_feat,
    calc_target_freq_by_feat,
)

warnings.filterwarnings("ignore")


def calc_max_jsd_f2r(df, key_col="발급회원번호", partition_col="is_syn", n_bins=20):
    """첫번째 파티션을 원본으로 보고, 나머지 파티션 별 데이터의 분포와 비교 (First to Rest)"""
    partitions = sorted(df[partition_col].unique())
    assert (
        len(partitions) > 1
    ), f"There must be at least 2 partitions for `{partition_col}` (Current: {len(partitions)})."
    # df_by_part = [df[df[partition_col]==part].pivot_table(index=key_col, columns=partition_col, aggfunc=lambda x: x, sort=False) for part in partitions]
    df_by_part = [
        df[df[partition_col] == part]
        .drop(columns=[key_col, partition_col])
        .reset_index(drop=True)
        for part in partitions
    ]

    # 값이 아예 생성되지 않은 컬럼 제외
    null_cols = []
    for df_part in df_by_part[1:]:
        null_cols = null_cols + [
            x[0]
            for x in df_by_part[0].columns
            if x[0] not in [y[0] for y in df_part.columns]
        ]
    for i, df_part in enumerate(df_by_part):
        for col in null_cols:
            if col in df_part:
                df_part = df_part.drop(columns=col)
        df_by_part[i] = df_part

    df_base_dist = generate_freq_by_feat(
        df_by_part[0], bin_method="equal_width", n_bins=n_bins
    )

    df_jsd = pd.concat(
        [
            calc_jsd_by_feat(calc_target_freq_by_feat(df_base_dist, x))
            .groupby("feat_index")["jsd_i"]
            .sum()
            .rename(partitions[i])
            for i, x in enumerate(df_by_part)
        ],
        axis=1,
    )
    # df_jsd.index = [x[0] for x in df_by_part[0].columns]
    df_jsd.index = df_by_part[0].columns

    return df_jsd.max(axis=1).to_dict()


def get_jsd_by_col(
    df_to_plot,
    key_col="발급회원번호",
    partition_col="is_syn",
    n_bins=20,
):
    """컬럼별 JSD 계산
    Args:
        df_to_plot (pd.DataFrame): 시각화 대상 컬럼 데이터
    """
    try:
        jsd_by_col = calc_max_jsd_f2r(
            df_to_plot, key_col=key_col, partition_col=partition_col, n_bins=n_bins
        )
        for col in [x for x in df_to_plot.columns if x not in [key_col, partition_col]]:
            if col not in jsd_by_col:  # 산출되지 않은 컬럼의 jsd는 -1로 설정
                jsd_by_col[col] = -1
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        jsd_by_col = {  # 오류 발생 시 모든 컬럼의 jsd를 -1로 설정
            x: -1 for x in df_to_plot.columns if x not in [key_col, partition_col]
        }

    return jsd_by_col


def get_jsd(
    df_merge: pd.DataFrame, key_col: str = "발급회원번호", partition_col: str = "is_syn"
) -> float:
    """jsd를 계산하기 위한 전처리 포함 전반적인 작업 수행
    Args:
        df_merge: 'partition_col' = [0,1] 로 구분 가능한 원본과 합성이 병합된 데이터
        key_col: 데이터의 pk 컬럼명
        partition_col: 원본과 합성의 구분 컬럼명
    Returns:
        float: JSD값
    """
    jsd_by_col = get_jsd_by_col(
        df_merge, key_col=key_col, partition_col=partition_col
    )

    jsd = np.mean(list(jsd_by_col.values()))
    return jsd


def calc_pmse(probs, y):
    N = len(y)
    r = y.sum() / N

    return ((probs - r) ** 2).sum() / N


def get_pmse(
    df_merge: pd.DataFrame,
    high_cardinality_cols: List[str] = None,
    partition_col: str = "is_syn",
) -> float:
    """pMSE를 계산하기 위한 샘플링부터 전처리까지 전반적인 작업 수행
    Args:
        df_merge: 'partition_col' = [0,1] 로 구분 가능한 원본과 합성이 병합된 데이터
        high_cardinality_cols: label enoding 수행할 매우 많은 범주수를 갖는 컬럼명들
        partition_col: 원본과 합성의 구분 컬럼명
    Returns:
        float: pMSE값
    """
    ## 동일 크기로 샘플링
    n1 = (df_merge[partition_col] == 0).sum()
    n2 = (df_merge[partition_col] == 1).sum()

    if n1 == n2:  # 동일 사이즈
        pass
    elif n1 > n2:  # 합성이 더 적음
        df_merge = pd.concat(
            [
                df_merge[df_merge[partition_col] == 0].sample(n=n2, random_state=0),
                df_merge[df_merge[partition_col] == 1],
            ]
        )
    else:  # 합성이 더 많음
        df_merge = pd.concat(
            [
                df_merge[df_merge[partition_col] == 0],
                df_merge[df_merge[partition_col] == 1].sample(n=n1, random_state=0),
            ]
        )

    X = df_merge.drop([partition_col], axis=1)
    # X = df_merge.drop([partition_col, BASE_YM_COL, KEY_COL], axis=1)
    y = df_merge[partition_col]

    ## 데이터 전처리
    # 범주형 처리
    # print("label enc")
    if high_cardinality_cols is None:
        high_cardinality_cols = []

    enc = LabelEncoder()
    for col in high_cardinality_cols:
        # print(col, X[col].dtype, X[col].unique())
        X[col] = enc.fit_transform(X[col])

    # 결측치 대체
    # print("null impute")
    X_imp = X.copy()
    # 숫자형 컬럼의 결측치를 평균값으로 채우기
    for col in X_imp.select_dtypes(include=[np.number]).columns:
        X_imp[col].fillna(X_imp[col].mean(), inplace=True)
    # 문자열 컬럼의 결측치를 'empty'으로 채우기
    for col in X_imp.select_dtypes(include=[object]).columns:
        X_imp[col].fillna("empty", inplace=True)

    # 범주형 변수 더미화
    X_dummy = pd.get_dummies(X_imp)

    # print("start lr fitting")
    lr_model = LogisticRegression(max_iter=5000, random_state=0, n_jobs=5)
    lr_model.fit(X_dummy, y)
    lr_model_probs = lr_model.predict_proba(X_dummy)[:, 1]

    pmse = calc_pmse(lr_model_probs, y)
    return pmse


def get_corrdiff(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    categorical_columns: Union[List[str], str] = "auto",
) -> float:
    """corr_diff의 계산하여 모든 컬럼쌍의 평균값을 출력
    수치형은 피어슨 상관계수, 범주형은 Theil'U 값을 계산
    Args:
        real: 원본데이터
        fake: 합성데이터
        categorical_columns: 범주형 컬럼명 리스트
    Returns:
        float: corr_diff의 평균값
    """
    real_corr = associations(
        real,
        nominal_columns=categorical_columns,
        nom_nom_assoc="theil",
        compute_only=True
    )
    fake_corr = associations(
        fake,
        nominal_columns=categorical_columns,
        nom_nom_assoc="theil",
        compute_only=True
    )

    corr_dist = real_corr["corr"] - fake_corr["corr"]
    corr_dist = np.abs(corr_dist.values).mean()

    return corr_dist
