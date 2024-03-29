import copy
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


def get_col_lists(col_info, min_nuniq_for_num=11, max_nuniq_for_cate=10):
    """컬럼 정보 정합성 체크 후 유형별 컬럼 목록 반환
    Args:
        col_info (dict): 컬럼 정보
        min_nuniq_for_num (int):
            수치형이 되기 위한 최소 고유값 수
            (해당 값보다 고유값이 적은 수치형은 categorical로 변경됨)
        max_nuniq_for_cate (int):
            범주형으로 취급할 최대 고유값 수
            (해당 값보다 고유값이 많은 범주형은 non_categorical로 변경됨)
    Returns:
        dict: CTABGAN+ 입력용 컬럼 목록
    """
    assert (
        min_nuniq_for_num <= max_nuniq_for_cate + 1
    ), "min_nuniq_for_num must be <= max_nuniq_for_cate + 1."

    def check_cond(cond, err_msg):
        if cond:
            return 1 - int(cond)
        else:
            print(f"Warning: {err_msg}")
            return 1 - int(cond)

    n_violations = 0
    for col, props in col_info.items():
        col_prop = props["properties"]

        # categorical인 경우
        if col_prop["is_categorical"] == 1:
            subject = f"Categorical column (`{col}`)"
            n_violations += check_cond(
                col_prop["is_mixed"] == 0, f"{subject} cannot be mixed."
            )
            n_violations += check_cond(
                col_prop["mode"] is None, f"{subject} should not have mode."
            )
            n_violations += check_cond(
                col_prop["category"] is not None,
                f"{subject} must have a list of categories.",
            )

            n_violations += check_cond(
                col_prop["is_integer"] == 0, f"{subject} cannot be integer."
            )
            n_violations += check_cond(
                col_prop["is_log"] == 0, f"{subject} cannot be log."
            )
            n_violations += check_cond(
                col_prop["is_positive"] == 0, f"{subject} cannot be positive."
            )
            n_violations += check_cond(
                col_prop["is_skewed"] == 0, f"{subject} cannot be skewed."
            )

        # mixed인 경우
        elif col_prop["is_mixed"] == 1:
            subject = f"Mixed column (`{col}`)"
            n_violations += check_cond(
                col_prop["mode"] is not None, f"{subject} must have mode."
            )

            n_violations += check_cond(
                col_prop["category"] is None, f"{subject} should not have category."
            )
            n_violations += check_cond(
                col_prop["is_general"] == 0, f"{subject} should not be general."
            )

        # continuous인 경우
        else:
            subject = f"Continuous column (`{col}`)"
            n_violations += check_cond(
                col_prop["category"] is None, f"{subject} should not have category."
            )
            n_violations += check_cond(
                col_prop["mode"] is None, f"{subject} should not have mode."
            )

    if n_violations > 0:
        raise Exception(f"Column info is invalid, with {n_violations} violation(s).")

    # num2cate
    num2cate = list(
        filter(
            lambda x: (x[1]["properties"]["is_categorical"] == 0)
            & (x[1]["properties"]["nunique"] < min_nuniq_for_num),
            col_info.items(),
        )
    )
    num2cate = list(map(lambda x: x[0], num2cate))
    print(f"[num2cate]: {len(num2cate)}")

    # cate
    _cate_tmp = list(
        filter(lambda x: x[1]["properties"]["is_categorical"] == 1, col_info.items())
    )
    cate_list = [k for k, v in _cate_tmp] + num2cate
    print(f"[cate]: {len(cate_list)}")

    # non_cate
    tmp = list(
        filter(lambda x: x[1]["properties"]["nunique"] > max_nuniq_for_cate, _cate_tmp)
    )
    non_cate_list = [k for k, v in tmp]
    print(f"[non_cate]: {len(non_cate_list)}")

    # log
    tmp = list(filter(lambda x: x[1]["properties"]["is_log"] == 1, col_info.items()))
    log_list = [k for k, v in tmp]
    print(f"[log]: {len(log_list)}")

    # mixed
    tmp = list(filter(lambda x: x[1]["properties"]["is_mixed"] == 1, col_info.items()))
    mixed_dict = {k: v["properties"]["mode"] for k, v in tmp}
    print(f"[mixed]: {len(mixed_dict)}")

    # general
    tmp = list(
        filter(lambda x: x[1]["properties"]["is_general"] == 1, col_info.items())
    )
    general_list = [k for k, v in tmp]
    print(f"[general]: {len(general_list)}")

    # skewed
    tmp = list(filter(lambda x: x[1]["properties"]["is_skewed"] == 1, col_info.items()))
    skewed_list = [k for k, v in tmp]
    print(f"[skewed]: {len(skewed_list)}")

    # integer
    tmp = list(
        filter(lambda x: x[1]["properties"]["is_integer"] == 1, col_info.items())
    )
    integer_list = [k for k, v in tmp]
    print(f"[integer]: {len(integer_list)}")

    return {
        "categorical_columns": cate_list,
        "log_columns": log_list,
        "mixed_columns": mixed_dict,
        "general_columns": general_list,
        "skewed_columns": skewed_list,
        "non_categorical_columns": non_cate_list,
        "integer_columns": integer_list,
    }


def date_to_recency(
    df, col, is_past=True, special_val=None, replace_val=np.nan, base_dt_col="기준일자_dt"
):
    """날짜 컬럼을 Recency로 변환"""
    if is_past:
        new_col = np.where(
            df[col] == special_val,
            replace_val,
            np.where(
                pd.isnull(df[col]),
                np.nan,
                (
                    df[base_dt_col]
                    - df[(~pd.isnull(df[col])) & (df[col] != special_val)][col]
                    .astype(int)
                    .apply(lambda x: datetime.strptime(str(x), "%Y%m%d"))
                ).dt.days,
            ),
        )
    else:
        new_col = np.where(
            df[col] == special_val,
            replace_val,
            np.where(
                pd.isnull(df[col]),
                np.nan,
                (
                    df[(~pd.isnull(df[col])) & (df[col] != special_val)][col]
                    .astype(int)
                    .apply(
                        lambda x: datetime.strptime(str(x), "%Y%m")
                        + relativedelta(months=1)
                        - relativedelta(days=1)
                    )
                    - df[base_dt_col]
                ).dt.days,
            ),
        )
    return new_col


def recency_to_date(
    df,
    new_col,
    is_past=True,
    special_val=None,
    replace_val=np.nan,
    base_dt_col="기준일자_dt",
):
    """Recency 컬럼을 날짜 값으로 변환"""
    if is_past:
        col = np.where(
            df[new_col] == replace_val,
            special_val,
            np.where(
                pd.isnull(df[new_col]),
                np.nan,
                df.apply(
                    lambda x: datetime.strftime(
                        x[base_dt_col] - relativedelta(days=x[new_col]), "%Y%m%d"
                    )
                    if (~pd.isnull(x[new_col])) & (x[new_col] != replace_val)
                    else "tmp",
                    axis=1,
                ),
            ),
        )
    else:
        col = np.where(
            df[new_col] == replace_val,
            special_val,
            np.where(
                pd.isnull(df[new_col]),
                np.nan,
                df.apply(
                    lambda x: datetime.strftime(
                        x[base_dt_col] + relativedelta(days=x[new_col]), "%Y%m"
                    )
                    if (~pd.isnull(x[new_col])) & (x[new_col] != replace_val)
                    else "tmp",
                    axis=1,
                ),
            ),
        )
    return col


def transform_date_cols(
    df_to_transform, date_replace_dict, base_ym_col="기준년월", inverse=False
):
    """날짜 컬럼을 지정된 규칙에 따라 Recency 컬럼으로 변환, 또는 그 역변환
    Recency 컬럼은 과거인 경우 `경과일수_`, 미래인 경우 `잔여일수_`를 날짜 컬럼명에 prefix로 붙임
    변환된 컬럼은 데이터 맨 뒤에 추가되며, 기존 컬럼은 제거됨
    Args:
        df (pd.DataFrame): 변환 대상 데이터
        date_replace_dict (dict): 날짜 컬럼별 변환 설정 정보
        base_ym_col (str): 기준년월 컬럼명
        inverse (bool): 역변환 여부
    Returns:
        pd.DataFrame: 변환 적용된 데이터
    """
    df = copy.deepcopy(df_to_transform)

    # 산출 기준일자 (기준년월의 말일)
    base_dt_col = "기준일자_dt"
    df[base_dt_col] = df[base_ym_col].apply(
        lambda x: datetime.strptime(str(x), "%Y%m")
        + relativedelta(months=1)
        - relativedelta(days=1)
    )
    if inverse == False:
        for col in date_replace_dict:
            new_col = (
                f"경과일수_{col}"
                if date_replace_dict[col]["is_past"] == True
                else f"잔여일수_{col}".replace("__", "_")
            )
            df[new_col] = date_to_recency(
                df,
                col,
                is_past=date_replace_dict[col]["is_past"],
                special_val=date_replace_dict[col]["special_val"],
                replace_val=date_replace_dict[col]["replace_val"],
                base_dt_col=base_dt_col,
            )
        df = df.drop(columns=[base_dt_col] + list(date_replace_dict.keys()))
    else:
        new_cols = []
        for col in date_replace_dict:
            new_col = (
                f"경과일수_{col}"
                if date_replace_dict[col]["is_past"] == True
                else f"잔여일수_{col}".replace("__", "_")
            )
            new_cols.append(new_col)
            df[col] = recency_to_date(
                df,
                new_col,
                is_past=date_replace_dict[col]["is_past"],
                special_val=date_replace_dict[col]["special_val"],
                replace_val=date_replace_dict[col]["replace_val"],
                base_dt_col=base_dt_col,
            )
        df = df.drop(columns=[base_dt_col] + new_cols)
    return df


def make_fn_dept(data: dict) -> (list, dict):
    """
    제약조건 데이터로부터 선후행 조건에 따라
    수행 역순의 함수 이름 목록과 수행되지 못한 제약 조건을 생성합니다.
    Args:
        data (dict): 제약 조건을 나타내는 딕셔너리들의 리스트입니다.
    Returns:
        tuple: 함수 실행 순서의 역순으로 정렬된 함수 이름 목록과 남은 제약 조건이 포함된 튜플을 반환합니다.
    """

    def process_data(cp_data: dict) -> (list, dict):
        """
        제약 조건 데이터로부터 정보를 추출하고 처리합니다.
        Args:
            cp_data (dict): 제약 조건을 담고 있는 딕셔너리입니다.
        Returns:
            tuple: 함수 이름 목록과 수정된 제약 조건 데이터가 포함된 튜플을 반환합니다.
        """
        all_output = set()
        all_input = set()

        # 제약 조건에서 columns set 추출
        for con in cp_data:
            all_output.add(con["output"])
            all_input.update(con["columns"])

        # 선행이 필요없는 칼럼명 추출
        lv0 = list(all_output - all_input)
        fun_names = []
        new_data = []
        for l in lv0:
            desired_dict = next(item for item in cp_data if item["output"] == l)
            # 추출된 칼럼으로 부터 fname을 추출하고 제약조건에서 삭제
            fun_names.append(desired_dict["fname"])
            cp_data.remove(desired_dict)
        return fun_names, cp_data

    if len(data) <= 1:
        return [], data
    # 현재 제약 조건 데이터를 처리하고 다음 레벨의 결과를 재귀 수행
    fun_names, new_data = process_data(data.copy())
    next_fun_names, next_data = make_fn_dept(new_data)
    # 현재 레벨의 함수 이름 목록과 다음 레벨 결과를 결합하여 반환
    return [fun_names] + next_fun_names, next_data
