# 1) 이용건수_할부_R6M <= 이용건수_할부_R12M
# 2) IF 강제한도감액횟수_R12M >0: 강제한도감액후경과월 IS NOT NULL
# 3) 유효카드수_신용체크 = 유효카드수_신용 + 유효카드수_체크

import pandas as pd
import numpy as np
from typing import Union, List
from functools import wraps


df = pd.read_csv("./Real_datasets/master_sample_10000.csv", delimiter="\t")
df = df.iloc[:100]
df.shape


prefix_docstr = """\
컬럼간 제약조건/파생수식 체크하는 함수.
입력 데이터와 출력 길이는 같음.\
"""


def constraint_udf(func):
    @wraps(func)
    def wrapper(df: pd.DataFrame):
        # 함수 수행
        ret = func(df)
        # 유효성체크 여기서
        assert len(ret) == len(df)
        return ret

    # docstring 추가
    wrapper.__doc__ = prefix_docstr + "\n" + func.__doc__
    return wrapper


constraints = [
    {
        "columns": ["이용건수_할부_R6M", "이용건수_할부_R12M"],
        "fname": "cc_00001",
        "type": "constraint",
        "content": "이용건수_할부_R6M <= 이용건수_할부_R12M",
    },
    {
        "columns": ["강제한도감액후경과월", "강제한도감액횟수_R12M"],
        "fname": "cc_00002",
        "type": "constraint",
        "content": "IF 강제한도감액횟수_R12M >0: 강제한도감액후경과월 IS NOT NULL",
    },
    {
        "columns": ["이용건수_할부_R6M", "이용건수_할부_R12M"],
        "fname": "cf_00001",
        "type": "formula",
        "content": "이용건수_할부_R6M <= 이용건수_할부_R12M",
    },
]

# --------- constraint/formula 함수 정의 ---------
# cc: check constraint
# cf: check formula


@constraint_udf
def cc_00001(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용건수_할부_R6M <= 이용건수_할부_R12M
    """
    r6m, r12m = df["이용건수_할부_R6M"], df["이용건수_할부_R12M"]  # pd.Series
    return r6m <= r12m


@constraint_udf
def cc_00002(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF 강제한도감액횟수_R12M >0: 강제한도감액후경과월 IS NOT NULL
    """
    dd = df[["강제한도감액후경과월", "강제한도감액횟수_R12M"]]
    ret = dd.apply(lambda x: not pd.isna(x[1]) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cf_00001(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        유효카드수_신용체크 = 유효카드수_신용 + 유효카드수_체크
    """
    c1, c2, c3 = df["유효카드수_신용"], df["유효카드수_체크"], df["유효카드수_신용체크"]
    return c3 == c1 + c2


cc_00001(df)
cc_00002(df)
cf_00001(df)
