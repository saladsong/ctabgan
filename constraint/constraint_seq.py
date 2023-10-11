# 컬럼 간 시계열 제약조건 / 파생수식 만족 여부 검증용
# (v.1) 6개월치 시계열 데이터 flat 하게 입력 시 발생하는 파생 관계
# for each constraint/formula fx,
# I: 전체 데이터프레임
# O: Boolean 컬럼

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Union, List
from functools import wraps


# df = pd.read_csv("./Real_datasets/master_sample_10000.csv", delimiter="\t")
# # df = pd.read_csv("master_sample_10000.csv", delimiter="\t")
# df = df.iloc[:100]
# df.shape


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


def isNaN(val):
    # NaN 검증용
    return (val is None) or (val != val)


seq_constraints = [
    # 1.회원 테이블 컬럼 Constraints
    # M08
    {
        "columns": ["M07_탈회횟수_누적", "M08_탈회횟수_누적"],
        "fname": "ccs_01_0115",
        "type": "constraint",
        "content": "M07_탈회횟수_누적 <= M08_탈회횟수_누적",
    },
    {
        "columns": ["M07_탈회횟수_발급6개월이내", "M08_탈회횟수_발급6개월이내"],
        "fname": "ccs_01_0117",
        "type": "constraint",
        "content": "M07_탈회횟수_발급6개월이내 <= M08_탈회횟수_발급6개월이내",
    },
    {
        "columns": ["M07_탈회횟수_발급1년이내", "M08_탈회횟수_발급1년이내"],
        "fname": "ccs_01_0118",
        "type": "constraint",
        "content": "M07_탈회횟수_발급1년이내 <= M08_탈회횟수_발급1년이내",
    },
    {
        "columns": ["M07_최종카드발급일자", "M08_최종카드발급일자"],
        "fname": "ccs_01_0154",
        "type": "constraint",
        "content": "M07_최종카드발급일자 <= M08_최종카드발급일자",
    },
    # M09
    {
        "columns": ["M08_탈회횟수_누적", "M09_탈회횟수_누적"],
        "fname": "ccs_01_0204",
        "type": "constraint",
        "content": "M08_탈회횟수_누적 <= M09_탈회횟수_누적",
    },
    {
        "columns": ["M08_탈회횟수_발급6개월이내", "M09_탈회횟수_발급6개월이내"],
        "fname": "ccs_01_0206",
        "type": "constraint",
        "content": "M08_탈회횟수_발급6개월이내 <= M09_탈회횟수_발급6개월이내",
    },
    {
        "columns": ["M08_탈회횟수_발급1년이내", "M09_탈회횟수_발급1년이내"],
        "fname": "ccs_01_0207",
        "type": "constraint",
        "content": "M08_탈회횟수_발급1년이내 <= M09_탈회횟수_발급1년이내",
    },
    {
        "columns": ["M08_최종카드발급일자", "M09_최종카드발급일자"],
        "fname": "ccs_01_0243",
        "type": "constraint",
        "content": "M08_최종카드발급일자 <= M09_최종카드발급일자",
    },
    # M10
    {
        "columns": ["M09_탈회횟수_누적", "M10_탈회횟수_누적"],
        "fname": "ccs_01_0293",
        "type": "constraint",
        "content": "M09_탈회횟수_누적 <= M10_탈회횟수_누적",
    },
    {
        "columns": ["M09_탈회횟수_발급6개월이내", "M10_탈회횟수_발급6개월이내"],
        "fname": "ccs_01_0295",
        "type": "constraint",
        "content": "M09_탈회횟수_발급6개월이내 <= M10_탈회횟수_발급6개월이내",
    },
    {
        "columns": ["M09_탈회횟수_발급1년이내", "M10_탈회횟수_발급1년이내"],
        "fname": "ccs_01_0296",
        "type": "constraint",
        "content": "M09_탈회횟수_발급1년이내 <= M10_탈회횟수_발급1년이내",
    },
    {
        "columns": ["M09_최종카드발급일자", "M10_최종카드발급일자"],
        "fname": "ccs_01_0332",
        "type": "constraint",
        "content": "M09_최종카드발급일자 <= M10_최종카드발급일자",
    },
    # M11
    {
        "columns": ["M10_탈회횟수_누적", "M11_탈회횟수_누적"],
        "fname": "ccs_01_0382",
        "type": "constraint",
        "content": "M10_탈회횟수_누적 <= M11_탈회횟수_누적",
    },
    {
        "columns": ["M10_탈회횟수_발급6개월이내", "M11_탈회횟수_발급6개월이내"],
        "fname": "ccs_01_0384",
        "type": "constraint",
        "content": "M10_탈회횟수_발급6개월이내 <= M11_탈회횟수_발급6개월이내",
    },
    {
        "columns": ["M10_탈회횟수_발급1년이내", "M11_탈회횟수_발급1년이내"],
        "fname": "ccs_01_0385",
        "type": "constraint",
        "content": "M10_탈회횟수_발급1년이내 <= M11_탈회횟수_발급1년이내",
    },
    {
        "columns": ["M10_최종카드발급일자", "M11_최종카드발급일자"],
        "fname": "ccs_01_0421",
        "type": "constraint",
        "content": "M10_최종카드발급일자 <= M11_최종카드발급일자",
    },
    # M12
    {
        "columns": ["M11_탈회횟수_누적", "M12_탈회횟수_누적"],
        "fname": "ccs_01_0471",
        "type": "constraint",
        "content": "M11_탈회횟수_누적 <= M12_탈회횟수_누적",
    },
    {
        "columns": ["M11_탈회횟수_발급6개월이내", "M12_탈회횟수_발급6개월이내"],
        "fname": "ccs_01_0473",
        "type": "constraint",
        "content": "M11_탈회횟수_발급6개월이내 <= M12_탈회횟수_발급6개월이내",
    },
    {
        "columns": ["M11_탈회횟수_발급1년이내", "M12_탈회횟수_발급1년이내"],
        "fname": "ccs_01_0474",
        "type": "constraint",
        "content": "M11_탈회횟수_발급1년이내 <= M12_탈회횟수_발급1년이내",
    },
    {
        "columns": ["M11_최종카드발급일자", "M12_최종카드발급일자"],
        "fname": "ccs_01_0510",
        "type": "constraint",
        "content": "M11_최종카드발급일자 <= M12_최종카드발급일자",
    },
    # 1.회원 테이블 컬럼 Formula
    # M08
    {
        "columns": ["M07_남녀구분코드"],
        "output": "M08_남녀구분코드",
        "fname": "cfs_01_0092",
        "type": "formula",
        "content": "M08_남녀구분코드 = M07_남녀구분코드",
    },
    {
        "columns": ["M07_연령"],
        "output": "M08_연령",
        "fname": "cfs_01_0093",
        "type": "formula",
        "content": "M08_연령 = M07_연령",
    },
    {
        "columns": ["M07_VIP등급코드"],
        "output": "M08_VIP등급코드",
        "fname": "cfs_01_0094",
        "type": "formula",
        "content": "M08_VIP등급코드 = M07_VIP등급코드",
    },
    {
        "columns": ["M07_최상위카드등급코드"],
        "output": "M08_최상위카드등급코드",
        "fname": "cfs_01_0095",
        "type": "formula",
        "content": "M08_최상위카드등급코드 = M07_최상위카드등급코드",
    },
    {
        "columns": ["M07_입회일자_신용"],
        "output": "M08_입회일자_신용",
        "fname": "cfs_01_0102",
        "type": "formula",
        "content": "M08_입회일자_신용 = M07_입회일자_신용",
    },
    {
        "columns": ["M07_입회경과개월수_신용"],
        "output": "M08_입회경과개월수_신용",
        "fname": "cfs_01_0103",
        "type": "formula",
        "content": "M08_입회경과개월수_신용 = M07_입회경과개월수_신용 + 1",
    },
    # {
    #     "columns": ["M07_유치경로코드_신용"],
    #     "output": "M08_유치경로코드_신용",
    #     "fname": "cfs_01_0101",
    #     "type": "formula",
    #     "content": "M08_유치경로코드_신용 = M07_유치경로코드_신용",
    # },
    # {
    #     "columns": ["M07_자사카드자격코드"],
    #     "output": "M08_자사카드자격코드",
    #     "fname": "cfs_01_0102",
    #     "type": "formula",
    #     "content": "M08_자사카드자격코드 = M07_자사카드자격코드",
    # },
    {
        "columns": ["M07_탈회횟수_누적", "M08_탈회횟수_누적", "M07_최종탈회후경과월"],
        "output": "M08_최종탈회후경과월",
        "fname": "cfs_01_0116",
        "type": "formula",
        "content": """IF (M08_탈회횟수_누적 > 0) & (M07_탈회횟수_누적 == M08_탈회횟수_누적)
                      THEN M08_최종탈회후경과월 = M07_최종탈회후경과월 + 1
                      ELSE M08_최종탈회후경과월 = 0""",
    },
    # M09
    {
        "columns": ["M07_남녀구분코드"],
        "output": "M09_남녀구분코드",
        "fname": "cfs_01_0181",
        "type": "formula",
        "content": "M09_남녀구분코드 = M07_남녀구분코드",
    },
    {
        "columns": ["M07_연령"],
        "output": "M09_연령",
        "fname": "cfs_01_0182",
        "type": "formula",
        "content": "M09_연령 = M07_연령",
    },
    {
        "columns": ["M07_VIP등급코드"],
        "output": "M09_VIP등급코드",
        "fname": "cfs_01_0183",
        "type": "formula",
        "content": "M09_VIP등급코드 = M07_VIP등급코드 ",
    },
    {
        "columns": ["M07_최상위카드등급코드"],
        "output": "M09_최상위카드등급코드",
        "fname": "cfs_01_0184",
        "type": "formula",
        "content": "M09_최상위카드등급코드 = M07_최상위카드등급코드",
    },
    {
        "columns": ["M07_입회일자_신용"],
        "output": "M09_입회일자_신용",
        "fname": "cfs_01_0191",
        "type": "formula",
        "content": "M09_입회일자_신용 = M07_입회일자_신용",
    },
    {
        "columns": ["M07_입회경과개월수_신용"],
        "output": "M09_입회경과개월수_신용",
        "fname": "cfs_01_0192",
        "type": "formula",
        "content": "M09_입회경과개월수_신용 = M07_입회경과개월수_신용 + 2",
    },
    # {
    #     "columns": ["M07_유치경로코드_신용"],
    #     "output": "M09_유치경로코드_신용",
    #     "fname": "cfs_01_0187",
    #     "type": "formula",
    #     "content": "M09_유치경로코드_신용 = M07_유치경로코드_신용",
    # },
    # {
    #     "columns": ["M07_자사카드자격코드"],
    #     "output": "M09_자사카드자격코드",
    #     "fname": "cfs_01_0188",
    #     "type": "formula",
    #     "content": "M09_자사카드자격코드 = M07_자사카드자격코드",
    # },
    {
        "columns": ["M08_탈회횟수_누적", "M09_탈회횟수_누적", "M08_최종탈회후경과월"],
        "output": "M09_최종탈회후경과월",
        "fname": "cfs_01_0205",
        "type": "formula",
        "content": """IF (M09_탈회횟수_누적 > 0) & (M08_탈회횟수_누적 == M09_탈회횟수_누적)
                      THEN M09_최종탈회후경과월 = M08_최종탈회후경과월 + 1
                      ELSE M09_최종탈회후경과월 = 0""",
    },
    # M10
    {
        "columns": ["M07_남녀구분코드"],
        "output": "M10_남녀구분코드",
        "fname": "cfs_01_0270",
        "type": "formula",
        "content": "M10_남녀구분코드 = M07_남녀구분코드",
    },
    {
        "columns": ["M07_연령"],
        "output": "M10_연령",
        "fname": "cfs_01_0271",
        "type": "formula",
        "content": "M10_연령 = M07_연령",
    },
    {
        "columns": ["M07_VIP등급코드"],
        "output": "M10_VIP등급코드",
        "fname": "cfs_01_0272",
        "type": "formula",
        "content": "M10_VIP등급코드 = M07_VIP등급코드 ",
    },
    {
        "columns": ["M07_최상위카드등급코드"],
        "output": "M10_최상위카드등급코드",
        "fname": "cfs_01_0273",
        "type": "formula",
        "content": "M10_최상위카드등급코드 = M07_최상위카드등급코드",
    },
    {
        "columns": ["M07_입회일자_신용"],
        "output": "M10_입회일자_신용",
        "fname": "cfs_01_0280",
        "type": "formula",
        "content": "M10_입회일자_신용 = M07_입회일자_신용",
    },
    {
        "columns": ["M07_입회경과개월수_신용"],
        "output": "M10_입회경과개월수_신용",
        "fname": "cfs_01_0281",
        "type": "formula",
        "content": "M10_입회경과개월수_신용 = M07_입회경과개월수_신용 + 3",
    },
    # {
    #     "columns": ["M07_유치경로코드_신용"],
    #     "output": "M10_유치경로코드_신용",
    #     "fname": "cfs_01_0273",
    #     "type": "formula",
    #     "content": "M10_유치경로코드_신용 = M07_유치경로코드_신용",
    # },
    # {
    #     "columns": ["M07_자사카드자격코드"],
    #     "output": "M10_자사카드자격코드",
    #     "fname": "cfs_01_0274",
    #     "type": "formula",
    #     "content": "M10_자사카드자격코드 = M07_자사카드자격코드",
    # },
    {
        "columns": ["M09_탈회횟수_누적", "M10_탈회횟수_누적", "M09_최종탈회후경과월"],
        "output": "M10_최종탈회후경과월",
        "fname": "cfs_01_0294",
        "type": "formula",
        "content": """IF (M10_탈회횟수_누적 > 0) & (M09_탈회횟수_누적 == M10_탈회횟수_누적)
                      THEN M10_최종탈회후경과월 = M09_최종탈회후경과월 + 1
                      ELSE M10_최종탈회후경과월 = 0""",
    },
    # M11
    {
        "columns": ["M07_남녀구분코드"],
        "output": "M11_남녀구분코드",
        "fname": "cfs_01_0359",
        "type": "formula",
        "content": "M11_남녀구분코드 = M07_남녀구분코드",
    },
    {
        "columns": ["M07_연령"],
        "output": "M11_연령",
        "fname": "cfs_01_0360",
        "type": "formula",
        "content": "M11_연령 = M07_연령",
    },
    {
        "columns": ["M07_VIP등급코드"],
        "output": "M11_VIP등급코드",
        "fname": "cfs_01_0361",
        "type": "formula",
        "content": "M11_VIP등급코드 = M07_VIP등급코드 ",
    },
    {
        "columns": ["M07_최상위카드등급코드"],
        "output": "M11_최상위카드등급코드",
        "fname": "cfs_01_0362",
        "type": "formula",
        "content": "M11_최상위카드등급코드 = M07_최상위카드등급코드",
    },
    {
        "columns": ["M07_입회일자_신용"],
        "output": "M11_입회일자_신용",
        "fname": "cfs_01_0369",
        "type": "formula",
        "content": "M11_입회일자_신용 = M07_입회일자_신용",
    },
    {
        "columns": ["M07_입회경과개월수_신용"],
        "output": "M11_입회경과개월수_신용",
        "fname": "cfs_01_0370",
        "type": "formula",
        "content": "M11_입회경과개월수_신용 = M07_입회경과개월수_신용 + 4",
    },
    # {
    #     "columns": ["M07_유치경로코드_신용"],
    #     "output": "M11_유치경로코드_신용",
    #     "fname": "cfs_01_0359",
    #     "type": "formula",
    #     "content": "M11_유치경로코드_신용 = M07_유치경로코드_신용",
    # },
    # {
    #     "columns": ["M07_자사카드자격코드"],
    #     "output": "M11_자사카드자격코드",
    #     "fname": "cfs_01_0360",
    #     "type": "formula",
    #     "content": "M11_자사카드자격코드 = M07_자사카드자격코드",
    # },
    {
        "columns": ["M10_탈회횟수_누적", "M11_탈회횟수_누적", "M10_최종탈회후경과월"],
        "output": "M11_최종탈회후경과월",
        "fname": "cfs_01_0383",
        "type": "formula",
        "content": """IF (M11_탈회횟수_누적 > 0) & (M10_탈회횟수_누적 == M11_탈회횟수_누적)
                      THEN M11_최종탈회후경과월 = M10_최종탈회후경과월 + 1
                      ELSE M11_최종탈회후경과월 = 0""",
    },
    # M12
    {
        "columns": ["M07_남녀구분코드"],
        "output": "M12_남녀구분코드",
        "fname": "cfs_01_0448",
        "type": "formula",
        "content": "M12_남녀구분코드 = M07_남녀구분코드",
    },
    {
        "columns": ["M07_연령"],
        "output": "M12_연령",
        "fname": "cfs_01_0449",
        "type": "formula",
        "content": "M12_연령 = M07_연령",
    },
    {
        "columns": ["M07_VIP등급코드"],
        "output": "M12_VIP등급코드",
        "fname": "cfs_01_0450",
        "type": "formula",
        "content": "M12_VIP등급코드 = M07_VIP등급코드 ",
    },
    {
        "columns": ["M07_최상위카드등급코드"],
        "output": "M12_최상위카드등급코드",
        "fname": "cfs_01_0451",
        "type": "formula",
        "content": "M12_최상위카드등급코드 = M07_최상위카드등급코드",
    },
    {
        "columns": ["M07_입회일자_신용"],
        "output": "M12_입회일자_신용",
        "fname": "cfs_01_0458",
        "type": "formula",
        "content": "M12_입회일자_신용 = M07_입회일자_신용",
    },
    {
        "columns": ["M07_입회경과개월수_신용"],
        "output": "M12_입회경과개월수_신용",
        "fname": "cfs_01_0459",
        "type": "formula",
        "content": "M12_입회경과개월수_신용 = M07_입회경과개월수_신용 + 5",
    },
    # {
    #     "columns": ["M07_유치경로코드_신용"],
    #     "output": "M12_유치경로코드_신용",
    #     "fname": "cfs_01_0445",
    #     "type": "formula",
    #     "content": "M12_유치경로코드_신용 = M07_유치경로코드_신용",
    # },
    # {
    #     "columns": ["M07_자사카드자격코드"],
    #     "output": "M12_자사카드자격코드",
    #     "fname": "cfs_01_0446",
    #     "type": "formula",
    #     "content": "M12_자사카드자격코드 = M07_자사카드자격코드",
    # },
    {
        "columns": ["M11_탈회횟수_누적", "M12_탈회횟수_누적", "M11_최종탈회후경과월"],
        "output": "M12_최종탈회후경과월",
        "fname": "cfs_01_0472",
        "type": "formula",
        "content": """IF (M12_탈회횟수_누적 > 0) & (M11_탈회횟수_누적 == M12_탈회횟수_누적)
                      THEN M12_최종탈회후경과월 = M11_최종탈회후경과월 + 1
                      ELSE M12_최종탈회후경과월 = 0""",
    },

    # 2.신용 테이블 컬럼 Constraints
    # M08
    {
        "columns": ["M07_RV신청일자", "M08_RV신청일자"],
        "fname": "ccs_02_0066",
        "type": "constraint",
        "content": "M07_RV신청일자 <= M08_RV신청일자",
    },
    # M09
    {
        "columns": ["M08_RV신청일자", "M09_RV신청일자"],
        "fname": "ccs_02_0120",
        "type": "constraint",
        "content": "M08_RV신청일자 <= M09_RV신청일자",
    },
    # M10
    {
        "columns": ["M09_RV신청일자", "M10_RV신청일자"],
        "fname": "ccs_02_0174",
        "type": "constraint",
        "content": "M09_RV신청일자 <= M10_RV신청일자",
    },
    # M11
    {
        "columns": ["M10_RV신청일자", "M11_RV신청일자"],
        "fname": "ccs_02_0228",
        "type": "constraint",
        "content": "M10_RV신청일자 <= M11_RV신청일자",
    },
    # M12
    {
        "columns": ["M11_RV신청일자", "M12_RV신청일자"],
        "fname": "ccs_02_0282",
        "type": "constraint",
        "content": "M11_RV신청일자 <= M12_RV신청일자",
    },
    # 2.신용 테이블 컬럼 Formula
    # M08
    {
        "columns": ["M07_최초한도금액"],
        "output": "M08_최초한도금액",
        "fname": "cfs_02_0057",
        "type": "formula",
        "content": "M08_최초한도금액 = M07_최초한도금액",
    },
    {
        "columns": ["M08_자발한도감액횟수_R12M", "M07_자발한도감액횟수_R12M", "M07_자발한도감액후경과월"],
        "output": "M08_자발한도감액후경과월",
        "fname": "cfs_02_0071",
        "type": "formula",
        "content": """IF M08_자발한도감액횟수_R12M == 0 THEN M08_자발한도감액후경과월 = 12
                      ELIF M08_자발한도감액횟수_R12M > M07_자발한도감액횟수_R12M THEN M08_자발한도감액후경과월 = 0
                      ELSE M08_자발한도감액후경과월 =  M07_자발한도감액후경과월 + 1""",
    },
    {
        "columns": ["M08_강제한도감액횟수_R12M", "M07_강제한도감액횟수_R12M", "M07_강제한도감액후경과월"],
        "output": "M08_강제한도감액후경과월",
        "fname": "cfs_02_0074",
        "type": "formula",
        "content": """IF M08_강제한도감액횟수_R12M == 0 THEN M08_강제한도감액후경과월 = 12
                      ELIF M08_강제한도감액횟수_R12M > M07_강제한도감액횟수_R12M THEN M08_강제한도감액후경과월 = 0
                      ELSE M08_강제한도감액후경과월 =  M07_강제한도감액후경과월 + 1""",
    },
    {
        "columns": ["M08_한도증액횟수_R12M", "M07_한도증액횟수_R12M", "M07_한도증액후경과월"],
        "output": "M08_한도증액후경과월",
        "fname": "cfs_02_0077",
        "type": "formula",
        "content": """IF M08_한도증액횟수_R12M == 0 THEN M08_한도증액후경과월 = 12
                      ELIF M08_한도증액횟수_R12M > M07_한도증액횟수_R12M THEN M08_한도증액후경과월 = 0
                      ELSE M08_한도증액후경과월 =  M07_한도증액후경과월 + 1""",
    },
    {
        "columns": ["M08_한도심사요청건수", "M07_한도심사요청건수", "M07_한도심사요청후경과월"],
        "output": "M08_한도심사요청후경과월",
        "fname": "cfs_02_0091",
        "type": "formula",
        "content": """IF M08_한도심사요청건수 == 0 THEN M08_한도심사요청후경과월 = 3
                      ELIF M08_한도심사요청건수 > M07_한도심사요청건수 THEN M08_한도심사요청후경과월 = 0
                      ELSE M08_한도심사요청후경과월 =  M07_한도심사요청후경과월 + 1""",
    },
    {
        "columns": ["M08_한도요청거절건수", "M07_한도요청거절건수", "M07_한도심사거절후경과월"],
        "output": "M08_한도심사거절후경과월",
        "fname": "cfs_02_0092",
        "type": "formula",
        "content": """IF M08_한도요청거절건수 == 0 THEN M08_한도심사거절후경과월 = 3
                      ELIF M08_한도요청거절건수 > M07_한도요청거절건수 THEN M08_한도심사거절후경과월 = 0
                      ELSE M08_한도심사거절후경과월 =  M07_한도심사거절후경과월 + 1""",
    },
    {
        "columns": ["M07_카드이용한도금액_B1M"],
        "output": "M08_카드이용한도금액_B2M",
        "fname": "cfs_02_0085",
        "type": "formula",
        "content": "M08_카드이용한도금액_B2M = M07_카드이용한도금액_B1M",
    },
    # M09
    {
        "columns": ["M07_최초한도금액"],
        "output": "M09_최초한도금액",
        "fname": "cfs_02_0111",
        "type": "formula",
        "content": "M09_최초한도금액 = M07_최초한도금액",
    },
    {
        "columns": ["M09_자발한도감액횟수_R12M", "M08_자발한도감액횟수_R12M", "M08_자발한도감액후경과월"],
        "output": "M09_자발한도감액후경과월",
        "fname": "cfs_02_0125",
        "type": "formula",
        "content": """IF M09_자발한도감액횟수_R12M == 0 THEN M09_자발한도감액후경과월 = 12
                      ELIF M09_자발한도감액횟수_R12M > M08_자발한도감액횟수_R12M THEN M09_자발한도감액후경과월 = 0
                      ELSE M09_자발한도감액후경과월 =  M08_자발한도감액후경과월 + 1""",
    },
    {
        "columns": ["M09_강제한도감액횟수_R12M", "M08_강제한도감액횟수_R12M", "M08_강제한도감액후경과월"],
        "output": "M09_강제한도감액후경과월",
        "fname": "cfs_02_0128",
        "type": "formula",
        "content": """IF M09_강제한도감액횟수_R12M == 0 THEN M09_강제한도감액후경과월 = 12
                      ELIF M09_강제한도감액횟수_R12M > M08_강제한도감액횟수_R12M THEN M09_강제한도감액후경과월 = 0
                      ELSE M09_강제한도감액후경과월 =  M08_강제한도감액후경과월 + 1""",
    },
    {
        "columns": ["M09_한도증액횟수_R12M", "M08_한도증액횟수_R12M", "M08_한도증액후경과월"],
        "output": "M09_한도증액후경과월",
        "fname": "cfs_02_0131",
        "type": "formula",
        "content": """IF M09_한도증액횟수_R12M == 0 THEN M09_한도증액후경과월 = 12
                      ELIF M09_한도증액횟수_R12M > M08_한도증액횟수_R12M THEN M09_한도증액후경과월 = 0
                      ELSE M09_한도증액후경과월 =  M08_한도증액후경과월 + 1""",
    },
    {
        "columns": ["M09_한도심사요청건수", "M08_한도심사요청건수", "M08_한도심사요청후경과월"],
        "output": "M09_한도심사요청후경과월",
        "fname": "cfs_02_0145",
        "type": "formula",
        "content": """IF M09_한도심사요청건수 == 0 THEN M09_한도심사요청후경과월 = 3
                      ELIF M09_한도심사요청건수 > M08_한도심사요청건수 THEN M09_한도심사요청후경과월 = 0
                      ELSE M09_한도심사요청후경과월 =  M08_한도심사요청후경과월 + 1""",
    },
    {
        "columns": ["M09_한도요청거절건수", "M08_한도요청거절건수", "M08_한도심사거절후경과월"],
        "output": "M09_한도심사거절후경과월",
        "fname": "cfs_02_0146",
        "type": "formula",
        "content": """IF M09_한도요청거절건수 == 0 THEN M09_한도심사거절후경과월 = 3
                      ELIF M09_한도요청거절건수 > M08_한도요청거절건수 THEN M09_한도심사거절후경과월 = 0
                      ELSE M09_한도심사거절후경과월 =  M08_한도심사거절후경과월 + 1""",
    },
    {
        "columns": ["M08_카드이용한도금액_B1M"],
        "output": "M09_카드이용한도금액_B2M",
        "fname": "cfs_02_0139",
        "type": "formula",
        "content": "M09_카드이용한도금액_B2M = M08_카드이용한도금액_B1M",
    },
    # M10
    {
        "columns": ["M07_최초한도금액"],
        "output": "M10_최초한도금액",
        "fname": "cfs_02_0165",
        "type": "formula",
        "content": "M10_최초한도금액 = M07_최초한도금액",
    },
    {
        "columns": ["M10_자발한도감액횟수_R12M", "M09_자발한도감액횟수_R12M", "M09_자발한도감액후경과월"],
        "output": "M10_자발한도감액후경과월",
        "fname": "cfs_02_0179",
        "type": "formula",
        "content": """IF M10_자발한도감액횟수_R12M == 0 THEN M10_자발한도감액후경과월 = 12
                      ELIF M10_자발한도감액횟수_R12M > M09_자발한도감액횟수_R12M THEN M10_자발한도감액후경과월 = 0
                      ELSE M10_자발한도감액후경과월 =  M09_자발한도감액후경과월 + 1""",
    },
    {
        "columns": ["M10_강제한도감액횟수_R12M", "M09_강제한도감액횟수_R12M", "M09_강제한도감액후경과월"],
        "output": "M10_강제한도감액후경과월",
        "fname": "cfs_02_0182",
        "type": "formula",
        "content": """IF M10_강제한도감액횟수_R12M == 0 THEN M10_강제한도감액후경과월 = 12
                      ELIF M10_강제한도감액횟수_R12M > M09_강제한도감액횟수_R12M THEN M10_강제한도감액후경과월 = 0
                      ELSE M10_강제한도감액후경과월 =  M09_강제한도감액후경과월 + 1""",
    },
    {
        "columns": ["M10_한도증액횟수_R12M", "M09_한도증액횟수_R12M", "M09_한도증액후경과월"],
        "output": "M10_한도증액후경과월",
        "fname": "cfs_02_0185",
        "type": "formula",
        "content": """IF M10_한도증액횟수_R12M == 0 THEN M10_한도증액후경과월 = 12
                      ELIF M10_한도증액횟수_R12M > M09_한도증액횟수_R12M THEN M10_한도증액후경과월 = 0
                      ELSE M10_한도증액후경과월 =  M09_한도증액후경과월 + 1""",
    },
    {
        "columns": ["M10_한도심사요청건수", "M09_한도심사요청건수", "M09_한도심사요청후경과월"],
        "output": "M10_한도심사요청후경과월",
        "fname": "cfs_02_0199",
        "type": "formula",
        "content": """IF M10_한도심사요청건수 == 0 THEN M10_한도심사요청후경과월 = 3
                      ELIF M10_한도심사요청건수 > M09_한도심사요청건수 THEN M10_한도심사요청후경과월 = 0
                      ELSE M10_한도심사요청후경과월 =  M09_한도심사요청후경과월 + 1""",
    },
    {
        "columns": ["M10_한도요청거절건수", "M09_한도요청거절건수", "M09_한도심사거절후경과월"],
        "output": "M10_한도심사거절후경과월",
        "fname": "cfs_02_0200",
        "type": "formula",
        "content": """IF M10_한도요청거절건수 == 0 THEN M10_한도심사거절후경과월 = 3
                      ELIF M10_한도요청거절건수 > M09_한도요청거절건수 THEN M10_한도심사거절후경과월 = 0
                      ELSE M10_한도심사거절후경과월 =  M09_한도심사거절후경과월 + 1""",
    },
    {
        "columns": ["M09_카드이용한도금액_B1M"],
        "output": "M10_카드이용한도금액_B2M",
        "fname": "cfs_02_0193",
        "type": "formula",
        "content": "M10_카드이용한도금액_B2M = M09_카드이용한도금액_B1M",
    },
    # M11
    {
        "columns": ["M07_최초한도금액"],
        "output": "M11_최초한도금액",
        "fname": "cfs_02_0219",
        "type": "formula",
        "content": "M11_최초한도금액 = M07_최초한도금액",
    },
    {
        "columns": ["M11_자발한도감액횟수_R12M", "M10_자발한도감액횟수_R12M", "M10_자발한도감액후경과월"],
        "output": "M11_자발한도감액후경과월",
        "fname": "cfs_02_0233",
        "type": "formula",
        "content": """IF M11_자발한도감액횟수_R12M == 0 THEN M11_자발한도감액후경과월 = 12
                      ELIF M11_자발한도감액횟수_R12M > M10_자발한도감액횟수_R12M THEN M11_자발한도감액후경과월 = 0
                      ELSE M11_자발한도감액후경과월 =  M10_자발한도감액후경과월 + 1""",
    },
    {
        "columns": ["M11_강제한도감액횟수_R12M", "M10_강제한도감액횟수_R12M", "M10_강제한도감액후경과월"],
        "output": "M11_강제한도감액후경과월",
        "fname": "cfs_02_0236",
        "type": "formula",
        "content": """IF M11_강제한도감액횟수_R12M == 0 THEN M11_강제한도감액후경과월 = 12
                      ELIF M11_강제한도감액횟수_R12M > M10_강제한도감액횟수_R12M THEN M11_강제한도감액후경과월 = 0
                      ELSE M11_강제한도감액후경과월 =  M10_강제한도감액후경과월 + 1""",
    },
    {
        "columns": ["M11_한도증액횟수_R12M", "M10_한도증액횟수_R12M", "M10_한도증액후경과월"],
        "output": "M11_한도증액후경과월",
        "fname": "cfs_02_0239",
        "type": "formula",
        "content": """IF M11_한도증액횟수_R12M == 0 THEN M11_한도증액후경과월 = 12
                      ELIF M11_한도증액횟수_R12M > M10_한도증액횟수_R12M THEN M11_한도증액후경과월 = 0
                      ELSE M11_한도증액후경과월 =  M10_한도증액후경과월 + 1""",
    },
    {
        "columns": ["M11_한도심사요청건수", "M10_한도심사요청건수", "M10_한도심사요청후경과월"],
        "output": "M11_한도심사요청후경과월",
        "fname": "cfs_02_0253",
        "type": "formula",
        "content": """IF M11_한도심사요청건수 == 0 THEN M11_한도심사요청후경과월 = 3
                      ELIF M11_한도심사요청건수 > M10_한도심사요청건수 THEN M11_한도심사요청후경과월 = 0
                      ELSE M11_한도심사요청후경과월 =  M10_한도심사요청후경과월 + 1""",
    },
    {
        "columns": ["M11_한도요청거절건수", "M10_한도요청거절건수", "M10_한도심사거절후경과월"],
        "output": "M11_한도심사거절후경과월",
        "fname": "cfs_02_0254",
        "type": "formula",
        "content": """IF M11_한도요청거절건수 == 0 THEN M11_한도심사거절후경과월 = 3
                      ELIF M11_한도요청거절건수 > M10_한도요청거절건수 THEN M11_한도심사거절후경과월 = 0
                      ELSE M11_한도심사거절후경과월 =  M10_한도심사거절후경과월 + 1""",
    },
    {
        "columns": ["M10_카드이용한도금액_B1M"],
        "output": "M11_카드이용한도금액_B2M",
        "fname": "cfs_02_0247",
        "type": "formula",
        "content": "M11_카드이용한도금액_B2M = M10_카드이용한도금액_B1M",
    },
    # M12
    {
        "columns": ["M07_최초한도금액"],
        "output": "M12_최초한도금액",
        "fname": "cfs_02_0273",
        "type": "formula",
        "content": "M12_최초한도금액 = M07_최초한도금액",
    },
    {
        "columns": ["M12_자발한도감액횟수_R12M", "M11_자발한도감액횟수_R12M", "M11_자발한도감액후경과월"],
        "output": "M12_자발한도감액후경과월",
        "fname": "cfs_02_0287",
        "type": "formula",
        "content": """IF M12_자발한도감액횟수_R12M == 0 THEN M12_자발한도감액후경과월 = 12
                      ELIF M12_자발한도감액횟수_R12M > M11_자발한도감액횟수_R12M THEN M12_자발한도감액후경과월 = 0
                      ELSE M12_자발한도감액후경과월 =  M11_자발한도감액후경과월 + 1""",
    },
    {
        "columns": ["M12_강제한도감액횟수_R12M", "M11_강제한도감액횟수_R12M", "M11_강제한도감액후경과월"],
        "output": "M12_강제한도감액후경과월",
        "fname": "cfs_02_0290",
        "type": "formula",
        "content": """IF M12_강제한도감액횟수_R12M == 0 THEN M12_강제한도감액후경과월 = 12
                      ELIF M12_강제한도감액횟수_R12M > M11_강제한도감액횟수_R12M THEN M12_강제한도감액후경과월 = 0
                      ELSE M12_강제한도감액후경과월 =  M11_강제한도감액후경과월 + 1""",
    },
    {
        "columns": ["M12_한도증액횟수_R12M", "M11_한도증액횟수_R12M", "M11_한도증액후경과월"],
        "output": "M12_한도증액후경과월",
        "fname": "cfs_02_0293",
        "type": "formula",
        "content": """IF M12_한도증액횟수_R12M == 0 THEN M12_한도증액후경과월 = 12
                      ELIF M12_한도증액횟수_R12M > M11_한도증액횟수_R12M THEN M12_한도증액후경과월 = 0
                      ELSE M12_한도증액후경과월 =  M11_한도증액후경과월 + 1""",
    },
    {
        "columns": ["M12_한도심사요청건수", "M11_한도심사요청건수", "M11_한도심사요청후경과월"],
        "output": "M12_한도심사요청후경과월",
        "fname": "cfs_02_0307",
        "type": "formula",
        "content": """IF M12_한도심사요청건수 == 0 THEN M12_한도심사요청후경과월 = 3
                      ELIF M12_한도심사요청건수 > M11_한도심사요청건수 THEN M12_한도심사요청후경과월 = 0
                      ELSE M12_한도심사요청후경과월 =  M11_한도심사요청후경과월 + 1""",
    },
    {
        "columns": ["M12_한도요청거절건수", "M11_한도요청거절건수", "M11_한도심사거절후경과월"],
        "output": "M12_한도심사거절후경과월",
        "fname": "cfs_02_0308",
        "type": "formula",
        "content": """IF M12_한도요청거절건수 == 0 THEN M12_한도심사거절후경과월 = 3
                      ELIF M12_한도요청거절건수 > M11_한도요청거절건수 THEN M12_한도심사거절후경과월 = 0
                      ELSE M12_한도심사거절후경과월 =  M11_한도심사거절후경과월 + 1""",
    },
    {
        "columns": ["M11_카드이용한도금액_B1M"],
        "output": "M12_카드이용한도금액_B2M",
        "fname": "cfs_02_0301",
        "type": "formula",
        "content": "M12_카드이용한도금액_B2M = M11_카드이용한도금액_B1M",
    },

    # 4.청구 테이블 컬럼 Formula
    # M09
    {
        "columns": ["M09_청구서발송여부_B0", "M08_청구서발송여부_B0", "M07_청구서발송여부_B0"],
        "output": "M09_청구서발송여부_R3M",
        "fname": "cfs_04_0112",
        "type": "formula",
        "content": "IF M07 & M08 & M09_청구서발송여부_B0 == '0' THEN M09_청구서발송여부_R3M = '0' ELSE '1'",
    },
    {
        "columns": ["M09_청구금액_B0", "M08_청구금액_B0", "M07_청구금액_B0"],
        "output": "M09_청구금액_R3M",
        "fname": "cfs_04_0115",
        "type": "formula",
        "content": "M09_청구금액_R3M = M09_청구금액_B0 + M08_청구금액_B0 + M07_청구금액_B0",
    },
    {
        "columns": ["M09_포인트_마일리지_건별_B0M", "M08_포인트_마일리지_건별_B0M", "M07_포인트_마일리지_건별_B0M"],
        "output": "M09_포인트_마일리지_건별_R3M",
        "fname": "cfs_04_0118",
        "type": "formula",
        "content": "M09_포인트_마일리지_건별_R3M = M09_포인트_마일리지_건별_B0M + M08_포인트_마일리지_건별_B0M + M07_포인트_마일리지_건별_B0M",
    },
    {
        "columns": ["M09_포인트_포인트_건별_B0M", "M08_포인트_포인트_건별_B0M", "M07_포인트_포인트_건별_B0M"],
        "output": "M09_포인트_포인트_건별_R3M",
        "fname": "cfs_04_0120",
        "type": "formula",
        "content": "M09_포인트_포인트_건별_R3M = M09_포인트_포인트_건별_B0M + M08_포인트_포인트_건별_B0M + M07_포인트_포인트_건별_B0M",
    },
    {
        "columns": ["M09_포인트_마일리지_월적립_B0M", "M08_포인트_마일리지_월적립_B0M", "M07_포인트_마일리지_월적립_B0M"],
        "output": "M09_포인트_마일리지_월적립_R3M",
        "fname": "cfs_04_0122",
        "type": "formula",
        "content": "M09_포인트_마일리지_월적립_R3M = M09_포인트_마일리지_월적립_B0M + M08_포인트_마일리지_월적립_B0M + M07_포인트_마일리지_월적립_B0M",
    },
    {
        "columns": ["M09_포인트_포인트_월적립_B0M", "M08_포인트_포인트_월적립_B0M", "M07_포인트_포인트_월적립_B0M"],
        "output": "M09_포인트_포인트_월적립_R3M",
        "fname": "cfs_04_0124",
        "type": "formula",
        "content": "M09_포인트_포인트_월적립_R3M = M09_포인트_포인트_월적립_B0M + M08_포인트_포인트_월적립_B0M + M07_포인트_포인트_건별_B0M",
    },
    # {
    #     "columns": ["M09_포인트_적립포인트_B0M", "M08_포인트_적립포인트_B0M", "M07_포인트_적립포인트_B0M"],
    #     "output": "M09_포인트_적립포인트_R3M",
    #     "fname": "cfs_04_0126",
    #     "type": "formula",
    #     "content": "M09_포인트_적립포인트_R3M = M09_포인트_적립포인트_B0M + M08_포인트_적립포인트_B0M + M07_포인트_적립포인트_B0M",
    # },
    # {
    #     "columns": ["M09_포인트_이용포인트_B0M", "M08_포인트_이용포인트_B0M", "M07_포인트_이용포인트_B0M"],
    #     "output": "M09_포인트_이용포인트_R3M",
    #     "fname": "cfs_04_0128",
    #     "type": "formula",
    #     "content": "M09_포인트_이용포인트_R3M = M09_포인트_이용포인트_B0M + M08_포인트_이용포인트_B0M + M07_포인트_이용포인트_B0M",
    # },
    # {
    #     "columns": ["M09_마일_적립포인트_B0M", "M08_마일_적립포인트_B0M", "M07_마일_적립포인트_B0M"],
    #     "output": "M09_마일_적립포인트_R3M",
    #     "fname": "cfs_04_0131",
    #     "type": "formula",
    #     "content": "M09_마일_적립포인트_R3M = M09_마일_적립포인트_B0M + M08_마일_적립포인트_B0M + M07_마일_적립포인트_B0M",
    # },
    # {
    #     "columns": ["M09_마일_이용포인트_B0M", "M08_마일_이용포인트_B0M", "M07_마일_이용포인트_B0M"],
    #     "output": "M09_마일_이용포인트_R3M",
    #     "fname": "cfs_04_0133",
    #     "type": "formula",
    #     "content": "M09_마일_이용포인트_R3M = M09_마일_이용포인트_B0M + M08_마일_이용포인트_B0M + M07_마일_이용포인트_B0M",
    # },
    {
        "columns": ["M09_할인건수_B0M", "M08_할인건수_B0M", "M07_할인건수_B0M"],
        "output": "M09_할인건수_R3M",
        "fname": "cfs_04_0135",
        "type": "formula",
        "content": "M09_할인건수_R3M = M09_할인건수_B0M + M08_할인건수_B0M + M07_할인건수_B0M",
    },
    {
        "columns": ["M09_할인금액_B0M", "M08_할인금액_B0M", "M07_할인금액_B0M"],
        "output": "M09_할인금액_R3M",
        "fname": "cfs_04_0136",
        "type": "formula",
        "content": "M09_할인금액_R3M = M09_할인금액_B0M + M08_할인금액_B0M + M07_할인금액_B0M",
    },
    {
        "columns": ["M09_할인금액_청구서_B0M", "M08_할인금액_청구서_B0M", "M07_할인금액_청구서_B0M"],
        "output": "M09_할인금액_청구서_R3M",
        "fname": "cfs_04_0139",
        "type": "formula",
        "content": "M09_할인금액_청구서_R3M = M09_할인금액_청구서_B0M + M08_할인금액_청구서_B0M + M07_할인금액_청구서_B0M",
    },
    {
        "columns": ["M09_혜택수혜금액", "M08_혜택수혜금액", "M07_혜택수혜금액"],
        "output": "M09_혜택수혜금액_R3M",
        "fname": "cfs_04_0141",
        "type": "formula",
        "content": "M09_혜택수혜금액_R3M = M09_혜택수혜금액 + M08_혜택수혜금액 + M07_혜택수혜금액",
    },
    # M10
    {
        "columns": ["M10_청구서발송여부_B0", "M09_청구서발송여부_B0", "M08_청구서발송여부_B0"],
        "output": "M10_청구서발송여부_R3M",
        "fname": "cfs_04_0163",
        "type": "formula",
        "content": "IF M08 & M09 & M10_청구서발송여부_B0 == '0' THEN M10_청구서발송여부_R3M = '0' ELSE '1'",
    },
    {
        "columns": ["M10_청구서발송여부_R3M", "M07_청구서발송여부_R3M"],
        "output": "M10_청구서발송여부_R6M",
        "fname": "cfs_04_0164",
        "type": "formula",
        "content": "IF M07 & M10_청구서발송여부_R3M == '0' THEN M10_청구서발송여부_R6M = '0' ELSE '1'",
    },
    {
        "columns": ["M10_청구금액_B0", "M09_청구금액_B0", "M08_청구금액_B0"],
        "output": "M10_청구금액_R3M",
        "fname": "cfs_04_0166",
        "type": "formula",
        "content": "M10_청구금액_R3M = M10_청구금액_B0 + M09_청구금액_B0 + M08_청구금액_B0",
    },
    {
        "columns": ["M10_청구금액_R3M", "M07_청구금액_R3M"],
        "output": "M10_청구금액_R6M",
        "fname": "cfs_04_0167",
        "type": "formula",
        "content": "M10_청구금액_R6M = M10_청구금액_R3M + M07_청구금액_R3M",
    },
    {
        "columns": ["M10_포인트_마일리지_건별_B0M", "M09_포인트_마일리지_건별_B0M", "M08_포인트_마일리지_건별_B0M"],
        "output": "M10_포인트_마일리지_건별_R3M",
        "fname": "cfs_04_0169",
        "type": "formula",
        "content": "M10_포인트_마일리지_건별_R3M = M10_포인트_마일리지_건별_B0M + M09_포인트_마일리지_건별_B0M + M08_포인트_마일리지_건별_B0M",
    },
    {
        "columns": ["M10_포인트_포인트_건별_B0M", "M09_포인트_포인트_건별_B0M", "M08_포인트_포인트_건별_B0M"],
        "output": "M10_포인트_포인트_건별_R3M",
        "fname": "cfs_04_0171",
        "type": "formula",
        "content": "M10_포인트_포인트_건별_R3M = M10_포인트_포인트_건별_B0M + M09_포인트_포인트_건별_B0M + M08_포인트_포인트_건별_B0M",
    },
    {
        "columns": ["M10_포인트_마일리지_월적립_B0M", "M09_포인트_마일리지_월적립_B0M", "M08_포인트_마일리지_월적립_B0M"],
        "output": "M10_포인트_마일리지_월적립_R3M",
        "fname": "cfs_04_0173",
        "type": "formula",
        "content": "M10_포인트_마일리지_월적립_R3M = M10_포인트_마일리지_월적립_B0M + M09_포인트_마일리지_월적립_B0M + M08_포인트_마일리지_월적립_B0M",
    },
    {
        "columns": ["M10_포인트_포인트_월적립_B0M", "M09_포인트_포인트_월적립_B0M", "M08_포인트_포인트_월적립_B0M"],
        "output": "M10_포인트_포인트_월적립_R3M",
        "fname": "cfs_04_0175",
        "type": "formula",
        "content": "M10_포인트_포인트_월적립_R3M = M10_포인트_포인트_월적립_B0M + M09_포인트_포인트_월적립_B0M + M08_포인트_포인트_건별_B0M",
    },
    # {
    #     "columns": ["M10_포인트_적립포인트_B0M", "M09_포인트_적립포인트_B0M", "M08_포인트_적립포인트_B0M"],
    #     "output": "M10_포인트_적립포인트_R3M",
    #     "fname": "cfs_04_0177",
    #     "type": "formula",
    #     "content": "M10_포인트_적립포인트_R3M = M10_포인트_적립포인트_B0M + M09_포인트_적립포인트_B0M + M08_포인트_적립포인트_B0M",
    # },
    # {
    #     "columns": ["M10_포인트_이용포인트_B0M", "M09_포인트_이용포인트_B0M", "M08_포인트_이용포인트_B0M"],
    #     "output": "M10_포인트_이용포인트_R3M",
    #     "fname": "cfs_04_0179",
    #     "type": "formula",
    #     "content": "M10_포인트_이용포인트_R3M = M10_포인트_이용포인트_B0M + M09_포인트_이용포인트_B0M + M08_포인트_이용포인트_B0M",
    # },
    # {
    #     "columns": ["M10_마일_적립포인트_B0M", "M09_마일_적립포인트_B0M", "M08_마일_적립포인트_B0M"],
    #     "output": "M10_마일_적립포인트_R3M",
    #     "fname": "cfs_04_0182",
    #     "type": "formula",
    #     "content": "M10_마일_적립포인트_R3M = M10_마일_적립포인트_B0M + M09_마일_적립포인트_B0M + M08_마일_적립포인트_B0M",
    # },
    # {
    #     "columns": ["M10_마일_이용포인트_B0M", "M09_마일_이용포인트_B0M", "M08_마일_이용포인트_B0M"],
    #     "output": "M10_마일_이용포인트_R3M",
    #     "fname": "cfs_04_0184",
    #     "type": "formula",
    #     "content": "M10_마일_이용포인트_R3M = M10_마일_이용포인트_B0M + M09_마일_이용포인트_B0M + M08_마일_이용포인트_B0M",
    # },
    {
        "columns": ["M10_할인건수_B0M", "M09_할인건수_B0M", "M08_할인건수_B0M"],
        "output": "M10_할인건수_R3M",
        "fname": "cfs_04_0186",
        "type": "formula",
        "content": "M10_할인건수_R3M = M10_할인건수_B0M + M09_할인건수_B0M + M08_할인건수_B0M",
    },
    {
        "columns": ["M10_할인금액_B0M", "M09_할인금액_B0M", "M08_할인금액_B0M"],
        "output": "M10_할인금액_R3M",
        "fname": "cfs_04_0187",
        "type": "formula",
        "content": "M10_할인금액_R3M = M10_할인금액_B0M + M09_할인금액_B0M + M08_할인금액_B0M",
    },
    {
        "columns": ["M10_할인금액_청구서_B0M", "M09_할인금액_청구서_B0M", "M08_할인금액_청구서_B0M"],
        "output": "M10_할인금액_청구서_R3M",
        "fname": "cfs_04_0190",
        "type": "formula",
        "content": "M10_할인금액_청구서_R3M = M10_할인금액_청구서_B0M + M09_할인금액_청구서_B0M + M08_할인금액_청구서_B0M",
    },
    {
        "columns": ["M10_혜택수혜금액", "M09_혜택수혜금액", "M08_혜택수혜금액"],
        "output": "M10_혜택수혜금액_R3M",
        "fname": "cfs_04_0192",
        "type": "formula",
        "content": "M10_혜택수혜금액_R3M = M10_혜택수혜금액 + M09_혜택수혜금액 + M08_혜택수혜금액",
    },
    # M11
    {
        "columns": ["M11_청구서발송여부_B0", "M10_청구서발송여부_B0", "M09_청구서발송여부_B0"],
        "output": "M11_청구서발송여부_R3M",
        "fname": "cfs_04_0214",
        "type": "formula",
        "content": "IF M09 & M10 & M11_청구서발송여부_B0 == '0'THEN M11_청구서발송여부_R3M = '0' ELSE '1'",
    },
    {
        "columns": ["M11_청구서발송여부_R3M", "M08_청구서발송여부_R3M"],
        "output": "M11_청구서발송여부_R6M",
        "fname": "cfs_04_0215",
        "type": "formula",
        "content": "IF M08 & M11_청구서발송여부_R3M == '0'THEN M11_청구서발송여부_R6M = '0' ELSE '1'",
    },
    {
        "columns": ["M11_청구금액_B0", "M10_청구금액_B0", "M09_청구금액_B0"],
        "output": "M11_청구금액_R3M",
        "fname": "cfs_04_0217",
        "type": "formula",
        "content": "M11_청구금액_R3M = M11_청구금액_B0 + M10_청구금액_B0 + M09_청구금액_B0",
    },
    {
        "columns": ["M11_청구금액_R3M", "M08_청구금액_R3M"],
        "output": "M11_청구금액_R6M",
        "fname": "cfs_04_0218",
        "type": "formula",
        "content": "M11_청구금액_R6M = M11_청구금액_R3M + M08_청구금액_R3M",
    },
    {
        "columns": ["M11_포인트_마일리지_건별_B0M", "M10_포인트_마일리지_건별_B0M", "M09_포인트_마일리지_건별_B0M"],
        "output": "M11_포인트_마일리지_건별_R3M",
        "fname": "cfs_04_0220",
        "type": "formula",
        "content": "M11_포인트_마일리지_건별_R3M = M11_포인트_마일리지_건별_B0M + M10_포인트_마일리지_건별_B0M + M09_포인트_마일리지_건별_B0M",
    },
    {
        "columns": ["M11_포인트_포인트_건별_B0M", "M10_포인트_포인트_건별_B0M", "M09_포인트_포인트_건별_B0M"],
        "output": "M11_포인트_포인트_건별_R3M",
        "fname": "cfs_04_0222",
        "type": "formula",
        "content": "M11_포인트_포인트_건별_R3M = M11_포인트_포인트_건별_B0M + M10_포인트_포인트_건별_B0M + M09_포인트_포인트_건별_B0M",
    },
    {
        "columns": ["M11_포인트_마일리지_월적립_B0M", "M10_포인트_마일리지_월적립_B0M", "M09_포인트_마일리지_월적립_B0M"],
        "output": "M11_포인트_마일리지_월적립_R3M",
        "fname": "cfs_04_0224",
        "type": "formula",
        "content": "M11_포인트_마일리지_월적립_R3M = M11_포인트_마일리지_월적립_B0M + M10_포인트_마일리지_월적립_B0M + M09_포인트_마일리지_월적립_B0M",
    },
    {
        "columns": ["M11_포인트_포인트_월적립_B0M", "M10_포인트_포인트_월적립_B0M", "M09_포인트_포인트_월적립_B0M"],
        "output": "M11_포인트_포인트_월적립_R3M",
        "fname": "cfs_04_0226",
        "type": "formula",
        "content": "M11_포인트_포인트_월적립_R3M = M11_포인트_포인트_월적립_B0M + M10_포인트_포인트_월적립_B0M + M09_포인트_포인트_건별_B0M",
    },
    # {
    #     "columns": ["M11_포인트_적립포인트_B0M", "M10_포인트_적립포인트_B0M", "M09_포인트_적립포인트_B0M"],
    #     "output": "M11_포인트_적립포인트_R3M",
    #     "fname": "cfs_04_0228",
    #     "type": "formula",
    #     "content": "M11_포인트_적립포인트_R3M = M11_포인트_적립포인트_B0M + M10_포인트_적립포인트_B0M + M09_포인트_적립포인트_B0M",
    # },
    # {
    #     "columns": ["M11_포인트_이용포인트_B0M", "M10_포인트_이용포인트_B0M", "M09_포인트_이용포인트_B0M"],
    #     "output": "M11_포인트_이용포인트_R3M",
    #     "fname": "cfs_04_0230",
    #     "type": "formula",
    #     "content": "M11_포인트_이용포인트_R3M = M11_포인트_이용포인트_B0M + M10_포인트_이용포인트_B0M + M09_포인트_이용포인트_B0M",
    # },
    # {
    #     "columns": ["M11_마일_적립포인트_B0M", "M10_마일_적립포인트_B0M", "M09_마일_적립포인트_B0M"],
    #     "output": "M11_마일_적립포인트_R3M",
    #     "fname": "cfs_04_0233",
    #     "type": "formula",
    #     "content": "M11_마일_적립포인트_R3M = M11_마일_적립포인트_B0M + M10_마일_적립포인트_B0M + M09_마일_적립포인트_B0M",
    # },
    # {
    #     "columns": ["M11_마일_이용포인트_B0M", "M10_마일_이용포인트_B0M", "M09_마일_이용포인트_B0M"],
    #     "output": "M11_마일_이용포인트_R3M",
    #     "fname": "cfs_04_0235",
    #     "type": "formula",
    #     "content": "M11_마일_이용포인트_R3M = M11_마일_이용포인트_B0M + M10_마일_이용포인트_B0M + M09_마일_이용포인트_B0M",
    # },
    {
        "columns": ["M11_할인건수_B0M", "M10_할인건수_B0M", "M09_할인건수_B0M"],
        "output": "M11_할인건수_R3M",
        "fname": "cfs_04_0237",
        "type": "formula",
        "content": "M11_할인건수_R3M = M11_할인건수_B0M + M10_할인건수_B0M + M09_할인건수_B0M",
    },
    {
        "columns": ["M11_할인금액_B0M", "M10_할인금액_B0M", "M09_할인금액_B0M"],
        "output": "M11_할인금액_R3M",
        "fname": "cfs_04_0238",
        "type": "formula",
        "content": "M11_할인금액_R3M = M11_할인금액_B0M + M10_할인금액_B0M + M09_할인금액_B0M",
    },
    {
        "columns": ["M11_할인금액_청구서_B0M", "M10_할인금액_청구서_B0M", "M09_할인금액_청구서_B0M"],
        "output": "M11_할인금액_청구서_R3M",
        "fname": "cfs_04_0241",
        "type": "formula",
        "content": "M11_할인금액_청구서_R3M = M11_할인금액_청구서_B0M + M10_할인금액_청구서_B0M + M09_할인금액_청구서_B0M",
    },
    {
        "columns": ["M11_혜택수혜금액", "M10_혜택수혜금액", "M09_혜택수혜금액"],
        "output": "M11_혜택수혜금액_R3M",
        "fname": "cfs_04_0243",
        "type": "formula",
        "content": "M11_혜택수혜금액_R3M = M11_혜택수혜금액 + M10_혜택수혜금액 + M09_혜택수혜금액",
    },
    # M12
    {
        "columns": ["M12_청구서발송여부_B0", "M11_청구서발송여부_B0", "M10_청구서발송여부_B0"],
        "output": "M12_청구서발송여부_R3M",
        "fname": "cfs_04_0265",
        "type": "formula",
        "content": "IF M10 & M11 & M12_청구서발송여부_B0 == '0'THEN M12_청구서발송여부_R3M = '0' ELSE '1'",
    },
    {
        "columns": ["M12_청구서발송여부_R3M", "M08_청구서발송여부_R3M"],
        "output": "M12_청구서발송여부_R6M",
        "fname": "cfs_04_0266",
        "type": "formula",
        "content": "IF M08 & M12_청구서발송여부_R3M == '0'THEN M12_청구서발송여부_R6M = '0' ELSE '1'",
    },
    {
        "columns": ["M12_청구금액_B0", "M11_청구금액_B0", "M10_청구금액_B0"],
        "output": "M12_청구금액_R3M",
        "fname": "cfs_04_0268",
        "type": "formula",
        "content": "M12_청구금액_R3M = M12_청구금액_B0 + M11_청구금액_B0 + M10_청구금액_B0",
    },
    {
        "columns": ["M12_청구금액_R3M", "M08_청구금액_R3M"],
        "output": "M12_청구금액_R6M",
        "fname": "cfs_04_0269",
        "type": "formula",
        "content": "M12_청구금액_R6M = M12_청구금액_R3M + M09_청구금액_R3M",
    },
    {
        "columns": ["M12_포인트_마일리지_건별_B0M", "M11_포인트_마일리지_건별_B0M", "M10_포인트_마일리지_건별_B0M"],
        "output": "M12_포인트_마일리지_건별_R3M",
        "fname": "cfs_04_0271",
        "type": "formula",
        "content": "M12_포인트_마일리지_건별_R3M = M12_포인트_마일리지_건별_B0M + M11_포인트_마일리지_건별_B0M + M10_포인트_마일리지_건별_B0M",
    },
    {
        "columns": ["M12_포인트_포인트_건별_B0M", "M11_포인트_포인트_건별_B0M", "M10_포인트_포인트_건별_B0M"],
        "output": "M12_포인트_포인트_건별_R3M",
        "fname": "cfs_04_0273",
        "type": "formula",
        "content": "M12_포인트_포인트_건별_R3M = M12_포인트_포인트_건별_B0M + M11_포인트_포인트_건별_B0M + M10_포인트_포인트_건별_B0M",
    },
    {
        "columns": ["M12_포인트_마일리지_월적립_B0M", "M11_포인트_마일리지_월적립_B0M", "M10_포인트_마일리지_월적립_B0M"],
        "output": "M12_포인트_마일리지_월적립_R3M",
        "fname": "cfs_04_0275",
        "type": "formula",
        "content": "M12_포인트_마일리지_월적립_R3M = M12_포인트_마일리지_월적립_B0M + M11_포인트_마일리지_월적립_B0M + M10_포인트_마일리지_월적립_B0M",
    },
    {
        "columns": ["M12_포인트_포인트_월적립_B0M", "M11_포인트_포인트_월적립_B0M", "M10_포인트_포인트_월적립_B0M"],
        "output": "M12_포인트_포인트_월적립_R3M",
        "fname": "cfs_04_0277",
        "type": "formula",
        "content": "M12_포인트_포인트_월적립_R3M = M12_포인트_포인트_월적립_B0M + M11_포인트_포인트_월적립_B0M + M10_포인트_포인트_건별_B0M",
    },
    # {
    #     "columns": ["M12_포인트_적립포인트_B0M", "M11_포인트_적립포인트_B0M", "M10_포인트_적립포인트_B0M"],
    #     "output": "M12_포인트_적립포인트_R3M",
    #     "fname": "cfs_04_0279",
    #     "type": "formula",
    #     "content": "M12_포인트_적립포인트_R3M = M12_포인트_적립포인트_B0M + M11_포인트_적립포인트_B0M + M10_포인트_적립포인트_B0M",
    # },
    # {
    #     "columns": ["M12_포인트_이용포인트_B0M", "M11_포인트_이용포인트_B0M", "M10_포인트_이용포인트_B0M"],
    #     "output": "M12_포인트_이용포인트_R3M",
    #     "fname": "cfs_04_0281",
    #     "type": "formula",
    #     "content": "M12_포인트_이용포인트_R3M = M12_포인트_이용포인트_B0M + M11_포인트_이용포인트_B0M + M10_포인트_이용포인트_B0M",
    # },
    # {
    #     "columns": ["M12_마일_적립포인트_B0M", "M11_마일_적립포인트_B0M", "M10_마일_적립포인트_B0M"],
    #     "output": "M12_마일_적립포인트_R3M",
    #     "fname": "cfs_04_0284",
    #     "type": "formula",
    #     "content": "M12_마일_적립포인트_R3M = M12_마일_적립포인트_B0M + M11_마일_적립포인트_B0M + M10_마일_적립포인트_B0M",
    # },
    # {
    #     "columns": ["M12_마일_이용포인트_B0M", "M11_마일_이용포인트_B0M", "M10_마일_이용포인트_B0M"],
    #     "output": "M12_마일_이용포인트_R3M",
    #     "fname": "cfs_04_0286",
    #     "type": "formula",
    #     "content": "M12_마일_이용포인트_R3M = M12_마일_이용포인트_B0M + M11_마일_이용포인트_B0M + M10_마일_이용포인트_B0M",
    # },
    {
        "columns": ["M12_할인건수_B0M", "M11_할인건수_B0M", "M10_할인건수_B0M"],
        "output": "M12_할인건수_R3M",
        "fname": "cfs_04_0288",
        "type": "formula",
        "content": "M12_할인건수_R3M = M12_할인건수_B0M + M11_할인건수_B0M + M10_할인건수_B0M",
    },
    {
        "columns": ["M12_할인금액_B0M", "M11_할인금액_B0M", "M10_할인금액_B0M"],
        "output": "M12_할인금액_R3M",
        "fname": "cfs_04_0289",
        "type": "formula",
        "content": "M12_할인금액_R3M = M12_할인금액_B0M + M11_할인금액_B0M + M10_할인금액_B0M",
    },
    {
        "columns": ["M12_할인금액_청구서_B0M", "M11_할인금액_청구서_B0M", "M10_할인금액_청구서_B0M"],
        "output": "M12_할인금액_청구서_R3M",
        "fname": "cfs_04_0292",
        "type": "formula",
        "content": "M12_할인금액_청구서_R3M = M12_할인금액_청구서_B0M + M11_할인금액_청구서_B0M + M10_할인금액_청구서_B0M",
    },
    {
        "columns": ["M12_혜택수혜금액", "M11_혜택수혜금액", "M10_혜택수혜금액"],
        "output": "M12_혜택수혜금액_R3M",
        "fname": "cfs_04_0294",
        "type": "formula",
        "content": "M12_혜택수혜금액_R3M = M12_혜택수혜금액 + M11_혜택수혜금액 + M10_혜택수혜금액",
    },

    # 5.잔액 테이블 컬럼 Formula
    # M07
    {
        "columns": ["M07_잔액_카드론_B0M"],
        "output": "M07_카드론잔액_최종경과월",
        "fname": "cfs_05_0013",
        "type": "formula",
        "content": "IF M07_잔액_카드론_B0M >0 THEN M07_카드론잔액_최종경과월 = 0 ELSE pass",
    },
    # M08
    {
        "columns": ["M08_잔액_카드론_B0M", "M07_잔액_카드론_B0M", "M07_카드론잔액_최종경과월"],
        "output": "M08_카드론잔액_최종경과월",
        "fname": "cfs_05_0114",
        "type": "formula",
        "content": """IF M08_잔액_카드론_B0M>0 THEN M08_카드론잔액_최종경과월 = 0
                      ELIF (M08_잔액_카드론_B0M==0 & M07_잔액_카드론_B0M==0) THEN M08_카드론잔액_최종경과월 = 0
                      ELIF (M08_잔액_카드론_B0M==0 & M07_잔액_카드론_B0M>0) THEN M08_카드론잔액_최종경과월 = 1
                      ELSE M08_카드론잔액_최종경과월 = M07_카드론잔액_최종경과월 + 1""",
    },
    {
        "columns": ["M07_잔액_현금서비스_B1M"],
        "output": "M08_잔액_현금서비스_B2M",
        "fname": "cfs_05_0123",
        "type": "formula",
        "content": "M08_잔액_현금서비스_B2M = M07_잔액_현금서비스_B1M",
    },
    {
        "columns": ["M07_잔액_카드론_B1M"],
        "output": "M08_잔액_카드론_B2M",
        "fname": "cfs_05_0125",
        "type": "formula",
        "content": "M08_잔액_카드론_B2M = M07_잔액_카드론_B1M",
    },
    {
        "columns": ["M07_잔액_카드론_B2M"],
        "output": "M08_잔액_카드론_B3M",
        "fname": "cfs_05_0126",
        "type": "formula",
        "content": "M08_잔액_카드론_B3M = M07_잔액_카드론_B2M",
    },
    {
        "columns": ["M07_잔액_카드론_B3M"],
        "output": "M08_잔액_카드론_B4M",
        "fname": "cfs_05_0127",
        "type": "formula",
        "content": "M08_잔액_카드론_B4M = M07_잔액_카드론_B3M",
    },
    {
        "columns": ["M07_잔액_카드론_B4M"],
        "output": "M08_잔액_카드론_B5M",
        "fname": "cfs_05_0128",
        "type": "formula",
        "content": "M08_잔액_카드론_B5M = M07_잔액_카드론_B4M",
    },
    {
        "columns": ["M07_잔액_할부_B1M"],
        "output": "M08_잔액_할부_B2M",
        "fname": "cfs_05_0130",
        "type": "formula",
        "content": "M08_잔액_할부_B2M = M07_잔액_할부_B1M",
    },
    {
        "columns": ["M07_잔액_일시불_B1M"],
        "output": "M08_잔액_일시불_B2M",
        "fname": "cfs_05_0132",
        "type": "formula",
        "content": "M08_잔액_일시불_B2M = M07_잔액_일시불_B1M",
    },
    {
        "columns": ["M07_연체일수_B1M"],
        "output": "M08_연체일수_B2M",
        "fname": "cfs_05_0134",
        "type": "formula",
        "content": "M08_연체일수_B2M = M07_연체일수_B1M",
    },
    {
        "columns": ["M07_연체원금_B1M"],
        "output": "M08_연체원금_B2M",
        "fname": "cfs_05_0136",
        "type": "formula",
        "content": "M08_연체원금_B2M = M07_연체원금_B1M",
    },
    # M09
    {
        "columns": ["M09_잔액_카드론_B0M", "M08_잔액_카드론_B0M", "M08_카드론잔액_최종경과월"],
        "output": "M09_카드론잔액_최종경과월",
        "fname": "cfs_05_0215",
        "type": "formula",
        "content": """IF M09_잔액_카드론_B0M>0 THEN M09_카드론잔액_최종경과월 = 0
                      ELIF (M09_잔액_카드론_B0M==0 & M08_잔액_카드론_B0M==0) THEN M09_카드론잔액_최종경과월 = 0
                      ELIF (M09_잔액_카드론_B0M==0 & M08_잔액_카드론_B0M>0) THEN M09_카드론잔액_최종경과월 = 1
                      ELSE M09_카드론잔액_최종경과월 = M08_카드론잔액_최종경과월 + 1""",
    },
    {
        "columns": ["M08_잔액_현금서비스_B1M"],
        "output": "M09_잔액_현금서비스_B2M",
        "fname": "cfs_05_0224",
        "type": "formula",
        "content": "M09_잔액_현금서비스_B2M = M08_잔액_현금서비스_B1M",
    },
    {
        "columns": ["M08_잔액_카드론_B1M"],
        "output": "M09_잔액_카드론_B2M",
        "fname": "cfs_05_0226",
        "type": "formula",
        "content": "M09_잔액_카드론_B2M = M08_잔액_카드론_B1M",
    },
    {
        "columns": ["M08_잔액_카드론_B2M"],
        "output": "M09_잔액_카드론_B3M",
        "fname": "cfs_05_0227",
        "type": "formula",
        "content": "M09_잔액_카드론_B3M = M08_잔액_카드론_B2M",
    },
    {
        "columns": ["M08_잔액_카드론_B3M"],
        "output": "M09_잔액_카드론_B4M",
        "fname": "cfs_05_0228",
        "type": "formula",
        "content": "M09_잔액_카드론_B4M = M08_잔액_카드론_B3M",
    },
    {
        "columns": ["M08_잔액_카드론_B4M"],
        "output": "M09_잔액_카드론_B5M",
        "fname": "cfs_05_0229",
        "type": "formula",
        "content": "M09_잔액_카드론_B5M = M08_잔액_카드론_B4M",
    },
    {
        "columns": ["M08_잔액_할부_B1M"],
        "output": "M09_잔액_할부_B2M",
        "fname": "cfs_05_0231",
        "type": "formula",
        "content": "M09_잔액_할부_B2M = M08_잔액_할부_B1M",
    },
    {
        "columns": ["M08_잔액_일시불_B1M"],
        "output": "M09_잔액_일시불_B2M",
        "fname": "cfs_05_0233",
        "type": "formula",
        "content": "M09_잔액_일시불_B2M = M08_잔액_일시불_B1M",
    },
    {
        "columns": ["M08_연체일수_B1M"],
        "output": "M09_연체일수_B2M",
        "fname": "cfs_05_0235",
        "type": "formula",
        "content": "M09_연체일수_B2M = M08_연체일수_B1M",
    },
    {
        "columns": ["M08_연체원금_B1M"],
        "output": "M09_연체원금_B2M",
        "fname": "cfs_05_0237",
        "type": "formula",
        "content": "M09_연체원금_B2M = M08_연체원금_B1M",
    },
    # M10
    {
        "columns": ["M10_잔액_카드론_B0M", "M09_잔액_카드론_B0M", "M09_카드론잔액_최종경과월"],
        "output": "M10_카드론잔액_최종경과월",
        "fname": "cfs_05_0316",
        "type": "formula",
        "content": """IF M10_잔액_카드론_B0M>0 THEN M10_카드론잔액_최종경과월 = 0
                      ELIF (M10_잔액_카드론_B0M==0 & M09_잔액_카드론_B0M==0) THEN M10_카드론잔액_최종경과월 = 0
                      ELIF (M10_잔액_카드론_B0M==0 & M09_잔액_카드론_B0M>0) THEN M10_카드론잔액_최종경과월 = 1
                      ELSE M10_카드론잔액_최종경과월 = M09_카드론잔액_최종경과월 + 1""",
    },
    {
        "columns": ["M09_잔액_현금서비스_B1M"],
        "output": "M10_잔액_현금서비스_B2M",
        "fname": "cfs_05_0325",
        "type": "formula",
        "content": "M10_잔액_현금서비스_B2M = M09_잔액_현금서비스_B1M",
    },
    {
        "columns": ["M09_잔액_카드론_B1M"],
        "output": "M10_잔액_카드론_B2M",
        "fname": "cfs_05_0327",
        "type": "formula",
        "content": "M10_잔액_카드론_B2M = M09_잔액_카드론_B1M",
    },
    {
        "columns": ["M09_잔액_카드론_B2M"],
        "output": "M10_잔액_카드론_B3M",
        "fname": "cfs_05_0328",
        "type": "formula",
        "content": "M10_잔액_카드론_B3M = M09_잔액_카드론_B2M",
    },
    {
        "columns": ["M09_잔액_카드론_B3M"],
        "output": "M10_잔액_카드론_B4M",
        "fname": "cfs_05_0329",
        "type": "formula",
        "content": "M10_잔액_카드론_B4M = M09_잔액_카드론_B3M",
    },
    {
        "columns": ["M09_잔액_카드론_B4M"],
        "output": "M10_잔액_카드론_B5M",
        "fname": "cfs_05_0330",
        "type": "formula",
        "content": "M10_잔액_카드론_B5M = M09_잔액_카드론_B4M",
    },
    {
        "columns": ["M09_잔액_할부_B1M"],
        "output": "M10_잔액_할부_B2M",
        "fname": "cfs_05_0332",
        "type": "formula",
        "content": "M10_잔액_할부_B2M = M09_잔액_할부_B1M",
    },
    {
        "columns": ["M09_잔액_일시불_B1M"],
        "output": "M10_잔액_일시불_B2M",
        "fname": "cfs_05_0334",
        "type": "formula",
        "content": "M10_잔액_일시불_B2M = M09_잔액_일시불_B1M",
    },
    {
        "columns": ["M09_연체일수_B1M"],
        "output": "M10_연체일수_B2M",
        "fname": "cfs_05_0336",
        "type": "formula",
        "content": "M10_연체일수_B2M = M09_연체일수_B1M",
    },
    {
        "columns": ["M09_연체원금_B1M"],
        "output": "M10_연체원금_B2M",
        "fname": "cfs_05_0338",
        "type": "formula",
        "content": "M10_연체원금_B2M = M09_연체원금_B1M",
    },
    {
        "columns": ["M10_RV_평균잔액_R3M", "M07_RV_평균잔액_R3M"],
        "output": "M10_RV_평균잔액_R6M",
        "fname": "cfs_05_0346",
        "type": "formula",
        "content": "M10_RV_평균잔액_R6M = avg(M10_RV_평균잔액_R3M, M07_RV_평균잔액_R3M)",
    },
    {
        "columns": ["M10_RV_최대잔액_R3M", "M07_RV_최대잔액_R3M"],
        "output": "M10_RV_최대잔액_R6M",
        "fname": "cfs_05_0347",
        "type": "formula",
        "content": "M10_RV_최대잔액_R6M = max(M10_RV_최대잔액_R3M, M07_RV_최대잔액_R3M)",
    },
    {
        "columns": ["M10_RV잔액이월횟수_R3M", "M07_RV잔액이월횟수_R3M"],
        "output": "M10_RV잔액이월횟수_R6M",
        "fname": "cfs_05_0357",
        "type": "formula",
        "content": "M10_RV잔액이월횟수_R6M = M10_RV잔액이월횟수_R3M + M07_RV잔액이월횟수_R3M",
    },
    # M11
    {
        "columns": ["M11_잔액_카드론_B0M", "M10_잔액_카드론_B0M", "M10_카드론잔액_최종경과월"],
        "output": "M11_카드론잔액_최종경과월",
        "fname": "cfs_05_0417",
        "type": "formula",
        "content": """IF M11_잔액_카드론_B0M>0 THEN M11_카드론잔액_최종경과월 = 0
                      ELIF (M11_잔액_카드론_B0M==0 & M10_잔액_카드론_B0M==0) THEN M11_카드론잔액_최종경과월 = 0
                      ELIF (M11_잔액_카드론_B0M==0 & M10_잔액_카드론_B0M>0) THEN M11_카드론잔액_최종경과월 = 1
                      ELSE M11_카드론잔액_최종경과월 = M10_카드론잔액_최종경과월 + 1""",
    },
    {
        "columns": ["M10_잔액_현금서비스_B1M"],
        "output": "M11_잔액_현금서비스_B2M",
        "fname": "cfs_05_0426",
        "type": "formula",
        "content": "M11_잔액_현금서비스_B2M = M10_잔액_현금서비스_B1M",
    },
    {
        "columns": ["M10_잔액_카드론_B1M"],
        "output": "M11_잔액_카드론_B2M",
        "fname": "cfs_05_0428",
        "type": "formula",
        "content": "M11_잔액_카드론_B2M = M10_잔액_카드론_B1M",
    },
    {
        "columns": ["M10_잔액_카드론_B2M"],
        "output": "M11_잔액_카드론_B3M",
        "fname": "cfs_05_0429",
        "type": "formula",
        "content": "M11_잔액_카드론_B3M = M10_잔액_카드론_B2M",
    },
    {
        "columns": ["M10_잔액_카드론_B3M"],
        "output": "M11_잔액_카드론_B4M",
        "fname": "cfs_05_0430",
        "type": "formula",
        "content": "M11_잔액_카드론_B4M = M10_잔액_카드론_B3M",
    },
    {
        "columns": ["M10_잔액_카드론_B4M"],
        "output": "M11_잔액_카드론_B5M",
        "fname": "cfs_05_0431",
        "type": "formula",
        "content": "M11_잔액_카드론_B5M = M10_잔액_카드론_B4M",
    },
    {
        "columns": ["M10_잔액_할부_B1M"],
        "output": "M11_잔액_할부_B2M",
        "fname": "cfs_05_0433",
        "type": "formula",
        "content": "M11_잔액_할부_B2M = M10_잔액_할부_B1M",
    },
    {
        "columns": ["M10_잔액_일시불_B1M"],
        "output": "M11_잔액_일시불_B2M",
        "fname": "cfs_05_0435",
        "type": "formula",
        "content": "M11_잔액_일시불_B2M = M10_잔액_일시불_B1M",
    },
    {
        "columns": ["M10_연체일수_B1M"],
        "output": "M11_연체일수_B2M",
        "fname": "cfs_05_0437",
        "type": "formula",
        "content": "M11_연체일수_B2M = M10_연체일수_B1M",
    },
    {
        "columns": ["M10_연체원금_B1M"],
        "output": "M11_연체원금_B2M",
        "fname": "cfs_05_0439",
        "type": "formula",
        "content": "M11_연체원금_B2M = M10_연체원금_B1M",
    },
    {
        "columns": ["M11_RV_평균잔액_R3M", "M08_RV_평균잔액_R3M"],
        "output": "M11_RV_평균잔액_R6M",
        "fname": "cfs_05_0447",
        "type": "formula",
        "content": "M11_RV_평균잔액_R6M = avg(M11_RV_평균잔액_R3M, M08_RV_평균잔액_R3M)",
    },
    {
        "columns": ["M11_RV_최대잔액_R3M", "M08_RV_최대잔액_R3M"],
        "output": "M11_RV_최대잔액_R6M",
        "fname": "cfs_05_0448",
        "type": "formula",
        "content": "M11_RV_최대잔액_R6M = max(M11_RV_최대잔액_R3M, M08_RV_최대잔액_R3M)",
    },
    {
        "columns": ["M11_RV잔액이월횟수_R3M", "M08_RV잔액이월횟수_R3M"],
        "output": "M11_RV잔액이월횟수_R6M",
        "fname": "cfs_05_0458",
        "type": "formula",
        "content": "M11_RV잔액이월횟수_R6M = M10_RV잔액이월횟수_R3M + M08_RV잔액이월횟수_R3M",
    },
    # M12
    {
        "columns": ["M12_잔액_카드론_B0M", "M11_잔액_카드론_B0M", "M11_카드론잔액_최종경과월"],
        "output": "M12_카드론잔액_최종경과월",
        "fname": "cfs_05_0518",
        "type": "formula",
        "content": """IF M12_잔액_카드론_B0M>0 THEN M12_카드론잔액_최종경과월 = 0
                      ELIF (M12_잔액_카드론_B0M==0 & M11_잔액_카드론_B0M==0) THEN M12_카드론잔액_최종경과월 = 0
                      ELIF (M12_잔액_카드론_B0M==0 & M11_잔액_카드론_B0M>0) THEN M12_카드론잔액_최종경과월 = 1
                      ELSE M12_카드론잔액_최종경과월 = M11_카드론잔액_최종경과월 + 1""",
    },
    {
        "columns": ["M11_잔액_현금서비스_B1M"],
        "output": "M12_잔액_현금서비스_B2M",
        "fname": "cfs_05_0527",
        "type": "formula",
        "content": "M12_잔액_현금서비스_B2M = M11_잔액_현금서비스_B1M",
    },
    {
        "columns": ["M11_잔액_카드론_B1M"],
        "output": "M12_잔액_카드론_B2M",
        "fname": "cfs_05_0529",
        "type": "formula",
        "content": "M12_잔액_카드론_B2M = M11_잔액_카드론_B1M",
    },
    {
        "columns": ["M11_잔액_카드론_B2M"],
        "output": "M12_잔액_카드론_B3M",
        "fname": "cfs_05_0530",
        "type": "formula",
        "content": "M12_잔액_카드론_B3M = M11_잔액_카드론_B2M",
    },
    {
        "columns": ["M11_잔액_카드론_B3M"],
        "output": "M12_잔액_카드론_B4M",
        "fname": "cfs_05_0531",
        "type": "formula",
        "content": "M12_잔액_카드론_B4M = M11_잔액_카드론_B3M",
    },
    {
        "columns": ["M11_잔액_카드론_B4M"],
        "output": "M12_잔액_카드론_B5M",
        "fname": "cfs_05_0532",
        "type": "formula",
        "content": "M12_잔액_카드론_B5M = M11_잔액_카드론_B4M",
    },
    {
        "columns": ["M11_잔액_할부_B1M"],
        "output": "M12_잔액_할부_B2M",
        "fname": "cfs_05_0534",
        "type": "formula",
        "content": "M12_잔액_할부_B2M = M11_잔액_할부_B1M",
    },
    {
        "columns": ["M11_잔액_일시불_B1M"],
        "output": "M12_잔액_일시불_B2M",
        "fname": "cfs_05_0536",
        "type": "formula",
        "content": "M12_잔액_일시불_B2M = M11_잔액_일시불_B1M",
    },
    {
        "columns": ["M11_연체일수_B1M"],
        "output": "M12_연체일수_B2M",
        "fname": "cfs_05_0538",
        "type": "formula",
        "content": "M12_연체일수_B2M = M11_연체일수_B1M",
    },
    {
        "columns": ["M11_연체원금_B1M"],
        "output": "M12_연체원금_B2M",
        "fname": "cfs_05_0540",
        "type": "formula",
        "content": "M12_연체원금_B2M = M11_연체원금_B1M",
    },
    {
        "columns": ["M12_RV_평균잔액_R3M", "M09_RV_평균잔액_R3M"],
        "output": "M12_RV_평균잔액_R6M",
        "fname": "cfs_05_0548",
        "type": "formula",
        "content": "M12_RV_평균잔액_R6M = avg(M12_RV_평균잔액_R3M, M09_RV_평균잔액_R3M)",
    },
    {
        "columns": ["M12_RV_최대잔액_R3M", "M09_RV_최대잔액_R3M"],
        "output": "M12_RV_최대잔액_R6M",
        "fname": "cfs_05_0549",
        "type": "formula",
        "content": "M12_RV_최대잔액_R6M = max(M12_RV_최대잔액_R3M, M09_RV_최대잔액_R3M)",
    },
    {
        "columns": ["M12_RV잔액이월횟수_R3M", "M09_RV잔액이월횟수_R3M"],
        "output": "M12_RV잔액이월횟수_R6M",
        "fname": "cfs_05_0559",
        "type": "formula",
        "content": "M12_RV잔액이월횟수_R6M = M11_RV잔액이월횟수_R3M + M09_RV잔액이월횟수_R3M",
    },

    # 6.채널활동 테이블 컬럼 Formula
    # M07
    {
        "columns": ["M07_인입횟수_ARS_B0M"],
        "output": "M07_인입후경과월_ARS",
        "fname": "cfs_06_0007",
        "type": "formula",
        "content": "IF M07_인입횟수_ARS_B0M >0 THEN M07_인입후경과월_ARS = 0 ELSE pass",
    },
    {
        "columns": ["M07_방문횟수_PC_B0M"],
        "output": "M07_방문후경과월_PC_R6M",
        "fname": "cfs_06_0014",
        "type": "formula",
        "content": "IF M07_방문횟수_PC_B0M >0 THEN M07_방문후경과월_PC_R6M = 0 ELSE pass",
    },
    {
        "columns": ["M07_방문횟수_앱_B0M"],
        "output": "M07_방문후경과월_앱_R6M",
        "fname": "cfs_06_0018",
        "type": "formula",
        "content": "IF M07_방문횟수_앱_B0M >0 THEN M07_방문후경과월_앱_R6M = 0 ELSE pass",
    },
    {
        "columns": ["M07_방문횟수_모바일웹_B0M"],
        "output": "M07_방문후경과월_모바일웹_R6M",
        "fname": "cfs_06_0022",
        "type": "formula",
        "content": "IF M07_방문횟수_모바일웹_B0M >0 THEN M07_방문후경과월_모바일웹_R6M = 0 ELSE pass",
    },
    {
        "columns": ["M07_인입횟수_IB_B0M"],
        "output": "M07_인입후경과월_IB_R6M",
        "fname": "cfs_06_0033",
        "type": "formula",
        "content": "IF M07_인입횟수_IB_B0M >0 THEN M07_인입후경과월_IB_R6M = 0 ELSE pass",
    },
    {
        "columns": ["M07_인입불만횟수_IB_B0M"],
        "output": "M07_인입불만후경과월_IB_R6M",
        "fname": "cfs_06_0040",
        "type": "formula",
        "content": "IF M07_인입불만횟수_IB_B0M >0 THEN M07_인입불만후경과월_IB_R6M = 0 ELSE pass",
    },
    {
        "columns": ["M07_불만제기건수_B0M"],
        "output": "M07_불만제기후경과월_R12M",
        "fname": "cfs_06_0092",
        "type": "formula",
        "content": "IF M07_불만제기건수_B0M >0 THEN M07_불만제기후경과월_R12M = 0 ELSE pass",
    },
    # M08
    {
        "columns": ["M08_인입횟수_ARS_B0M", "M07_인입횟수_ARS_B0M", "M07_인입후경과월_ARS"],
        "output": "M08_인입후경과월_ARS",
        "fname": "cfs_06_0114",
        "type": "formula",
        "content": """IF (M08_인입횟수_ARS_B0M>0) THEN M08_인입후경과월_ARS = 0
                      ELIF (M07_인입횟수_ARS_B0M==0)&(M07_인입후경과월_ARS==0) THEN M08_인입후경과월_ARS = 0
                      ELIF (M07_인입횟수_ARS_B0M>0)&(M07_인입후경과월_ARS==0) THEN M08_인입후경과월_ARS = 1
                      ELIF (M07_인입후경과월_ARS==5) THEN M08_인입후경과월_ARS = 0
                      ELIF M08_인입후경과월_ARS = M07_인입후경과월_ARS + 1""",
    },
    {
        "columns": ["M08_방문횟수_PC_B0M", "M07_방문후경과월_PC_R6M"],
        "output": "M08_방문후경과월_PC_R6M",
        "fname": "cfs_06_0121",
        "type": "formula",
        "content": """IF M08_방문횟수_PC_B0M > 0 THEN M08_방문후경과월_PC_R6M = 0
                      ELIF M08_방문후경과월_PC_R6M = min(M07_방문후경과월_PC_R6M + 1, 6)""",
    },
    {
        "columns": ["M08_방문횟수_앱_B0M", "M07_방문후경과월_앱_R6M"],
        "output": "M08_방문후경과월_앱_R6M",
        "fname": "cfs_06_0125",
        "type": "formula",
        "content": """IF M08_방문횟수_앱_B0M > 0 THEN M08_방문후경과월_앱_R6M = 0
                      ELIF M08_방문후경과월_앱_R6M = min(M07_방문후경과월_앱_R6M + 1, 6)""",
    },
    {
        "columns": ["M08_방문횟수_모바일웹_B0M", "M07_방문후경과월_모바일웹_R6M"],
        "output": "M08_방문후경과월_모바일웹_R6M",
        "fname": "cfs_06_0129",
        "type": "formula",
        "content": """IF M08_방문횟수_모바일웹_B0M > 0 THEN M08_방문후경과월_모바일웹_R6M = 0
                      ELIF M08_방문후경과월_모바일웹_R6M = min(M07_방문후경과월_모바일웹_R6M + 1, 6)""",
    },
    {
        "columns": ["M08_인입횟수_IB_B0M", "M07_인입후경과월_IB_R6M"],
        "output": "M08_인입후경과월_IB_R6M",
        "fname": "cfs_06_0140",
        "type": "formula",
        "content": """IF M08_인입횟수_IB_B0M > 0 THEN M08_인입후경과월_IB_R6M = 0
                      ELIF M08_인입후경과월_IB_R6M = min(M07_인입후경과월_IB_R6M + 1, 6)""",
    },
    {
        "columns": ["M08_인입불만횟수_IB_B0M", "M07_인입불만후경과월_IB_R6M"],
        "output": "M08_인입불만후경과월_IB_R6M",
        "fname": "cfs_06_0147",
        "type": "formula",
        "content": """IF M08_인입불만횟수_IB_B0M > 0 THEN M08_인입불만후경과월_IB_R6M = 0
                      ELIF M08_인입불만후경과월_IB_R6M = min(M07_인입불만후경과월_IB_R6M + 1, 6)""",
    },
    {
        "columns": ["M08_불만제기건수_B0M", "M07_불만제기건수_B0M", "M07_불만제기후경과월_R12M"],
        "output": "M08_불만제기후경과월_R12M",
        "fname": "cfs_06_0199",
        "type": "formula",
        "content": """IF M08_불만제기건수_B0M > 0 THEN M08_불만제기후경과월_R12M = 0
                      ELIF (M07_불만제기건수_B0M==0 & M07_불만제기후경과월_R12M==0) THEN M08_불만제기후경과월_R12M = 0
                      ELIF (M07_불만제기건수_B0M==0 & M07_불만제기후경과월_R12M==12) THEN M08_불만제기후경과월_R12M = 12
                      ELIF M08_불만제기후경과월_R12M = min(M07_불만제기후경과월_R12M + 1, 12)""",
    },
    # M09
    {
        "columns": ["M09_인입횟수_ARS_B0M", "M08_인입횟수_ARS_B0M", "M08_인입후경과월_ARS"],
        "output": "M09_인입후경과월_ARS",
        "fname": "cfs_06_0221",
        "type": "formula",
        "content": """IF (M09_인입횟수_ARS_B0M>0) THEN M09_인입후경과월_ARS = 0
                      ELIF (M08_인입횟수_ARS_B0M==0)&(M08_인입후경과월_ARS==0) THEN M09_인입후경과월_ARS = 0
                      ELIF (M08_인입횟수_ARS_B0M>0)&(M08_인입후경과월_ARS==0) THEN M09_인입후경과월_ARS = 1
                      ELIF (M08_인입후경과월_ARS==5) THEN M09_인입후경과월_ARS = 0
                      ELIF M09_인입후경과월_ARS = M08_인입후경과월_ARS + 1""",
    },
    {
        "columns": ["M09_방문횟수_PC_B0M", "M08_방문후경과월_PC_R6M"],
        "output": "M09_방문후경과월_PC_R6M",
        "fname": "cfs_06_0228",
        "type": "formula",
        "content": """IF M09_방문횟수_PC_B0M > 0 THEN M09_방문후경과월_PC_R6M = 0
                      ELIF M09_방문후경과월_PC_R6M = min(M08_방문후경과월_PC_R6M + 1, 6)""",
    },
    {
        "columns": ["M09_방문횟수_앱_B0M", "M08_방문후경과월_앱_R6M"],
        "output": "M09_방문후경과월_앱_R6M",
        "fname": "cfs_06_0232",
        "type": "formula",
        "content": """IF M09_방문횟수_앱_B0M > 0 THEN M09_방문후경과월_앱_R6M = 0
                      ELIF M09_방문후경과월_앱_R6M = min(M08_방문후경과월_앱_R6M + 1, 6)""",
    },
    {
        "columns": ["M09_방문횟수_모바일웹_B0M", "M08_방문후경과월_모바일웹_R6M"],
        "output": "M09_방문후경과월_모바일웹_R6M",
        "fname": "cfs_06_0236",
        "type": "formula",
        "content": """IF M09_방문횟수_모바일웹_B0M > 0 THEN M09_방문후경과월_모바일웹_R6M = 0
                      ELIF M09_방문후경과월_모바일웹_R6M = min(M08_방문후경과월_모바일웹_R6M + 1, 6)""",
    },
    {
        "columns": ["M09_인입횟수_IB_B0M", "M08_인입후경과월_IB_R6M"],
        "output": "M09_인입후경과월_IB_R6M",
        "fname": "cfs_06_0247",
        "type": "formula",
        "content": """IF M09_인입횟수_IB_B0M > 0 THEN M09_인입후경과월_IB_R6M = 0
                      ELIF M09_인입후경과월_IB_R6M = min(M08_인입후경과월_IB_R6M + 1, 6)""",
    },
    {
        "columns": ["M09_인입불만횟수_IB_B0M", "M08_인입불만후경과월_IB_R6M"],
        "output": "M09_인입불만후경과월_IB_R6M",
        "fname": "cfs_06_0254",
        "type": "formula",
        "content": """IF M09_인입불만횟수_IB_B0M > 0 THEN M09_인입불만후경과월_IB_R6M = 0
                      ELIF M09_인입불만후경과월_IB_R6M = min(M08_인입불만후경과월_IB_R6M + 1, 6)""",
    },
    {
        "columns": ["M09_불만제기건수_B0M", "M08_불만제기건수_B0M", "M08_불만제기후경과월_R12M"],
        "output": "M09_불만제기후경과월_R12M",
        "fname": "cfs_06_0306",
        "type": "formula",
        "content": """IF M09_불만제기건수_B0M > 0 THEN M09_불만제기후경과월_R12M = 0
                      ELIF (M08_불만제기건수_B0M==0 & M08_불만제기후경과월_R12M==0) THEN M09_불만제기후경과월_R12M = 0
                      ELIF (M08_불만제기건수_B0M==0 & M08_불만제기후경과월_R12M==12) THEN M09_불만제기후경과월_R12M = 12
                      ELIF M09_불만제기후경과월_R12M = min(M08_불만제기후경과월_R12M + 1, 12)""",
    },
    # M10
    {
        "columns": ["M10_인입횟수_ARS_B0M", "M09_인입횟수_ARS_B0M", "M09_인입후경과월_ARS"],
        "output": "M10_인입후경과월_ARS",
        "fname": "cfs_06_0328",
        "type": "formula",
        "content": """IF (M10_인입횟수_ARS_B0M>0) THEN M10_인입후경과월_ARS = 0
                      ELIF (M09_인입횟수_ARS_B0M==0)&(M09_인입후경과월_ARS==0) THEN M10_인입후경과월_ARS = 0
                      ELIF (M09_인입횟수_ARS_B0M>0)&(M09_인입후경과월_ARS==0) THEN M10_인입후경과월_ARS = 1
                      ELIF (M09_인입후경과월_ARS==5) THEN M10_인입후경과월_ARS = 0
                      ELIF M10_인입후경과월_ARS = M09_인입후경과월_ARS + 1""",
    },
    {
        "columns": ["M10_방문횟수_PC_B0M", "M09_방문후경과월_PC_R6M"],
        "output": "M10_방문후경과월_PC_R6M",
        "fname": "cfs_06_0335",
        "type": "formula",
        "content": """IF M10_방문횟수_PC_B0M > 0 THEN M10_방문후경과월_PC_R6M = 0
                      ELIF M10_방문후경과월_PC_R6M = min(M09_방문후경과월_PC_R6M + 1, 6)""",
    },
    {
        "columns": ["M10_방문횟수_앱_B0M", "M09_방문후경과월_앱_R6M"],
        "output": "M10_방문후경과월_앱_R6M",
        "fname": "cfs_06_0339",
        "type": "formula",
        "content": """IF M10_방문횟수_앱_B0M > 0 THEN M10_방문후경과월_앱_R6M = 0
                      ELIF M10_방문후경과월_앱_R6M = min(M09_방문후경과월_앱_R6M + 1, 6)""",
    },
    {
        "columns": ["M10_방문횟수_모바일웹_B0M", "M09_방문후경과월_모바일웹_R6M"],
        "output": "M10_방문후경과월_모바일웹_R6M",
        "fname": "cfs_06_0343",
        "type": "formula",
        "content": """IF M10_방문횟수_모바일웹_B0M > 0 THEN M10_방문후경과월_모바일웹_R6M = 0
                      ELIF M10_방문후경과월_모바일웹_R6M = min(M09_방문후경과월_모바일웹_R6M + 1, 6)""",
    },
    {
        "columns": ["M10_인입횟수_IB_B0M", "M09_인입후경과월_IB_R6M"],
        "output": "M10_인입후경과월_IB_R6M",
        "fname": "cfs_06_0354",
        "type": "formula",
        "content": """IF M10_인입횟수_IB_B0M > 0 THEN M10_인입후경과월_IB_R6M = 0
                      ELIF M10_인입후경과월_IB_R6M = min(M09_인입후경과월_IB_R6M + 1, 6)""",
    },
    {
        "columns": ["M10_인입불만횟수_IB_B0M", "M09_인입불만후경과월_IB_R6M"],
        "output": "M10_인입불만후경과월_IB_R6M",
        "fname": "cfs_06_0361",
        "type": "formula",
        "content": """IF M10_인입불만횟수_IB_B0M > 0 THEN M10_인입불만후경과월_IB_R6M = 0
                      ELIF M10_인입불만후경과월_IB_R6M = min(M09_인입불만후경과월_IB_R6M + 1, 6)""",
    },
    {
        "columns": ["M10_불만제기건수_B0M", "M09_불만제기건수_B0M", "M09_불만제기후경과월_R12M"],
        "output": "M10_불만제기후경과월_R12M",
        "fname": "cfs_06_0413",
        "type": "formula",
        "content": """IF M10_불만제기건수_B0M > 0 THEN M10_불만제기후경과월_R12M = 0
                      ELIF (M09_불만제기건수_B0M==0 & M09_불만제기후경과월_R12M==0) THEN M10_불만제기후경과월_R12M = 0
                      ELIF (M09_불만제기건수_B0M==0 & M09_불만제기후경과월_R12M==12) THEN M10_불만제기후경과월_R12M = 12
                      ELIF M10_불만제기후경과월_R12M = min(M09_불만제기후경과월_R12M + 1, 12)""",
    },
    {
        "columns": ["M10_홈페이지_금융건수_R3M", "M07_홈페이지_금융건수_R3M"],
        "output": "M10_홈페이지_금융건수_R6M",
        "fname": "cfs_06_0423",
        "type": "formula",
        "content": "M10_홈페이지_금융건수_R6M = M10_홈페이지_금융건수_R3M + M07_홈페이지_금융건수_R3M",
    },
    {
        "columns": ["M10_홈페이지_선결제건수_R3M", "M07_홈페이지_선결제건수_R3M"],
        "output": "M10_홈페이지_선결제건수_R6M",
        "fname": "cfs_06_0424",
        "type": "formula",
        "content": "M10_홈페이지_선결제건수_R6M = M10_홈페이지_선결제건수_R3M + M07_홈페이지_선결제건수_R3M",
    },
    # M11
    {
        "columns": ["M11_인입횟수_ARS_B0M", "M10_인입횟수_ARS_B0M", "M10_인입후경과월_ARS"],
        "output": "M11_인입후경과월_ARS",
        "fname": "cfs_06_0435",
        "type": "formula",
        "content": """"IF (M11_인입횟수_ARS_B0M>0) THEN M11_인입후경과월_ARS = 0
                      ELIF (M10_인입횟수_ARS_B0M==0)&(M10_인입후경과월_ARS==0) THEN M11_인입후경과월_ARS = 0
                      ELIF (M10_인입횟수_ARS_B0M>0)&(M10_인입후경과월_ARS==0) THEN M11_인입후경과월_ARS = 1
                      ELIF (M10_인입후경과월_ARS==5) THEN M11_인입후경과월_ARS = 0
                      ELIF M11_인입후경과월_ARS = M10_인입후경과월_ARS + 1""",
    },
    {
        "columns": ["M11_방문횟수_PC_B0M", "M10_방문후경과월_PC_R6M"],
        "output": "M11_방문후경과월_PC_R6M",
        "fname": "cfs_06_0442",
        "type": "formula",
        "content": """IF M11_방문횟수_PC_B0M > 0 THEN M11_방문후경과월_PC_R6M = 0
                      ELIF M11_방문후경과월_PC_R6M = min(M10_방문후경과월_PC_R6M + 1, 6)""",
    },
    {
        "columns": ["M11_방문횟수_앱_B0M", "M10_방문후경과월_앱_R6M"],
        "output": "M11_방문후경과월_앱_R6M",
        "fname": "cfs_06_0446",
        "type": "formula",
        "content": """IF M11_방문횟수_앱_B0M > 0 THEN M11_방문후경과월_앱_R6M = 0
                      ELIF M11_방문후경과월_앱_R6M = min(M10_방문후경과월_앱_R6M + 1, 6)""",
    },
    {
        "columns": ["M11_방문횟수_모바일웹_B0M", "M10_방문후경과월_모바일웹_R6M"],
        "output": "M11_방문후경과월_모바일웹_R6M",
        "fname": "cfs_06_0450",
        "type": "formula",
        "content": """IF M11_방문횟수_모바일웹_B0M > 0 THEN M11_방문후경과월_모바일웹_R6M = 0
                      ELIF M11_방문후경과월_모바일웹_R6M = min(M10_방문후경과월_모바일웹_R6M + 1, 6)""",
    },
    {
        "columns": ["M11_인입횟수_IB_B0M", "M10_인입후경과월_IB_R6M"],
        "output": "M11_인입후경과월_IB_R6M",
        "fname": "cfs_06_0461",
        "type": "formula",
        "content": """IF M11_인입횟수_IB_B0M > 0 THEN M11_인입후경과월_IB_R6M = 0
                      ELIF M11_인입후경과월_IB_R6M = min(M10_인입후경과월_IB_R6M + 1, 6)""",
    },
    {
        "columns": ["M11_인입불만횟수_IB_B0M", "M10_인입불만후경과월_IB_R6M"],
        "output": "M11_인입불만후경과월_IB_R6M",
        "fname": "cfs_06_0468",
        "type": "formula",
        "content": """IF M11_인입불만횟수_IB_B0M > 0 THEN M11_인입불만후경과월_IB_R6M = 0
                      ELIF M11_인입불만후경과월_IB_R6M = min(M10_인입불만후경과월_IB_R6M + 1, 6)""",
    },
    {
        "columns": ["M11_불만제기건수_B0M", "M10_불만제기건수_B0M", "M10_불만제기후경과월_R12M"],
        "output": "M11_불만제기후경과월_R12M",
        "fname": "cfs_06_0520",
        "type": "formula",
        "content": """IF M11_불만제기건수_B0M > 0 THEN M11_불만제기후경과월_R12M = 0
                      ELIF (M10_불만제기건수_B0M==0 & M10_불만제기후경과월_R12M==0) THEN M11_불만제기후경과월_R12M = 0
                      ELIF (M10_불만제기건수_B0M==0 & M10_불만제기후경과월_R12M==12) THEN M11_불만제기후경과월_R12M = 12
                      ELIF M11_불만제기후경과월_R12M = min(M10_불만제기후경과월_R12M + 1, 12)""",
    },
    {
        "columns": ["M11_홈페이지_금융건수_R3M", "M08_홈페이지_금융건수_R3M"],
        "output": "M11_홈페이지_금융건수_R6M",
        "fname": "cfs_06_0530",
        "type": "formula",
        "content": "M11_홈페이지_금융건수_R6M = M11_홈페이지_금융건수_R3M + M08_홈페이지_금융건수_R3M",
    },
    {
        "columns": ["M11_홈페이지_선결제건수_R3M", "M08_홈페이지_선결제건수_R3M"],
        "output": "M11_홈페이지_선결제건수_R6M",
        "fname": "cfs_06_0531",
        "type": "formula",
        "content": "M11_홈페이지_선결제건수_R6M = M11_홈페이지_선결제건수_R3M + M08_홈페이지_선결제건수_R3M",
    },
    # M12
    {
        "columns": ["M12_인입횟수_ARS_B0M", "M11_인입횟수_ARS_B0M", "M11_인입후경과월_ARS"],
        "output": "M12_인입후경과월_ARS",
        "fname": "cfs_06_0542",
        "type": "formula",
        "content": """IF (M12_인입횟수_ARS_B0M>0) THEN M12_인입후경과월_ARS = 0
                      ELIF (M11_인입횟수_ARS_B0M==0)&(M11_인입후경과월_ARS==0) THEN M12_인입후경과월_ARS = 0
                      ELIF (M11_인입횟수_ARS_B0M>0)&(M11_인입후경과월_ARS==0) THEN M12_인입후경과월_ARS = 1
                      ELIF (M11_인입후경과월_ARS==5) THEN M12_인입후경과월_ARS = 0
                      ELIF M12_인입후경과월_ARS = M11_인입후경과월_ARS + 1""",
    },
    {
        "columns": ["M12_방문횟수_PC_B0M", "M11_방문후경과월_PC_R6M"],
        "output": "M12_방문후경과월_PC_R6M",
        "fname": "cfs_06_0549",
        "type": "formula",
        "content": """IF M12_방문횟수_PC_B0M > 0 THEN M12_방문후경과월_PC_R6M = 0
                      ELIF M12_방문후경과월_PC_R6M = min(M11_방문후경과월_PC_R6M + 1, 6)""",
    },
    {
        "columns": ["M12_방문횟수_앱_B0M", "M11_방문후경과월_앱_R6M"],
        "output": "M12_방문후경과월_앱_R6M",
        "fname": "cfs_06_0553",
        "type": "formula",
        "content": """IF M12_방문횟수_앱_B0M > 0 THEN M12_방문후경과월_앱_R6M = 0
                      ELIF M12_방문후경과월_앱_R6M = min(M11_방문후경과월_앱_R6M + 1, 6)""",
    },
    {
        "columns": ["M12_방문횟수_모바일웹_B0M", "M11_방문후경과월_모바일웹_R6M"],
        "output": "M12_방문후경과월_모바일웹_R6M",
        "fname": "cfs_06_0557",
        "type": "formula",
        "content": """IF M12_방문횟수_모바일웹_B0M > 0 THEN M12_방문후경과월_모바일웹_R6M = 0
                      ELIF M12_방문후경과월_모바일웹_R6M = min(M11_방문후경과월_모바일웹_R6M + 1, 6)""",
    },
    {
        "columns": ["M12_인입횟수_IB_B0M", "M11_인입후경과월_IB_R6M"],
        "output": "M12_인입후경과월_IB_R6M",
        "fname": "cfs_06_0568",
        "type": "formula",
        "content": """IF M12_인입횟수_IB_B0M > 0 THEN M12_인입후경과월_IB_R6M = 0
                      ELIF M12_인입후경과월_IB_R6M = min(M11_인입후경과월_IB_R6M + 1, 6)""",
    },
    {
        "columns": ["M12_인입불만횟수_IB_B0M", "M11_인입불만후경과월_IB_R6M"],
        "output": "M12_인입불만후경과월_IB_R6M",
        "fname": "cfs_06_0575",
        "type": "formula",
        "content": """IF M12_인입불만횟수_IB_B0M > 0 THEN M12_인입불만후경과월_IB_R6M = 0
                      ELIF M12_인입불만후경과월_IB_R6M = min(M11_인입불만후경과월_IB_R6M + 1, 6)""",
    },
    {
        "columns": ["M12_불만제기건수_B0M", "M11_불만제기건수_B0M", "M11_불만제기후경과월_R12M"],
        "output": "M12_불만제기후경과월_R12M",
        "fname": "cfs_06_0627",
        "type": "formula",
        "content": """IF M12_불만제기건수_B0M > 0 THEN M12_불만제기후경과월_R12M = 0
                      ELIF (M11_불만제기건수_B0M==0 & M11_불만제기후경과월_R12M==0) THEN M12_불만제기후경과월_R12M = 0
                      ELIF (M11_불만제기건수_B0M==0 & M11_불만제기후경과월_R12M==12) THEN M12_불만제기후경과월_R12M = 12
                      ELIF M12_불만제기후경과월_R12M = min(M11_불만제기후경과월_R12M + 1, 12)""",
    },
    {
        "columns": [
            "M07_인입횟수_ARS_B0M",
            "M08_인입횟수_ARS_B0M",
            "M09_인입횟수_ARS_B0M",
            "M10_인입횟수_ARS_B0M",
            "M11_인입횟수_ARS_B0M",
            "M12_인입횟수_ARS_B0M"
        ],
        "output": "M12_인입횟수_ARS_R6M",
        "fname": "cfs_06_0538",
        "type": "formula",
        "content": "M12_인입횟수_ARS_R6M = SUM(M07~M12_인입횟수_ARS_B0M)",
    },
    {
        "columns": [
            "M07_이용메뉴건수_ARS_B0M",
            "M08_이용메뉴건수_ARS_B0M",
            "M09_이용메뉴건수_ARS_B0M",
            "M10_이용메뉴건수_ARS_B0M",
            "M11_이용메뉴건수_ARS_B0M",
            "M12_이용메뉴건수_ARS_B0M"
        ],
        "output": "M12_이용메뉴건수_ARS_R6M",
        "fname": "cfs_06_0539",
        "type": "formula",
        "content": "M12_이용메뉴건수_ARS_R6M = SUM(M07~M12_이용메뉴건수_ARS_B0M)",
    },
    {
        "columns": [
            "M07_인입일수_ARS_B0M",
            "M08_인입일수_ARS_B0M",
            "M09_인입일수_ARS_B0M",
            "M10_인입일수_ARS_B0M",
            "M11_인입일수_ARS_B0M",
            "M12_인입일수_ARS_B0M"
        ],
        "output": "M12_인입일수_ARS_R6M",
        "fname": "cfs_06_0540",
        "type": "formula",
        "content": "M12_인입일수_ARS_R6M = SUM(M07~M12_인입일수_ARS_B0M)",
    },
    {
        "columns": [
            "M07_방문횟수_PC_B0M",
            "M08_방문횟수_PC_B0M",
            "M09_방문횟수_PC_B0M",
            "M10_방문횟수_PC_B0M",
            "M11_방문횟수_PC_B0M",
            "M12_방문횟수_PC_B0M"
        ],
        "output": "M12_방문횟수_PC_R6M",
        "fname": "cfs_06_0546",
        "type": "formula",
        "content": "M12_방문횟수_PC_R6M = SUM(M07~M12_방문횟수_PC_B0M)",
    },
    {
        "columns": [
            "M07_방문일수_PC_B0M",
            "M08_방문일수_PC_B0M",
            "M09_방문일수_PC_B0M",
            "M10_방문일수_PC_B0M",
            "M11_방문일수_PC_B0M",
            "M12_방문일수_PC_B0M"
        ],
        "output": "M12_방문일수_PC_R6M",
        "fname": "cfs_06_0547",
        "type": "formula",
        "content": "M12_방문일수_PC_R6M = SUM(M07~M12_방문일수_PC_B0M)",
    },
    {
        "columns": [
            "M07_방문횟수_앱_B0M",
            "M08_방문횟수_앱_B0M",
            "M09_방문횟수_앱_B0M",
            "M10_방문횟수_앱_B0M",
            "M11_방문횟수_앱_B0M",
            "M12_방문횟수_앱_B0M"
        ],
        "output": "M12_방문횟수_앱_R6M",
        "fname": "cfs_06_0550",
        "type": "formula",
        "content": "M12_방문횟수_앱_R6M = SUM(M07~M12_방문횟수_앱_B0M)",
    },
    {
        "columns": [
            "M07_방문일수_앱_B0M",
            "M08_방문일수_앱_B0M",
            "M09_방문일수_앱_B0M",
            "M10_방문일수_앱_B0M",
            "M11_방문일수_앱_B0M",
            "M12_방문일수_앱_B0M"
        ],
        "output": "M12_방문일수_앱_R6M",
        "fname": "cfs_06_0551",
        "type": "formula",
        "content": "M12_방문일수_앱_R6M = SUM(M07~M12_방문일수_앱_B0M)",
    },
    {
        "columns": [
            "M07_방문횟수_모바일웹_B0M",
            "M08_방문횟수_모바일웹_B0M",
            "M09_방문횟수_모바일웹_B0M",
            "M10_방문횟수_모바일웹_B0M",
            "M11_방문횟수_모바일웹_B0M",
            "M12_방문횟수_모바일웹_B0M"
        ],
        "output": "M12_방문횟수_모바일웹_R6M",
        "fname": "cfs_06_0554",
        "type": "formula",
        "content": "M12_방문횟수_모바일웹_R6M = SUM(M07~M12_방문횟수_모바일웹_B0M)",
    },
    {
        "columns": [
            "M07_방문일수_모바일웹_B0M",
            "M08_방문일수_모바일웹_B0M",
            "M09_방문일수_모바일웹_B0M",
            "M10_방문일수_모바일웹_B0M",
            "M11_방문일수_모바일웹_B0M",
            "M12_방문일수_모바일웹_B0M"
        ],
        "output": "M12_방문일수_모바일웹_R6M",
        "fname": "cfs_06_0555",
        "type": "formula",
        "content": "M12_방문일수_모바일웹_R6M = SUM(M07~M12_방문일수_모바일웹_B0M)",
    },
    {
        "columns": [
            "M07_인입횟수_IB_B0M",
            "M08_인입횟수_IB_B0M",
            "M09_인입횟수_IB_B0M",
            "M10_인입횟수_IB_B0M",
            "M11_인입횟수_IB_B0M",
            "M12_인입횟수_IB_B0M"
        ],
        "output": "M12_인입횟수_IB_R6M",
        "fname": "cfs_06_0564",
        "type": "formula",
        "content": "M12_인입횟수_IB_R6M = SUM(M07~M12_인입횟수_IB_B0M)",
    },
    {
        "columns": [
            "M07_인입일수_IB_B0M",
            "M08_인입일수_IB_B0M",
            "M09_인입일수_IB_B0M",
            "M10_인입일수_IB_B0M",
            "M11_인입일수_IB_B0M",
            "M12_인입일수_IB_B0M"
        ],
        "output": "M12_인입일수_IB_R6M",
        "fname": "cfs_06_0565",
        "type": "formula",
        "content": "M12_인입일수_IB_R6M = SUM(M07~M12_인입일수_IB_B0M)",
    },
    {
        "columns": [
            "M07_이용메뉴건수_IB_B0M",
            "M08_이용메뉴건수_IB_B0M",
            "M09_이용메뉴건수_IB_B0M",
            "M10_이용메뉴건수_IB_B0M",
            "M11_이용메뉴건수_IB_B0M",
            "M12_이용메뉴건수_IB_B0M"
        ],
        "output": "M12_이용메뉴건수_IB_R6M",
        "fname": "cfs_06_0567",
        "type": "formula",
        "content": "M12_이용메뉴건수_IB_R6M = SUM(M07~M12_이용메뉴건수_IB_B0M)",
    },
    {
        "columns": [
            "M07_인입불만횟수_IB_B0M",
            "M08_인입불만횟수_IB_B0M",
            "M09_인입불만횟수_IB_B0M",
            "M10_인입불만횟수_IB_B0M",
            "M11_인입불만횟수_IB_B0M",
            "M12_인입불만횟수_IB_B0M"
        ],
        "output": "M12_인입불만횟수_IB_R6M",
        "fname": "cfs_06_0572",
        "type": "formula",
        "content": "M12_인입불만횟수_IB_R6M = SUM(M07~M12_인입불만횟수_IB_B0M)",
    },
    {
        "columns": [
            "M07_인입불만일수_IB_B0M",
            "M08_인입불만일수_IB_B0M",
            "M09_인입불만일수_IB_B0M",
            "M10_인입불만일수_IB_B0M",
            "M11_인입불만일수_IB_B0M",
            "M12_인입불만일수_IB_B0M"
        ],
        "output": "M12_인입불만일수_IB_R6M",
        "fname": "cfs_06_0573",
        "type": "formula",
        "content": "M12_인입불만일수_IB_R6M = SUM(M07~M12_인입불만일수_IB_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_사용승인내역_B0M",
            "M08_IB문의건수_사용승인내역_B0M",
            "M09_IB문의건수_사용승인내역_B0M",
            "M10_IB문의건수_사용승인내역_B0M",
            "M11_IB문의건수_사용승인내역_B0M",
            "M12_IB문의건수_사용승인내역_B0M"
        ],
        "output": "M12_IB문의건수_사용승인내역_R6M",
        "fname": "cfs_06_0602",
        "type": "formula",
        "content": "M12_IB문의건수_사용승인내역_R6M = SUM(M07~M12_IB문의건수_사용승인내역_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_한도_B0M",
            "M08_IB문의건수_한도_B0M",
            "M09_IB문의건수_한도_B0M",
            "M10_IB문의건수_한도_B0M",
            "M11_IB문의건수_한도_B0M",
            "M12_IB문의건수_한도_B0M"
        ],
        "output": "M12_IB문의건수_한도_R6M",
        "fname": "cfs_06_0603",
        "type": "formula",
        "content": "M12_IB문의건수_한도_R6M = SUM(M07~M12_IB문의건수_한도_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_선결제_B0M",
            "M08_IB문의건수_선결제_B0M",
            "M09_IB문의건수_선결제_B0M",
            "M10_IB문의건수_선결제_B0M",
            "M11_IB문의건수_선결제_B0M",
            "M12_IB문의건수_선결제_B0M"
        ],
        "output": "M12_IB문의건수_선결제_R6M",
        "fname": "cfs_06_0604",
        "type": "formula",
        "content": "M12_IB문의건수_선결제_R6M = SUM(M07~M12_IB문의건수_선결제_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_결제_B0M",
            "M08_IB문의건수_결제_B0M",
            "M09_IB문의건수_결제_B0M",
            "M10_IB문의건수_결제_B0M",
            "M11_IB문의건수_결제_B0M",
            "M12_IB문의건수_결제_B0M"
        ],
        "output": "M12_IB문의건수_결제_R6M",
        "fname": "cfs_06_0605",
        "type": "formula",
        "content": "M12_IB문의건수_결제_R6M = SUM(M07~M12_IB문의건수_결제_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_할부_B0M",
            "M08_IB문의건수_할부_B0M",
            "M09_IB문의건수_할부_B0M",
            "M10_IB문의건수_할부_B0M",
            "M11_IB문의건수_할부_B0M",
            "M12_IB문의건수_할부_B0M"
        ],
        "output": "M12_IB문의건수_할부_R6M",
        "fname": "cfs_06_0606",
        "type": "formula",
        "content": "M12_IB문의건수_할부_R6M = SUM(M07~M12_IB문의건수_할부_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_정보변경_B0M",
            "M08_IB문의건수_정보변경_B0M",
            "M09_IB문의건수_정보변경_B0M",
            "M10_IB문의건수_정보변경_B0M",
            "M11_IB문의건수_정보변경_B0M",
            "M12_IB문의건수_정보변경_B0M"
        ],
        "output": "M12_IB문의건수_정보변경_R6M",
        "fname": "cfs_06_0607",
        "type": "formula",
        "content": "M12_IB문의건수_정보변경_R6M = SUM(M07~M12_IB문의건수_정보변경_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_결제일변경_B0M",
            "M08_IB문의건수_결제일변경_B0M",
            "M09_IB문의건수_결제일변경_B0M",
            "M10_IB문의건수_결제일변경_B0M",
            "M11_IB문의건수_결제일변경_B0M",
            "M12_IB문의건수_결제일변경_B0M"
        ],
        "output": "M12_IB문의건수_결제일변경_R6M",
        "fname": "cfs_06_0608",
        "type": "formula",
        "content": "M12_IB문의건수_결제일변경_R6M = SUM(M07~M12_IB문의건수_결제일변경_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_명세서_B0M",
            "M08_IB문의건수_명세서_B0M",
            "M09_IB문의건수_명세서_B0M",
            "M10_IB문의건수_명세서_B0M",
            "M11_IB문의건수_명세서_B0M",
            "M12_IB문의건수_명세서_B0M"
        ],
        "output": "M12_IB문의건수_명세서_R6M",
        "fname": "cfs_06_0609",
        "type": "formula",
        "content": "M12_IB문의건수_명세서_R6M = SUM(M07~M12_IB문의건수_명세서_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_비밀번호_B0M",
            "M08_IB문의건수_비밀번호_B0M",
            "M09_IB문의건수_비밀번호_B0M",
            "M10_IB문의건수_비밀번호_B0M",
            "M11_IB문의건수_비밀번호_B0M",
            "M12_IB문의건수_비밀번호_B0M"
        ],
        "output": "M12_IB문의건수_비밀번호_R6M",
        "fname": "cfs_06_0610",
        "type": "formula",
        "content": "M12_IB문의건수_비밀번호_R6M = SUM(M07~M12_IB문의건수_비밀번호_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_SMS_B0M",
            "M08_IB문의건수_SMS_B0M",
            "M09_IB문의건수_SMS_B0M",
            "M10_IB문의건수_SMS_B0M",
            "M11_IB문의건수_SMS_B0M",
            "M12_IB문의건수_SMS_B0M"
        ],
        "output": "M12_IB문의건수_SMS_R6M",
        "fname": "cfs_06_0611",
        "type": "formula",
        "content": "M12_IB문의건수_SMS_R6M = SUM(M07~M12_IB문의건수_SMS_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_APP_B0M",
            "M08_IB문의건수_APP_B0M",
            "M09_IB문의건수_APP_B0M",
            "M10_IB문의건수_APP_B0M",
            "M11_IB문의건수_APP_B0M",
            "M12_IB문의건수_APP_B0M"
        ],
        "output": "M12_IB문의건수_APP_R6M",
        "fname": "cfs_06_0612",
        "type": "formula",
        "content": "M12_IB문의건수_APP_R6M = SUM(M07~M12_IB문의건수_APP_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_부대서비스_B0M",
            "M08_IB문의건수_부대서비스_B0M",
            "M09_IB문의건수_부대서비스_B0M",
            "M10_IB문의건수_부대서비스_B0M",
            "M11_IB문의건수_부대서비스_B0M",
            "M12_IB문의건수_부대서비스_B0M"
        ],
        "output": "M12_IB문의건수_부대서비스_R6M",
        "fname": "cfs_06_0613",
        "type": "formula",
        "content": "M12_IB문의건수_부대서비스_R6M = SUM(M07~M12_IB문의건수_부대서비스_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_포인트_B0M",
            "M08_IB문의건수_포인트_B0M",
            "M09_IB문의건수_포인트_B0M",
            "M10_IB문의건수_포인트_B0M",
            "M11_IB문의건수_포인트_B0M",
            "M12_IB문의건수_포인트_B0M"
        ],
        "output": "M12_IB문의건수_포인트_R6M",
        "fname": "cfs_06_0614",
        "type": "formula",
        "content": "M12_IB문의건수_포인트_R6M = SUM(M07~M12_IB문의건수_포인트_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_카드발급_B0M",
            "M08_IB문의건수_카드발급_B0M",
            "M09_IB문의건수_카드발급_B0M",
            "M10_IB문의건수_카드발급_B0M",
            "M11_IB문의건수_카드발급_B0M",
            "M12_IB문의건수_카드발급_B0M"
        ],
        "output": "M12_IB문의건수_카드발급_R6M",
        "fname": "cfs_06_0615",
        "type": "formula",
        "content": "M12_IB문의건수_카드발급_R6M = SUM(M07~M12_IB문의건수_카드발급_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_BL_B0M",
            "M08_IB문의건수_BL_B0M",
            "M09_IB문의건수_BL_B0M",
            "M10_IB문의건수_BL_B0M",
            "M11_IB문의건수_BL_B0M",
            "M12_IB문의건수_BL_B0M"
        ],
        "output": "M12_IB문의건수_BL_R6M",
        "fname": "cfs_06_0616",
        "type": "formula",
        "content": "M12_IB문의건수_BL_R6M = SUM(M07~M12_IB문의건수_BL_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_분실도난_B0M",
            "M08_IB문의건수_분실도난_B0M",
            "M09_IB문의건수_분실도난_B0M",
            "M10_IB문의건수_분실도난_B0M",
            "M11_IB문의건수_분실도난_B0M",
            "M12_IB문의건수_분실도난_B0M"
        ],
        "output": "M12_IB문의건수_분실도난_R6M",
        "fname": "cfs_06_0617",
        "type": "formula",
        "content": "M12_IB문의건수_분실도난_R6M = SUM(M07~M12_IB문의건수_분실도난_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_CA_B0M",
            "M08_IB문의건수_CA_B0M",
            "M09_IB문의건수_CA_B0M",
            "M10_IB문의건수_CA_B0M",
            "M11_IB문의건수_CA_B0M",
            "M12_IB문의건수_CA_B0M"
        ],
        "output": "M12_IB문의건수_CA_R6M",
        "fname": "cfs_06_0618",
        "type": "formula",
        "content": "M12_IB문의건수_CA_R6M = SUM(M07~M12_IB문의건수_CA_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_CL_RV_B0M",
            "M08_IB문의건수_CL_RV_B0M",
            "M09_IB문의건수_CL_RV_B0M",
            "M10_IB문의건수_CL_RV_B0M",
            "M11_IB문의건수_CL_RV_B0M",
            "M12_IB문의건수_CL_RV_B0M"
        ],
        "output": "M12_IB문의건수_CL_RV_R6M",
        "fname": "cfs_06_0619",
        "type": "formula",
        "content": "M12_IB문의건수_CL_RV_R6M = SUM(M07~M12_IB문의건수_CL_RV_B0M)",
    },
    {
        "columns": [
            "M07_IB문의건수_CS_B0M",
            "M08_IB문의건수_CS_B0M",
            "M09_IB문의건수_CS_B0M",
            "M10_IB문의건수_CS_B0M",
            "M11_IB문의건수_CS_B0M",
            "M12_IB문의건수_CS_B0M"
        ],
        "output": "M12_IB문의건수_CS_R6M",
        "fname": "cfs_06_0620",
        "type": "formula",
        "content": "M12_IB문의건수_CS_R6M = SUM(M07~M12_IB문의건수_CS_B0M)",
    },
    {
        "columns": [
            "M07_IB상담건수_VOC_B0M",
            "M08_IB상담건수_VOC_B0M",
            "M09_IB상담건수_VOC_B0M",
            "M10_IB상담건수_VOC_B0M",
            "M11_IB상담건수_VOC_B0M",
            "M12_IB상담건수_VOC_B0M"
        ],
        "output": "M12_IB상담건수_VOC_R6M",
        "fname": "cfs_06_0621",
        "type": "formula",
        "content": "M12_IB상담건수_VOC_R6M = SUM(M07~M12_IB상담건수_VOC_B0M)",
    },
    {
        "columns": [
            "M07_IB상담건수_VOC민원_B0M",
            "M08_IB상담건수_VOC민원_B0M",
            "M09_IB상담건수_VOC민원_B0M",
            "M10_IB상담건수_VOC민원_B0M",
            "M11_IB상담건수_VOC민원_B0M",
            "M12_IB상담건수_VOC민원_B0M"
        ],
        "output": "M12_IB상담건수_VOC민원_R6M",
        "fname": "cfs_06_0622",
        "type": "formula",
        "content": "M12_IB상담건수_VOC민원_R6M = SUM(M07~M12_IB상담건수_VOC민원_B0M)",
    },
    {
        "columns": [
            "M07_IB상담건수_VOC불만_B0M",
            "M08_IB상담건수_VOC불만_B0M",
            "M09_IB상담건수_VOC불만_B0M",
            "M10_IB상담건수_VOC불만_B0M",
            "M11_IB상담건수_VOC불만_B0M",
            "M12_IB상담건수_VOC불만_B0M"
        ],
        "output": "M12_IB상담건수_VOC불만_R6M",
        "fname": "cfs_06_0623",
        "type": "formula",
        "content": "M12_IB상담건수_VOC불만_R6M = SUM(M07~M12_IB상담건수_VOC불만_B0M)",
    },
    {
        "columns": [
            "M07_IB상담건수_금감원_B0M",
            "M08_IB상담건수_금감원_B0M",
            "M09_IB상담건수_금감원_B0M",
            "M10_IB상담건수_금감원_B0M",
            "M11_IB상담건수_금감원_B0M",
            "M12_IB상담건수_금감원_B0M"
        ],
        "output": "M12_IB상담건수_금감원_R6M",
        "fname": "cfs_06_0624",
        "type": "formula",
        "content": "M12_IB상담건수_금감원_R6M = SUM(M07~M12_IB상담건수_금감원_B0M)",
    },
    {
        "columns": [
            "M07_당사PAY_방문횟수_B0M",
            "M08_당사PAY_방문횟수_B0M",
            "M09_당사PAY_방문횟수_B0M",
            "M10_당사PAY_방문횟수_B0M",
            "M11_당사PAY_방문횟수_B0M",
            "M12_당사PAY_방문횟수_B0M"
        ],
        "output": "M12_당사PAY_방문횟수_R6M",
        "fname": "cfs_06_0629",
        "type": "formula",
        "content": "M12_당사PAY_방문횟수_R6M = SUM(M07~M12_당사PAY_방문횟수_B0M)",
    },
    {
        "columns": [
            "M07_당사멤버쉽_방문횟수_B0M",
            "M08_당사멤버쉽_방문횟수_B0M",
            "M09_당사멤버쉽_방문횟수_B0M",
            "M10_당사멤버쉽_방문횟수_B0M",
            "M11_당사멤버쉽_방문횟수_B0M",
            "M12_당사멤버쉽_방문횟수_B0M"
        ],
        "output": "M12_당사멤버쉽_방문횟수_R6M",
        "fname": "cfs_06_0632",
        "type": "formula",
        "content": "M12_당사멤버쉽_방문횟수_R6M = SUM(M07~M12_당사멤버쉽_방문횟수_B0M)",
    },
    {
        "columns": ["M12_홈페이지_금융건수_R3M", "M09_홈페이지_금융건수_R3M"],
        "output": "M12_홈페이지_금융건수_R6M",
        "fname": "cfs_06_0637",
        "type": "formula",
        "content": "M12_홈페이지_금융건수_R6M = M12_홈페이지_금융건수_R3M + M09_홈페이지_금융건수_R3M",
    },
    {
        "columns": ["M12_홈페이지_선결제건수_R3M", "M09_홈페이지_선결제건수_R3M"],
        "output": "M12_홈페이지_선결제건수_R6M",
        "fname": "cfs_06_0638",
        "type": "formula",
        "content": "M12_홈페이지_선결제건수_R6M = M12_홈페이지_선결제건수_R3M + M09_홈페이지_선결제건수_R3M",
    },

    # 7.마케팅 테이블 컬럼 Formula
    # M12
    {
        "columns": [
            "M07_컨택건수_카드론_TM_B0M",
            "M08_컨택건수_카드론_TM_B0M",
            "M09_컨택건수_카드론_TM_B0M",
            "M10_컨택건수_카드론_TM_B0M",
            "M11_컨택건수_카드론_TM_B0M",
            "M12_컨택건수_카드론_TM_B0M"
        ],
        "output": "M12_컨택건수_카드론_TM_R6M",
        "fname": "cfs_07_0351",
        "type": "formula",
        "content": "M12_컨택건수_카드론_TM_R6M = SUM(M07~M12_컨택건수_카드론_TM_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_리볼빙_TM_B0M",
            "M08_컨택건수_리볼빙_TM_B0M",
            "M09_컨택건수_리볼빙_TM_B0M",
            "M10_컨택건수_리볼빙_TM_B0M",
            "M11_컨택건수_리볼빙_TM_B0M",
            "M12_컨택건수_리볼빙_TM_B0M"
        ],
        "output": "M12_컨택건수_리볼빙_TM_R6M",
        "fname": "cfs_07_0352",
        "type": "formula",
        "content": "M12_컨택건수_리볼빙_TM_R6M = SUM(M07~M12_컨택건수_리볼빙_TM_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_CA_TM_B0M",
            "M08_컨택건수_CA_TM_B0M",
            "M09_컨택건수_CA_TM_B0M",
            "M10_컨택건수_CA_TM_B0M",
            "M11_컨택건수_CA_TM_B0M",
            "M12_컨택건수_CA_TM_B0M"
        ],
        "output": "M12_컨택건수_CA_TM_R6M",
        "fname": "cfs_07_0353",
        "type": "formula",
        "content": "M12_컨택건수_CA_TM_R6M = SUM(M07~M12_컨택건수_CA_TM_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_이용유도_TM_B0M",
            "M08_컨택건수_이용유도_TM_B0M",
            "M09_컨택건수_이용유도_TM_B0M",
            "M10_컨택건수_이용유도_TM_B0M",
            "M11_컨택건수_이용유도_TM_B0M",
            "M12_컨택건수_이용유도_TM_B0M"
        ],
        "output": "M12_컨택건수_이용유도_TM_R6M",
        "fname": "cfs_07_0354",
        "type": "formula",
        "content": "M12_컨택건수_이용유도_TM_R6M = SUM(M07~M12_컨택건수_이용유도_TM_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_신용발급_TM_B0M",
            "M08_컨택건수_신용발급_TM_B0M",
            "M09_컨택건수_신용발급_TM_B0M",
            "M10_컨택건수_신용발급_TM_B0M",
            "M11_컨택건수_신용발급_TM_B0M",
            "M12_컨택건수_신용발급_TM_B0M"
        ],
        "output": "M12_컨택건수_신용발급_TM_R6M",
        "fname": "cfs_07_0355",
        "type": "formula",
        "content": "M12_컨택건수_신용발급_TM_R6M = SUM(M07~M12_컨택건수_신용발급_TM_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_부대서비스_TM_B0M",
            "M08_컨택건수_부대서비스_TM_B0M",
            "M09_컨택건수_부대서비스_TM_B0M",
            "M10_컨택건수_부대서비스_TM_B0M",
            "M11_컨택건수_부대서비스_TM_B0M",
            "M12_컨택건수_부대서비스_TM_B0M"
        ],
        "output": "M12_컨택건수_부대서비스_TM_R6M",
        "fname": "cfs_07_0356",
        "type": "formula",
        "content": "M12_컨택건수_부대서비스_TM_R6M = SUM(M07~M12_컨택건수_부대서비스_TM_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_포인트소진_TM_B0M",
            "M08_컨택건수_포인트소진_TM_B0M",
            "M09_컨택건수_포인트소진_TM_B0M",
            "M10_컨택건수_포인트소진_TM_B0M",
            "M11_컨택건수_포인트소진_TM_B0M",
            "M12_컨택건수_포인트소진_TM_B0M"
        ],
        "output": "M12_컨택건수_포인트소진_TM_R6M",
        "fname": "cfs_07_0357",
        "type": "formula",
        "content": "M12_컨택건수_포인트소진_TM_R6M = SUM(M07~M12_컨택건수_포인트소진_TM_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_보험_TM_B0M",
            "M08_컨택건수_보험_TM_B0M",
            "M09_컨택건수_보험_TM_B0M",
            "M10_컨택건수_보험_TM_B0M",
            "M11_컨택건수_보험_TM_B0M",
            "M12_컨택건수_보험_TM_B0M"
        ],
        "output": "M12_컨택건수_보험_TM_R6M",
        "fname": "cfs_07_0358",
        "type": "formula",
        "content": "M12_컨택건수_보험_TM_R6M = SUM(M07~M12_컨택건수_보험_TM_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_카드론_LMS_B0M",
            "M08_컨택건수_카드론_LMS_B0M",
            "M09_컨택건수_카드론_LMS_B0M",
            "M10_컨택건수_카드론_LMS_B0M",
            "M11_컨택건수_카드론_LMS_B0M",
            "M12_컨택건수_카드론_LMS_B0M"
        ],
        "output": "M12_컨택건수_카드론_LMS_R6M",
        "fname": "cfs_07_0359",
        "type": "formula",
        "content": "M12_컨택건수_카드론_LMS_R6M = SUM(M07~M12_컨택건수_카드론_LMS_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_CA_LMS_B0M",
            "M08_컨택건수_CA_LMS_B0M",
            "M09_컨택건수_CA_LMS_B0M",
            "M10_컨택건수_CA_LMS_B0M",
            "M11_컨택건수_CA_LMS_B0M",
            "M12_컨택건수_CA_LMS_B0M"
        ],
        "output": "M12_컨택건수_CA_LMS_R6M",
        "fname": "cfs_07_0360",
        "type": "formula",
        "content": "M12_컨택건수_CA_LMS_R6M = SUM(M07~M12_컨택건수_CA_LMS_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_리볼빙_LMS_B0M",
            "M08_컨택건수_리볼빙_LMS_B0M",
            "M09_컨택건수_리볼빙_LMS_B0M",
            "M10_컨택건수_리볼빙_LMS_B0M",
            "M11_컨택건수_리볼빙_LMS_B0M",
            "M12_컨택건수_리볼빙_LMS_B0M"
        ],
        "output": "M12_컨택건수_리볼빙_LMS_R6M",
        "fname": "cfs_07_0361",
        "type": "formula",
        "content": "M12_컨택건수_리볼빙_LMS_R6M = SUM(M07~M12_컨택건수_리볼빙_LMS_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_이용유도_LMS_B0M",
            "M08_컨택건수_이용유도_LMS_B0M",
            "M09_컨택건수_이용유도_LMS_B0M",
            "M10_컨택건수_이용유도_LMS_B0M",
            "M11_컨택건수_이용유도_LMS_B0M",
            "M12_컨택건수_이용유도_LMS_B0M"
        ],
        "output": "M12_컨택건수_이용유도_LMS_R6M",
        "fname": "cfs_07_0362",
        "type": "formula",
        "content": "M12_컨택건수_이용유도_LMS_R6M = SUM(M07~M12_컨택건수_이용유도_LMS_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_카드론_EM_B0M",
            "M08_컨택건수_카드론_EM_B0M",
            "M09_컨택건수_카드론_EM_B0M",
            "M10_컨택건수_카드론_EM_B0M",
            "M11_컨택건수_카드론_EM_B0M",
            "M12_컨택건수_카드론_EM_B0M"
        ],
        "output": "M12_컨택건수_카드론_EM_R6M",
        "fname": "cfs_07_0363",
        "type": "formula",
        "content": "M12_컨택건수_카드론_EM_R6M = SUM(M07~M12_컨택건수_카드론_EM_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_CA_EM_B0M",
            "M08_컨택건수_CA_EM_B0M",
            "M09_컨택건수_CA_EM_B0M",
            "M10_컨택건수_CA_EM_B0M",
            "M11_컨택건수_CA_EM_B0M",
            "M12_컨택건수_CA_EM_B0M"
        ],
        "output": "M12_컨택건수_CA_EM_R6M",
        "fname": "cfs_07_0364",
        "type": "formula",
        "content": "M12_컨택건수_CA_EM_R6M = SUM(M07~M12_컨택건수_CA_EM_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_리볼빙_EM_B0M",
            "M08_컨택건수_리볼빙_EM_B0M",
            "M09_컨택건수_리볼빙_EM_B0M",
            "M10_컨택건수_리볼빙_EM_B0M",
            "M11_컨택건수_리볼빙_EM_B0M",
            "M12_컨택건수_리볼빙_EM_B0M"
        ],
        "output": "M12_컨택건수_리볼빙_EM_R6M",
        "fname": "cfs_07_0365",
        "type": "formula",
        "content": "M12_컨택건수_리볼빙_EM_R6M = SUM(M07~M12_컨택건수_리볼빙_EM_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_이용유도_EM_B0M",
            "M08_컨택건수_이용유도_EM_B0M",
            "M09_컨택건수_이용유도_EM_B0M",
            "M10_컨택건수_이용유도_EM_B0M",
            "M11_컨택건수_이용유도_EM_B0M",
            "M12_컨택건수_이용유도_EM_B0M"
        ],
        "output": "M12_컨택건수_이용유도_EM_R6M",
        "fname": "cfs_07_0366",
        "type": "formula",
        "content": "M12_컨택건수_이용유도_EM_R6M = SUM(M07~M12_컨택건수_이용유도_EM_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_카드론_청구서_B0M",
            "M08_컨택건수_카드론_청구서_B0M",
            "M09_컨택건수_카드론_청구서_B0M",
            "M10_컨택건수_카드론_청구서_B0M",
            "M11_컨택건수_카드론_청구서_B0M",
            "M12_컨택건수_카드론_청구서_B0M"
        ],
        "output": "M12_컨택건수_카드론_청구서_R6M",
        "fname": "cfs_07_0367",
        "type": "formula",
        "content": "M12_컨택건수_카드론_청구서_R6M = SUM(M07~M12_컨택건수_카드론_청구서_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_CA_청구서_B0M",
            "M08_컨택건수_CA_청구서_B0M",
            "M09_컨택건수_CA_청구서_B0M",
            "M10_컨택건수_CA_청구서_B0M",
            "M11_컨택건수_CA_청구서_B0M",
            "M12_컨택건수_CA_청구서_B0M"
        ],
        "output": "M12_컨택건수_CA_청구서_R6M",
        "fname": "cfs_07_0368",
        "type": "formula",
        "content": "M12_컨택건수_CA_청구서_R6M = SUM(M07~M12_컨택건수_CA_청구서_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_리볼빙_청구서_B0M",
            "M08_컨택건수_리볼빙_청구서_B0M",
            "M09_컨택건수_리볼빙_청구서_B0M",
            "M10_컨택건수_리볼빙_청구서_B0M",
            "M11_컨택건수_리볼빙_청구서_B0M",
            "M12_컨택건수_리볼빙_청구서_B0M"
        ],
        "output": "M12_컨택건수_리볼빙_청구서_R6M",
        "fname": "cfs_07_0369",
        "type": "formula",
        "content": "M12_컨택건수_리볼빙_청구서_R6M = SUM(M07~M12_컨택건수_리볼빙_청구서_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_이용유도_청구서_B0M",
            "M08_컨택건수_이용유도_청구서_B0M",
            "M09_컨택건수_이용유도_청구서_B0M",
            "M10_컨택건수_이용유도_청구서_B0M",
            "M11_컨택건수_이용유도_청구서_B0M",
            "M12_컨택건수_이용유도_청구서_B0M"
        ],
        "output": "M12_컨택건수_이용유도_청구서_R6M",
        "fname": "cfs_07_0370",
        "type": "formula",
        "content": "M12_컨택건수_이용유도_청구서_R6M = SUM(M07~M12_컨택건수_이용유도_청구서_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_카드론_인터넷_B0M",
            "M08_컨택건수_카드론_인터넷_B0M",
            "M09_컨택건수_카드론_인터넷_B0M",
            "M10_컨택건수_카드론_인터넷_B0M",
            "M11_컨택건수_카드론_인터넷_B0M",
            "M12_컨택건수_카드론_인터넷_B0M"
        ],
        "output": "M12_컨택건수_카드론_인터넷_R6M",
        "fname": "cfs_07_0371",
        "type": "formula",
        "content": "M12_컨택건수_카드론_인터넷_R6M = SUM(M07~M12_컨택건수_카드론_인터넷_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_CA_인터넷_B0M",
            "M08_컨택건수_CA_인터넷_B0M",
            "M09_컨택건수_CA_인터넷_B0M",
            "M10_컨택건수_CA_인터넷_B0M",
            "M11_컨택건수_CA_인터넷_B0M",
            "M12_컨택건수_CA_인터넷_B0M"
        ],
        "output": "M12_컨택건수_CA_인터넷_R6M",
        "fname": "cfs_07_0372",
        "type": "formula",
        "content": "M12_컨택건수_CA_인터넷_R6M = SUM(M07~M12_컨택건수_CA_인터넷_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_리볼빙_인터넷_B0M",
            "M08_컨택건수_리볼빙_인터넷_B0M",
            "M09_컨택건수_리볼빙_인터넷_B0M",
            "M10_컨택건수_리볼빙_인터넷_B0M",
            "M11_컨택건수_리볼빙_인터넷_B0M",
            "M12_컨택건수_리볼빙_인터넷_B0M"
        ],
        "output": "M12_컨택건수_리볼빙_인터넷_R6M",
        "fname": "cfs_07_0373",
        "type": "formula",
        "content": "M12_컨택건수_리볼빙_인터넷_R6M = SUM(M07~M12_컨택건수_리볼빙_인터넷_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_이용유도_인터넷_B0M",
            "M08_컨택건수_이용유도_인터넷_B0M",
            "M09_컨택건수_이용유도_인터넷_B0M",
            "M10_컨택건수_이용유도_인터넷_B0M",
            "M11_컨택건수_이용유도_인터넷_B0M",
            "M12_컨택건수_이용유도_인터넷_B0M"
        ],
        "output": "M12_컨택건수_이용유도_인터넷_R6M",
        "fname": "cfs_07_0374",
        "type": "formula",
        "content": "M12_컨택건수_이용유도_인터넷_R6M = SUM(M07~M12_컨택건수_이용유도_인터넷_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_카드론_당사앱_B0M",
            "M08_컨택건수_카드론_당사앱_B0M",
            "M09_컨택건수_카드론_당사앱_B0M",
            "M10_컨택건수_카드론_당사앱_B0M",
            "M11_컨택건수_카드론_당사앱_B0M",
            "M12_컨택건수_카드론_당사앱_B0M"
        ],
        "output": "M12_컨택건수_카드론_당사앱_R6M",
        "fname": "cfs_07_0375",
        "type": "formula",
        "content": "M12_컨택건수_카드론_당사앱_R6M = SUM(M07~M12_컨택건수_카드론_당사앱_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_CA_당사앱_B0M",
            "M08_컨택건수_CA_당사앱_B0M",
            "M09_컨택건수_CA_당사앱_B0M",
            "M10_컨택건수_CA_당사앱_B0M",
            "M11_컨택건수_CA_당사앱_B0M",
            "M12_컨택건수_CA_당사앱_B0M"
        ],
        "output": "M12_컨택건수_CA_당사앱_R6M",
        "fname": "cfs_07_0376",
        "type": "formula",
        "content": "M12_컨택건수_CA_당사앱_R6M = SUM(M07~M12_컨택건수_CA_당사앱_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_리볼빙_당사앱_B0M",
            "M08_컨택건수_리볼빙_당사앱_B0M",
            "M09_컨택건수_리볼빙_당사앱_B0M",
            "M10_컨택건수_리볼빙_당사앱_B0M",
            "M11_컨택건수_리볼빙_당사앱_B0M",
            "M12_컨택건수_리볼빙_당사앱_B0M"
        ],
        "output": "M12_컨택건수_리볼빙_당사앱_R6M",
        "fname": "cfs_07_0377",
        "type": "formula",
        "content": "M12_컨택건수_리볼빙_당사앱_R6M = SUM(M07~M12_컨택건수_리볼빙_당사앱_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_이용유도_당사앱_B0M",
            "M08_컨택건수_이용유도_당사앱_B0M",
            "M09_컨택건수_이용유도_당사앱_B0M",
            "M10_컨택건수_이용유도_당사앱_B0M",
            "M11_컨택건수_이용유도_당사앱_B0M",
            "M12_컨택건수_이용유도_당사앱_B0M"
        ],
        "output": "M12_컨택건수_이용유도_당사앱_R6M",
        "fname": "cfs_07_0378",
        "type": "formula",
        "content": "M12_컨택건수_이용유도_당사앱_R6M = SUM(M07~M12_컨택건수_이용유도_당사앱_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_채권_B0M",
            "M08_컨택건수_채권_B0M",
            "M09_컨택건수_채권_B0M",
            "M10_컨택건수_채권_B0M",
            "M11_컨택건수_채권_B0M",
            "M12_컨택건수_채권_B0M"
        ],
        "output": "M12_컨택건수_채권_R6M",
        "fname": "cfs_07_0381",
        "type": "formula",
        "content": "M12_컨택건수_채권_R6M = SUM(M07~M12_컨택건수_채권_B0M)",
    },
    {
        "columns": [
            "M07_컨택건수_FDS_B0M",
            "M08_컨택건수_FDS_B0M",
            "M09_컨택건수_FDS_B0M",
            "M10_컨택건수_FDS_B0M",
            "M11_컨택건수_FDS_B0M",
            "M12_컨택건수_FDS_B0M"
        ],
        "output": "M12_컨택건수_FDS_R6M",
        "fname": "cfs_07_0382",
        "type": "formula",
        "content": "M12_컨택건수_FDS_R6M = SUM(M07~M12_컨택건수_FDS_B0M)",
    },

    # 8.성과 테이블 컬럼 Formula
    # M12
    {
        "columns": ["M12_잔액_신판최대한도소진율_r3m", "M09_잔액_신판최대한도소진율_r3m"],
        "output": "M12_잔액_신판최대한도소진율_r6m",
        "fname": "cfs_08_0444",
        "type": "formula",
        "content": "M12_잔액_신판최대한도소진율_r6m = MAX(M12_잔액_신판최대한도소진율_r3m, M09_잔액_신판최대한도소진율_r3m)",
    },
    {
        "columns": ["M12_잔액_신판ca최대한도소진율_r3m", "M09_잔액_신판ca최대한도소진율_r3m"],
        "output": "M12_잔액_신판ca최대한도소진율_r6m",
        "fname": "cfs_08_0448",
        "type": "formula",
        "content": "M12_잔액_신판ca최대한도소진율_r6m = MAX(M12_잔액_신판ca최대한도소진율_r3m, M09_잔액_신판ca최대한도소진율_r3m)",
    },

    # 3.승인.매출 테이블 컬럼 Constraint
    # M08
    {
        "columns": ["M07_최종이용일자_기본", "M08_최종이용일자_기본"],
        "fname": "ccs_03_0001",
        "type": "constraint",
        "content": "M07_최종이용일자_기본 <= M08_최종이용일자_기본",
    },
    {
        "columns": ["M07_최종이용일자_신판", "M08_최종이용일자_신판"],
        "fname": "ccs_03_0002",
        "type": "constraint",
        "content": "M07_최종이용일자_신판 <= M08_최종이용일자_신판",
    },
    {
        "columns": ["M07_최종이용일자_CA", "M08_최종이용일자_CA"],
        "fname": "ccs_03_0003",
        "type": "constraint",
        "content": "M07_최종이용일자_CA <= M08_최종이용일자_CA",
    },
    {
        "columns": ["M07_최종이용일자_카드론", "M08_최종이용일자_카드론"],
        "fname": "ccs_03_0004",
        "type": "constraint",
        "content": "M07_최종이용일자_카드론 <= M08_최종이용일자_카드론",
    },
    {
        "columns": ["M07_최종이용일자_체크", "M08_최종이용일자_체크"],
        "fname": "ccs_03_0005",
        "type": "constraint",
        "content": "M07_최종이용일자_체크 <= M08_최종이용일자_체크",
    },
    {
        "columns": ["M07_최종이용일자_일시불", "M08_최종이용일자_일시불"],
        "fname": "ccs_03_0006",
        "type": "constraint",
        "content": "M07_최종이용일자_일시불 <= M08_최종이용일자_일시불",
    },
    {
        "columns": ["M07_최종이용일자_할부", "M08_최종이용일자_할부"],
        "fname": "ccs_03_0007",
        "type": "constraint",
        "content": "M07_최종이용일자_할부 <= M08_최종이용일자_할부",
    },
    {
        "columns": ["M07_카드론이용건수_누적", "M08_카드론이용건수_누적"],
        "fname": "ccs_03_0008",
        "type": "constraint",
        "content": "M07_카드론이용건수_누적 <= M08_카드론이용건수_누적",
    },
    {
        "columns": ["M07_카드론이용월수_누적", "M08_카드론이용월수_누적"],
        "fname": "ccs_03_0009",
        "type": "constraint",
        "content": "M07_카드론이용월수_누적 <= M08_카드론이용월수_누적",
    },
    {
        "columns": ["M07_카드론이용금액_누적", "M08_카드론이용금액_누적"],
        "fname": "ccs_03_0010",
        "type": "constraint",
        "content": "M07_카드론이용금액_누적 <= M08_카드론이용금액_누적",
    },
    # M09
    {
        "columns": ["M08_최종이용일자_기본", "M09_최종이용일자_기본"],
        "fname": "ccs_03_0011",
        "type": "constraint",
        "content": "M08_최종이용일자_기본 <= M09_최종이용일자_기본",
    },
    {
        "columns": ["M08_최종이용일자_신판", "M09_최종이용일자_신판"],
        "fname": "ccs_03_0012",
        "type": "constraint",
        "content": "M08_최종이용일자_신판 <= M09_최종이용일자_신판",
    },
    {
        "columns": ["M08_최종이용일자_CA", "M09_최종이용일자_CA"],
        "fname": "ccs_03_0013",
        "type": "constraint",
        "content": "M08_최종이용일자_CA <= M09_최종이용일자_CA",
    },
    {
        "columns": ["M08_최종이용일자_카드론", "M09_최종이용일자_카드론"],
        "fname": "ccs_03_0014",
        "type": "constraint",
        "content": "M08_최종이용일자_카드론 <= M09_최종이용일자_카드론",
    },
    {
        "columns": ["M08_최종이용일자_체크", "M09_최종이용일자_체크"],
        "fname": "ccs_03_0015",
        "type": "constraint",
        "content": "M08_최종이용일자_체크 <= M09_최종이용일자_체크",
    },
    {
        "columns": ["M08_최종이용일자_일시불", "M09_최종이용일자_일시불"],
        "fname": "ccs_03_0016",
        "type": "constraint",
        "content": "M08_최종이용일자_일시불 <= M09_최종이용일자_일시불",
    },
    {
        "columns": ["M08_최종이용일자_할부", "M09_최종이용일자_할부"],
        "fname": "ccs_03_0017",
        "type": "constraint",
        "content": "M08_최종이용일자_할부 <= M09_최종이용일자_할부",
    },
    {
        "columns": ["M08_카드론이용건수_누적", "M09_카드론이용건수_누적"],
        "fname": "ccs_03_0018",
        "type": "constraint",
        "content": "M08_카드론이용건수_누적 <= M09_카드론이용건수_누적",
    },
    {
        "columns": ["M08_카드론이용월수_누적", "M09_카드론이용월수_누적"],
        "fname": "ccs_03_0019",
        "type": "constraint",
        "content": "M08_카드론이용월수_누적 <= M09_카드론이용월수_누적",
    },
    {
        "columns": ["M08_카드론이용금액_누적", "M09_카드론이용금액_누적"],
        "fname": "ccs_03_0020",
        "type": "constraint",
        "content": "M08_카드론이용금액_누적 <= M09_카드론이용금액_누적",
    },
    # M10
    {
        "columns": ["M09_최종이용일자_기본", "M10_최종이용일자_기본"],
        "fname": "ccs_03_0021",
        "type": "constraint",
        "content": "M09_최종이용일자_기본 <= M10_최종이용일자_기본",
    },
    {
        "columns": ["M09_최종이용일자_신판", "M10_최종이용일자_신판"],
        "fname": "ccs_03_0022",
        "type": "constraint",
        "content": "M09_최종이용일자_신판 <= M10_최종이용일자_신판",
    },
    {
        "columns": ["M09_최종이용일자_CA", "M10_최종이용일자_CA"],
        "fname": "ccs_03_0023",
        "type": "constraint",
        "content": "M09_최종이용일자_CA <= M10_최종이용일자_CA",
    },
    {
        "columns": ["M09_최종이용일자_카드론", "M10_최종이용일자_카드론"],
        "fname": "ccs_03_0024",
        "type": "constraint",
        "content": "M09_최종이용일자_카드론 <= M10_최종이용일자_카드론",
    },
    {
        "columns": ["M09_최종이용일자_체크", "M10_최종이용일자_체크"],
        "fname": "ccs_03_0025",
        "type": "constraint",
        "content": "M09_최종이용일자_체크 <= M10_최종이용일자_체크",
    },
    {
        "columns": ["M09_최종이용일자_일시불", "M10_최종이용일자_일시불"],
        "fname": "ccs_03_0026",
        "type": "constraint",
        "content": "M09_최종이용일자_일시불 <= M10_최종이용일자_일시불",
    },
    {
        "columns": ["M09_최종이용일자_할부", "M10_최종이용일자_할부"],
        "fname": "ccs_03_0027",
        "type": "constraint",
        "content": "M09_최종이용일자_할부 <= M10_최종이용일자_할부",
    },
    {
        "columns": ["M09_카드론이용건수_누적", "M10_카드론이용건수_누적"],
        "fname": "ccs_03_0028",
        "type": "constraint",
        "content": "M09_카드론이용건수_누적 <= M10_카드론이용건수_누적",
    },
    {
        "columns": ["M09_카드론이용월수_누적", "M10_카드론이용월수_누적"],
        "fname": "ccs_03_0029",
        "type": "constraint",
        "content": "M09_카드론이용월수_누적 <= M10_카드론이용월수_누적",
    },
    {
        "columns": ["M09_카드론이용금액_누적", "M10_카드론이용금액_누적"],
        "fname": "ccs_03_0030",
        "type": "constraint",
        "content": "M09_카드론이용금액_누적 <= M10_카드론이용금액_누적",
    },
    # M11
    {
        "columns": ["M10_최종이용일자_기본", "M11_최종이용일자_기본"],
        "fname": "ccs_03_0031",
        "type": "constraint",
        "content": "M10_최종이용일자_기본 <= M11_최종이용일자_기본",
    },
    {
        "columns": ["M10_최종이용일자_신판", "M11_최종이용일자_신판"],
        "fname": "ccs_03_0032",
        "type": "constraint",
        "content": "M10_최종이용일자_신판 <= M11_최종이용일자_신판",
    },
    {
        "columns": ["M10_최종이용일자_CA", "M11_최종이용일자_CA"],
        "fname": "ccs_03_0033",
        "type": "constraint",
        "content": "M10_최종이용일자_CA <= M11_최종이용일자_CA",
    },
    {
        "columns": ["M10_최종이용일자_카드론", "M11_최종이용일자_카드론"],
        "fname": "ccs_03_0034",
        "type": "constraint",
        "content": "M10_최종이용일자_카드론 <= M11_최종이용일자_카드론",
    },
    {
        "columns": ["M10_최종이용일자_체크", "M11_최종이용일자_체크"],
        "fname": "ccs_03_0035",
        "type": "constraint",
        "content": "M10_최종이용일자_체크 <= M11_최종이용일자_체크",
    },
    {
        "columns": ["M10_최종이용일자_일시불", "M11_최종이용일자_일시불"],
        "fname": "ccs_03_0036",
        "type": "constraint",
        "content": "M10_최종이용일자_일시불 <= M11_최종이용일자_일시불",
    },
    {
        "columns": ["M10_최종이용일자_할부", "M11_최종이용일자_할부"],
        "fname": "ccs_03_0037",
        "type": "constraint",
        "content": "M10_최종이용일자_할부 <= M11_최종이용일자_할부",
    },
    {
        "columns": ["M10_카드론이용건수_누적", "M11_카드론이용건수_누적"],
        "fname": "ccs_03_0038",
        "type": "constraint",
        "content": "M10_카드론이용건수_누적 <= M11_카드론이용건수_누적",
    },
    {
        "columns": ["M10_카드론이용월수_누적", "M11_카드론이용월수_누적"],
        "fname": "ccs_03_0039",
        "type": "constraint",
        "content": "M10_카드론이용월수_누적 <= M11_카드론이용월수_누적",
    },
    {
        "columns": ["M10_카드론이용금액_누적", "M11_카드론이용금액_누적"],
        "fname": "ccs_03_0040",
        "type": "constraint",
        "content": "M10_카드론이용금액_누적 <= M11_카드론이용금액_누적",
    },
    # M12
    {
        "columns": ["M11_최종이용일자_기본", "M12_최종이용일자_기본"],
        "fname": "ccs_03_0041",
        "type": "constraint",
        "content": "M11_최종이용일자_기본 <= M12_최종이용일자_기본",
    },
    {
        "columns": ["M11_최종이용일자_신판", "M12_최종이용일자_신판"],
        "fname": "ccs_03_0042",
        "type": "constraint",
        "content": "M11_최종이용일자_신판 <= M12_최종이용일자_신판",
    },
    {
        "columns": ["M11_최종이용일자_CA", "M12_최종이용일자_CA"],
        "fname": "ccs_03_0043",
        "type": "constraint",
        "content": "M11_최종이용일자_CA <= M12_최종이용일자_CA",
    },
    {
        "columns": ["M11_최종이용일자_카드론", "M12_최종이용일자_카드론"],
        "fname": "ccs_03_0044",
        "type": "constraint",
        "content": "M11_최종이용일자_카드론 <= M12_최종이용일자_카드론",
    },
    {
        "columns": ["M11_최종이용일자_체크", "M12_최종이용일자_체크"],
        "fname": "ccs_03_0045",
        "type": "constraint",
        "content": "M11_최종이용일자_체크 <= M12_최종이용일자_체크",
    },
    {
        "columns": ["M11_최종이용일자_일시불", "M12_최종이용일자_일시불"],
        "fname": "ccs_03_0046",
        "type": "constraint",
        "content": "M11_최종이용일자_일시불 <= M12_최종이용일자_일시불",
    },
    {
        "columns": ["M11_최종이용일자_할부", "M12_최종이용일자_할부"],
        "fname": "ccs_03_0047",
        "type": "constraint",
        "content": "M11_최종이용일자_할부 <= M12_최종이용일자_할부",
    },
    {
        "columns": ["M11_카드론이용건수_누적", "M12_카드론이용건수_누적"],
        "fname": "ccs_03_0048",
        "type": "constraint",
        "content": "M11_카드론이용건수_누적 <= M12_카드론이용건수_누적",
    },
    {
        "columns": ["M11_카드론이용월수_누적", "M12_카드론이용월수_누적"],
        "fname": "ccs_03_0049",
        "type": "constraint",
        "content": "M11_카드론이용월수_누적 <= M12_카드론이용월수_누적",
    },
    {
        "columns": ["M11_카드론이용금액_누적", "M12_카드론이용금액_누적"],
        "fname": "ccs_03_0050",
        "type": "constraint",
        "content": "M11_카드론이용금액_누적 <= M12_카드론이용금액_누적",
    },
    # 3.승인.매출 테이블 컬럼 Formula
    # M07
    {
        "columns": ["M07_RP건수_B0M", "M07_RP후경과월"],
        "output": "M07_RP후경과월",
        "fname": "cfs_03_0262",
        "type": "formula",
        "content": """IF M07_RP건수_B0M > 0 THEN M07_RP후경과월 = 0
                      ELSE M07_RP후경과월 = M07_RP후경과월.clip(1,6)""",
    },
    {
        "columns": ["M07_RP건수_통신_B0M", "M07_RP후경과월_통신"],
        "output": "M07_RP후경과월_통신",
        "fname": "cfs_03_0263",
        "type": "formula",
        "content": """IF M07_RP건수_통신_B0M > 0 THEN M07_RP후경과월_통신 = 0
                      ELSE M07_RP후경과월_통신 = M07_RP후경과월_통신.clip(1,6)""",
    },
    {
        "columns": ["M07_RP건수_아파트_B0M", "M07_RP후경과월_아파트"],
        "output": "M07_RP후경과월_아파트",
        "fname": "cfs_03_0264",
        "type": "formula",
        "content": """IF M07_RP건수_아파트_B0M > 0 THEN M07_RP후경과월_아파트 = 0
                      ELSE M07_RP후경과월_아파트 = M07_RP후경과월_아파트.clip(1,6)""",
    },
    {
        "columns": ["M07_RP건수_제휴사서비스직접판매_B0M", "M07_RP후경과월_제휴사서비스직접판매"],
        "output": "M07_RP후경과월_제휴사서비스직접판매",
        "fname": "cfs_03_0265",
        "type": "formula",
        "content": """IF M07_RP건수_제휴사서비스직접판매_B0M > 0 THEN M07_RP후경과월_제휴사서비스직접판매 = 0
                      ELSE M07_RP후경과월_제휴사서비스직접판매 = M07_RP후경과월_제휴사서비스직접판매.clip(1,6)""",
    },
    {
        "columns": ["M07_RP건수_렌탈_B0M", "M07_RP후경과월_렌탈"],
        "output": "M07_RP후경과월_렌탈",
        "fname": "cfs_03_0266",
        "type": "formula",
        "content": """IF M07_RP건수_렌탈_B0M > 0 THEN M07_RP후경과월_렌탈 = 0
                      ELSE M07_RP후경과월_렌탈 = M07_RP후경과월_렌탈.clip(1,6)""",
    },
    {
        "columns": ["M07_RP건수_가스_B0M", "M07_RP후경과월_가스"],
        "output": "M07_RP후경과월_가스",
        "fname": "cfs_03_0267",
        "type": "formula",
        "content": """IF M07_RP건수_가스_B0M > 0 THEN M07_RP후경과월_가스 = 0
                      ELSE M07_RP후경과월_가스 = M07_RP후경과월_가스.clip(1,6)""",
    },
    {
        "columns": ["M07_RP건수_전기_B0M", "M07_RP후경과월_전기"],
        "output": "M07_RP후경과월_전기",
        "fname": "cfs_03_0268",
        "type": "formula",
        "content": """IF M07_RP건수_전기_B0M > 0 THEN M07_RP후경과월_전기 = 0
                      ELSE M07_RP후경과월_전기 = M07_RP후경과월_전기.clip(1,6)""",
    },
    {
        "columns": ["M07_RP건수_보험_B0M", "M07_RP후경과월_보험"],
        "output": "M07_RP후경과월_보험",
        "fname": "cfs_03_0269",
        "type": "formula",
        "content": """IF M07_RP건수_보험_B0M > 0 THEN M07_RP후경과월_보험 = 0
                      ELSE M07_RP후경과월_보험 = M07_RP후경과월_보험.clip(1,6)""",
    },
    {
        "columns": ["M07_RP건수_학습비_B0M", "M07_RP후경과월_학습비"],
        "output": "M07_RP후경과월_학습비",
        "fname": "cfs_03_0270",
        "type": "formula",
        "content": """IF M07_RP건수_학습비_B0M > 0 THEN M07_RP후경과월_학습비 = 0
                      ELSE M07_RP후경과월_학습비 = M07_RP후경과월_학습비.clip(1,6)""",
    },
    {
        "columns": ["M07_RP건수_유선방송_B0M", "M07_RP후경과월_유선방송"],
        "output": "M07_RP후경과월_유선방송",
        "fname": "cfs_03_0271",
        "type": "formula",
        "content": """IF M07_RP건수_유선방송_B0M > 0 THEN M07_RP후경과월_유선방송 = 0
                      ELSE M07_RP후경과월_유선방송 = M07_RP후경과월_유선방송.clip(1,6)""",
    },
    {
        "columns": ["M07_RP건수_건강_B0M", "M07_RP후경과월_건강"],
        "output": "M07_RP후경과월_건강",
        "fname": "cfs_03_0272",
        "type": "formula",
        "content": """IF M07_RP건수_건강_B0M > 0 THEN M07_RP후경과월_건강 = 0
                      ELSE M07_RP후경과월_건강 = M07_RP후경과월_건강.clip(1,6)""",
    },
    {
        "columns": ["M07_RP건수_교통_B0M", "M07_RP후경과월_교통"],
        "output": "M07_RP후경과월_교통",
        "fname": "cfs_03_0273",
        "type": "formula",
        "content": """IF M07_RP건수_교통_B0M > 0 THEN M07_RP후경과월_교통 = 0
                      ELSE M07_RP후경과월_교통 = M07_RP후경과월_교통.clip(1,6)""",
    },
    # M08
    {
        "columns": ["M08_RP건수_B0M", "M07_RP후경과월"],
        "output": "M08_RP후경과월",
        "fname": "cfs_03_0723",
        "type": "formula",
        "content": """IF M08_RP건수_B0M > 0 THEN M08_RP후경과월 = 0
                      ELSE M08_RP후경과월 = min(M07_RP후경과월 + 1, 6)""",
    },
    {
        "columns": ["M08_RP건수_통신_B0M", "M07_RP후경과월_통신"],
        "output": "M08_RP후경과월_통신",
        "fname": "cfs_03_0724",
        "type": "formula",
        "content": """IF M08_RP건수_통신_B0M > 0 THEN M08_RP후경과월_통신 = 0
                      ELSE M08_RP후경과월_통신 = min(M07_RP후경과월_통신 + 1, 6)""",
    },
    {
        "columns": ["M08_RP건수_아파트_B0M", "M07_RP후경과월_아파트"],
        "output": "M08_RP후경과월_아파트",
        "fname": "cfs_03_0725",
        "type": "formula",
        "content": """IF M08_RP건수_아파트_B0M > 0 THEN M08_RP후경과월_아파트 = 0
                      ELSE M08_RP후경과월_아파트 = min(M07_RP후경과월_아파트 + 1, 6)""",
    },
    {
        "columns": ["M08_RP건수_제휴사서비스직접판매_B0M", "M07_RP후경과월_제휴사서비스직접판매"],
        "output": "M08_RP후경과월_제휴사서비스직접판매",
        "fname": "cfs_03_0726",
        "type": "formula",
        "content": """IF M08_RP건수_제휴사서비스직접판매_B0M > 0 THEN M08_RP후경과월_제휴사서비스직접판매 = 0
                      ELSE M08_RP후경과월_제휴사서비스직접판매 = min(M07_RP후경과월_제휴사서비스직접판매 + 1, 6)""",
    },
    {
        "columns": ["M08_RP건수_렌탈_B0M", "M07_RP후경과월_렌탈"],
        "output": "M08_RP후경과월_렌탈",
        "fname": "cfs_03_0727",
        "type": "formula",
        "content": """IF M08_RP건수_렌탈_B0M > 0 THEN M08_RP후경과월_렌탈 = 0
                      ELSE M08_RP후경과월_렌탈 = min(M07_RP후경과월_렌탈 + 1, 6)""",
    },
    {
        "columns": ["M08_RP건수_가스_B0M", "M07_RP후경과월_가스"],
        "output": "M08_RP후경과월_가스",
        "fname": "cfs_03_0728",
        "type": "formula",
        "content": """IF M08_RP건수_가스_B0M > 0 THEN M08_RP후경과월_가스 = 0
                      ELSE M08_RP후경과월_가스 = min(M07_RP후경과월_가스 + 1, 6)""",
    },
    {
        "columns": ["M08_RP건수_전기_B0M", "M07_RP후경과월_전기"],
        "output": "M08_RP후경과월_전기",
        "fname": "cfs_03_0729",
        "type": "formula",
        "content": """IF M08_RP건수_전기_B0M > 0 THEN M08_RP후경과월_전기 = 0
                      ELSE M08_RP후경과월_전기 = min(M07_RP후경과월_전기 + 1, 6)""",
    },
    {
        "columns": ["M08_RP건수_보험_B0M", "M07_RP후경과월_보험"],
        "output": "M08_RP후경과월_보험",
        "fname": "cfs_03_0730",
        "type": "formula",
        "content": """IF M08_RP건수_보험_B0M > 0 THEN M08_RP후경과월_보험 = 0
                      ELSE M08_RP후경과월_보험 = min(M07_RP후경과월_보험 + 1, 6)""",
    },
    {
        "columns": ["M08_RP건수_학습비_B0M", "M07_RP후경과월_학습비"],
        "output": "M08_RP후경과월_학습비",
        "fname": "cfs_03_0731",
        "type": "formula",
        "content": """IF M08_RP건수_학습비_B0M > 0 THEN M08_RP후경과월_학습비 = 0
                      ELSE M08_RP후경과월_학습비 = min(M07_RP후경과월_학습비 + 1, 6)""",
    },
    {
        "columns": ["M08_RP건수_유선방송_B0M", "M07_RP후경과월_유선방송"],
        "output": "M08_RP후경과월_유선방송",
        "fname": "cfs_03_0732",
        "type": "formula",
        "content": """IF M08_RP건수_유선방송_B0M > 0 THEN M08_RP후경과월_유선방송 = 0
                      ELSE M08_RP후경과월_유선방송 = min(M07_RP후경과월_유선방송 + 1, 6)""",
    },
    {
        "columns": ["M08_RP건수_건강_B0M", "M07_RP후경과월_건강"],
        "output": "M08_RP후경과월_건강",
        "fname": "cfs_03_0733",
        "type": "formula",
        "content": """IF M08_RP건수_건강_B0M > 0 THEN M08_RP후경과월_건강 = 0
                      ELSE M08_RP후경과월_건강 = min(M07_RP후경과월_건강 + 1, 6)""",
    },
    {
        "columns": ["M08_RP건수_교통_B0M", "M07_RP후경과월_교통"],
        "output": "M08_RP후경과월_교통",
        "fname": "cfs_03_0734",
        "type": "formula",
        "content": """IF M08_RP건수_교통_B0M > 0 THEN M08_RP후경과월_교통 = 0
                      ELSE M08_RP후경과월_교통 = min(M07_RP후경과월_교통 + 1, 6)""",
    },
    {
        "columns": ["M08_최종카드론_대출일자", "M07_최종카드론_대출일자", "M07_최종카드론이용경과월"],
        "output": "M08_최종카드론이용경과월",
        "fname": "cfs_03_0736",
        "type": "formula",
        "content": """IF M08_최종카드론_대출일자 IS NULL
                      THEN M08_최종카드론이용경과월 = 999
                      ELIF M08_최종카드론_대출일자 = M07_최종카드론_대출일자
                      THEN M08_최종카드론이용경과월 = M07_최종카드론이용경과월 + 1
                      ELSE M08_최종카드론이용경과월 = 0""",
    },
    {
        "columns": ["M07_가맹점매출금액_B1M"],
        "output": "M08_가맹점매출금액_B2M",
        "fname": "cfs_03_0843",
        "type": "formula",
        "content": "M08_가맹점매출금액_B2M = M07_가맹점매출금액_B1M",
    },
    {
        "columns": ["M07_RP건수_B0M", "M08_RP건수_B0M"],
        "output": "M08_증감_RP건수_전월",
        "fname": "cfs_03_0876",
        "type": "formula",
        "content": "M08_증감_RP건수_전월 = M07_RP건수_B0M - M08_RP건수_B0M",
    },
    {
        "columns": ["M07_RP건수_통신_B0M", "M08_RP건수_통신_B0M"],
        "output": "M08_증감_RP건수_통신_전월",
        "fname": "cfs_03_0878",
        "type": "formula",
        "content": "M08_증감_RP건수_통신_전월 = M07_RP건수_통신_B0M - M08_RP건수_통신_B0M",
    },
    {
        "columns": ["M07_RP건수_아파트_B0M", "M08_RP건수_아파트_B0M"],
        "output": "M08_증감_RP건수_아파트_전월",
        "fname": "cfs_03_0879",
        "type": "formula",
        "content": "M08_증감_RP건수_아파트_전월 = M07_RP건수_아파트_B0M - M08_RP건수_아파트_B0M",
    },
    {
        "columns": ["M07_RP건수_제휴사서비스직접판매_B0M", "M08_RP건수_제휴사서비스직접판매_B0M"],
        "output": "M08_증감_RP건수_제휴사서비스직접판매_전월",
        "fname": "cfs_03_0880",
        "type": "formula",
        "content": "M08_증감_RP건수_제휴사서비스직접판매_전월 = M07_RP건수_제휴사서비스직접판매_B0M - M08_RP건수_제휴사서비스직접판매_B0M",
    },
    {
        "columns": ["M07_RP건수_렌탈_B0M", "M08_RP건수_렌탈_B0M"],
        "output": "M08_증감_RP건수_렌탈_전월",
        "fname": "cfs_03_0881",
        "type": "formula",
        "content": "M08_증감_RP건수_렌탈_전월 = M07_RP건수_렌탈_B0M - M08_RP건수_렌탈_B0M",
    },
    {
        "columns": ["M07_RP건수_가스_B0M", "M08_RP건수_가스_B0M"],
        "output": "M08_증감_RP건수_가스_전월",
        "fname": "cfs_03_0882",
        "type": "formula",
        "content": "M08_증감_RP건수_가스_전월 = M07_RP건수_가스_B0M - M08_RP건수_가스_B0M",
    },
    {
        "columns": ["M07_RP건수_전기_B0M", "M08_RP건수_전기_B0M"],
        "output": "M08_증감_RP건수_전기_전월",
        "fname": "cfs_03_0883",
        "type": "formula",
        "content": "M08_증감_RP건수_전기_전월 = M07_RP건수_전기_B0M - M08_RP건수_전기_B0M",
    },
    {
        "columns": ["M07_RP건수_보험_B0M", "M08_RP건수_보험_B0M"],
        "output": "M08_증감_RP건수_보험_전월",
        "fname": "cfs_03_0884",
        "type": "formula",
        "content": "M08_증감_RP건수_보험_전월 = M07_RP건수_보험_B0M - M08_RP건수_보험_B0M",
    },
    {
        "columns": ["M07_RP건수_학습비_B0M", "M08_RP건수_학습비_B0M"],
        "output": "M08_증감_RP건수_학습비_전월",
        "fname": "cfs_03_0885",
        "type": "formula",
        "content": "M08_증감_RP건수_학습비_전월 = M07_RP건수_학습비_B0M - M08_RP건수_학습비_B0M",
    },
    {
        "columns": ["M07_RP건수_유선방송_B0M", "M08_RP건수_유선방송_B0M"],
        "output": "M08_증감_RP건수_유선방송_전월",
        "fname": "cfs_03_0886",
        "type": "formula",
        "content": "M08_증감_RP건수_유선방송_전월 = M07_RP건수_유선방송_B0M - M08_RP건수_유선방송_B0M",
    },
    {
        "columns": ["M07_RP건수_건강_B0M", "M08_RP건수_건강_B0M"],
        "output": "M08_증감_RP건수_건강_전월",
        "fname": "cfs_03_0887",
        "type": "formula",
        "content": "M08_증감_RP건수_건강_전월 = M07_RP건수_건강_B0M - M08_RP건수_건강_B0M",
    },
    {
        "columns": ["M07_RP건수_교통_B0M", "M08_RP건수_교통_B0M"],
        "output": "M08_증감_RP건수_교통_전월",
        "fname": "cfs_03_0888",
        "type": "formula",
        "content": "M08_증감_RP건수_교통_전월 = M07_RP건수_교통_B0M - M08_RP건수_교통_B0M",
    },
    # M09
    {
        "columns": ["M09_RP건수_B0M", "M08_RP후경과월"],
        "output": "M09_RP후경과월",
        "fname": "cfs_03_1184",
        "type": "formula",
        "content": """IF M09_RP건수_B0M > 0 THEN M09_RP후경과월 = 0
                      ELSE M09_RP후경과월 = min(M08_RP후경과월 + 1, 6)""",
    },
    {
        "columns": ["M09_RP건수_통신_B0M", "M08_RP후경과월_통신"],
        "output": "M09_RP후경과월_통신",
        "fname": "cfs_03_1185",
        "type": "formula",
        "content": """IF M09_RP건수_통신_B0M > 0 THEN M09_RP후경과월_통신 = 0
                      ELSE M09_RP후경과월_통신 = min(M08_RP후경과월_통신 + 1, 6)""",
    },
    {
        "columns": ["M09_RP건수_아파트_B0M", "M08_RP후경과월_아파트"],
        "output": "M09_RP후경과월_아파트",
        "fname": "cfs_03_1186",
        "type": "formula",
        "content": """IF M09_RP건수_아파트_B0M > 0 THEN M09_RP후경과월_아파트 = 0
                      ELSE M09_RP후경과월_아파트 = min(M08_RP후경과월_아파트 + 1, 6)""",
    },
    {
        "columns": ["M09_RP건수_제휴사서비스직접판매_B0M", "M08_RP후경과월_제휴사서비스직접판매"],
        "output": "M09_RP후경과월_제휴사서비스직접판매",
        "fname": "cfs_03_1187",
        "type": "formula",
        "content": """IF M09_RP건수_제휴사서비스직접판매_B0M > 0 THEN M09_RP후경과월_제휴사서비스직접판매 = 0
                      ELSE M09_RP후경과월_제휴사서비스직접판매 = min(M08_RP후경과월_제휴사서비스직접판매 + 1, 6)""",
    },
    {
        "columns": ["M09_RP건수_렌탈_B0M", "M08_RP후경과월_렌탈"],
        "output": "M09_RP후경과월_렌탈",
        "fname": "cfs_03_1188",
        "type": "formula",
        "content": """IF M09_RP건수_렌탈_B0M > 0 THEN M09_RP후경과월_렌탈 = 0
                      ELSE M09_RP후경과월_렌탈 = min(M08_RP후경과월_렌탈 + 1, 6)""",
    },
    {
        "columns": ["M09_RP건수_가스_B0M", "M08_RP후경과월_가스"],
        "output": "M09_RP후경과월_가스",
        "fname": "cfs_03_1189",
        "type": "formula",
        "content": """IF M09_RP건수_가스_B0M > 0 THEN M09_RP후경과월_가스 = 0
                      ELSE M09_RP후경과월_가스 = min(M08_RP후경과월_가스 + 1, 6)""",
    },
    {
        "columns": ["M09_RP건수_전기_B0M", "M08_RP후경과월_전기"],
        "output": "M09_RP후경과월_전기",
        "fname": "cfs_03_1190",
        "type": "formula",
        "content": """IF M09_RP건수_전기_B0M > 0 THEN M09_RP후경과월_전기 = 0
                      ELSE M09_RP후경과월_전기 = min(M08_RP후경과월_전기 + 1, 6)""",
    },
    {
        "columns": ["M09_RP건수_보험_B0M", "M08_RP후경과월_보험"],
        "output": "M09_RP후경과월_보험",
        "fname": "cfs_03_1191",
        "type": "formula",
        "content": """IF M09_RP건수_보험_B0M > 0 THEN M09_RP후경과월_보험 = 0
                      ELSE M09_RP후경과월_보험 = min(M08_RP후경과월_보험 + 1, 6)""",
    },
    {
        "columns": ["M09_RP건수_학습비_B0M", "M08_RP후경과월_학습비"],
        "output": "M09_RP후경과월_학습비",
        "fname": "cfs_03_1192",
        "type": "formula",
        "content": """IF M09_RP건수_학습비_B0M > 0 THEN M09_RP후경과월_학습비 = 0
                      ELSE M09_RP후경과월_학습비 = min(M08_RP후경과월_학습비 + 1, 6)""",
    },
    {
        "columns": ["M09_RP건수_유선방송_B0M", "M08_RP후경과월_유선방송"],
        "output": "M09_RP후경과월_유선방송",
        "fname": "cfs_03_1193",
        "type": "formula",
        "content": """IF M09_RP건수_유선방송_B0M > 0 THEN M09_RP후경과월_유선방송 = 0
                      ELSE M09_RP후경과월_유선방송 = min(M08_RP후경과월_유선방송 + 1, 6)""",
    },
    {
        "columns": ["M09_RP건수_건강_B0M", "M08_RP후경과월_건강"],
        "output": "M09_RP후경과월_건강",
        "fname": "cfs_03_1194",
        "type": "formula",
        "content": """IF M09_RP건수_건강_B0M > 0 THEN M09_RP후경과월_건강 = 0
                      ELSE M09_RP후경과월_건강 = min(M08_RP후경과월_건강 + 1, 6)""",
    },
    {
        "columns": ["M09_RP건수_교통_B0M", "M08_RP후경과월_교통"],
        "output": "M09_RP후경과월_교통",
        "fname": "cfs_03_1195",
        "type": "formula",
        "content": """IF M09_RP건수_교통_B0M > 0 THEN M09_RP후경과월_교통 = 0
                      ELSE M09_RP후경과월_교통 = min(M08_RP후경과월_교통 + 1, 6)""",
    },
    {
        "columns": ["M07_승인거절건수_B0M", "M08_승인거절건수_B0M", "M09_승인거절건수_B0M"],
        "output": "M09_승인거절건수_R3M",
        "fname": "cfs_03_1369",
        "type": "formula",
        "content": "M09_승인거절건수_R3M = M07_승인거절건수_B0M + M08_승인거절건수_B0M + M09_승인거절건수_B0M",
    },
    {
        "columns": ["M07_승인거절건수_한도초과_B0M", "M08_승인거절건수_한도초과_B0M", "M09_승인거절건수_한도초과_B0M"],
        "output": "M09_승인거절건수_한도초과_R3M",
        "fname": "cfs_03_1370",
        "type": "formula",
        "content": "M09_승인거절건수_한도초과_R3M = M07_승인거절건수_한도초과_B0M + M08_승인거절건수_한도초과_B0M + M09_승인거절건수_한도초과_B0M",
    },
    {
        "columns": ["M07_승인거절건수_BL_B0M", "M08_승인거절건수_BL_B0M", "M09_승인거절건수_BL_B0M"],
        "output": "M09_승인거절건수_BL_R3M",
        "fname": "cfs_03_1371",
        "type": "formula",
        "content": "M09_승인거절건수_BL_R3M = M07_승인거절건수_BL_B0M + M08_승인거절건수_BL_B0M + M09_승인거절건수_BL_B0M",
    },
    {
        "columns": ["M07_승인거절건수_입력오류_B0M", "M08_승인거절건수_입력오류_B0M", "M09_승인거절건수_입력오류_B0M"],
        "output": "M09_승인거절건수_입력오류_R3M",
        "fname": "cfs_03_1372",
        "type": "formula",
        "content": "M09_승인거절건수_입력오류_R3M = M07_승인거절건수_입력오류_B0M + M08_승인거절건수_입력오류_B0M + M09_승인거절건수_입력오류_B0M",
    },
    {
        "columns": ["M07_승인거절건수_기타_B0M", "M08_승인거절건수_기타_B0M", "M09_승인거절건수_기타_B0M"],
        "output": "M09_승인거절건수_기타_R3M",
        "fname": "cfs_03_1373",
        "type": "formula",
        "content": "M09_승인거절건수_기타_R3M = M07_승인거절건수_기타_B0M + M08_승인거절건수_기타_B0M + M09_승인거절건수_기타_B0M",
    },
    {
        "columns": ["M09_이용건수_신용_B0M", "M08_이용건수_신용_B0M", "M07_이용건수_신용_B0M"],
        "output": "M09_이용건수_신용_R3M",
        "fname": "cfs_03_1032",
        "type": "formula",
        "content": "M09_이용건수_신용_R3M = M09_이용건수_신용_B0M + M08_이용건수_신용_B0M + M07_이용건수_신용_B0M",
    },
    {
        "columns": ["M09_이용건수_신판_B0M", "M08_이용건수_신판_B0M", "M07_이용건수_신판_B0M"],
        "output": "M09_이용건수_신판_R3M",
        "fname": "cfs_03_1033",
        "type": "formula",
        "content": "M09_이용건수_신판_R3M = M09_이용건수_신판_B0M + M08_이용건수_신판_B0M + M07_이용건수_신판_B0M",
    },
    {
        "columns": ["M09_이용건수_일시불_B0M", "M08_이용건수_일시불_B0M", "M07_이용건수_일시불_B0M"],
        "output": "M09_이용건수_일시불_R3M",
        "fname": "cfs_03_1034",
        "type": "formula",
        "content": "M09_이용건수_일시불_R3M = M09_이용건수_일시불_B0M + M08_이용건수_일시불_B0M + M07_이용건수_일시불_B0M",
    },
    {
        "columns": ["M09_이용건수_할부_B0M", "M08_이용건수_할부_B0M", "M07_이용건수_할부_B0M"],
        "output": "M09_이용건수_할부_R3M",
        "fname": "cfs_03_1035",
        "type": "formula",
        "content": "M09_이용건수_할부_R3M = M09_이용건수_할부_B0M + M08_이용건수_할부_B0M + M07_이용건수_할부_B0M",
    },
    {
        "columns": ["M09_이용건수_할부_유이자_B0M", "M08_이용건수_할부_유이자_B0M", "M07_이용건수_할부_유이자_B0M"],
        "output": "M09_이용건수_할부_유이자_R3M",
        "fname": "cfs_03_1036",
        "type": "formula",
        "content": "M09_이용건수_할부_유이자_R3M = M09_이용건수_할부_유이자_B0M + M08_이용건수_할부_유이자_B0M + M07_이용건수_할부_유이자_B0M",
    },
    {
        "columns": ["M09_이용건수_할부_무이자_B0M", "M08_이용건수_할부_무이자_B0M", "M07_이용건수_할부_무이자_B0M"],
        "output": "M09_이용건수_할부_무이자_R3M",
        "fname": "cfs_03_1037",
        "type": "formula",
        "content": "M09_이용건수_할부_무이자_R3M = M09_이용건수_할부_무이자_B0M + M08_이용건수_할부_무이자_B0M + M07_이용건수_할부_무이자_B0M",
    },
    {
        "columns": ["M09_이용건수_부분무이자_B0M", "M08_이용건수_부분무이자_B0M", "M07_이용건수_부분무이자_B0M"],
        "output": "M09_이용건수_부분무이자_R3M",
        "fname": "cfs_03_1038",
        "type": "formula",
        "content": "M09_이용건수_부분무이자_R3M = M09_이용건수_부분무이자_B0M + M08_이용건수_부분무이자_B0M + M07_이용건수_부분무이자_B0M",
    },
    {
        "columns": ["M09_이용건수_CA_B0M", "M08_이용건수_CA_B0M", "M07_이용건수_CA_B0M"],
        "output": "M09_이용건수_CA_R3M",
        "fname": "cfs_03_1039",
        "type": "formula",
        "content": "M09_이용건수_CA_R3M = M09_이용건수_CA_B0M + M08_이용건수_CA_B0M + M07_이용건수_CA_B0M",
    },
    {
        "columns": ["M09_이용건수_체크_B0M", "M08_이용건수_체크_B0M", "M07_이용건수_체크_B0M"],
        "output": "M09_이용건수_체크_R3M",
        "fname": "cfs_03_1040",
        "type": "formula",
        "content": "M09_이용건수_체크_R3M = M09_이용건수_체크_B0M + M08_이용건수_체크_B0M + M07_이용건수_체크_B0M",
    },
    {
        "columns": ["M09_이용건수_카드론_B0M", "M08_이용건수_카드론_B0M", "M07_이용건수_카드론_B0M"],
        "output": "M09_이용건수_카드론_R3M",
        "fname": "cfs_03_1041",
        "type": "formula",
        "content": "M09_이용건수_카드론_R3M = M09_이용건수_카드론_B0M + M08_이용건수_카드론_B0M + M07_이용건수_카드론_B0M",
    },
    {
        "columns": ["M09_이용금액_신용_B0M", "M08_이용금액_신용_B0M", "M07_이용금액_신용_B0M"],
        "output": "M09_이용금액_신용_R3M",
        "fname": "cfs_03_1042",
        "type": "formula",
        "content": "M09_이용금액_신용_R3M = M09_이용금액_신용_B0M + M08_이용금액_신용_B0M + M07_이용금액_신용_B0M",
    },
    {
        "columns": ["M09_이용금액_신판_B0M", "M08_이용금액_신판_B0M", "M07_이용금액_신판_B0M"],
        "output": "M09_이용금액_신판_R3M",
        "fname": "cfs_03_1043",
        "type": "formula",
        "content": "M09_이용금액_신판_R3M = M09_이용금액_신판_B0M + M08_이용금액_신판_B0M + M07_이용금액_신판_B0M",
    },
    {
        "columns": ["M09_이용금액_일시불_B0M", "M08_이용금액_일시불_B0M", "M07_이용금액_일시불_B0M"],
        "output": "M09_이용금액_일시불_R3M",
        "fname": "cfs_03_1044",
        "type": "formula",
        "content": "M09_이용금액_일시불_R3M = M09_이용금액_일시불_B0M + M08_이용금액_일시불_B0M + M07_이용금액_일시불_B0M",
    },
    {
        "columns": ["M09_이용금액_할부_B0M", "M08_이용금액_할부_B0M", "M07_이용금액_할부_B0M"],
        "output": "M09_이용금액_할부_R3M",
        "fname": "cfs_03_1045",
        "type": "formula",
        "content": "M09_이용금액_할부_R3M = M09_이용금액_할부_B0M + M08_이용금액_할부_B0M + M07_이용금액_할부_B0M",
    },
    {
        "columns": ["M09_이용금액_할부_유이자_B0M", "M08_이용금액_할부_유이자_B0M", "M07_이용금액_할부_유이자_B0M"],
        "output": "M09_이용금액_할부_유이자_R3M",
        "fname": "cfs_03_1046",
        "type": "formula",
        "content": "M09_이용금액_할부_유이자_R3M = M09_이용금액_할부_유이자_B0M + M08_이용금액_할부_유이자_B0M + M07_이용금액_할부_유이자_B0M",
    },
    {
        "columns": ["M09_이용금액_할부_무이자_B0M", "M08_이용금액_할부_무이자_B0M", "M07_이용금액_할부_무이자_B0M"],
        "output": "M09_이용금액_할부_무이자_R3M",
        "fname": "cfs_03_1047",
        "type": "formula",
        "content": "M09_이용금액_할부_무이자_R3M = M09_이용금액_할부_무이자_B0M + M08_이용금액_할부_무이자_B0M + M07_이용금액_할부_무이자_B0M",
    },
    {
        "columns": ["M09_이용금액_부분무이자_B0M", "M08_이용금액_부분무이자_B0M", "M07_이용금액_부분무이자_B0M"],
        "output": "M09_이용금액_부분무이자_R3M",
        "fname": "cfs_03_1048",
        "type": "formula",
        "content": "M09_이용금액_부분무이자_R3M = M09_이용금액_부분무이자_B0M + M08_이용금액_부분무이자_B0M + M07_이용금액_부분무이자_B0M",
    },
    {
        "columns": ["M09_이용금액_CA_B0M", "M08_이용금액_CA_B0M", "M07_이용금액_CA_B0M"],
        "output": "M09_이용금액_CA_R3M",
        "fname": "cfs_03_1049",
        "type": "formula",
        "content": "M09_이용금액_CA_R3M = M09_이용금액_CA_B0M + M08_이용금액_CA_B0M + M07_이용금액_CA_B0M",
    },
    {
        "columns": ["M09_이용금액_체크_B0M", "M08_이용금액_체크_B0M", "M07_이용금액_체크_B0M"],
        "output": "M09_이용금액_체크_R3M",
        "fname": "cfs_03_1050",
        "type": "formula",
        "content": "M09_이용금액_체크_R3M = M09_이용금액_체크_B0M + M08_이용금액_체크_B0M + M07_이용금액_체크_B0M",
    },
    {
        "columns": ["M09_이용금액_카드론_B0M", "M08_이용금액_카드론_B0M", "M07_이용금액_카드론_B0M"],
        "output": "M09_이용금액_카드론_R3M",
        "fname": "cfs_03_1051",
        "type": "formula",
        "content": "M09_이용금액_카드론_R3M = M09_이용금액_카드론_B0M + M08_이용금액_카드론_B0M + M07_이용금액_카드론_B0M",
    },
    {
        "columns": ["M09_최종카드론_대출일자", "M08_최종카드론_대출일자", "M08_최종카드론이용경과월"],
        "output": "M09_최종카드론이용경과월",
        "fname": "cfs_03_1197",
        "type": "formula",
        "content": """IF M09_최종카드론_대출일자 IS NULL
                      THEN M09_최종카드론이용경과월 = 999
                      ELIF M09_최종카드론_대출일자 = M08_최종카드론_대출일자
                      THEN M09_최종카드론이용경과월 = M08_최종카드론이용경과월 + 1
                      ELSE M09_최종카드론이용경과월 = 0""",
    },
    {
        "columns": ["M09_이용금액_온라인_B0M", "M08_이용금액_온라인_B0M", "M07_이용금액_온라인_B0M"],
        "output": "M09_이용금액_온라인_R3M",
        "fname": "cfs_03_1222",
        "type": "formula",
        "content": "M09_이용금액_온라인_R3M = M09_이용금액_온라인_B0M + M08_이용금액_온라인_B0M + M07_이용금액_온라인_B0M",
    },
    {
        "columns": ["M09_이용금액_오프라인_B0M", "M08_이용금액_오프라인_B0M", "M07_이용금액_오프라인_B0M"],
        "output": "M09_이용금액_오프라인_R3M",
        "fname": "cfs_03_1223",
        "type": "formula",
        "content": "M09_이용금액_오프라인_R3M = M09_이용금액_오프라인_B0M + M08_이용금액_오프라인_B0M + M07_이용금액_오프라인_B0M",
    },
    {
        "columns": ["M09_이용건수_온라인_B0M", "M08_이용건수_온라인_B0M", "M07_이용건수_온라인_B0M"],
        "output": "M09_이용건수_온라인_R3M",
        "fname": "cfs_03_1224",
        "type": "formula",
        "content": "M09_이용건수_온라인_R3M = M09_이용건수_온라인_B0M + M08_이용건수_온라인_B0M + M07_이용건수_온라인_B0M",
    },
    {
        "columns": ["M09_이용건수_오프라인_B0M", "M08_이용건수_오프라인_B0M", "M07_이용건수_오프라인_B0M"],
        "output": "M09_이용건수_오프라인_R3M",
        "fname": "cfs_03_1225",
        "type": "formula",
        "content": "M09_이용건수_오프라인_R3M = M09_이용건수_오프라인_B0M + M08_이용건수_오프라인_B0M + M07_이용건수_오프라인_B0M",
    },
    {
        "columns": ["M09_이용금액_페이_온라인_B0M", "M08_이용금액_페이_온라인_B0M", "M07_이용금액_페이_온라인_B0M"],
        "output": "M09_이용금액_페이_온라인_R3M",
        "fname": "cfs_03_1236",
        "type": "formula",
        "content": "M09_이용금액_페이_온라인_R3M = M09_이용금액_페이_온라인_B0M + M08_이용금액_페이_온라인_B0M + M07_이용금액_페이_온라인_B0M",
    },
    {
        "columns": ["M09_이용금액_페이_오프라인_B0M", "M08_이용금액_페이_오프라인_B0M", "M07_이용금액_페이_오프라인_B0M"],
        "output": "M09_이용금액_페이_오프라인_R3M",
        "fname": "cfs_03_1237",
        "type": "formula",
        "content": "M09_이용금액_페이_오프라인_R3M = M09_이용금액_페이_오프라인_B0M + M08_이용금액_페이_오프라인_B0M + M07_이용금액_페이_오프라인_B0M",
    },
    {
        "columns": ["M09_이용건수_페이_온라인_B0M", "M08_이용건수_페이_온라인_B0M", "M07_이용건수_페이_온라인_B0M"],
        "output": "M09_이용건수_페이_온라인_R3M",
        "fname": "cfs_03_1238",
        "type": "formula",
        "content": "M09_이용건수_페이_온라인_R3M = M09_이용건수_페이_온라인_B0M + M08_이용건수_페이_온라인_B0M + M07_이용건수_페이_온라인_B0M",
    },
    {
        "columns": ["M09_이용건수_페이_오프라인_B0M", "M08_이용건수_페이_오프라인_B0M", "M07_이용건수_페이_오프라인_B0M"],
        "output": "M09_이용건수_페이_오프라인_R3M",
        "fname": "cfs_03_1239",
        "type": "formula",
        "content": "M09_이용건수_페이_오프라인_R3M = M09_이용건수_페이_오프라인_B0M + M08_이용건수_페이_오프라인_B0M + M07_이용건수_페이_오프라인_B0M",
    },
    {
        "columns": ["M09_이용금액_간편결제_B0M", "M08_이용금액_간편결제_B0M", "M07_이용금액_간편결제_B0M"],
        "output": "M09_이용금액_간편결제_R3M",
        "fname": "cfs_03_1265",
        "type": "formula",
        "content": "M09_이용금액_간편결제_R3M = M09_이용금액_간편결제_B0M + M08_이용금액_간편결제_B0M + M07_이용금액_간편결제_B0M",
    },
    {
        "columns": ["M09_이용금액_당사페이_B0M", "M08_이용금액_당사페이_B0M", "M07_이용금액_당사페이_B0M"],
        "output": "M09_이용금액_당사페이_R3M",
        "fname": "cfs_03_1266",
        "type": "formula",
        "content": "M09_이용금액_당사페이_R3M = M09_이용금액_당사페이_B0M + M08_이용금액_당사페이_B0M + M07_이용금액_당사페이_B0M",
    },
    {
        "columns": ["M09_이용금액_당사기타_B0M", "M08_이용금액_당사기타_B0M", "M07_이용금액_당사기타_B0M"],
        "output": "M09_이용금액_당사기타_R3M",
        "fname": "cfs_03_1267",
        "type": "formula",
        "content": "M09_이용금액_당사기타_R3M = M09_이용금액_당사기타_B0M + M08_이용금액_당사기타_B0M + M07_이용금액_당사기타_B0M",
    },
    {
        "columns": ["M09_이용금액_A페이_B0M", "M08_이용금액_A페이_B0M", "M07_이용금액_A페이_B0M"],
        "output": "M09_이용금액_A페이_R3M",
        "fname": "cfs_03_1268",
        "type": "formula",
        "content": "M09_이용금액_A페이_R3M = M09_이용금액_A페이_B0M + M08_이용금액_A페이_B0M + M07_이용금액_A페이_B0M",
    },
    {
        "columns": ["M09_이용금액_B페이_B0M", "M08_이용금액_B페이_B0M", "M07_이용금액_B페이_B0M"],
        "output": "M09_이용금액_B페이_R3M",
        "fname": "cfs_03_1269",
        "type": "formula",
        "content": "M09_이용금액_B페이_R3M = M09_이용금액_B페이_B0M + M08_이용금액_B페이_B0M + M07_이용금액_B페이_B0M",
    },
    {
        "columns": ["M09_이용금액_C페이_B0M", "M08_이용금액_C페이_B0M", "M07_이용금액_C페이_B0M"],
        "output": "M09_이용금액_C페이_R3M",
        "fname": "cfs_03_1270",
        "type": "formula",
        "content": "M09_이용금액_C페이_R3M = M09_이용금액_C페이_B0M + M08_이용금액_C페이_B0M + M07_이용금액_C페이_B0M",
    },
    {
        "columns": ["M09_이용금액_D페이_B0M", "M08_이용금액_D페이_B0M", "M07_이용금액_D페이_B0M"],
        "output": "M09_이용금액_D페이_R3M",
        "fname": "cfs_03_1271",
        "type": "formula",
        "content": "M09_이용금액_D페이_R3M = M09_이용금액_D페이_B0M + M08_이용금액_D페이_B0M + M07_이용금액_D페이_B0M",
    },
    {
        "columns": ["M09_이용건수_간편결제_B0M", "M08_이용건수_간편결제_B0M", "M07_이용건수_간편결제_B0M"],
        "output": "M09_이용건수_간편결제_R3M",
        "fname": "cfs_03_1272",
        "type": "formula",
        "content": "M09_이용건수_간편결제_R3M = M09_이용건수_간편결제_B0M + M08_이용건수_간편결제_B0M + M07_이용건수_간편결제_B0M",
    },
    {
        "columns": ["M09_이용건수_당사페이_B0M", "M08_이용건수_당사페이_B0M", "M07_이용건수_당사페이_B0M"],
        "output": "M09_이용건수_당사페이_R3M",
        "fname": "cfs_03_1273",
        "type": "formula",
        "content": "M09_이용건수_당사페이_R3M = M09_이용건수_당사페이_B0M + M08_이용건수_당사페이_B0M + M07_이용건수_당사페이_B0M",
    },
    {
        "columns": ["M09_이용건수_당사기타_B0M", "M08_이용건수_당사기타_B0M", "M07_이용건수_당사기타_B0M"],
        "output": "M09_이용건수_당사기타_R3M",
        "fname": "cfs_03_1274",
        "type": "formula",
        "content": "M09_이용건수_당사기타_R3M = M09_이용건수_당사기타_B0M + M08_이용건수_당사기타_B0M + M07_이용건수_당사기타_B0M",
    },
    {
        "columns": ["M09_이용건수_A페이_B0M", "M08_이용건수_A페이_B0M", "M07_이용건수_A페이_B0M"],
        "output": "M09_이용건수_A페이_R3M",
        "fname": "cfs_03_1275",
        "type": "formula",
        "content": "M09_이용건수_A페이_R3M = M09_이용건수_A페이_B0M + M08_이용건수_A페이_B0M + M07_이용건수_A페이_B0M",
    },
    {
        "columns": ["M09_이용건수_B페이_B0M", "M08_이용건수_B페이_B0M", "M07_이용건수_B페이_B0M"],
        "output": "M09_이용건수_B페이_R3M",
        "fname": "cfs_03_1276",
        "type": "formula",
        "content": "M09_이용건수_B페이_R3M = M09_이용건수_B페이_B0M + M08_이용건수_B페이_B0M + M07_이용건수_B페이_B0M",
    },
    {
        "columns": ["M09_이용건수_C페이_B0M", "M08_이용건수_C페이_B0M", "M07_이용건수_C페이_B0M"],
        "output": "M09_이용건수_C페이_R3M",
        "fname": "cfs_03_1277",
        "type": "formula",
        "content": "M09_이용건수_C페이_R3M = M09_이용건수_C페이_B0M + M08_이용건수_C페이_B0M + M07_이용건수_C페이_B0M",
    },
    {
        "columns": ["M09_이용건수_D페이_B0M", "M08_이용건수_D페이_B0M", "M07_이용건수_D페이_B0M"],
        "output": "M09_이용건수_D페이_R3M",
        "fname": "cfs_03_1278",
        "type": "formula",
        "content": "M09_이용건수_D페이_R3M = M09_이용건수_D페이_B0M + M08_이용건수_D페이_B0M + M07_이용건수_D페이_B0M",
    },
    {
        "columns": ["M09_이용횟수_선결제_B0M", "M08_이용횟수_선결제_B0M", "M07_이용횟수_선결제_B0M"],
        "output": "M09_이용횟수_선결제_R3M",
        "fname": "cfs_03_1297",
        "type": "formula",
        "content": "M09_이용횟수_선결제_R3M = M09_이용횟수_선결제_B0M + M08_이용횟수_선결제_B0M + M07_이용횟수_선결제_B0M",
    },
    {
        "columns": ["M09_이용금액_선결제_B0M", "M08_이용금액_선결제_B0M", "M07_이용금액_선결제_B0M"],
        "output": "M09_이용금액_선결제_R3M",
        "fname": "cfs_03_1298",
        "type": "formula",
        "content": "M09_이용금액_선결제_R3M = M09_이용금액_선결제_B0M + M08_이용금액_선결제_B0M + M07_이용금액_선결제_B0M",
    },
    {
        "columns": ["M09_이용건수_선결제_B0M", "M08_이용건수_선결제_B0M", "M07_이용건수_선결제_B0M"],
        "output": "M09_이용건수_선결제_R3M",
        "fname": "cfs_03_1299",
        "type": "formula",
        "content": "M09_이용건수_선결제_R3M = M09_이용건수_선결제_B0M + M08_이용건수_선결제_B0M + M07_이용건수_선결제_B0M",
    },
    {
        "columns": ["M08_가맹점매출금액_B1M"],
        "output": "M09_가맹점매출금액_B2M",
        "fname": "cfs_03_1304",
        "type": "formula",
        "content": "M09_가맹점매출금액_B2M = M08_가맹점매출금액_B1M",
    },
    {
        "columns": ["M07_정상청구원금_B0M"],
        "output": "M09_정상청구원금_B2M",
        "fname": "cfs_03_1306",
        "type": "formula",
        "content": "M09_정상청구원금_B2M = M07_정상청구원금_B0M",
    },
    {
        "columns": ["M07_선입금원금_B0M"],
        "output": "M09_선입금원금_B2M",
        "fname": "cfs_03_1309",
        "type": "formula",
        "content": "M09_선입금원금_B2M = M07_선입금원금_B0M",
    },
    {
        "columns": ["M07_정상입금원금_B0M"],
        "output": "M09_정상입금원금_B2M",
        "fname": "cfs_03_1312",
        "type": "formula",
        "content": "M09_정상입금원금_B2M = M07_정상입금원금_B0M",
    },
    {
        "columns": ["M07_연체입금원금_B0M"],
        "output": "M09_연체입금원금_B2M",
        "fname": "cfs_03_1315",
        "type": "formula",
        "content": "M09_연체입금원금_B2M = M07_연체입금원금_B0M",
    },
    {
        "columns": ["M09_이용횟수_연체_B0M", "M08_이용횟수_연체_B0M", "M07_이용횟수_연체_B0M"],
        "output": "M09_이용횟수_연체_R3M",
        "fname": "cfs_03_1323",
        "type": "formula",
        "content": "M09_이용횟수_연체_R3M = M09_이용횟수_연체_B0M + M08_이용횟수_연체_B0M + M07_이용횟수_연체_B0M",
    },
    {
        "columns": ["M09_이용금액_연체_B0M", "M08_이용금액_연체_B0M", "M07_이용금액_연체_B0M"],
        "output": "M09_이용금액_연체_R3M",
        "fname": "cfs_03_1324",
        "type": "formula",
        "content": "M09_이용금액_연체_R3M = M09_이용금액_연체_B0M + M08_이용금액_연체_B0M + M07_이용금액_연체_B0M",
    },
    {
        "columns": ["M08_RP건수_B0M", "M09_RP건수_B0M"],
        "output": "M09_증감_RP건수_전월",
        "fname": "cfs_03_1337",
        "type": "formula",
        "content": "M09_증감_RP건수_전월 = M08_RP건수_B0M - M09_RP건수_B0M",
    },
    {
        "columns": ["M08_RP건수_통신_B0M", "M09_RP건수_통신_B0M"],
        "output": "M09_증감_RP건수_통신_전월",
        "fname": "cfs_03_1339",
        "type": "formula",
        "content": "M09_증감_RP건수_통신_전월 = M08_RP건수_통신_B0M - M09_RP건수_통신_B0M",
    },
    {
        "columns": ["M08_RP건수_아파트_B0M", "M09_RP건수_아파트_B0M"],
        "output": "M09_증감_RP건수_아파트_전월",
        "fname": "cfs_03_1340",
        "type": "formula",
        "content": "M09_증감_RP건수_아파트_전월 = M08_RP건수_아파트_B0M - M09_RP건수_아파트_B0M",
    },
    {
        "columns": ["M08_RP건수_제휴사서비스직접판매_B0M", "M09_RP건수_제휴사서비스직접판매_B0M"],
        "output": "M09_증감_RP건수_제휴사서비스직접판매_전월",
        "fname": "cfs_03_1341",
        "type": "formula",
        "content": "M09_증감_RP건수_제휴사서비스직접판매_전월 = M08_RP건수_제휴사서비스직접판매_B0M - M09_RP건수_제휴사서비스직접판매_B0M",
    },
    {
        "columns": ["M08_RP건수_렌탈_B0M", "M09_RP건수_렌탈_B0M"],
        "output": "M09_증감_RP건수_렌탈_전월",
        "fname": "cfs_03_1342",
        "type": "formula",
        "content": "M09_증감_RP건수_렌탈_전월 = M08_RP건수_렌탈_B0M - M09_RP건수_렌탈_B0M",
    },
    {
        "columns": ["M08_RP건수_가스_B0M", "M09_RP건수_가스_B0M"],
        "output": "M09_증감_RP건수_가스_전월",
        "fname": "cfs_03_1343",
        "type": "formula",
        "content": "M09_증감_RP건수_가스_전월 = M08_RP건수_가스_B0M - M09_RP건수_가스_B0M",
    },
    {
        "columns": ["M08_RP건수_전기_B0M", "M09_RP건수_전기_B0M"],
        "output": "M09_증감_RP건수_전기_전월",
        "fname": "cfs_03_1344",
        "type": "formula",
        "content": "M09_증감_RP건수_전기_전월 = M08_RP건수_전기_B0M - M09_RP건수_전기_B0M",
    },
    {
        "columns": ["M08_RP건수_보험_B0M", "M09_RP건수_보험_B0M"],
        "output": "M09_증감_RP건수_보험_전월",
        "fname": "cfs_03_1345",
        "type": "formula",
        "content": "M09_증감_RP건수_보험_전월 = M08_RP건수_보험_B0M - M09_RP건수_보험_B0M",
    },
    {
        "columns": ["M08_RP건수_학습비_B0M", "M09_RP건수_학습비_B0M"],
        "output": "M09_증감_RP건수_학습비_전월",
        "fname": "cfs_03_1346",
        "type": "formula",
        "content": "M09_증감_RP건수_학습비_전월 = M08_RP건수_학습비_B0M - M09_RP건수_학습비_B0M",
    },
    {
        "columns": ["M08_RP건수_유선방송_B0M", "M09_RP건수_유선방송_B0M"],
        "output": "M09_증감_RP건수_유선방송_전월",
        "fname": "cfs_03_1347",
        "type": "formula",
        "content": "M09_증감_RP건수_유선방송_전월 = M08_RP건수_유선방송_B0M - M09_RP건수_유선방송_B0M",
    },
    {
        "columns": ["M08_RP건수_건강_B0M", "M09_RP건수_건강_B0M"],
        "output": "M09_증감_RP건수_건강_전월",
        "fname": "cfs_03_1348",
        "type": "formula",
        "content": "M09_증감_RP건수_건강_전월 = M08_RP건수_건강_B0M - M09_RP건수_건강_B0M",
    },
    {
        "columns": ["M08_RP건수_교통_B0M", "M09_RP건수_교통_B0M"],
        "output": "M09_증감_RP건수_교통_전월",
        "fname": "cfs_03_1349",
        "type": "formula",
        "content": "M09_증감_RP건수_교통_전월 = M08_RP건수_교통_B0M - M09_RP건수_교통_B0M",
    },
    # M10
    {
        "columns": ["M10_RP건수_B0M", "M09_RP후경과월"],
        "output": "M10_RP후경과월",
        "fname": "cfs_03_1645",
        "type": "formula",
        "content": """IF M10_RP건수_B0M > 0 THEN M10_RP후경과월 = 0
                      ELSE M10_RP후경과월 = min(M09_RP후경과월 + 1, 6)""",
    },
    {
        "columns": ["M10_RP건수_통신_B0M", "M09_RP후경과월_통신"],
        "output": "M10_RP후경과월_통신",
        "fname": "cfs_03_1646",
        "type": "formula",
        "content": """IF M10_RP건수_통신_B0M > 0 THEN M10_RP후경과월_통신 = 0
                      ELSE M10_RP후경과월_통신 = min(M09_RP후경과월_통신 + 1, 6)""",
    },
    {
        "columns": ["M10_RP건수_아파트_B0M", "M09_RP후경과월_아파트"],
        "output": "M10_RP후경과월_아파트",
        "fname": "cfs_03_1647",
        "type": "formula",
        "content": """IF M10_RP건수_아파트_B0M > 0 THEN M10_RP후경과월_아파트 = 0
                      ELSE M10_RP후경과월_아파트 = min(M09_RP후경과월_아파트 + 1, 6)""",
    },
    {
        "columns": ["M10_RP건수_제휴사서비스직접판매_B0M", "M09_RP후경과월_제휴사서비스직접판매"],
        "output": "M10_RP후경과월_제휴사서비스직접판매",
        "fname": "cfs_03_1648",
        "type": "formula",
        "content": """IF M10_RP건수_제휴사서비스직접판매_B0M > 0 THEN M10_RP후경과월_제휴사서비스직접판매 = 0
                      ELSE M10_RP후경과월_제휴사서비스직접판매 = min(M09_RP후경과월_제휴사서비스직접판매 + 1, 6)""",
    },
    {
        "columns": ["M10_RP건수_렌탈_B0M", "M09_RP후경과월_렌탈"],
        "output": "M10_RP후경과월_렌탈",
        "fname": "cfs_03_1649",
        "type": "formula",
        "content": """IF M10_RP건수_렌탈_B0M > 0 THEN M10_RP후경과월_렌탈 = 0
                      ELSE M10_RP후경과월_렌탈 = min(M09_RP후경과월_렌탈 + 1, 6)""",
    },
    {
        "columns": ["M10_RP건수_가스_B0M", "M09_RP후경과월_가스"],
        "output": "M10_RP후경과월_가스",
        "fname": "cfs_03_1650",
        "type": "formula",
        "content": """IF M10_RP건수_가스_B0M > 0 THEN M10_RP후경과월_가스 = 0
                      ELSE M10_RP후경과월_가스 = min(M09_RP후경과월_가스 + 1, 6)""",
    },
    {
        "columns": ["M10_RP건수_전기_B0M", "M09_RP후경과월_전기"],
        "output": "M10_RP후경과월_전기",
        "fname": "cfs_03_1651",
        "type": "formula",
        "content": """IF M10_RP건수_전기_B0M > 0 THEN M10_RP후경과월_전기 = 0
                      ELSE M10_RP후경과월_전기 = min(M09_RP후경과월_전기 + 1, 6)""",
    },
    {
        "columns": ["M10_RP건수_보험_B0M", "M09_RP후경과월_보험"],
        "output": "M10_RP후경과월_보험",
        "fname": "cfs_03_1652",
        "type": "formula",
        "content": """IF M10_RP건수_보험_B0M > 0 THEN M10_RP후경과월_보험 = 0
                      ELSE M10_RP후경과월_보험 = min(M09_RP후경과월_보험 + 1, 6)""",
    },
    {
        "columns": ["M10_RP건수_학습비_B0M", "M09_RP후경과월_학습비"],
        "output": "M10_RP후경과월_학습비",
        "fname": "cfs_03_1653",
        "type": "formula",
        "content": """IF M10_RP건수_학습비_B0M > 0 THEN M10_RP후경과월_학습비 = 0
                      ELSE M10_RP후경과월_학습비 = min(M09_RP후경과월_학습비 + 1, 6)""",
    },
    {
        "columns": ["M10_RP건수_유선방송_B0M", "M09_RP후경과월_유선방송"],
        "output": "M10_RP후경과월_유선방송",
        "fname": "cfs_03_1654",
        "type": "formula",
        "content": """IF M10_RP건수_유선방송_B0M > 0 THEN M10_RP후경과월_유선방송 = 0
                      ELSE M10_RP후경과월_유선방송 = min(M09_RP후경과월_유선방송 + 1, 6)""",
    },
    {
        "columns": ["M10_RP건수_건강_B0M", "M09_RP후경과월_건강"],
        "output": "M10_RP후경과월_건강",
        "fname": "cfs_03_1655",
        "type": "formula",
        "content": """IF M10_RP건수_건강_B0M > 0 THEN M10_RP후경과월_건강 = 0
                      ELSE M10_RP후경과월_건강 = min(M09_RP후경과월_건강 + 1, 6)""",
    },
    {
        "columns": ["M10_RP건수_교통_B0M", "M09_RP후경과월_교통"],
        "output": "M10_RP후경과월_교통",
        "fname": "cfs_03_1656",
        "type": "formula",
        "content": """IF M10_RP건수_교통_B0M > 0 THEN M10_RP후경과월_교통 = 0
                      ELSE M10_RP후경과월_교통 = min(M09_RP후경과월_교통 + 1, 6)""",
    },
    {
        "columns": ["M08_승인거절건수_B0M", "M09_승인거절건수_B0M", "M10_승인거절건수_B0M"],
        "output": "M10_승인거절건수_R3M",
        "fname": "cfs_03_1830",
        "type": "formula",
        "content": "M10_승인거절건수_R3M = M08_승인거절건수_B0M + M09_승인거절건수_B0M + M10_승인거절건수_B0M",
    },
    {
        "columns": ["M08_승인거절건수_한도초과_B0M", "M09_승인거절건수_한도초과_B0M", "M10_승인거절건수_한도초과_B0M"],
        "output": "M10_승인거절건수_한도초과_R3M",
        "fname": "cfs_03_1831",
        "type": "formula",
        "content": "M10_승인거절건수_한도초과_R3M = M08_승인거절건수_한도초과_B0M + M09_승인거절건수_한도초과_B0M + M10_승인거절건수_한도초과_B0M",
    },
    {
        "columns": ["M08_승인거절건수_BL_B0M", "M09_승인거절건수_BL_B0M", "M10_승인거절건수_BL_B0M"],
        "output": "M10_승인거절건수_BL_R3M",
        "fname": "cfs_03_1832",
        "type": "formula",
        "content": "M10_승인거절건수_BL_R3M = M08_승인거절건수_BL_B0M + M09_승인거절건수_BL_B0M + M10_승인거절건수_BL_B0M",
    },
    {
        "columns": ["M08_승인거절건수_입력오류_B0M", "M09_승인거절건수_입력오류_B0M", "M10_승인거절건수_입력오류_B0M"],
        "output": "M10_승인거절건수_입력오류_R3M",
        "fname": "cfs_03_1833",
        "type": "formula",
        "content": "M10_승인거절건수_입력오류_R3M = M08_승인거절건수_입력오류_B0M + M09_승인거절건수_입력오류_B0M + M10_승인거절건수_입력오류_B0M",
    },
    {
        "columns": ["M08_승인거절건수_기타_B0M", "M09_승인거절건수_기타_B0M", "M10_승인거절건수_기타_B0M"],
        "output": "M10_승인거절건수_기타_R3M",
        "fname": "cfs_03_1834",
        "type": "formula",
        "content": "M10_승인거절건수_기타_R3M = M08_승인거절건수_기타_B0M + M09_승인거절건수_기타_B0M + M10_승인거절건수_기타_B0M",
    },
    {
        "columns": ["M10_이용건수_신용_R3M", "M07_이용건수_신용_R3M"],
        "output": "M10_이용건수_신용_R6M",
        "fname": "cfs_03_1463",
        "type": "formula",
        "content": "M10_이용건수_신용_R6M = M10_이용건수_신용_R3M + M07_이용건수_신용_R3M",
    },
    {
        "columns": ["M10_이용건수_신판_R3M", "M07_이용건수_신판_R3M"],
        "output": "M10_이용건수_신판_R6M",
        "fname": "cfs_03_1464",
        "type": "formula",
        "content": "M10_이용건수_신판_R6M = M10_이용건수_신판_R3M + M07_이용건수_신판_R3M",
    },
    {
        "columns": ["M10_이용건수_일시불_R3M", "M07_이용건수_일시불_R3M"],
        "output": "M10_이용건수_일시불_R6M",
        "fname": "cfs_03_1465",
        "type": "formula",
        "content": "M10_이용건수_일시불_R6M = M10_이용건수_일시불_R3M + M07_이용건수_일시불_R3M",
    },
    {
        "columns": ["M10_이용건수_할부_R3M", "M07_이용건수_할부_R3M"],
        "output": "M10_이용건수_할부_R6M",
        "fname": "cfs_03_1466",
        "type": "formula",
        "content": "M10_이용건수_할부_R6M = M10_이용건수_할부_R3M + M07_이용건수_할부_R3M",
    },
    {
        "columns": ["M10_이용건수_할부_유이자_R3M", "M07_이용건수_할부_유이자_R3M"],
        "output": "M10_이용건수_할부_유이자_R6M",
        "fname": "cfs_03_1467",
        "type": "formula",
        "content": "M10_이용건수_할부_유이자_R6M = M10_이용건수_할부_유이자_R3M + M07_이용건수_할부_유이자_R3M",
    },
    {
        "columns": ["M10_이용건수_할부_무이자_R3M", "M07_이용건수_할부_무이자_R3M"],
        "output": "M10_이용건수_할부_무이자_R6M",
        "fname": "cfs_03_1468",
        "type": "formula",
        "content": "M10_이용건수_할부_무이자_R6M = M10_이용건수_할부_무이자_R3M + M07_이용건수_할부_무이자_R3M",
    },
    {
        "columns": ["M10_이용건수_부분무이자_R3M", "M07_이용건수_부분무이자_R3M"],
        "output": "M10_이용건수_부분무이자_R6M",
        "fname": "cfs_03_1469",
        "type": "formula",
        "content": "M10_이용건수_부분무이자_R6M = M10_이용건수_부분무이자_R3M + M07_이용건수_부분무이자_R3M",
    },
    {
        "columns": ["M10_이용건수_CA_R3M", "M07_이용건수_CA_R3M"],
        "output": "M10_이용건수_CA_R6M",
        "fname": "cfs_03_1470",
        "type": "formula",
        "content": "M10_이용건수_CA_R6M = M10_이용건수_CA_R3M + M07_이용건수_CA_R3M",
    },
    {
        "columns": ["M10_이용건수_체크_R3M", "M07_이용건수_체크_R3M"],
        "output": "M10_이용건수_체크_R6M",
        "fname": "cfs_03_1471",
        "type": "formula",
        "content": "M10_이용건수_체크_R6M = M10_이용건수_체크_R3M + M07_이용건수_체크_R3M",
    },
    {
        "columns": ["M10_이용건수_카드론_R3M", "M07_이용건수_카드론_R3M"],
        "output": "M10_이용건수_카드론_R6M",
        "fname": "cfs_03_1472",
        "type": "formula",
        "content": "M10_이용건수_카드론_R6M = M10_이용건수_카드론_R3M + M07_이용건수_카드론_R3M",
    },
    {
        "columns": ["M10_이용금액_신용_R3M", "M07_이용금액_신용_R3M"],
        "output": "M10_이용금액_신용_R6M",
        "fname": "cfs_03_1473",
        "type": "formula",
        "content": "M10_이용금액_신용_R6M = M10_이용금액_신용_R3M + M07_이용금액_신용_R3M",
    },
    {
        "columns": ["M10_이용금액_신판_R3M", "M07_이용금액_신판_R3M"],
        "output": "M10_이용금액_신판_R6M",
        "fname": "cfs_03_1474",
        "type": "formula",
        "content": "M10_이용금액_신판_R6M = M10_이용금액_신판_R3M + M07_이용금액_신판_R3M",
    },
    {
        "columns": ["M10_이용금액_일시불_R3M", "M07_이용금액_일시불_R3M"],
        "output": "M10_이용금액_일시불_R6M",
        "fname": "cfs_03_1475",
        "type": "formula",
        "content": "M10_이용금액_일시불_R6M = M10_이용금액_일시불_R3M + M07_이용금액_일시불_R3M",
    },
    {
        "columns": ["M10_이용금액_할부_R3M", "M07_이용금액_할부_R3M"],
        "output": "M10_이용금액_할부_R6M",
        "fname": "cfs_03_1476",
        "type": "formula",
        "content": "M10_이용금액_할부_R6M = M10_이용금액_할부_R3M + M07_이용금액_할부_R3M",
    },
    {
        "columns": ["M10_이용금액_할부_유이자_R3M", "M07_이용금액_할부_유이자_R3M"],
        "output": "M10_이용금액_할부_유이자_R6M",
        "fname": "cfs_03_1477",
        "type": "formula",
        "content": "M10_이용금액_할부_유이자_R6M = M10_이용금액_할부_유이자_R3M + M07_이용금액_할부_유이자_R3M",
    },
    {
        "columns": ["M10_이용금액_할부_무이자_R3M", "M07_이용금액_할부_무이자_R3M"],
        "output": "M10_이용금액_할부_무이자_R6M",
        "fname": "cfs_03_1478",
        "type": "formula",
        "content": "M10_이용금액_할부_무이자_R6M = M10_이용금액_할부_무이자_R3M + M07_이용금액_할부_무이자_R3M",
    },
    {
        "columns": ["M10_이용금액_부분무이자_R3M", "M07_이용금액_부분무이자_R3M"],
        "output": "M10_이용금액_부분무이자_R6M",
        "fname": "cfs_03_1479",
        "type": "formula",
        "content": "M10_이용금액_부분무이자_R6M = M10_이용금액_부분무이자_R3M + M07_이용금액_부분무이자_R3M",
    },
    {
        "columns": ["M10_이용금액_CA_R3M", "M07_이용금액_CA_R3M"],
        "output": "M10_이용금액_CA_R6M",
        "fname": "cfs_03_1480",
        "type": "formula",
        "content": "M10_이용금액_CA_R6M = M10_이용금액_CA_R3M + M07_이용금액_CA_R3M",
    },
    {
        "columns": ["M10_이용금액_체크_R3M", "M07_이용금액_체크_R3M"],
        "output": "M10_이용금액_체크_R6M",
        "fname": "cfs_03_1481",
        "type": "formula",
        "content": "M10_이용금액_체크_R6M = M10_이용금액_체크_R3M + M07_이용금액_체크_R3M",
    },
    {
        "columns": ["M10_이용금액_카드론_R3M", "M07_이용금액_카드론_R3M"],
        "output": "M10_이용금액_카드론_R6M",
        "fname": "cfs_03_1482",
        "type": "formula",
        "content": "M10_이용금액_카드론_R6M = M10_이용금액_카드론_R3M + M07_이용금액_카드론_R3M",
    },
    {
        "columns": ["M10_이용건수_신용_B0M", "M09_이용건수_신용_B0M", "M08_이용건수_신용_B0M"],
        "output": "M10_이용건수_신용_R3M",
        "fname": "cfs_03_1493",
        "type": "formula",
        "content": "M10_이용건수_신용_R3M = M10_이용건수_신용_B0M + M09_이용건수_신용_B0M + M08_이용건수_신용_B0M",
    },
    {
        "columns": ["M10_이용건수_신판_B0M", "M09_이용건수_신판_B0M", "M08_이용건수_신판_B0M"],
        "output": "M10_이용건수_신판_R3M",
        "fname": "cfs_03_1494",
        "type": "formula",
        "content": "M10_이용건수_신판_R3M = M10_이용건수_신판_B0M + M09_이용건수_신판_B0M + M08_이용건수_신판_B0M",
    },
    {
        "columns": ["M10_이용건수_일시불_B0M", "M09_이용건수_일시불_B0M", "M08_이용건수_일시불_B0M"],
        "output": "M10_이용건수_일시불_R3M",
        "fname": "cfs_03_1495",
        "type": "formula",
        "content": "M10_이용건수_일시불_R3M = M10_이용건수_일시불_B0M + M09_이용건수_일시불_B0M + M08_이용건수_일시불_B0M",
    },
    {
        "columns": ["M10_이용건수_할부_B0M", "M09_이용건수_할부_B0M", "M08_이용건수_할부_B0M"],
        "output": "M10_이용건수_할부_R3M",
        "fname": "cfs_03_1496",
        "type": "formula",
        "content": "M10_이용건수_할부_R3M = M10_이용건수_할부_B0M + M09_이용건수_할부_B0M + M08_이용건수_할부_B0M",
    },
    {
        "columns": ["M10_이용건수_할부_유이자_B0M", "M09_이용건수_할부_유이자_B0M", "M08_이용건수_할부_유이자_B0M"],
        "output": "M10_이용건수_할부_유이자_R3M",
        "fname": "cfs_03_1497",
        "type": "formula",
        "content": "M10_이용건수_할부_유이자_R3M = M10_이용건수_할부_유이자_B0M + M09_이용건수_할부_유이자_B0M + M08_이용건수_할부_유이자_B0M",
    },
    {
        "columns": ["M10_이용건수_할부_무이자_B0M", "M09_이용건수_할부_무이자_B0M", "M08_이용건수_할부_무이자_B0M"],
        "output": "M10_이용건수_할부_무이자_R3M",
        "fname": "cfs_03_1498",
        "type": "formula",
        "content": "M10_이용건수_할부_무이자_R3M = M10_이용건수_할부_무이자_B0M + M09_이용건수_할부_무이자_B0M + M08_이용건수_할부_무이자_B0M",
    },
    {
        "columns": ["M10_이용건수_부분무이자_B0M", "M09_이용건수_부분무이자_B0M", "M08_이용건수_부분무이자_B0M"],
        "output": "M10_이용건수_부분무이자_R3M",
        "fname": "cfs_03_1499",
        "type": "formula",
        "content": "M10_이용건수_부분무이자_R3M = M10_이용건수_부분무이자_B0M + M09_이용건수_부분무이자_B0M + M08_이용건수_부분무이자_B0M",
    },
    {
        "columns": ["M10_이용건수_CA_B0M", "M09_이용건수_CA_B0M", "M08_이용건수_CA_B0M"],
        "output": "M10_이용건수_CA_R3M",
        "fname": "cfs_03_1500",
        "type": "formula",
        "content": "M10_이용건수_CA_R3M = M10_이용건수_CA_B0M + M09_이용건수_CA_B0M + M08_이용건수_CA_B0M",
    },
    {
        "columns": ["M10_이용건수_체크_B0M", "M09_이용건수_체크_B0M", "M08_이용건수_체크_B0M"],
        "output": "M10_이용건수_체크_R3M",
        "fname": "cfs_03_1501",
        "type": "formula",
        "content": "M10_이용건수_체크_R3M = M10_이용건수_체크_B0M + M09_이용건수_체크_B0M + M08_이용건수_체크_B0M",
    },
    {
        "columns": ["M10_이용건수_카드론_B0M", "M09_이용건수_카드론_B0M", "M08_이용건수_카드론_B0M"],
        "output": "M10_이용건수_카드론_R3M",
        "fname": "cfs_03_1502",
        "type": "formula",
        "content": "M10_이용건수_카드론_R3M = M10_이용건수_카드론_B0M + M09_이용건수_카드론_B0M + M08_이용건수_카드론_B0M",
    },
    {
        "columns": ["M10_이용금액_신용_B0M", "M09_이용금액_신용_B0M", "M08_이용금액_신용_B0M"],
        "output": "M10_이용금액_신용_R3M",
        "fname": "cfs_03_1503",
        "type": "formula",
        "content": "M10_이용금액_신용_R3M = M10_이용금액_신용_B0M + M09_이용금액_신용_B0M + M08_이용금액_신용_B0M",
    },
    {
        "columns": ["M10_이용금액_신판_B0M", "M09_이용금액_신판_B0M", "M08_이용금액_신판_B0M"],
        "output": "M10_이용금액_신판_R3M",
        "fname": "cfs_03_1504",
        "type": "formula",
        "content": "M10_이용금액_신판_R3M = M10_이용금액_신판_B0M + M09_이용금액_신판_B0M + M08_이용금액_신판_B0M",
    },
    {
        "columns": ["M10_이용금액_일시불_B0M", "M09_이용금액_일시불_B0M", "M08_이용금액_일시불_B0M"],
        "output": "M10_이용금액_일시불_R3M",
        "fname": "cfs_03_1505",
        "type": "formula",
        "content": "M10_이용금액_일시불_R3M = M10_이용금액_일시불_B0M + M09_이용금액_일시불_B0M + M08_이용금액_일시불_B0M",
    },
    {
        "columns": ["M10_이용금액_할부_B0M", "M09_이용금액_할부_B0M", "M08_이용금액_할부_B0M"],
        "output": "M10_이용금액_할부_R3M",
        "fname": "cfs_03_1506",
        "type": "formula",
        "content": "M10_이용금액_할부_R3M = M10_이용금액_할부_B0M + M09_이용금액_할부_B0M + M08_이용금액_할부_B0M",
    },
    {
        "columns": ["M10_이용금액_할부_유이자_B0M", "M09_이용금액_할부_유이자_B0M", "M08_이용금액_할부_유이자_B0M"],
        "output": "M10_이용금액_할부_유이자_R3M",
        "fname": "cfs_03_1507",
        "type": "formula",
        "content": "M10_이용금액_할부_유이자_R3M = M10_이용금액_할부_유이자_B0M + M09_이용금액_할부_유이자_B0M + M08_이용금액_할부_유이자_B0M",
    },
    {
        "columns": ["M10_이용금액_할부_무이자_B0M", "M09_이용금액_할부_무이자_B0M", "M08_이용금액_할부_무이자_B0M"],
        "output": "M10_이용금액_할부_무이자_R3M",
        "fname": "cfs_03_1508",
        "type": "formula",
        "content": "M10_이용금액_할부_무이자_R3M = M10_이용금액_할부_무이자_B0M + M09_이용금액_할부_무이자_B0M + M08_이용금액_할부_무이자_B0M",
    },
    {
        "columns": ["M10_이용금액_부분무이자_B0M", "M09_이용금액_부분무이자_B0M", "M08_이용금액_부분무이자_B0M"],
        "output": "M10_이용금액_부분무이자_R3M",
        "fname": "cfs_03_1509",
        "type": "formula",
        "content": "M10_이용금액_부분무이자_R3M = M10_이용금액_부분무이자_B0M + M09_이용금액_부분무이자_B0M + M08_이용금액_부분무이자_B0M",
    },
    {
        "columns": ["M10_이용금액_CA_B0M", "M09_이용금액_CA_B0M", "M08_이용금액_CA_B0M"],
        "output": "M10_이용금액_CA_R3M",
        "fname": "cfs_03_1510",
        "type": "formula",
        "content": "M10_이용금액_CA_R3M = M10_이용금액_CA_B0M + M09_이용금액_CA_B0M + M08_이용금액_CA_B0M",
    },
    {
        "columns": ["M10_이용금액_체크_B0M", "M09_이용금액_체크_B0M", "M08_이용금액_체크_B0M"],
        "output": "M10_이용금액_체크_R3M",
        "fname": "cfs_03_1511",
        "type": "formula",
        "content": "M10_이용금액_체크_R3M = M10_이용금액_체크_B0M + M09_이용금액_체크_B0M + M08_이용금액_체크_B0M",
    },
    {
        "columns": ["M10_이용금액_카드론_B0M", "M09_이용금액_카드론_B0M", "M08_이용금액_카드론_B0M"],
        "output": "M10_이용금액_카드론_R3M",
        "fname": "cfs_03_1512",
        "type": "formula",
        "content": "M10_이용금액_카드론_R3M = M10_이용금액_카드론_B0M + M09_이용금액_카드론_B0M + M08_이용금액_카드론_B0M",
    },
    {
        "columns": ["M10_건수_할부전환_R3M", "M07_건수_할부전환_R3M"],
        "output": "M10_건수_할부전환_R6M",
        "fname": "cfs_03_1527",
        "type": "formula",
        "content": "M10_건수_할부전환_R6M = M10_건수_할부전환_R3M + M07_건수_할부전환_R3M",
    },
    {
        "columns": ["M10_금액_할부전환_R3M", "M07_금액_할부전환_R3M"],
        "output": "M10_금액_할부전환_R6M",
        "fname": "cfs_03_1528",
        "type": "formula",
        "content": "M10_금액_할부전환_R6M = M10_금액_할부전환_R3M + M07_금액_할부전환_R3M",
    },
    {
        "columns": ["M10_최종카드론_대출일자", "M09_최종카드론_대출일자", "M09_최종카드론이용경과월"],
        "output": "M10_최종카드론이용경과월",
        "fname": "cfs_03_1658",
        "type": "formula",
        "content": """IF M10_최종카드론_대출일자 IS NULL
                      THEN M10_최종카드론이용경과월 = 999
                      ELIF M10_최종카드론_대출일자 = M09_최종카드론_대출일자
                      THEN M10_최종카드론이용경과월 = M09_최종카드론이용경과월 + 1
                      ELSE M10_최종카드론이용경과월 = 0""",
    },
    {
        "columns": ["M10_이용금액_온라인_R3M", "M07_이용금액_온라인_R3M"],
        "output": "M10_이용금액_온라인_R6M",
        "fname": "cfs_03_1679",
        "type": "formula",
        "content": "M10_이용금액_온라인_R6M = M10_이용금액_온라인_R3M + M07_이용금액_온라인_R3M",
    },
    {
        "columns": ["M10_이용금액_오프라인_R3M", "M07_이용금액_오프라인_R3M"],
        "output": "M10_이용금액_오프라인_R6M",
        "fname": "cfs_03_1680",
        "type": "formula",
        "content": "M10_이용금액_오프라인_R6M = M10_이용금액_오프라인_R3M + M07_이용금액_오프라인_R3M",
    },
    {
        "columns": ["M10_이용건수_온라인_R3M", "M07_이용건수_온라인_R3M"],
        "output": "M10_이용건수_온라인_R6M",
        "fname": "cfs_03_1681",
        "type": "formula",
        "content": "M10_이용건수_온라인_R6M = M10_이용건수_온라인_R3M + M07_이용건수_온라인_R3M",
    },
    {
        "columns": ["M10_이용건수_오프라인_R3M", "M07_이용건수_오프라인_R3M"],
        "output": "M10_이용건수_오프라인_R6M",
        "fname": "cfs_03_1682",
        "type": "formula",
        "content": "M10_이용건수_오프라인_R6M = M10_이용건수_오프라인_R3M + M07_이용건수_오프라인_R3M",
    },
    {
        "columns": ["M10_이용금액_온라인_B0M", "M09_이용금액_온라인_B0M", "M08_이용금액_온라인_B0M"],
        "output": "M10_이용금액_온라인_R3M",
        "fname": "cfs_03_1683",
        "type": "formula",
        "content": "M10_이용금액_온라인_R3M = M10_이용금액_온라인_B0M + M09_이용금액_온라인_B0M + M08_이용금액_온라인_B0M",
    },
    {
        "columns": ["M10_이용금액_오프라인_B0M", "M09_이용금액_오프라인_B0M", "M08_이용금액_오프라인_B0M"],
        "output": "M10_이용금액_오프라인_R3M",
        "fname": "cfs_03_1684",
        "type": "formula",
        "content": "M10_이용금액_오프라인_R3M = M10_이용금액_오프라인_B0M + M09_이용금액_오프라인_B0M + M08_이용금액_오프라인_B0M",
    },
    {
        "columns": ["M10_이용건수_온라인_B0M", "M09_이용건수_온라인_B0M", "M08_이용건수_온라인_B0M"],
        "output": "M10_이용건수_온라인_R3M",
        "fname": "cfs_03_1685",
        "type": "formula",
        "content": "M10_이용건수_온라인_R3M = M10_이용건수_온라인_B0M + M09_이용건수_온라인_B0M + M08_이용건수_온라인_B0M",
    },
    {
        "columns": ["M10_이용건수_오프라인_B0M", "M09_이용건수_오프라인_B0M", "M08_이용건수_오프라인_B0M"],
        "output": "M10_이용건수_오프라인_R3M",
        "fname": "cfs_03_1686",
        "type": "formula",
        "content": "M10_이용건수_오프라인_R3M = M10_이용건수_오프라인_B0M + M09_이용건수_오프라인_B0M + M08_이용건수_오프라인_B0M",
    },
    {
        "columns": ["M10_이용금액_페이_온라인_R3M", "M07_이용금액_페이_온라인_R3M"],
        "output": "M10_이용금액_페이_온라인_R6M",
        "fname": "cfs_03_1693",
        "type": "formula",
        "content": "M10_이용금액_페이_온라인_R6M = M10_이용금액_페이_온라인_R3M + M07_이용금액_페이_온라인_R3M",
    },
    {
        "columns": ["M10_이용금액_페이_오프라인_R3M", "M07_이용금액_페이_오프라인_R3M"],
        "output": "M10_이용금액_페이_오프라인_R6M",
        "fname": "cfs_03_1694",
        "type": "formula",
        "content": "M10_이용금액_페이_오프라인_R6M = M10_이용금액_페이_오프라인_R3M + M07_이용금액_페이_오프라인_R3M",
    },
    {
        "columns": ["M10_이용건수_페이_온라인_R3M", "M07_이용건수_페이_온라인_R3M"],
        "output": "M10_이용건수_페이_온라인_R6M",
        "fname": "cfs_03_1695",
        "type": "formula",
        "content": "M10_이용건수_페이_온라인_R6M = M10_이용건수_페이_온라인_R3M + M07_이용건수_페이_온라인_R3M",
    },
    {
        "columns": ["M10_이용건수_페이_오프라인_R3M", "M07_이용건수_페이_오프라인_R3M"],
        "output": "M10_이용건수_페이_오프라인_R6M",
        "fname": "cfs_03_1696",
        "type": "formula",
        "content": "M10_이용건수_페이_오프라인_R6M = M10_이용건수_페이_오프라인_R3M + M07_이용건수_페이_오프라인_R3M",
    },
    {
        "columns": ["M10_이용금액_페이_온라인_B0M", "M09_이용금액_페이_온라인_B0M", "M08_이용금액_페이_온라인_B0M"],
        "output": "M10_이용금액_페이_온라인_R3M",
        "fname": "cfs_03_1697",
        "type": "formula",
        "content": "M10_이용금액_페이_온라인_R3M = M10_이용금액_페이_온라인_B0M + M09_이용금액_페이_온라인_B0M + M08_이용금액_페이_온라인_B0M",
    },
    {
        "columns": ["M10_이용금액_페이_오프라인_B0M", "M09_이용금액_페이_오프라인_B0M", "M08_이용금액_페이_오프라인_B0M"],
        "output": "M10_이용금액_페이_오프라인_R3M",
        "fname": "cfs_03_1698",
        "type": "formula",
        "content": "M10_이용금액_페이_오프라인_R3M = M10_이용금액_페이_오프라인_B0M + M09_이용금액_페이_오프라인_B0M + M08_이용금액_페이_오프라인_B0M",
    },
    {
        "columns": ["M10_이용건수_페이_온라인_B0M", "M09_이용건수_페이_온라인_B0M", "M08_이용건수_페이_온라인_B0M"],
        "output": "M10_이용건수_페이_온라인_R3M",
        "fname": "cfs_03_1699",
        "type": "formula",
        "content": "M10_이용건수_페이_온라인_R3M = M10_이용건수_페이_온라인_B0M + M09_이용건수_페이_온라인_B0M + M08_이용건수_페이_온라인_B0M",
    },
    {
        "columns": ["M10_이용건수_페이_오프라인_B0M", "M09_이용건수_페이_오프라인_B0M", "M08_이용건수_페이_오프라인_B0M"],
        "output": "M10_이용건수_페이_오프라인_R3M",
        "fname": "cfs_03_1700",
        "type": "formula",
        "content": "M10_이용건수_페이_오프라인_R3M = M10_이용건수_페이_오프라인_B0M + M09_이용건수_페이_오프라인_B0M + M08_이용건수_페이_오프라인_B0M",
    },
    {
        "columns": ["M10_이용금액_간편결제_R3M", "M07_이용금액_간편결제_R3M"],
        "output": "M10_이용금액_간편결제_R6M",
        "fname": "cfs_03_1712",
        "type": "formula",
        "content": "M10_이용금액_간편결제_R6M = M10_이용금액_간편결제_R3M + M07_이용금액_간편결제_R3M",
    },
    {
        "columns": ["M10_이용금액_당사페이_R3M", "M07_이용금액_당사페이_R3M"],
        "output": "M10_이용금액_당사페이_R6M",
        "fname": "cfs_03_1713",
        "type": "formula",
        "content": "M10_이용금액_당사페이_R6M = M10_이용금액_당사페이_R3M + M07_이용금액_당사페이_R3M",
    },
    {
        "columns": ["M10_이용금액_당사기타_R3M", "M07_이용금액_당사기타_R3M"],
        "output": "M10_이용금액_당사기타_R6M",
        "fname": "cfs_03_1714",
        "type": "formula",
        "content": "M10_이용금액_당사기타_R6M = M10_이용금액_당사기타_R3M + M07_이용금액_당사기타_R3M",
    },
    {
        "columns": ["M10_이용금액_A페이_R3M", "M07_이용금액_A페이_R3M"],
        "output": "M10_이용금액_A페이_R6M",
        "fname": "cfs_03_1715",
        "type": "formula",
        "content": "M10_이용금액_A페이_R6M = M10_이용금액_A페이_R3M + M07_이용금액_A페이_R3M",
    },
    {
        "columns": ["M10_이용금액_B페이_R3M", "M07_이용금액_B페이_R3M"],
        "output": "M10_이용금액_B페이_R6M",
        "fname": "cfs_03_1716",
        "type": "formula",
        "content": "M10_이용금액_B페이_R6M = M10_이용금액_B페이_R3M + M07_이용금액_B페이_R3M",
    },
    {
        "columns": ["M10_이용금액_C페이_R3M", "M07_이용금액_C페이_R3M"],
        "output": "M10_이용금액_C페이_R6M",
        "fname": "cfs_03_1717",
        "type": "formula",
        "content": "M10_이용금액_C페이_R6M = M10_이용금액_C페이_R3M + M07_이용금액_C페이_R3M",
    },
    {
        "columns": ["M10_이용금액_D페이_R3M", "M07_이용금액_D페이_R3M"],
        "output": "M10_이용금액_D페이_R6M",
        "fname": "cfs_03_1718",
        "type": "formula",
        "content": "M10_이용금액_D페이_R6M = M10_이용금액_D페이_R3M + M07_이용금액_D페이_R3M",
    },
    {
        "columns": ["M10_이용건수_간편결제_R3M", "M07_이용건수_간편결제_R3M"],
        "output": "M10_이용건수_간편결제_R6M",
        "fname": "cfs_03_1719",
        "type": "formula",
        "content": "M10_이용건수_간편결제_R6M = M10_이용건수_간편결제_R3M + M07_이용건수_간편결제_R3M",
    },
    {
        "columns": ["M10_이용건수_당사페이_R3M", "M07_이용건수_당사페이_R3M"],
        "output": "M10_이용건수_당사페이_R6M",
        "fname": "cfs_03_1720",
        "type": "formula",
        "content": "M10_이용건수_당사페이_R6M = M10_이용건수_당사페이_R3M + M07_이용건수_당사페이_R3M",
    },
    {
        "columns": ["M10_이용건수_당사기타_R3M", "M07_이용건수_당사기타_R3M"],
        "output": "M10_이용건수_당사기타_R6M",
        "fname": "cfs_03_1721",
        "type": "formula",
        "content": "M10_이용건수_당사기타_R6M = M10_이용건수_당사기타_R3M + M07_이용건수_당사기타_R3M",
    },
    {
        "columns": ["M10_이용건수_A페이_R3M", "M07_이용건수_A페이_R3M"],
        "output": "M10_이용건수_A페이_R6M",
        "fname": "cfs_03_1722",
        "type": "formula",
        "content": "M10_이용건수_A페이_R6M = M10_이용건수_A페이_R3M + M07_이용건수_A페이_R3M",
    },
    {
        "columns": ["M10_이용건수_B페이_R3M", "M07_이용건수_B페이_R3M"],
        "output": "M10_이용건수_B페이_R6M",
        "fname": "cfs_03_1723",
        "type": "formula",
        "content": "M10_이용건수_B페이_R6M = M10_이용건수_B페이_R3M + M07_이용건수_B페이_R3M",
    },
    {
        "columns": ["M10_이용건수_C페이_R3M", "M07_이용건수_C페이_R3M"],
        "output": "M10_이용건수_C페이_R6M",
        "fname": "cfs_03_1724",
        "type": "formula",
        "content": "M10_이용건수_C페이_R6M = M10_이용건수_C페이_R3M + M07_이용건수_C페이_R3M",
    },
    {
        "columns": ["M10_이용건수_D페이_R3M", "M07_이용건수_D페이_R3M"],
        "output": "M10_이용건수_D페이_R6M",
        "fname": "cfs_03_1725",
        "type": "formula",
        "content": "M10_이용건수_D페이_R6M = M10_이용건수_D페이_R3M + M07_이용건수_D페이_R3M",
    },
    {
        "columns": ["M10_이용금액_간편결제_B0M", "M09_이용금액_간편결제_B0M", "M08_이용금액_간편결제_B0M"],
        "output": "M10_이용금액_간편결제_R3M",
        "fname": "cfs_03_1726",
        "type": "formula",
        "content": "M10_이용금액_간편결제_R3M = M10_이용금액_간편결제_B0M + M09_이용금액_간편결제_B0M + M08_이용금액_간편결제_B0M",
    },
    {
        "columns": ["M10_이용금액_당사페이_B0M", "M09_이용금액_당사페이_B0M", "M08_이용금액_당사페이_B0M"],
        "output": "M10_이용금액_당사페이_R3M",
        "fname": "cfs_03_1727",
        "type": "formula",
        "content": "M10_이용금액_당사페이_R3M = M10_이용금액_당사페이_B0M + M09_이용금액_당사페이_B0M + M08_이용금액_당사페이_B0M",
    },
    {
        "columns": ["M10_이용금액_당사기타_B0M", "M09_이용금액_당사기타_B0M", "M08_이용금액_당사기타_B0M"],
        "output": "M10_이용금액_당사기타_R3M",
        "fname": "cfs_03_1728",
        "type": "formula",
        "content": "M10_이용금액_당사기타_R3M = M10_이용금액_당사기타_B0M + M09_이용금액_당사기타_B0M + M08_이용금액_당사기타_B0M",
    },
    {
        "columns": ["M10_이용금액_A페이_B0M", "M09_이용금액_A페이_B0M", "M08_이용금액_A페이_B0M"],
        "output": "M10_이용금액_A페이_R3M",
        "fname": "cfs_03_1729",
        "type": "formula",
        "content": "M10_이용금액_A페이_R3M = M10_이용금액_A페이_B0M + M09_이용금액_A페이_B0M + M08_이용금액_A페이_B0M",
    },
    {
        "columns": ["M10_이용금액_B페이_B0M", "M09_이용금액_B페이_B0M", "M08_이용금액_B페이_B0M"],
        "output": "M10_이용금액_B페이_R3M",
        "fname": "cfs_03_1730",
        "type": "formula",
        "content": "M10_이용금액_B페이_R3M = M10_이용금액_B페이_B0M + M09_이용금액_B페이_B0M + M08_이용금액_B페이_B0M",
    },
    {
        "columns": ["M10_이용금액_C페이_B0M", "M09_이용금액_C페이_B0M", "M08_이용금액_C페이_B0M"],
        "output": "M10_이용금액_C페이_R3M",
        "fname": "cfs_03_1731",
        "type": "formula",
        "content": "M10_이용금액_C페이_R3M = M10_이용금액_C페이_B0M + M09_이용금액_C페이_B0M + M08_이용금액_C페이_B0M",
    },
    {
        "columns": ["M10_이용금액_D페이_B0M", "M09_이용금액_D페이_B0M", "M08_이용금액_D페이_B0M"],
        "output": "M10_이용금액_D페이_R3M",
        "fname": "cfs_03_1732",
        "type": "formula",
        "content": "M10_이용금액_D페이_R3M = M10_이용금액_D페이_B0M + M09_이용금액_D페이_B0M + M08_이용금액_D페이_B0M",
    },
    {
        "columns": ["M10_이용건수_간편결제_B0M", "M09_이용건수_간편결제_B0M", "M08_이용건수_간편결제_B0M"],
        "output": "M10_이용건수_간편결제_R3M",
        "fname": "cfs_03_1733",
        "type": "formula",
        "content": "M10_이용건수_간편결제_R3M = M10_이용건수_간편결제_B0M + M09_이용건수_간편결제_B0M + M08_이용건수_간편결제_B0M",
    },
    {
        "columns": ["M10_이용건수_당사페이_B0M", "M09_이용건수_당사페이_B0M", "M08_이용건수_당사페이_B0M"],
        "output": "M10_이용건수_당사페이_R3M",
        "fname": "cfs_03_1734",
        "type": "formula",
        "content": "M10_이용건수_당사페이_R3M = M10_이용건수_당사페이_B0M + M09_이용건수_당사페이_B0M + M08_이용건수_당사페이_B0M",
    },
    {
        "columns": ["M10_이용건수_당사기타_B0M", "M09_이용건수_당사기타_B0M", "M08_이용건수_당사기타_B0M"],
        "output": "M10_이용건수_당사기타_R3M",
        "fname": "cfs_03_1735",
        "type": "formula",
        "content": "M10_이용건수_당사기타_R3M = M10_이용건수_당사기타_B0M + M09_이용건수_당사기타_B0M + M08_이용건수_당사기타_B0M",
    },
    {
        "columns": ["M10_이용건수_A페이_B0M", "M09_이용건수_A페이_B0M", "M08_이용건수_A페이_B0M"],
        "output": "M10_이용건수_A페이_R3M",
        "fname": "cfs_03_1736",
        "type": "formula",
        "content": "M10_이용건수_A페이_R3M = M10_이용건수_A페이_B0M + M09_이용건수_A페이_B0M + M08_이용건수_A페이_B0M",
    },
    {
        "columns": ["M10_이용건수_B페이_B0M", "M09_이용건수_B페이_B0M", "M08_이용건수_B페이_B0M"],
        "output": "M10_이용건수_B페이_R3M",
        "fname": "cfs_03_1737",
        "type": "formula",
        "content": "M10_이용건수_B페이_R3M = M10_이용건수_B페이_B0M + M09_이용건수_B페이_B0M + M08_이용건수_B페이_B0M",
    },
    {
        "columns": ["M10_이용건수_C페이_B0M", "M09_이용건수_C페이_B0M", "M08_이용건수_C페이_B0M"],
        "output": "M10_이용건수_C페이_R3M",
        "fname": "cfs_03_1738",
        "type": "formula",
        "content": "M10_이용건수_C페이_R3M = M10_이용건수_C페이_B0M + M09_이용건수_C페이_B0M + M08_이용건수_C페이_B0M",
    },
    {
        "columns": ["M10_이용건수_D페이_B0M", "M09_이용건수_D페이_B0M", "M08_이용건수_D페이_B0M"],
        "output": "M10_이용건수_D페이_R3M",
        "fname": "cfs_03_1739",
        "type": "formula",
        "content": "M10_이용건수_D페이_R3M = M10_이용건수_D페이_B0M + M09_이용건수_D페이_B0M + M08_이용건수_D페이_B0M",
    },
    {
        "columns": ["M10_이용횟수_선결제_R3M", "M07_이용횟수_선결제_R3M"],
        "output": "M10_이용횟수_선결제_R6M",
        "fname": "cfs_03_1755",
        "type": "formula",
        "content": "M10_이용횟수_선결제_R6M = M10_이용횟수_선결제_R3M + M07_이용횟수_선결제_R3M",
    },
    {
        "columns": ["M10_이용금액_선결제_R3M", "M07_이용금액_선결제_R3M"],
        "output": "M10_이용금액_선결제_R6M",
        "fname": "cfs_03_1756",
        "type": "formula",
        "content": "M10_이용금액_선결제_R6M = M10_이용금액_선결제_R3M + M07_이용금액_선결제_R3M",
    },
    {
        "columns": ["M10_이용건수_선결제_R3M", "M07_이용건수_선결제_R3M"],
        "output": "M10_이용건수_선결제_R6M",
        "fname": "cfs_03_1757",
        "type": "formula",
        "content": "M10_이용건수_선결제_R6M = M10_이용건수_선결제_R3M + M07_이용건수_선결제_R3M",
    },
    {
        "columns": ["M10_이용횟수_선결제_B0M", "M09_이용횟수_선결제_B0M", "M08_이용횟수_선결제_B0M"],
        "output": "M10_이용횟수_선결제_R3M",
        "fname": "cfs_03_1758",
        "type": "formula",
        "content": "M10_이용횟수_선결제_R3M = M10_이용횟수_선결제_B0M + M09_이용횟수_선결제_B0M + M08_이용횟수_선결제_B0M",
    },
    {
        "columns": ["M10_이용금액_선결제_B0M", "M09_이용금액_선결제_B0M", "M08_이용금액_선결제_B0M"],
        "output": "M10_이용금액_선결제_R3M",
        "fname": "cfs_03_1759",
        "type": "formula",
        "content": "M10_이용금액_선결제_R3M = M10_이용금액_선결제_B0M + M09_이용금액_선결제_B0M + M08_이용금액_선결제_B0M",
    },
    {
        "columns": ["M10_이용건수_선결제_B0M", "M09_이용건수_선결제_B0M", "M08_이용건수_선결제_B0M"],
        "output": "M10_이용건수_선결제_R3M",
        "fname": "cfs_03_1760",
        "type": "formula",
        "content": "M10_이용건수_선결제_R3M = M10_이용건수_선결제_B0M + M09_이용건수_선결제_B0M + M08_이용건수_선결제_B0M",
    },
    {
        "columns": ["M09_가맹점매출금액_B1M"],
        "output": "M10_가맹점매출금액_B2M",
        "fname": "cfs_03_1765",
        "type": "formula",
        "content": "M10_가맹점매출금액_B2M = M09_가맹점매출금액_B1M",
    },
    {
        "columns": ["M08_정상청구원금_B0M"],
        "output": "M10_정상청구원금_B2M",
        "fname": "cfs_03_1767",
        "type": "formula",
        "content": "M10_정상청구원금_B2M = M08_정상청구원금_B0M",
    },
    {
        "columns": ["M08_선입금원금_B0M"],
        "output": "M10_선입금원금_B2M",
        "fname": "cfs_03_1770",
        "type": "formula",
        "content": "M10_선입금원금_B2M = M08_선입금원금_B0M",
    },
    {
        "columns": ["M08_정상입금원금_B0M"],
        "output": "M10_정상입금원금_B2M",
        "fname": "cfs_03_1773",
        "type": "formula",
        "content": "M10_정상입금원금_B2M = M08_정상입금원금_B0M",
    },
    {
        "columns": ["M08_연체입금원금_B0M"],
        "output": "M10_연체입금원금_B2M",
        "fname": "cfs_03_1776",
        "type": "formula",
        "content": "M10_연체입금원금_B2M = M08_연체입금원금_B0M",
    },
    {
        "columns": ["M10_이용횟수_연체_R3M", "M07_이용횟수_연체_R3M"],
        "output": "M10_이용횟수_연체_R6M",
        "fname": "cfs_03_1782",
        "type": "formula",
        "content": "M10_이용횟수_연체_R6M = M10_이용횟수_연체_R3M + M07_이용횟수_연체_R3M",
    },
    {
        "columns": ["M10_이용금액_연체_R3M", "M07_이용금액_연체_R3M"],
        "output": "M10_이용금액_연체_R6M",
        "fname": "cfs_03_1783",
        "type": "formula",
        "content": "M10_이용금액_연체_R6M = M10_이용금액_연체_R3M + M07_이용금액_연체_R3M",
    },
    {
        "columns": ["M10_이용횟수_연체_B0M", "M09_이용횟수_연체_B0M", "M08_이용횟수_연체_B0M"],
        "output": "M10_이용횟수_연체_R3M",
        "fname": "cfs_03_1784",
        "type": "formula",
        "content": "M10_이용횟수_연체_R3M = M10_이용횟수_연체_B0M + M09_이용횟수_연체_B0M + M08_이용횟수_연체_B0M",
    },
    {
        "columns": ["M10_이용금액_연체_B0M", "M09_이용금액_연체_B0M", "M08_이용금액_연체_B0M"],
        "output": "M10_이용금액_연체_R3M",
        "fname": "cfs_03_1785",
        "type": "formula",
        "content": "M10_이용금액_연체_R3M = M10_이용금액_연체_B0M + M09_이용금액_연체_B0M + M08_이용금액_연체_B0M",
    },
    {
        "columns": ["M09_RP건수_B0M", "M10_RP건수_B0M"],
        "output": "M10_증감_RP건수_전월",
        "fname": "cfs_03_1798",
        "type": "formula",
        "content": "M10_증감_RP건수_전월 = M09_RP건수_B0M - M10_RP건수_B0M",
    },
    {
        "columns": ["M09_RP건수_통신_B0M", "M10_RP건수_통신_B0M"],
        "output": "M10_증감_RP건수_통신_전월",
        "fname": "cfs_03_1800",
        "type": "formula",
        "content": "M10_증감_RP건수_통신_전월 = M09_RP건수_통신_B0M - M10_RP건수_통신_B0M",
    },
    {
        "columns": ["M09_RP건수_아파트_B0M", "M10_RP건수_아파트_B0M"],
        "output": "M10_증감_RP건수_아파트_전월",
        "fname": "cfs_03_1801",
        "type": "formula",
        "content": "M10_증감_RP건수_아파트_전월 = M09_RP건수_아파트_B0M - M10_RP건수_아파트_B0M",
    },
    {
        "columns": ["M09_RP건수_제휴사서비스직접판매_B0M", "M10_RP건수_제휴사서비스직접판매_B0M"],
        "output": "M10_증감_RP건수_제휴사서비스직접판매_전월",
        "fname": "cfs_03_1802",
        "type": "formula",
        "content": "M10_증감_RP건수_제휴사서비스직접판매_전월 = M09_RP건수_제휴사서비스직접판매_B0M - M10_RP건수_제휴사서비스직접판매_B0M",
    },
    {
        "columns": ["M09_RP건수_렌탈_B0M", "M10_RP건수_렌탈_B0M"],
        "output": "M10_증감_RP건수_렌탈_전월",
        "fname": "cfs_03_1803",
        "type": "formula",
        "content": "M10_증감_RP건수_렌탈_전월 = M09_RP건수_렌탈_B0M - M10_RP건수_렌탈_B0M",
    },
    {
        "columns": ["M09_RP건수_가스_B0M", "M10_RP건수_가스_B0M"],
        "output": "M10_증감_RP건수_가스_전월",
        "fname": "cfs_03_1804",
        "type": "formula",
        "content": "M10_증감_RP건수_가스_전월 = M09_RP건수_가스_B0M - M10_RP건수_가스_B0M",
    },
    {
        "columns": ["M09_RP건수_전기_B0M", "M10_RP건수_전기_B0M"],
        "output": "M10_증감_RP건수_전기_전월",
        "fname": "cfs_03_1805",
        "type": "formula",
        "content": "M10_증감_RP건수_전기_전월 = M09_RP건수_전기_B0M - M10_RP건수_전기_B0M",
    },
    {
        "columns": ["M09_RP건수_보험_B0M", "M10_RP건수_보험_B0M"],
        "output": "M10_증감_RP건수_보험_전월",
        "fname": "cfs_03_1806",
        "type": "formula",
        "content": "M10_증감_RP건수_보험_전월 = M09_RP건수_보험_B0M - M10_RP건수_보험_B0M",
    },
    {
        "columns": ["M09_RP건수_학습비_B0M", "M10_RP건수_학습비_B0M"],
        "output": "M10_증감_RP건수_학습비_전월",
        "fname": "cfs_03_1807",
        "type": "formula",
        "content": "M10_증감_RP건수_학습비_전월 = M09_RP건수_학습비_B0M - M10_RP건수_학습비_B0M",
    },
    {
        "columns": ["M09_RP건수_유선방송_B0M", "M10_RP건수_유선방송_B0M"],
        "output": "M10_증감_RP건수_유선방송_전월",
        "fname": "cfs_03_1808",
        "type": "formula",
        "content": "M10_증감_RP건수_유선방송_전월 = M09_RP건수_유선방송_B0M - M10_RP건수_유선방송_B0M",
    },
    {
        "columns": ["M09_RP건수_건강_B0M", "M10_RP건수_건강_B0M"],
        "output": "M10_증감_RP건수_건강_전월",
        "fname": "cfs_03_1809",
        "type": "formula",
        "content": "M10_증감_RP건수_건강_전월 = M09_RP건수_건강_B0M - M10_RP건수_건강_B0M",
    },
    {
        "columns": ["M09_RP건수_교통_B0M", "M10_RP건수_교통_B0M"],
        "output": "M10_증감_RP건수_교통_전월",
        "fname": "cfs_03_1810",
        "type": "formula",
        "content": "M10_증감_RP건수_교통_전월 = M09_RP건수_교통_B0M - M10_RP건수_교통_B0M",
    },
    # M11
    {
        "columns": ["M11_RP건수_B0M", "M10_RP후경과월"],
        "output": "M11_RP후경과월",
        "fname": "cfs_03_2106",
        "type": "formula",
        "content": """IF M11_RP건수_B0M > 0 THEN M11_RP후경과월 = 0
                      ELSE M11_RP후경과월 = min(M10_RP후경과월 + 1, 6)""",
    },
    {
        "columns": ["M11_RP건수_통신_B0M", "M10_RP후경과월_통신"],
        "output": "M11_RP후경과월_통신",
        "fname": "cfs_03_2107",
        "type": "formula",
        "content": """IF M11_RP건수_통신_B0M > 0 THEN M11_RP후경과월_통신 = 0
                      ELSE M11_RP후경과월_통신 = min(M10_RP후경과월_통신 + 1, 6)""",
    },
    {
        "columns": ["M11_RP건수_아파트_B0M", "M10_RP후경과월_아파트"],
        "output": "M11_RP후경과월_아파트",
        "fname": "cfs_03_2108",
        "type": "formula",
        "content": """IF M11_RP건수_아파트_B0M > 0 THEN M11_RP후경과월_아파트 = 0
                      ELSE M11_RP후경과월_아파트 = min(M10_RP후경과월_아파트 + 1, 6)""",
    },
    {
        "columns": ["M11_RP건수_제휴사서비스직접판매_B0M", "M10_RP후경과월_제휴사서비스직접판매"],
        "output": "M11_RP후경과월_제휴사서비스직접판매",
        "fname": "cfs_03_2109",
        "type": "formula",
        "content": """IF M11_RP건수_제휴사서비스직접판매_B0M > 0 THEN M11_RP후경과월_제휴사서비스직접판매 = 0
                      ELSE M11_RP후경과월_제휴사서비스직접판매 = min(M10_RP후경과월_제휴사서비스직접판매 + 1, 6)""",
    },
    {
        "columns": ["M11_RP건수_렌탈_B0M", "M10_RP후경과월_렌탈"],
        "output": "M11_RP후경과월_렌탈",
        "fname": "cfs_03_2110",
        "type": "formula",
        "content": """IF M11_RP건수_렌탈_B0M > 0 THEN M11_RP후경과월_렌탈 = 0
                      ELSE M11_RP후경과월_렌탈 = min(M10_RP후경과월_렌탈 + 1, 6)""",
    },
    {
        "columns": ["M11_RP건수_가스_B0M", "M10_RP후경과월_가스"],
        "output": "M11_RP후경과월_가스",
        "fname": "cfs_03_2111",
        "type": "formula",
        "content": """IF M11_RP건수_가스_B0M > 0 THEN M11_RP후경과월_가스 = 0
                      ELSE M11_RP후경과월_가스 = min(M10_RP후경과월_가스 + 1, 6)""",
    },
    {
        "columns": ["M11_RP건수_전기_B0M", "M10_RP후경과월_전기"],
        "output": "M11_RP후경과월_전기",
        "fname": "cfs_03_2112",
        "type": "formula",
        "content": """IF M11_RP건수_전기_B0M > 0 THEN M11_RP후경과월_전기 = 0
                      ELSE M11_RP후경과월_전기 = min(M10_RP후경과월_전기 + 1, 6)""",
    },
    {
        "columns": ["M11_RP건수_보험_B0M", "M10_RP후경과월_보험"],
        "output": "M11_RP후경과월_보험",
        "fname": "cfs_03_2113",
        "type": "formula",
        "content": """IF M11_RP건수_보험_B0M > 0 THEN M11_RP후경과월_보험 = 0
                      ELSE M11_RP후경과월_보험 = min(M10_RP후경과월_보험 + 1, 6)""",
    },
    {
        "columns": ["M11_RP건수_학습비_B0M", "M10_RP후경과월_학습비"],
        "output": "M11_RP후경과월_학습비",
        "fname": "cfs_03_2114",
        "type": "formula",
        "content": """IF M11_RP건수_학습비_B0M > 0 THEN M11_RP후경과월_학습비 = 0
                      ELSE M11_RP후경과월_학습비 = min(M10_RP후경과월_학습비 + 1, 6)""",
    },
    {
        "columns": ["M11_RP건수_유선방송_B0M", "M10_RP후경과월_유선방송"],
        "output": "M11_RP후경과월_유선방송",
        "fname": "cfs_03_2115",
        "type": "formula",
        "content": """IF M11_RP건수_유선방송_B0M > 0 THEN M11_RP후경과월_유선방송 = 0
                      ELSE M11_RP후경과월_유선방송 = min(M10_RP후경과월_유선방송 + 1, 6)""",
    },
    {
        "columns": ["M11_RP건수_건강_B0M", "M10_RP후경과월_건강"],
        "output": "M11_RP후경과월_건강",
        "fname": "cfs_03_2116",
        "type": "formula",
        "content": """IF M11_RP건수_건강_B0M > 0 THEN M11_RP후경과월_건강 = 0
                      ELSE M11_RP후경과월_건강 = min(M10_RP후경과월_건강 + 1, 6)""",
    },
    {
        "columns": ["M11_RP건수_교통_B0M", "M10_RP후경과월_교통"],
        "output": "M11_RP후경과월_교통",
        "fname": "cfs_03_2117",
        "type": "formula",
        "content": """IF M11_RP건수_교통_B0M > 0 THEN M11_RP후경과월_교통 = 0
                      ELSE M11_RP후경과월_교통 = min(M10_RP후경과월_교통 + 1, 6)""",
    },
    {
        "columns": ["M09_승인거절건수_B0M", "M10_승인거절건수_B0M", "M11_승인거절건수_B0M"],
        "output": "M11_승인거절건수_R3M",
        "fname": "cfs_03_2291",
        "type": "formula",
        "content": "M11_승인거절건수_R3M = M09_승인거절건수_B0M + M10_승인거절건수_B0M + M11_승인거절건수_B0M",
    },
    {
        "columns": ["M09_승인거절건수_한도초과_B0M", "M10_승인거절건수_한도초과_B0M", "M11_승인거절건수_한도초과_B0M"],
        "output": "M11_승인거절건수_한도초과_R3M",
        "fname": "cfs_03_2292",
        "type": "formula",
        "content": "M11_승인거절건수_한도초과_R3M = M09_승인거절건수_한도초과_B0M + M10_승인거절건수_한도초과_B0M + M11_승인거절건수_한도초과_B0M",
    },
    {
        "columns": ["M09_승인거절건수_BL_B0M", "M10_승인거절건수_BL_B0M", "M11_승인거절건수_BL_B0M"],
        "output": "M11_승인거절건수_BL_R3M",
        "fname": "cfs_03_2293",
        "type": "formula",
        "content": "M11_승인거절건수_BL_R3M = M09_승인거절건수_BL_B0M + M10_승인거절건수_BL_B0M + M11_승인거절건수_BL_B0M",
    },
    {
        "columns": ["M09_승인거절건수_입력오류_B0M", "M10_승인거절건수_입력오류_B0M", "M11_승인거절건수_입력오류_B0M"],
        "output": "M11_승인거절건수_입력오류_R3M",
        "fname": "cfs_03_2294",
        "type": "formula",
        "content": "M11_승인거절건수_입력오류_R3M = M09_승인거절건수_입력오류_B0M + M10_승인거절건수_입력오류_B0M + M11_승인거절건수_입력오류_B0M",
    },
    {
        "columns": ["M09_승인거절건수_기타_B0M", "M10_승인거절건수_기타_B0M", "M11_승인거절건수_기타_B0M"],
        "output": "M11_승인거절건수_기타_R3M",
        "fname": "cfs_03_2295",
        "type": "formula",
        "content": "M11_승인거절건수_기타_R3M = M09_승인거절건수_기타_B0M + M10_승인거절건수_기타_B0M + M11_승인거절건수_기타_B0M",
    },
    {
        "columns": ["M11_이용건수_신용_R3M", "M08_이용건수_신용_R3M"],
        "output": "M11_이용건수_신용_R6M",
        "fname": "cfs_03_1924",
        "type": "formula",
        "content": "M11_이용건수_신용_R6M = M11_이용건수_신용_R3M + M08_이용건수_신용_R3M",
    },
    {
        "columns": ["M11_이용건수_신판_R3M", "M08_이용건수_신판_R3M"],
        "output": "M11_이용건수_신판_R6M",
        "fname": "cfs_03_1925",
        "type": "formula",
        "content": "M11_이용건수_신판_R6M = M11_이용건수_신판_R3M + M08_이용건수_신판_R3M",
    },
    {
        "columns": ["M11_이용건수_일시불_R3M", "M08_이용건수_일시불_R3M"],
        "output": "M11_이용건수_일시불_R6M",
        "fname": "cfs_03_1926",
        "type": "formula",
        "content": "M11_이용건수_일시불_R6M = M11_이용건수_일시불_R3M + M08_이용건수_일시불_R3M",
    },
    {
        "columns": ["M11_이용건수_할부_R3M", "M08_이용건수_할부_R3M"],
        "output": "M11_이용건수_할부_R6M",
        "fname": "cfs_03_1927",
        "type": "formula",
        "content": "M11_이용건수_할부_R6M = M11_이용건수_할부_R3M + M08_이용건수_할부_R3M",
    },
    {
        "columns": ["M11_이용건수_할부_유이자_R3M", "M08_이용건수_할부_유이자_R3M"],
        "output": "M11_이용건수_할부_유이자_R6M",
        "fname": "cfs_03_1928",
        "type": "formula",
        "content": "M11_이용건수_할부_유이자_R6M = M11_이용건수_할부_유이자_R3M + M08_이용건수_할부_유이자_R3M",
    },
    {
        "columns": ["M11_이용건수_할부_무이자_R3M", "M08_이용건수_할부_무이자_R3M"],
        "output": "M11_이용건수_할부_무이자_R6M",
        "fname": "cfs_03_1929",
        "type": "formula",
        "content": "M11_이용건수_할부_무이자_R6M = M11_이용건수_할부_무이자_R3M + M08_이용건수_할부_무이자_R3M",
    },
    {
        "columns": ["M11_이용건수_부분무이자_R3M", "M08_이용건수_부분무이자_R3M"],
        "output": "M11_이용건수_부분무이자_R6M",
        "fname": "cfs_03_1930",
        "type": "formula",
        "content": "M11_이용건수_부분무이자_R6M = M11_이용건수_부분무이자_R3M + M08_이용건수_부분무이자_R3M",
    },
    {
        "columns": ["M11_이용건수_CA_R3M", "M08_이용건수_CA_R3M"],
        "output": "M11_이용건수_CA_R6M",
        "fname": "cfs_03_1931",
        "type": "formula",
        "content": "M11_이용건수_CA_R6M = M11_이용건수_CA_R3M + M08_이용건수_CA_R3M",
    },
    {
        "columns": ["M11_이용건수_체크_R3M", "M08_이용건수_체크_R3M"],
        "output": "M11_이용건수_체크_R6M",
        "fname": "cfs_03_1932",
        "type": "formula",
        "content": "M11_이용건수_체크_R6M = M11_이용건수_체크_R3M + M08_이용건수_체크_R3M",
    },
    {
        "columns": ["M11_이용건수_카드론_R3M", "M08_이용건수_카드론_R3M"],
        "output": "M11_이용건수_카드론_R6M",
        "fname": "cfs_03_1933",
        "type": "formula",
        "content": "M11_이용건수_카드론_R6M = M11_이용건수_카드론_R3M + M08_이용건수_카드론_R3M",
    },
    {
        "columns": ["M11_이용금액_신용_R3M", "M08_이용금액_신용_R3M"],
        "output": "M11_이용금액_신용_R6M",
        "fname": "cfs_03_1934",
        "type": "formula",
        "content": "M11_이용금액_신용_R6M = M11_이용금액_신용_R3M + M08_이용금액_신용_R3M",
    },
    {
        "columns": ["M11_이용금액_신판_R3M", "M08_이용금액_신판_R3M"],
        "output": "M11_이용금액_신판_R6M",
        "fname": "cfs_03_1935",
        "type": "formula",
        "content": "M11_이용금액_신판_R6M = M11_이용금액_신판_R3M + M08_이용금액_신판_R3M",
    },
    {
        "columns": ["M11_이용금액_일시불_R3M", "M08_이용금액_일시불_R3M"],
        "output": "M11_이용금액_일시불_R6M",
        "fname": "cfs_03_1936",
        "type": "formula",
        "content": "M11_이용금액_일시불_R6M = M11_이용금액_일시불_R3M + M08_이용금액_일시불_R3M",
    },
    {
        "columns": ["M11_이용금액_할부_R3M", "M08_이용금액_할부_R3M"],
        "output": "M11_이용금액_할부_R6M",
        "fname": "cfs_03_1937",
        "type": "formula",
        "content": "M11_이용금액_할부_R6M = M11_이용금액_할부_R3M + M08_이용금액_할부_R3M",
    },
    {
        "columns": ["M11_이용금액_할부_유이자_R3M", "M08_이용금액_할부_유이자_R3M"],
        "output": "M11_이용금액_할부_유이자_R6M",
        "fname": "cfs_03_1938",
        "type": "formula",
        "content": "M11_이용금액_할부_유이자_R6M = M11_이용금액_할부_유이자_R3M + M08_이용금액_할부_유이자_R3M",
    },
    {
        "columns": ["M11_이용금액_할부_무이자_R3M", "M08_이용금액_할부_무이자_R3M"],
        "output": "M11_이용금액_할부_무이자_R6M",
        "fname": "cfs_03_1939",
        "type": "formula",
        "content": "M11_이용금액_할부_무이자_R6M = M11_이용금액_할부_무이자_R3M + M08_이용금액_할부_무이자_R3M",
    },
    {
        "columns": ["M11_이용금액_부분무이자_R3M", "M08_이용금액_부분무이자_R3M"],
        "output": "M11_이용금액_부분무이자_R6M",
        "fname": "cfs_03_1940",
        "type": "formula",
        "content": "M11_이용금액_부분무이자_R6M = M11_이용금액_부분무이자_R3M + M08_이용금액_부분무이자_R3M",
    },
    {
        "columns": ["M11_이용금액_CA_R3M", "M08_이용금액_CA_R3M"],
        "output": "M11_이용금액_CA_R6M",
        "fname": "cfs_03_1941",
        "type": "formula",
        "content": "M11_이용금액_CA_R6M = M11_이용금액_CA_R3M + M08_이용금액_CA_R3M",
    },
    {
        "columns": ["M11_이용금액_체크_R3M", "M08_이용금액_체크_R3M"],
        "output": "M11_이용금액_체크_R6M",
        "fname": "cfs_03_1942",
        "type": "formula",
        "content": "M11_이용금액_체크_R6M = M11_이용금액_체크_R3M + M08_이용금액_체크_R3M",
    },
    {
        "columns": ["M11_이용금액_카드론_R3M", "M08_이용금액_카드론_R3M"],
        "output": "M11_이용금액_카드론_R6M",
        "fname": "cfs_03_1943",
        "type": "formula",
        "content": "M11_이용금액_카드론_R6M = M11_이용금액_카드론_R3M + M08_이용금액_카드론_R3M",
    },
    {
        "columns": ["M11_이용건수_신용_B0M", "M10_이용건수_신용_B0M", "M09_이용건수_신용_B0M"],
        "output": "M11_이용건수_신용_R3M",
        "fname": "cfs_03_1954",
        "type": "formula",
        "content": "M11_이용건수_신용_R3M = M11_이용건수_신용_B0M + M10_이용건수_신용_B0M + M09_이용건수_신용_B0M",
    },
    {
        "columns": ["M11_이용건수_신판_B0M", "M10_이용건수_신판_B0M", "M09_이용건수_신판_B0M"],
        "output": "M11_이용건수_신판_R3M",
        "fname": "cfs_03_1955",
        "type": "formula",
        "content": "M11_이용건수_신판_R3M = M11_이용건수_신판_B0M + M10_이용건수_신판_B0M + M09_이용건수_신판_B0M",
    },
    {
        "columns": ["M11_이용건수_일시불_B0M", "M10_이용건수_일시불_B0M", "M09_이용건수_일시불_B0M"],
        "output": "M11_이용건수_일시불_R3M",
        "fname": "cfs_03_1956",
        "type": "formula",
        "content": "M11_이용건수_일시불_R3M = M11_이용건수_일시불_B0M + M10_이용건수_일시불_B0M + M09_이용건수_일시불_B0M",
    },
    {
        "columns": ["M11_이용건수_할부_B0M", "M10_이용건수_할부_B0M", "M09_이용건수_할부_B0M"],
        "output": "M11_이용건수_할부_R3M",
        "fname": "cfs_03_1957",
        "type": "formula",
        "content": "M11_이용건수_할부_R3M = M11_이용건수_할부_B0M + M10_이용건수_할부_B0M + M09_이용건수_할부_B0M",
    },
    {
        "columns": ["M11_이용건수_할부_유이자_B0M", "M10_이용건수_할부_유이자_B0M", "M09_이용건수_할부_유이자_B0M"],
        "output": "M11_이용건수_할부_유이자_R3M",
        "fname": "cfs_03_1958",
        "type": "formula",
        "content": "M11_이용건수_할부_유이자_R3M = M11_이용건수_할부_유이자_B0M + M10_이용건수_할부_유이자_B0M + M09_이용건수_할부_유이자_B0M",
    },
    {
        "columns": ["M11_이용건수_할부_무이자_B0M", "M10_이용건수_할부_무이자_B0M", "M09_이용건수_할부_무이자_B0M"],
        "output": "M11_이용건수_할부_무이자_R3M",
        "fname": "cfs_03_1959",
        "type": "formula",
        "content": "M11_이용건수_할부_무이자_R3M = M11_이용건수_할부_무이자_B0M + M10_이용건수_할부_무이자_B0M + M09_이용건수_할부_무이자_B0M",
    },
    {
        "columns": ["M11_이용건수_부분무이자_B0M", "M10_이용건수_부분무이자_B0M", "M09_이용건수_부분무이자_B0M"],
        "output": "M11_이용건수_부분무이자_R3M",
        "fname": "cfs_03_1960",
        "type": "formula",
        "content": "M11_이용건수_부분무이자_R3M = M11_이용건수_부분무이자_B0M + M10_이용건수_부분무이자_B0M + M09_이용건수_부분무이자_B0M",
    },
    {
        "columns": ["M11_이용건수_CA_B0M", "M10_이용건수_CA_B0M", "M09_이용건수_CA_B0M"],
        "output": "M11_이용건수_CA_R3M",
        "fname": "cfs_03_1961",
        "type": "formula",
        "content": "M11_이용건수_CA_R3M = M11_이용건수_CA_B0M + M10_이용건수_CA_B0M + M09_이용건수_CA_B0M",
    },
    {
        "columns": ["M11_이용건수_체크_B0M", "M10_이용건수_체크_B0M", "M09_이용건수_체크_B0M"],
        "output": "M11_이용건수_체크_R3M",
        "fname": "cfs_03_1962",
        "type": "formula",
        "content": "M11_이용건수_체크_R3M = M11_이용건수_체크_B0M + M10_이용건수_체크_B0M + M09_이용건수_체크_B0M",
    },
    {
        "columns": ["M11_이용건수_카드론_B0M", "M10_이용건수_카드론_B0M", "M09_이용건수_카드론_B0M"],
        "output": "M11_이용건수_카드론_R3M",
        "fname": "cfs_03_1963",
        "type": "formula",
        "content": "M11_이용건수_카드론_R3M = M11_이용건수_카드론_B0M + M10_이용건수_카드론_B0M + M09_이용건수_카드론_B0M",
    },
    {
        "columns": ["M11_이용금액_신용_B0M", "M10_이용금액_신용_B0M", "M09_이용금액_신용_B0M"],
        "output": "M11_이용금액_신용_R3M",
        "fname": "cfs_03_1964",
        "type": "formula",
        "content": "M11_이용금액_신용_R3M = M11_이용금액_신용_B0M + M10_이용금액_신용_B0M + M09_이용금액_신용_B0M",
    },
    {
        "columns": ["M11_이용금액_신판_B0M", "M10_이용금액_신판_B0M", "M09_이용금액_신판_B0M"],
        "output": "M11_이용금액_신판_R3M",
        "fname": "cfs_03_1965",
        "type": "formula",
        "content": "M11_이용금액_신판_R3M = M11_이용금액_신판_B0M + M10_이용금액_신판_B0M + M09_이용금액_신판_B0M",
    },
    {
        "columns": ["M11_이용금액_일시불_B0M", "M10_이용금액_일시불_B0M", "M09_이용금액_일시불_B0M"],
        "output": "M11_이용금액_일시불_R3M",
        "fname": "cfs_03_1966",
        "type": "formula",
        "content": "M11_이용금액_일시불_R3M = M11_이용금액_일시불_B0M + M10_이용금액_일시불_B0M + M09_이용금액_일시불_B0M",
    },
    {
        "columns": ["M11_이용금액_할부_B0M", "M10_이용금액_할부_B0M", "M09_이용금액_할부_B0M"],
        "output": "M11_이용금액_할부_R3M",
        "fname": "cfs_03_1967",
        "type": "formula",
        "content": "M11_이용금액_할부_R3M = M11_이용금액_할부_B0M + M10_이용금액_할부_B0M + M09_이용금액_할부_B0M",
    },
    {
        "columns": ["M11_이용금액_할부_유이자_B0M", "M10_이용금액_할부_유이자_B0M", "M09_이용금액_할부_유이자_B0M"],
        "output": "M11_이용금액_할부_유이자_R3M",
        "fname": "cfs_03_1968",
        "type": "formula",
        "content": "M11_이용금액_할부_유이자_R3M = M11_이용금액_할부_유이자_B0M + M10_이용금액_할부_유이자_B0M + M09_이용금액_할부_유이자_B0M",
    },
    {
        "columns": ["M11_이용금액_할부_무이자_B0M", "M10_이용금액_할부_무이자_B0M", "M09_이용금액_할부_무이자_B0M"],
        "output": "M11_이용금액_할부_무이자_R3M",
        "fname": "cfs_03_1969",
        "type": "formula",
        "content": "M11_이용금액_할부_무이자_R3M = M11_이용금액_할부_무이자_B0M + M10_이용금액_할부_무이자_B0M + M09_이용금액_할부_무이자_B0M",
    },
    {
        "columns": ["M11_이용금액_부분무이자_B0M", "M10_이용금액_부분무이자_B0M", "M09_이용금액_부분무이자_B0M"],
        "output": "M11_이용금액_부분무이자_R3M",
        "fname": "cfs_03_1970",
        "type": "formula",
        "content": "M11_이용금액_부분무이자_R3M = M11_이용금액_부분무이자_B0M + M10_이용금액_부분무이자_B0M + M09_이용금액_부분무이자_B0M",
    },
    {
        "columns": ["M11_이용금액_CA_B0M", "M10_이용금액_CA_B0M", "M09_이용금액_CA_B0M"],
        "output": "M11_이용금액_CA_R3M",
        "fname": "cfs_03_1971",
        "type": "formula",
        "content": "M11_이용금액_CA_R3M = M11_이용금액_CA_B0M + M10_이용금액_CA_B0M + M09_이용금액_CA_B0M",
    },
    {
        "columns": ["M11_이용금액_체크_B0M", "M10_이용금액_체크_B0M", "M09_이용금액_체크_B0M"],
        "output": "M11_이용금액_체크_R3M",
        "fname": "cfs_03_1972",
        "type": "formula",
        "content": "M11_이용금액_체크_R3M = M11_이용금액_체크_B0M + M10_이용금액_체크_B0M + M09_이용금액_체크_B0M",
    },
    {
        "columns": ["M11_이용금액_카드론_B0M", "M10_이용금액_카드론_B0M", "M09_이용금액_카드론_B0M"],
        "output": "M11_이용금액_카드론_R3M",
        "fname": "cfs_03_1973",
        "type": "formula",
        "content": "M11_이용금액_카드론_R3M = M11_이용금액_카드론_B0M + M10_이용금액_카드론_B0M + M09_이용금액_카드론_B0M",
    },
    {
        "columns": ["M11_건수_할부전환_R3M", "M08_건수_할부전환_R3M"],
        "output": "M11_건수_할부전환_R6M",
        "fname": "cfs_03_1988",
        "type": "formula",
        "content": "M11_건수_할부전환_R6M = M11_건수_할부전환_R3M + M08_건수_할부전환_R3M",
    },
    {
        "columns": ["M11_금액_할부전환_R3M", "M08_금액_할부전환_R3M"],
        "output": "M11_금액_할부전환_R6M",
        "fname": "cfs_03_1989",
        "type": "formula",
        "content": "M11_금액_할부전환_R6M = M11_금액_할부전환_R3M + M08_금액_할부전환_R3M",
    },
    {
        "columns": ["M11_최종카드론_대출일자", "M10_최종카드론_대출일자", "M10_최종카드론이용경과월"],
        "output": "M11_최종카드론이용경과월",
        "fname": "cfs_03_2119",
        "type": "formula",
        "content": """IF M11_최종카드론_대출일자 IS NULL
                      THEN M11_최종카드론이용경과월 = 999
                      ELIF M11_최종카드론_대출일자 = M10_최종카드론_대출일자
                      THEN M11_최종카드론이용경과월 = M10_최종카드론이용경과월 + 1
                      ELSE M11_최종카드론이용경과월 = 0""",
    },
    {
        "columns": ["M11_이용금액_온라인_R3M", "M08_이용금액_온라인_R3M"],
        "output": "M11_이용금액_온라인_R6M",
        "fname": "cfs_03_2140",
        "type": "formula",
        "content": "M11_이용금액_온라인_R6M = M11_이용금액_온라인_R3M + M08_이용금액_온라인_R3M",
    },
    {
        "columns": ["M11_이용금액_오프라인_R3M", "M08_이용금액_오프라인_R3M"],
        "output": "M11_이용금액_오프라인_R6M",
        "fname": "cfs_03_2141",
        "type": "formula",
        "content": "M11_이용금액_오프라인_R6M = M11_이용금액_오프라인_R3M + M08_이용금액_오프라인_R3M",
    },
    {
        "columns": ["M11_이용건수_온라인_R3M", "M08_이용건수_온라인_R3M"],
        "output": "M11_이용건수_온라인_R6M",
        "fname": "cfs_03_2142",
        "type": "formula",
        "content": "M11_이용건수_온라인_R6M = M11_이용건수_온라인_R3M + M08_이용건수_온라인_R3M",
    },
    {
        "columns": ["M11_이용건수_오프라인_R3M", "M08_이용건수_오프라인_R3M"],
        "output": "M11_이용건수_오프라인_R6M",
        "fname": "cfs_03_2143",
        "type": "formula",
        "content": "M11_이용건수_오프라인_R6M = M11_이용건수_오프라인_R3M + M08_이용건수_오프라인_R3M",
    },
    {
        "columns": ["M11_이용금액_온라인_B0M", "M10_이용금액_온라인_B0M", "M09_이용금액_온라인_B0M"],
        "output": "M11_이용금액_온라인_R3M",
        "fname": "cfs_03_2144",
        "type": "formula",
        "content": "M11_이용금액_온라인_R3M = M11_이용금액_온라인_B0M + M10_이용금액_온라인_B0M + M09_이용금액_온라인_B0M",
    },
    {
        "columns": ["M11_이용금액_오프라인_B0M", "M10_이용금액_오프라인_B0M", "M09_이용금액_오프라인_B0M"],
        "output": "M11_이용금액_오프라인_R3M",
        "fname": "cfs_03_2145",
        "type": "formula",
        "content": "M11_이용금액_오프라인_R3M = M11_이용금액_오프라인_B0M + M10_이용금액_오프라인_B0M + M09_이용금액_오프라인_B0M",
    },
    {
        "columns": ["M11_이용건수_온라인_B0M", "M10_이용건수_온라인_B0M", "M09_이용건수_온라인_B0M"],
        "output": "M11_이용건수_온라인_R3M",
        "fname": "cfs_03_2146",
        "type": "formula",
        "content": "M11_이용건수_온라인_R3M = M11_이용건수_온라인_B0M + M10_이용건수_온라인_B0M + M09_이용건수_온라인_B0M",
    },
    {
        "columns": ["M11_이용건수_오프라인_B0M", "M10_이용건수_오프라인_B0M", "M09_이용건수_오프라인_B0M"],
        "output": "M11_이용건수_오프라인_R3M",
        "fname": "cfs_03_2147",
        "type": "formula",
        "content": "M11_이용건수_오프라인_R3M = M11_이용건수_오프라인_B0M + M10_이용건수_오프라인_B0M + M09_이용건수_오프라인_B0M",
    },
    {
        "columns": ["M11_이용금액_페이_온라인_R3M", "M08_이용금액_페이_온라인_R3M"],
        "output": "M11_이용금액_페이_온라인_R6M",
        "fname": "cfs_03_2154",
        "type": "formula",
        "content": "M11_이용금액_페이_온라인_R6M = M11_이용금액_페이_온라인_R3M + M08_이용금액_페이_온라인_R3M",
    },
    {
        "columns": ["M11_이용금액_페이_오프라인_R3M", "M08_이용금액_페이_오프라인_R3M"],
        "output": "M11_이용금액_페이_오프라인_R6M",
        "fname": "cfs_03_2155",
        "type": "formula",
        "content": "M11_이용금액_페이_오프라인_R6M = M11_이용금액_페이_오프라인_R3M + M08_이용금액_페이_오프라인_R3M",
    },
    {
        "columns": ["M11_이용건수_페이_온라인_R3M", "M08_이용건수_페이_온라인_R3M"],
        "output": "M11_이용건수_페이_온라인_R6M",
        "fname": "cfs_03_2156",
        "type": "formula",
        "content": "M11_이용건수_페이_온라인_R6M = M11_이용건수_페이_온라인_R3M + M08_이용건수_페이_온라인_R3M",
    },
    {
        "columns": ["M11_이용건수_페이_오프라인_R3M", "M08_이용건수_페이_오프라인_R3M"],
        "output": "M11_이용건수_페이_오프라인_R6M",
        "fname": "cfs_03_2157",
        "type": "formula",
        "content": "M11_이용건수_페이_오프라인_R6M = M11_이용건수_페이_오프라인_R3M + M08_이용건수_페이_오프라인_R3M",
    },
    {
        "columns": ["M11_이용금액_페이_온라인_B0M", "M10_이용금액_페이_온라인_B0M", "M09_이용금액_페이_온라인_B0M"],
        "output": "M11_이용금액_페이_온라인_R3M",
        "fname": "cfs_03_2158",
        "type": "formula",
        "content": "M11_이용금액_페이_온라인_R3M = M11_이용금액_페이_온라인_B0M + M10_이용금액_페이_온라인_B0M + M09_이용금액_페이_온라인_B0M",
    },
    {
        "columns": ["M11_이용금액_페이_오프라인_B0M", "M10_이용금액_페이_오프라인_B0M", "M09_이용금액_페이_오프라인_B0M"],
        "output": "M11_이용금액_페이_오프라인_R3M",
        "fname": "cfs_03_2159",
        "type": "formula",
        "content": "M11_이용금액_페이_오프라인_R3M = M11_이용금액_페이_오프라인_B0M + M10_이용금액_페이_오프라인_B0M + M09_이용금액_페이_오프라인_B0M",
    },
    {
        "columns": ["M11_이용건수_페이_온라인_B0M", "M10_이용건수_페이_온라인_B0M", "M09_이용건수_페이_온라인_B0M"],
        "output": "M11_이용건수_페이_온라인_R3M",
        "fname": "cfs_03_2160",
        "type": "formula",
        "content": "M11_이용건수_페이_온라인_R3M = M11_이용건수_페이_온라인_B0M + M10_이용건수_페이_온라인_B0M + M09_이용건수_페이_온라인_B0M",
    },
    {
        "columns": ["M11_이용건수_페이_오프라인_B0M", "M10_이용건수_페이_오프라인_B0M", "M09_이용건수_페이_오프라인_B0M"],
        "output": "M11_이용건수_페이_오프라인_R3M",
        "fname": "cfs_03_2161",
        "type": "formula",
        "content": "M11_이용건수_페이_오프라인_R3M = M11_이용건수_페이_오프라인_B0M + M10_이용건수_페이_오프라인_B0M + M09_이용건수_페이_오프라인_B0M",
    },
    {
        "columns": ["M11_이용금액_간편결제_R3M", "M08_이용금액_간편결제_R3M"],
        "output": "M11_이용금액_간편결제_R6M",
        "fname": "cfs_03_2173",
        "type": "formula",
        "content": "M11_이용금액_간편결제_R6M = M11_이용금액_간편결제_R3M + M08_이용금액_간편결제_R3M",
    },
    {
        "columns": ["M11_이용금액_당사페이_R3M", "M08_이용금액_당사페이_R3M"],
        "output": "M11_이용금액_당사페이_R6M",
        "fname": "cfs_03_2174",
        "type": "formula",
        "content": "M11_이용금액_당사페이_R6M = M11_이용금액_당사페이_R3M + M08_이용금액_당사페이_R3M",
    },
    {
        "columns": ["M11_이용금액_당사기타_R3M", "M08_이용금액_당사기타_R3M"],
        "output": "M11_이용금액_당사기타_R6M",
        "fname": "cfs_03_2175",
        "type": "formula",
        "content": "M11_이용금액_당사기타_R6M = M11_이용금액_당사기타_R3M + M08_이용금액_당사기타_R3M",
    },
    {
        "columns": ["M11_이용금액_A페이_R3M", "M08_이용금액_A페이_R3M"],
        "output": "M11_이용금액_A페이_R6M",
        "fname": "cfs_03_2176",
        "type": "formula",
        "content": "M11_이용금액_A페이_R6M = M11_이용금액_A페이_R3M + M08_이용금액_A페이_R3M",
    },
    {
        "columns": ["M11_이용금액_B페이_R3M", "M08_이용금액_B페이_R3M"],
        "output": "M11_이용금액_B페이_R6M",
        "fname": "cfs_03_2177",
        "type": "formula",
        "content": "M11_이용금액_B페이_R6M = M11_이용금액_B페이_R3M + M08_이용금액_B페이_R3M",
    },
    {
        "columns": ["M11_이용금액_C페이_R3M", "M08_이용금액_C페이_R3M"],
        "output": "M11_이용금액_C페이_R6M",
        "fname": "cfs_03_2178",
        "type": "formula",
        "content": "M11_이용금액_C페이_R6M = M11_이용금액_C페이_R3M + M08_이용금액_C페이_R3M",
    },
    {
        "columns": ["M11_이용금액_D페이_R3M", "M08_이용금액_D페이_R3M"],
        "output": "M11_이용금액_D페이_R6M",
        "fname": "cfs_03_2179",
        "type": "formula",
        "content": "M11_이용금액_D페이_R6M = M11_이용금액_D페이_R3M + M08_이용금액_D페이_R3M",
    },
    {
        "columns": ["M11_이용건수_간편결제_R3M", "M08_이용건수_간편결제_R3M"],
        "output": "M11_이용건수_간편결제_R6M",
        "fname": "cfs_03_2180",
        "type": "formula",
        "content": "M11_이용건수_간편결제_R6M = M11_이용건수_간편결제_R3M + M08_이용건수_간편결제_R3M",
    },
    {
        "columns": ["M11_이용건수_당사페이_R3M", "M08_이용건수_당사페이_R3M"],
        "output": "M11_이용건수_당사페이_R6M",
        "fname": "cfs_03_2181",
        "type": "formula",
        "content": "M11_이용건수_당사페이_R6M = M11_이용건수_당사페이_R3M + M08_이용건수_당사페이_R3M",
    },
    {
        "columns": ["M11_이용건수_당사기타_R3M", "M08_이용건수_당사기타_R3M"],
        "output": "M11_이용건수_당사기타_R6M",
        "fname": "cfs_03_2182",
        "type": "formula",
        "content": "M11_이용건수_당사기타_R6M = M11_이용건수_당사기타_R3M + M08_이용건수_당사기타_R3M",
    },
    {
        "columns": ["M11_이용건수_A페이_R3M", "M08_이용건수_A페이_R3M"],
        "output": "M11_이용건수_A페이_R6M",
        "fname": "cfs_03_2183",
        "type": "formula",
        "content": "M11_이용건수_A페이_R6M = M11_이용건수_A페이_R3M + M08_이용건수_A페이_R3M",
    },
    {
        "columns": ["M11_이용건수_B페이_R3M", "M08_이용건수_B페이_R3M"],
        "output": "M11_이용건수_B페이_R6M",
        "fname": "cfs_03_2184",
        "type": "formula",
        "content": "M11_이용건수_B페이_R6M = M11_이용건수_B페이_R3M + M08_이용건수_B페이_R3M",
    },
    {
        "columns": ["M11_이용건수_C페이_R3M", "M08_이용건수_C페이_R3M"],
        "output": "M11_이용건수_C페이_R6M",
        "fname": "cfs_03_2185",
        "type": "formula",
        "content": "M11_이용건수_C페이_R6M = M11_이용건수_C페이_R3M + M08_이용건수_C페이_R3M",
    },
    {
        "columns": ["M11_이용건수_D페이_R3M", "M08_이용건수_D페이_R3M"],
        "output": "M11_이용건수_D페이_R6M",
        "fname": "cfs_03_2186",
        "type": "formula",
        "content": "M11_이용건수_D페이_R6M = M11_이용건수_D페이_R3M + M08_이용건수_D페이_R3M",
    },
    {
        "columns": ["M11_이용금액_간편결제_B0M", "M10_이용금액_간편결제_B0M", "M09_이용금액_간편결제_B0M"],
        "output": "M11_이용금액_간편결제_R3M",
        "fname": "cfs_03_2187",
        "type": "formula",
        "content": "M11_이용금액_간편결제_R3M = M11_이용금액_간편결제_B0M + M10_이용금액_간편결제_B0M + M09_이용금액_간편결제_B0M",
    },
    {
        "columns": ["M11_이용금액_당사페이_B0M", "M10_이용금액_당사페이_B0M", "M09_이용금액_당사페이_B0M"],
        "output": "M11_이용금액_당사페이_R3M",
        "fname": "cfs_03_2188",
        "type": "formula",
        "content": "M11_이용금액_당사페이_R3M = M11_이용금액_당사페이_B0M + M10_이용금액_당사페이_B0M + M09_이용금액_당사페이_B0M",
    },
    {
        "columns": ["M11_이용금액_당사기타_B0M", "M10_이용금액_당사기타_B0M", "M09_이용금액_당사기타_B0M"],
        "output": "M11_이용금액_당사기타_R3M",
        "fname": "cfs_03_2189",
        "type": "formula",
        "content": "M11_이용금액_당사기타_R3M = M11_이용금액_당사기타_B0M + M10_이용금액_당사기타_B0M + M09_이용금액_당사기타_B0M",
    },
    {
        "columns": ["M11_이용금액_A페이_B0M", "M10_이용금액_A페이_B0M", "M09_이용금액_A페이_B0M"],
        "output": "M11_이용금액_A페이_R3M",
        "fname": "cfs_03_2190",
        "type": "formula",
        "content": "M11_이용금액_A페이_R3M = M11_이용금액_A페이_B0M + M10_이용금액_A페이_B0M + M09_이용금액_A페이_B0M",
    },
    {
        "columns": ["M11_이용금액_B페이_B0M", "M10_이용금액_B페이_B0M", "M09_이용금액_B페이_B0M"],
        "output": "M11_이용금액_B페이_R3M",
        "fname": "cfs_03_2191",
        "type": "formula",
        "content": "M11_이용금액_B페이_R3M = M11_이용금액_B페이_B0M + M10_이용금액_B페이_B0M + M09_이용금액_B페이_B0M",
    },
    {
        "columns": ["M11_이용금액_C페이_B0M", "M10_이용금액_C페이_B0M", "M09_이용금액_C페이_B0M"],
        "output": "M11_이용금액_C페이_R3M",
        "fname": "cfs_03_2192",
        "type": "formula",
        "content": "M11_이용금액_C페이_R3M = M11_이용금액_C페이_B0M + M10_이용금액_C페이_B0M + M09_이용금액_C페이_B0M",
    },
    {
        "columns": ["M11_이용금액_D페이_B0M", "M10_이용금액_D페이_B0M", "M09_이용금액_D페이_B0M"],
        "output": "M11_이용금액_D페이_R3M",
        "fname": "cfs_03_2193",
        "type": "formula",
        "content": "M11_이용금액_D페이_R3M = M11_이용금액_D페이_B0M + M10_이용금액_D페이_B0M + M09_이용금액_D페이_B0M",
    },
    {
        "columns": ["M11_이용건수_간편결제_B0M", "M10_이용건수_간편결제_B0M", "M09_이용건수_간편결제_B0M"],
        "output": "M11_이용건수_간편결제_R3M",
        "fname": "cfs_03_2194",
        "type": "formula",
        "content": "M11_이용건수_간편결제_R3M = M11_이용건수_간편결제_B0M + M10_이용건수_간편결제_B0M + M09_이용건수_간편결제_B0M",
    },
    {
        "columns": ["M11_이용건수_당사페이_B0M", "M10_이용건수_당사페이_B0M", "M09_이용건수_당사페이_B0M"],
        "output": "M11_이용건수_당사페이_R3M",
        "fname": "cfs_03_2195",
        "type": "formula",
        "content": "M11_이용건수_당사페이_R3M = M11_이용건수_당사페이_B0M + M10_이용건수_당사페이_B0M + M09_이용건수_당사페이_B0M",
    },
    {
        "columns": ["M11_이용건수_당사기타_B0M", "M10_이용건수_당사기타_B0M", "M09_이용건수_당사기타_B0M"],
        "output": "M11_이용건수_당사기타_R3M",
        "fname": "cfs_03_2196",
        "type": "formula",
        "content": "M11_이용건수_당사기타_R3M = M11_이용건수_당사기타_B0M + M10_이용건수_당사기타_B0M + M09_이용건수_당사기타_B0M",
    },
    {
        "columns": ["M11_이용건수_A페이_B0M", "M10_이용건수_A페이_B0M", "M09_이용건수_A페이_B0M"],
        "output": "M11_이용건수_A페이_R3M",
        "fname": "cfs_03_2197",
        "type": "formula",
        "content": "M11_이용건수_A페이_R3M = M11_이용건수_A페이_B0M + M10_이용건수_A페이_B0M + M09_이용건수_A페이_B0M",
    },
    {
        "columns": ["M11_이용건수_B페이_B0M", "M10_이용건수_B페이_B0M", "M09_이용건수_B페이_B0M"],
        "output": "M11_이용건수_B페이_R3M",
        "fname": "cfs_03_2198",
        "type": "formula",
        "content": "M11_이용건수_B페이_R3M = M11_이용건수_B페이_B0M + M10_이용건수_B페이_B0M + M09_이용건수_B페이_B0M",
    },
    {
        "columns": ["M11_이용건수_C페이_B0M", "M10_이용건수_C페이_B0M", "M09_이용건수_C페이_B0M"],
        "output": "M11_이용건수_C페이_R3M",
        "fname": "cfs_03_2199",
        "type": "formula",
        "content": "M11_이용건수_C페이_R3M = M11_이용건수_C페이_B0M + M10_이용건수_C페이_B0M + M09_이용건수_C페이_B0M",
    },
    {
        "columns": ["M11_이용건수_D페이_B0M", "M10_이용건수_D페이_B0M", "M09_이용건수_D페이_B0M"],
        "output": "M11_이용건수_D페이_R3M",
        "fname": "cfs_03_2200",
        "type": "formula",
        "content": "M11_이용건수_D페이_R3M = M11_이용건수_D페이_B0M + M10_이용건수_D페이_B0M + M09_이용건수_D페이_B0M",
    },
    {
        "columns": ["M11_이용횟수_선결제_R3M", "M08_이용횟수_선결제_R3M"],
        "output": "M11_이용횟수_선결제_R6M",
        "fname": "cfs_03_2216",
        "type": "formula",
        "content": "M11_이용횟수_선결제_R6M = M11_이용횟수_선결제_R3M + M08_이용횟수_선결제_R3M",
    },
    {
        "columns": ["M11_이용금액_선결제_R3M", "M08_이용금액_선결제_R3M"],
        "output": "M11_이용금액_선결제_R6M",
        "fname": "cfs_03_2217",
        "type": "formula",
        "content": "M11_이용금액_선결제_R6M = M11_이용금액_선결제_R3M + M08_이용금액_선결제_R3M",
    },
    {
        "columns": ["M11_이용건수_선결제_R3M", "M08_이용건수_선결제_R3M"],
        "output": "M11_이용건수_선결제_R6M",
        "fname": "cfs_03_2218",
        "type": "formula",
        "content": "M11_이용건수_선결제_R6M = M11_이용건수_선결제_R3M + M08_이용건수_선결제_R3M",
    },
    {
        "columns": ["M11_이용횟수_선결제_B0M", "M10_이용횟수_선결제_B0M", "M09_이용횟수_선결제_B0M"],
        "output": "M11_이용횟수_선결제_R3M",
        "fname": "cfs_03_2219",
        "type": "formula",
        "content": "M11_이용횟수_선결제_R3M = M11_이용횟수_선결제_B0M + M10_이용횟수_선결제_B0M + M09_이용횟수_선결제_B0M",
    },
    {
        "columns": ["M11_이용금액_선결제_B0M", "M10_이용금액_선결제_B0M", "M09_이용금액_선결제_B0M"],
        "output": "M11_이용금액_선결제_R3M",
        "fname": "cfs_03_2220",
        "type": "formula",
        "content": "M11_이용금액_선결제_R3M = M11_이용금액_선결제_B0M + M10_이용금액_선결제_B0M + M09_이용금액_선결제_B0M",
    },
    {
        "columns": ["M11_이용건수_선결제_B0M", "M10_이용건수_선결제_B0M", "M09_이용건수_선결제_B0M"],
        "output": "M11_이용건수_선결제_R3M",
        "fname": "cfs_03_2221",
        "type": "formula",
        "content": "M11_이용건수_선결제_R3M = M11_이용건수_선결제_B0M + M10_이용건수_선결제_B0M + M09_이용건수_선결제_B0M",
    },
    {
        "columns": ["M10_가맹점매출금액_B1M"],
        "output": "M11_가맹점매출금액_B2M",
        "fname": "cfs_03_2226",
        "type": "formula",
        "content": "M11_가맹점매출금액_B2M = M10_가맹점매출금액_B1M",
    },
    {
        "columns": ["M09_정상청구원금_B0M"],
        "output": "M11_정상청구원금_B2M",
        "fname": "cfs_03_2228",
        "type": "formula",
        "content": "M11_정상청구원금_B2M = M09_정상청구원금_B0M",
    },
    {
        "columns": ["M09_선입금원금_B0M"],
        "output": "M11_선입금원금_B2M",
        "fname": "cfs_03_2231",
        "type": "formula",
        "content": "M11_선입금원금_B2M = M09_선입금원금_B0M",
    },
    {
        "columns": ["M09_정상입금원금_B0M"],
        "output": "M11_정상입금원금_B2M",
        "fname": "cfs_03_2234",
        "type": "formula",
        "content": "M11_정상입금원금_B2M = M09_정상입금원금_B0M",
    },
    {
        "columns": ["M09_연체입금원금_B0M"],
        "output": "M11_연체입금원금_B2M",
        "fname": "cfs_03_2237",
        "type": "formula",
        "content": "M11_연체입금원금_B2M = M09_연체입금원금_B0M",
    },
    {
        "columns": ["M11_이용횟수_연체_R3M", "M08_이용횟수_연체_R3M"],
        "output": "M11_이용횟수_연체_R6M",
        "fname": "cfs_03_2243",
        "type": "formula",
        "content": "M11_이용횟수_연체_R6M = M11_이용횟수_연체_R3M + M08_이용횟수_연체_R3M",
    },
    {
        "columns": ["M11_이용금액_연체_R3M", "M08_이용금액_연체_R3M"],
        "output": "M11_이용금액_연체_R6M",
        "fname": "cfs_03_2244",
        "type": "formula",
        "content": "M11_이용금액_연체_R6M = M11_이용금액_연체_R3M + M08_이용금액_연체_R3M",
    },
    {
        "columns": ["M11_이용횟수_연체_B0M", "M10_이용횟수_연체_B0M", "M09_이용횟수_연체_B0M"],
        "output": "M11_이용횟수_연체_R3M",
        "fname": "cfs_03_2245",
        "type": "formula",
        "content": "M11_이용횟수_연체_R3M = M11_이용횟수_연체_B0M + M10_이용횟수_연체_B0M + M09_이용횟수_연체_B0M",
    },
    {
        "columns": ["M11_이용금액_연체_B0M", "M10_이용금액_연체_B0M", "M09_이용금액_연체_B0M"],
        "output": "M11_이용금액_연체_R3M",
        "fname": "cfs_03_2246",
        "type": "formula",
        "content": "M11_이용금액_연체_R3M = M11_이용금액_연체_B0M + M10_이용금액_연체_B0M + M09_이용금액_연체_B0M",
    },
    {
        "columns": ["M10_RP건수_B0M", "M11_RP건수_B0M"],
        "output": "M11_증감_RP건수_전월",
        "fname": "cfs_03_2259",
        "type": "formula",
        "content": "M11_증감_RP건수_전월 = M10_RP건수_B0M - M11_RP건수_B0M",
    },
    {
        "columns": ["M10_RP건수_통신_B0M", "M11_RP건수_통신_B0M"],
        "output": "M11_증감_RP건수_통신_전월",
        "fname": "cfs_03_2261",
        "type": "formula",
        "content": "M11_증감_RP건수_통신_전월 = M10_RP건수_통신_B0M - M11_RP건수_통신_B0M",
    },
    {
        "columns": ["M10_RP건수_아파트_B0M", "M11_RP건수_아파트_B0M"],
        "output": "M11_증감_RP건수_아파트_전월",
        "fname": "cfs_03_2262",
        "type": "formula",
        "content": "M11_증감_RP건수_아파트_전월 = M10_RP건수_아파트_B0M - M11_RP건수_아파트_B0M",
    },
    {
        "columns": ["M10_RP건수_제휴사서비스직접판매_B0M", "M11_RP건수_제휴사서비스직접판매_B0M"],
        "output": "M11_증감_RP건수_제휴사서비스직접판매_전월",
        "fname": "cfs_03_2263",
        "type": "formula",
        "content": "M11_증감_RP건수_제휴사서비스직접판매_전월 = M10_RP건수_제휴사서비스직접판매_B0M - M11_RP건수_제휴사서비스직접판매_B0M",
    },
    {
        "columns": ["M10_RP건수_렌탈_B0M", "M11_RP건수_렌탈_B0M"],
        "output": "M11_증감_RP건수_렌탈_전월",
        "fname": "cfs_03_2264",
        "type": "formula",
        "content": "M11_증감_RP건수_렌탈_전월 = M10_RP건수_렌탈_B0M - M11_RP건수_렌탈_B0M",
    },
    {
        "columns": ["M10_RP건수_가스_B0M", "M11_RP건수_가스_B0M"],
        "output": "M11_증감_RP건수_가스_전월",
        "fname": "cfs_03_2265",
        "type": "formula",
        "content": "M11_증감_RP건수_가스_전월 = M10_RP건수_가스_B0M - M11_RP건수_가스_B0M",
    },
    {
        "columns": ["M10_RP건수_전기_B0M", "M11_RP건수_전기_B0M"],
        "output": "M11_증감_RP건수_전기_전월",
        "fname": "cfs_03_2266",
        "type": "formula",
        "content": "M11_증감_RP건수_전기_전월 = M10_RP건수_전기_B0M - M11_RP건수_전기_B0M",
    },
    {
        "columns": ["M10_RP건수_보험_B0M", "M11_RP건수_보험_B0M"],
        "output": "M11_증감_RP건수_보험_전월",
        "fname": "cfs_03_2267",
        "type": "formula",
        "content": "M11_증감_RP건수_보험_전월 = M10_RP건수_보험_B0M - M11_RP건수_보험_B0M",
    },
    {
        "columns": ["M10_RP건수_학습비_B0M", "M11_RP건수_학습비_B0M"],
        "output": "M11_증감_RP건수_학습비_전월",
        "fname": "cfs_03_2268",
        "type": "formula",
        "content": "M11_증감_RP건수_학습비_전월 = M10_RP건수_학습비_B0M - M11_RP건수_학습비_B0M",
    },
    {
        "columns": ["M10_RP건수_유선방송_B0M", "M11_RP건수_유선방송_B0M"],
        "output": "M11_증감_RP건수_유선방송_전월",
        "fname": "cfs_03_2269",
        "type": "formula",
        "content": "M11_증감_RP건수_유선방송_전월 = M10_RP건수_유선방송_B0M - M11_RP건수_유선방송_B0M",
    },
    {
        "columns": ["M10_RP건수_건강_B0M", "M11_RP건수_건강_B0M"],
        "output": "M11_증감_RP건수_건강_전월",
        "fname": "cfs_03_2270",
        "type": "formula",
        "content": "M11_증감_RP건수_건강_전월 = M10_RP건수_건강_B0M - M11_RP건수_건강_B0M",
    },
    {
        "columns": ["M10_RP건수_교통_B0M", "M11_RP건수_교통_B0M"],
        "output": "M11_증감_RP건수_교통_전월",
        "fname": "cfs_03_2271",
        "type": "formula",
        "content": "M11_증감_RP건수_교통_전월 = M10_RP건수_교통_B0M - M11_RP건수_교통_B0M",
    },
    # M12
    {
        "columns": ["M12_RP건수_B0M", "M11_RP후경과월"],
        "output": "M12_RP후경과월",
        "fname": "cfs_03_2567",
        "type": "formula",
        "content": """IF M12_RP건수_B0M > 0 THEN M12_RP후경과월 = 0
                      ELSE M12_RP후경과월 = min(M11_RP후경과월 + 1, 6)""",
    },
    {
        "columns": ["M12_RP건수_통신_B0M", "M11_RP후경과월_통신"],
        "output": "M12_RP후경과월_통신",
        "fname": "cfs_03_2568",
        "type": "formula",
        "content": """IF M12_RP건수_통신_B0M > 0 THEN M12_RP후경과월_통신 = 0
                      ELSE M12_RP후경과월_통신 = min(M11_RP후경과월_통신 + 1, 6)""",
    },
    {
        "columns": ["M12_RP건수_아파트_B0M", "M11_RP후경과월_아파트"],
        "output": "M12_RP후경과월_아파트",
        "fname": "cfs_03_2569",
        "type": "formula",
        "content": """IF M12_RP건수_아파트_B0M > 0 THEN M12_RP후경과월_아파트 = 0
                      ELSE M12_RP후경과월_아파트 = min(M11_RP후경과월_아파트 + 1, 6)""",
    },
    {
        "columns": ["M12_RP건수_제휴사서비스직접판매_B0M", "M11_RP후경과월_제휴사서비스직접판매"],
        "output": "M12_RP후경과월_제휴사서비스직접판매",
        "fname": "cfs_03_2570",
        "type": "formula",
        "content": """IF M12_RP건수_제휴사서비스직접판매_B0M > 0 THEN M12_RP후경과월_제휴사서비스직접판매 = 0
                      ELSE M12_RP후경과월_제휴사서비스직접판매 = min(M11_RP후경과월_제휴사서비스직접판매 + 1, 6)""",
    },
    {
        "columns": ["M12_RP건수_렌탈_B0M", "M11_RP후경과월_렌탈"],
        "output": "M12_RP후경과월_렌탈",
        "fname": "cfs_03_2571",
        "type": "formula",
        "content": """IF M12_RP건수_렌탈_B0M > 0 THEN M12_RP후경과월_렌탈 = 0
                      ELSE M12_RP후경과월_렌탈 = min(M11_RP후경과월_렌탈 + 1, 6)""",
    },
    {
        "columns": ["M12_RP건수_가스_B0M", "M11_RP후경과월_가스"],
        "output": "M12_RP후경과월_가스",
        "fname": "cfs_03_2572",
        "type": "formula",
        "content": """IF M12_RP건수_가스_B0M > 0 THEN M12_RP후경과월_가스 = 0
                      ELSE M12_RP후경과월_가스 = min(M11_RP후경과월_가스 + 1, 6)""",
    },
    {
        "columns": ["M12_RP건수_전기_B0M", "M11_RP후경과월_전기"],
        "output": "M12_RP후경과월_전기",
        "fname": "cfs_03_2573",
        "type": "formula",
        "content": """IF M12_RP건수_전기_B0M > 0 THEN M12_RP후경과월_전기 = 0
                      ELSE M12_RP후경과월_전기 = min(M11_RP후경과월_전기 + 1, 6)""",
    },
    {
        "columns": ["M12_RP건수_보험_B0M", "M11_RP후경과월_보험"],
        "output": "M12_RP후경과월_보험",
        "fname": "cfs_03_2574",
        "type": "formula",
        "content": """IF M12_RP건수_보험_B0M > 0 THEN M12_RP후경과월_보험 = 0
                      ELSE M12_RP후경과월_보험 = min(M11_RP후경과월_보험 + 1, 6)""",
    },
    {
        "columns": ["M12_RP건수_학습비_B0M", "M11_RP후경과월_학습비"],
        "output": "M12_RP후경과월_학습비",
        "fname": "cfs_03_2575",
        "type": "formula",
        "content": """IF M12_RP건수_학습비_B0M > 0 THEN M12_RP후경과월_학습비 = 0
                      ELSE M12_RP후경과월_학습비 = min(M11_RP후경과월_학습비 + 1, 6)""",
    },
    {
        "columns": ["M12_RP건수_유선방송_B0M", "M11_RP후경과월_유선방송"],
        "output": "M12_RP후경과월_유선방송",
        "fname": "cfs_03_2576",
        "type": "formula",
        "content": """IF M12_RP건수_유선방송_B0M > 0 THEN M12_RP후경과월_유선방송 = 0
                      ELSE M12_RP후경과월_유선방송 = min(M11_RP후경과월_유선방송 + 1, 6)""",
    },
    {
        "columns": ["M12_RP건수_건강_B0M", "M11_RP후경과월_건강"],
        "output": "M12_RP후경과월_건강",
        "fname": "cfs_03_2577",
        "type": "formula",
        "content": """IF M12_RP건수_건강_B0M > 0 THEN M12_RP후경과월_건강 = 0
                      ELSE M12_RP후경과월_건강 = min(M11_RP후경과월_건강 + 1, 6)""",
    },
    {
        "columns": ["M12_RP건수_교통_B0M", "M11_RP후경과월_교통"],
        "output": "M12_RP후경과월_교통",
        "fname": "cfs_03_2578",
        "type": "formula",
        "content": """IF M12_RP건수_교통_B0M > 0 THEN M12_RP후경과월_교통 = 0
                      ELSE M12_RP후경과월_교통 = min(M11_RP후경과월_교통 + 1, 6)""",
    },
    {
        "columns": ["M10_승인거절건수_B0M", "M11_승인거절건수_B0M", "M12_승인거절건수_B0M"],
        "output": "M12_승인거절건수_R3M",
        "fname": "cfs_03_2752",
        "type": "formula",
        "content": "M12_승인거절건수_R3M = M10_승인거절건수_B0M + M11_승인거절건수_B0M + M12_승인거절건수_B0M",
    },
    {
        "columns": ["M10_승인거절건수_한도초과_B0M", "M11_승인거절건수_한도초과_B0M", "M12_승인거절건수_한도초과_B0M"],
        "output": "M12_승인거절건수_한도초과_R3M",
        "fname": "cfs_03_2753",
        "type": "formula",
        "content": "M12_승인거절건수_한도초과_R3M = M10_승인거절건수_한도초과_B0M + M11_승인거절건수_한도초과_B0M + M12_승인거절건수_한도초과_B0M",
    },
    {
        "columns": ["M10_승인거절건수_BL_B0M", "M11_승인거절건수_BL_B0M", "M12_승인거절건수_BL_B0M"],
        "output": "M12_승인거절건수_BL_R3M",
        "fname": "cfs_03_2754",
        "type": "formula",
        "content": "M12_승인거절건수_BL_R3M = M10_승인거절건수_BL_B0M + M11_승인거절건수_BL_B0M + M12_승인거절건수_BL_B0M",
    },
    {
        "columns": ["M10_승인거절건수_입력오류_B0M", "M11_승인거절건수_입력오류_B0M", "M12_승인거절건수_입력오류_B0M"],
        "output": "M12_승인거절건수_입력오류_R3M",
        "fname": "cfs_03_2755",
        "type": "formula",
        "content": "M12_승인거절건수_입력오류_R3M = M10_승인거절건수_입력오류_B0M + M11_승인거절건수_입력오류_B0M + M12_승인거절건수_입력오류_B0M",
    },
    {
        "columns": ["M10_승인거절건수_기타_B0M", "M11_승인거절건수_기타_B0M", "M12_승인거절건수_기타_B0M"],
        "output": "M12_승인거절건수_기타_R3M",
        "fname": "cfs_03_2756",
        "type": "formula",
        "content": "M12_승인거절건수_기타_R3M = M10_승인거절건수_기타_B0M + M11_승인거절건수_기타_B0M + M12_승인거절건수_기타_B0M",
    },
    {
        "columns": ["M12_이용건수_신용_R3M", "M09_이용건수_신용_R3M"],
        "output": "M12_이용건수_신용_R6M",
        "fname": "cfs_03_2385",
        "type": "formula",
        "content": "M12_이용건수_신용_R6M = M12_이용건수_신용_R3M + M09_이용건수_신용_R3M",
    },
    {
        "columns": ["M12_이용건수_신판_R3M", "M09_이용건수_신판_R3M"],
        "output": "M12_이용건수_신판_R6M",
        "fname": "cfs_03_2386",
        "type": "formula",
        "content": "M12_이용건수_신판_R6M = M12_이용건수_신판_R3M + M09_이용건수_신판_R3M",
    },
    {
        "columns": ["M12_이용건수_일시불_R3M", "M09_이용건수_일시불_R3M"],
        "output": "M12_이용건수_일시불_R6M",
        "fname": "cfs_03_2387",
        "type": "formula",
        "content": "M12_이용건수_일시불_R6M = M12_이용건수_일시불_R3M + M09_이용건수_일시불_R3M",
    },
    {
        "columns": ["M12_이용건수_할부_R3M", "M09_이용건수_할부_R3M"],
        "output": "M12_이용건수_할부_R6M",
        "fname": "cfs_03_2388",
        "type": "formula",
        "content": "M12_이용건수_할부_R6M = M12_이용건수_할부_R3M + M09_이용건수_할부_R3M",
    },
    {
        "columns": ["M12_이용건수_할부_유이자_R3M", "M09_이용건수_할부_유이자_R3M"],
        "output": "M12_이용건수_할부_유이자_R6M",
        "fname": "cfs_03_2389",
        "type": "formula",
        "content": "M12_이용건수_할부_유이자_R6M = M12_이용건수_할부_유이자_R3M + M09_이용건수_할부_유이자_R3M",
    },
    {
        "columns": ["M12_이용건수_할부_무이자_R3M", "M09_이용건수_할부_무이자_R3M"],
        "output": "M12_이용건수_할부_무이자_R6M",
        "fname": "cfs_03_2390",
        "type": "formula",
        "content": "M12_이용건수_할부_무이자_R6M = M12_이용건수_할부_무이자_R3M + M09_이용건수_할부_무이자_R3M",
    },
    {
        "columns": ["M12_이용건수_부분무이자_R3M", "M09_이용건수_부분무이자_R3M"],
        "output": "M12_이용건수_부분무이자_R6M",
        "fname": "cfs_03_2391",
        "type": "formula",
        "content": "M12_이용건수_부분무이자_R6M = M12_이용건수_부분무이자_R3M + M09_이용건수_부분무이자_R3M",
    },
    {
        "columns": ["M12_이용건수_CA_R3M", "M09_이용건수_CA_R3M"],
        "output": "M12_이용건수_CA_R6M",
        "fname": "cfs_03_2392",
        "type": "formula",
        "content": "M12_이용건수_CA_R6M = M12_이용건수_CA_R3M + M09_이용건수_CA_R3M",
    },
    {
        "columns": ["M12_이용건수_체크_R3M", "M09_이용건수_체크_R3M"],
        "output": "M12_이용건수_체크_R6M",
        "fname": "cfs_03_2393",
        "type": "formula",
        "content": "M12_이용건수_체크_R6M = M12_이용건수_체크_R3M + M09_이용건수_체크_R3M",
    },
    {
        "columns": ["M12_이용건수_카드론_R3M", "M09_이용건수_카드론_R3M"],
        "output": "M12_이용건수_카드론_R6M",
        "fname": "cfs_03_2394",
        "type": "formula",
        "content": "M12_이용건수_카드론_R6M = M12_이용건수_카드론_R3M + M09_이용건수_카드론_R3M",
    },
    {
        "columns": ["M12_이용금액_신용_R3M", "M09_이용금액_신용_R3M"],
        "output": "M12_이용금액_신용_R6M",
        "fname": "cfs_03_2395",
        "type": "formula",
        "content": "M12_이용금액_신용_R6M = M12_이용금액_신용_R3M + M09_이용금액_신용_R3M",
    },
    {
        "columns": ["M12_이용금액_신판_R3M", "M09_이용금액_신판_R3M"],
        "output": "M12_이용금액_신판_R6M",
        "fname": "cfs_03_2396",
        "type": "formula",
        "content": "M12_이용금액_신판_R6M = M12_이용금액_신판_R3M + M09_이용금액_신판_R3M",
    },
    {
        "columns": ["M12_이용금액_일시불_R3M", "M09_이용금액_일시불_R3M"],
        "output": "M12_이용금액_일시불_R6M",
        "fname": "cfs_03_2397",
        "type": "formula",
        "content": "M12_이용금액_일시불_R6M = M12_이용금액_일시불_R3M + M09_이용금액_일시불_R3M",
    },
    {
        "columns": ["M12_이용금액_할부_R3M", "M09_이용금액_할부_R3M"],
        "output": "M12_이용금액_할부_R6M",
        "fname": "cfs_03_2398",
        "type": "formula",
        "content": "M12_이용금액_할부_R6M = M12_이용금액_할부_R3M + M09_이용금액_할부_R3M",
    },
    {
        "columns": ["M12_이용금액_할부_유이자_R3M", "M09_이용금액_할부_유이자_R3M"],
        "output": "M12_이용금액_할부_유이자_R6M",
        "fname": "cfs_03_2399",
        "type": "formula",
        "content": "M12_이용금액_할부_유이자_R6M = M12_이용금액_할부_유이자_R3M + M09_이용금액_할부_유이자_R3M",
    },
    {
        "columns": ["M12_이용금액_할부_무이자_R3M", "M09_이용금액_할부_무이자_R3M"],
        "output": "M12_이용금액_할부_무이자_R6M",
        "fname": "cfs_03_2400",
        "type": "formula",
        "content": "M12_이용금액_할부_무이자_R6M = M12_이용금액_할부_무이자_R3M + M09_이용금액_할부_무이자_R3M",
    },
    {
        "columns": ["M12_이용금액_부분무이자_R3M", "M09_이용금액_부분무이자_R3M"],
        "output": "M12_이용금액_부분무이자_R6M",
        "fname": "cfs_03_2401",
        "type": "formula",
        "content": "M12_이용금액_부분무이자_R6M = M12_이용금액_부분무이자_R3M + M09_이용금액_부분무이자_R3M",
    },
    {
        "columns": ["M12_이용금액_CA_R3M", "M09_이용금액_CA_R3M"],
        "output": "M12_이용금액_CA_R6M",
        "fname": "cfs_03_2402",
        "type": "formula",
        "content": "M12_이용금액_CA_R6M = M12_이용금액_CA_R3M + M09_이용금액_CA_R3M",
    },
    {
        "columns": ["M12_이용금액_체크_R3M", "M09_이용금액_체크_R3M"],
        "output": "M12_이용금액_체크_R6M",
        "fname": "cfs_03_2403",
        "type": "formula",
        "content": "M12_이용금액_체크_R6M = M12_이용금액_체크_R3M + M09_이용금액_체크_R3M",
    },
    {
        "columns": ["M12_이용금액_카드론_R3M", "M09_이용금액_카드론_R3M"],
        "output": "M12_이용금액_카드론_R6M",
        "fname": "cfs_03_2404",
        "type": "formula",
        "content": "M12_이용금액_카드론_R6M = M12_이용금액_카드론_R3M + M09_이용금액_카드론_R3M",
    },
    {
        "columns": ["M12_이용개월수_신용_R3M", "M09_이용개월수_신용_R3M"],
        "output": "M12_이용개월수_신용_R6M",
        "fname": "cfs_03_2405",
        "type": "formula",
        "content": "M12_이용개월수_신용_R6M = M12_이용개월수_신용_R3M + M09_이용개월수_신용_R3M",
    },
    {
        "columns": ["M12_이용개월수_신판_R3M", "M09_이용개월수_신판_R3M"],
        "output": "M12_이용개월수_신판_R6M",
        "fname": "cfs_03_2406",
        "type": "formula",
        "content": "M12_이용개월수_신판_R6M = M12_이용개월수_신판_R3M + M09_이용개월수_신판_R3M",
    },
    {
        "columns": ["M12_이용개월수_일시불_R3M", "M09_이용개월수_일시불_R3M"],
        "output": "M12_이용개월수_일시불_R6M",
        "fname": "cfs_03_2407",
        "type": "formula",
        "content": "M12_이용개월수_일시불_R6M = M12_이용개월수_일시불_R3M + M09_이용개월수_일시불_R3M",
    },
    {
        "columns": ["M12_이용개월수_할부_R3M", "M09_이용개월수_할부_R3M"],
        "output": "M12_이용개월수_할부_R6M",
        "fname": "cfs_03_2408",
        "type": "formula",
        "content": "M12_이용개월수_할부_R6M = M12_이용개월수_할부_R3M + M09_이용개월수_할부_R3M",
    },
    {
        "columns": ["M12_이용개월수_할부_유이자_R3M", "M09_이용개월수_할부_유이자_R3M"],
        "output": "M12_이용개월수_할부_유이자_R6M",
        "fname": "cfs_03_2409",
        "type": "formula",
        "content": "M12_이용개월수_할부_유이자_R6M = M12_이용개월수_할부_유이자_R3M + M09_이용개월수_할부_유이자_R3M",
    },
    {
        "columns": ["M12_이용개월수_할부_무이자_R3M", "M09_이용개월수_할부_무이자_R3M"],
        "output": "M12_이용개월수_할부_무이자_R6M",
        "fname": "cfs_03_2410",
        "type": "formula",
        "content": "M12_이용개월수_할부_무이자_R6M = M12_이용개월수_할부_무이자_R3M + M09_이용개월수_할부_무이자_R3M",
    },
    {
        "columns": ["M12_이용개월수_부분무이자_R3M", "M09_이용개월수_부분무이자_R3M"],
        "output": "M12_이용개월수_부분무이자_R6M",
        "fname": "cfs_03_2411",
        "type": "formula",
        "content": "M12_이용개월수_부분무이자_R6M = M12_이용개월수_부분무이자_R3M + M09_이용개월수_부분무이자_R3M",
    },
    {
        "columns": ["M12_이용개월수_CA_R3M", "M09_이용개월수_CA_R3M"],
        "output": "M12_이용개월수_CA_R6M",
        "fname": "cfs_03_2412",
        "type": "formula",
        "content": "M12_이용개월수_CA_R6M = M12_이용개월수_CA_R3M + M09_이용개월수_CA_R3M",
    },
    {
        "columns": ["M12_이용개월수_체크_R3M", "M09_이용개월수_체크_R3M"],
        "output": "M12_이용개월수_체크_R6M",
        "fname": "cfs_03_2413",
        "type": "formula",
        "content": "M12_이용개월수_체크_R6M = M12_이용개월수_체크_R3M + M09_이용개월수_체크_R3M",
    },
    {
        "columns": ["M12_이용개월수_카드론_R3M", "M09_이용개월수_카드론_R3M"],
        "output": "M12_이용개월수_카드론_R6M",
        "fname": "cfs_03_2414",
        "type": "formula",
        "content": "M12_이용개월수_카드론_R6M = M12_이용개월수_카드론_R3M + M09_이용개월수_카드론_R3M",
    },
    {
        "columns": ["M12_이용건수_신용_B0M", "M11_이용건수_신용_B0M", "M10_이용건수_신용_B0M"],
        "output": "M12_이용건수_신용_R3M",
        "fname": "cfs_03_2415",
        "type": "formula",
        "content": "M12_이용건수_신용_R3M = M12_이용건수_신용_B0M + M11_이용건수_신용_B0M + M10_이용건수_신용_B0M",
    },
    {
        "columns": ["M12_이용건수_신판_B0M", "M11_이용건수_신판_B0M", "M10_이용건수_신판_B0M"],
        "output": "M12_이용건수_신판_R3M",
        "fname": "cfs_03_2416",
        "type": "formula",
        "content": "M12_이용건수_신판_R3M = M12_이용건수_신판_B0M + M11_이용건수_신판_B0M + M10_이용건수_신판_B0M",
    },
    {
        "columns": ["M12_이용건수_일시불_B0M", "M11_이용건수_일시불_B0M", "M10_이용건수_일시불_B0M"],
        "output": "M12_이용건수_일시불_R3M",
        "fname": "cfs_03_2417",
        "type": "formula",
        "content": "M12_이용건수_일시불_R3M = M12_이용건수_일시불_B0M + M11_이용건수_일시불_B0M + M10_이용건수_일시불_B0M",
    },
    {
        "columns": ["M12_이용건수_할부_B0M", "M11_이용건수_할부_B0M", "M10_이용건수_할부_B0M"],
        "output": "M12_이용건수_할부_R3M",
        "fname": "cfs_03_2418",
        "type": "formula",
        "content": "M12_이용건수_할부_R3M = M12_이용건수_할부_B0M + M11_이용건수_할부_B0M + M10_이용건수_할부_B0M",
    },
    {
        "columns": ["M12_이용건수_할부_유이자_B0M", "M11_이용건수_할부_유이자_B0M", "M10_이용건수_할부_유이자_B0M"],
        "output": "M12_이용건수_할부_유이자_R3M",
        "fname": "cfs_03_2419",
        "type": "formula",
        "content": "M12_이용건수_할부_유이자_R3M = M12_이용건수_할부_유이자_B0M + M11_이용건수_할부_유이자_B0M + M10_이용건수_할부_유이자_B0M",
    },
    {
        "columns": ["M12_이용건수_할부_무이자_B0M", "M11_이용건수_할부_무이자_B0M", "M10_이용건수_할부_무이자_B0M"],
        "output": "M12_이용건수_할부_무이자_R3M",
        "fname": "cfs_03_2420",
        "type": "formula",
        "content": "M12_이용건수_할부_무이자_R3M = M12_이용건수_할부_무이자_B0M + M11_이용건수_할부_무이자_B0M + M10_이용건수_할부_무이자_B0M",
    },
    {
        "columns": ["M12_이용건수_부분무이자_B0M", "M11_이용건수_부분무이자_B0M", "M10_이용건수_부분무이자_B0M"],
        "output": "M12_이용건수_부분무이자_R3M",
        "fname": "cfs_03_2421",
        "type": "formula",
        "content": "M12_이용건수_부분무이자_R3M = M12_이용건수_부분무이자_B0M + M11_이용건수_부분무이자_B0M + M10_이용건수_부분무이자_B0M",
    },
    {
        "columns": ["M12_이용건수_CA_B0M", "M11_이용건수_CA_B0M", "M10_이용건수_CA_B0M"],
        "output": "M12_이용건수_CA_R3M",
        "fname": "cfs_03_2422",
        "type": "formula",
        "content": "M12_이용건수_CA_R3M = M12_이용건수_CA_B0M + M11_이용건수_CA_B0M + M10_이용건수_CA_B0M",
    },
    {
        "columns": ["M12_이용건수_체크_B0M", "M11_이용건수_체크_B0M", "M10_이용건수_체크_B0M"],
        "output": "M12_이용건수_체크_R3M",
        "fname": "cfs_03_2423",
        "type": "formula",
        "content": "M12_이용건수_체크_R3M = M12_이용건수_체크_B0M + M11_이용건수_체크_B0M + M10_이용건수_체크_B0M",
    },
    {
        "columns": ["M12_이용건수_카드론_B0M", "M11_이용건수_카드론_B0M", "M10_이용건수_카드론_B0M"],
        "output": "M12_이용건수_카드론_R3M",
        "fname": "cfs_03_2424",
        "type": "formula",
        "content": "M12_이용건수_카드론_R3M = M12_이용건수_카드론_B0M + M11_이용건수_카드론_B0M + M10_이용건수_카드론_B0M",
    },
    {
        "columns": ["M12_이용금액_신용_B0M", "M11_이용금액_신용_B0M", "M10_이용금액_신용_B0M"],
        "output": "M12_이용금액_신용_R3M",
        "fname": "cfs_03_2425",
        "type": "formula",
        "content": "M12_이용금액_신용_R3M = M12_이용금액_신용_B0M + M11_이용금액_신용_B0M + M10_이용금액_신용_B0M",
    },
    {
        "columns": ["M12_이용금액_신판_B0M", "M11_이용금액_신판_B0M", "M10_이용금액_신판_B0M"],
        "output": "M12_이용금액_신판_R3M",
        "fname": "cfs_03_2426",
        "type": "formula",
        "content": "M12_이용금액_신판_R3M = M12_이용금액_신판_B0M + M11_이용금액_신판_B0M + M10_이용금액_신판_B0M",
    },
    {
        "columns": ["M12_이용금액_일시불_B0M", "M11_이용금액_일시불_B0M", "M10_이용금액_일시불_B0M"],
        "output": "M12_이용금액_일시불_R3M",
        "fname": "cfs_03_2427",
        "type": "formula",
        "content": "M12_이용금액_일시불_R3M = M12_이용금액_일시불_B0M + M11_이용금액_일시불_B0M + M10_이용금액_일시불_B0M",
    },
    {
        "columns": ["M12_이용금액_할부_B0M", "M11_이용금액_할부_B0M", "M10_이용금액_할부_B0M"],
        "output": "M12_이용금액_할부_R3M",
        "fname": "cfs_03_2428",
        "type": "formula",
        "content": "M12_이용금액_할부_R3M = M12_이용금액_할부_B0M + M11_이용금액_할부_B0M + M10_이용금액_할부_B0M",
    },
    {
        "columns": ["M12_이용금액_할부_유이자_B0M", "M11_이용금액_할부_유이자_B0M", "M10_이용금액_할부_유이자_B0M"],
        "output": "M12_이용금액_할부_유이자_R3M",
        "fname": "cfs_03_2429",
        "type": "formula",
        "content": "M12_이용금액_할부_유이자_R3M = M12_이용금액_할부_유이자_B0M + M11_이용금액_할부_유이자_B0M + M10_이용금액_할부_유이자_B0M",
    },
    {
        "columns": ["M12_이용금액_할부_무이자_B0M", "M11_이용금액_할부_무이자_B0M", "M10_이용금액_할부_무이자_B0M"],
        "output": "M12_이용금액_할부_무이자_R3M",
        "fname": "cfs_03_2430",
        "type": "formula",
        "content": "M12_이용금액_할부_무이자_R3M = M12_이용금액_할부_무이자_B0M + M11_이용금액_할부_무이자_B0M + M10_이용금액_할부_무이자_B0M",
    },
    {
        "columns": ["M12_이용금액_부분무이자_B0M", "M11_이용금액_부분무이자_B0M", "M10_이용금액_부분무이자_B0M"],
        "output": "M12_이용금액_부분무이자_R3M",
        "fname": "cfs_03_2431",
        "type": "formula",
        "content": "M12_이용금액_부분무이자_R3M = M12_이용금액_부분무이자_B0M + M11_이용금액_부분무이자_B0M + M10_이용금액_부분무이자_B0M",
    },
    {
        "columns": ["M12_이용금액_CA_B0M", "M11_이용금액_CA_B0M", "M10_이용금액_CA_B0M"],
        "output": "M12_이용금액_CA_R3M",
        "fname": "cfs_03_2432",
        "type": "formula",
        "content": "M12_이용금액_CA_R3M = M12_이용금액_CA_B0M + M11_이용금액_CA_B0M + M10_이용금액_CA_B0M",
    },
    {
        "columns": ["M12_이용금액_체크_B0M", "M11_이용금액_체크_B0M", "M10_이용금액_체크_B0M"],
        "output": "M12_이용금액_체크_R3M",
        "fname": "cfs_03_2433",
        "type": "formula",
        "content": "M12_이용금액_체크_R3M = M12_이용금액_체크_B0M + M11_이용금액_체크_B0M + M10_이용금액_체크_B0M",
    },
    {
        "columns": ["M12_이용금액_카드론_B0M", "M11_이용금액_카드론_B0M", "M10_이용금액_카드론_B0M"],
        "output": "M12_이용금액_카드론_R3M",
        "fname": "cfs_03_2434",
        "type": "formula",
        "content": "M12_이용금액_카드론_R3M = M12_이용금액_카드론_B0M + M11_이용금액_카드론_B0M + M10_이용금액_카드론_B0M",
    },
    {
        "columns": ["M12_건수_할부전환_R3M", "M09_건수_할부전환_R3M"],
        "output": "M12_건수_할부전환_R6M",
        "fname": "cfs_03_2449",
        "type": "formula",
        "content": "M12_건수_할부전환_R6M = M12_건수_할부전환_R3M + M09_건수_할부전환_R3M",
    },
    {
        "columns": ["M12_금액_할부전환_R3M", "M09_금액_할부전환_R3M"],
        "output": "M12_금액_할부전환_R6M",
        "fname": "cfs_03_2450",
        "type": "formula",
        "content": "M12_금액_할부전환_R6M = M12_금액_할부전환_R3M + M09_금액_할부전환_R3M",
    },
    {
        "columns": ["M12_최종카드론_대출일자", "M11_최종카드론_대출일자", "M11_최종카드론이용경과월"],
        "output": "M12_최종카드론이용경과월",
        "fname": "cfs_03_2580",
        "type": "formula",
        "content": """IF M12_최종카드론_대출일자 IS NULL
                      THEN M12_최종카드론이용경과월 = 999
                      ELIF M12_최종카드론_대출일자 = M11_최종카드론_대출일자
                      THEN M12_최종카드론이용경과월 = M11_최종카드론이용경과월 + 1
                      ELSE M12_최종카드론이용경과월 = 0""",
    },
    {
        "columns": [
            "M12_신청건수_ATM_CA_B0",
            "M11_신청건수_ATM_CA_B0",
            "M10_신청건수_ATM_CA_B0",
            "M09_신청건수_ATM_CA_B0",
            "M08_신청건수_ATM_CA_B0",
            "M07_신청건수_ATM_CA_B0"
        ],
        "output": "M12_신청건수_ATM_CA_R6M",
        "fname": "cfs_03_2596",
        "type": "formula",
        "content": "M12_신청건수_ATM_CA_R6M = SUM(M07~M12_신청건수_ATM_CA_B0)",
    },
    {
        "columns": [
            "M12_신청건수_ATM_CL_B0",
            "M11_신청건수_ATM_CL_B0",
            "M10_신청건수_ATM_CL_B0",
            "M09_신청건수_ATM_CL_B0",
            "M08_신청건수_ATM_CL_B0",
            "M07_신청건수_ATM_CL_B0"
        ],
        "output": "M12_신청건수_ATM_CL_R6M",
        "fname": "cfs_03_2597",
        "type": "formula",
        "content": "M12_신청건수_ATM_CL_R6M = SUM(M07~M12_신청건수_ATM_CL_B0)",
    },
    {
        "columns": [
            "M12_이용건수_온라인_B0M",
            "M11_이용건수_온라인_B0M",
            "M10_이용건수_온라인_B0M",
            "M09_이용건수_온라인_B0M",
            "M08_이용건수_온라인_B0M",
            "M07_이용건수_온라인_B0M"
        ],
        "output": "M12_이용개월수_온라인_R6M",
        "fname": "cfs_03_2599",
        "type": "formula",
        "content": "M12_이용개월수_온라인_R6M = SUM(1 IF M0X_이용건수_온라인_B0M > 0 ELSE 0)",
    },
    {
        "columns": [
            "M12_이용건수_오프라인_B0M",
            "M11_이용건수_오프라인_B0M",
            "M10_이용건수_오프라인_B0M",
            "M09_이용건수_오프라인_B0M",
            "M08_이용건수_오프라인_B0M",
            "M07_이용건수_오프라인_B0M"
        ],
        "output": "M12_이용개월수_오프라인_R6M",
        "fname": "cfs_03_2600",
        "type": "formula",
        "content": "M12_이용개월수_오프라인_R6M = SUM(1 IF M0X_이용건수_오프라인_B0M > 0 ELSE 0)",
    },
    {
        "columns": ["M12_이용금액_온라인_R3M", "M09_이용금액_온라인_R3M"],
        "output": "M12_이용금액_온라인_R6M",
        "fname": "cfs_03_2601",
        "type": "formula",
        "content": "M12_이용금액_온라인_R6M = M12_이용금액_온라인_R3M + M09_이용금액_온라인_R3M",
    },
    {
        "columns": ["M12_이용금액_오프라인_R3M", "M09_이용금액_오프라인_R3M"],
        "output": "M12_이용금액_오프라인_R6M",
        "fname": "cfs_03_2602",
        "type": "formula",
        "content": "M12_이용금액_오프라인_R6M = M12_이용금액_오프라인_R3M + M09_이용금액_오프라인_R3M",
    },
    {
        "columns": ["M12_이용건수_온라인_R3M", "M09_이용건수_온라인_R3M"],
        "output": "M12_이용건수_온라인_R6M",
        "fname": "cfs_03_2603",
        "type": "formula",
        "content": "M12_이용건수_온라인_R6M = M12_이용건수_온라인_R3M + M09_이용건수_온라인_R3M",
    },
    {
        "columns": ["M12_이용건수_오프라인_R3M", "M09_이용건수_오프라인_R3M"],
        "output": "M12_이용건수_오프라인_R6M",
        "fname": "cfs_03_2604",
        "type": "formula",
        "content": "M12_이용건수_오프라인_R6M = M12_이용건수_오프라인_R3M + M09_이용건수_오프라인_R3M",
    },
    {
        "columns": ["M12_이용금액_온라인_B0M", "M11_이용금액_온라인_B0M", "M10_이용금액_온라인_B0M"],
        "output": "M12_이용금액_온라인_R3M",
        "fname": "cfs_03_2605",
        "type": "formula",
        "content": "M12_이용금액_온라인_R3M = M12_이용금액_온라인_B0M + M11_이용금액_온라인_B0M + M10_이용금액_온라인_B0M",
    },
    {
        "columns": ["M12_이용금액_오프라인_B0M", "M11_이용금액_오프라인_B0M", "M10_이용금액_오프라인_B0M"],
        "output": "M12_이용금액_오프라인_R3M",
        "fname": "cfs_03_2606",
        "type": "formula",
        "content": "M12_이용금액_오프라인_R3M = M12_이용금액_오프라인_B0M + M11_이용금액_오프라인_B0M + M10_이용금액_오프라인_B0M",
    },
    {
        "columns": ["M12_이용건수_온라인_B0M", "M11_이용건수_온라인_B0M", "M10_이용건수_온라인_B0M"],
        "output": "M12_이용건수_온라인_R3M",
        "fname": "cfs_03_2607",
        "type": "formula",
        "content": "M12_이용건수_온라인_R3M = M12_이용건수_온라인_B0M + M11_이용건수_온라인_B0M + M10_이용건수_온라인_B0M",
    },
    {
        "columns": ["M12_이용건수_오프라인_B0M", "M11_이용건수_오프라인_B0M", "M10_이용건수_오프라인_B0M"],
        "output": "M12_이용건수_오프라인_R3M",
        "fname": "cfs_03_2608",
        "type": "formula",
        "content": "M12_이용건수_오프라인_R3M = M12_이용건수_오프라인_B0M + M11_이용건수_오프라인_B0M + M10_이용건수_오프라인_B0M",
    },
    {
        "columns": [
            "M12_이용건수_페이_온라인_B0M",
            "M11_이용건수_페이_온라인_B0M",
            "M10_이용건수_페이_온라인_B0M",
            "M09_이용건수_페이_온라인_B0M",
            "M08_이용건수_페이_온라인_B0M",
            "M07_이용건수_페이_온라인_B0M"
        ],
        "output": "M12_이용개월수_페이_온라인_R6M",
        "fname": "cfs_03_2613",
        "type": "formula",
        "content": "M12_이용개월수_페이_온라인_R6M = SUM(1 IF M0X_이용건수_페이_온라인_B0M > 0 ELSE 0)",
    },
    {
        "columns": [
            "M12_이용건수_페이_오프라인_B0M",
            "M11_이용건수_페이_오프라인_B0M",
            "M10_이용건수_페이_오프라인_B0M",
            "M09_이용건수_페이_오프라인_B0M",
            "M08_이용건수_페이_오프라인_B0M",
            "M07_이용건수_페이_오프라인_B0M"
        ],
        "output": "M12_이용개월수_페이_오프라인_R6M",
        "fname": "cfs_03_2614",
        "type": "formula",
        "content": "M12_이용개월수_페이_오프라인_R6M = SUM(1 IF M0X_이용건수_페이_오프라인_B0M > 0 ELSE 0)",
    },
    {
        "columns": ["M12_이용금액_페이_온라인_R3M", "M09_이용금액_페이_온라인_R3M"],
        "output": "M12_이용금액_페이_온라인_R6M",
        "fname": "cfs_03_2615",
        "type": "formula",
        "content": "M12_이용금액_페이_온라인_R6M = M12_이용금액_페이_온라인_R3M + M09_이용금액_페이_온라인_R3M",
    },
    {
        "columns": ["M12_이용금액_페이_오프라인_R3M", "M09_이용금액_페이_오프라인_R3M"],
        "output": "M12_이용금액_페이_오프라인_R6M",
        "fname": "cfs_03_2616",
        "type": "formula",
        "content": "M12_이용금액_페이_오프라인_R6M = M12_이용금액_페이_오프라인_R3M + M09_이용금액_페이_오프라인_R3M",
    },
    {
        "columns": ["M12_이용건수_페이_온라인_R3M", "M09_이용건수_페이_온라인_R3M"],
        "output": "M12_이용건수_페이_온라인_R6M",
        "fname": "cfs_03_2617",
        "type": "formula",
        "content": "M12_이용건수_페이_온라인_R6M = M12_이용건수_페이_온라인_R3M + M09_이용건수_페이_온라인_R3M",
    },
    {
        "columns": ["M12_이용건수_페이_오프라인_R3M", "M09_이용건수_페이_오프라인_R3M"],
        "output": "M12_이용건수_페이_오프라인_R6M",
        "fname": "cfs_03_2618",
        "type": "formula",
        "content": "M12_이용건수_페이_오프라인_R6M = M12_이용건수_페이_오프라인_R3M + M09_이용건수_페이_오프라인_R3M",
    },
    {
        "columns": ["M12_이용금액_페이_온라인_B0M", "M11_이용금액_페이_온라인_B0M", "M10_이용금액_페이_온라인_B0M"],
        "output": "M12_이용금액_페이_온라인_R3M",
        "fname": "cfs_03_2619",
        "type": "formula",
        "content": "M12_이용금액_페이_온라인_R3M = M12_이용금액_페이_온라인_B0M + M11_이용금액_페이_온라인_B0M + M10_이용금액_페이_온라인_B0M",
    },
    {
        "columns": ["M12_이용금액_페이_오프라인_B0M", "M11_이용금액_페이_오프라인_B0M", "M10_이용금액_페이_오프라인_B0M"],
        "output": "M12_이용금액_페이_오프라인_R3M",
        "fname": "cfs_03_2620",
        "type": "formula",
        "content": "M12_이용금액_페이_오프라인_R3M = M12_이용금액_페이_오프라인_B0M + M11_이용금액_페이_오프라인_B0M + M10_이용금액_페이_오프라인_B0M",
    },
    {
        "columns": ["M12_이용건수_페이_온라인_B0M", "M11_이용건수_페이_온라인_B0M", "M10_이용건수_페이_온라인_B0M"],
        "output": "M12_이용건수_페이_온라인_R3M",
        "fname": "cfs_03_2621",
        "type": "formula",
        "content": "M12_이용건수_페이_온라인_R3M = M12_이용건수_페이_온라인_B0M + M11_이용건수_페이_온라인_B0M + M10_이용건수_페이_온라인_B0M",
    },
    {
        "columns": ["M12_이용건수_페이_오프라인_B0M", "M11_이용건수_페이_오프라인_B0M", "M10_이용건수_페이_오프라인_B0M"],
        "output": "M12_이용건수_페이_오프라인_R3M",
        "fname": "cfs_03_2622",
        "type": "formula",
        "content": "M12_이용건수_페이_오프라인_R3M = M12_이용건수_페이_오프라인_B0M + M11_이용건수_페이_오프라인_B0M + M10_이용건수_페이_오프라인_B0M",
    },
    {
        "columns": [
            "M12_이용건수_간편결제_B0M",
            "M11_이용건수_간편결제_B0M",
            "M10_이용건수_간편결제_B0M",
            "M09_이용건수_간편결제_B0M",
            "M08_이용건수_간편결제_B0M",
            "M07_이용건수_간편결제_B0M"
        ],
        "output": "M12_이용개월수_간편결제_R6M",
        "fname": "cfs_03_2627",
        "type": "formula",
        "content": "M12_이용개월수_간편결제_R6M = SUM(1 IF M0X_이용건수_간편결제_B0M > 0 ELSE 0)",
    },
    {
        "columns": [
            "M12_이용건수_당사페이_B0M",
            "M11_이용건수_당사페이_B0M",
            "M10_이용건수_당사페이_B0M",
            "M09_이용건수_당사페이_B0M",
            "M08_이용건수_당사페이_B0M",
            "M07_이용건수_당사페이_B0M"
        ],
        "output": "M12_이용개월수_당사페이_R6M",
        "fname": "cfs_03_2628",
        "type": "formula",
        "content": "M12_이용개월수_당사페이_R6M = SUM(1 IF M0X_이용건수_당사페이_B0M > 0 ELSE 0)",
    },
    {
        "columns": [
            "M12_이용건수_당사기타_B0M",
            "M11_이용건수_당사기타_B0M",
            "M10_이용건수_당사기타_B0M",
            "M09_이용건수_당사기타_B0M",
            "M08_이용건수_당사기타_B0M",
            "M07_이용건수_당사기타_B0M"
        ],
        "output": "M12_이용개월수_당사기타_R6M",
        "fname": "cfs_03_2629",
        "type": "formula",
        "content": "M12_이용개월수_당사기타_R6M = SUM(1 IF M0X_이용건수_당사기타_B0M > 0 ELSE 0)",
    },
    {
        "columns": [
            "M12_이용건수_A페이_B0M",
            "M11_이용건수_A페이_B0M",
            "M10_이용건수_A페이_B0M",
            "M09_이용건수_A페이_B0M",
            "M08_이용건수_A페이_B0M",
            "M07_이용건수_A페이_B0M"
        ],
        "output": "M12_이용개월수_A페이_R6M",
        "fname": "cfs_03_2630",
        "type": "formula",
        "content": "M12_이용개월수_A페이_R6M = SUM(1 IF M0X_이용건수_A페이_B0M > 0 ELSE 0)",
    },
    {
        "columns": [
            "M12_이용건수_B페이_B0M",
            "M11_이용건수_B페이_B0M",
            "M10_이용건수_B페이_B0M",
            "M09_이용건수_B페이_B0M",
            "M08_이용건수_B페이_B0M",
            "M07_이용건수_B페이_B0M"
        ],
        "output": "M12_이용개월수_B페이_R6M",
        "fname": "cfs_03_2631",
        "type": "formula",
        "content": "M12_이용개월수_B페이_R6M = SUM(1 IF M0X_이용건수_B페이_B0M > 0 ELSE 0)",
    },
    {
        "columns": [
            "M12_이용건수_C페이_B0M",
            "M11_이용건수_C페이_B0M",
            "M10_이용건수_C페이_B0M",
            "M09_이용건수_C페이_B0M",
            "M08_이용건수_C페이_B0M",
            "M07_이용건수_C페이_B0M"
        ],
        "output": "M12_이용개월수_C페이_R6M",
        "fname": "cfs_03_2632",
        "type": "formula",
        "content": "M12_이용개월수_C페이_R6M = SUM(1 IF M0X_이용건수_C페이_B0M > 0 ELSE 0)",
    },
    {
        "columns": [
            "M12_이용건수_D페이_B0M",
            "M11_이용건수_D페이_B0M",
            "M10_이용건수_D페이_B0M",
            "M09_이용건수_D페이_B0M",
            "M08_이용건수_D페이_B0M",
            "M07_이용건수_D페이_B0M"
        ],
        "output": "M12_이용개월수_D페이_R6M",
        "fname": "cfs_03_2633",
        "type": "formula",
        "content": "M12_이용개월수_D페이_R6M = SUM(1 IF M0X_이용건수_D페이_B0M > 0 ELSE 0)",
    },
    {
        "columns": ["M12_이용금액_간편결제_R3M", "M09_이용금액_간편결제_R3M"],
        "output": "M12_이용금액_간편결제_R6M",
        "fname": "cfs_03_2634",
        "type": "formula",
        "content": "M12_이용금액_간편결제_R6M = M12_이용금액_간편결제_R3M + M09_이용금액_간편결제_R3M",
    },
    {
        "columns": ["M12_이용금액_당사페이_R3M", "M09_이용금액_당사페이_R3M"],
        "output": "M12_이용금액_당사페이_R6M",
        "fname": "cfs_03_2635",
        "type": "formula",
        "content": "M12_이용금액_당사페이_R6M = M12_이용금액_당사페이_R3M + M09_이용금액_당사페이_R3M",
    },
    {
        "columns": ["M12_이용금액_당사기타_R3M", "M09_이용금액_당사기타_R3M"],
        "output": "M12_이용금액_당사기타_R6M",
        "fname": "cfs_03_2636",
        "type": "formula",
        "content": "M12_이용금액_당사기타_R6M = M12_이용금액_당사기타_R3M + M09_이용금액_당사기타_R3M",
    },
    {
        "columns": ["M12_이용금액_A페이_R3M", "M09_이용금액_A페이_R3M"],
        "output": "M12_이용금액_A페이_R6M",
        "fname": "cfs_03_2637",
        "type": "formula",
        "content": "M12_이용금액_A페이_R6M = M12_이용금액_A페이_R3M + M09_이용금액_A페이_R3M",
    },
    {
        "columns": ["M12_이용금액_B페이_R3M", "M09_이용금액_B페이_R3M"],
        "output": "M12_이용금액_B페이_R6M",
        "fname": "cfs_03_2638",
        "type": "formula",
        "content": "M12_이용금액_B페이_R6M = M12_이용금액_B페이_R3M + M09_이용금액_B페이_R3M",
    },
    {
        "columns": ["M12_이용금액_C페이_R3M", "M09_이용금액_C페이_R3M"],
        "output": "M12_이용금액_C페이_R6M",
        "fname": "cfs_03_2639",
        "type": "formula",
        "content": "M12_이용금액_C페이_R6M = M12_이용금액_C페이_R3M + M09_이용금액_C페이_R3M",
    },
    {
        "columns": ["M12_이용금액_D페이_R3M", "M09_이용금액_D페이_R3M"],
        "output": "M12_이용금액_D페이_R6M",
        "fname": "cfs_03_2640",
        "type": "formula",
        "content": "M12_이용금액_D페이_R6M = M12_이용금액_D페이_R3M + M09_이용금액_D페이_R3M",
    },
    {
        "columns": ["M12_이용건수_간편결제_R3M", "M09_이용건수_간편결제_R3M"],
        "output": "M12_이용건수_간편결제_R6M",
        "fname": "cfs_03_2641",
        "type": "formula",
        "content": "M12_이용건수_간편결제_R6M = M12_이용건수_간편결제_R3M + M09_이용건수_간편결제_R3M",
    },
    {
        "columns": ["M12_이용건수_당사페이_R3M", "M09_이용건수_당사페이_R3M"],
        "output": "M12_이용건수_당사페이_R6M",
        "fname": "cfs_03_2642",
        "type": "formula",
        "content": "M12_이용건수_당사페이_R6M = M12_이용건수_당사페이_R3M + M09_이용건수_당사페이_R3M",
    },
    {
        "columns": ["M12_이용건수_당사기타_R3M", "M09_이용건수_당사기타_R3M"],
        "output": "M12_이용건수_당사기타_R6M",
        "fname": "cfs_03_2643",
        "type": "formula",
        "content": "M12_이용건수_당사기타_R6M = M12_이용건수_당사기타_R3M + M09_이용건수_당사기타_R3M",
    },
    {
        "columns": ["M12_이용건수_A페이_R3M", "M09_이용건수_A페이_R3M"],
        "output": "M12_이용건수_A페이_R6M",
        "fname": "cfs_03_2644",
        "type": "formula",
        "content": "M12_이용건수_A페이_R6M = M12_이용건수_A페이_R3M + M09_이용건수_A페이_R3M",
    },
    {
        "columns": ["M12_이용건수_B페이_R3M", "M09_이용건수_B페이_R3M"],
        "output": "M12_이용건수_B페이_R6M",
        "fname": "cfs_03_2645",
        "type": "formula",
        "content": "M12_이용건수_B페이_R6M = M12_이용건수_B페이_R3M + M09_이용건수_B페이_R3M",
    },
    {
        "columns": ["M12_이용건수_C페이_R3M", "M09_이용건수_C페이_R3M"],
        "output": "M12_이용건수_C페이_R6M",
        "fname": "cfs_03_2646",
        "type": "formula",
        "content": "M12_이용건수_C페이_R6M = M12_이용건수_C페이_R3M + M09_이용건수_C페이_R3M",
    },
    {
        "columns": ["M12_이용건수_D페이_R3M", "M09_이용건수_D페이_R3M"],
        "output": "M12_이용건수_D페이_R6M",
        "fname": "cfs_03_2647",
        "type": "formula",
        "content": "M12_이용건수_D페이_R6M = M12_이용건수_D페이_R3M + M09_이용건수_D페이_R3M",
    },
    {
        "columns": ["M12_이용금액_간편결제_B0M", "M11_이용금액_간편결제_B0M", "M10_이용금액_간편결제_B0M"],
        "output": "M12_이용금액_간편결제_R3M",
        "fname": "cfs_03_2648",
        "type": "formula",
        "content": "M12_이용금액_간편결제_R3M = M12_이용금액_간편결제_B0M + M11_이용금액_간편결제_B0M + M10_이용금액_간편결제_B0M",
    },
    {
        "columns": ["M12_이용금액_당사페이_B0M", "M11_이용금액_당사페이_B0M", "M10_이용금액_당사페이_B0M"],
        "output": "M12_이용금액_당사페이_R3M",
        "fname": "cfs_03_2649",
        "type": "formula",
        "content": "M12_이용금액_당사페이_R3M = M12_이용금액_당사페이_B0M + M11_이용금액_당사페이_B0M + M10_이용금액_당사페이_B0M",
    },
    {
        "columns": ["M12_이용금액_당사기타_B0M", "M11_이용금액_당사기타_B0M", "M10_이용금액_당사기타_B0M"],
        "output": "M12_이용금액_당사기타_R3M",
        "fname": "cfs_03_2650",
        "type": "formula",
        "content": "M12_이용금액_당사기타_R3M = M12_이용금액_당사기타_B0M + M11_이용금액_당사기타_B0M + M10_이용금액_당사기타_B0M",
    },
    {
        "columns": ["M12_이용금액_A페이_B0M", "M11_이용금액_A페이_B0M", "M10_이용금액_A페이_B0M"],
        "output": "M12_이용금액_A페이_R3M",
        "fname": "cfs_03_2651",
        "type": "formula",
        "content": "M12_이용금액_A페이_R3M = M12_이용금액_A페이_B0M + M11_이용금액_A페이_B0M + M10_이용금액_A페이_B0M",
    },
    {
        "columns": ["M12_이용금액_B페이_B0M", "M11_이용금액_B페이_B0M", "M10_이용금액_B페이_B0M"],
        "output": "M12_이용금액_B페이_R3M",
        "fname": "cfs_03_2652",
        "type": "formula",
        "content": "M12_이용금액_B페이_R3M = M12_이용금액_B페이_B0M + M11_이용금액_B페이_B0M + M10_이용금액_B페이_B0M",
    },
    {
        "columns": ["M12_이용금액_C페이_B0M", "M11_이용금액_C페이_B0M", "M10_이용금액_C페이_B0M"],
        "output": "M12_이용금액_C페이_R3M",
        "fname": "cfs_03_2653",
        "type": "formula",
        "content": "M12_이용금액_C페이_R3M = M12_이용금액_C페이_B0M + M11_이용금액_C페이_B0M + M10_이용금액_C페이_B0M",
    },
    {
        "columns": ["M12_이용금액_D페이_B0M", "M11_이용금액_D페이_B0M", "M10_이용금액_D페이_B0M"],
        "output": "M12_이용금액_D페이_R3M",
        "fname": "cfs_03_2654",
        "type": "formula",
        "content": "M12_이용금액_D페이_R3M = M12_이용금액_D페이_B0M + M11_이용금액_D페이_B0M + M10_이용금액_D페이_B0M",
    },
    {
        "columns": ["M12_이용건수_간편결제_B0M", "M11_이용건수_간편결제_B0M", "M10_이용건수_간편결제_B0M"],
        "output": "M12_이용건수_간편결제_R3M",
        "fname": "cfs_03_2655",
        "type": "formula",
        "content": "M12_이용건수_간편결제_R3M = M12_이용건수_간편결제_B0M + M11_이용건수_간편결제_B0M + M10_이용건수_간편결제_B0M",
    },
    {
        "columns": ["M12_이용건수_당사페이_B0M", "M11_이용건수_당사페이_B0M", "M10_이용건수_당사페이_B0M"],
        "output": "M12_이용건수_당사페이_R3M",
        "fname": "cfs_03_2656",
        "type": "formula",
        "content": "M12_이용건수_당사페이_R3M = M12_이용건수_당사페이_B0M + M11_이용건수_당사페이_B0M + M10_이용건수_당사페이_B0M",
    },
    {
        "columns": ["M12_이용건수_당사기타_B0M", "M11_이용건수_당사기타_B0M", "M10_이용건수_당사기타_B0M"],
        "output": "M12_이용건수_당사기타_R3M",
        "fname": "cfs_03_2657",
        "type": "formula",
        "content": "M12_이용건수_당사기타_R3M = M12_이용건수_당사기타_B0M + M11_이용건수_당사기타_B0M + M10_이용건수_당사기타_B0M",
    },
    {
        "columns": ["M12_이용건수_A페이_B0M", "M11_이용건수_A페이_B0M", "M10_이용건수_A페이_B0M"],
        "output": "M12_이용건수_A페이_R3M",
        "fname": "cfs_03_2658",
        "type": "formula",
        "content": "M12_이용건수_A페이_R3M = M12_이용건수_A페이_B0M + M11_이용건수_A페이_B0M + M10_이용건수_A페이_B0M",
    },
    {
        "columns": ["M12_이용건수_B페이_B0M", "M11_이용건수_B페이_B0M", "M10_이용건수_B페이_B0M"],
        "output": "M12_이용건수_B페이_R3M",
        "fname": "cfs_03_2659",
        "type": "formula",
        "content": "M12_이용건수_B페이_R3M = M12_이용건수_B페이_B0M + M11_이용건수_B페이_B0M + M10_이용건수_B페이_B0M",
    },
    {
        "columns": ["M12_이용건수_C페이_B0M", "M11_이용건수_C페이_B0M", "M10_이용건수_C페이_B0M"],
        "output": "M12_이용건수_C페이_R3M",
        "fname": "cfs_03_2660",
        "type": "formula",
        "content": "M12_이용건수_C페이_R3M = M12_이용건수_C페이_B0M + M11_이용건수_C페이_B0M + M10_이용건수_C페이_B0M",
    },
    {
        "columns": ["M12_이용건수_D페이_B0M", "M11_이용건수_D페이_B0M", "M10_이용건수_D페이_B0M"],
        "output": "M12_이용건수_D페이_R3M",
        "fname": "cfs_03_2661",
        "type": "formula",
        "content": "M12_이용건수_D페이_R3M = M12_이용건수_D페이_B0M + M11_이용건수_D페이_B0M + M10_이용건수_D페이_B0M",
    },
    {
        "columns": [
            "M12_이용건수_선결제_B0M",
            "M11_이용건수_선결제_B0M",
            "M10_이용건수_선결제_B0M",
            "M09_이용건수_선결제_B0M",
            "M08_이용건수_선결제_B0M",
            "M07_이용건수_선결제_B0M"
        ],
        "output": "M12_이용개월수_선결제_R6M",
        "fname": "cfs_03_2676",
        "type": "formula",
        "content": "M12_이용개월수_선결제_R6M = SUM(1 IF M0X_이용건수_선결제_B0M > 0 ELSE 0)",
    },
    {
        "columns": ["M12_이용횟수_선결제_R3M", "M09_이용횟수_선결제_R3M"],
        "output": "M12_이용횟수_선결제_R6M",
        "fname": "cfs_03_2677",
        "type": "formula",
        "content": "M12_이용횟수_선결제_R6M = M12_이용횟수_선결제_R3M + M09_이용횟수_선결제_R3M",
    },
    {
        "columns": ["M12_이용금액_선결제_R3M", "M09_이용금액_선결제_R3M"],
        "output": "M12_이용금액_선결제_R6M",
        "fname": "cfs_03_2678",
        "type": "formula",
        "content": "M12_이용금액_선결제_R6M = M12_이용금액_선결제_R3M + M09_이용금액_선결제_R3M",
    },
    {
        "columns": ["M12_이용건수_선결제_R3M", "M09_이용건수_선결제_R3M"],
        "output": "M12_이용건수_선결제_R6M",
        "fname": "cfs_03_2679",
        "type": "formula",
        "content": "M12_이용건수_선결제_R6M = M12_이용건수_선결제_R3M + M09_이용건수_선결제_R3M",
    },
    {
        "columns": ["M12_이용횟수_선결제_B0M", "M11_이용횟수_선결제_B0M", "M10_이용횟수_선결제_B0M"],
        "output": "M12_이용횟수_선결제_R3M",
        "fname": "cfs_03_2680",
        "type": "formula",
        "content": "M12_이용횟수_선결제_R3M = M12_이용횟수_선결제_B0M + M11_이용횟수_선결제_B0M + M10_이용횟수_선결제_B0M",
    },
    {
        "columns": ["M12_이용금액_선결제_B0M", "M11_이용금액_선결제_B0M", "M10_이용금액_선결제_B0M"],
        "output": "M12_이용금액_선결제_R3M",
        "fname": "cfs_03_2681",
        "type": "formula",
        "content": "M12_이용금액_선결제_R3M = M12_이용금액_선결제_B0M + M11_이용금액_선결제_B0M + M10_이용금액_선결제_B0M",
    },
    {
        "columns": ["M12_이용건수_선결제_B0M", "M11_이용건수_선결제_B0M", "M10_이용건수_선결제_B0M"],
        "output": "M12_이용건수_선결제_R3M",
        "fname": "cfs_03_2682",
        "type": "formula",
        "content": "M12_이용건수_선결제_R3M = M12_이용건수_선결제_B0M + M11_이용건수_선결제_B0M + M10_이용건수_선결제_B0M",
    },
    {
        "columns": ["M11_가맹점매출금액_B1M"],
        "output": "M12_가맹점매출금액_B2M",
        "fname": "cfs_03_2687",
        "type": "formula",
        "content": "M12_가맹점매출금액_B2M = M11_가맹점매출금액_B1M",
    },
    {
        "columns": ["M10_정상청구원금_B0M"],
        "output": "M12_정상청구원금_B2M",
        "fname": "cfs_03_2689",
        "type": "formula",
        "content": "M12_정상청구원금_B2M = M10_정상청구원금_B0M",
    },
    {
        "columns": ["M07_정상청구원금_B0M"],
        "output": "M12_정상청구원금_B5M",
        "fname": "cfs_03_2690",
        "type": "formula",
        "content": "M12_정상청구원금_B5M = M07_정상청구원금_B0M",
    },
    {
        "columns": ["M10_선입금원금_B0M"],
        "output": "M12_선입금원금_B2M",
        "fname": "cfs_03_2692",
        "type": "formula",
        "content": "M12_선입금원금_B2M = M10_선입금원금_B0M",
    },
    {
        "columns": ["M07_선입금원금_B0M"],
        "output": "M12_선입금원금_B5M",
        "fname": "cfs_03_2693",
        "type": "formula",
        "content": "M12_선입금원금_B5M = M07_선입금원금_B0M",
    },
    {
        "columns": ["M10_정상입금원금_B0M"],
        "output": "M12_정상입금원금_B2M",
        "fname": "cfs_03_2695",
        "type": "formula",
        "content": "M12_정상입금원금_B2M = M10_정상입금원금_B0M",
    },
    {
        "columns": ["M07_정상입금원금_B0M"],
        "output": "M12_정상입금원금_B5M",
        "fname": "cfs_03_2696",
        "type": "formula",
        "content": "M12_정상입금원금_B5M = M07_정상입금원금_B0M",
    },
    {
        "columns": ["M10_연체입금원금_B0M"],
        "output": "M12_연체입금원금_B2M",
        "fname": "cfs_03_2698",
        "type": "formula",
        "content": "M12_연체입금원금_B2M = M10_연체입금원금_B0M",
    },
    {
        "columns": ["M07_연체입금원금_B0M"],
        "output": "M12_연체입금원금_B5M",
        "fname": "cfs_03_2699",
        "type": "formula",
        "content": "M12_연체입금원금_B5M = M07_연체입금원금_B0M",
    },
    {
        "columns": ["M12_이용개월수_전체_R3M", "M09_이용개월수_전체_R3M"],
        "output": "M12_이용개월수_전체_R6M",
        "fname": "cfs_03_2700",
        "type": "formula",
        "content": "M12_이용개월수_전체_R6M = M12_이용개월수_전체_R3M + M09_이용개월수_전체_R3M",
    },
    {
        "columns": ["M12_이용횟수_연체_R3M", "M09_이용횟수_연체_R3M"],
        "output": "M12_이용횟수_연체_R6M",
        "fname": "cfs_03_2704",
        "type": "formula",
        "content": "M12_이용횟수_연체_R6M = M12_이용횟수_연체_R3M + M09_이용횟수_연체_R3M",
    },
    {
        "columns": ["M12_이용금액_연체_R3M", "M09_이용금액_연체_R3M"],
        "output": "M12_이용금액_연체_R6M",
        "fname": "cfs_03_2705",
        "type": "formula",
        "content": "M12_이용금액_연체_R6M = M12_이용금액_연체_R3M + M09_이용금액_연체_R3M",
    },
    {
        "columns": ["M12_이용횟수_연체_B0M", "M11_이용횟수_연체_B0M", "M10_이용횟수_연체_B0M"],
        "output": "M12_이용횟수_연체_R3M",
        "fname": "cfs_03_2706",
        "type": "formula",
        "content": "M12_이용횟수_연체_R3M = M12_이용횟수_연체_B0M + M11_이용횟수_연체_B0M + M10_이용횟수_연체_B0M",
    },
    {
        "columns": ["M12_이용금액_연체_B0M", "M11_이용금액_연체_B0M", "M10_이용금액_연체_B0M"],
        "output": "M12_이용금액_연체_R3M",
        "fname": "cfs_03_2707",
        "type": "formula",
        "content": "M12_이용금액_연체_R3M = M12_이용금액_연체_B0M + M11_이용금액_연체_B0M + M10_이용금액_연체_B0M",
    },
    {
        "columns": ["M11_RP건수_B0M", "M12_RP건수_B0M"],
        "output": "M12_증감_RP건수_전월",
        "fname": "cfs_03_2720",
        "type": "formula",
        "content": "M12_증감_RP건수_전월 = M11_RP건수_B0M - M12_RP건수_B0M",
    },
    {
        "columns": ["M11_RP건수_통신_B0M", "M12_RP건수_통신_B0M"],
        "output": "M12_증감_RP건수_통신_전월",
        "fname": "cfs_03_2722",
        "type": "formula",
        "content": "M12_증감_RP건수_통신_전월 = M11_RP건수_통신_B0M - M12_RP건수_통신_B0M",
    },
    {
        "columns": ["M11_RP건수_아파트_B0M", "M12_RP건수_아파트_B0M"],
        "output": "M12_증감_RP건수_아파트_전월",
        "fname": "cfs_03_2723",
        "type": "formula",
        "content": "M12_증감_RP건수_아파트_전월 = M11_RP건수_아파트_B0M - M12_RP건수_아파트_B0M",
    },
    {
        "columns": ["M11_RP건수_제휴사서비스직접판매_B0M", "M12_RP건수_제휴사서비스직접판매_B0M"],
        "output": "M12_증감_RP건수_제휴사서비스직접판매_전월",
        "fname": "cfs_03_2724",
        "type": "formula",
        "content": "M12_증감_RP건수_제휴사서비스직접판매_전월 = M11_RP건수_제휴사서비스직접판매_B0M - M12_RP건수_제휴사서비스직접판매_B0M",
    },
    {
        "columns": ["M11_RP건수_렌탈_B0M", "M12_RP건수_렌탈_B0M"],
        "output": "M12_증감_RP건수_렌탈_전월",
        "fname": "cfs_03_2725",
        "type": "formula",
        "content": "M12_증감_RP건수_렌탈_전월 = M11_RP건수_렌탈_B0M - M12_RP건수_렌탈_B0M",
    },
    {
        "columns": ["M11_RP건수_가스_B0M", "M12_RP건수_가스_B0M"],
        "output": "M12_증감_RP건수_가스_전월",
        "fname": "cfs_03_2726",
        "type": "formula",
        "content": "M12_증감_RP건수_가스_전월 = M11_RP건수_가스_B0M - M12_RP건수_가스_B0M",
    },
    {
        "columns": ["M11_RP건수_전기_B0M", "M12_RP건수_전기_B0M"],
        "output": "M12_증감_RP건수_전기_전월",
        "fname": "cfs_03_2727",
        "type": "formula",
        "content": "M12_증감_RP건수_전기_전월 = M11_RP건수_전기_B0M - M12_RP건수_전기_B0M",
    },
    {
        "columns": ["M11_RP건수_보험_B0M", "M12_RP건수_보험_B0M"],
        "output": "M12_증감_RP건수_보험_전월",
        "fname": "cfs_03_2728",
        "type": "formula",
        "content": "M12_증감_RP건수_보험_전월 = M11_RP건수_보험_B0M - M12_RP건수_보험_B0M",
    },
    {
        "columns": ["M11_RP건수_학습비_B0M", "M12_RP건수_학습비_B0M"],
        "output": "M12_증감_RP건수_학습비_전월",
        "fname": "cfs_03_2729",
        "type": "formula",
        "content": "M12_증감_RP건수_학습비_전월 = M11_RP건수_학습비_B0M - M12_RP건수_학습비_B0M",
    },
    {
        "columns": ["M11_RP건수_유선방송_B0M", "M12_RP건수_유선방송_B0M"],
        "output": "M12_증감_RP건수_유선방송_전월",
        "fname": "cfs_03_2730",
        "type": "formula",
        "content": "M12_증감_RP건수_유선방송_전월 = M11_RP건수_유선방송_B0M - M12_RP건수_유선방송_B0M",
    },
    {
        "columns": ["M11_RP건수_건강_B0M", "M12_RP건수_건강_B0M"],
        "output": "M12_증감_RP건수_건강_전월",
        "fname": "cfs_03_2731",
        "type": "formula",
        "content": "M12_증감_RP건수_건강_전월 = M11_RP건수_건강_B0M - M12_RP건수_건강_B0M",
    },
    {
        "columns": ["M11_RP건수_교통_B0M", "M12_RP건수_교통_B0M"],
        "output": "M12_증감_RP건수_교통_전월",
        "fname": "cfs_03_2732",
        "type": "formula",
        "content": "M12_증감_RP건수_교통_전월 = M11_RP건수_교통_B0M - M12_RP건수_교통_B0M",
    },
]

# --------- constraint/formula 함수 정의 ---------
# cc: check constraint
# cf: check formula


# 01_Constraints
@constraint_udf
def ccs_01_0115(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_탈회횟수_누적 <= M08_탈회횟수_누적
    """
    c1, c2 = df["M07_탈회횟수_누적"], df["M08_탈회횟수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_01_0117(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_탈회횟수_발급6개월이내 <= M08_탈회횟수_발급6개월이내
    """
    c1, c2 = df["M07_탈회횟수_발급6개월이내"], df["M08_탈회횟수_발급6개월이내"]
    return c1 <= c2


@constraint_udf
def ccs_01_0118(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_탈회횟수_발급1년이내 <= M08_탈회횟수_발급1년이내
    """
    c1, c2 = df["M07_탈회횟수_발급1년이내"], df["M08_탈회횟수_발급1년이내"]
    return c1 <= c2


@constraint_udf
def ccs_01_0154(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_최종카드발급일자 <= M08_최종카드발급일자
    """
    dd = df[["M07_최종카드발급일자", "M08_최종카드발급일자"]]
    res = dd.apply(lambda x: x[0] <= x[1] if (not isNaN(x[0])) & (not isNaN(x[1])) else True, axis=1)
    return res


@constraint_udf
def ccs_01_0204(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_탈회횟수_누적 <= M09_탈회횟수_누적
    """
    c1, c2 = df["M08_탈회횟수_누적"], df["M09_탈회횟수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_01_0206(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_탈회횟수_발급6개월이내 <= M09_탈회횟수_발급6개월이내
    """
    c1, c2 = df["M08_탈회횟수_발급6개월이내"], df["M09_탈회횟수_발급6개월이내"]
    return c1 <= c2


@constraint_udf
def ccs_01_0207(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_탈회횟수_발급1년이내 <= M09_탈회횟수_발급1년이내
    """
    c1, c2 = df["M08_탈회횟수_발급1년이내"], df["M09_탈회횟수_발급1년이내"]
    return c1 <= c2


@constraint_udf
def ccs_01_0243(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_최종카드발급일자 <= M09_최종카드발급일자
    """
    dd = df[["M08_최종카드발급일자", "M09_최종카드발급일자"]]
    res = dd.apply(lambda x: x[0] <= x[1] if (not isNaN(x[0])) & (not isNaN(x[1])) else True, axis=1)
    return res


@constraint_udf
def ccs_01_0293(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_탈회횟수_누적 <= M10_탈회횟수_누적
    """
    c1, c2 = df["M09_탈회횟수_누적"], df["M10_탈회횟수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_01_0295(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_탈회횟수_발급6개월이내 <= M10_탈회횟수_발급6개월이내
    """
    c1, c2 = df["M09_탈회횟수_발급6개월이내"], df["M10_탈회횟수_발급6개월이내"]
    return c1 <= c2


@constraint_udf
def ccs_01_0296(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_탈회횟수_발급1년이내 <= M10_탈회횟수_발급1년이내
    """
    c1, c2 = df["M09_탈회횟수_발급1년이내"], df["M10_탈회횟수_발급1년이내"]
    return c1 <= c2


@constraint_udf
def ccs_01_0332(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_최종카드발급일자 <= M10_최종카드발급일자
    """
    dd = df[["M09_최종카드발급일자", "M10_최종카드발급일자"]]
    res = dd.apply(lambda x: x[0] <= x[1] if (not isNaN(x[0])) & (not isNaN(x[1])) else True, axis=1)
    return res


@constraint_udf
def ccs_01_0382(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_탈회횟수_누적 <= M11_탈회횟수_누적
    """
    c1, c2 = df["M10_탈회횟수_누적"], df["M11_탈회횟수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_01_0384(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_탈회횟수_발급6개월이내 <= M11_탈회횟수_발급6개월이내
    """
    c1, c2 = df["M10_탈회횟수_발급6개월이내"], df["M11_탈회횟수_발급6개월이내"]
    return c1 <= c2


@constraint_udf
def ccs_01_0385(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_탈회횟수_발급1년이내 <= M11_탈회횟수_발급1년이내
    """
    c1, c2 = df["M10_탈회횟수_발급1년이내"], df["M11_탈회횟수_발급1년이내"]
    return c1 <= c2


@constraint_udf
def ccs_01_0421(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_최종카드발급일자 <= M11_최종카드발급일자
    """
    dd = df[["M10_최종카드발급일자", "M11_최종카드발급일자"]]
    res = dd.apply(lambda x: x[0] <= x[1] if (not isNaN(x[0])) & (not isNaN(x[1])) else True, axis=1)
    return res


@constraint_udf
def ccs_01_0471(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_탈회횟수_누적 <= M12_탈회횟수_누적
    """
    c1, c2 = df["M11_탈회횟수_누적"], df["M12_탈회횟수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_01_0473(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_탈회횟수_발급6개월이내 <= M12_탈회횟수_발급6개월이내
    """
    c1, c2 = df["M11_탈회횟수_발급6개월이내"], df["M12_탈회횟수_발급6개월이내"]
    return c1 <= c2


@constraint_udf
def ccs_01_0474(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_탈회횟수_발급1년이내 <= M12_탈회횟수_발급1년이내
    """
    c1, c2 = df["M11_탈회횟수_발급1년이내"], df["M12_탈회횟수_발급1년이내"]
    return c1 <= c2


@constraint_udf
def ccs_01_0510(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_최종카드발급일자 <= M12_최종카드발급일자
    """
    dd = df[["M11_최종카드발급일자", "M12_최종카드발급일자"]]
    res = dd.apply(lambda x: x[0] <= x[1] if (not isNaN(x[0])) & (not isNaN(x[1])) else True, axis=1)
    return res


# 01_M08
@constraint_udf
def cfs_01_0092(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_남녀구분코드 = M07_남녀구분코드
    """
    res = df["M07_남녀구분코드"]

    c = df["M08_남녀구분코드"]
    return c == res


@constraint_udf
def cfs_01_0093(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_연령 = M07_연령
    """
    res = df["M07_연령"]

    c = df["M08_연령"]
    return c == res


@constraint_udf
def cfs_01_0094(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_VIP등급코드 = M07_VIP등급코드
    """
    res = df["M07_VIP등급코드"]

    c = df["M08_VIP등급코드"]
    return c == res


@constraint_udf
def cfs_01_0095(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_최상위카드등급코드 = M07_최상위카드등급코드
    """
    res = df["M07_최상위카드등급코드"]

    c = df["M08_최상위카드등급코드"]
    return c == res


@constraint_udf
def cfs_01_0102(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_입회일자_신용 = M07_입회일자_신용
    """
    res = df["M07_입회일자_신용"]

    c = df["M08_입회일자_신용"]
    return c == res


@constraint_udf
def cfs_01_0103(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_입회경과개월수_신용 = M07_입회경과개월수_신용 + 1
    """
    dd = df[["M07_입회경과개월수_신용"]]
    res = dd.apply(lambda x: x[0] + 1, axis=1)

    c = df["M08_입회경과개월수_신용"]
    return c == res


# @constraint_udf
# def cfs_01_0101(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M08_유치경로코드_신용 = M07_유치경로코드_신용
#     """
#     res = df["M07_유치경로코드_신용"]
#
#    c = df["M08_유치경로코드_신용"]
#    return c == res


# @constraint_udf
# def cfs_01_0102(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M08_자사카드자격코드 = M07_자사카드자격코드
#     """
#     res = df["M07_자사카드자격코드"]
#
#    c = df["M08_자사카드자격코드"]
#    return c == res


@constraint_udf
def cfs_01_0116(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF (M08_탈회횟수_누적 > 0) & (M07_탈회횟수_누적 == M08_탈회횟수_누적)
        THEN M08_최종탈회후경과월 = M07_최종탈회후경과월 + 1
        ELSE M08_최종탈회후경과월 = 0
    """
    dd = df[["M07_탈회횟수_누적", "M08_탈회횟수_누적", "M07_최종탈회후경과월"]]
    res = dd.apply(lambda x: x[2] + 1 if (x[0] > 0) & (x[0] == x[1]) else 0, axis=1)

    c = df["M08_최종탈회후경과월"]
    return c == res


# 01_M09
@constraint_udf
def cfs_01_0181(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_남녀구분코드 = M07_남녀구분코드
    """
    res = df["M07_남녀구분코드"]

    c = df["M08_남녀구분코드"]
    return c == res


@constraint_udf
def cfs_01_0182(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_연령 = M07_연령
    """
    res = df["M07_연령"]

    c = df["M09_연령"]
    return c == res


@constraint_udf
def cfs_01_0183(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_VIP등급코드 = M07_VIP등급코드
    """
    res = df["M07_VIP등급코드"]

    c = df["M09_VIP등급코드"]
    return c == res


@constraint_udf
def cfs_01_0184(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_최상위카드등급코드 = M07_최상위카드등급코드
    """
    res = df["M07_최상위카드등급코드"]

    c = df["M09_최상위카드등급코드"]
    return c == res


@constraint_udf
def cfs_01_0191(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_입회일자_신용 = M07_입회일자_신용
    """
    res = df["M07_입회일자_신용"]

    c = df["M09_입회일자_신용"]
    return c == res


@constraint_udf
def cfs_01_0192(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_입회경과개월수_신용 = M07_입회경과개월수_신용 + 2
    """
    dd = df[["M07_입회경과개월수_신용"]]
    res = dd.apply(lambda x: x[0] + 2, axis=1)

    c = df["M09_입회경과개월수_신용"]
    return c == res


# @constraint_udf
# def cfs_01_0187(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M09_유치경로코드_신용 = M07_유치경로코드_신용
#     """
#     res = df["M07_유치경로코드_신용"]
#
#    c = df["M09_유치경로코드_신용"]
#    return c == res


# @constraint_udf
# def cfs_01_0188(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M09_자사카드자격코드 = M07_자사카드자격코드
#     """
#     res = df["M07_자사카드자격코드"]
#
#    c = df["M09_자사카드자격코드"]
#    return c == res


@constraint_udf
def cfs_01_0205(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF (M09_탈회횟수_누적 > 0) & (M08_탈회횟수_누적 == M09_탈회횟수_누적)
        THEN M09_최종탈회후경과월 = M08_최종탈회후경과월 + 1
        ELSE M09_최종탈회후경과월 = 0
    """
    dd = df[["M08_탈회횟수_누적", "M09_탈회횟수_누적", "M08_최종탈회후경과월"]]
    res = dd.apply(lambda x: x[2] + 1 if (x[0] > 0) & (x[0] == x[1]) else 0, axis=1)

    c = df["M09_최종탈회후경과월"]
    return c == res


# 01_M10
@constraint_udf
def cfs_01_0270(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_남녀구분코드 = M07_남녀구분코드
    """
    res = df["M07_남녀구분코드"]

    c = df["M10_남녀구분코드"]
    return c == res


@constraint_udf
def cfs_01_0271(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_연령 = M07_연령
    """
    res = df["M07_연령"]

    c = df["M10_연령"]
    return c == res


@constraint_udf
def cfs_01_0272(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_VIP등급코드 = M07_VIP등급코드
    """
    res = df["M07_VIP등급코드"]

    c = df["M10_VIP등급코드"]
    return c == res


@constraint_udf
def cfs_01_0273(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_최상위카드등급코드 = M07_최상위카드등급코드
    """
    res = df["M07_최상위카드등급코드"]

    c = df["M10_최상위카드등급코드"]
    return c == res


@constraint_udf
def cfs_01_0280(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_입회일자_신용 = M07_입회일자_신용
    """
    res = df["M07_입회일자_신용"]

    c = df["M10_입회일자_신용"]
    return c == res


@constraint_udf
def cfs_01_0281(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_입회경과개월수_신용 = M07_입회경과개월수_신용 + 3
    """
    dd = df[["M07_입회경과개월수_신용"]]
    res = dd.apply(lambda x: x[0] + 3, axis=1)

    c = df["M10_입회경과개월수_신용"]
    return c == res


# @constraint_udf
# def cfs_01_0273(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M10_유치경로코드_신용 = M07_유치경로코드_신용
#     """
#     res = df["M07_유치경로코드_신용"]
#
#    c = df["M10_유치경로코드_신용"]
#    return c == res


# @constraint_udf
# def cfs_01_0274(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M10_자사카드자격코드 = M07_자사카드자격코드
#     """
#     res = df["M07_자사카드자격코드"]
#
#    c = df["M10_자사카드자격코드"]
#    return c == res


@constraint_udf
def cfs_01_0294(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF (M10_탈회횟수_누적 > 0) & (M09_탈회횟수_누적 == M10_탈회횟수_누적)
        THEN M10_최종탈회후경과월 = M09_최종탈회후경과월 + 1
        ELSE M10_최종탈회후경과월 = 0
    """
    dd = df[["M09_탈회횟수_누적", "M10_탈회횟수_누적", "M09_최종탈회후경과월"]]
    res = dd.apply(lambda x: x[2] + 1 if (x[0] > 0) & (x[0] == x[1]) else 0, axis=1)

    c = df["M10_최종탈회후경과월"]
    return c == res


# 01_M11
@constraint_udf
def cfs_01_0359(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_남녀구분코드 = M07_남녀구분코드
    """
    res = df["M07_남녀구분코드"]

    c = df["M11_남녀구분코드"]
    return c == res


@constraint_udf
def cfs_01_0360(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_연령 = M07_연령
    """
    res = df["M07_연령"]

    c = df["M11_연령"]
    return c == res


@constraint_udf
def cfs_01_0361(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_VIP등급코드 = M07_VIP등급코드
    """
    res = df["M07_VIP등급코드"]

    c = df["M11_VIP등급코드"]
    return c == res


@constraint_udf
def cfs_01_0362(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_최상위카드등급코드 = M07_최상위카드등급코드
    """
    res = df["M07_최상위카드등급코드"]

    c = df["M11_최상위카드등급코드"]
    return c == res


@constraint_udf
def cfs_01_0369(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_입회일자_신용 = M07_입회일자_신용
    """
    res = df["M07_입회일자_신용"]

    c = df["M11_입회일자_신용"]
    return c == res


@constraint_udf
def cfs_01_0370(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_입회경과개월수_신용 = M07_입회경과개월수_신용 + 4
    """
    dd = df[["M07_입회경과개월수_신용"]]
    res = dd.apply(lambda x: x[0] + 4, axis=1)

    c = df["M11_입회경과개월수_신용"]
    return c == res


# @constraint_udf
# def cfs_01_0359(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M11_유치경로코드_신용 = M07_유치경로코드_신용
#     """
#     res = df["M07_유치경로코드_신용"]
#
#    c = df["M11_유치경로코드_신용"]
#    return c == res


# @constraint_udf
# def cfs_01_0360(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M11_자사카드자격코드 = M07_자사카드자격코드
#     """
#     res = df["M07_자사카드자격코드"]
#
#    c = df["M11_자사카드자격코드"]
#    return c == res


@constraint_udf
def cfs_01_0383(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF (M11_탈회횟수_누적 > 0) & (M10_탈회횟수_누적 == M11_탈회횟수_누적)
        THEN M11_최종탈회후경과월 = M10_최종탈회후경과월 + 1
        ELSE M11_최종탈회후경과월 = 0
    """
    dd = df[["M10_탈회횟수_누적", "M11_탈회횟수_누적", "M10_최종탈회후경과월"]]
    res = dd.apply(lambda x: x[2] + 1 if (x[0] > 0) & (x[0] == x[1]) else 0, axis=1)

    c = df["M11_최종탈회후경과월"]
    return c == res


# 01_M12
@constraint_udf
def cfs_01_0448(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_남녀구분코드 = M07_남녀구분코드
    """
    res = df["M07_남녀구분코드"]

    c = df["M12_남녀구분코드"]
    return c == res


@constraint_udf
def cfs_01_0449(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_연령 = M07_연령
    """
    res = df["M07_연령"]

    c = df["M12_연령"]
    return c == res


@constraint_udf
def cfs_01_0450(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_VIP등급코드 = M07_VIP등급코드
    """
    res = df["M07_VIP등급코드"]

    c = df["M12_VIP등급코드"]
    return c == res


@constraint_udf
def cfs_01_0451(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_최상위카드등급코드 = M07_최상위카드등급코드
    """
    res = df["M07_최상위카드등급코드"]

    c = df["M12_최상위카드등급코드"]
    return c == res


@constraint_udf
def cfs_01_0458(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_입회일자_신용 = M07_입회일자_신용
    """
    res = df["M07_입회일자_신용"]

    c = df["M12_입회일자_신용"]
    return c == res


@constraint_udf
def cfs_01_0459(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_입회경과개월수_신용 = M07_입회경과개월수_신용 + 5
    """
    dd = df[["M07_입회경과개월수_신용"]]
    res = dd.apply(lambda x: x[0] + 5, axis=1)

    c = df["M12_입회경과개월수_신용"]
    return c == res


# @constraint_udf
# def cfs_01_0445(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M12_유치경로코드_신용 = M07_유치경로코드_신용
#     """
#     res = df["M07_유치경로코드_신용"]
#
#    c = df["M12_유치경로코드_신용"]
#    return c == res


# @constraint_udf
# def cfs_01_0446(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M12_자사카드자격코드 = M07_자사카드자격코드
#     """
#     res = df["M07_자사카드자격코드"]
#
#    c = df["M12_자사카드자격코드"]
#    return c == res


@constraint_udf
def cfs_01_0472(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF (M12_탈회횟수_누적 > 0) & (M11_탈회횟수_누적 == M12_탈회횟수_누적)
        THEN M12_최종탈회후경과월 = M11_최종탈회후경과월 + 1
        ELSE M12_최종탈회후경과월 = 0
    """
    dd = df[["M11_탈회횟수_누적", "M12_탈회횟수_누적", "M11_최종탈회후경과월"]]
    res = dd.apply(lambda x: x[2] + 1 if (x[0] > 0) & (x[0] == x[1]) else 0, axis=1)

    c = df["M12_최종탈회후경과월"]
    return c == res


# 02_Constraints
@constraint_udf
def ccs_02_0066(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_RV신청일자 <= M08_RV신청일자
    """
    dd = df[["M07_RV신청일자", "M08_RV신청일자"]]
    res = dd.apply(lambda x: x[0] <= x[1] if (not isNaN(x[0])) & (not isNaN(x[1])) else True, axis=1)
    return res


@constraint_udf
def ccs_02_0120(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_RV신청일자 <= M09_RV신청일자
    """
    dd = df[["M08_RV신청일자", "M09_RV신청일자"]]
    res = dd.apply(lambda x: x[0] <= x[1] if (not isNaN(x[0])) & (not isNaN(x[1])) else True, axis=1)
    return res


@constraint_udf
def ccs_02_0174(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_RV신청일자 <= M10_RV신청일자
    """
    dd = df[["M09_RV신청일자", "M10_RV신청일자"]]
    res = dd.apply(lambda x: x[0] <= x[1] if (not isNaN(x[0])) & (not isNaN(x[1])) else True, axis=1)
    return res


@constraint_udf
def ccs_02_0228(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_RV신청일자 <= M11_RV신청일자
    """
    dd = df[["M10_RV신청일자", "M11_RV신청일자"]]
    res = dd.apply(lambda x: x[0] <= x[1] if (not isNaN(x[0])) & (not isNaN(x[1])) else True, axis=1)
    return res


@constraint_udf
def ccs_02_0282(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_RV신청일자 <= M12_RV신청일자
    """
    dd = df[["M11_RV신청일자", "M12_RV신청일자"]]
    res = dd.apply(lambda x: x[0] <= x[1] if (not isNaN(x[0])) & (not isNaN(x[1])) else True, axis=1)
    return res


# 02_M08
@constraint_udf
def cfs_02_0057(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_최초한도금액 = M07_최초한도금액
    """
    res = df["M07_최초한도금액"]

    c = df["M08_최초한도금액"]
    return c == res


@constraint_udf
def cfs_02_0071(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_자발한도감액횟수_R12M == 0 THEN M08_자발한도감액후경과월 = 12
        ELIF M08_자발한도감액횟수_R12M > M07_자발한도감액횟수_R12M THEN M08_자발한도감액후경과월 = 0
        ELSE M08_자발한도감액후경과월 =  M07_자발한도감액후경과월 + 1
    """
    dd = df[["M08_자발한도감액횟수_R12M", "M07_자발한도감액횟수_R12M", "M07_자발한도감액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M08_자발한도감액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0074(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_강제한도감액횟수_R12M == 0 THEN M08_강제한도감액후경과월 = 12
        ELIF M08_강제한도감액횟수_R12M > M07_강제한도감액횟수_R12M THEN M08_강제한도감액후경과월 = 0
        ELSE M08_강제한도감액후경과월 =  M07_강제한도감액후경과월 + 1
    """
    dd = df[["M08_강제한도감액횟수_R12M", "M07_강제한도감액횟수_R12M", "M07_강제한도감액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M08_강제한도감액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0077(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_한도증액횟수_R12M == 0 THEN M08_한도증액후경과월 = 12
        ELIF M08_한도증액횟수_R12M > M07_한도증액횟수_R12M THEN M08_한도증액후경과월 = 0
        ELSE M08_한도증액후경과월 =  M07_한도증액후경과월 + 1
    """
    dd = df[["M08_한도증액횟수_R12M", "M07_한도증액횟수_R12M", "M07_한도증액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M08_한도증액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0091(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_한도심사요청건수 == 0 THEN M08_한도심사요청후경과월 = 3
        ELIF M08_한도심사요청건수 > M07_한도심사요청건수 THEN M08_한도심사요청후경과월 = 0
        ELSE M08_한도심사요청후경과월 =  M07_한도심사요청후경과월 + 1
    """
    dd = df[["M08_한도심사요청건수", "M07_한도심사요청건수", "M07_한도심사요청후경과월"]]
    res = dd.apply(lambda x: 3 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M08_한도심사요청후경과월"]
    return c == res


@constraint_udf
def cfs_02_0092(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_한도요청거절건수 == 0 THEN M08_한도심사거절후경과월 = 3
        ELIF M08_한도요청거절건수 > M07_한도요청거절건수 THEN M08_한도심사거절후경과월 = 0
        ELSE M08_한도심사거절후경과월 =  M07_한도심사거절후경과월 + 1
    """
    dd = df[["M08_한도요청거절건수", "M07_한도요청거절건수", "M07_한도심사거절후경과월"]]
    res = dd.apply(lambda x: 3 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M08_한도심사거절후경과월"]
    return c == res


@constraint_udf
def cfs_02_0085(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_카드이용한도금액_B2M = M07_카드이용한도금액_B1M
    """
    res = df["M07_카드이용한도금액_B1M"]

    c = df["M08_카드이용한도금액_B2M"]
    return c == res


# 02_M09
@constraint_udf
def cfs_02_0111(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_최초한도금액 = M07_최초한도금액
    """
    res = df["M07_최초한도금액"]

    c = df["M09_최초한도금액"]
    return c == res


@constraint_udf
def cfs_02_0125(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_자발한도감액횟수_R12M == 0 THEN M09_자발한도감액후경과월 = 12
        ELIF M09_자발한도감액횟수_R12M > M08_자발한도감액횟수_R12M THEN M09_자발한도감액후경과월 = 0
        ELSE M09_자발한도감액후경과월 =  M08_자발한도감액후경과월 + 1
    """
    dd = df[["M09_자발한도감액횟수_R12M", "M08_자발한도감액횟수_R12M", "M08_자발한도감액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M09_자발한도감액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0128(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_강제한도감액횟수_R12M == 0 THEN M09_강제한도감액후경과월 = 12
        ELIF M09_강제한도감액횟수_R12M > M08_강제한도감액횟수_R12M THEN M09_강제한도감액후경과월 = 0
        ELSE M09_강제한도감액후경과월 =  M08_강제한도감액후경과월 + 1
    """
    dd = df[["M09_강제한도감액횟수_R12M", "M08_강제한도감액횟수_R12M", "M08_강제한도감액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M09_강제한도감액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0131(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_한도증액횟수_R12M == 0 THEN M09_한도증액후경과월 = 12
        ELIF M09_한도증액횟수_R12M > M08_한도증액횟수_R12M THEN M09_한도증액후경과월 = 0
        ELSE M09_한도증액후경과월 =  M08_한도증액후경과월 + 1
    """
    dd = df[["M09_한도증액횟수_R12M", "M08_한도증액횟수_R12M", "M08_한도증액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M09_한도증액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0145(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_한도심사요청건수 == 0 THEN M09_한도심사요청후경과월 = 3
        ELIF M09_한도심사요청건수 > M08_한도심사요청건수 THEN M09_한도심사요청후경과월 = 0
        ELSE M09_한도심사요청후경과월 =  M08_한도심사요청후경과월 + 1
    """
    dd = df[["M09_한도심사요청건수", "M08_한도심사요청건수", "M08_한도심사요청후경과월"]]
    res = dd.apply(lambda x: 3 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M09_한도심사요청후경과월"]
    return c == res


@constraint_udf
def cfs_02_0146(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_한도요청거절건수 == 0 THEN M09_한도심사거절후경과월 = 3
        ELIF M09_한도요청거절건수 > M08_한도요청거절건수 THEN M09_한도심사거절후경과월 = 0
        ELSE M09_한도심사거절후경과월 =  M08_한도심사거절후경과월 + 1
    """
    dd = df[["M09_한도요청거절건수", "M08_한도요청거절건수", "M08_한도심사거절후경과월"]]
    res = dd.apply(lambda x: 3 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M09_한도심사거절후경과월"]
    return c == res


@constraint_udf
def cfs_02_0139(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_카드이용한도금액_B2M = M08_카드이용한도금액_B1M
    """
    res = df["M08_카드이용한도금액_B1M"]

    c = df["M09_카드이용한도금액_B2M"]
    return c == res


# 02_M10
@constraint_udf
def cfs_02_0165(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_최초한도금액 = M07_최초한도금액
    """
    res = df["M07_최초한도금액"]

    c = df["M10_최초한도금액"]
    return c == res


@constraint_udf
def cfs_02_0179(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_자발한도감액횟수_R12M == 0 THEN M10_자발한도감액후경과월 = 12
        ELIF M10_자발한도감액횟수_R12M > M09_자발한도감액횟수_R12M THEN M10_자발한도감액후경과월 = 0
        ELSE M10_자발한도감액후경과월 =  M09_자발한도감액후경과월 + 1
    """
    dd = df[["M10_자발한도감액횟수_R12M", "M09_자발한도감액횟수_R12M", "M09_자발한도감액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M10_자발한도감액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0182(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_강제한도감액횟수_R12M == 0 THEN M10_강제한도감액후경과월 = 12
        ELIF M10_강제한도감액횟수_R12M > M09_강제한도감액횟수_R12M THEN M10_강제한도감액후경과월 = 0
        ELSE M10_강제한도감액후경과월 =  M09_강제한도감액후경과월 + 1
    """
    dd = df[["M10_강제한도감액횟수_R12M", "M09_강제한도감액횟수_R12M", "M09_강제한도감액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M10_강제한도감액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0185(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_한도증액횟수_R12M == 0 THEN M10_한도증액후경과월 = 12
        ELIF M10_한도증액횟수_R12M > M09_한도증액횟수_R12M THEN M10_한도증액후경과월 = 0
        ELSE M10_한도증액후경과월 =  M09_한도증액후경과월 + 1
    """
    dd = df[["M10_한도증액횟수_R12M", "M09_한도증액횟수_R12M", "M09_한도증액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M10_한도증액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0199(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_한도심사요청건수 == 0 THEN M10_한도심사요청후경과월 = 3
        ELIF M10_한도심사요청건수 > M09_한도심사요청건수 THEN M10_한도심사요청후경과월 = 0
        ELSE M10_한도심사요청후경과월 =  M09_한도심사요청후경과월 + 1
    """
    dd = df[["M10_한도심사요청건수", "M09_한도심사요청건수", "M09_한도심사요청후경과월"]]
    res = dd.apply(lambda x: 3 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M10_한도심사요청후경과월"]
    return c == res


@constraint_udf
def cfs_02_0200(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_한도요청거절건수 == 0 THEN M10_한도심사거절후경과월 = 3
        ELIF M10_한도요청거절건수 > M09_한도요청거절건수 THEN M10_한도심사거절후경과월 = 0
        ELSE M10_한도심사거절후경과월 =  M09_한도심사거절후경과월 + 1
    """
    dd = df[["M10_한도요청거절건수", "M09_한도요청거절건수", "M09_한도심사거절후경과월"]]
    res = dd.apply(lambda x: 3 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M10_한도심사거절후경과월"]
    return c == res


@constraint_udf
def cfs_02_0193(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_카드이용한도금액_B2M = M09_카드이용한도금액_B1M
    """
    res = df["M09_카드이용한도금액_B1M"]

    c = df["M10_카드이용한도금액_B2M"]
    return c == res


# 02_M11
@constraint_udf
def cfs_02_0219(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_최초한도금액 = M07_최초한도금액
    """
    res = df["M07_최초한도금액"]

    c = df["M11_최초한도금액"]
    return c == res


@constraint_udf
def cfs_02_0233(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_자발한도감액횟수_R12M == 0 THEN M11_자발한도감액후경과월 = 12
        ELIF M11_자발한도감액횟수_R12M > M10_자발한도감액횟수_R12M THEN M11_자발한도감액후경과월 = 0
        ELSE M11_자발한도감액후경과월 =  M10_자발한도감액후경과월 + 1
    """
    dd = df[["M11_자발한도감액횟수_R12M", "M10_자발한도감액횟수_R12M", "M10_자발한도감액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M11_자발한도감액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0236(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_강제한도감액횟수_R12M == 0 THEN M11_강제한도감액후경과월 = 12
        ELIF M11_강제한도감액횟수_R12M > M10_강제한도감액횟수_R12M THEN M11_강제한도감액후경과월 = 0
        ELSE M11_강제한도감액후경과월 =  M10_강제한도감액후경과월 + 1
    """
    dd = df[["M11_강제한도감액횟수_R12M", "M10_강제한도감액횟수_R12M", "M10_강제한도감액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M11_강제한도감액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0239(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_한도증액횟수_R12M == 0 THEN M11_한도증액후경과월 = 12
        ELIF M11_한도증액횟수_R12M > M10_한도증액횟수_R12M THEN M11_한도증액후경과월 = 0
        ELSE M11_한도증액후경과월 =  M10_한도증액후경과월 + 1
    """
    dd = df[["M11_한도증액횟수_R12M", "M10_한도증액횟수_R12M", "M10_한도증액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M11_한도증액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0253(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_한도심사요청건수 == 0 THEN M11_한도심사요청후경과월 = 3
        ELIF M11_한도심사요청건수 > M10_한도심사요청건수 THEN M11_한도심사요청후경과월 = 0
        ELSE M11_한도심사요청후경과월 =  M10_한도심사요청후경과월 + 1
    """
    dd = df[["M11_한도심사요청건수", "M10_한도심사요청건수", "M10_한도심사요청후경과월"]]
    res = dd.apply(lambda x: 3 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M11_한도심사요청후경과월"]
    return c == res


@constraint_udf
def cfs_02_0254(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_한도요청거절건수 == 0 THEN M11_한도심사거절후경과월 = 3
        ELIF M11_한도요청거절건수 > M10_한도요청거절건수 THEN M11_한도심사거절후경과월 = 0
        ELSE M11_한도심사거절후경과월 =  M10_한도심사거절후경과월 + 1
    """
    dd = df[["M11_한도요청거절건수", "M10_한도요청거절건수", "M10_한도심사거절후경과월"]]
    res = dd.apply(lambda x: 3 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M11_한도심사거절후경과월"]
    return c == res


@constraint_udf
def cfs_02_0247(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_카드이용한도금액_B2M = M10_카드이용한도금액_B1M
    """
    res = df["M10_카드이용한도금액_B1M"]

    c = df["M11_카드이용한도금액_B2M"]
    return c == res


# 02_M12
@constraint_udf
def cfs_02_0273(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_최초한도금액 = M07_최초한도금액
    """
    res = df["M07_최초한도금액"]

    c = df["M12_최초한도금액"]
    return c == res


@constraint_udf
def cfs_02_0287(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_자발한도감액횟수_R12M == 0 THEN M12_자발한도감액후경과월 = 12
        ELIF M12_자발한도감액횟수_R12M > M11_자발한도감액횟수_R12M THEN M12_자발한도감액후경과월 = 0
        ELSE M12_자발한도감액후경과월 =  M11_자발한도감액후경과월 + 1
    """
    dd = df[["M12_자발한도감액횟수_R12M", "M11_자발한도감액횟수_R12M", "M11_자발한도감액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M12_자발한도감액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0290(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_강제한도감액횟수_R12M == 0 THEN M12_강제한도감액후경과월 = 12
        ELIF M12_강제한도감액횟수_R12M > M11_강제한도감액횟수_R12M THEN M12_강제한도감액후경과월 = 0
        ELSE M12_강제한도감액후경과월 =  M11_강제한도감액후경과월 + 1
    """
    dd = df[["M12_강제한도감액횟수_R12M", "M11_강제한도감액횟수_R12M", "M11_강제한도감액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M12_강제한도감액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0293(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_한도증액횟수_R12M == 0 THEN M12_한도증액후경과월 = 12
        ELIF M12_한도증액횟수_R12M > M11_한도증액횟수_R12M THEN M12_한도증액후경과월 = 0
        ELSE M12_한도증액후경과월 =  M11_한도증액후경과월 + 1
    """
    dd = df[["M12_한도증액횟수_R12M", "M11_한도증액횟수_R12M", "M11_한도증액후경과월"]]
    res = dd.apply(lambda x: 12 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M12_한도증액후경과월"]
    return c == res


@constraint_udf
def cfs_02_0307(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_한도심사요청건수 == 0 THEN M12_한도심사요청후경과월 = 3
        ELIF M12_한도심사요청건수 > M11_한도심사요청건수 THEN M12_한도심사요청후경과월 = 0
        ELSE M12_한도심사요청후경과월 =  M11_한도심사요청후경과월 + 1
    """
    dd = df[["M12_한도심사요청건수", "M11_한도심사요청건수", "M11_한도심사요청후경과월"]]
    res = dd.apply(lambda x: 3 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M12_한도심사요청후경과월"]
    return c == res


@constraint_udf
def cfs_02_0308(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_한도요청거절건수 == 0 THEN M12_한도심사거절후경과월 = 12
        ELIF M12_한도요청거절건수 > M11_한도요청거절건수 THEN M12_한도심사거절후경과월 = 0
        ELSE M12_한도심사거절후경과월 =  M11_한도심사거절후경과월 + 1
    """
    dd = df[["M12_한도요청거절건수", "M11_한도요청거절건수", "M11_한도심사거절후경과월"]]
    res = dd.apply(lambda x: 3 if x[0] == 0 else (0 if x[0] > x[1] else x[2] + 1), axis=1)

    c = df["M12_한도심사거절후경과월"]
    return c == res


@constraint_udf
def cfs_02_0301(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_카드이용한도금액_B2M = M11_카드이용한도금액_B1M
    """
    res = df["M11_카드이용한도금액_B1M"]

    c = df["M12_카드이용한도금액_B2M"]
    return c == res


# 04_M09
@constraint_udf
def cfs_04_0112(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07 & M08 & M09_청구서발송여부_B0 == '0' THEN M09_청구서발송여부_R3M = '0' ELSE '1'
    """
    dd = df[["M09_청구서발송여부_B0", "M08_청구서발송여부_B0", "M07_청구서발송여부_B0"]]
    res = dd.apply(lambda x: '0'
                   if (x[0] == '0') & (x[1] == '0') & (x[2] == '0')
                   else '1',
                   axis=1)

    c = df["M09_청구서발송여부_R3M"]
    return c == res


@constraint_udf
def cfs_04_0115(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_청구금액_R3M = M09_청구금액_B0 + M08_청구금액_B0 + M07_청구금액_B0
    """
    dd = df[["M09_청구금액_B0", "M08_청구금액_B0", "M07_청구금액_B0"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_청구금액_R3M"]
    return c == res


@constraint_udf
def cfs_04_0118(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_포인트_마일리지_건별_R3M = M09_포인트_마일리지_건별_B0M + M08_포인트_마일리지_건별_B0M + M07_포인트_마일리지_건별_B0M
    """
    dd = df[["M09_포인트_마일리지_건별_B0M", "M08_포인트_마일리지_건별_B0M", "M07_포인트_마일리지_건별_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_포인트_마일리지_건별_R3M"]
    return c == res


@constraint_udf
def cfs_04_0120(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        "M09_포인트_포인트_건별_R3M = M09_포인트_포인트_건별_B0M + M08_포인트_포인트_건별_B0M + M07_포인트_포인트_건별_B0M
    """
    dd = df[["M09_포인트_포인트_건별_B0M", "M08_포인트_포인트_건별_B0M", "M07_포인트_포인트_건별_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_포인트_포인트_건별_R3M"]
    return c == res


@constraint_udf
def cfs_04_0122(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_포인트_마일리지_월적립_R3M = M09_포인트_마일리지_월적립_B0M + M08_포인트_마일리지_월적립_B0M + M07_포인트_마일리지_월적립_B0M
    """
    dd = df[["M09_포인트_마일리지_월적립_B0M", "M08_포인트_마일리지_월적립_B0M", "M07_포인트_마일리지_월적립_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_포인트_마일리지_월적립_R3M"]
    return c == res


@constraint_udf
def cfs_04_0124(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_포인트_포인트_월적립_R3M = M09_포인트_포인트_월적립_B0M + M08_포인트_포인트_월적립_B0M + M07_포인트_포인트_건별_B0M
    """
    dd = df[["M09_포인트_포인트_월적립_B0M", "M08_포인트_포인트_월적립_B0M", "M07_포인트_포인트_월적립_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_포인트_포인트_월적립_R3M"]
    return c == res


# @constraint_udf
# def cfs_04_0126(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M09_포인트_적립포인트_R3M = M09_포인트_적립포인트_B0M + M08_포인트_적립포인트_B0M + M07_포인트_적립포인트_B0M
#     """
#     dd = df[["M09_포인트_적립포인트_B0M", "M08_포인트_적립포인트_B0M", "M07_포인트_적립포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M09_포인트_적립포인트_R3M"]
#     return c == res


# @constraint_udf
# def cfs_04_0128(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M09_포인트_이용포인트_R3M = M09_포인트_이용포인트_B0M + M08_포인트_이용포인트_B0M + M07_포인트_이용포인트_B0M
#     """
#     dd = df[["M09_포인트_이용포인트_B0M", "M08_포인트_이용포인트_B0M", "M07_포인트_이용포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M09_포인트_이용포인트_R3M"]
#     return c == res


# @constraint_udf
# def cfs_04_0131(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M09_마일_적립포인트_R3M = M09_마일_적립포인트_B0M + M08_마일_적립포인트_B0M + M07_마일_적립포인트_B0M
#     """
#     dd = df[["M09_마일_적립포인트_B0M", "M08_마일_적립포인트_B0M", "M07_마일_적립포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M09_마일_적립포인트_R3M"]
#     return c == res


# @constraint_udf
# def cfs_04_0133(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M09_마일_이용포인트_R3M = M09_마일_이용포인트_B0M + M08_마일_이용포인트_B0M + M07_마일_이용포인트_B0M
#     """
#     dd = df[["M09_마일_이용포인트_B0M", "M08_마일_이용포인트_B0M", "M07_마일_이용포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M09_마일_이용포인트_R3M"]
#     return c == res


@constraint_udf
def cfs_04_0135(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_할인건수_R3M = M09_할인건수_B0M + M08_할인건수_B0M + M07_할인건수_B0M
    """
    dd = df[["M09_할인건수_B0M", "M08_할인건수_B0M", "M07_할인건수_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_할인건수_R3M"]
    return c == res


@constraint_udf
def cfs_04_0136(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_할인금액_R3M = M09_할인금액_B0M + M08_할인금액_B0M + M07_할인금액_B0M
    """
    dd = df[["M09_할인금액_B0M", "M08_할인금액_B0M", "M07_할인금액_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_할인금액_R3M"]
    return c == res


@constraint_udf
def cfs_04_0139(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_할인금액_청구서_R3M = M09_할인금액_청구서_B0M + M08_할인금액_청구서_B0M + M07_할인금액_청구서_B0M
    """
    dd = df[["M09_할인금액_청구서_B0M", "M08_할인금액_청구서_B0M", "M07_할인금액_청구서_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_할인금액_청구서_R3M"]
    return c == res


@constraint_udf
def cfs_04_0141(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       M09_혜택수혜금액_R3M = M09_혜택수혜금액 + M08_혜택수혜금액 + M07_혜택수혜금액
    """
    dd = df[["M09_혜택수혜금액", "M08_혜택수혜금액", "M07_혜택수혜금액"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_혜택수혜금액_R3M"]
    return c == res


# 04_M10
@constraint_udf
def cfs_04_0163(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08 & M09 & M10_청구서발송여부_B0 == '0' THEN M10_청구서발송여부_R3M = '0' ELSE '1'
    """
    dd = df[["M10_청구서발송여부_B0", "M09_청구서발송여부_B0", "M08_청구서발송여부_B0"]]
    res = dd.apply(lambda x: '0'
                   if (x[0] == '0') & (x[1] == '0') & (x[2] == '0')
                   else '1',
                   axis=1)

    c = df["M10_청구서발송여부_R3M"]
    return c == res


@constraint_udf
def cfs_04_0164(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07 & M10_청구서발송여부_R3M == '0' THEN M10_청구서발송여부_R6M = '0' ELSE '1'
    """
    dd = df[["M10_청구서발송여부_R3M", "M07_청구서발송여부_R3M"]]
    res = dd.apply(lambda x: '0'
                   if (x[0] == '0') & (x[1] == '0')
                   else '1',
                   axis=1)

    c = df["M10_청구서발송여부_R6M"]
    return c == res


@constraint_udf
def cfs_04_0166(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_청구금액_R3M = M10_청구금액_B0 + M09_청구금액_B0 + M08_청구금액_B0
    """
    dd = df[["M10_청구금액_B0", "M09_청구금액_B0", "M08_청구금액_B0"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_청구금액_R3M"]
    return c == res


@constraint_udf
def cfs_04_0167(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_청구금액_R6M = M10_청구금액_R3M + M07_청구금액_R3M
    """
    dd = df[["M10_청구금액_R3M", "M07_청구금액_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_청구금액_R6M"]
    return c == res


@constraint_udf
def cfs_04_0169(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_포인트_마일리지_건별_R3M = M10_포인트_마일리지_건별_B0M + M09_포인트_마일리지_건별_B0M + M08_포인트_마일리지_건별_B0M
    """
    dd = df[["M10_포인트_마일리지_건별_B0M", "M09_포인트_마일리지_건별_B0M", "M08_포인트_마일리지_건별_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_포인트_마일리지_건별_R3M"]
    return c == res


@constraint_udf
def cfs_04_0171(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        "M10_포인트_포인트_건별_R3M = M10_포인트_포인트_건별_B0M + M09_포인트_포인트_건별_B0M + M08_포인트_포인트_건별_B0M
    """
    dd = df[["M10_포인트_포인트_건별_B0M", "M09_포인트_포인트_건별_B0M", "M08_포인트_포인트_건별_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_포인트_포인트_건별_R3M"]
    return c == res


@constraint_udf
def cfs_04_0173(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_포인트_마일리지_월적립_R3M = M10_포인트_마일리지_월적립_B0M + M09_포인트_마일리지_월적립_B0M + M08_포인트_마일리지_월적립_B0M
    """
    dd = df[["M10_포인트_마일리지_월적립_B0M", "M09_포인트_마일리지_월적립_B0M", "M08_포인트_마일리지_월적립_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_포인트_마일리지_월적립_R3M"]
    return c == res


@constraint_udf
def cfs_04_0175(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_포인트_포인트_월적립_R3M = M10_포인트_포인트_월적립_B0M + M09_포인트_포인트_월적립_B0M + M08_포인트_포인트_건별_B0M
    """
    dd = df[["M10_포인트_포인트_월적립_B0M", "M09_포인트_포인트_월적립_B0M", "M08_포인트_포인트_월적립_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_포인트_포인트_월적립_R3M"]
    return c == res


# @constraint_udf
# def cfs_04_0177(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M10_포인트_적립포인트_R3M = M10_포인트_적립포인트_B0M + M09_포인트_적립포인트_B0M + M08_포인트_적립포인트_B0M
#     """
#     dd = df[["M10_포인트_적립포인트_B0M", "M09_포인트_적립포인트_B0M", "M08_포인트_적립포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M10_포인트_적립포인트_R3M"]
#     return c == res


# @constraint_udf
# def cfs_04_0179(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M10_포인트_이용포인트_R3M = M10_포인트_이용포인트_B0M + M09_포인트_이용포인트_B0M + M08_포인트_이용포인트_B0M
#     """
#     dd = df[["M10_포인트_이용포인트_B0M", "M09_포인트_이용포인트_B0M", "M08_포인트_이용포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M10_포인트_이용포인트_R3M"]
#     return c == res


# @constraint_udf
# def cfs_04_0182(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M10_마일_적립포인트_R3M = M10_마일_적립포인트_B0M + M09_마일_적립포인트_B0M + M08_마일_적립포인트_B0M
#     """
#     dd = df[["M10_마일_적립포인트_B0M", "M09_마일_적립포인트_B0M", "M08_마일_적립포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M10_마일_적립포인트_R3M"]
#     return c == res


# @constraint_udf
# def cfs_04_0184(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M10_마일_이용포인트_R3M = M10_마일_이용포인트_B0M + M09_마일_이용포인트_B0M + M08_마일_이용포인트_B0M
#     """
#     dd = df[["M10_마일_이용포인트_B0M", "M09_마일_이용포인트_B0M", "M08_마일_이용포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M10_마일_이용포인트_R3M"]
#     return c == res


@constraint_udf
def cfs_04_0186(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_할인건수_R3M = M10_할인건수_B0M + M09_할인건수_B0M + M08_할인건수_B0M
    """
    dd = df[["M10_할인건수_B0M", "M09_할인건수_B0M", "M08_할인건수_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_할인건수_R3M"]
    return c == res


@constraint_udf
def cfs_04_0187(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_할인금액_R3M = M10_할인금액_B0M + M09_할인금액_B0M + M08_할인금액_B0M
    """
    dd = df[["M10_할인금액_B0M", "M09_할인금액_B0M", "M08_할인금액_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_할인금액_R3M"]
    return c == res


@constraint_udf
def cfs_04_0190(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_할인금액_청구서_R3M = M10_할인금액_청구서_B0M + M09_할인금액_청구서_B0M + M08_할인금액_청구서_B0M
    """
    dd = df[["M10_할인금액_청구서_B0M", "M09_할인금액_청구서_B0M", "M08_할인금액_청구서_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_할인금액_청구서_R3M"]
    return c == res


@constraint_udf
def cfs_04_0192(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       M10_혜택수혜금액_R3M = M10_혜택수혜금액 + M09_혜택수혜금액 + M08_혜택수혜금액
    """
    dd = df[["M10_혜택수혜금액", "M09_혜택수혜금액", "M08_혜택수혜금액"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_혜택수혜금액_R3M"]
    return c == res


# 04_M11
@constraint_udf
def cfs_04_0214(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09 & M10 & M11_청구서발송여부_B0 == '0' THEN M11_청구서발송여부_R3M = '0' ELSE '1'
    """
    dd = df[["M11_청구서발송여부_B0", "M10_청구서발송여부_B0", "M09_청구서발송여부_B0"]]
    res = dd.apply(lambda x: '0'
                   if (x[0] == '0') & (x[1] == '0') & (x[2] == '0')
                   else '1',
                   axis=1)

    c = df["M11_청구서발송여부_R3M"]
    return c == res


@constraint_udf
def cfs_04_0215(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08 & M11_청구서발송여부_R3M == '0' THEN M11_청구서발송여부_R6M = '0' ELSE '1'
    """
    dd = df[["M11_청구서발송여부_R3M", "M08_청구서발송여부_R3M"]]
    res = dd.apply(lambda x: '0'
                   if (x[0] == '0') & (x[1] == '0')
                   else '1',
                   axis=1)

    c = df["M11_청구서발송여부_R6M"]
    return c == res


@constraint_udf
def cfs_04_0217(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_청구금액_R3M = M11_청구금액_B0 + M10_청구금액_B0 + M09_청구금액_B0
    """
    dd = df[["M11_청구금액_B0", "M10_청구금액_B0", "M09_청구금액_B0"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_청구금액_R3M"]
    return c == res


@constraint_udf
def cfs_04_0218(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_청구금액_R6M = M11_청구금액_R3M + M08_청구금액_R3M
    """
    dd = df[["M11_청구금액_R3M", "M08_청구금액_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_청구금액_R6M"]
    return c == res


@constraint_udf
def cfs_04_0220(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_포인트_마일리지_건별_R3M = M11_포인트_마일리지_건별_B0M + M10_포인트_마일리지_건별_B0M + M09_포인트_마일리지_건별_B0M
    """
    dd = df[["M11_포인트_마일리지_건별_B0M", "M10_포인트_마일리지_건별_B0M", "M09_포인트_마일리지_건별_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_포인트_마일리지_건별_R3M"]
    return c == res


@constraint_udf
def cfs_04_0222(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_포인트_포인트_건별_R3M = M11_포인트_포인트_건별_B0M + M10_포인트_포인트_건별_B0M + M09_포인트_포인트_건별_B0M
    """
    dd = df[["M11_포인트_포인트_건별_B0M", "M10_포인트_포인트_건별_B0M", "M09_포인트_포인트_건별_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_포인트_포인트_건별_R3M"]
    return c == res


@constraint_udf
def cfs_04_0224(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_포인트_마일리지_월적립_R3M = M11_포인트_마일리지_월적립_B0M + M10_포인트_마일리지_월적립_B0M + M09_포인트_마일리지_월적립_B0M
    """
    dd = df[["M11_포인트_마일리지_월적립_B0M", "M10_포인트_마일리지_월적립_B0M", "M09_포인트_마일리지_월적립_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_포인트_마일리지_월적립_R3M"]
    return c == res


@constraint_udf
def cfs_04_0226(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_포인트_포인트_월적립_R3M = M11_포인트_포인트_월적립_B0M + M10_포인트_포인트_월적립_B0M + M09_포인트_포인트_건별_B0M
    """
    dd = df[["M11_포인트_포인트_월적립_B0M", "M10_포인트_포인트_월적립_B0M", "M09_포인트_포인트_월적립_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_포인트_포인트_월적립_R3M"]
    return c == res


# @constraint_udf
# def cfs_04_0228(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M11_포인트_적립포인트_R3M = M11_포인트_적립포인트_B0M + M10_포인트_적립포인트_B0M + M09_포인트_적립포인트_B0M
#     """
#     dd = df[["M11_포인트_적립포인트_B0M", "M10_포인트_적립포인트_B0M", "M09_포인트_적립포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M11_포인트_적립포인트_R3M"]
#     return c == res


# @constraint_udf
# def cfs_04_0230(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M11_포인트_이용포인트_R3M = M11_포인트_이용포인트_B0M + M10_포인트_이용포인트_B0M + M09_포인트_이용포인트_B0M
#     """
#     dd = df[["M11_포인트_이용포인트_B0M", "M10_포인트_이용포인트_B0M", "M09_포인트_이용포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M11_포인트_이용포인트_R3M"]
#     return c == res


# @constraint_udf
# def cfs_04_0233(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M11_마일_적립포인트_R3M = M11_마일_적립포인트_B0M + M10_마일_적립포인트_B0M + M09_마일_적립포인트_B0M
#     """
#     dd = df[["M11_마일_적립포인트_B0M", "M10_마일_적립포인트_B0M", "M09_마일_적립포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M11_마일_적립포인트_R3M"]
#     return c == res


# @constraint_udf
# def cfs_04_0235(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M11_마일_이용포인트_R3M = M11_마일_이용포인트_B0M + M10_마일_이용포인트_B0M + M09_마일_이용포인트_B0M
#     """
#     dd = df[["M11_마일_이용포인트_B0M", "M10_마일_이용포인트_B0M", "M09_마일_이용포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M11_마일_이용포인트_R3M"]
#     return c == res


@constraint_udf
def cfs_04_0237(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_할인건수_R3M = M11_할인건수_B0M + M10_할인건수_B0M + M09_할인건수_B0M
    """
    dd = df[["M11_할인건수_B0M", "M10_할인건수_B0M", "M09_할인건수_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_할인건수_R3M"]
    return c == res


@constraint_udf
def cfs_04_0238(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_할인금액_R3M = M11_할인금액_B0M + M10_할인금액_B0M + M09_할인금액_B0M
    """
    dd = df[["M11_할인금액_B0M", "M10_할인금액_B0M", "M09_할인금액_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_할인금액_R3M"]
    return c == res


@constraint_udf
def cfs_04_0241(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_할인금액_청구서_R3M = M11_할인금액_청구서_B0M + M10_할인금액_청구서_B0M + M09_할인금액_청구서_B0M
    """
    dd = df[["M11_할인금액_청구서_B0M", "M10_할인금액_청구서_B0M", "M09_할인금액_청구서_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_할인금액_청구서_R3M"]
    return c == res


@constraint_udf
def cfs_04_0243(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       M11_혜택수혜금액_R3M = M11_혜택수혜금액 + M10_혜택수혜금액 + M09_혜택수혜금액
    """
    dd = df[["M11_혜택수혜금액", "M10_혜택수혜금액", "M09_혜택수혜금액"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_혜택수혜금액_R3M"]
    return c == res


# 04_M12
@constraint_udf
def cfs_04_0265(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10 & M11 & M12_청구서발송여부_B0 == '0' THEN M12_청구서발송여부_R3M = '0' ELSE '1'
    """
    dd = df[["M12_청구서발송여부_B0", "M11_청구서발송여부_B0", "M10_청구서발송여부_B0"]]
    res = dd.apply(lambda x: '0'
                   if (x[0] == '0') & (x[1] == '0') & (x[2] == '0')
                   else '1',
                   axis=1)

    c = df["M12_청구서발송여부_R3M"]
    return c == res


@constraint_udf
def cfs_04_0266(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09 & M12_청구서발송여부_R3M == '0' THEN M12_청구서발송여부_R6M = '0' ELSE '1'
    """
    dd = df[["M12_청구서발송여부_R3M", "M09_청구서발송여부_R3M"]]
    res = dd.apply(lambda x: '0'
                   if (x[0] == '0') & (x[1] == '0')
                   else '1',
                   axis=1)

    c = df["M12_청구서발송여부_R6M"]
    return c == res


@constraint_udf
def cfs_04_0268(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_청구금액_R3M = M12_청구금액_B0 + M11_청구금액_B0 + M10_청구금액_B0
    """
    dd = df[["M12_청구금액_B0", "M11_청구금액_B0", "M10_청구금액_B0"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_청구금액_R3M"]
    return c == res


@constraint_udf
def cfs_04_0269(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_청구금액_R6M = M12_청구금액_R3M + M09_청구금액_R3M
    """
    dd = df[["M12_청구금액_R3M", "M09_청구금액_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_청구금액_R6M"]
    return c == res


@constraint_udf
def cfs_04_0271(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_포인트_마일리지_건별_R3M = M12_포인트_마일리지_건별_B0M + M11_포인트_마일리지_건별_B0M + M10_포인트_마일리지_건별_B0M
    """
    dd = df[["M12_포인트_마일리지_건별_B0M", "M11_포인트_마일리지_건별_B0M", "M10_포인트_마일리지_건별_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_포인트_마일리지_건별_R3M"]
    return c == res


@constraint_udf
def cfs_04_0273(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_포인트_포인트_건별_R3M = M12_포인트_포인트_건별_B0M + M11_포인트_포인트_건별_B0M + M10_포인트_포인트_건별_B0M
    """
    dd = df[["M12_포인트_포인트_건별_B0M", "M11_포인트_포인트_건별_B0M", "M10_포인트_포인트_건별_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_포인트_포인트_건별_R3M"]
    return c == res


@constraint_udf
def cfs_04_0275(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_포인트_마일리지_월적립_R3M = M12_포인트_마일리지_월적립_B0M + M11_포인트_마일리지_월적립_B0M + M10_포인트_마일리지_월적립_B0M
    """
    dd = df[["M12_포인트_마일리지_월적립_B0M", "M11_포인트_마일리지_월적립_B0M", "M10_포인트_마일리지_월적립_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_포인트_마일리지_월적립_R3M"]
    return c == res


@constraint_udf
def cfs_04_0277(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_포인트_포인트_월적립_R3M = M12_포인트_포인트_월적립_B0M + M11_포인트_포인트_월적립_B0M + M10_포인트_포인트_건별_B0M
    """
    dd = df[["M12_포인트_포인트_월적립_B0M", "M11_포인트_포인트_월적립_B0M", "M10_포인트_포인트_월적립_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_포인트_포인트_월적립_R3M"]
    return c == res


# @constraint_udf
# def cfs_04_0279(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M12_포인트_적립포인트_R3M = M12_포인트_적립포인트_B0M + M11_포인트_적립포인트_B0M + M10_포인트_적립포인트_B0M
#     """
#     dd = df[["M12_포인트_적립포인트_B0M", "M11_포인트_적립포인트_B0M", "M10_포인트_적립포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M12_포인트_적립포인트_R3M"]
#     return c == res


# @constraint_udf
# def cfs_04_0281(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M12_포인트_이용포인트_R3M = M12_포인트_이용포인트_B0M + M11_포인트_이용포인트_B0M + M10_포인트_이용포인트_B0M
#     """
#     dd = df[["M12_포인트_이용포인트_B0M", "M11_포인트_이용포인트_B0M", "M10_포인트_이용포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M12_포인트_이용포인트_R3M"]
#     return c == res


# @constraint_udf
# def cfs_04_0284(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M12_마일_적립포인트_R3M = M12_마일_적립포인트_B0M + M11_마일_적립포인트_B0M + M10_마일_적립포인트_B0M
#     """
#     dd = df[["M12_마일_적립포인트_B0M", "M11_마일_적립포인트_B0M", "M10_마일_적립포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M12_마일_적립포인트_R3M"]
#     return c == res


# @constraint_udf
# def cfs_04_0286(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         M12_마일_이용포인트_R3M = M12_마일_이용포인트_B0M + M11_마일_이용포인트_B0M + M10_마일_이용포인트_B0M
#     """
#     dd = df[["M12_마일_이용포인트_B0M", "M11_마일_이용포인트_B0M", "M10_마일_이용포인트_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df["M12_마일_이용포인트_R3M"]
#     return c == res


@constraint_udf
def cfs_04_0288(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_할인건수_R3M = M12_할인건수_B0M + M11_할인건수_B0M + M10_할인건수_B0M
    """
    dd = df[["M12_할인건수_B0M", "M11_할인건수_B0M", "M10_할인건수_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_할인건수_R3M"]
    return c == res


@constraint_udf
def cfs_04_0289(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_할인금액_R3M = M12_할인금액_B0M + M11_할인금액_B0M + M10_할인금액_B0M
    """
    dd = df[["M12_할인금액_B0M", "M11_할인금액_B0M", "M10_할인금액_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_할인금액_R3M"]
    return c == res


@constraint_udf
def cfs_04_0292(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_할인금액_청구서_R3M = M12_할인금액_청구서_B0M + M11_할인금액_청구서_B0M + M10_할인금액_청구서_B0M
    """
    dd = df[["M12_할인금액_청구서_B0M", "M11_할인금액_청구서_B0M", "M10_할인금액_청구서_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_할인금액_청구서_R3M"]
    return c == res


@constraint_udf
def cfs_04_0294(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       M12_혜택수혜금액_R3M = M12_혜택수혜금액 + M11_혜택수혜금액 + M10_혜택수혜금액
    """
    dd = df[["M12_혜택수혜금액", "M11_혜택수혜금액", "M10_혜택수혜금액"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_혜택수혜금액_R3M"]
    return c == res


# 05_M07
@constraint_udf
def cfs_05_0013(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_잔액_카드론_B0M >0 THEN M07_카드론잔액_최종경과월 = 0 ELSE pass
    """
    dd = df[["M07_잔액_카드론_B0M", "M07_카드론잔액_최종경과월"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else x[1], axis=1)

    c = df["M07_카드론잔액_최종경과월"]
    return c == res


# 05_M08
@constraint_udf
def cfs_05_0114(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_잔액_카드론_B0M>0 THEN M08_카드론잔액_최종경과월 = 0
        ELIF (M08_잔액_카드론_B0M==0 & M07_잔액_카드론_B0M==0) THEN M08_카드론잔액_최종경과월 = 0
        ELIF (M08_잔액_카드론_B0M==0 & M07_잔액_카드론_B0M>0) THEN M08_카드론잔액_최종경과월 = 1
        ELSE M08_카드론잔액_최종경과월 = M07_카드론잔액_최종경과월 + 1
    """
    dd = df[["M08_잔액_카드론_B0M", "M07_잔액_카드론_B0M", "M07_카드론잔액_최종경과월"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[0] == 0) & (x[1] == 0)
                   else (1 if (x[0] == 0) & (x[1] > 0) else x[2] + 1)), axis=1)

    c = df["M08_카드론잔액_최종경과월"]
    return c == res


@constraint_udf
def cfs_05_0123(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_잔액_현금서비스_B2M = M07_잔액_현금서비스_B1M
    """
    res = df["M07_잔액_현금서비스_B1M"]

    c = df["M08_잔액_현금서비스_B2M"]
    return c == res


@constraint_udf
def cfs_05_0125(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_잔액_카드론_B2M = M07_잔액_카드론_B1M
    """
    res = df["M07_잔액_카드론_B1M"]

    c = df["M08_잔액_카드론_B2M"]
    return c == res


@constraint_udf
def cfs_05_0126(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_잔액_카드론_B3M = M07_잔액_카드론_B2M
    """
    res = df["M07_잔액_카드론_B2M"]

    c = df["M08_잔액_카드론_B3M"]
    return c == res


@constraint_udf
def cfs_05_0127(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_잔액_카드론_B4M = M07_잔액_카드론_B3M
    """
    res = df["M07_잔액_카드론_B3M"]

    c = df["M08_잔액_카드론_B4M"]
    return c == res


@constraint_udf
def cfs_05_0128(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_잔액_카드론_B5M = M07_잔액_카드론_B4M
    """
    res = df["M07_잔액_카드론_B4M"]

    c = df["M08_잔액_카드론_B5M"]
    return c == res


@constraint_udf
def cfs_05_0130(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_잔액_할부_B2M = M07_잔액_할부_B1M
    """
    res = df["M07_잔액_할부_B1M"]

    c = df["M08_잔액_할부_B2M"]
    return c == res


@constraint_udf
def cfs_05_0132(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_잔액_일시불_B2M = M07_잔액_일시불_B1M
    """
    res = df["M07_잔액_일시불_B1M"]

    c = df["M08_잔액_일시불_B2M"]
    return c == res


@constraint_udf
def cfs_05_0134(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_연체일수_B2M = M07_연체일수_B1M
    """
    res = df["M07_연체일수_B1M"]

    c = df["M08_연체일수_B2M"]
    return c == res


@constraint_udf
def cfs_05_0136(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_연체원금_B2M = M07_연체원금_B1M
    """
    res = df["M07_연체원금_B1M"]

    c = df["M08_연체원금_B2M"]
    return c == res


# 05_M09
@constraint_udf
def cfs_05_0215(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_잔액_카드론_B0M>0 THEN M09_카드론잔액_최종경과월 = 0
        ELIF (M09_잔액_카드론_B0M==0 & M08_잔액_카드론_B0M==0) THEN M09_카드론잔액_최종경과월 = 0
        ELIF (M09_잔액_카드론_B0M==0 & M08_잔액_카드론_B0M>0) THEN M09_카드론잔액_최종경과월 = 1
        ELSE M09_카드론잔액_최종경과월 = M08_카드론잔액_최종경과월 + 1
    """
    dd = df[["M09_잔액_카드론_B0M", "M08_잔액_카드론_B0M", "M08_카드론잔액_최종경과월"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[0] == 0) & (x[1] == 0)
                   else (1 if (x[0] == 0) & (x[1] > 0) else x[2] + 1)), axis=1)

    c = df["M09_카드론잔액_최종경과월"]
    return c == res


@constraint_udf
def cfs_05_0224(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_잔액_현금서비스_B2M = M08_잔액_현금서비스_B1M
    """
    res = df["M08_잔액_현금서비스_B1M"]

    c = df["M09_잔액_현금서비스_B2M"]
    return c == res


@constraint_udf
def cfs_05_0226(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_잔액_카드론_B2M = M08_잔액_카드론_B1M
    """
    res = df["M08_잔액_카드론_B1M"]

    c = df["M09_잔액_카드론_B2M"]
    return c == res


@constraint_udf
def cfs_05_0227(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_잔액_카드론_B3M = M08_잔액_카드론_B2M
    """
    res = df["M08_잔액_카드론_B2M"]

    c = df["M09_잔액_카드론_B3M"]
    return c == res


@constraint_udf
def cfs_05_0228(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_잔액_카드론_B4M = M08_잔액_카드론_B3M
    """
    res = df["M08_잔액_카드론_B3M"]

    c = df["M09_잔액_카드론_B4M"]
    return c == res


@constraint_udf
def cfs_05_0229(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_잔액_카드론_B5M = M08_잔액_카드론_B4M
    """
    res = df["M08_잔액_카드론_B4M"]

    c = df["M09_잔액_카드론_B5M"]
    return c == res


@constraint_udf
def cfs_05_0231(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_잔액_할부_B2M = M08_잔액_할부_B1M
    """
    res = df["M08_잔액_할부_B1M"]

    c = df["M09_잔액_할부_B2M"]
    return c == res


@constraint_udf
def cfs_05_0233(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_잔액_일시불_B2M = M08_잔액_일시불_B1M
    """
    res = df["M08_잔액_일시불_B1M"]

    c = df["M09_잔액_일시불_B2M"]
    return c == res


@constraint_udf
def cfs_05_0235(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_연체일수_B2M = M08_연체일수_B1M
    """
    res = df["M08_연체일수_B1M"]

    c = df["M09_연체일수_B2M"]
    return c == res


@constraint_udf
def cfs_05_0237(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_연체원금_B2M = M08_연체원금_B1M
    """
    res = df["M08_연체원금_B1M"]

    c = df["M09_연체원금_B2M"]
    return c == res


# 05_M10
@constraint_udf
def cfs_05_0316(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_잔액_카드론_B0M>0 THEN M10_카드론잔액_최종경과월 = 0
        ELIF (M10_잔액_카드론_B0M==0 & M09_잔액_카드론_B0M==0) THEN M10_카드론잔액_최종경과월 = 0
        ELIF (M10_잔액_카드론_B0M==0 & M09_잔액_카드론_B0M>0) THEN M10_카드론잔액_최종경과월 = 1
        ELSE M10_카드론잔액_최종경과월 = M09_카드론잔액_최종경과월 + 1
    """
    dd = df[["M10_잔액_카드론_B0M", "M09_잔액_카드론_B0M", "M09_카드론잔액_최종경과월"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[0] == 0) & (x[1] == 0)
                   else (1 if (x[0] == 0) & (x[1] > 0) else x[2] + 1)), axis=1)

    c = df["M10_카드론잔액_최종경과월"]
    return c == res


@constraint_udf
def cfs_05_0325(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_잔액_현금서비스_B2M = M09_잔액_현금서비스_B1M
    """
    res = df["M09_잔액_현금서비스_B1M"]

    c = df["M10_잔액_현금서비스_B2M"]
    return c == res


@constraint_udf
def cfs_05_0327(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_잔액_카드론_B2M = M09_잔액_카드론_B1M
    """
    res = df["M09_잔액_카드론_B1M"]

    c = df["M10_잔액_카드론_B2M"]
    return c == res


@constraint_udf
def cfs_05_0328(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_잔액_카드론_B3M = M09_잔액_카드론_B2M
    """
    res = df["M09_잔액_카드론_B2M"]

    c = df["M10_잔액_카드론_B3M"]
    return c == res


@constraint_udf
def cfs_05_0329(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_잔액_카드론_B4M = M09_잔액_카드론_B3M
    """
    res = df["M09_잔액_카드론_B3M"]

    c = df["M10_잔액_카드론_B4M"]
    return c == res


@constraint_udf
def cfs_05_0330(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_잔액_카드론_B5M = M09_잔액_카드론_B4M
    """
    res = df["M09_잔액_카드론_B4M"]

    c = df["M10_잔액_카드론_B5M"]
    return c == res


@constraint_udf
def cfs_05_0332(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_잔액_할부_B2M = M09_잔액_할부_B1M
    """
    res = df["M09_잔액_할부_B1M"]

    c = df["M10_잔액_할부_B2M"]
    return c == res


@constraint_udf
def cfs_05_0334(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_잔액_일시불_B2M = M09_잔액_일시불_B1M
    """
    res = df["M09_잔액_일시불_B1M"]

    c = df["M10_잔액_일시불_B2M"]
    return c == res


@constraint_udf
def cfs_05_0336(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_연체일수_B2M = M09_연체일수_B1M
    """
    res = df["M09_연체일수_B1M"]

    c = df["M10_연체일수_B2M"]
    return c == res


@constraint_udf
def cfs_05_0338(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_연체원금_B2M = M09_연체원금_B1M
    """
    res = df["M09_연체원금_B1M"]

    c = df["M10_연체원금_B2M"]
    return c == res


@constraint_udf
def cfs_05_0346(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_RV_평균잔액_R6M = avg(M10_RV_평균잔액_R3M, M07_RV_평균잔액_R3M)
    """
    dd = df[["M10_RV_평균잔액_R3M", "M07_RV_평균잔액_R3M"]]
    res = dd.mean(axis=1).astype(int)

    c = df["M10_RV_평균잔액_R6M"]
    return c == res


@constraint_udf
def cfs_05_0347(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_RV_최대잔액_R6M = max(M10_RV_최대잔액_R3M, M07_RV_최대잔액_R3M)
    """
    dd = df[["M10_RV_최대잔액_R3M", "M07_RV_최대잔액_R3M"]]
    res = dd.max(axis=1).astype(int)

    c = df["M10_RV_최대잔액_R6M"]
    return c == res


@constraint_udf
def cfs_05_0357(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_RV잔액이월횟수_R6M = M10_RV잔액이월횟수_R3M + M07_RV잔액이월횟수_R3M
    """
    dd = df[["M10_RV잔액이월횟수_R3M", "M07_RV잔액이월횟수_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_RV잔액이월횟수_R6M"]
    return c == res


# 05_M11
@constraint_udf
def cfs_05_0417(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_잔액_카드론_B0M>0 THEN M11_카드론잔액_최종경과월 = 0
        ELIF (M11_잔액_카드론_B0M==0 & M10_잔액_카드론_B0M==0) THEN M11_카드론잔액_최종경과월 = 0
        ELIF (M11_잔액_카드론_B0M==0 & M10_잔액_카드론_B0M>0) THEN M11_카드론잔액_최종경과월 = 1
        ELSE M11_카드론잔액_최종경과월 = M10_카드론잔액_최종경과월 + 1
    """
    dd = df[["M11_잔액_카드론_B0M", "M10_잔액_카드론_B0M", "M10_카드론잔액_최종경과월"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[0] == 0) & (x[1] == 0)
                   else (1 if (x[0] == 0) & (x[1] > 0) else x[2] + 1)), axis=1)

    c = df["M11_카드론잔액_최종경과월"]
    return c == res


@constraint_udf
def cfs_05_0426(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_잔액_현금서비스_B2M = M10_잔액_현금서비스_B1M
    """
    res = df["M10_잔액_현금서비스_B1M"]

    c = df["M11_잔액_현금서비스_B2M"]
    return c == res


@constraint_udf
def cfs_05_0428(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_잔액_카드론_B2M = M10_잔액_카드론_B1M
    """
    res = df["M10_잔액_카드론_B1M"]

    c = df["M11_잔액_카드론_B2M"]
    return c == res


@constraint_udf
def cfs_05_0429(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_잔액_카드론_B3M = M10_잔액_카드론_B2M
    """
    res = df["M10_잔액_카드론_B2M"]

    c = df["M11_잔액_카드론_B3M"]
    return c == res


@constraint_udf
def cfs_05_0430(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_잔액_카드론_B4M = M10_잔액_카드론_B3M
    """
    res = df["M10_잔액_카드론_B3M"]

    c = df["M11_잔액_카드론_B4M"]
    return c == res


@constraint_udf
def cfs_05_0431(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_잔액_카드론_B5M = M10_잔액_카드론_B4M
    """
    res = df["M10_잔액_카드론_B4M"]

    c = df["M11_잔액_카드론_B5M"]
    return c == res


@constraint_udf
def cfs_05_0433(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_잔액_할부_B2M = M10_잔액_할부_B1M
    """
    res = df["M10_잔액_할부_B1M"]

    c = df["M11_잔액_할부_B2M"]
    return c == res


@constraint_udf
def cfs_05_0435(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_잔액_일시불_B2M = M10_잔액_일시불_B1M
    """
    res = df["M10_잔액_일시불_B1M"]

    c = df["M11_잔액_일시불_B2M"]
    return c == res


@constraint_udf
def cfs_05_0437(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_연체일수_B2M = M10_연체일수_B1M
    """
    res = df["M10_연체일수_B1M"]

    c = df["M11_연체일수_B2M"]
    return c == res


@constraint_udf
def cfs_05_0439(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_연체원금_B2M = M10_연체원금_B1M
    """
    res = df["M10_연체원금_B1M"]

    c = df["M11_연체원금_B2M"]
    return c == res


@constraint_udf
def cfs_05_0447(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_RV_평균잔액_R6M = avg(M11_RV_평균잔액_R3M, M08_RV_평균잔액_R3M)
    """
    dd = df[["M11_RV_평균잔액_R3M", "M08_RV_평균잔액_R3M"]]
    res = dd.mean(axis=1).astype(int)

    c = df["M11_RV_평균잔액_R6M"]
    return c == res


@constraint_udf
def cfs_05_0448(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_RV_최대잔액_R6M = max(M11_RV_최대잔액_R3M, M08_RV_최대잔액_R3M)
    """
    dd = df[["M11_RV_최대잔액_R3M", "M08_RV_최대잔액_R3M"]]
    res = dd.max(axis=1).astype(int)

    c = df["M11_RV_최대잔액_R6M"]
    return c == res


@constraint_udf
def cfs_05_0458(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_RV잔액이월횟수_R6M = M11_RV잔액이월횟수_R3M + M08_RV잔액이월횟수_R3M
    """
    dd = df[["M11_RV잔액이월횟수_R3M", "M08_RV잔액이월횟수_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_RV잔액이월횟수_R6M"]
    return c == res


# 05_M12
@constraint_udf
def cfs_05_0518(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_잔액_카드론_B0M>0 THEN M12_카드론잔액_최종경과월 = 0
        ELIF (M12_잔액_카드론_B0M==0 & M11_잔액_카드론_B0M==0) THEN M12_카드론잔액_최종경과월 = 0
        ELIF (M12_잔액_카드론_B0M==0 & M11_잔액_카드론_B0M>0) THEN M12_카드론잔액_최종경과월 = 1
        ELSE M12_카드론잔액_최종경과월 = M11_카드론잔액_최종경과월 + 1
    """
    dd = df[["M12_잔액_카드론_B0M", "M11_잔액_카드론_B0M", "M11_카드론잔액_최종경과월"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[0] == 0) & (x[1] == 0)
                   else (1 if (x[0] == 0) & (x[1] > 0) else x[2] + 1)), axis=1)

    c = df["M12_카드론잔액_최종경과월"]
    return c == res


@constraint_udf
def cfs_05_0527(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_잔액_현금서비스_B2M = M11_잔액_현금서비스_B1M
    """
    res = df["M11_잔액_현금서비스_B1M"]

    c = df["M12_잔액_현금서비스_B2M"]
    return c == res


@constraint_udf
def cfs_05_0529(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_잔액_카드론_B2M = M11_잔액_카드론_B1M
    """
    res = df["M11_잔액_카드론_B1M"]

    c = df["M12_잔액_카드론_B2M"]
    return c == res


@constraint_udf
def cfs_05_0530(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_잔액_카드론_B3M = M11_잔액_카드론_B2M
    """
    res = df["M11_잔액_카드론_B2M"]

    c = df["M12_잔액_카드론_B3M"]
    return c == res


@constraint_udf
def cfs_05_0531(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_잔액_카드론_B4M = M11_잔액_카드론_B3M
    """
    res = df["M11_잔액_카드론_B3M"]

    c = df["M12_잔액_카드론_B4M"]
    return c == res


@constraint_udf
def cfs_05_0532(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_잔액_카드론_B5M = M11_잔액_카드론_B4M
    """
    res = df["M11_잔액_카드론_B4M"]

    c = df["M12_잔액_카드론_B5M"]
    return c == res


@constraint_udf
def cfs_05_0534(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_잔액_할부_B2M = M11_잔액_할부_B1M
    """
    res = df["M11_잔액_할부_B1M"]

    c = df["M12_잔액_할부_B2M"]
    return c == res


@constraint_udf
def cfs_05_0536(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_잔액_일시불_B2M = M11_잔액_일시불_B1M
    """
    res = df["M11_잔액_일시불_B1M"]

    c = df["M12_잔액_일시불_B2M"]
    return c == res


@constraint_udf
def cfs_05_0538(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_연체일수_B2M = M11_연체일수_B1M
    """
    res = df["M11_연체일수_B1M"]

    c = df["M12_연체일수_B2M"]
    return c == res


@constraint_udf
def cfs_05_0540(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_연체원금_B2M = M11_연체원금_B1M
    """
    res = df["M11_연체원금_B1M"]

    c = df["M12_연체원금_B2M"]
    return c == res


@constraint_udf
def cfs_05_0548(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_RV_평균잔액_R6M = avg(M12_RV_평균잔액_R3M, M09_RV_평균잔액_R3M)
    """
    dd = df[["M12_RV_평균잔액_R3M", "M09_RV_평균잔액_R3M"]]
    res = dd.mean(axis=1).astype(int)

    c = df["M12_RV_평균잔액_R6M"]
    return c == res


@constraint_udf
def cfs_05_0549(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_RV_최대잔액_R6M = max(M12_RV_최대잔액_R3M, M09_RV_최대잔액_R3M)
    """
    dd = df[["M12_RV_최대잔액_R3M", "M09_RV_최대잔액_R3M"]]
    res = dd.max(axis=1).astype(int)

    c = df["M12_RV_최대잔액_R6M"]
    return c == res


@constraint_udf
def cfs_05_0559(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_RV잔액이월횟수_R6M = M12_RV잔액이월횟수_R3M + M09_RV잔액이월횟수_R3M
    """
    dd = df[["M12_RV잔액이월횟수_R3M", "M09_RV잔액이월횟수_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_RV잔액이월횟수_R6M"]
    return c == res


# 06_M07
@constraint_udf
def cfs_06_0007(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_인입횟수_ARS_B0M >0 THEN M07_인입후경과월_ARS = 0 ELSE pass
    """
    dd = df[["M07_인입횟수_ARS_B0M", "M07_인입후경과월_ARS"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else x[1], axis=1)

    c = df["M07_인입후경과월_ARS"]
    return c == res


@constraint_udf
def cfs_06_0014(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_방문횟수_PC_B0M >0 THEN M07_방문후경과월_PC_R6M = 0 ELSE pass
    """
    dd = df[["M07_방문횟수_PC_B0M", "M07_방문후경과월_PC_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else x[1], axis=1)

    c = df["M07_방문후경과월_PC_R6M"]
    return c == res


@constraint_udf
def cfs_06_0018(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_방문횟수_앱_B0M >0 THEN M07_방문후경과월_앱_R6M = 0 ELSE pass
    """
    dd = df[["M07_방문횟수_앱_B0M", "M07_방문후경과월_앱_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else x[1], axis=1)

    c = df["M07_방문후경과월_앱_R6M"]
    return c == res


@constraint_udf
def cfs_06_0022(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_방문횟수_모바일웹_B0M >0 THEN M07_방문후경과월_모바일웹_R6M = 0 ELSE pas
    """
    dd = df[["M07_방문횟수_모바일웹_B0M", "M07_방문후경과월_모바일웹_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else x[1], axis=1)

    c = df["M07_방문후경과월_모바일웹_R6M"]
    return c == res


@constraint_udf
def cfs_06_0033(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_인입횟수_IB_B0M >0 THEN M07_인입후경과월_IB_R6M = 0 ELSE pass
    """
    dd = df[["M07_인입횟수_IB_B0M", "M07_인입후경과월_IB_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else x[1], axis=1)

    c = df["M07_인입후경과월_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0040(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_인입불만횟수_IB_B0M >0 THEN M07_인입불만후경과월_IB_R6M = 0 ELSE pass
    """
    dd = df[["M07_인입불만횟수_IB_B0M", "M07_인입불만후경과월_IB_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else x[1], axis=1)

    c = df["M07_인입불만후경과월_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0092(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_불만제기건수_B0M >0 THEN M07_불만제기후경과월_R12M = 0 ELSE pass
    """
    dd = df[["M07_불만제기건수_B0M", "M07_불만제기후경과월_R12M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else x[1], axis=1)

    c = df["M07_불만제기후경과월_R12M"]
    return c == res


# 06_M08
@constraint_udf
def cfs_06_0114(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF (M08_인입횟수_ARS_B0M>0) THEN M08_인입후경과월_ARS = 0
        ELIF (M07_인입횟수_ARS_B0M==0)&(M07_인입후경과월_ARS==0) THEN M08_인입후경과월_ARS = 0
        ELIF (M07_인입횟수_ARS_B0M>0)&(M07_인입후경과월_ARS==0) THEN M08_인입후경과월_ARS = 1
        ELIF (M07_인입후경과월_ARS==5) THEN M08_인입후경과월_ARS = 0
        ELIF M08_인입후경과월_ARS = M07_인입후경과월_ARS + 1
    """
    dd = df[["M08_인입횟수_ARS_B0M", "M07_인입횟수_ARS_B0M", "M07_인입후경과월_ARS"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[1] == 0) & (x[2] == 0)
                   else (1 if (x[1] > 0) & (x[2] == 0)
                         else (0 if x[2] == 5 else x[2] + 1))), axis=1)

    c = df["M08_인입후경과월_ARS"]
    return c == res


@constraint_udf
def cfs_06_0121(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_방문횟수_PC_B0M > 0 THEN M08_방문후경과월_PC_R6M = 0
        ELIF M08_방문후경과월_PC_R6M = min(M07_방문후경과월_PC_R6M + 1, 6)
    """
    dd = df[["M08_방문횟수_PC_B0M", "M07_방문후경과월_PC_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_방문후경과월_PC_R6M"]
    return c == res


@constraint_udf
def cfs_06_0125(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_방문횟수_앱_B0M > 0 THEN M08_방문후경과월_앱_R6M = 0
        ELIF M08_방문후경과월_앱_R6M = min(M07_방문후경과월_앱_R6M + 1, 6)
    """
    dd = df[["M08_방문횟수_앱_B0M", "M07_방문후경과월_앱_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_방문후경과월_앱_R6M"]
    return c == res


@constraint_udf
def cfs_06_0129(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_방문횟수_모바일웹_B0M > 0 THEN M08_방문후경과월_모바일웹_R6M = 0
        ELIF M08_방문후경과월_모바일웹_R6M = min(M07_방문후경과월_모바일웹_R6M + 1, 6)
    """
    dd = df[["M08_방문횟수_모바일웹_B0M", "M07_방문후경과월_모바일웹_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_방문후경과월_모바일웹_R6M"]
    return c == res


@constraint_udf
def cfs_06_0140(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_인입횟수_IB_B0M > 0 THEN M08_인입후경과월_IB_R6M = 0
        ELIF M08_인입후경과월_IB_R6M = min(M07_인입후경과월_IB_R6M + 1, 6)
    """
    dd = df[["M08_인입횟수_IB_B0M", "M07_인입후경과월_IB_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_인입후경과월_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0147(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_인입불만횟수_IB_B0M > 0 THEN M08_인입불만후경과월_IB_R6M = 0
        ELIF M08_인입불만후경과월_IB_R6M = min(M07_인입불만후경과월_IB_R6M + 1, 6)
    """
    dd = df[["M08_인입불만횟수_IB_B0M", "M07_인입불만후경과월_IB_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_인입불만후경과월_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0199(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_불만제기건수_B0M > 0 THEN M08_불만제기후경과월_R12M = 0
        ELIF (M07_불만제기건수_B0M==0 & M07_불만제기후경과월_R12M==0) THEN M08_불만제기후경과월_R12M = 0
        ELIF (M07_불만제기건수_B0M==0 & M07_불만제기후경과월_R12M==12) THEN M08_불만제기후경과월_R12M = 12
        ELIF M08_불만제기후경과월_R12M = min(M07_불만제기후경과월_R12M + 1, 12)
    """
    dd = df[["M08_불만제기건수_B0M", "M07_불만제기건수_B0M", "M07_불만제기후경과월_R12M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[1] == 0) & (x[2] == 0)
                   else (12 if (x[1] == 0) & (x[2] == 12) else min(x[2] + 1, 12))), axis=1)

    c = df["M08_불만제기후경과월_R12M"]
    return c == res


# 06_M09
@constraint_udf
def cfs_06_0221(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF (M09_인입횟수_ARS_B0M>0) THEN M09_인입후경과월_ARS = 0
        ELIF (M08_인입횟수_ARS_B0M==0)&(M08_인입후경과월_ARS==0) THEN M09_인입후경과월_ARS = 0
        ELIF (M08_인입횟수_ARS_B0M>0)&(M08_인입후경과월_ARS==0) THEN M09_인입후경과월_ARS = 1
        ELIF (M08_인입후경과월_ARS==5) THEN M09_인입후경과월_ARS = 0
        ELIF M09_인입후경과월_ARS = M08_인입후경과월_ARS + 1
    """
    dd = df[["M09_인입횟수_ARS_B0M", "M08_인입횟수_ARS_B0M", "M08_인입후경과월_ARS"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[1] == 0) & (x[2] == 0)
                   else (1 if (x[1] > 0) & (x[2] == 0)
                         else (0 if x[2] == 5 else x[2] + 1))), axis=1)

    c = df["M09_인입후경과월_ARS"]
    return c == res


@constraint_udf
def cfs_06_0228(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_방문횟수_PC_B0M > 0 THEN M09_방문후경과월_PC_R6M = 0
        ELIF M09_방문후경과월_PC_R6M = min(M08_방문후경과월_PC_R6M + 1, 6)
    """
    dd = df[["M09_방문횟수_PC_B0M", "M08_방문후경과월_PC_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_방문후경과월_PC_R6M"]
    return c == res


@constraint_udf
def cfs_06_0232(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_방문횟수_앱_B0M > 0 THEN M09_방문후경과월_앱_R6M = 0
        ELIF M09_방문후경과월_앱_R6M = min(M08_방문후경과월_앱_R6M + 1, 6)
    """
    dd = df[["M09_방문횟수_앱_B0M", "M08_방문후경과월_앱_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_방문후경과월_앱_R6M"]
    return c == res


@constraint_udf
def cfs_06_0236(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_방문횟수_모바일웹_B0M > 0 THEN M09_방문후경과월_모바일웹_R6M = 0
        ELIF M09_방문후경과월_모바일웹_R6M = min(M08_방문후경과월_모바일웹_R6M + 1, 6)
    """
    dd = df[["M09_방문횟수_모바일웹_B0M", "M08_방문후경과월_모바일웹_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_방문후경과월_모바일웹_R6M"]
    return c == res


@constraint_udf
def cfs_06_0247(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_인입횟수_IB_B0M > 0 THEN M09_인입후경과월_IB_R6M = 0
        ELIF M09_인입후경과월_IB_R6M = min(M08_인입후경과월_IB_R6M + 1, 6)
    """
    dd = df[["M09_인입횟수_IB_B0M", "M08_인입후경과월_IB_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_인입후경과월_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0254(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_인입불만횟수_IB_B0M > 0 THEN M09_인입불만후경과월_IB_R6M = 0
        ELIF M09_인입불만후경과월_IB_R6M = min(M08_인입불만후경과월_IB_R6M + 1, 6)
    """
    dd = df[["M09_인입불만횟수_IB_B0M", "M08_인입불만후경과월_IB_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_인입불만후경과월_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0306(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_불만제기건수_B0M > 0 THEN M09_불만제기후경과월_R12M = 0
        ELIF (M08_불만제기건수_B0M==0 & M08_불만제기후경과월_R12M==0) THEN M09_불만제기후경과월_R12M = 0
        ELIF (M08_불만제기건수_B0M==0 & M08_불만제기후경과월_R12M==12) THEN M09_불만제기후경과월_R12M = 12
        ELIF M09_불만제기후경과월_R12M = min(M08_불만제기후경과월_R12M + 1, 12)
    """
    dd = df[["M09_불만제기건수_B0M", "M08_불만제기건수_B0M", "M08_불만제기후경과월_R12M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[1] == 0) & (x[2] == 0)
                   else (12 if (x[1] == 0) & (x[2] == 12) else min(x[2] + 1, 12))), axis=1)

    c = df["M09_불만제기후경과월_R12M"]
    return c == res


# 06_M10
@constraint_udf
def cfs_06_0328(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF (M10_인입횟수_ARS_B0M>0) THEN M10_인입후경과월_ARS = 0
        ELIF (M09_인입횟수_ARS_B0M==0)&(M09_인입후경과월_ARS==0) THEN M10_인입후경과월_ARS = 0
        ELIF (M09_인입횟수_ARS_B0M>0)&(M09_인입후경과월_ARS==0) THEN M10_인입후경과월_ARS = 1
        ELIF (M09_인입후경과월_ARS==5) THEN M10_인입후경과월_ARS = 0
        ELIF M10_인입후경과월_ARS = M09_인입후경과월_ARS + 1
    """
    dd = df[["M10_인입횟수_ARS_B0M", "M09_인입횟수_ARS_B0M", "M09_인입후경과월_ARS"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[1] == 0) & (x[2] == 0)
                   else (1 if (x[1] > 0) & (x[2] == 0)
                         else (0 if x[2] == 5 else x[2] + 1))), axis=1)

    c = df["M10_인입후경과월_ARS"]
    return c == res


@constraint_udf
def cfs_06_0335(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_방문횟수_PC_B0M > 0 THEN M10_방문후경과월_PC_R6M = 0
        ELIF M10_방문후경과월_PC_R6M = min(M09_방문후경과월_PC_R6M + 1, 6)
    """
    dd = df[["M10_방문횟수_PC_B0M", "M09_방문후경과월_PC_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_방문후경과월_PC_R6M"]
    return c == res


@constraint_udf
def cfs_06_0339(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_방문횟수_앱_B0M > 0 THEN M10_방문후경과월_앱_R6M = 0
        ELIF M10_방문후경과월_앱_R6M = min(M09_방문후경과월_앱_R6M + 1, 6)
    """
    dd = df[["M10_방문횟수_앱_B0M", "M09_방문후경과월_앱_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_방문후경과월_앱_R6M"]
    return c == res


@constraint_udf
def cfs_06_0343(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_방문횟수_모바일웹_B0M > 0 THEN M10_방문후경과월_모바일웹_R6M = 0
        ELIF M10_방문후경과월_모바일웹_R6M = min(M09_방문후경과월_모바일웹_R6M + 1, 6)
    """
    dd = df[["M10_방문횟수_모바일웹_B0M", "M09_방문후경과월_모바일웹_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_방문후경과월_모바일웹_R6M"]
    return c == res


@constraint_udf
def cfs_06_0354(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_인입횟수_IB_B0M > 0 THEN M10_인입후경과월_IB_R6M = 0
        ELIF M10_인입후경과월_IB_R6M = min(M09_인입후경과월_IB_R6M + 1, 6)
    """
    dd = df[["M10_인입횟수_IB_B0M", "M09_인입후경과월_IB_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_인입후경과월_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0361(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_인입불만횟수_IB_B0M > 0 THEN M10_인입불만후경과월_IB_R6M = 0
        ELIF M10_인입불만후경과월_IB_R6M = min(M09_인입불만후경과월_IB_R6M + 1, 6)
    """
    dd = df[["M10_인입불만횟수_IB_B0M", "M09_인입불만후경과월_IB_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_인입불만후경과월_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0413(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_불만제기건수_B0M > 0 THEN M10_불만제기후경과월_R12M = 0
        ELIF (M09_불만제기건수_B0M==0 & M09_불만제기후경과월_R12M==0) THEN M10_불만제기후경과월_R12M = 0
        ELIF (M09_불만제기건수_B0M==0 & M09_불만제기후경과월_R12M==12) THEN M10_불만제기후경과월_R12M = 12
        ELIF M10_불만제기후경과월_R12M = min(M09_불만제기후경과월_R12M + 1, 12)
    """
    dd = df[["M10_불만제기건수_B0M", "M09_불만제기건수_B0M", "M09_불만제기후경과월_R12M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[1] == 0) & (x[2] == 0)
                   else (12 if (x[1] == 0) & (x[2] == 12) else min(x[2] + 1, 12))), axis=1)

    c = df["M10_불만제기후경과월_R12M"]
    return c == res


@constraint_udf
def cfs_06_0423(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_홈페이지_금융건수_R6M = M10_홈페이지_금융건수_R3M + M07_홈페이지_금융건수_R3M
    """
    c1, c2 = df["M10_홈페이지_금융건수_R3M"], df["M07_홈페이지_금융건수_R3M"]
    res = c1 + c2

    c = df["M10_홈페이지_금융건수_R6M"]
    return c == res


@constraint_udf
def cfs_06_0424(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_홈페이지_선결제건수_R6M = M10_홈페이지_선결제건수_R3M + M07_홈페이지_선결제건수_R3M
    """
    c1, c2 = df["M10_홈페이지_선결제건수_R3M"], df["M07_홈페이지_선결제건수_R3M"]
    res = c1 + c2

    c = df["M10_홈페이지_선결제건수_R6M"]
    return c == res


# 06_M11
@constraint_udf
def cfs_06_0435(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF (M11_인입횟수_ARS_B0M>0) THEN M11_인입후경과월_ARS = 0
        ELIF (M10_인입횟수_ARS_B0M==0)&(M10_인입후경과월_ARS==0) THEN M11_인입후경과월_ARS = 0
        ELIF (M10_인입횟수_ARS_B0M>0)&(M10_인입후경과월_ARS==0) THEN M11_인입후경과월_ARS = 1
        ELIF (M10_인입후경과월_ARS==5) THEN M11_인입후경과월_ARS = 0
        ELIF M11_인입후경과월_ARS = M10_인입후경과월_ARS + 1
    """
    dd = df[["M11_인입횟수_ARS_B0M", "M10_인입횟수_ARS_B0M", "M10_인입후경과월_ARS"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[1] == 0) & (x[2] == 0)
                   else (1 if (x[1] > 0) & (x[2] == 0)
                         else (0 if x[2] == 5 else x[2] + 1))), axis=1)

    c = df["M11_인입후경과월_ARS"]
    return c == res


@constraint_udf
def cfs_06_0442(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_방문횟수_PC_B0M > 0 THEN M11_방문후경과월_PC_R6M = 0
        ELIF M11_방문후경과월_PC_R6M = min(M10_방문후경과월_PC_R6M + 1, 6)
    """
    dd = df[["M11_방문횟수_PC_B0M", "M10_방문후경과월_PC_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_방문후경과월_PC_R6M"]
    return c == res


@constraint_udf
def cfs_06_0446(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_방문횟수_앱_B0M > 0 THEN M11_방문후경과월_앱_R6M = 0
        ELIF M11_방문후경과월_앱_R6M = min(M10_방문후경과월_앱_R6M + 1, 6)
    """
    dd = df[["M11_방문횟수_앱_B0M", "M10_방문후경과월_앱_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_방문후경과월_앱_R6M"]
    return c == res


@constraint_udf
def cfs_06_0450(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_방문횟수_모바일웹_B0M > 0 THEN M11_방문후경과월_모바일웹_R6M = 0
        ELIF M11_방문후경과월_모바일웹_R6M = min(M10_방문후경과월_모바일웹_R6M + 1, 6)
    """
    dd = df[["M11_방문횟수_모바일웹_B0M", "M10_방문후경과월_모바일웹_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_방문후경과월_모바일웹_R6M"]
    return c == res


@constraint_udf
def cfs_06_0461(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_인입횟수_IB_B0M > 0 THEN M11_인입후경과월_IB_R6M = 0
        ELIF M11_인입후경과월_IB_R6M = min(M10_인입후경과월_IB_R6M + 1, 6)
    """
    dd = df[["M11_인입횟수_IB_B0M", "M10_인입후경과월_IB_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_인입후경과월_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0468(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_인입불만횟수_IB_B0M > 0 THEN M11_인입불만후경과월_IB_R6M = 0
        ELIF M11_인입불만후경과월_IB_R6M = min(M10_인입불만후경과월_IB_R6M + 1, 6)
    """
    dd = df[["M11_인입불만횟수_IB_B0M", "M10_인입불만후경과월_IB_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_인입불만후경과월_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0520(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_불만제기건수_B0M > 0 THEN M11_불만제기후경과월_R12M = 0
        ELIF (M10_불만제기건수_B0M==0 & M10_불만제기후경과월_R12M==0) THEN M11_불만제기후경과월_R12M = 0
        ELIF (M10_불만제기건수_B0M==0 & M10_불만제기후경과월_R12M==12) THEN M11_불만제기후경과월_R12M = 12
        ELIF M11_불만제기후경과월_R12M = min(M10_불만제기후경과월_R12M + 1, 12)
    """
    dd = df[["M11_불만제기건수_B0M", "M10_불만제기건수_B0M", "M10_불만제기후경과월_R12M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[1] == 0) & (x[2] == 0)
                   else (12 if (x[1] == 0) & (x[2] == 12) else min(x[2] + 1, 12))), axis=1)

    c = df["M11_불만제기후경과월_R12M"]
    return c == res


@constraint_udf
def cfs_06_0530(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_홈페이지_금융건수_R6M = M11_홈페이지_금융건수_R3M + M08_홈페이지_금융건수_R3M
    """
    c1, c2 = df["M11_홈페이지_금융건수_R3M"], df["M08_홈페이지_금융건수_R3M"]
    res = c1 + c2

    c = df["M11_홈페이지_금융건수_R6M"]
    return c == res


@constraint_udf
def cfs_06_0531(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_홈페이지_선결제건수_R6M = M11_홈페이지_선결제건수_R3M + M08_홈페이지_선결제건수_R3M
    """
    c1, c2 = df["M11_홈페이지_선결제건수_R3M"], df["M08_홈페이지_선결제건수_R3M"]
    res = c1 + c2

    c = df["M11_홈페이지_선결제건수_R6M"]
    return c == res


# 06_M12
@constraint_udf
def cfs_06_0542(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF (M12_인입횟수_ARS_B0M>0) THEN M12_인입후경과월_ARS = 0
        ELIF (M11_인입횟수_ARS_B0M==0)&(M11_인입후경과월_ARS==0) THEN M12_인입후경과월_ARS = 0
        ELIF (M11_인입횟수_ARS_B0M>0)&(M11_인입후경과월_ARS==0) THEN M12_인입후경과월_ARS = 1
        ELIF (M11_인입후경과월_ARS==5) THEN M12_인입후경과월_ARS = 0
        ELIF M12_인입후경과월_ARS = M11_인입후경과월_ARS + 1
    """
    dd = df[["M12_인입횟수_ARS_B0M", "M11_인입횟수_ARS_B0M", "M11_인입후경과월_ARS"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[1] == 0) & (x[2] == 0)
                   else (1 if (x[1] > 0) & (x[2] == 0)
                         else (0 if x[2] == 5 else x[2] + 1))), axis=1)

    c = df["M12_인입후경과월_ARS"]
    return c == res


@constraint_udf
def cfs_06_0549(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_방문횟수_PC_B0M > 0 THEN M12_방문후경과월_PC_R6M = 0
        ELIF M12_방문후경과월_PC_R6M = min(M11_방문후경과월_PC_R6M + 1, 6)
    """
    dd = df[["M12_방문횟수_PC_B0M", "M11_방문후경과월_PC_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_방문후경과월_PC_R6M"]
    return c == res


@constraint_udf
def cfs_06_0553(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_방문횟수_앱_B0M > 0 THEN M12_방문후경과월_앱_R6M = 0
        ELIF M12_방문후경과월_앱_R6M = min(M11_방문후경과월_앱_R6M + 1, 6)
    """
    dd = df[["M12_방문횟수_앱_B0M", "M11_방문후경과월_앱_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_방문후경과월_앱_R6M"]
    return c == res


@constraint_udf
def cfs_06_0557(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_방문횟수_모바일웹_B0M > 0 THEN M12_방문후경과월_모바일웹_R6M = 0
        ELIF M12_방문후경과월_모바일웹_R6M = min(M11_방문후경과월_모바일웹_R6M + 1, 6)
    """
    dd = df[["M12_방문횟수_모바일웹_B0M", "M11_방문후경과월_모바일웹_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_방문후경과월_모바일웹_R6M"]
    return c == res


@constraint_udf
def cfs_06_0568(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_인입횟수_IB_B0M > 0 THEN M12_인입후경과월_IB_R6M = 0
        ELIF M12_인입후경과월_IB_R6M = min(M11_인입후경과월_IB_R6M + 1, 6)
    """
    dd = df[["M12_인입횟수_IB_B0M", "M11_인입후경과월_IB_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_인입후경과월_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0575(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_인입불만횟수_IB_B0M > 0 THEN M12_인입불만후경과월_IB_R6M = 0
        ELIF M12_인입불만후경과월_IB_R6M = min(M11_인입불만후경과월_IB_R6M + 1, 6)
    """
    dd = df[["M12_인입불만횟수_IB_B0M", "M11_인입불만후경과월_IB_R6M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_인입불만후경과월_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0627(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_불만제기건수_B0M > 0 THEN M12_불만제기후경과월_R12M = 0
        ELIF (M11_불만제기건수_B0M==0 & M11_불만제기후경과월_R12M==0) THEN M12_불만제기후경과월_R12M = 0
        ELIF (M11_불만제기건수_B0M==0 & M11_불만제기후경과월_R12M==12) THEN M12_불만제기후경과월_R12M = 12
        ELIF M12_불만제기후경과월_R12M = min(M11_불만제기후경과월_R12M + 1, 12)
    """
    dd = df[["M12_불만제기건수_B0M", "M11_불만제기건수_B0M", "M11_불만제기후경과월_R12M"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (0 if (x[1] == 0) & (x[2] == 0) else (12 if (x[1] == 0) & (x[2] == 12) else min(x[2] + 1, 12))), axis=1)

    c = df["M12_불만제기후경과월_R12M"]
    return c == res


@constraint_udf
def cfs_06_0538(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_인입횟수_ARS_R6M = SUM(M07~M12_인입횟수_ARS_B0M)
    """
    dd = df[["M07_인입횟수_ARS_B0M",
             "M08_인입횟수_ARS_B0M",
             "M09_인입횟수_ARS_B0M",
             "M10_인입횟수_ARS_B0M",
             "M11_인입횟수_ARS_B0M",
             "M12_인입횟수_ARS_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_인입횟수_ARS_R6M"]
    return c == res


@constraint_udf
def cfs_06_0539(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용메뉴건수_ARS_R6M = SUM(M07~M12_이용메뉴건수_ARS_B0M)
    """
    dd = df[["M07_이용메뉴건수_ARS_B0M",
             "M08_이용메뉴건수_ARS_B0M",
             "M09_이용메뉴건수_ARS_B0M",
             "M10_이용메뉴건수_ARS_B0M",
             "M11_이용메뉴건수_ARS_B0M",
             "M12_이용메뉴건수_ARS_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용메뉴건수_ARS_R6M"]
    return c == res


@constraint_udf
def cfs_06_0540(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_인입일수_ARS_R6M = SUM(M07~M12_인입일수_ARS_B0M)
    """
    dd = df[["M07_인입일수_ARS_B0M",
             "M08_인입일수_ARS_B0M",
             "M09_인입일수_ARS_B0M",
             "M10_인입일수_ARS_B0M",
             "M11_인입일수_ARS_B0M",
             "M12_인입일수_ARS_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_인입일수_ARS_R6M"]
    return c == res


@constraint_udf
def cfs_06_0546(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_방문횟수_PC_R6M = SUM(M07~M12_방문횟수_PC_B0M)
    """
    dd = df[["M07_방문횟수_PC_B0M",
             "M08_방문횟수_PC_B0M",
             "M09_방문횟수_PC_B0M",
             "M10_방문횟수_PC_B0M",
             "M11_방문횟수_PC_B0M",
             "M12_방문횟수_PC_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_방문횟수_PC_R6M"]
    return c == res


@constraint_udf
def cfs_06_0547(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_방문일수_PC_R6M = SUM(M07~M12_방문일수_PC_B0M)
    """
    dd = df[["M07_방문일수_PC_B0M",
             "M08_방문일수_PC_B0M",
             "M09_방문일수_PC_B0M",
             "M10_방문일수_PC_B0M",
             "M11_방문일수_PC_B0M",
             "M12_방문일수_PC_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_방문일수_PC_R6M"]
    return c == res


@constraint_udf
def cfs_06_0550(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_방문횟수_앱_R6M = SUM(M07~M12_방문횟수_앱_B0M)
    """
    dd = df[["M07_방문횟수_앱_B0M",
             "M08_방문횟수_앱_B0M",
             "M09_방문횟수_앱_B0M",
             "M10_방문횟수_앱_B0M",
             "M11_방문횟수_앱_B0M",
             "M12_방문횟수_앱_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_방문횟수_앱_R6M"]
    return c == res


@constraint_udf
def cfs_06_0551(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_방문일수_앱_R6M = SUM(M07~M12_방문일수_앱_B0M)
    """
    dd = df[["M07_방문일수_앱_B0M",
             "M08_방문일수_앱_B0M",
             "M09_방문일수_앱_B0M",
             "M10_방문일수_앱_B0M",
             "M11_방문일수_앱_B0M",
             "M12_방문일수_앱_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_방문일수_앱_R6M"]
    return c == res


@constraint_udf
def cfs_06_0554(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_방문횟수_모바일웹_R6M = SUM(M07~M12_방문횟수_모바일웹_B0M)
    """
    dd = df[["M07_방문횟수_모바일웹_B0M",
             "M08_방문횟수_모바일웹_B0M",
             "M09_방문횟수_모바일웹_B0M",
             "M10_방문횟수_모바일웹_B0M",
             "M11_방문횟수_모바일웹_B0M",
             "M12_방문횟수_모바일웹_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_방문횟수_모바일웹_R6M"]
    return c == res


@constraint_udf
def cfs_06_0555(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_방문일수_모바일웹_R6M = SUM(M07~M12_방문일수_모바일웹_B0M)
    """
    dd = df[["M07_방문일수_모바일웹_B0M",
             "M08_방문일수_모바일웹_B0M",
             "M09_방문일수_모바일웹_B0M",
             "M10_방문일수_모바일웹_B0M",
             "M11_방문일수_모바일웹_B0M",
             "M12_방문일수_모바일웹_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_방문일수_모바일웹_R6M"]
    return c == res


@constraint_udf
def cfs_06_0564(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_인입횟수_IB_R6M = SUM(M07~M12_인입횟수_IB_B0M)
    """
    dd = df[["M07_인입횟수_IB_B0M",
             "M08_인입횟수_IB_B0M",
             "M09_인입횟수_IB_B0M",
             "M10_인입횟수_IB_B0M",
             "M11_인입횟수_IB_B0M",
             "M12_인입횟수_IB_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_인입횟수_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0565(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_인입일수_IB_R6M = SUM(M07~M12_인입일수_IB_B0M)
    """
    dd = df[["M07_인입일수_IB_B0M",
             "M08_인입일수_IB_B0M",
             "M09_인입일수_IB_B0M",
             "M10_인입일수_IB_B0M",
             "M11_인입일수_IB_B0M",
             "M12_인입일수_IB_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_인입일수_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0567(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용메뉴건수_IB_R6M = SUM(M07~M12_이용메뉴건수_IB_B0M)
    """
    dd = df[["M07_이용메뉴건수_IB_B0M",
             "M08_이용메뉴건수_IB_B0M",
             "M09_이용메뉴건수_IB_B0M",
             "M10_이용메뉴건수_IB_B0M",
             "M11_이용메뉴건수_IB_B0M",
             "M12_이용메뉴건수_IB_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용메뉴건수_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0572(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_인입불만횟수_IB_R6M = SUM(M07~M12_인입불만횟수_IB_B0M)
    """
    dd = df[["M07_인입불만횟수_IB_B0M",
             "M08_인입불만횟수_IB_B0M",
             "M09_인입불만횟수_IB_B0M",
             "M10_인입불만횟수_IB_B0M",
             "M11_인입불만횟수_IB_B0M",
             "M12_인입불만횟수_IB_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_인입불만횟수_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0573(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_인입불만일수_IB_R6M = SUM(M07~M12_인입불만일수_IB_B0M)
    """
    dd = df[["M07_인입불만일수_IB_B0M",
             "M08_인입불만일수_IB_B0M",
             "M09_인입불만일수_IB_B0M",
             "M10_인입불만일수_IB_B0M",
             "M11_인입불만일수_IB_B0M",
             "M12_인입불만일수_IB_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_인입불만일수_IB_R6M"]
    return c == res


@constraint_udf
def cfs_06_0602(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_사용승인내역_R6M = SUM(M07~M12_IB문의건수_사용승인내역_B0M)
    """
    dd = df[["M07_IB문의건수_사용승인내역_B0M",
             "M08_IB문의건수_사용승인내역_B0M",
             "M09_IB문의건수_사용승인내역_B0M",
             "M10_IB문의건수_사용승인내역_B0M",
             "M11_IB문의건수_사용승인내역_B0M",
             "M12_IB문의건수_사용승인내역_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_사용승인내역_R6M"]
    return c == res


@constraint_udf
def cfs_06_0603(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_한도_R6M = SUM(M07~M12_IB문의건수_한도_B0M)
    """
    dd = df[["M07_IB문의건수_한도_B0M",
             "M08_IB문의건수_한도_B0M",
             "M09_IB문의건수_한도_B0M",
             "M10_IB문의건수_한도_B0M",
             "M11_IB문의건수_한도_B0M",
             "M12_IB문의건수_한도_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_한도_R6M"]
    return c == res


@constraint_udf
def cfs_06_0604(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_선결제_R6M = SUM(M07~M12_IB문의건수_선결제_B0M)
    """
    dd = df[["M07_IB문의건수_선결제_B0M",
             "M08_IB문의건수_선결제_B0M",
             "M09_IB문의건수_선결제_B0M",
             "M10_IB문의건수_선결제_B0M",
             "M11_IB문의건수_선결제_B0M",
             "M12_IB문의건수_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_선결제_R6M"]
    return c == res


@constraint_udf
def cfs_06_0605(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_결제_R6M = SUM(M07~M12_IB문의건수_결제_B0M)
    """
    dd = df[["M07_IB문의건수_결제_B0M",
             "M08_IB문의건수_결제_B0M",
             "M09_IB문의건수_결제_B0M",
             "M10_IB문의건수_결제_B0M",
             "M11_IB문의건수_결제_B0M",
             "M12_IB문의건수_결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_결제_R6M"]
    return c == res


@constraint_udf
def cfs_06_0606(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_할부_R6M = SUM(M07~M12_IB문의건수_할부_B0M)
    """
    dd = df[["M07_IB문의건수_할부_B0M",
             "M08_IB문의건수_할부_B0M",
             "M09_IB문의건수_할부_B0M",
             "M10_IB문의건수_할부_B0M",
             "M11_IB문의건수_할부_B0M",
             "M12_IB문의건수_할부_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_할부_R6M"]
    return c == res


@constraint_udf
def cfs_06_0607(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_정보변경_R6M = SUM(M07~M12_IB문의건수_정보변경_B0M)
    """
    dd = df[["M07_IB문의건수_정보변경_B0M",
             "M08_IB문의건수_정보변경_B0M",
             "M09_IB문의건수_정보변경_B0M",
             "M10_IB문의건수_정보변경_B0M",
             "M11_IB문의건수_정보변경_B0M",
             "M12_IB문의건수_정보변경_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_정보변경_R6M"]
    return c == res


@constraint_udf
def cfs_06_0608(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_결제일변경_R6M = SUM(M07~M12_IB문의건수_결제일변경_B0M)
    """
    dd = df[["M07_IB문의건수_결제일변경_B0M",
             "M08_IB문의건수_결제일변경_B0M",
             "M09_IB문의건수_결제일변경_B0M",
             "M10_IB문의건수_결제일변경_B0M",
             "M11_IB문의건수_결제일변경_B0M",
             "M12_IB문의건수_결제일변경_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_결제일변경_R6M"]
    return c == res


@constraint_udf
def cfs_06_0609(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_명세서_R6M = SUM(M07~M12_IB문의건수_명세서_B0M)
    """
    dd = df[["M07_IB문의건수_명세서_B0M",
             "M08_IB문의건수_명세서_B0M",
             "M09_IB문의건수_명세서_B0M",
             "M10_IB문의건수_명세서_B0M",
             "M11_IB문의건수_명세서_B0M",
             "M12_IB문의건수_명세서_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_명세서_R6M"]
    return c == res


@constraint_udf
def cfs_06_0610(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_비밀번호_R6M = SUM(M07~M12_IB문의건수_비밀번호_B0M)
    """
    dd = df[["M07_IB문의건수_비밀번호_B0M",
             "M08_IB문의건수_비밀번호_B0M",
             "M09_IB문의건수_비밀번호_B0M",
             "M10_IB문의건수_비밀번호_B0M",
             "M11_IB문의건수_비밀번호_B0M",
             "M12_IB문의건수_비밀번호_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_비밀번호_R6M"]
    return c == res


@constraint_udf
def cfs_06_0611(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_SMS_R6M = SUM(M07~M12_IB문의건수_SMS_B0M)
    """
    dd = df[["M07_IB문의건수_SMS_B0M",
             "M08_IB문의건수_SMS_B0M",
             "M09_IB문의건수_SMS_B0M",
             "M10_IB문의건수_SMS_B0M",
             "M11_IB문의건수_SMS_B0M",
             "M12_IB문의건수_SMS_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_SMS_R6M"]
    return c == res


@constraint_udf
def cfs_06_0612(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_APP_R6M = SUM(M07~M12_IB문의건수_APP_B0M)
    """
    dd = df[["M07_IB문의건수_APP_B0M",
             "M08_IB문의건수_APP_B0M",
             "M09_IB문의건수_APP_B0M",
             "M10_IB문의건수_APP_B0M",
             "M11_IB문의건수_APP_B0M",
             "M12_IB문의건수_APP_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_APP_R6M"]
    return c == res


@constraint_udf
def cfs_06_0613(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_부대서비스_R6M = SUM(M07~M12_IB문의건수_부대서비스_B0M)
    """
    dd = df[["M07_IB문의건수_부대서비스_B0M",
             "M08_IB문의건수_부대서비스_B0M",
             "M09_IB문의건수_부대서비스_B0M",
             "M10_IB문의건수_부대서비스_B0M",
             "M11_IB문의건수_부대서비스_B0M",
             "M12_IB문의건수_부대서비스_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_부대서비스_R6M"]
    return c == res


@constraint_udf
def cfs_06_0614(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_포인트_R6M = SUM(M07~M12_IB문의건수_포인트_B0M)
    """
    dd = df[["M07_IB문의건수_포인트_B0M",
             "M08_IB문의건수_포인트_B0M",
             "M09_IB문의건수_포인트_B0M",
             "M10_IB문의건수_포인트_B0M",
             "M11_IB문의건수_포인트_B0M",
             "M12_IB문의건수_포인트_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_포인트_R6M"]
    return c == res


@constraint_udf
def cfs_06_0615(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_카드발급_R6M = SUM(M07~M12_IB문의건수_카드발급_B0M)
    """
    dd = df[["M07_IB문의건수_카드발급_B0M",
             "M08_IB문의건수_카드발급_B0M",
             "M09_IB문의건수_카드발급_B0M",
             "M10_IB문의건수_카드발급_B0M",
             "M11_IB문의건수_카드발급_B0M",
             "M12_IB문의건수_카드발급_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_카드발급_R6M"]
    return c == res


@constraint_udf
def cfs_06_0616(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_BL_R6M = SUM(M07~M12_IB문의건수_BL_B0M)
    """
    dd = df[["M07_IB문의건수_BL_B0M",
             "M08_IB문의건수_BL_B0M",
             "M09_IB문의건수_BL_B0M",
             "M10_IB문의건수_BL_B0M",
             "M11_IB문의건수_BL_B0M",
             "M12_IB문의건수_BL_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_BL_R6M"]
    return c == res


@constraint_udf
def cfs_06_0617(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_분실도난_R6M = SUM(M07~M12_IB문의건수_분실도난_B0M)
    """
    dd = df[["M07_IB문의건수_분실도난_B0M",
             "M08_IB문의건수_분실도난_B0M",
             "M09_IB문의건수_분실도난_B0M",
             "M10_IB문의건수_분실도난_B0M",
             "M11_IB문의건수_분실도난_B0M",
             "M12_IB문의건수_분실도난_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_분실도난_R6M"]
    return c == res


@constraint_udf
def cfs_06_0618(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_CA_R6M = SUM(M07~M12_IB문의건수_CA_B0M)
    """
    dd = df[["M07_IB문의건수_CA_B0M",
             "M08_IB문의건수_CA_B0M",
             "M09_IB문의건수_CA_B0M",
             "M10_IB문의건수_CA_B0M",
             "M11_IB문의건수_CA_B0M",
             "M12_IB문의건수_CA_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_CA_R6M"]
    return c == res


@constraint_udf
def cfs_06_0619(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_CL_RV_R6M = SUM(M07~M12_IB문의건수_CL_RV_B0M)
    """
    dd = df[["M07_IB문의건수_CL_RV_B0M",
             "M08_IB문의건수_CL_RV_B0M",
             "M09_IB문의건수_CL_RV_B0M",
             "M10_IB문의건수_CL_RV_B0M",
             "M11_IB문의건수_CL_RV_B0M",
             "M12_IB문의건수_CL_RV_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_CL_RV_R6M"]
    return c == res


@constraint_udf
def cfs_06_0620(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB문의건수_CS_R6M = SUM(M07~M12_IB문의건수_CS_B0M)
    """
    dd = df[["M07_IB문의건수_CS_B0M",
             "M08_IB문의건수_CS_B0M",
             "M09_IB문의건수_CS_B0M",
             "M10_IB문의건수_CS_B0M",
             "M11_IB문의건수_CS_B0M",
             "M12_IB문의건수_CS_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB문의건수_CS_R6M"]
    return c == res


@constraint_udf
def cfs_06_0621(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB상담건수_VOC_R6M = SUM(M07~M12_IB상담건수_VOC_B0M)
    """
    dd = df[["M07_IB상담건수_VOC_B0M",
             "M08_IB상담건수_VOC_B0M",
             "M09_IB상담건수_VOC_B0M",
             "M10_IB상담건수_VOC_B0M",
             "M11_IB상담건수_VOC_B0M",
             "M12_IB상담건수_VOC_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB상담건수_VOC_R6M"]
    return c == res


@constraint_udf
def cfs_06_0622(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB상담건수_VOC민원_R6M = SUM(M07~M12_IB상담건수_VOC민원_B0M)
    """
    dd = df[["M07_IB상담건수_VOC민원_B0M",
             "M08_IB상담건수_VOC민원_B0M",
             "M09_IB상담건수_VOC민원_B0M",
             "M10_IB상담건수_VOC민원_B0M",
             "M11_IB상담건수_VOC민원_B0M",
             "M12_IB상담건수_VOC민원_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB상담건수_VOC민원_R6M"]
    return c == res


@constraint_udf
def cfs_06_0623(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB상담건수_VOC불만_R6M = SUM(M07~M12_IB상담건수_VOC불만_B0M)
    """
    dd = df[["M07_IB상담건수_VOC불만_B0M",
             "M08_IB상담건수_VOC불만_B0M",
             "M09_IB상담건수_VOC불만_B0M",
             "M10_IB상담건수_VOC불만_B0M",
             "M11_IB상담건수_VOC불만_B0M",
             "M12_IB상담건수_VOC불만_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB상담건수_VOC불만_R6M"]
    return c == res


@constraint_udf
def cfs_06_0624(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_IB상담건수_금감원_R6M = SUM(M07~M12_IB상담건수_금감원_B0M)
    """
    dd = df[["M07_IB상담건수_금감원_B0M",
             "M08_IB상담건수_금감원_B0M",
             "M09_IB상담건수_금감원_B0M",
             "M10_IB상담건수_금감원_B0M",
             "M11_IB상담건수_금감원_B0M",
             "M12_IB상담건수_금감원_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_IB상담건수_금감원_R6M"]
    return c == res


@constraint_udf
def cfs_06_0629(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_당사PAY_방문횟수_R6M = SUM(M07~M12_당사PAY_방문횟수_B0M)
    """
    dd = df[["M07_당사PAY_방문횟수_B0M",
             "M08_당사PAY_방문횟수_B0M",
             "M09_당사PAY_방문횟수_B0M",
             "M10_당사PAY_방문횟수_B0M",
             "M11_당사PAY_방문횟수_B0M",
             "M12_당사PAY_방문횟수_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_당사PAY_방문횟수_R6M"]
    return c == res


@constraint_udf
def cfs_06_0632(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_당사멤버쉽_방문횟수_R6M = SUM(M07~M12_당사멤버쉽_방문횟수_B0M)
    """
    dd = df[["M07_당사멤버쉽_방문횟수_B0M",
             "M08_당사멤버쉽_방문횟수_B0M",
             "M09_당사멤버쉽_방문횟수_B0M",
             "M10_당사멤버쉽_방문횟수_B0M",
             "M11_당사멤버쉽_방문횟수_B0M",
             "M12_당사멤버쉽_방문횟수_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_당사멤버쉽_방문횟수_R6M"]
    return c == res


@constraint_udf
def cfs_06_0637(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_홈페이지_금융건수_R6M = M12_홈페이지_금융건수_R3M + M09_홈페이지_금융건수_R3M
    """
    c1, c2 = df["M12_홈페이지_금융건수_R3M"], df["M09_홈페이지_금융건수_R3M"]
    res = c1 + c2

    c = df["M12_홈페이지_금융건수_R6M"]
    return c == res


@constraint_udf
def cfs_06_0638(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_홈페이지_선결제건수_R6M = M12_홈페이지_선결제건수_R3M + M09_홈페이지_선결제건수_R3M
    """
    c1, c2 = df["M12_홈페이지_선결제건수_R3M"], df["M09_홈페이지_선결제건수_R3M"]
    res = c1 + c2

    c = df["M12_홈페이지_선결제건수_R6M"]
    return c == res


# 07_M12
@constraint_udf
def cfs_07_0351(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_카드론_TM_R6M = SUM(M07~M12_컨택건수_카드론_TM_B0M)
    """
    dd = df[["M07_컨택건수_카드론_TM_B0M",
             "M08_컨택건수_카드론_TM_B0M",
             "M09_컨택건수_카드론_TM_B0M",
             "M10_컨택건수_카드론_TM_B0M",
             "M11_컨택건수_카드론_TM_B0M",
             "M12_컨택건수_카드론_TM_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_카드론_TM_R6M"]
    return c == res


@constraint_udf
def cfs_07_0352(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_리볼빙_TM_R6M = SUM(M07~M12_컨택건수_리볼빙_TM_B0M)
    """
    dd = df[["M07_컨택건수_리볼빙_TM_B0M",
             "M08_컨택건수_리볼빙_TM_B0M",
             "M09_컨택건수_리볼빙_TM_B0M",
             "M10_컨택건수_리볼빙_TM_B0M",
             "M11_컨택건수_리볼빙_TM_B0M",
             "M12_컨택건수_리볼빙_TM_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_리볼빙_TM_R6M"]
    return c == res


@constraint_udf
def cfs_07_0353(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_CA_TM_R6M = SUM(M07~M12_컨택건수_CA_TM_B0M)
    """
    dd = df[["M07_컨택건수_CA_TM_B0M",
             "M08_컨택건수_CA_TM_B0M",
             "M09_컨택건수_CA_TM_B0M",
             "M10_컨택건수_CA_TM_B0M",
             "M11_컨택건수_CA_TM_B0M",
             "M12_컨택건수_CA_TM_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_CA_TM_R6M"]
    return c == res


@constraint_udf
def cfs_07_0354(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_이용유도_TM_R6M = SUM(M07~M12_컨택건수_이용유도_TM_B0M)
    """
    dd = df[["M07_컨택건수_이용유도_TM_B0M",
             "M08_컨택건수_이용유도_TM_B0M",
             "M09_컨택건수_이용유도_TM_B0M",
             "M10_컨택건수_이용유도_TM_B0M",
             "M11_컨택건수_이용유도_TM_B0M",
             "M12_컨택건수_이용유도_TM_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_이용유도_TM_R6M"]
    return c == res


@constraint_udf
def cfs_07_0355(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_신용발급_TM_R6M = SUM(M07~M12_컨택건수_신용발급_TM_B0M)
    """
    dd = df[["M07_컨택건수_신용발급_TM_B0M",
             "M08_컨택건수_신용발급_TM_B0M",
             "M09_컨택건수_신용발급_TM_B0M",
             "M10_컨택건수_신용발급_TM_B0M",
             "M11_컨택건수_신용발급_TM_B0M",
             "M12_컨택건수_신용발급_TM_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_신용발급_TM_R6M"]
    return c == res


@constraint_udf
def cfs_07_0356(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_부대서비스_TM_R6M = SUM(M07~M12_컨택건수_부대서비스_TM_B0M)
    """
    dd = df[["M07_컨택건수_부대서비스_TM_B0M",
             "M08_컨택건수_부대서비스_TM_B0M",
             "M09_컨택건수_부대서비스_TM_B0M",
             "M10_컨택건수_부대서비스_TM_B0M",
             "M11_컨택건수_부대서비스_TM_B0M",
             "M12_컨택건수_부대서비스_TM_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_부대서비스_TM_R6M"]
    return c == res


@constraint_udf
def cfs_07_0357(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_포인트소진_TM_R6M = SUM(M07~M12_컨택건수_포인트소진_TM_B0M)
    """
    dd = df[["M07_컨택건수_포인트소진_TM_B0M",
             "M08_컨택건수_포인트소진_TM_B0M",
             "M09_컨택건수_포인트소진_TM_B0M",
             "M10_컨택건수_포인트소진_TM_B0M",
             "M11_컨택건수_포인트소진_TM_B0M",
             "M12_컨택건수_포인트소진_TM_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_포인트소진_TM_R6M"]
    return c == res


@constraint_udf
def cfs_07_0358(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_보험_TM_R6M = SUM(M07~M12_컨택건수_보험_TM_B0M)
    """
    dd = df[["M07_컨택건수_보험_TM_B0M",
             "M08_컨택건수_보험_TM_B0M",
             "M09_컨택건수_보험_TM_B0M",
             "M10_컨택건수_보험_TM_B0M",
             "M11_컨택건수_보험_TM_B0M",
             "M12_컨택건수_보험_TM_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_보험_TM_R6M"]
    return c == res


@constraint_udf
def cfs_07_0359(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_카드론_LMS_R6M = SUM(M07~M12_컨택건수_카드론_LMS_B0M)
    """
    dd = df[["M07_컨택건수_카드론_LMS_B0M",
             "M08_컨택건수_카드론_LMS_B0M",
             "M09_컨택건수_카드론_LMS_B0M",
             "M10_컨택건수_카드론_LMS_B0M",
             "M11_컨택건수_카드론_LMS_B0M",
             "M12_컨택건수_카드론_LMS_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_카드론_LMS_R6M"]
    return c == res


@constraint_udf
def cfs_07_0360(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_CA_LMS_R6M = SUM(M07~M12_컨택건수_CA_LMS_B0M)
    """
    dd = df[["M07_컨택건수_CA_LMS_B0M",
             "M08_컨택건수_CA_LMS_B0M",
             "M09_컨택건수_CA_LMS_B0M",
             "M10_컨택건수_CA_LMS_B0M",
             "M11_컨택건수_CA_LMS_B0M",
             "M12_컨택건수_CA_LMS_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_CA_LMS_R6M"]
    return c == res


@constraint_udf
def cfs_07_0361(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_리볼빙_LMS_R6M = SUM(M07~M12_컨택건수_리볼빙_LMS_B0M)
    """
    dd = df[["M07_컨택건수_리볼빙_LMS_B0M",
             "M08_컨택건수_리볼빙_LMS_B0M",
             "M09_컨택건수_리볼빙_LMS_B0M",
             "M10_컨택건수_리볼빙_LMS_B0M",
             "M11_컨택건수_리볼빙_LMS_B0M",
             "M12_컨택건수_리볼빙_LMS_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_리볼빙_LMS_R6M"]
    return c == res


@constraint_udf
def cfs_07_0362(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_이용유도_LMS_R6M = SUM(M07~M12_컨택건수_이용유도_LMS_B0M)
    """
    dd = df[["M07_컨택건수_이용유도_LMS_B0M",
             "M08_컨택건수_이용유도_LMS_B0M",
             "M09_컨택건수_이용유도_LMS_B0M",
             "M10_컨택건수_이용유도_LMS_B0M",
             "M11_컨택건수_이용유도_LMS_B0M",
             "M12_컨택건수_이용유도_LMS_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_이용유도_LMS_R6M"]
    return c == res


@constraint_udf
def cfs_07_0363(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_카드론_EM_R6M = SUM(M07~M12_컨택건수_카드론_EM_B0M)
    """
    dd = df[["M07_컨택건수_카드론_EM_B0M",
             "M08_컨택건수_카드론_EM_B0M",
             "M09_컨택건수_카드론_EM_B0M",
             "M10_컨택건수_카드론_EM_B0M",
             "M11_컨택건수_카드론_EM_B0M",
             "M12_컨택건수_카드론_EM_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_카드론_EM_R6M"]
    return c == res


@constraint_udf
def cfs_07_0364(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_CA_EM_R6M = SUM(M07~M12_컨택건수_CA_EM_B0M)
    """
    dd = df[["M07_컨택건수_CA_EM_B0M",
             "M08_컨택건수_CA_EM_B0M",
             "M09_컨택건수_CA_EM_B0M",
             "M10_컨택건수_CA_EM_B0M",
             "M11_컨택건수_CA_EM_B0M",
             "M12_컨택건수_CA_EM_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_CA_EM_R6M"]
    return c == res


@constraint_udf
def cfs_07_0365(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_리볼빙_EM_R6M = SUM(M07~M12_컨택건수_리볼빙_EM_B0M)
    """
    dd = df[["M07_컨택건수_리볼빙_EM_B0M",
             "M08_컨택건수_리볼빙_EM_B0M",
             "M09_컨택건수_리볼빙_EM_B0M",
             "M10_컨택건수_리볼빙_EM_B0M",
             "M11_컨택건수_리볼빙_EM_B0M",
             "M12_컨택건수_리볼빙_EM_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_리볼빙_EM_R6M"]
    return c == res


@constraint_udf
def cfs_07_0366(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_이용유도_EM_R6M = SUM(M07~M12_컨택건수_이용유도_EM_B0M)
    """
    dd = df[["M07_컨택건수_이용유도_EM_B0M",
             "M08_컨택건수_이용유도_EM_B0M",
             "M09_컨택건수_이용유도_EM_B0M",
             "M10_컨택건수_이용유도_EM_B0M",
             "M11_컨택건수_이용유도_EM_B0M",
             "M12_컨택건수_이용유도_EM_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_이용유도_EM_R6M"]
    return c == res


@constraint_udf
def cfs_07_0367(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_카드론_청구서_R6M = SUM(M07~M12_컨택건수_카드론_청구서_B0M)
    """
    dd = df[["M07_컨택건수_카드론_청구서_B0M",
             "M08_컨택건수_카드론_청구서_B0M",
             "M09_컨택건수_카드론_청구서_B0M",
             "M10_컨택건수_카드론_청구서_B0M",
             "M11_컨택건수_카드론_청구서_B0M",
             "M12_컨택건수_카드론_청구서_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_카드론_청구서_R6M"]
    return c == res


@constraint_udf
def cfs_07_0368(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_CA_청구서_R6M = SUM(M07~M12_컨택건수_CA_청구서_B0M)
    """
    dd = df[["M07_컨택건수_CA_청구서_B0M",
             "M08_컨택건수_CA_청구서_B0M",
             "M09_컨택건수_CA_청구서_B0M",
             "M10_컨택건수_CA_청구서_B0M",
             "M11_컨택건수_CA_청구서_B0M",
             "M12_컨택건수_CA_청구서_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_CA_청구서_R6M"]
    return c == res


@constraint_udf
def cfs_07_0369(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_리볼빙_청구서_R6M = SUM(M07~M12_컨택건수_리볼빙_청구서_B0M)
    """
    dd = df[["M07_컨택건수_리볼빙_청구서_B0M",
             "M08_컨택건수_리볼빙_청구서_B0M",
             "M09_컨택건수_리볼빙_청구서_B0M",
             "M10_컨택건수_리볼빙_청구서_B0M",
             "M11_컨택건수_리볼빙_청구서_B0M",
             "M12_컨택건수_리볼빙_청구서_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_리볼빙_청구서_R6M"]
    return c == res


@constraint_udf
def cfs_07_0370(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_이용유도_청구서_R6M = SUM(M07~M12_컨택건수_이용유도_청구서_B0M)
    """
    dd = df[["M07_컨택건수_이용유도_청구서_B0M",
             "M08_컨택건수_이용유도_청구서_B0M",
             "M09_컨택건수_이용유도_청구서_B0M",
             "M10_컨택건수_이용유도_청구서_B0M",
             "M11_컨택건수_이용유도_청구서_B0M",
             "M12_컨택건수_이용유도_청구서_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_이용유도_청구서_R6M"]
    return c == res


@constraint_udf
def cfs_07_0371(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_카드론_인터넷_R6M = SUM(M07~M12_컨택건수_카드론_인터넷_B0M)
    """
    dd = df[["M07_컨택건수_카드론_인터넷_B0M",
             "M08_컨택건수_카드론_인터넷_B0M",
             "M09_컨택건수_카드론_인터넷_B0M",
             "M10_컨택건수_카드론_인터넷_B0M",
             "M11_컨택건수_카드론_인터넷_B0M",
             "M12_컨택건수_카드론_인터넷_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_카드론_인터넷_R6M"]
    return c == res


@constraint_udf
def cfs_07_0372(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_CA_인터넷_R6M = SUM(M07~M12_컨택건수_CA_인터넷_B0M)
    """
    dd = df[["M07_컨택건수_CA_인터넷_B0M",
             "M08_컨택건수_CA_인터넷_B0M",
             "M09_컨택건수_CA_인터넷_B0M",
             "M10_컨택건수_CA_인터넷_B0M",
             "M11_컨택건수_CA_인터넷_B0M",
             "M12_컨택건수_CA_인터넷_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_CA_인터넷_R6M"]
    return c == res


@constraint_udf
def cfs_07_0373(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_리볼빙_인터넷_R6M = SUM(M07~M12_컨택건수_리볼빙_인터넷_B0M)
    """
    dd = df[["M07_컨택건수_리볼빙_인터넷_B0M",
             "M08_컨택건수_리볼빙_인터넷_B0M",
             "M09_컨택건수_리볼빙_인터넷_B0M",
             "M10_컨택건수_리볼빙_인터넷_B0M",
             "M11_컨택건수_리볼빙_인터넷_B0M",
             "M12_컨택건수_리볼빙_인터넷_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_리볼빙_인터넷_R6M"]
    return c == res


@constraint_udf
def cfs_07_0374(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_이용유도_인터넷_R6M = SUM(M07~M12_컨택건수_이용유도_인터넷_B0M)
    """
    dd = df[["M07_컨택건수_이용유도_인터넷_B0M",
             "M08_컨택건수_이용유도_인터넷_B0M",
             "M09_컨택건수_이용유도_인터넷_B0M",
             "M10_컨택건수_이용유도_인터넷_B0M",
             "M11_컨택건수_이용유도_인터넷_B0M",
             "M12_컨택건수_이용유도_인터넷_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_이용유도_인터넷_R6M"]
    return c == res


@constraint_udf
def cfs_07_0375(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_카드론_당사앱_R6M = SUM(M07~M12_컨택건수_카드론_당사앱_B0M)
    """
    dd = df[["M07_컨택건수_카드론_당사앱_B0M",
             "M08_컨택건수_카드론_당사앱_B0M",
             "M09_컨택건수_카드론_당사앱_B0M",
             "M10_컨택건수_카드론_당사앱_B0M",
             "M11_컨택건수_카드론_당사앱_B0M",
             "M12_컨택건수_카드론_당사앱_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_카드론_당사앱_R6M"]
    return c == res


@constraint_udf
def cfs_07_0376(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_CA_당사앱_R6M = SUM(M07~M12_컨택건수_CA_당사앱_B0M)
    """
    dd = df[["M07_컨택건수_CA_당사앱_B0M",
             "M08_컨택건수_CA_당사앱_B0M",
             "M09_컨택건수_CA_당사앱_B0M",
             "M10_컨택건수_CA_당사앱_B0M",
             "M11_컨택건수_CA_당사앱_B0M",
             "M12_컨택건수_CA_당사앱_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_CA_당사앱_R6M"]
    return c == res


@constraint_udf
def cfs_07_0377(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_리볼빙_당사앱_R6M = SUM(M07~M12_컨택건수_리볼빙_당사앱_B0M)
    """
    dd = df[["M07_컨택건수_리볼빙_당사앱_B0M",
             "M08_컨택건수_리볼빙_당사앱_B0M",
             "M09_컨택건수_리볼빙_당사앱_B0M",
             "M10_컨택건수_리볼빙_당사앱_B0M",
             "M11_컨택건수_리볼빙_당사앱_B0M",
             "M12_컨택건수_리볼빙_당사앱_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_리볼빙_당사앱_R6M"]
    return c == res


@constraint_udf
def cfs_07_0378(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_이용유도_당사앱_R6M = SUM(M07~M12_컨택건수_이용유도_당사앱_B0M)
    """
    dd = df[["M07_컨택건수_이용유도_당사앱_B0M",
             "M08_컨택건수_이용유도_당사앱_B0M",
             "M09_컨택건수_이용유도_당사앱_B0M",
             "M10_컨택건수_이용유도_당사앱_B0M",
             "M11_컨택건수_이용유도_당사앱_B0M",
             "M12_컨택건수_이용유도_당사앱_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_이용유도_당사앱_R6M"]
    return c == res


@constraint_udf
def cfs_07_0381(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_채권_R6M = SUM(M07~M12_컨택건수_채권_B0M)
    """
    dd = df[["M07_컨택건수_채권_B0M",
             "M08_컨택건수_채권_B0M",
             "M09_컨택건수_채권_B0M",
             "M10_컨택건수_채권_B0M",
             "M11_컨택건수_채권_B0M",
             "M12_컨택건수_채권_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_채권_R6M"]
    return c == res


@constraint_udf
def cfs_07_0382(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_컨택건수_FDS_R6M = SUM(M07~M12_컨택건수_FDS_B0M)
    """
    dd = df[["M07_컨택건수_FDS_B0M",
             "M08_컨택건수_FDS_B0M",
             "M09_컨택건수_FDS_B0M",
             "M10_컨택건수_FDS_B0M",
             "M11_컨택건수_FDS_B0M",
             "M12_컨택건수_FDS_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_컨택건수_FDS_R6M"]
    return c == res


# 08_M12
@constraint_udf
def cfs_08_0444(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_잔액_신판최대한도소진율_r6m = MAX(M12_잔액_신판최대한도소진율_r3m, M09_잔액_신판최대한도소진율_r3m)
    """
    dd = df[["M12_잔액_신판최대한도소진율_r3m", "M09_잔액_신판최대한도소진율_r3m"]]
    res = dd.max(axis=1)

    c = df["M12_잔액_신판최대한도소진율_r6m"]
    return c == res


@constraint_udf
def cfs_08_0448(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_잔액_신판ca최대한도소진율_r6m = MAX(M12_잔액_신판ca최대한도소진율_r3m, M09_잔액_신판ca최대한도소진율_r3m)
    """
    dd = df[["M12_잔액_신판ca최대한도소진율_r3m", "M09_잔액_신판ca최대한도소진율_r3m"]]
    res = dd.max(axis=1)

    c = df["M12_잔액_신판ca최대한도소진율_r6m"]
    return c == res


# 03_Constraints
@constraint_udf
def ccs_03_0001(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_최종이용일자_기본 <= M08_최종이용일자_기본
    """
    c1, c2 = df["M07_최종이용일자_기본"], df["M08_최종이용일자_기본"]
    return c1 <= c2


@constraint_udf
def ccs_03_0002(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_최종이용일자_신판 <= M08_최종이용일자_신판
    """
    c1, c2 = df["M07_최종이용일자_신판"], df["M08_최종이용일자_신판"]
    return c1 <= c2


@constraint_udf
def ccs_03_0003(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_최종이용일자_CA <= M08_최종이용일자_CA
    """
    c1, c2 = df["M07_최종이용일자_CA"], df["M08_최종이용일자_CA"]
    return c1 <= c2


@constraint_udf
def ccs_03_0004(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_최종이용일자_카드론 <= M08_최종이용일자_카드론
    """
    c1, c2 = df["M07_최종이용일자_카드론"], df["M08_최종이용일자_카드론"]
    return c1 <= c2


@constraint_udf
def ccs_03_0005(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_최종이용일자_체크 <= M08_최종이용일자_체크
    """
    c1, c2 = df["M07_최종이용일자_체크"], df["M08_최종이용일자_체크"]
    return c1 <= c2


@constraint_udf
def ccs_03_0006(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_최종이용일자_일시불 <= M08_최종이용일자_일시불
    """
    c1, c2 = df["M07_최종이용일자_일시불"], df["M08_최종이용일자_일시불"]
    return c1 <= c2


@constraint_udf
def ccs_03_0007(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_최종이용일자_할부 <= M08_최종이용일자_할부
    """
    c1, c2 = df["M07_최종이용일자_할부"], df["M08_최종이용일자_할부"]
    return c1 <= c2


@constraint_udf
def ccs_03_0008(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_카드론이용건수_누적 <= M08_카드론이용건수_누적
    """
    c1, c2 = df["M07_카드론이용건수_누적"], df["M08_카드론이용건수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0009(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_카드론이용월수_누적 <= M08_카드론이용월수_누적
    """
    c1, c2 = df["M07_카드론이용월수_누적"], df["M08_카드론이용월수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0010(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M07_카드론이용금액_누적 <= M08_카드론이용금액_누적
    """
    c1, c2 = df["M07_카드론이용금액_누적"], df["M08_카드론이용금액_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0011(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_최종이용일자_기본 <= M09_최종이용일자_기본
    """
    c1, c2 = df["M08_최종이용일자_기본"], df["M09_최종이용일자_기본"]
    return c1 <= c2


@constraint_udf
def ccs_03_0012(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_최종이용일자_신판 <= M09_최종이용일자_신판
    """
    c1, c2 = df["M08_최종이용일자_신판"], df["M09_최종이용일자_신판"]
    return c1 <= c2


@constraint_udf
def ccs_03_0013(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_최종이용일자_CA <= M09_최종이용일자_CA
    """
    c1, c2 = df["M08_최종이용일자_CA"], df["M09_최종이용일자_CA"]
    return c1 <= c2


@constraint_udf
def ccs_03_0014(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_최종이용일자_카드론 <= M09_최종이용일자_카드론
    """
    c1, c2 = df["M08_최종이용일자_카드론"], df["M09_최종이용일자_카드론"]
    return c1 <= c2


@constraint_udf
def ccs_03_0015(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_최종이용일자_체크 <= M09_최종이용일자_체크
    """
    c1, c2 = df["M08_최종이용일자_체크"], df["M09_최종이용일자_체크"]
    return c1 <= c2


@constraint_udf
def ccs_03_0016(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_최종이용일자_일시불 <= M09_최종이용일자_일시불
    """
    c1, c2 = df["M08_최종이용일자_일시불"], df["M09_최종이용일자_일시불"]
    return c1 <= c2


@constraint_udf
def ccs_03_0017(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_최종이용일자_할부 <= M09_최종이용일자_할부
    """
    c1, c2 = df["M08_최종이용일자_할부"], df["M09_최종이용일자_할부"]
    return c1 <= c2


@constraint_udf
def ccs_03_0018(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_카드론이용건수_누적 <= M09_카드론이용건수_누적
    """
    c1, c2 = df["M08_카드론이용건수_누적"], df["M09_카드론이용건수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0019(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_카드론이용월수_누적 <= M09_카드론이용월수_누적
    """
    c1, c2 = df["M08_카드론이용월수_누적"], df["M09_카드론이용월수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0020(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M08_카드론이용금액_누적 <= M09_카드론이용금액_누적
    """
    c1, c2 = df["M08_카드론이용금액_누적"], df["M09_카드론이용금액_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0021(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_최종이용일자_기본 <= M10_최종이용일자_기본
    """
    c1, c2 = df["M09_최종이용일자_기본"], df["M10_최종이용일자_기본"]
    return c1 <= c2


@constraint_udf
def ccs_03_0022(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_최종이용일자_신판 <= M10_최종이용일자_신판
    """
    c1, c2 = df["M09_최종이용일자_신판"], df["M10_최종이용일자_신판"]
    return c1 <= c2


@constraint_udf
def ccs_03_0023(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_최종이용일자_CA <= M10_최종이용일자_CA
    """
    c1, c2 = df["M09_최종이용일자_CA"], df["M10_최종이용일자_CA"]
    return c1 <= c2


@constraint_udf
def ccs_03_0024(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_최종이용일자_카드론 <= M10_최종이용일자_카드론
    """
    c1, c2 = df["M09_최종이용일자_카드론"], df["M10_최종이용일자_카드론"]
    return c1 <= c2


@constraint_udf
def ccs_03_0025(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_최종이용일자_체크 <= M10_최종이용일자_체크
    """
    c1, c2 = df["M09_최종이용일자_체크"], df["M10_최종이용일자_체크"]
    return c1 <= c2


@constraint_udf
def ccs_03_0026(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_최종이용일자_일시불 <= M10_최종이용일자_일시불
    """
    c1, c2 = df["M09_최종이용일자_일시불"], df["M10_최종이용일자_일시불"]
    return c1 <= c2


@constraint_udf
def ccs_03_0027(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_최종이용일자_할부 <= M10_최종이용일자_할부
    """
    c1, c2 = df["M09_최종이용일자_할부"], df["M10_최종이용일자_할부"]
    return c1 <= c2


@constraint_udf
def ccs_03_0028(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_카드론이용건수_누적 <= M10_카드론이용건수_누적
    """
    c1, c2 = df["M09_카드론이용건수_누적"], df["M10_카드론이용건수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0029(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_카드론이용월수_누적 <= M10_카드론이용월수_누적
    """
    c1, c2 = df["M09_카드론이용월수_누적"], df["M10_카드론이용월수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0030(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M09_카드론이용금액_누적 <= M10_카드론이용금액_누적
    """
    c1, c2 = df["M09_카드론이용금액_누적"], df["M10_카드론이용금액_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0031(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_최종이용일자_기본 <= M11_최종이용일자_기본
    """
    c1, c2 = df["M10_최종이용일자_기본"], df["M11_최종이용일자_기본"]
    return c1 <= c2


@constraint_udf
def ccs_03_0032(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_최종이용일자_신판 <= M11_최종이용일자_신판
    """
    c1, c2 = df["M10_최종이용일자_신판"], df["M11_최종이용일자_신판"]
    return c1 <= c2


@constraint_udf
def ccs_03_0033(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_최종이용일자_CA <= M11_최종이용일자_CA
    """
    c1, c2 = df["M10_최종이용일자_CA"], df["M11_최종이용일자_CA"]
    return c1 <= c2


@constraint_udf
def ccs_03_0034(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_최종이용일자_카드론 <= M11_최종이용일자_카드론
    """
    c1, c2 = df["M10_최종이용일자_카드론"], df["M11_최종이용일자_카드론"]
    return c1 <= c2


@constraint_udf
def ccs_03_0035(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_최종이용일자_체크 <= M11_최종이용일자_체크
    """
    c1, c2 = df["M10_최종이용일자_체크"], df["M11_최종이용일자_체크"]
    return c1 <= c2


@constraint_udf
def ccs_03_0036(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_최종이용일자_일시불 <= M11_최종이용일자_일시불
    """
    c1, c2 = df["M10_최종이용일자_일시불"], df["M11_최종이용일자_일시불"]
    return c1 <= c2


@constraint_udf
def ccs_03_0037(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_최종이용일자_할부 <= M11_최종이용일자_할부
    """
    c1, c2 = df["M10_최종이용일자_할부"], df["M11_최종이용일자_할부"]
    return c1 <= c2


@constraint_udf
def ccs_03_0038(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_카드론이용건수_누적 <= M11_카드론이용건수_누적
    """
    c1, c2 = df["M10_카드론이용건수_누적"], df["M11_카드론이용건수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0039(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_카드론이용월수_누적 <= M11_카드론이용월수_누적
    """
    c1, c2 = df["M10_카드론이용월수_누적"], df["M11_카드론이용월수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0040(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M10_카드론이용금액_누적 <= M11_카드론이용금액_누적
    """
    c1, c2 = df["M10_카드론이용금액_누적"], df["M11_카드론이용금액_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0041(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_최종이용일자_기본 <= M12_최종이용일자_기본
    """
    c1, c2 = df["M11_최종이용일자_기본"], df["M12_최종이용일자_기본"]
    return c1 <= c2


@constraint_udf
def ccs_03_0042(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_최종이용일자_신판 <= M12_최종이용일자_신판
    """
    c1, c2 = df["M11_최종이용일자_신판"], df["M12_최종이용일자_신판"]
    return c1 <= c2


@constraint_udf
def ccs_03_0043(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_최종이용일자_CA <= M12_최종이용일자_CA
    """
    c1, c2 = df["M11_최종이용일자_CA"], df["M12_최종이용일자_CA"]
    return c1 <= c2


@constraint_udf
def ccs_03_0044(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_최종이용일자_카드론 <= M12_최종이용일자_카드론
    """
    c1, c2 = df["M11_최종이용일자_카드론"], df["M12_최종이용일자_카드론"]
    return c1 <= c2


@constraint_udf
def ccs_03_0045(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_최종이용일자_체크 <= M12_최종이용일자_체크
    """
    c1, c2 = df["M11_최종이용일자_체크"], df["M12_최종이용일자_체크"]
    return c1 <= c2


@constraint_udf
def ccs_03_0046(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_최종이용일자_일시불 <= M12_최종이용일자_일시불
    """
    c1, c2 = df["M11_최종이용일자_일시불"], df["M12_최종이용일자_일시불"]
    return c1 <= c2


@constraint_udf
def ccs_03_0047(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_최종이용일자_할부 <= M12_최종이용일자_할부
    """
    c1, c2 = df["M11_최종이용일자_할부"], df["M12_최종이용일자_할부"]
    return c1 <= c2


@constraint_udf
def ccs_03_0048(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_카드론이용건수_누적 <= M12_카드론이용건수_누적
    """
    c1, c2 = df["M11_카드론이용건수_누적"], df["M12_카드론이용건수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0049(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_카드론이용월수_누적 <= M12_카드론이용월수_누적
    """
    c1, c2 = df["M11_카드론이용월수_누적"], df["M12_카드론이용월수_누적"]
    return c1 <= c2


@constraint_udf
def ccs_03_0050(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        M11_카드론이용금액_누적 <= M12_카드론이용금액_누적
    """
    c1, c2 = df["M11_카드론이용금액_누적"], df["M12_카드론이용금액_누적"]
    return c1 <= c2


# 03_M07
@constraint_udf
def cfs_03_0262(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_RP건수_B0M > 0 THEN M07_RP후경과월 = 0
        ELSE M07_RP후경과월 = M07_RP후경과월.clip(1,6)
    """
    dd = df[["M07_RP건수_B0M", "M07_RP후경과월"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (6 if x[1] > 6 else (1 if x[1] < 1 else x[1])), axis=1)

    c = df["M07_RP후경과월"]
    return c == res


@constraint_udf
def cfs_03_0263(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_RP건수_통신_B0M > 0 THEN M07_RP후경과월_통신 = 0
        ELSE M07_RP후경과월_통신 = M07_RP후경과월_통신.clip(1,6)
    """
    dd = df[["M07_RP건수_통신_B0M", "M07_RP후경과월_통신"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (6 if x[1] > 6 else (1 if x[1] < 1 else x[1])), axis=1)

    c = df["M07_RP후경과월_통신"]
    return c == res


@constraint_udf
def cfs_03_0264(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_RP건수_아파트_B0M > 0 THEN M07_RP후경과월_아파트 = 0
        ELSE M07_RP후경과월_아파트 = M07_RP후경과월_아파트.clip(1,6)
    """
    dd = df[["M07_RP건수_아파트_B0M", "M07_RP후경과월_아파트"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (6 if x[1] > 6 else (1 if x[1] < 1 else x[1])), axis=1)

    c = df["M07_RP후경과월_아파트"]
    return c == res


@constraint_udf
def cfs_03_0265(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_RP건수_제휴사서비스직접판매_B0M > 0 THEN M07_RP후경과월_제휴사서비스직접판매 = 0
        ELSE M07_RP후경과월_제휴사서비스직접판매 = M07_RP후경과월_제휴사서비스직접판매.clip(1,6)
    """
    dd = df[["M07_RP건수_제휴사서비스직접판매_B0M", "M07_RP후경과월_제휴사서비스직접판매"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (6 if x[1] > 6 else (1 if x[1] < 1 else x[1])), axis=1)

    c = df["M07_RP후경과월_제휴사서비스직접판매"]
    return c == res


@constraint_udf
def cfs_03_0266(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_RP건수_렌탈_B0M > 0 THEN M07_RP후경과월_렌탈 = 0
        ELSE M07_RP후경과월_렌탈 = M07_RP후경과월_렌탈.clip(1,6)
    """
    dd = df[["M07_RP건수_렌탈_B0M", "M07_RP후경과월_렌탈"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (6 if x[1] > 6 else (1 if x[1] < 1 else x[1])), axis=1)

    c = df["M07_RP후경과월_렌탈"]
    return c == res


@constraint_udf
def cfs_03_0267(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_RP건수_가스_B0M > 0 THEN M07_RP후경과월_가스 = 0
        ELSE M07_RP후경과월_가스 = M07_RP후경과월_가스.clip(1,6)
    """
    dd = df[["M07_RP건수_가스_B0M", "M07_RP후경과월_가스"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (6 if x[1] > 6 else (1 if x[1] < 1 else x[1])), axis=1)

    c = df["M07_RP후경과월_가스"]
    return c == res


@constraint_udf
def cfs_03_0268(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_RP건수_전기_B0M > 0 THEN M07_RP후경과월_전기 = 0
        ELSE M07_RP후경과월_전기 = M07_RP후경과월_전기.clip(1,6)
    """
    dd = df[["M07_RP건수_전기_B0M", "M07_RP후경과월_전기"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (6 if x[1] > 6 else (1 if x[1] < 1 else x[1])), axis=1)

    c = df["M07_RP후경과월_전기"]
    return c == res


@constraint_udf
def cfs_03_0269(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_RP건수_보험_B0M > 0 THEN M07_RP후경과월_보험 = 0
        ELSE M07_RP후경과월_보험 = M07_RP후경과월_보험.clip(1,6)
    """
    dd = df[["M07_RP건수_보험_B0M", "M07_RP후경과월_보험"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (6 if x[1] > 6 else (1 if x[1] < 1 else x[1])), axis=1)

    c = df["M07_RP후경과월_보험"]
    return c == res


@constraint_udf
def cfs_03_0270(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_RP건수_학습비_B0M > 0 THEN M07_RP후경과월_학습비 = 0
        ELSE M07_RP후경과월_학습비 = M07_RP후경과월_학습비.clip(1,6)
    """
    dd = df[["M07_RP건수_학습비_B0M", "M07_RP후경과월_학습비"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (6 if x[1] > 6 else (1 if x[1] < 1 else x[1])), axis=1)

    c = df["M07_RP후경과월_학습비"]
    return c == res


@constraint_udf
def cfs_03_0271(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_RP건수_유선방송_B0M > 0 THEN M07_RP후경과월_유선방송 = 0
        ELSE M07_RP후경과월_유선방송 = M07_RP후경과월_유선방송.clip(1,6)
    """
    dd = df[["M07_RP건수_유선방송_B0M", "M07_RP후경과월_유선방송"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (6 if x[1] > 6 else (1 if x[1] < 1 else x[1])), axis=1)

    c = df["M07_RP후경과월_유선방송"]
    return c == res


@constraint_udf
def cfs_03_0272(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_RP건수_건강_B0M > 0 THEN M07_RP후경과월_건강 = 0
        ELSE M07_RP후경과월_건강 = M07_RP후경과월_건강.clip(1,6)
    """
    dd = df[["M07_RP건수_건강_B0M", "M07_RP후경과월_건강"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (6 if x[1] > 6 else (1 if x[1] < 1 else x[1])), axis=1)

    c = df["M07_RP후경과월_건강"]
    return c == res


@constraint_udf
def cfs_03_0273(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M07_RP건수_교통_B0M > 0 THEN M07_RP후경과월_교통 = 0
        ELSE M07_RP후경과월_교통 = M07_RP후경과월_교통.clip(1,6)
    """
    dd = df[["M07_RP건수_교통_B0M", "M07_RP후경과월_교통"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else (6 if x[1] > 6 else (1 if x[1] < 1 else x[1])), axis=1)

    c = df["M07_RP후경과월_교통"]
    return c == res


# 03_M08
@constraint_udf
def cfs_03_0723(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_RP건수_B0M > 0 THEN M08_RP후경과월 = 0
        ELSE M08_RP후경과월 = min(M07_RP후경과월 + 1, 6)
    """
    dd = df[["M08_RP건수_B0M", "M07_RP후경과월"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_RP후경과월"]
    return c == res


@constraint_udf
def cfs_03_0724(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_RP건수_통신_B0M > 0 THEN M08_RP후경과월_통신 = 0
        ELSE M08_RP후경과월_통신 = min(M07_RP후경과월_통신 + 1, 6)
    """
    dd = df[["M08_RP건수_통신_B0M", "M07_RP후경과월_통신"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_RP후경과월_통신"]
    return c == res


@constraint_udf
def cfs_03_0725(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_RP건수_아파트_B0M > 0 THEN M08_RP후경과월_아파트 = 0
        ELSE M08_RP후경과월_아파트 = min(M07_RP후경과월_아파트 + 1, 6)
    """
    dd = df[["M08_RP건수_아파트_B0M", "M07_RP후경과월_아파트"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_RP후경과월_아파트"]
    return c == res


@constraint_udf
def cfs_03_0726(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_RP건수_제휴사서비스직접판매_B0M > 0 THEN M08_RP후경과월_제휴사서비스직접판매 = 0
        ELSE M08_RP후경과월_제휴사서비스직접판매 = min(M07_RP후경과월_제휴사서비스직접판매 + 1, 6)
    """
    dd = df[["M08_RP건수_제휴사서비스직접판매_B0M", "M07_RP후경과월_제휴사서비스직접판매"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_RP후경과월_제휴사서비스직접판매"]
    return c == res


@constraint_udf
def cfs_03_0727(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_RP건수_렌탈_B0M > 0 THEN M08_RP후경과월_렌탈 = 0
        ELSE M08_RP후경과월_렌탈 = min(M07_RP후경과월_렌탈 + 1, 6)
    """
    dd = df[["M08_RP건수_렌탈_B0M", "M07_RP후경과월_렌탈"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_RP후경과월_렌탈"]
    return c == res


@constraint_udf
def cfs_03_0728(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_RP건수_가스_B0M > 0 THEN M08_RP후경과월_가스 = 0
        ELSE M08_RP후경과월_가스 = min(M07_RP후경과월_가스 + 1, 6)
    """
    dd = df[["M08_RP건수_가스_B0M", "M07_RP후경과월_가스"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_RP후경과월_가스"]
    return c == res


@constraint_udf
def cfs_03_0729(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_RP건수_전기_B0M > 0 THEN M08_RP후경과월_전기 = 0
        ELSE M08_RP후경과월_전기 = min(M07_RP후경과월_전기 + 1, 6)
    """
    dd = df[["M08_RP건수_전기_B0M", "M07_RP후경과월_전기"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_RP후경과월_전기"]
    return c == res


@constraint_udf
def cfs_03_0730(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_RP건수_보험_B0M > 0 THEN M08_RP후경과월_보험 = 0
        ELSE M08_RP후경과월_보험 = min(M07_RP후경과월_보험 + 1, 6)
    """
    dd = df[["M08_RP건수_보험_B0M", "M07_RP후경과월_보험"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_RP후경과월_보험"]
    return c == res


@constraint_udf
def cfs_03_0731(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_RP건수_학습비_B0M > 0 THEN M08_RP후경과월_학습비 = 0
        ELSE M08_RP후경과월_학습비 = min(M07_RP후경과월_학습비 + 1, 6)
    """
    dd = df[["M08_RP건수_학습비_B0M", "M07_RP후경과월_학습비"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_RP후경과월_학습비"]
    return c == res


@constraint_udf
def cfs_03_0732(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_RP건수_유선방송_B0M > 0 THEN M08_RP후경과월_유선방송 = 0
        ELSE M08_RP후경과월_유선방송 = min(M07_RP후경과월_유선방송 + 1, 6)
    """
    dd = df[["M08_RP건수_유선방송_B0M", "M07_RP후경과월_유선방송"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_RP후경과월_유선방송"]
    return c == res


@constraint_udf
def cfs_03_0733(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_RP건수_건강_B0M > 0 THEN M08_RP후경과월_건강 = 0
        ELSE M08_RP후경과월_건강 = min(M07_RP후경과월_건강 + 1, 6)
    """
    dd = df[["M08_RP건수_건강_B0M", "M07_RP후경과월_건강"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_RP후경과월_건강"]
    return c == res


@constraint_udf
def cfs_03_0734(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_RP건수_교통_B0M > 0 THEN M08_RP후경과월_교통 = 0
        ELSE M08_RP후경과월_교통 = min(M07_RP후경과월_교통 + 1, 6)
    """
    dd = df[["M08_RP건수_교통_B0M", "M07_RP후경과월_교통"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M08_RP후경과월_교통"]
    return c == res


@constraint_udf
def cfs_03_0736(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M08_최종카드론_대출일자 IS NULL
        THEN M08_최종카드론이용경과월 = 999
        ELIF M08_최종카드론_대출일자 = M07_최종카드론_대출일자
        THEN M08_최종카드론이용경과월 = M07_최종카드론이용경과월 + 1
        ELSE M08_최종카드론이용경과월 = 0
    """
    dd = df[["M08_최종카드론_대출일자", "M07_최종카드론_대출일자", "M07_최종카드론이용경과월"]]
    res = dd.apply(lambda x: 999 if pd.isna(x[0]) else (x[2] + 1 if x[0] == x[1] else 0), axis=1)

    c = df["M08_최종카드론이용경과월"]
    return c == res


@constraint_udf
def cfs_03_0843(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_가맹점매출금액_B2M = M07_가맹점매출금액_B1M
    """
    res = df["M07_가맹점매출금액_B1M"]

    c = df["M08_가맹점매출금액_B2M"]
    return c == res


@constraint_udf
def cfs_03_0876(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_증감_RP건수_전월 = M07_RP건수_B0M - M08_RP건수_B0M
    """
    dd = df[["M07_RP건수_B0M", "M08_RP건수_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M08_증감_RP건수_전월"]
    return c == res


@constraint_udf
def cfs_03_0878(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_증감_RP건수_통신_전월 = M07_RP건수_통신_B0M - M08_RP건수_통신_B0M
    """
    dd = df[["M07_RP건수_통신_B0M", "M08_RP건수_통신_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M08_증감_RP건수_통신_전월"]
    return c == res


@constraint_udf
def cfs_03_0879(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_증감_RP건수_아파트_전월 = M07_RP건수_아파트_B0M - M08_RP건수_아파트_B0M
    """
    dd = df[["M07_RP건수_아파트_B0M", "M08_RP건수_아파트_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M08_증감_RP건수_아파트_전월"]
    return c == res


@constraint_udf
def cfs_03_0880(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_증감_RP건수_제휴사서비스직접판매_전월 = M07_RP건수_제휴사서비스직접판매_B0M - M08_RP건수_제휴사서비스직접판매_B0M
    """
    dd = df[["M07_RP건수_제휴사서비스직접판매_B0M", "M08_RP건수_제휴사서비스직접판매_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M08_증감_RP건수_제휴사서비스직접판매_전월"]
    return c == res


@constraint_udf
def cfs_03_0881(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_증감_RP건수_렌탈_전월 = M07_RP건수_렌탈_B0M - M08_RP건수_렌탈_B0M
    """
    dd = df[["M07_RP건수_렌탈_B0M", "M08_RP건수_렌탈_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M08_증감_RP건수_렌탈_전월"]
    return c == res


@constraint_udf
def cfs_03_0882(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_증감_RP건수_가스_전월 = M07_RP건수_가스_B0M - M08_RP건수_가스_B0M
    """
    dd = df[["M07_RP건수_가스_B0M", "M08_RP건수_가스_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M08_증감_RP건수_가스_전월"]
    return c == res


@constraint_udf
def cfs_03_0883(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_증감_RP건수_전기_전월 = M07_RP건수_전기_B0M - M08_RP건수_전기_B0M
    """
    dd = df[["M07_RP건수_전기_B0M", "M08_RP건수_전기_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M08_증감_RP건수_전기_전월"]
    return c == res


@constraint_udf
def cfs_03_0884(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_증감_RP건수_보험_전월 = M07_RP건수_보험_B0M - M08_RP건수_보험_B0M
    """
    dd = df[["M07_RP건수_보험_B0M", "M08_RP건수_보험_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M08_증감_RP건수_보험_전월"]
    return c == res


@constraint_udf
def cfs_03_0885(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_증감_RP건수_학습비_전월 = M07_RP건수_학습비_B0M - M08_RP건수_학습비_B0M
    """
    dd = df[["M07_RP건수_학습비_B0M", "M08_RP건수_학습비_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M08_증감_RP건수_학습비_전월"]
    return c == res


@constraint_udf
def cfs_03_0886(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_증감_RP건수_유선방송_전월 = M07_RP건수_유선방송_B0M - M08_RP건수_유선방송_B0M
    """
    dd = df[["M07_RP건수_유선방송_B0M", "M08_RP건수_유선방송_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M08_증감_RP건수_유선방송_전월"]
    return c == res


@constraint_udf
def cfs_03_0887(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_증감_RP건수_건강_전월 = M07_RP건수_건강_B0M - M08_RP건수_건강_B0M
    """
    dd = df[["M07_RP건수_건강_B0M", "M08_RP건수_건강_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M08_증감_RP건수_건강_전월"]
    return c == res


@constraint_udf
def cfs_03_0888(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M08_증감_RP건수_교통_전월 = M07_RP건수_교통_B0M - M08_RP건수_교통_B0M
    """
    dd = df[["M07_RP건수_교통_B0M", "M08_RP건수_교통_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M08_증감_RP건수_교통_전월"]
    return c == res


# 03_M09
@constraint_udf
def cfs_03_1032(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_신용_R3M = M09_이용건수_신용_B0M + M08_이용건수_신용_B0M + M07_이용건수_신용_B0M
    """
    dd = df[["M09_이용건수_신용_B0M", "M08_이용건수_신용_B0M", "M07_이용건수_신용_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_신용_R3M"]
    return c == res


@constraint_udf
def cfs_03_1033(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_신판_R3M = M09_이용건수_신판_B0M + M08_이용건수_신판_B0M + M07_이용건수_신판_B0M
    """
    dd = df[["M09_이용건수_신판_B0M", "M08_이용건수_신판_B0M", "M07_이용건수_신판_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_신판_R3M"]
    return c == res


@constraint_udf
def cfs_03_1034(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_일시불_R3M = M09_이용건수_일시불_B0M + M08_이용건수_일시불_B0M + M07_이용건수_일시불_B0M
    """
    dd = df[["M09_이용건수_일시불_B0M", "M08_이용건수_일시불_B0M", "M07_이용건수_일시불_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_일시불_R3M"]
    return c == res


@constraint_udf
def cfs_03_1035(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_할부_R3M = M09_이용건수_할부_B0M + M08_이용건수_할부_B0M + M07_이용건수_할부_B0M
    """
    dd = df[["M09_이용건수_할부_B0M", "M08_이용건수_할부_B0M", "M07_이용건수_할부_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_할부_R3M"]
    return c == res


@constraint_udf
def cfs_03_1036(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_할부_유이자_R3M = M09_이용건수_할부_유이자_B0M + M08_이용건수_할부_유이자_B0M + M07_이용건수_할부_유이자_B0M
    """
    dd = df[["M09_이용건수_할부_유이자_B0M", "M08_이용건수_할부_유이자_B0M", "M07_이용건수_할부_유이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_할부_유이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1037(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_할부_무이자_R3M = M09_이용건수_할부_무이자_B0M + M08_이용건수_할부_무이자_B0M + M07_이용건수_할부_무이자_B0M
    """
    dd = df[["M09_이용건수_할부_무이자_B0M", "M08_이용건수_할부_무이자_B0M", "M07_이용건수_할부_무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_할부_무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1038(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_부분무이자_R3M = M09_이용건수_부분무이자_B0M + M08_이용건수_부분무이자_B0M + M07_이용건수_부분무이자_B0M
    """
    dd = df[["M09_이용건수_부분무이자_B0M", "M08_이용건수_부분무이자_B0M", "M07_이용건수_부분무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_부분무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1039(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_CA_R3M = M09_이용건수_CA_B0M + M08_이용건수_CA_B0M + M07_이용건수_CA_B0M
    """
    dd = df[["M09_이용건수_CA_B0M", "M08_이용건수_CA_B0M", "M07_이용건수_CA_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_CA_R3M"]
    return c == res


@constraint_udf
def cfs_03_1040(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_체크_R3M = M09_이용건수_체크_B0M + M08_이용건수_체크_B0M + M07_이용건수_체크_B0M
    """
    dd = df[["M09_이용건수_체크_B0M", "M08_이용건수_체크_B0M", "M07_이용건수_체크_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_체크_R3M"]
    return c == res


@constraint_udf
def cfs_03_1041(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_카드론_R3M = M09_이용건수_카드론_B0M + M08_이용건수_카드론_B0M + M07_이용건수_카드론_B0M
    """
    dd = df[["M09_이용건수_카드론_B0M", "M08_이용건수_카드론_B0M", "M07_이용건수_카드론_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_카드론_R3M"]
    return c == res


@constraint_udf
def cfs_03_1042(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_신용_R3M = M09_이용금액_신용_B0M + M08_이용금액_신용_B0M + M07_이용금액_신용_B0M
    """
    dd = df[["M09_이용금액_신용_B0M", "M08_이용금액_신용_B0M", "M07_이용금액_신용_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_신용_R3M"]
    return c == res


@constraint_udf
def cfs_03_1043(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_신판_R3M = M09_이용금액_신판_B0M + M08_이용금액_신판_B0M + M07_이용금액_신판_B0M
    """
    dd = df[["M09_이용금액_신판_B0M", "M08_이용금액_신판_B0M", "M07_이용금액_신판_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_신판_R3M"]
    return c == res


@constraint_udf
def cfs_03_1044(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_일시불_R3M = M09_이용금액_일시불_B0M + M08_이용금액_일시불_B0M + M07_이용금액_일시불_B0M
    """
    dd = df[["M09_이용금액_일시불_B0M", "M08_이용금액_일시불_B0M", "M07_이용금액_일시불_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_일시불_R3M"]
    return c == res


@constraint_udf
def cfs_03_1045(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_할부_R3M = M09_이용금액_할부_B0M + M08_이용금액_할부_B0M + M07_이용금액_할부_B0M
    """
    dd = df[["M09_이용금액_할부_B0M", "M08_이용금액_할부_B0M", "M07_이용금액_할부_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_할부_R3M"]
    return c == res


@constraint_udf
def cfs_03_1046(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_할부_유이자_R3M = M09_이용금액_할부_유이자_B0M + M08_이용금액_할부_유이자_B0M + M07_이용금액_할부_유이자_B0M
    """
    dd = df[["M09_이용금액_할부_유이자_B0M", "M08_이용금액_할부_유이자_B0M", "M07_이용금액_할부_유이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_할부_유이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1047(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_할부_무이자_R3M = M09_이용금액_할부_무이자_B0M + M08_이용금액_할부_무이자_B0M + M07_이용금액_할부_무이자_B0M
    """
    dd = df[["M09_이용금액_할부_무이자_B0M", "M08_이용금액_할부_무이자_B0M", "M07_이용금액_할부_무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_할부_무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1048(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_부분무이자_R3M = M09_이용금액_부분무이자_B0M + M08_이용금액_부분무이자_B0M + M07_이용금액_부분무이자_B0M
    """
    dd = df[["M09_이용금액_부분무이자_B0M", "M08_이용금액_부분무이자_B0M", "M07_이용금액_부분무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_부분무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1049(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_CA_R3M = M09_이용금액_CA_B0M + M08_이용금액_CA_B0M + M07_이용금액_CA_B0M
    """
    dd = df[["M09_이용금액_CA_B0M", "M08_이용금액_CA_B0M", "M07_이용금액_CA_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_CA_R3M"]
    return c == res


@constraint_udf
def cfs_03_1050(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_체크_R3M = M09_이용금액_체크_B0M + M08_이용금액_체크_B0M + M07_이용금액_체크_B0M
    """
    dd = df[["M09_이용금액_체크_B0M", "M08_이용금액_체크_B0M", "M07_이용금액_체크_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_체크_R3M"]
    return c == res


@constraint_udf
def cfs_03_1051(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_카드론_R3M = M09_이용금액_카드론_B0M + M08_이용금액_카드론_B0M + M07_이용금액_카드론_B0M
    """
    dd = df[["M09_이용금액_카드론_B0M", "M08_이용금액_카드론_B0M", "M07_이용금액_카드론_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_카드론_R3M"]
    return c == res


@constraint_udf
def cfs_03_1184(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_RP건수_B0M > 0 THEN M09_RP후경과월 = 0
        ELSE M09_RP후경과월 = min(M08_RP후경과월 + 1, 6)
    """
    dd = df[["M09_RP건수_B0M", "M08_RP후경과월"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_RP후경과월"]
    return c == res


@constraint_udf
def cfs_03_1185(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_RP건수_통신_B0M > 0 THEN M09_RP후경과월_통신 = 0
        ELSE M09_RP후경과월_통신 = min(M08_RP후경과월_통신 + 1, 6)
    """
    dd = df[["M09_RP건수_통신_B0M", "M08_RP후경과월_통신"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_RP후경과월_통신"]
    return c == res


@constraint_udf
def cfs_03_1186(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_RP건수_아파트_B0M > 0 THEN M09_RP후경과월_아파트 = 0
        ELSE M09_RP후경과월_아파트 = min(M08_RP후경과월_아파트 + 1, 6)
    """
    dd = df[["M09_RP건수_아파트_B0M", "M08_RP후경과월_아파트"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_RP후경과월_아파트"]
    return c == res


@constraint_udf
def cfs_03_1187(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_RP건수_제휴사서비스직접판매_B0M > 0 THEN M09_RP후경과월_제휴사서비스직접판매 = 0
        ELSE M09_RP후경과월_제휴사서비스직접판매 = min(M08_RP후경과월_제휴사서비스직접판매 + 1, 6)
    """
    dd = df[["M09_RP건수_제휴사서비스직접판매_B0M", "M08_RP후경과월_제휴사서비스직접판매"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_RP후경과월_제휴사서비스직접판매"]
    return c == res


@constraint_udf
def cfs_03_1188(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_RP건수_렌탈_B0M > 0 THEN M09_RP후경과월_렌탈 = 0
        ELSE M09_RP후경과월_렌탈 = min(M08_RP후경과월_렌탈 + 1, 6)
    """
    dd = df[["M09_RP건수_렌탈_B0M", "M08_RP후경과월_렌탈"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_RP후경과월_렌탈"]
    return c == res


@constraint_udf
def cfs_03_1189(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_RP건수_가스_B0M > 0 THEN M09_RP후경과월_가스 = 0
        ELSE M09_RP후경과월_가스 = min(M08_RP후경과월_가스 + 1, 6)
    """
    dd = df[["M09_RP건수_가스_B0M", "M08_RP후경과월_가스"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_RP후경과월_가스"]
    return c == res


@constraint_udf
def cfs_03_1190(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_RP건수_전기_B0M > 0 THEN M09_RP후경과월_전기 = 0
        ELSE M09_RP후경과월_전기 = min(M08_RP후경과월_전기 + 1, 6)
    """
    dd = df[["M09_RP건수_전기_B0M", "M08_RP후경과월_전기"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_RP후경과월_전기"]
    return c == res


@constraint_udf
def cfs_03_1191(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_RP건수_보험_B0M > 0 THEN M09_RP후경과월_보험 = 0
        ELSE M09_RP후경과월_보험 = min(M08_RP후경과월_보험 + 1, 6)
    """
    dd = df[["M09_RP건수_보험_B0M", "M08_RP후경과월_보험"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_RP후경과월_보험"]
    return c == res


@constraint_udf
def cfs_03_1192(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_RP건수_학습비_B0M > 0 THEN M09_RP후경과월_학습비 = 0
        ELSE M09_RP후경과월_학습비 = min(M08_RP후경과월_학습비 + 1, 6)
    """
    dd = df[["M09_RP건수_학습비_B0M", "M08_RP후경과월_학습비"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_RP후경과월_학습비"]
    return c == res


@constraint_udf
def cfs_03_1193(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_RP건수_유선방송_B0M > 0 THEN M09_RP후경과월_유선방송 = 0
        ELSE M09_RP후경과월_유선방송 = min(M08_RP후경과월_유선방송 + 1, 6)
    """
    dd = df[["M09_RP건수_유선방송_B0M", "M08_RP후경과월_유선방송"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_RP후경과월_유선방송"]
    return c == res


@constraint_udf
def cfs_03_1194(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_RP건수_건강_B0M > 0 THEN M09_RP후경과월_건강 = 0
        ELSE M09_RP후경과월_건강 = min(M08_RP후경과월_건강 + 1, 6)
    """
    dd = df[["M09_RP건수_건강_B0M", "M08_RP후경과월_건강"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_RP후경과월_건강"]
    return c == res


@constraint_udf
def cfs_03_1195(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_RP건수_교통_B0M > 0 THEN M09_RP후경과월_교통 = 0
        ELSE M09_RP후경과월_교통 = min(M08_RP후경과월_교통 + 1, 6)
    """
    dd = df[["M09_RP건수_교통_B0M", "M08_RP후경과월_교통"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M09_RP후경과월_교통"]
    return c == res


@constraint_udf
def cfs_03_1369(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_승인거절건수_R3M = M07_승인거절건수_B0M + M08_승인거절건수_B0M + M09_승인거절건수_B0M
    """
    dd = df[["M07_승인거절건수_B0M", "M08_승인거절건수_B0M", "M09_승인거절건수_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_승인거절건수_R3M"]
    return c == res


@constraint_udf
def cfs_03_1370(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_승인거절건수_한도초과_R3M = M07_승인거절건수_한도초과_B0M + M08_승인거절건수_한도초과_B0M + M09_승인거절건수_한도초과_B0M
    """
    dd = df[["M07_승인거절건수_한도초과_B0M", "M08_승인거절건수_한도초과_B0M", "M09_승인거절건수_한도초과_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_승인거절건수_한도초과_R3M"]
    return c == res


@constraint_udf
def cfs_03_1371(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_승인거절건수_BL_R3M = M07_승인거절건수_BL_B0M + M08_승인거절건수_BL_B0M + M09_승인거절건수_BL_B0M
    """
    dd = df[["M07_승인거절건수_BL_B0M", "M08_승인거절건수_BL_B0M", "M09_승인거절건수_BL_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_승인거절건수_BL_R3M"]
    return c == res


@constraint_udf
def cfs_03_1372(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_승인거절건수_입력오류_R3M = M07_승인거절건수_입력오류_B0M + M08_승인거절건수_입력오류_B0M + M09_승인거절건수_입력오류_B0M
    """
    dd = df[["M07_승인거절건수_입력오류_B0M", "M08_승인거절건수_입력오류_B0M", "M09_승인거절건수_입력오류_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_승인거절건수_입력오류_R3M"]
    return c == res


@constraint_udf
def cfs_03_1373(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_승인거절건수_기타_R3M = M07_승인거절건수_기타_B0M + M08_승인거절건수_기타_B0M + M09_승인거절건수_기타_B0M
    """
    dd = df[["M07_승인거절건수_기타_B0M", "M08_승인거절건수_기타_B0M", "M09_승인거절건수_기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_승인거절건수_기타_R3M"]
    return c == res


@constraint_udf
def cfs_03_1197(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M09_최종카드론_대출일자 IS NULL
        THEN M09_최종카드론이용경과월 = 999
        ELIF M09_최종카드론_대출일자 = M08_최종카드론_대출일자
        THEN M09_최종카드론이용경과월 = M08_최종카드론이용경과월 + 1
        ELSE M09_최종카드론이용경과월 = 0
    """
    dd = df[["M09_최종카드론_대출일자", "M08_최종카드론_대출일자", "M08_최종카드론이용경과월"]]
    res = dd.apply(lambda x: 999 if pd.isna(x[0]) else (x[2] + 1 if x[0] == x[1] else 0), axis=1)

    c = df["M09_최종카드론이용경과월"]
    return c == res


@constraint_udf
def cfs_03_1222(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_온라인_R3M = M09_이용금액_온라인_B0M + M08_이용금액_온라인_B0M + M07_이용금액_온라인_B0M
    """
    dd = df[["M09_이용금액_온라인_B0M", "M08_이용금액_온라인_B0M", "M07_이용금액_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1223(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_오프라인_R3M = M09_이용금액_오프라인_B0M + M08_이용금액_오프라인_B0M + M07_이용금액_오프라인_B0M
    """
    dd = df[["M09_이용금액_오프라인_B0M", "M08_이용금액_오프라인_B0M", "M07_이용금액_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1224(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_온라인_R3M = M09_이용건수_온라인_B0M + M08_이용건수_온라인_B0M + M07_이용건수_온라인_B0M
    """
    dd = df[["M09_이용건수_온라인_B0M", "M08_이용건수_온라인_B0M", "M07_이용건수_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1225(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_오프라인_R3M = M09_이용건수_오프라인_B0M + M08_이용건수_오프라인_B0M + M07_이용건수_오프라인_B0M
    """
    dd = df[["M09_이용건수_오프라인_B0M", "M08_이용건수_오프라인_B0M", "M07_이용건수_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1236(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_페이_온라인_R3M = M09_이용금액_페이_온라인_B0M + M08_이용금액_페이_온라인_B0M + M07_이용금액_페이_온라인_B0M
    """
    dd = df[["M09_이용금액_페이_온라인_B0M", "M08_이용금액_페이_온라인_B0M", "M07_이용금액_페이_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_페이_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1237(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_페이_오프라인_R3M = M09_이용금액_페이_오프라인_B0M + M08_이용금액_페이_오프라인_B0M + M07_이용금액_페이_오프라인_B0M
    """
    dd = df[["M09_이용금액_페이_오프라인_B0M", "M08_이용금액_페이_오프라인_B0M", "M07_이용금액_페이_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_페이_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1238(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_페이_온라인_R3M = M09_이용건수_페이_온라인_B0M + M08_이용건수_페이_온라인_B0M + M07_이용건수_페이_온라인_B0M
    """
    dd = df[["M09_이용건수_페이_온라인_B0M", "M08_이용건수_페이_온라인_B0M", "M07_이용건수_페이_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_페이_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1239(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_페이_오프라인_R3M = M09_이용건수_페이_오프라인_B0M + M08_이용건수_페이_오프라인_B0M + M07_이용건수_페이_오프라인_B0M
    """
    dd = df[["M09_이용건수_페이_오프라인_B0M", "M08_이용건수_페이_오프라인_B0M", "M07_이용건수_페이_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_페이_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1265(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_간편결제_R3M = M09_이용금액_간편결제_B0M + M08_이용금액_간편결제_B0M + M07_이용금액_간편결제_B0M
    """
    dd = df[["M09_이용금액_간편결제_B0M", "M08_이용금액_간편결제_B0M", "M07_이용금액_간편결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_간편결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_1266(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_당사페이_R3M = M09_이용금액_당사페이_B0M + M08_이용금액_당사페이_B0M + M07_이용금액_당사페이_B0M
    """
    dd = df[["M09_이용금액_당사페이_B0M", "M08_이용금액_당사페이_B0M", "M07_이용금액_당사페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_당사페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1267(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_당사기타_R3M = M09_이용금액_당사기타_B0M + M08_이용금액_당사기타_B0M + M07_이용금액_당사기타_B0M
    """
    dd = df[["M09_이용금액_당사기타_B0M", "M08_이용금액_당사기타_B0M", "M07_이용금액_당사기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_당사기타_R3M"]
    return c == res


@constraint_udf
def cfs_03_1268(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_A페이_R3M = M09_이용금액_A페이_B0M + M08_이용금액_A페이_B0M + M07_이용금액_A페이_B0M
    """
    dd = df[["M09_이용금액_A페이_B0M", "M08_이용금액_A페이_B0M", "M07_이용금액_A페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_A페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1269(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_B페이_R3M = M09_이용금액_B페이_B0M + M08_이용금액_B페이_B0M + M07_이용금액_B페이_B0M
    """
    dd = df[["M09_이용금액_B페이_B0M", "M08_이용금액_B페이_B0M", "M07_이용금액_B페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_B페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1270(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_C페이_R3M = M09_이용금액_C페이_B0M + M08_이용금액_C페이_B0M + M07_이용금액_C페이_B0M
    """
    dd = df[["M09_이용금액_C페이_B0M", "M08_이용금액_C페이_B0M", "M07_이용금액_C페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_C페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1271(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_D페이_R3M = M09_이용금액_D페이_B0M + M08_이용금액_D페이_B0M + M07_이용금액_D페이_B0M
    """
    dd = df[["M09_이용금액_D페이_B0M", "M08_이용금액_D페이_B0M", "M07_이용금액_D페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_D페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1272(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_간편결제_R3M = M09_이용건수_간편결제_B0M + M08_이용건수_간편결제_B0M + M07_이용건수_간편결제_B0M
    """
    dd = df[["M09_이용건수_간편결제_B0M", "M08_이용건수_간편결제_B0M", "M07_이용건수_간편결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_간편결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_1273(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_당사페이_R3M = M09_이용건수_당사페이_B0M + M08_이용건수_당사페이_B0M + M07_이용건수_당사페이_B0M
    """
    dd = df[["M09_이용건수_당사페이_B0M", "M08_이용건수_당사페이_B0M", "M07_이용건수_당사페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_당사페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1274(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_당사기타_R3M = M09_이용건수_당사기타_B0M + M08_이용건수_당사기타_B0M + M07_이용건수_당사기타_B0M
    """
    dd = df[["M09_이용건수_당사기타_B0M", "M08_이용건수_당사기타_B0M", "M07_이용건수_당사기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_당사기타_R3M"]
    return c == res


@constraint_udf
def cfs_03_1275(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_A페이_R3M = M09_이용건수_A페이_B0M + M08_이용건수_A페이_B0M + M07_이용건수_A페이_B0M
    """
    dd = df[["M09_이용건수_A페이_B0M", "M08_이용건수_A페이_B0M", "M07_이용건수_A페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_A페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1276(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_B페이_R3M = M09_이용건수_B페이_B0M + M08_이용건수_B페이_B0M + M07_이용건수_B페이_B0M
    """
    dd = df[["M09_이용건수_B페이_B0M", "M08_이용건수_B페이_B0M", "M07_이용건수_B페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_B페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1277(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_C페이_R3M = M09_이용건수_C페이_B0M + M08_이용건수_C페이_B0M + M07_이용건수_C페이_B0M
    """
    dd = df[["M09_이용건수_C페이_B0M", "M08_이용건수_C페이_B0M", "M07_이용건수_C페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_C페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1278(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_D페이_R3M = M09_이용건수_D페이_B0M + M08_이용건수_D페이_B0M + M07_이용건수_D페이_B0M
    """
    dd = df[["M09_이용건수_D페이_B0M", "M08_이용건수_D페이_B0M", "M07_이용건수_D페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_D페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1297(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용횟수_선결제_R3M = M09_이용횟수_선결제_B0M + M08_이용횟수_선결제_B0M + M07_이용횟수_선결제_B0M
    """
    dd = df[["M09_이용횟수_선결제_B0M", "M08_이용횟수_선결제_B0M", "M07_이용횟수_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용횟수_선결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_1298(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_선결제_R3M = M09_이용금액_선결제_B0M + M08_이용금액_선결제_B0M + M07_이용금액_선결제_B0M
    """
    dd = df[["M09_이용금액_선결제_B0M", "M08_이용금액_선결제_B0M", "M07_이용금액_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_선결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_1299(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용건수_선결제_R3M = M09_이용건수_선결제_B0M + M08_이용건수_선결제_B0M + M07_이용건수_선결제_B0M
    """
    dd = df[["M09_이용건수_선결제_B0M", "M08_이용건수_선결제_B0M", "M07_이용건수_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용건수_선결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_1304(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_가맹점매출금액_B2M = M08_가맹점매출금액_B1M
    """
    res = df["M08_가맹점매출금액_B1M"]

    c = df["M09_가맹점매출금액_B2M"]
    return c == res


@constraint_udf
def cfs_03_1306(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_정상청구원금_B2M = M07_정상청구원금_B0M
    """
    res = df["M07_정상청구원금_B0M"]

    c = df["M09_정상청구원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_1309(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_선입금원금_B2M = M07_선입금원금_B0M
    """
    res = df["M07_선입금원금_B0M"]

    c = df["M09_선입금원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_1312(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_정상입금원금_B2M = M07_정상입금원금_B0M
    """
    res = df["M07_정상입금원금_B0M"]

    c = df["M09_정상입금원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_1315(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_연체입금원금_B2M = M07_연체입금원금_B0M
    """
    res = df["M07_연체입금원금_B0M"]

    c = df["M09_연체입금원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_1323(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용횟수_연체_R3M = M09_이용횟수_연체_B0M + M08_이용횟수_연체_B0M + M07_이용횟수_연체_B0M
    """
    dd = df[["M09_이용횟수_연체_B0M", "M08_이용횟수_연체_B0M", "M07_이용횟수_연체_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용횟수_연체_R3M"]
    return c == res


@constraint_udf
def cfs_03_1324(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_이용금액_연체_R3M = M09_이용금액_연체_B0M + M08_이용금액_연체_B0M + M07_이용금액_연체_B0M
    """
    dd = df[["M09_이용금액_연체_B0M", "M08_이용금액_연체_B0M", "M07_이용금액_연체_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M09_이용금액_연체_R3M"]
    return c == res


@constraint_udf
def cfs_03_1337(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_증감_RP건수_전월 = M08_RP건수_B0M - M09_RP건수_B0M
    """
    dd = df[["M08_RP건수_B0M", "M09_RP건수_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M09_증감_RP건수_전월"]
    return c == res


@constraint_udf
def cfs_03_1339(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_증감_RP건수_통신_전월 = M08_RP건수_통신_B0M - M09_RP건수_통신_B0M
    """
    dd = df[["M08_RP건수_통신_B0M", "M09_RP건수_통신_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M09_증감_RP건수_통신_전월"]
    return c == res


@constraint_udf
def cfs_03_1340(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_증감_RP건수_아파트_전월 = M08_RP건수_아파트_B0M - M09_RP건수_아파트_B0M
    """
    dd = df[["M08_RP건수_아파트_B0M", "M09_RP건수_아파트_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M09_증감_RP건수_아파트_전월"]
    return c == res


@constraint_udf
def cfs_03_1341(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_증감_RP건수_제휴사서비스직접판매_전월 = M08_RP건수_제휴사서비스직접판매_B0M - M09_RP건수_제휴사서비스직접판매_B0M
    """
    dd = df[["M08_RP건수_제휴사서비스직접판매_B0M", "M09_RP건수_제휴사서비스직접판매_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M09_증감_RP건수_제휴사서비스직접판매_전월"]
    return c == res


@constraint_udf
def cfs_03_1342(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_증감_RP건수_렌탈_전월 = M08_RP건수_렌탈_B0M - M09_RP건수_렌탈_B0M
    """
    dd = df[["M08_RP건수_렌탈_B0M", "M09_RP건수_렌탈_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M09_증감_RP건수_렌탈_전월"]
    return c == res


@constraint_udf
def cfs_03_1343(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_증감_RP건수_가스_전월 = M08_RP건수_가스_B0M - M09_RP건수_가스_B0M
    """
    dd = df[["M08_RP건수_가스_B0M", "M09_RP건수_가스_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M09_증감_RP건수_가스_전월"]
    return c == res


@constraint_udf
def cfs_03_1344(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_증감_RP건수_전기_전월 = M08_RP건수_전기_B0M - M09_RP건수_전기_B0M
    """
    dd = df[["M08_RP건수_전기_B0M", "M09_RP건수_전기_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M09_증감_RP건수_전기_전월"]
    return c == res


@constraint_udf
def cfs_03_1345(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_증감_RP건수_보험_전월 = M08_RP건수_보험_B0M - M09_RP건수_보험_B0M
    """
    dd = df[["M08_RP건수_보험_B0M", "M09_RP건수_보험_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M09_증감_RP건수_보험_전월"]
    return c == res


@constraint_udf
def cfs_03_1346(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_증감_RP건수_학습비_전월 = M08_RP건수_학습비_B0M - M09_RP건수_학습비_B0M
    """
    dd = df[["M08_RP건수_학습비_B0M", "M09_RP건수_학습비_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M09_증감_RP건수_학습비_전월"]
    return c == res


@constraint_udf
def cfs_03_1347(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_증감_RP건수_유선방송_전월 = M08_RP건수_유선방송_B0M - M09_RP건수_유선방송_B0M
    """
    dd = df[["M08_RP건수_유선방송_B0M", "M09_RP건수_유선방송_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M09_증감_RP건수_유선방송_전월"]
    return c == res


@constraint_udf
def cfs_03_1348(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_증감_RP건수_건강_전월 = M08_RP건수_건강_B0M - M09_RP건수_건강_B0M
    """
    dd = df[["M08_RP건수_건강_B0M", "M09_RP건수_건강_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M09_증감_RP건수_건강_전월"]
    return c == res


@constraint_udf
def cfs_03_1349(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M09_증감_RP건수_교통_전월 = M08_RP건수_교통_B0M - M09_RP건수_교통_B0M
    """
    dd = df[["M08_RP건수_교통_B0M", "M09_RP건수_교통_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M09_증감_RP건수_교통_전월"]
    return c == res


# 03_M10
@constraint_udf
def cfs_03_1463(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_신용_R6M = M10_이용건수_신용_R3M + M07_이용건수_신용_R3M
    """
    dd = df[["M10_이용건수_신용_R3M", "M07_이용건수_신용_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_신용_R6M"]
    return c == res


@constraint_udf
def cfs_03_1464(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_신판_R6M = M10_이용건수_신판_R3M + M07_이용건수_신판_R3M
    """
    dd = df[["M10_이용건수_신판_R3M", "M07_이용건수_신판_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_신판_R6M"]
    return c == res


@constraint_udf
def cfs_03_1465(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_일시불_R6M = M10_이용건수_일시불_R3M + M07_이용건수_일시불_R3M
    """
    dd = df[["M10_이용건수_일시불_R3M", "M07_이용건수_일시불_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_일시불_R6M"]
    return c == res


@constraint_udf
def cfs_03_1466(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_할부_R6M = M10_이용건수_할부_R3M +  M07_이용건수_할부_R3M
    """
    dd = df[["M10_이용건수_할부_R3M", "M07_이용건수_할부_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_할부_R6M"]
    return c == res


@constraint_udf
def cfs_03_1467(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_할부_유이자_R6M = M10_이용건수_할부_유이자_R3M + M07_이용건수_할부_유이자_R3M
    """
    dd = df[["M10_이용건수_할부_유이자_R3M", "M07_이용건수_할부_유이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_할부_유이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_1468(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_할부_무이자_R6M = M10_이용건수_할부_무이자_R3M + M07_이용건수_할부_무이자_R3M
    """
    dd = df[["M10_이용건수_할부_무이자_R3M", "M07_이용건수_할부_무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_할부_무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_1469(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_부분무이자_R6M = M10_이용건수_부분무이자_R3M + M07_이용건수_부분무이자_R3M
    """
    dd = df[["M10_이용건수_부분무이자_R3M", "M07_이용건수_부분무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_부분무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_1470(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_CA_R6M = M10_이용건수_CA_R3M + M07_이용건수_CA_R3M
    """
    dd = df[["M10_이용건수_CA_R3M", "M07_이용건수_CA_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_CA_R6M"]
    return c == res


@constraint_udf
def cfs_03_1471(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_체크_R6M = M10_이용건수_체크_R3M + M07_이용건수_체크_R3M
    """
    dd = df[["M10_이용건수_체크_R3M", "M07_이용건수_체크_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_체크_R6M"]
    return c == res


@constraint_udf
def cfs_03_1472(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_카드론_R6M = M10_이용건수_카드론_R3M + M07_이용건수_카드론_R3M
    """
    dd = df[["M10_이용건수_카드론_R3M", "M07_이용건수_카드론_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_카드론_R6M"]
    return c == res


@constraint_udf
def cfs_03_1473(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_신용_R6M = M10_이용금액_신용_R3M + M07_이용금액_신용_R3M
    """
    dd = df[["M10_이용금액_신용_R3M", "M07_이용금액_신용_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_신용_R6M"]
    return c == res


@constraint_udf
def cfs_03_1474(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_신판_R6M = M10_이용금액_신판_R3M + M07_이용금액_신판_R3M
    """
    dd = df[["M10_이용금액_신판_R3M", "M07_이용금액_신판_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_신판_R6M"]
    return c == res


@constraint_udf
def cfs_03_1475(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_일시불_R6M = M10_이용금액_일시불_R3M + M07_이용금액_일시불_R3M
    """
    dd = df[["M10_이용금액_일시불_R3M", "M07_이용금액_일시불_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_일시불_R6M"]
    return c == res


@constraint_udf
def cfs_03_1476(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_할부_R6M = M10_이용금액_할부_R3M + M07_이용금액_할부_R3M
    """
    dd = df[["M10_이용금액_할부_R3M", "M07_이용금액_할부_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_할부_R6M"]
    return c == res


@constraint_udf
def cfs_03_1477(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_할부_유이자_R6M = M10_이용금액_할부_유이자_R3M + M07_이용금액_할부_유이자_R3M
    """
    dd = df[["M10_이용금액_할부_유이자_R3M", "M07_이용금액_할부_유이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_할부_유이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_1478(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_할부_무이자_R6M = M10_이용금액_할부_무이자_R3M + M07_이용금액_할부_무이자_R3M
    """
    dd = df[["M10_이용금액_할부_무이자_R3M", "M07_이용금액_할부_무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_할부_무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_1479(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_부분무이자_R6M = M10_이용금액_부분무이자_R3M +  M07_이용금액_부분무이자_R3M
    """
    dd = df[["M10_이용금액_부분무이자_R3M", "M07_이용금액_부분무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_부분무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_1480(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_CA_R6M = M10_이용금액_CA_R3M + M07_이용금액_CA_R3M
    """
    dd = df[["M10_이용금액_CA_R3M", "M07_이용금액_CA_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_CA_R6M"]
    return c == res


@constraint_udf
def cfs_03_1481(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_체크_R6M = M10_이용금액_체크_R3M + M07_이용금액_체크_R3M
    """
    dd = df[["M10_이용금액_체크_R3M", "M07_이용금액_체크_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_체크_R6M"]
    return c == res


@constraint_udf
def cfs_03_1482(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_카드론_R6M = M10_이용금액_카드론_R3M + M07_이용금액_카드론_R3M
    """
    dd = df[["M10_이용금액_카드론_R3M", "M07_이용금액_카드론_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_카드론_R6M"]
    return c == res


@constraint_udf
def cfs_03_1493(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_신용_R3M = M10_이용건수_신용_B0M + M09_이용건수_신용_B0M + M08_이용건수_신용_B0M
    """
    dd = df[["M10_이용건수_신용_B0M", "M09_이용건수_신용_B0M", "M08_이용건수_신용_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_신용_R3M"]
    return c == res


@constraint_udf
def cfs_03_1494(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_신판_R3M = M10_이용건수_신판_B0M + M09_이용건수_신판_B0M + M08_이용건수_신판_B0M
    """
    dd = df[["M10_이용건수_신판_B0M", "M09_이용건수_신판_B0M", "M08_이용건수_신판_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_신판_R3M"]
    return c == res


@constraint_udf
def cfs_03_1495(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_일시불_R3M = M10_이용건수_일시불_B0M + M09_이용건수_일시불_B0M + M08_이용건수_일시불_B0M
    """
    dd = df[["M10_이용건수_일시불_B0M", "M09_이용건수_일시불_B0M", "M08_이용건수_일시불_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_일시불_R3M"]
    return c == res


@constraint_udf
def cfs_03_1496(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_할부_R3M = M10_이용건수_할부_B0M + M09_이용건수_할부_B0M + M08_이용건수_할부_B0M
    """
    dd = df[["M10_이용건수_할부_B0M", "M09_이용건수_할부_B0M", "M08_이용건수_할부_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_할부_R3M"]
    return c == res


@constraint_udf
def cfs_03_1497(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_할부_유이자_R3M = M10_이용건수_할부_유이자_B0M + M09_이용건수_할부_유이자_B0M + M08_이용건수_할부_유이자_B0M
    """
    dd = df[["M10_이용건수_할부_유이자_B0M", "M09_이용건수_할부_유이자_B0M", "M08_이용건수_할부_유이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_할부_유이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1498(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_할부_무이자_R3M = M10_이용건수_할부_무이자_B0M + M09_이용건수_할부_무이자_B0M + M08_이용건수_할부_무이자_B0M
    """
    dd = df[["M10_이용건수_할부_무이자_B0M", "M09_이용건수_할부_무이자_B0M", "M08_이용건수_할부_무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_할부_무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1499(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_부분무이자_R3M = M10_이용건수_부분무이자_B0M + M09_이용건수_부분무이자_B0M + M08_이용건수_부분무이자_B0M
    """
    dd = df[["M10_이용건수_부분무이자_B0M", "M09_이용건수_부분무이자_B0M", "M08_이용건수_부분무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_부분무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1500(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_CA_R3M = M10_이용건수_CA_B0M + M09_이용건수_CA_B0M + M08_이용건수_CA_B0M
    """
    dd = df[["M10_이용건수_CA_B0M", "M09_이용건수_CA_B0M", "M08_이용건수_CA_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_CA_R3M"]
    return c == res


@constraint_udf
def cfs_03_1501(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_체크_R3M = M10_이용건수_체크_B0M + M09_이용건수_체크_B0M + M08_이용건수_체크_B0M
    """
    dd = df[["M10_이용건수_체크_B0M", "M09_이용건수_체크_B0M", "M08_이용건수_체크_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_체크_R3M"]
    return c == res


@constraint_udf
def cfs_03_1502(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_카드론_R3M = M10_이용건수_카드론_B0M + M09_이용건수_카드론_B0M + M08_이용건수_카드론_B0M
    """
    dd = df[["M10_이용건수_카드론_B0M", "M09_이용건수_카드론_B0M", "M08_이용건수_카드론_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_카드론_R3M"]
    return c == res


@constraint_udf
def cfs_03_1503(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_신용_R3M = M10_이용금액_신용_B0M + M09_이용금액_신용_B0M + M08_이용금액_신용_B0M
    """
    dd = df[["M10_이용금액_신용_B0M", "M09_이용금액_신용_B0M", "M08_이용금액_신용_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_신용_R3M"]
    return c == res


@constraint_udf
def cfs_03_1504(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_신판_R3M = M10_이용금액_신판_B0M + M09_이용금액_신판_B0M + M08_이용금액_신판_B0M
    """
    dd = df[["M10_이용금액_신판_B0M", "M09_이용금액_신판_B0M", "M08_이용금액_신판_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_신판_R3M"]
    return c == res


@constraint_udf
def cfs_03_1505(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_일시불_R3M = M10_이용금액_일시불_B0M + M09_이용금액_일시불_B0M + M08_이용금액_일시불_B0M
    """
    dd = df[["M10_이용금액_일시불_B0M", "M09_이용금액_일시불_B0M", "M08_이용금액_일시불_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_일시불_R3M"]
    return c == res


@constraint_udf
def cfs_03_1506(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_할부_R3M = M10_이용금액_할부_B0M + M09_이용금액_할부_B0M + M08_이용금액_할부_B0M
    """
    dd = df[["M10_이용금액_할부_B0M", "M09_이용금액_할부_B0M", "M08_이용금액_할부_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_할부_R3M"]
    return c == res


@constraint_udf
def cfs_03_1507(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_할부_유이자_R3M = M10_이용금액_할부_유이자_B0M + M09_이용금액_할부_유이자_B0M + M08_이용금액_할부_유이자_B0M
    """
    dd = df[["M10_이용금액_할부_유이자_B0M", "M09_이용금액_할부_유이자_B0M", "M08_이용금액_할부_유이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_할부_유이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1508(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_할부_무이자_R3M = M10_이용금액_할부_무이자_B0M + M09_이용금액_할부_무이자_B0M + M08_이용금액_할부_무이자_B0M
    """
    dd = df[["M10_이용금액_할부_무이자_B0M", "M09_이용금액_할부_무이자_B0M", "M08_이용금액_할부_무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_할부_무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1509(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_부분무이자_R3M = M10_이용금액_부분무이자_B0M + M09_이용금액_부분무이자_B0M + M08_이용금액_부분무이자_B0M
    """
    dd = df[["M10_이용금액_부분무이자_B0M", "M09_이용금액_부분무이자_B0M", "M08_이용금액_부분무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_부분무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1510(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_CA_R3M = M10_이용금액_CA_B0M + M09_이용금액_CA_B0M + M08_이용금액_CA_B0M
    """
    dd = df[["M10_이용금액_CA_B0M", "M09_이용금액_CA_B0M", "M08_이용금액_CA_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_CA_R3M"]
    return c == res


@constraint_udf
def cfs_03_1511(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_체크_R3M = M10_이용금액_체크_B0M + M09_이용금액_체크_B0M + M08_이용금액_체크_B0M
    """
    dd = df[["M10_이용금액_체크_B0M", "M09_이용금액_체크_B0M", "M08_이용금액_체크_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_체크_R3M"]
    return c == res


@constraint_udf
def cfs_03_1512(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_카드론_R3M = M10_이용금액_카드론_B0M + M09_이용금액_카드론_B0M + M08_이용금액_카드론_B0M
    """
    dd = df[["M10_이용금액_카드론_B0M", "M09_이용금액_카드론_B0M", "M08_이용금액_카드론_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_카드론_R3M"]
    return c == res


@constraint_udf
def cfs_03_1527(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_건수_할부전환_R6M = M10_건수_할부전환_R3M + M07_건수_할부전환_R3M
    """
    dd = df[["M10_건수_할부전환_R3M", "M07_건수_할부전환_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_건수_할부전환_R6M"]
    return c == res


@constraint_udf
def cfs_03_1528(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_금액_할부전환_R6M = M10_금액_할부전환_R3M + M07_금액_할부전환_R3M
    """
    dd = df[["M10_금액_할부전환_R3M", "M07_금액_할부전환_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_금액_할부전환_R6M"]
    return c == res


@constraint_udf
def cfs_03_1645(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_RP건수_B0M > 0 THEN M10_RP후경과월 = 0
        ELSE M10_RP후경과월 = min(M09_RP후경과월 + 1, 6)
    """
    dd = df[["M10_RP건수_B0M", "M09_RP후경과월"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_RP후경과월"]
    return c == res


@constraint_udf
def cfs_03_1646(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_RP건수_통신_B0M > 0 THEN M10_RP후경과월_통신 = 0
        ELSE M10_RP후경과월_통신 = min(M09_RP후경과월_통신 + 1, 6)
    """
    dd = df[["M10_RP건수_통신_B0M", "M09_RP후경과월_통신"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_RP후경과월_통신"]
    return c == res


@constraint_udf
def cfs_03_1647(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_RP건수_아파트_B0M > 0 THEN M10_RP후경과월_아파트 = 0
        ELSE M10_RP후경과월_아파트 = min(M09_RP후경과월_아파트 + 1, 6)
    """
    dd = df[["M10_RP건수_아파트_B0M", "M09_RP후경과월_아파트"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_RP후경과월_아파트"]
    return c == res


@constraint_udf
def cfs_03_1648(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_RP건수_제휴사서비스직접판매_B0M > 0 THEN M10_RP후경과월_제휴사서비스직접판매 = 0
        ELSE M10_RP후경과월_제휴사서비스직접판매 = min(M09_RP후경과월_제휴사서비스직접판매 + 1, 6)
    """
    dd = df[["M10_RP건수_제휴사서비스직접판매_B0M", "M09_RP후경과월_제휴사서비스직접판매"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_RP후경과월_제휴사서비스직접판매"]
    return c == res


@constraint_udf
def cfs_03_1649(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_RP건수_렌탈_B0M > 0 THEN M10_RP후경과월_렌탈 = 0
        ELSE M10_RP후경과월_렌탈 = min(M09_RP후경과월_렌탈 + 1, 6)
    """
    dd = df[["M10_RP건수_렌탈_B0M", "M09_RP후경과월_렌탈"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_RP후경과월_렌탈"]
    return c == res


@constraint_udf
def cfs_03_1650(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_RP건수_가스_B0M > 0 THEN M10_RP후경과월_가스 = 0
        ELSE M10_RP후경과월_가스 = min(M09_RP후경과월_가스 + 1, 6)
    """
    dd = df[["M10_RP건수_가스_B0M", "M09_RP후경과월_가스"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_RP후경과월_가스"]
    return c == res


@constraint_udf
def cfs_03_1651(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_RP건수_전기_B0M > 0 THEN M10_RP후경과월_전기 = 0
        ELSE M10_RP후경과월_전기 = min(M09_RP후경과월_전기 + 1, 6)
    """
    dd = df[["M10_RP건수_전기_B0M", "M09_RP후경과월_전기"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_RP후경과월_전기"]
    return c == res


@constraint_udf
def cfs_03_1652(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_RP건수_보험_B0M > 0 THEN M10_RP후경과월_보험 = 0
        ELSE M10_RP후경과월_보험 = min(M09_RP후경과월_보험 + 1, 6)
    """
    dd = df[["M10_RP건수_보험_B0M", "M09_RP후경과월_보험"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_RP후경과월_보험"]
    return c == res


@constraint_udf
def cfs_03_1653(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_RP건수_학습비_B0M > 0 THEN M10_RP후경과월_학습비 = 0
        ELSE M10_RP후경과월_학습비 = min(M09_RP후경과월_학습비 + 1, 6)
    """
    dd = df[["M10_RP건수_학습비_B0M", "M09_RP후경과월_학습비"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_RP후경과월_학습비"]
    return c == res


@constraint_udf
def cfs_03_1654(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_RP건수_유선방송_B0M > 0 THEN M10_RP후경과월_유선방송 = 0
        ELSE M10_RP후경과월_유선방송 = min(M09_RP후경과월_유선방송 + 1, 6)
    """
    dd = df[["M10_RP건수_유선방송_B0M", "M09_RP후경과월_유선방송"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_RP후경과월_유선방송"]
    return c == res


@constraint_udf
def cfs_03_1655(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_RP건수_건강_B0M > 0 THEN M10_RP후경과월_건강 = 0
        ELSE M10_RP후경과월_건강 = min(M09_RP후경과월_건강 + 1, 6)
    """
    dd = df[["M10_RP건수_건강_B0M", "M09_RP후경과월_건강"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_RP후경과월_건강"]
    return c == res


@constraint_udf
def cfs_03_1656(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_RP건수_교통_B0M > 0 THEN M10_RP후경과월_교통 = 0
        ELSE M10_RP후경과월_교통 = min(M09_RP후경과월_교통 + 1, 6)
    """
    dd = df[["M10_RP건수_교통_B0M", "M09_RP후경과월_교통"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M10_RP후경과월_교통"]
    return c == res


@constraint_udf
def cfs_03_1830(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_승인거절건수_R3M = M08_승인거절건수_B0M + M09_승인거절건수_B0M + M10_승인거절건수_B0M
    """
    dd = df[["M08_승인거절건수_B0M", "M09_승인거절건수_B0M", "M10_승인거절건수_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_승인거절건수_R3M"]
    return c == res


@constraint_udf
def cfs_03_1831(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_승인거절건수_한도초과_R3M = M08_승인거절건수_한도초과_B0M + M09_승인거절건수_한도초과_B0M + M10_승인거절건수_한도초과_B0M
    """
    dd = df[["M08_승인거절건수_한도초과_B0M", "M09_승인거절건수_한도초과_B0M", "M10_승인거절건수_한도초과_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_승인거절건수_한도초과_R3M"]
    return c == res


@constraint_udf
def cfs_03_1832(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_승인거절건수_BL_R3M = M08_승인거절건수_BL_B0M + M09_승인거절건수_BL_B0M + M10_승인거절건수_BL_B0M
    """
    dd = df[["M08_승인거절건수_BL_B0M", "M09_승인거절건수_BL_B0M", "M10_승인거절건수_BL_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_승인거절건수_BL_R3M"]
    return c == res


@constraint_udf
def cfs_03_1833(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_승인거절건수_입력오류_R3M = M08_승인거절건수_입력오류_B0M + M09_승인거절건수_입력오류_B0M + M10_승인거절건수_입력오류_B0M
    """
    dd = df[["M08_승인거절건수_입력오류_B0M", "M09_승인거절건수_입력오류_B0M", "M10_승인거절건수_입력오류_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_승인거절건수_입력오류_R3M"]
    return c == res


@constraint_udf
def cfs_03_1834(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_승인거절건수_기타_R3M = M08_승인거절건수_기타_B0M + M09_승인거절건수_기타_B0M + M10_승인거절건수_기타_B0M
    """
    dd = df[["M08_승인거절건수_기타_B0M", "M09_승인거절건수_기타_B0M", "M10_승인거절건수_기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_승인거절건수_기타_R3M"]
    return c == res


@constraint_udf
def cfs_03_1658(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M10_최종카드론_대출일자 IS NULL
        THEN M10_최종카드론이용경과월 = 999
        ELIF M10_최종카드론_대출일자 = M09_최종카드론_대출일자
        THEN M10_최종카드론이용경과월 = M09_최종카드론이용경과월 + 1
        ELSE M10_최종카드론이용경과월 = 0
    """
    dd = df[["M10_최종카드론_대출일자", "M09_최종카드론_대출일자", "M09_최종카드론이용경과월"]]
    res = dd.apply(lambda x: 999 if pd.isna(x[0]) else (x[2] + 1 if x[0] == x[1] else 0), axis=1)

    c = df["M10_최종카드론이용경과월"]
    return c == res


@constraint_udf
def cfs_03_1679(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_온라인_R6M = M10_이용금액_온라인_R3M + M07_이용금액_온라인_R3M
    """
    dd = df[["M10_이용금액_온라인_R3M", "M07_이용금액_온라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_1680(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_오프라인_R6M = M10_이용금액_오프라인_R3M + M07_이용금액_오프라인_R3M
    """
    dd = df[["M10_이용금액_오프라인_R3M", "M07_이용금액_오프라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_1681(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_온라인_R6M = M10_이용건수_온라인_R3M + M07_이용건수_온라인_R3M
    """
    dd = df[["M10_이용건수_온라인_R3M", "M07_이용건수_온라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_1682(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_오프라인_R6M = M10_이용건수_오프라인_R3M + M07_이용건수_오프라인_R3M
    """
    dd = df[["M10_이용건수_오프라인_R3M", "M07_이용건수_오프라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_1683(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_온라인_R3M = M10_이용금액_온라인_B0M + M09_이용금액_온라인_B0M + M08_이용금액_온라인_B0M
    """
    dd = df[["M10_이용금액_온라인_B0M", "M09_이용금액_온라인_B0M", "M08_이용금액_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1684(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_오프라인_R3M = M10_이용금액_오프라인_B0M + M09_이용금액_오프라인_B0M + M08_이용금액_오프라인_B0M
    """
    dd = df[["M10_이용금액_오프라인_B0M", "M09_이용금액_오프라인_B0M", "M08_이용금액_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1685(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_온라인_R3M = M10_이용건수_온라인_B0M + M09_이용건수_온라인_B0M + M08_이용건수_온라인_B0M
    """
    dd = df[["M10_이용건수_온라인_B0M", "M09_이용건수_온라인_B0M", "M08_이용건수_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1686(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_오프라인_R3M = M10_이용건수_오프라인_B0M + M09_이용건수_오프라인_B0M + M08_이용건수_오프라인_B0M
    """
    dd = df[["M10_이용건수_오프라인_B0M", "M09_이용건수_오프라인_B0M", "M08_이용건수_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1693(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_페이_온라인_R6M = M10_이용금액_페이_온라인_R3M + M07_이용금액_페이_온라인_R3M
    """
    dd = df[["M10_이용금액_페이_온라인_R3M", "M07_이용금액_페이_온라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_페이_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_1694(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_페이_오프라인_R6M = M10_이용금액_페이_오프라인_R3M + M07_이용금액_페이_오프라인_R3M
    """
    dd = df[["M10_이용금액_페이_오프라인_R3M", "M07_이용금액_페이_오프라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_페이_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_1695(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_페이_온라인_R6M = M10_이용건수_페이_온라인_R3M + M07_이용건수_페이_온라인_R3M
    """
    dd = df[["M10_이용건수_페이_온라인_R3M", "M07_이용건수_페이_온라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_페이_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_1696(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_페이_오프라인_R6M = M10_이용건수_페이_오프라인_R3M + M07_이용건수_페이_오프라인_R3M
    """
    dd = df[["M10_이용건수_페이_오프라인_R3M", "M07_이용건수_페이_오프라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_페이_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_1697(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_페이_온라인_R3M = M10_이용금액_페이_온라인_B0M + M09_이용금액_페이_온라인_B0M + M08_이용금액_페이_온라인_B0M
    """
    dd = df[["M10_이용금액_페이_온라인_B0M", "M09_이용금액_페이_온라인_B0M", "M08_이용금액_페이_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_페이_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1698(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_페이_오프라인_R3M = M10_이용금액_페이_오프라인_B0M + M09_이용금액_페이_오프라인_B0M + M08_이용금액_페이_오프라인_B0M
    """
    dd = df[["M10_이용금액_페이_오프라인_B0M", "M09_이용금액_페이_오프라인_B0M", "M08_이용금액_페이_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_페이_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1699(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_페이_온라인_R3M = M10_이용건수_페이_온라인_B0M + M09_이용건수_페이_온라인_B0M + M08_이용건수_페이_온라인_B0M
    """
    dd = df[["M10_이용건수_페이_온라인_B0M", "M09_이용건수_페이_온라인_B0M", "M08_이용건수_페이_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_페이_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1700(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_페이_오프라인_R3M = M10_이용건수_페이_오프라인_B0M + M09_이용건수_페이_오프라인_B0M + M08_이용건수_페이_오프라인_B0M
    """
    dd = df[["M10_이용건수_페이_오프라인_B0M", "M09_이용건수_페이_오프라인_B0M", "M08_이용건수_페이_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_페이_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_1712(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_간편결제_R6M = M10_이용금액_간편결제_R3M + M07_이용금액_간편결제_R3M
    """
    dd = df[["M10_이용금액_간편결제_R3M", "M07_이용금액_간편결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_간편결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_1713(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_당사페이_R6M = M10_이용금액_당사페이_R3M + M07_이용금액_당사페이_R3M
    """
    dd = df[["M10_이용금액_당사페이_R3M", "M07_이용금액_당사페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_당사페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_1714(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_당사기타_R6M = M10_이용금액_당사기타_R3M + M07_이용금액_당사기타_R3M
    """
    dd = df[["M10_이용금액_당사기타_R3M", "M07_이용금액_당사기타_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_당사기타_R6M"]
    return c == res


@constraint_udf
def cfs_03_1715(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_A페이_R6M = M10_이용금액_A페이_R3M + M07_이용금액_A페이_R3M
    """
    dd = df[["M10_이용금액_A페이_R3M", "M07_이용금액_A페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_A페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_1716(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_B페이_R6M = M10_이용금액_B페이_R3M + M07_이용금액_B페이_R3M
    """
    dd = df[["M10_이용금액_B페이_R3M", "M07_이용금액_B페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_B페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_1717(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_C페이_R6M = M10_이용금액_C페이_R3M + M07_이용금액_C페이_R3M
    """
    dd = df[["M10_이용금액_C페이_R3M", "M07_이용금액_C페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_C페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_1718(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_D페이_R6M = M10_이용금액_D페이_R3M + M07_이용금액_D페이_R3M
    """
    dd = df[["M10_이용금액_D페이_R3M", "M07_이용금액_D페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_D페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_1719(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_간편결제_R6M = M10_이용건수_간편결제_R3M + M07_이용건수_간편결제_R3M
    """
    dd = df[["M10_이용건수_간편결제_R3M", "M07_이용건수_간편결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_간편결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_1720(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_당사페이_R6M = M10_이용건수_당사페이_R3M + M07_이용건수_당사페이_R3M
    """
    dd = df[["M10_이용건수_당사페이_R3M", "M07_이용건수_당사페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_당사페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_1721(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_당사기타_R6M = M10_이용건수_당사기타_R3M + M07_이용건수_당사기타_R3M
    """
    dd = df[["M10_이용건수_당사기타_R3M", "M07_이용건수_당사기타_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_당사기타_R6M"]
    return c == res


@constraint_udf
def cfs_03_1722(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_A페이_R6M = M10_이용건수_A페이_R3M + M07_이용건수_A페이_R3M
    """
    dd = df[["M10_이용건수_A페이_R3M", "M07_이용건수_A페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_A페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_1723(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_B페이_R6M = M10_이용건수_B페이_R3M + M07_이용건수_B페이_R3M
    """
    dd = df[["M10_이용건수_B페이_R3M", "M07_이용건수_B페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_B페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_1724(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_C페이_R6M = M10_이용건수_C페이_R3M + M07_이용건수_C페이_R3M
    """
    dd = df[["M10_이용건수_C페이_R3M", "M07_이용건수_C페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_C페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_1725(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_D페이_R6M = M10_이용건수_D페이_R3M + M07_이용건수_D페이_R3M
    """
    dd = df[["M10_이용건수_D페이_R3M", "M07_이용건수_D페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_D페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_1726(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_간편결제_R3M = M10_이용금액_간편결제_B0M + M09_이용금액_간편결제_B0M + M08_이용금액_간편결제_B0M
    """
    dd = df[["M10_이용금액_간편결제_B0M", "M09_이용금액_간편결제_B0M", "M08_이용금액_간편결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_간편결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_1727(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_당사페이_R3M = M10_이용금액_당사페이_B0M + M09_이용금액_당사페이_B0M + M08_이용금액_당사페이_B0M
    """
    dd = df[["M10_이용금액_당사페이_B0M", "M09_이용금액_당사페이_B0M", "M08_이용금액_당사페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_당사페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1728(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_당사기타_R3M = M10_이용금액_당사기타_B0M + M09_이용금액_당사기타_B0M + M08_이용금액_당사기타_B0M
    """
    dd = df[["M10_이용금액_당사기타_B0M", "M09_이용금액_당사기타_B0M", "M08_이용금액_당사기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_당사기타_R3M"]
    return c == res


@constraint_udf
def cfs_03_1729(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_A페이_R3M = M10_이용금액_A페이_B0M + M09_이용금액_A페이_B0M + M08_이용금액_A페이_B0M
    """
    dd = df[["M10_이용금액_A페이_B0M", "M09_이용금액_A페이_B0M", "M08_이용금액_A페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_A페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1730(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_B페이_R3M = M10_이용금액_B페이_B0M + M09_이용금액_B페이_B0M + M08_이용금액_B페이_B0M
    """
    dd = df[["M10_이용금액_B페이_B0M", "M09_이용금액_B페이_B0M", "M08_이용금액_B페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_B페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1731(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_C페이_R3M = M10_이용금액_C페이_B0M + M09_이용금액_C페이_B0M + M08_이용금액_C페이_B0M
    """
    dd = df[["M10_이용금액_C페이_B0M", "M09_이용금액_C페이_B0M", "M08_이용금액_C페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_C페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1732(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_D페이_R3M = M10_이용금액_D페이_B0M + M09_이용금액_D페이_B0M + M08_이용금액_D페이_B0M
    """
    dd = df[["M10_이용금액_D페이_B0M", "M09_이용금액_D페이_B0M", "M08_이용금액_D페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_D페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1733(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_간편결제_R3M = M10_이용건수_간편결제_B0M + M09_이용건수_간편결제_B0M + M08_이용건수_간편결제_B0M
    """
    dd = df[["M10_이용건수_간편결제_B0M", "M09_이용건수_간편결제_B0M", "M08_이용건수_간편결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_간편결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_1734(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_당사페이_R3M = M10_이용건수_당사페이_B0M + M09_이용건수_당사페이_B0M + M08_이용건수_당사페이_B0M
    """
    dd = df[["M10_이용건수_당사페이_B0M", "M09_이용건수_당사페이_B0M", "M08_이용건수_당사페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_당사페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1735(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_당사기타_R3M = M10_이용건수_당사기타_B0M + M09_이용건수_당사기타_B0M + M08_이용건수_당사기타_B0M
    """
    dd = df[["M10_이용건수_당사기타_B0M", "M09_이용건수_당사기타_B0M", "M08_이용건수_당사기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_당사기타_R3M"]
    return c == res


@constraint_udf
def cfs_03_1736(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_A페이_R3M = M10_이용건수_A페이_B0M + M09_이용건수_A페이_B0M + M08_이용건수_A페이_B0M
    """
    dd = df[["M10_이용건수_A페이_B0M", "M09_이용건수_A페이_B0M", "M08_이용건수_A페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_A페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1737(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_B페이_R3M = M10_이용건수_B페이_B0M + M09_이용건수_B페이_B0M + M08_이용건수_B페이_B0M
    """
    dd = df[["M10_이용건수_B페이_B0M", "M09_이용건수_B페이_B0M", "M08_이용건수_B페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_B페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1738(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_C페이_R3M = M10_이용건수_C페이_B0M + M09_이용건수_C페이_B0M + M08_이용건수_C페이_B0M
    """
    dd = df[["M10_이용건수_C페이_B0M", "M09_이용건수_C페이_B0M", "M08_이용건수_C페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_C페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1739(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_D페이_R3M = M10_이용건수_D페이_B0M + M09_이용건수_D페이_B0M + M08_이용건수_D페이_B0M
    """
    dd = df[["M10_이용건수_D페이_B0M", "M09_이용건수_D페이_B0M", "M08_이용건수_D페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_D페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_1755(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용횟수_선결제_R6M = M10_이용횟수_선결제_R3M + M07_이용횟수_선결제_R3M
    """
    dd = df[["M10_이용횟수_선결제_R3M", "M07_이용횟수_선결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용횟수_선결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_1756(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_선결제_R6M = M10_이용금액_선결제_R3M + M07_이용금액_선결제_R3M
    """
    dd = df[["M10_이용금액_선결제_R3M", "M07_이용금액_선결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_선결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_1757(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_선결제_R6M = M10_이용건수_선결제_R3M + M07_이용건수_선결제_R3M
    """
    dd = df[["M10_이용건수_선결제_R3M", "M07_이용건수_선결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_선결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_1758(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용횟수_선결제_R3M = M10_이용횟수_선결제_B0M + M09_이용횟수_선결제_B0M + M08_이용횟수_선결제_B0M
    """
    dd = df[["M10_이용횟수_선결제_B0M", "M09_이용횟수_선결제_B0M", "M08_이용횟수_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용횟수_선결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_1759(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_선결제_R3M = M10_이용금액_선결제_B0M + M09_이용금액_선결제_B0M + M08_이용금액_선결제_B0M
    """
    dd = df[["M10_이용금액_선결제_B0M", "M09_이용금액_선결제_B0M", "M08_이용금액_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_선결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_1760(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용건수_선결제_R3M = M10_이용건수_선결제_B0M + M09_이용건수_선결제_B0M + M08_이용건수_선결제_B0M
    """
    dd = df[["M10_이용건수_선결제_B0M", "M09_이용건수_선결제_B0M", "M08_이용건수_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용건수_선결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_1765(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_가맹점매출금액_B2M = M09_가맹점매출금액_B1M
    """
    res = df["M09_가맹점매출금액_B1M"]

    c = df["M10_가맹점매출금액_B2M"]
    return c == res


@constraint_udf
def cfs_03_1767(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_정상청구원금_B2M = M08_정상청구원금_B0M
    """
    res = df["M08_정상청구원금_B0M"]

    c = df["M10_정상청구원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_1770(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_선입금원금_B2M = M08_선입금원금_B0M
    """
    res = df["M08_선입금원금_B0M"]

    c = df["M10_선입금원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_1773(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_정상입금원금_B2M = M08_정상입금원금_B0M
    """
    res = df["M08_정상입금원금_B0M"]

    c = df["M10_정상입금원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_1776(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_연체입금원금_B2M = M08_연체입금원금_B0M
    """
    res = df["M08_연체입금원금_B0M"]

    c = df["M10_연체입금원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_1782(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용횟수_연체_R6M = M10_이용횟수_연체_R3M + M07_이용횟수_연체_R3M
    """
    dd = df[["M10_이용횟수_연체_R3M", "M07_이용횟수_연체_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용횟수_연체_R6M"]
    return c == res


@constraint_udf
def cfs_03_1783(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_연체_R6M = M10_이용금액_연체_R3M + M07_이용금액_연체_R3M
    """
    dd = df[["M10_이용금액_연체_R3M", "M07_이용금액_연체_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_연체_R6M"]
    return c == res


@constraint_udf
def cfs_03_1784(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용횟수_연체_R3M = M10_이용횟수_연체_B0M + M09_이용횟수_연체_B0M + M08_이용횟수_연체_B0M
    """
    dd = df[["M10_이용횟수_연체_B0M", "M09_이용횟수_연체_B0M", "M08_이용횟수_연체_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용횟수_연체_R3M"]
    return c == res


@constraint_udf
def cfs_03_1785(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_이용금액_연체_R3M = M10_이용금액_연체_B0M + M09_이용금액_연체_B0M + M08_이용금액_연체_B0M
    """
    dd = df[["M10_이용금액_연체_B0M", "M09_이용금액_연체_B0M", "M08_이용금액_연체_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M10_이용금액_연체_R3M"]
    return c == res


@constraint_udf
def cfs_03_1798(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_증감_RP건수_전월 = M09_RP건수_B0M - M10_RP건수_B0M
    """
    dd = df[["M09_RP건수_B0M", "M10_RP건수_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M10_증감_RP건수_전월"]
    return c == res


@constraint_udf
def cfs_03_1800(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_증감_RP건수_통신_전월 = M09_RP건수_통신_B0M - M10_RP건수_통신_B0M
    """
    dd = df[["M09_RP건수_통신_B0M", "M10_RP건수_통신_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M10_증감_RP건수_통신_전월"]
    return c == res


@constraint_udf
def cfs_03_1801(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_증감_RP건수_아파트_전월 = M09_RP건수_아파트_B0M - M10_RP건수_아파트_B0M
    """
    dd = df[["M09_RP건수_아파트_B0M", "M10_RP건수_아파트_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M10_증감_RP건수_아파트_전월"]
    return c == res


@constraint_udf
def cfs_03_1802(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_증감_RP건수_제휴사서비스직접판매_전월 = M09_RP건수_제휴사서비스직접판매_B0M - M10_RP건수_제휴사서비스직접판매_B0M
    """
    dd = df[["M09_RP건수_제휴사서비스직접판매_B0M", "M10_RP건수_제휴사서비스직접판매_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M10_증감_RP건수_제휴사서비스직접판매_전월"]
    return c == res


@constraint_udf
def cfs_03_1803(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_증감_RP건수_렌탈_전월 = M09_RP건수_렌탈_B0M - M10_RP건수_렌탈_B0M
    """
    dd = df[["M09_RP건수_렌탈_B0M", "M10_RP건수_렌탈_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M10_증감_RP건수_렌탈_전월"]
    return c == res


@constraint_udf
def cfs_03_1804(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_증감_RP건수_가스_전월 = M09_RP건수_가스_B0M - M10_RP건수_가스_B0M
    """
    dd = df[["M09_RP건수_가스_B0M", "M10_RP건수_가스_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M10_증감_RP건수_가스_전월"]
    return c == res


@constraint_udf
def cfs_03_1805(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_증감_RP건수_전기_전월 = M09_RP건수_전기_B0M - M10_RP건수_전기_B0M
    """
    dd = df[["M09_RP건수_전기_B0M", "M10_RP건수_전기_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M10_증감_RP건수_전기_전월"]
    return c == res


@constraint_udf
def cfs_03_1806(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_증감_RP건수_보험_전월 = M09_RP건수_보험_B0M - M10_RP건수_보험_B0M
    """
    dd = df[["M09_RP건수_보험_B0M", "M10_RP건수_보험_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M10_증감_RP건수_보험_전월"]
    return c == res


@constraint_udf
def cfs_03_1807(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_증감_RP건수_학습비_전월 = M09_RP건수_학습비_B0M - M10_RP건수_학습비_B0M
    """
    dd = df[["M09_RP건수_학습비_B0M", "M10_RP건수_학습비_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M10_증감_RP건수_학습비_전월"]
    return c == res


@constraint_udf
def cfs_03_1808(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_증감_RP건수_유선방송_전월 = M09_RP건수_유선방송_B0M - M10_RP건수_유선방송_B0M
    """
    dd = df[["M09_RP건수_유선방송_B0M", "M10_RP건수_유선방송_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M10_증감_RP건수_유선방송_전월"]
    return c == res


@constraint_udf
def cfs_03_1809(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_증감_RP건수_건강_전월 = M09_RP건수_건강_B0M - M10_RP건수_건강_B0M
    """
    dd = df[["M09_RP건수_건강_B0M", "M10_RP건수_건강_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M10_증감_RP건수_건강_전월"]
    return c == res


@constraint_udf
def cfs_03_1810(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M10_증감_RP건수_교통_전월 = M09_RP건수_교통_B0M - M10_RP건수_교통_B0M
    """
    dd = df[["M09_RP건수_교통_B0M", "M10_RP건수_교통_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M10_증감_RP건수_교통_전월"]
    return c == res


# 03_M11
@constraint_udf
def cfs_03_1924(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_신용_R6M = M11_이용건수_신용_R3M + M08_이용건수_신용_R3M
    """
    dd = df[["M11_이용건수_신용_R3M", "M08_이용건수_신용_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_신용_R6M"]
    return c == res


@constraint_udf
def cfs_03_1925(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_신판_R6M = M11_이용건수_신판_R3M + M08_이용건수_신판_R3M
    """
    dd = df[["M11_이용건수_신판_R3M", "M08_이용건수_신판_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_신판_R6M"]
    return c == res


@constraint_udf
def cfs_03_1926(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_일시불_R6M = M11_이용건수_일시불_R3M + M08_이용건수_일시불_R3M
    """
    dd = df[["M11_이용건수_일시불_R3M", "M08_이용건수_일시불_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_일시불_R6M"]
    return c == res


@constraint_udf
def cfs_03_1927(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_할부_R6M = M11_이용건수_할부_R3M +  M08_이용건수_할부_R3M
    """
    dd = df[["M11_이용건수_할부_R3M", "M08_이용건수_할부_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_할부_R6M"]
    return c == res


@constraint_udf
def cfs_03_1928(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_할부_유이자_R6M = M11_이용건수_할부_유이자_R3M + M08_이용건수_할부_유이자_R3M
    """
    dd = df[["M11_이용건수_할부_유이자_R3M", "M08_이용건수_할부_유이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_할부_유이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_1929(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_할부_무이자_R6M = M11_이용건수_할부_무이자_R3M + M08_이용건수_할부_무이자_R3M
    """
    dd = df[["M11_이용건수_할부_무이자_R3M", "M08_이용건수_할부_무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_할부_무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_1930(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_부분무이자_R6M = M11_이용건수_부분무이자_R3M + M08_이용건수_부분무이자_R3M
    """
    dd = df[["M11_이용건수_부분무이자_R3M", "M08_이용건수_부분무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_부분무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_1931(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_CA_R6M = M11_이용건수_CA_R3M + M08_이용건수_CA_R3M
    """
    dd = df[["M11_이용건수_CA_R3M", "M08_이용건수_CA_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_CA_R6M"]
    return c == res


@constraint_udf
def cfs_03_1932(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_체크_R6M = M11_이용건수_체크_R3M + M08_이용건수_체크_R3M
    """
    dd = df[["M11_이용건수_체크_R3M", "M08_이용건수_체크_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_체크_R6M"]
    return c == res


@constraint_udf
def cfs_03_1933(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_카드론_R6M = M11_이용건수_카드론_R3M + M08_이용건수_카드론_R3M
    """
    dd = df[["M11_이용건수_카드론_R3M", "M08_이용건수_카드론_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_카드론_R6M"]
    return c == res


@constraint_udf
def cfs_03_1934(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_신용_R6M = M11_이용금액_신용_R3M + M08_이용금액_신용_R3M
    """
    dd = df[["M11_이용금액_신용_R3M", "M08_이용금액_신용_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_신용_R6M"]
    return c == res


@constraint_udf
def cfs_03_1935(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_신판_R6M = M11_이용금액_신판_R3M + M08_이용금액_신판_R3M
    """
    dd = df[["M11_이용금액_신판_R3M", "M08_이용금액_신판_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_신판_R6M"]
    return c == res


@constraint_udf
def cfs_03_1936(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_일시불_R6M = M11_이용금액_일시불_R3M + M08_이용금액_일시불_R3M
    """
    dd = df[["M11_이용금액_일시불_R3M", "M08_이용금액_일시불_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_일시불_R6M"]
    return c == res


@constraint_udf
def cfs_03_1937(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_할부_R6M = M11_이용금액_할부_R3M + M08_이용금액_할부_R3M
    """
    dd = df[["M11_이용금액_할부_R3M", "M08_이용금액_할부_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_할부_R6M"]
    return c == res


@constraint_udf
def cfs_03_1938(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_할부_유이자_R6M = M11_이용금액_할부_유이자_R3M + M08_이용금액_할부_유이자_R3M
    """
    dd = df[["M11_이용금액_할부_유이자_R3M", "M08_이용금액_할부_유이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_할부_유이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_1939(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_할부_무이자_R6M = M11_이용금액_할부_무이자_R3M + M08_이용금액_할부_무이자_R3M
    """
    dd = df[["M11_이용금액_할부_무이자_R3M", "M08_이용금액_할부_무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_할부_무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_1940(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_부분무이자_R6M = M11_이용금액_부분무이자_R3M +  M08_이용금액_부분무이자_R3M
    """
    dd = df[["M11_이용금액_부분무이자_R3M", "M08_이용금액_부분무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_부분무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_1941(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_CA_R6M = M11_이용금액_CA_R3M + M08_이용금액_CA_R3M
    """
    dd = df[["M11_이용금액_CA_R3M", "M08_이용금액_CA_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_CA_R6M"]
    return c == res


@constraint_udf
def cfs_03_1942(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_체크_R6M = M11_이용금액_체크_R3M + M08_이용금액_체크_R3M
    """
    dd = df[["M11_이용금액_체크_R3M", "M08_이용금액_체크_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_체크_R6M"]
    return c == res


@constraint_udf
def cfs_03_1943(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_카드론_R6M = M11_이용금액_카드론_R3M + M08_이용금액_카드론_R3M
    """
    dd = df[["M11_이용금액_카드론_R3M", "M08_이용금액_카드론_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_카드론_R6M"]
    return c == res


@constraint_udf
def cfs_03_1954(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_신용_R3M = M11_이용건수_신용_B0M + M10_이용건수_신용_B0M + M09_이용건수_신용_B0M
    """
    dd = df[["M11_이용건수_신용_B0M", "M10_이용건수_신용_B0M", "M09_이용건수_신용_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_신용_R3M"]
    return c == res


@constraint_udf
def cfs_03_1955(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_신판_R3M = M11_이용건수_신판_B0M + M10_이용건수_신판_B0M + M09_이용건수_신판_B0M
    """
    dd = df[["M11_이용건수_신판_B0M", "M10_이용건수_신판_B0M", "M09_이용건수_신판_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_신판_R3M"]
    return c == res


@constraint_udf
def cfs_03_1956(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_일시불_R3M = M11_이용건수_일시불_B0M + M10_이용건수_일시불_B0M + M09_이용건수_일시불_B0M
    """
    dd = df[["M11_이용건수_일시불_B0M", "M10_이용건수_일시불_B0M", "M09_이용건수_일시불_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_일시불_R3M"]
    return c == res


@constraint_udf
def cfs_03_1957(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_할부_R3M = M11_이용건수_할부_B0M + M10_이용건수_할부_B0M + M09_이용건수_할부_B0M
    """
    dd = df[["M11_이용건수_할부_B0M", "M10_이용건수_할부_B0M", "M09_이용건수_할부_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_할부_R3M"]
    return c == res


@constraint_udf
def cfs_03_1958(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_할부_유이자_R3M = M11_이용건수_할부_유이자_B0M + M10_이용건수_할부_유이자_B0M + M09_이용건수_할부_유이자_B0M
    """
    dd = df[["M11_이용건수_할부_유이자_B0M", "M10_이용건수_할부_유이자_B0M", "M09_이용건수_할부_유이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_할부_유이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1959(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_할부_무이자_R3M = M11_이용건수_할부_무이자_B0M + M10_이용건수_할부_무이자_B0M + M09_이용건수_할부_무이자_B0M
    """
    dd = df[["M11_이용건수_할부_무이자_B0M", "M10_이용건수_할부_무이자_B0M", "M09_이용건수_할부_무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_할부_무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1960(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_부분무이자_R3M = M11_이용건수_부분무이자_B0M + M10_이용건수_부분무이자_B0M + M09_이용건수_부분무이자_B0M
    """
    dd = df[["M11_이용건수_부분무이자_B0M", "M10_이용건수_부분무이자_B0M", "M09_이용건수_부분무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_부분무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1961(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_CA_R3M = M11_이용건수_CA_B0M + M10_이용건수_CA_B0M + M09_이용건수_CA_B0M
    """
    dd = df[["M11_이용건수_CA_B0M", "M10_이용건수_CA_B0M", "M09_이용건수_CA_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_CA_R3M"]
    return c == res


@constraint_udf
def cfs_03_1962(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_체크_R3M = M11_이용건수_체크_B0M + M10_이용건수_체크_B0M + M09_이용건수_체크_B0M
    """
    dd = df[["M11_이용건수_체크_B0M", "M10_이용건수_체크_B0M", "M09_이용건수_체크_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_체크_R3M"]
    return c == res


@constraint_udf
def cfs_03_1963(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_카드론_R3M = M11_이용건수_카드론_B0M + M10_이용건수_카드론_B0M + M09_이용건수_카드론_B0M
    """
    dd = df[["M11_이용건수_카드론_B0M", "M10_이용건수_카드론_B0M", "M09_이용건수_카드론_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_카드론_R3M"]
    return c == res


@constraint_udf
def cfs_03_1964(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_신용_R3M = M11_이용금액_신용_B0M + M10_이용금액_신용_B0M + M09_이용금액_신용_B0M
    """
    dd = df[["M11_이용금액_신용_B0M", "M10_이용금액_신용_B0M", "M09_이용금액_신용_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_신용_R3M"]
    return c == res


@constraint_udf
def cfs_03_1965(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_신판_R3M = M11_이용금액_신판_B0M + M10_이용금액_신판_B0M + M09_이용금액_신판_B0M
    """
    dd = df[["M11_이용금액_신판_B0M", "M10_이용금액_신판_B0M", "M09_이용금액_신판_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_신판_R3M"]
    return c == res


@constraint_udf
def cfs_03_1966(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_일시불_R3M = M11_이용금액_일시불_B0M + M10_이용금액_일시불_B0M + M09_이용금액_일시불_B0M
    """
    dd = df[["M11_이용금액_일시불_B0M", "M10_이용금액_일시불_B0M", "M09_이용금액_일시불_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_일시불_R3M"]
    return c == res


@constraint_udf
def cfs_03_1967(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_할부_R3M = M11_이용금액_할부_B0M + M10_이용금액_할부_B0M + M09_이용금액_할부_B0M
    """
    dd = df[["M11_이용금액_할부_B0M", "M10_이용금액_할부_B0M", "M09_이용금액_할부_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_할부_R3M"]
    return c == res


@constraint_udf
def cfs_03_1968(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_할부_유이자_R3M = M11_이용금액_할부_유이자_B0M + M10_이용금액_할부_유이자_B0M + M09_이용금액_할부_유이자_B0M
    """
    dd = df[["M11_이용금액_할부_유이자_B0M", "M10_이용금액_할부_유이자_B0M", "M09_이용금액_할부_유이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_할부_유이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1969(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_할부_무이자_R3M = M11_이용금액_할부_무이자_B0M + M10_이용금액_할부_무이자_B0M + M09_이용금액_할부_무이자_B0M
    """
    dd = df[["M11_이용금액_할부_무이자_B0M", "M10_이용금액_할부_무이자_B0M", "M09_이용금액_할부_무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_할부_무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1970(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_부분무이자_R3M = M11_이용금액_부분무이자_B0M + M10_이용금액_부분무이자_B0M + M09_이용금액_부분무이자_B0M
    """
    dd = df[["M11_이용금액_부분무이자_B0M", "M10_이용금액_부분무이자_B0M", "M09_이용금액_부분무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_부분무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_1971(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_CA_R3M = M11_이용금액_CA_B0M + M10_이용금액_CA_B0M + M09_이용금액_CA_B0M
    """
    dd = df[["M11_이용금액_CA_B0M", "M10_이용금액_CA_B0M", "M09_이용금액_CA_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_CA_R3M"]
    return c == res


@constraint_udf
def cfs_03_1972(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_체크_R3M = M11_이용금액_체크_B0M + M10_이용금액_체크_B0M + M09_이용금액_체크_B0M
    """
    dd = df[["M11_이용금액_체크_B0M", "M10_이용금액_체크_B0M", "M09_이용금액_체크_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_체크_R3M"]
    return c == res


@constraint_udf
def cfs_03_1973(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_카드론_R3M = M11_이용금액_카드론_B0M + M10_이용금액_카드론_B0M + M09_이용금액_카드론_B0M
    """
    dd = df[["M11_이용금액_카드론_B0M", "M10_이용금액_카드론_B0M", "M09_이용금액_카드론_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_카드론_R3M"]
    return c == res


@constraint_udf
def cfs_03_1988(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_건수_할부전환_R6M = M11_건수_할부전환_R3M + M08_건수_할부전환_R3M
    """
    dd = df[["M11_건수_할부전환_R3M", "M08_건수_할부전환_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_건수_할부전환_R6M"]
    return c == res


@constraint_udf
def cfs_03_1989(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_금액_할부전환_R6M = M11_금액_할부전환_R3M + M08_금액_할부전환_R3M
    """
    dd = df[["M11_금액_할부전환_R3M", "M08_금액_할부전환_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_금액_할부전환_R6M"]
    return c == res


@constraint_udf
def cfs_03_2106(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_RP건수_B0M > 0 THEN M11_RP후경과월 = 0
        ELSE M11_RP후경과월 = min(M10_RP후경과월 + 1, 6)
    """
    dd = df[["M11_RP건수_B0M", "M10_RP후경과월"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_RP후경과월"]
    return c == res


@constraint_udf
def cfs_03_2107(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_RP건수_통신_B0M > 0 THEN M11_RP후경과월_통신 = 0
        ELSE M11_RP후경과월_통신 = min(M10_RP후경과월_통신 + 1, 6)
    """
    dd = df[["M11_RP건수_통신_B0M", "M10_RP후경과월_통신"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_RP후경과월_통신"]
    return c == res


@constraint_udf
def cfs_03_2108(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_RP건수_아파트_B0M > 0 THEN M11_RP후경과월_아파트 = 0
        ELSE M11_RP후경과월_아파트 = min(M10_RP후경과월_아파트 + 1, 6)
    """
    dd = df[["M11_RP건수_아파트_B0M", "M10_RP후경과월_아파트"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_RP후경과월_아파트"]
    return c == res


@constraint_udf
def cfs_03_2109(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_RP건수_제휴사서비스직접판매_B0M > 0 THEN M11_RP후경과월_제휴사서비스직접판매 = 0
        ELSE M11_RP후경과월_제휴사서비스직접판매 = min(M10_RP후경과월_제휴사서비스직접판매 + 1, 6)
    """
    dd = df[["M11_RP건수_제휴사서비스직접판매_B0M", "M10_RP후경과월_제휴사서비스직접판매"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_RP후경과월_제휴사서비스직접판매"]
    return c == res


@constraint_udf
def cfs_03_2110(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_RP건수_렌탈_B0M > 0 THEN M11_RP후경과월_렌탈 = 0
        ELSE M11_RP후경과월_렌탈 = min(M10_RP후경과월_렌탈 + 1, 6)
    """
    dd = df[["M11_RP건수_렌탈_B0M", "M10_RP후경과월_렌탈"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_RP후경과월_렌탈"]
    return c == res


@constraint_udf
def cfs_03_2111(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_RP건수_가스_B0M > 0 THEN M11_RP후경과월_가스 = 0
        ELSE M11_RP후경과월_가스 = min(M10_RP후경과월_가스 + 1, 6)
    """
    dd = df[["M11_RP건수_가스_B0M", "M10_RP후경과월_가스"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_RP후경과월_가스"]
    return c == res


@constraint_udf
def cfs_03_2112(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_RP건수_전기_B0M > 0 THEN M11_RP후경과월_전기 = 0
        ELSE M11_RP후경과월_전기 = min(M10_RP후경과월_전기 + 1, 6)
    """
    dd = df[["M11_RP건수_전기_B0M", "M10_RP후경과월_전기"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_RP후경과월_전기"]
    return c == res


@constraint_udf
def cfs_03_2113(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_RP건수_보험_B0M > 0 THEN M11_RP후경과월_보험 = 0
        ELSE M11_RP후경과월_보험 = min(M10_RP후경과월_보험 + 1, 6)
    """
    dd = df[["M11_RP건수_보험_B0M", "M10_RP후경과월_보험"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_RP후경과월_보험"]
    return c == res


@constraint_udf
def cfs_03_2114(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_RP건수_학습비_B0M > 0 THEN M11_RP후경과월_학습비 = 0
        ELSE M11_RP후경과월_학습비 = min(M10_RP후경과월_학습비 + 1, 6)
    """
    dd = df[["M11_RP건수_학습비_B0M", "M10_RP후경과월_학습비"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_RP후경과월_학습비"]
    return c == res


@constraint_udf
def cfs_03_2115(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_RP건수_유선방송_B0M > 0 THEN M11_RP후경과월_유선방송 = 0
        ELSE M11_RP후경과월_유선방송 = min(M10_RP후경과월_유선방송 + 1, 6)
    """
    dd = df[["M11_RP건수_유선방송_B0M", "M10_RP후경과월_유선방송"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_RP후경과월_유선방송"]
    return c == res


@constraint_udf
def cfs_03_2116(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_RP건수_건강_B0M > 0 THEN M11_RP후경과월_건강 = 0
        ELSE M11_RP후경과월_건강 = min(M10_RP후경과월_건강 + 1, 6)
    """
    dd = df[["M11_RP건수_건강_B0M", "M10_RP후경과월_건강"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_RP후경과월_건강"]
    return c == res


@constraint_udf
def cfs_03_2117(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_RP건수_교통_B0M > 0 THEN M11_RP후경과월_교통 = 0
        ELSE M11_RP후경과월_교통 = min(M10_RP후경과월_교통 + 1, 6)
    """
    dd = df[["M11_RP건수_교통_B0M", "M10_RP후경과월_교통"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M11_RP후경과월_교통"]
    return c == res


@constraint_udf
def cfs_03_2291(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_승인거절건수_R3M = M09_승인거절건수_B0M + M10_승인거절건수_B0M + M11_승인거절건수_B0M
    """
    dd = df[["M09_승인거절건수_B0M", "M10_승인거절건수_B0M", "M11_승인거절건수_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_승인거절건수_R3M"]
    return c == res


@constraint_udf
def cfs_03_2292(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_승인거절건수_한도초과_R3M = M09_승인거절건수_한도초과_B0M + M10_승인거절건수_한도초과_B0M + M11_승인거절건수_한도초과_B0M
    """
    dd = df[["M09_승인거절건수_한도초과_B0M", "M10_승인거절건수_한도초과_B0M", "M11_승인거절건수_한도초과_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_승인거절건수_한도초과_R3M"]
    return c == res


@constraint_udf
def cfs_03_2293(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_승인거절건수_BL_R3M = M09_승인거절건수_BL_B0M + M10_승인거절건수_BL_B0M + M11_승인거절건수_BL_B0M
    """
    dd = df[["M09_승인거절건수_BL_B0M", "M10_승인거절건수_BL_B0M", "M11_승인거절건수_BL_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_승인거절건수_BL_R3M"]
    return c == res


@constraint_udf
def cfs_03_2294(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_승인거절건수_입력오류_R3M = M09_승인거절건수_입력오류_B0M + M10_승인거절건수_입력오류_B0M + M11_승인거절건수_입력오류_B0M
    """
    dd = df[["M09_승인거절건수_입력오류_B0M", "M10_승인거절건수_입력오류_B0M", "M11_승인거절건수_입력오류_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_승인거절건수_입력오류_R3M"]
    return c == res


@constraint_udf
def cfs_03_2295(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_승인거절건수_기타_R3M = M09_승인거절건수_기타_B0M + M10_승인거절건수_기타_B0M + M11_승인거절건수_기타_B0M
    """
    dd = df[["M09_승인거절건수_기타_B0M", "M10_승인거절건수_기타_B0M", "M11_승인거절건수_기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_승인거절건수_기타_R3M"]
    return c == res


@constraint_udf
def cfs_03_2119(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M11_최종카드론_대출일자 IS NULL
        THEN M11_최종카드론이용경과월 = 999
        ELIF M11_최종카드론_대출일자 = M10_최종카드론_대출일자
        THEN M11_최종카드론이용경과월 = M10_최종카드론이용경과월 + 1
        ELSE M11_최종카드론이용경과월 = 0
    """
    dd = df[["M11_최종카드론_대출일자", "M10_최종카드론_대출일자", "M10_최종카드론이용경과월"]]
    res = dd.apply(lambda x: 999 if pd.isna(x[0]) else (x[2] + 1 if x[0] == x[1] else 0), axis=1)

    c = df["M11_최종카드론이용경과월"]
    return c == res


@constraint_udf
def cfs_03_2140(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_온라인_R6M = M11_이용금액_온라인_R3M + M08_이용금액_온라인_R3M
    """
    dd = df[["M11_이용금액_온라인_R3M", "M08_이용금액_온라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2141(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_오프라인_R6M = M11_이용금액_오프라인_R3M + M08_이용금액_오프라인_R3M
    """
    dd = df[["M11_이용금액_오프라인_R3M", "M08_이용금액_오프라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2142(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_온라인_R6M = M11_이용건수_온라인_R3M + M08_이용건수_온라인_R3M
    """
    dd = df[["M11_이용건수_온라인_R3M", "M08_이용건수_온라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2143(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_오프라인_R6M = M11_이용건수_오프라인_R3M + M08_이용건수_오프라인_R3M
    """
    dd = df[["M11_이용건수_오프라인_R3M", "M08_이용건수_오프라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2144(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_온라인_R3M = M11_이용금액_온라인_B0M + M10_이용금액_온라인_B0M + M09_이용금액_온라인_B0M
    """
    dd = df[["M11_이용금액_온라인_B0M", "M10_이용금액_온라인_B0M", "M09_이용금액_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2145(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_오프라인_R3M = M11_이용금액_오프라인_B0M + M10_이용금액_오프라인_B0M + M09_이용금액_오프라인_B0M
    """
    dd = df[["M11_이용금액_오프라인_B0M", "M10_이용금액_오프라인_B0M", "M09_이용금액_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2146(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_온라인_R3M = M11_이용건수_온라인_B0M + M10_이용건수_온라인_B0M + M09_이용건수_온라인_B0M
    """
    dd = df[["M11_이용건수_온라인_B0M", "M10_이용건수_온라인_B0M", "M09_이용건수_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2147(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_오프라인_R3M = M11_이용건수_오프라인_B0M + M10_이용건수_오프라인_B0M + M09_이용건수_오프라인_B0M
    """
    dd = df[["M11_이용건수_오프라인_B0M", "M10_이용건수_오프라인_B0M", "M09_이용건수_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2154(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_페이_온라인_R6M = M11_이용금액_페이_온라인_R3M + M08_이용금액_페이_온라인_R3M
    """
    dd = df[["M11_이용금액_페이_온라인_R3M", "M08_이용금액_페이_온라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_페이_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2155(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_페이_오프라인_R6M = M11_이용금액_페이_오프라인_R3M + M08_이용금액_페이_오프라인_R3M
    """
    dd = df[["M11_이용금액_페이_오프라인_R3M", "M08_이용금액_페이_오프라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_페이_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2156(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_페이_온라인_R6M = M11_이용건수_페이_온라인_R3M + M08_이용건수_페이_온라인_R3M
    """
    dd = df[["M11_이용건수_페이_온라인_R3M", "M08_이용건수_페이_온라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_페이_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2157(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_페이_오프라인_R6M = M11_이용건수_페이_오프라인_R3M + M08_이용건수_페이_오프라인_R3M
    """
    dd = df[["M11_이용건수_페이_오프라인_R3M", "M08_이용건수_페이_오프라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_페이_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2158(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_페이_온라인_R3M = M11_이용금액_페이_온라인_B0M + M10_이용금액_페이_온라인_B0M + M09_이용금액_페이_온라인_B0M
    """
    dd = df[["M11_이용금액_페이_온라인_B0M", "M10_이용금액_페이_온라인_B0M", "M09_이용금액_페이_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_페이_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2159(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_페이_오프라인_R3M = M11_이용금액_페이_오프라인_B0M + M10_이용금액_페이_오프라인_B0M + M09_이용금액_페이_오프라인_B0M
    """
    dd = df[["M11_이용금액_페이_오프라인_B0M", "M10_이용금액_페이_오프라인_B0M", "M09_이용금액_페이_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_페이_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2160(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_페이_온라인_R3M = M11_이용건수_페이_온라인_B0M + M10_이용건수_페이_온라인_B0M + M09_이용건수_페이_온라인_B0M
    """
    dd = df[["M11_이용건수_페이_온라인_B0M", "M10_이용건수_페이_온라인_B0M", "M09_이용건수_페이_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_페이_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2161(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_페이_오프라인_R3M = M11_이용건수_페이_오프라인_B0M + M10_이용건수_페이_오프라인_B0M + M09_이용건수_페이_오프라인_B0M
    """
    dd = df[["M11_이용건수_페이_오프라인_B0M", "M10_이용건수_페이_오프라인_B0M", "M09_이용건수_페이_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_페이_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2173(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_간편결제_R6M = M11_이용금액_간편결제_R3M + M08_이용금액_간편결제_R3M
    """
    dd = df[["M11_이용금액_간편결제_R3M", "M08_이용금액_간편결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_간편결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_2174(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_당사페이_R6M = M11_이용금액_당사페이_R3M + M08_이용금액_당사페이_R3M
    """
    dd = df[["M11_이용금액_당사페이_R3M", "M08_이용금액_당사페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_당사페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2175(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_당사기타_R6M = M11_이용금액_당사기타_R3M + M08_이용금액_당사기타_R3M
    """
    dd = df[["M11_이용금액_당사기타_R3M", "M08_이용금액_당사기타_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_당사기타_R6M"]
    return c == res


@constraint_udf
def cfs_03_2176(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_A페이_R6M = M11_이용금액_A페이_R3M + M08_이용금액_A페이_R3M
    """
    dd = df[["M11_이용금액_A페이_R3M", "M08_이용금액_A페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_A페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2177(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_B페이_R6M = M11_이용금액_B페이_R3M + M08_이용금액_B페이_R3M
    """
    dd = df[["M11_이용금액_B페이_R3M", "M08_이용금액_B페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_B페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2178(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_C페이_R6M = M11_이용금액_C페이_R3M + M08_이용금액_C페이_R3M
    """
    dd = df[["M11_이용금액_C페이_R3M", "M08_이용금액_C페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_C페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2179(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_D페이_R6M = M11_이용금액_D페이_R3M + M08_이용금액_D페이_R3M
    """
    dd = df[["M11_이용금액_D페이_R3M", "M08_이용금액_D페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_D페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2180(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_간편결제_R6M = M11_이용건수_간편결제_R3M + M08_이용건수_간편결제_R3M
    """
    dd = df[["M11_이용건수_간편결제_R3M", "M08_이용건수_간편결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_간편결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_2181(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_당사페이_R6M = M11_이용건수_당사페이_R3M + M08_이용건수_당사페이_R3M
    """
    dd = df[["M11_이용건수_당사페이_R3M", "M08_이용건수_당사페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_당사페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2182(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_당사기타_R6M = M11_이용건수_당사기타_R3M + M08_이용건수_당사기타_R3M
    """
    dd = df[["M11_이용건수_당사기타_R3M", "M08_이용건수_당사기타_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_당사기타_R6M"]
    return c == res


@constraint_udf
def cfs_03_2183(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_A페이_R6M = M11_이용건수_A페이_R3M + M08_이용건수_A페이_R3M
    """
    dd = df[["M11_이용건수_A페이_R3M", "M08_이용건수_A페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_A페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2184(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_B페이_R6M = M11_이용건수_B페이_R3M + M08_이용건수_B페이_R3M
    """
    dd = df[["M11_이용건수_B페이_R3M", "M08_이용건수_B페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_B페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2185(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_C페이_R6M = M11_이용건수_C페이_R3M + M08_이용건수_C페이_R3M
    """
    dd = df[["M11_이용건수_C페이_R3M", "M08_이용건수_C페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_C페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2186(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_D페이_R6M = M11_이용건수_D페이_R3M + M08_이용건수_D페이_R3M
    """
    dd = df[["M11_이용건수_D페이_R3M", "M08_이용건수_D페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_D페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2187(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_간편결제_R3M = M11_이용금액_간편결제_B0M + M10_이용금액_간편결제_B0M + M09_이용금액_간편결제_B0M
    """
    dd = df[["M11_이용금액_간편결제_B0M", "M10_이용금액_간편결제_B0M", "M09_이용금액_간편결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_간편결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_2188(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_당사페이_R3M = M11_이용금액_당사페이_B0M + M10_이용금액_당사페이_B0M + M09_이용금액_당사페이_B0M
    """
    dd = df[["M11_이용금액_당사페이_B0M", "M10_이용금액_당사페이_B0M", "M09_이용금액_당사페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_당사페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2189(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_당사기타_R3M = M11_이용금액_당사기타_B0M + M10_이용금액_당사기타_B0M + M09_이용금액_당사기타_B0M
    """
    dd = df[["M11_이용금액_당사기타_B0M", "M10_이용금액_당사기타_B0M", "M09_이용금액_당사기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_당사기타_R3M"]
    return c == res


@constraint_udf
def cfs_03_2190(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_A페이_R3M = M11_이용금액_A페이_B0M + M10_이용금액_A페이_B0M + M09_이용금액_A페이_B0M
    """
    dd = df[["M11_이용금액_A페이_B0M", "M10_이용금액_A페이_B0M", "M09_이용금액_A페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_A페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2191(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_B페이_R3M = M11_이용금액_B페이_B0M + M10_이용금액_B페이_B0M + M09_이용금액_B페이_B0M
    """
    dd = df[["M11_이용금액_B페이_B0M", "M10_이용금액_B페이_B0M", "M09_이용금액_B페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_B페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2192(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_C페이_R3M = M11_이용금액_C페이_B0M + M10_이용금액_C페이_B0M + M09_이용금액_C페이_B0M
    """
    dd = df[["M11_이용금액_C페이_B0M", "M10_이용금액_C페이_B0M", "M09_이용금액_C페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_C페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2193(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_D페이_R3M = M11_이용금액_D페이_B0M + M10_이용금액_D페이_B0M + M09_이용금액_D페이_B0M
    """
    dd = df[["M11_이용금액_D페이_B0M", "M10_이용금액_D페이_B0M", "M09_이용금액_D페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_D페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2194(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_간편결제_R3M = M11_이용건수_간편결제_B0M + M10_이용건수_간편결제_B0M + M09_이용건수_간편결제_B0M
    """
    dd = df[["M11_이용건수_간편결제_B0M", "M10_이용건수_간편결제_B0M", "M09_이용건수_간편결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_간편결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_2195(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_당사페이_R3M = M11_이용건수_당사페이_B0M + M10_이용건수_당사페이_B0M + M09_이용건수_당사페이_B0M
    """
    dd = df[["M11_이용건수_당사페이_B0M", "M10_이용건수_당사페이_B0M", "M09_이용건수_당사페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_당사페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2196(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_당사기타_R3M = M11_이용건수_당사기타_B0M + M10_이용건수_당사기타_B0M + M09_이용건수_당사기타_B0M
    """
    dd = df[["M11_이용건수_당사기타_B0M", "M10_이용건수_당사기타_B0M", "M09_이용건수_당사기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_당사기타_R3M"]
    return c == res


@constraint_udf
def cfs_03_2197(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_A페이_R3M = M11_이용건수_A페이_B0M + M10_이용건수_A페이_B0M + M09_이용건수_A페이_B0M
    """
    dd = df[["M11_이용건수_A페이_B0M", "M10_이용건수_A페이_B0M", "M09_이용건수_A페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_A페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2198(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_B페이_R3M = M11_이용건수_B페이_B0M + M10_이용건수_B페이_B0M + M09_이용건수_B페이_B0M
    """
    dd = df[["M11_이용건수_B페이_B0M", "M10_이용건수_B페이_B0M", "M09_이용건수_B페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_B페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2199(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_C페이_R3M = M11_이용건수_C페이_B0M + M10_이용건수_C페이_B0M + M09_이용건수_C페이_B0M
    """
    dd = df[["M11_이용건수_C페이_B0M", "M10_이용건수_C페이_B0M", "M09_이용건수_C페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_C페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2200(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_D페이_R3M = M11_이용건수_D페이_B0M + M10_이용건수_D페이_B0M + M09_이용건수_D페이_B0M
    """
    dd = df[["M11_이용건수_D페이_B0M", "M10_이용건수_D페이_B0M", "M09_이용건수_D페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_D페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2216(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용횟수_선결제_R6M = M11_이용횟수_선결제_R3M + M08_이용횟수_선결제_R3M
    """
    dd = df[["M11_이용횟수_선결제_R3M", "M08_이용횟수_선결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용횟수_선결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_2217(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_선결제_R6M = M11_이용금액_선결제_R3M + M08_이용금액_선결제_R3M
    """
    dd = df[["M11_이용금액_선결제_R3M", "M08_이용금액_선결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_선결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_2218(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_선결제_R6M = M11_이용건수_선결제_R3M + M08_이용건수_선결제_R3M
    """
    dd = df[["M11_이용건수_선결제_R3M", "M08_이용건수_선결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_선결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_2219(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용횟수_선결제_R3M = M11_이용횟수_선결제_B0M + M10_이용횟수_선결제_B0M + M09_이용횟수_선결제_B0M
    """
    dd = df[["M11_이용횟수_선결제_B0M", "M10_이용횟수_선결제_B0M", "M09_이용횟수_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용횟수_선결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_2220(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_선결제_R3M = M11_이용금액_선결제_B0M + M10_이용금액_선결제_B0M + M09_이용금액_선결제_B0M
    """
    dd = df[["M11_이용금액_선결제_B0M", "M10_이용금액_선결제_B0M", "M09_이용금액_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_선결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_2221(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용건수_선결제_R3M = M11_이용건수_선결제_B0M + M10_이용건수_선결제_B0M + M09_이용건수_선결제_B0M
    """
    dd = df[["M11_이용건수_선결제_B0M", "M10_이용건수_선결제_B0M", "M09_이용건수_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용건수_선결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_2226(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_가맹점매출금액_B2M = M10_가맹점매출금액_B1M
    """
    res = df["M10_가맹점매출금액_B1M"]

    c = df["M11_가맹점매출금액_B2M"]
    return c == res


@constraint_udf
def cfs_03_2228(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_정상청구원금_B2M = M09_정상청구원금_B0M
    """
    res = df["M09_정상청구원금_B0M"]

    c = df["M11_정상청구원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_2231(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_선입금원금_B2M = M09_선입금원금_B0M
    """
    res = df["M09_선입금원금_B0M"]

    c = df["M11_선입금원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_2234(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_정상입금원금_B2M = M09_정상입금원금_B0M
    """
    res = df["M09_정상입금원금_B0M"]

    c = df["M11_정상입금원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_2237(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_연체입금원금_B2M = M09_연체입금원금_B0M
    """
    res = df["M09_연체입금원금_B0M"]

    c = df["M11_연체입금원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_2243(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용횟수_연체_R6M = M11_이용횟수_연체_R3M + M08_이용횟수_연체_R3M
    """
    dd = df[["M11_이용횟수_연체_R3M", "M08_이용횟수_연체_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용횟수_연체_R6M"]
    return c == res


@constraint_udf
def cfs_03_2244(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_연체_R6M = M11_이용금액_연체_R3M + M08_이용금액_연체_R3M
    """
    dd = df[["M11_이용금액_연체_R3M", "M08_이용금액_연체_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_연체_R6M"]
    return c == res


@constraint_udf
def cfs_03_2245(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용횟수_연체_R3M = M11_이용횟수_연체_B0M + M10_이용횟수_연체_B0M + M09_이용횟수_연체_B0M
    """
    dd = df[["M11_이용횟수_연체_B0M", "M10_이용횟수_연체_B0M", "M09_이용횟수_연체_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용횟수_연체_R3M"]
    return c == res


@constraint_udf
def cfs_03_2246(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_이용금액_연체_R3M = M11_이용금액_연체_B0M + M10_이용금액_연체_B0M + M09_이용금액_연체_B0M
    """
    dd = df[["M11_이용금액_연체_B0M", "M10_이용금액_연체_B0M", "M09_이용금액_연체_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M11_이용금액_연체_R3M"]
    return c == res


@constraint_udf
def cfs_03_2259(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_증감_RP건수_전월 = M10_RP건수_B0M - M11_RP건수_B0M
    """
    dd = df[["M10_RP건수_B0M", "M11_RP건수_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M11_증감_RP건수_전월"]
    return c == res


@constraint_udf
def cfs_03_2261(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_증감_RP건수_통신_전월 = M10_RP건수_통신_B0M - M11_RP건수_통신_B0M
    """
    dd = df[["M10_RP건수_통신_B0M", "M11_RP건수_통신_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M11_증감_RP건수_통신_전월"]
    return c == res


@constraint_udf
def cfs_03_2262(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_증감_RP건수_아파트_전월 = M10_RP건수_아파트_B0M - M11_RP건수_아파트_B0M
    """
    dd = df[["M10_RP건수_아파트_B0M", "M11_RP건수_아파트_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M11_증감_RP건수_아파트_전월"]
    return c == res


@constraint_udf
def cfs_03_2263(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_증감_RP건수_제휴사서비스직접판매_전월 = M10_RP건수_제휴사서비스직접판매_B0M - M11_RP건수_제휴사서비스직접판매_B0M
    """
    dd = df[["M10_RP건수_제휴사서비스직접판매_B0M", "M11_RP건수_제휴사서비스직접판매_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M11_증감_RP건수_제휴사서비스직접판매_전월"]
    return c == res


@constraint_udf
def cfs_03_2264(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_증감_RP건수_렌탈_전월 = M10_RP건수_렌탈_B0M - M11_RP건수_렌탈_B0M
    """
    dd = df[["M10_RP건수_렌탈_B0M", "M11_RP건수_렌탈_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M11_증감_RP건수_렌탈_전월"]
    return c == res


@constraint_udf
def cfs_03_2265(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_증감_RP건수_가스_전월 = M10_RP건수_가스_B0M - M11_RP건수_가스_B0M
    """
    dd = df[["M10_RP건수_가스_B0M", "M11_RP건수_가스_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M11_증감_RP건수_가스_전월"]
    return c == res


@constraint_udf
def cfs_03_2266(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_증감_RP건수_전기_전월 = M10_RP건수_전기_B0M - M11_RP건수_전기_B0M
    """
    dd = df[["M10_RP건수_전기_B0M", "M11_RP건수_전기_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M11_증감_RP건수_전기_전월"]
    return c == res


@constraint_udf
def cfs_03_2267(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_증감_RP건수_보험_전월 = M10_RP건수_보험_B0M - M11_RP건수_보험_B0M
    """
    dd = df[["M10_RP건수_보험_B0M", "M11_RP건수_보험_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M11_증감_RP건수_보험_전월"]
    return c == res


@constraint_udf
def cfs_03_2268(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_증감_RP건수_학습비_전월 = M10_RP건수_학습비_B0M - M11_RP건수_학습비_B0M
    """
    dd = df[["M10_RP건수_학습비_B0M", "M11_RP건수_학습비_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M11_증감_RP건수_학습비_전월"]
    return c == res


@constraint_udf
def cfs_03_2269(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_증감_RP건수_유선방송_전월 = M10_RP건수_유선방송_B0M - M11_RP건수_유선방송_B0M
    """
    dd = df[["M10_RP건수_유선방송_B0M", "M11_RP건수_유선방송_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M11_증감_RP건수_유선방송_전월"]
    return c == res


@constraint_udf
def cfs_03_2270(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_증감_RP건수_건강_전월 = M10_RP건수_건강_B0M - M11_RP건수_건강_B0M
    """
    dd = df[["M10_RP건수_건강_B0M", "M11_RP건수_건강_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M11_증감_RP건수_건강_전월"]
    return c == res


@constraint_udf
def cfs_03_2271(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M11_증감_RP건수_교통_전월 = M10_RP건수_교통_B0M - M11_RP건수_교통_B0M
    """
    dd = df[["M10_RP건수_교통_B0M", "M11_RP건수_교통_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M11_증감_RP건수_교통_전월"]
    return c == res


# 03_M12
@constraint_udf
def cfs_03_2385(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_신용_R6M = M12_이용건수_신용_R3M + M09_이용건수_신용_R3M
    """
    dd = df[["M12_이용건수_신용_R3M", "M09_이용건수_신용_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_신용_R6M"]
    return c == res


@constraint_udf
def cfs_03_2386(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_신판_R6M = M12_이용건수_신판_R3M + M09_이용건수_신판_R3M
    """
    dd = df[["M12_이용건수_신판_R3M", "M09_이용건수_신판_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_신판_R6M"]
    return c == res


@constraint_udf
def cfs_03_2387(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_일시불_R6M = M12_이용건수_일시불_R3M + M09_이용건수_일시불_R3M
    """
    dd = df[["M12_이용건수_일시불_R3M", "M09_이용건수_일시불_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_일시불_R6M"]
    return c == res


@constraint_udf
def cfs_03_2388(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_할부_R6M = M12_이용건수_할부_R3M +  M09_이용건수_할부_R3M
    """
    dd = df[["M12_이용건수_할부_R3M", "M09_이용건수_할부_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_할부_R6M"]
    return c == res


@constraint_udf
def cfs_03_2389(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_할부_유이자_R6M = M12_이용건수_할부_유이자_R3M + M09_이용건수_할부_유이자_R3M
    """
    dd = df[["M12_이용건수_할부_유이자_R3M", "M09_이용건수_할부_유이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_할부_유이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_2390(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_할부_무이자_R6M = M12_이용건수_할부_무이자_R3M + M09_이용건수_할부_무이자_R3M
    """
    dd = df[["M12_이용건수_할부_무이자_R3M", "M09_이용건수_할부_무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_할부_무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_2391(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_부분무이자_R6M = M12_이용건수_부분무이자_R3M + M09_이용건수_부분무이자_R3M
    """
    dd = df[["M12_이용건수_부분무이자_R3M", "M09_이용건수_부분무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_부분무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_2392(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_CA_R6M = M12_이용건수_CA_R3M + M09_이용건수_CA_R3M
    """
    dd = df[["M12_이용건수_CA_R3M", "M09_이용건수_CA_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_CA_R6M"]
    return c == res


@constraint_udf
def cfs_03_2393(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_체크_R6M = M12_이용건수_체크_R3M + M09_이용건수_체크_R3M
    """
    dd = df[["M12_이용건수_체크_R3M", "M09_이용건수_체크_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_체크_R6M"]
    return c == res


@constraint_udf
def cfs_03_2394(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_카드론_R6M = M12_이용건수_카드론_R3M + M09_이용건수_카드론_R3M
    """
    dd = df[["M12_이용건수_카드론_R3M", "M09_이용건수_카드론_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_카드론_R6M"]
    return c == res


@constraint_udf
def cfs_03_2395(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_신용_R6M = M12_이용금액_신용_R3M + M09_이용금액_신용_R3M
    """
    dd = df[["M12_이용금액_신용_R3M", "M09_이용금액_신용_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_신용_R6M"]
    return c == res


@constraint_udf
def cfs_03_2396(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_신판_R6M = M12_이용금액_신판_R3M + M09_이용금액_신판_R3M
    """
    dd = df[["M12_이용금액_신판_R3M", "M09_이용금액_신판_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_신판_R6M"]
    return c == res


@constraint_udf
def cfs_03_2397(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_일시불_R6M = M12_이용금액_일시불_R3M + M09_이용금액_일시불_R3M
    """
    dd = df[["M12_이용금액_일시불_R3M", "M09_이용금액_일시불_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_일시불_R6M"]
    return c == res


@constraint_udf
def cfs_03_2398(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_할부_R6M = M12_이용금액_할부_R3M + M09_이용금액_할부_R3M
    """
    dd = df[["M12_이용금액_할부_R3M", "M09_이용금액_할부_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_할부_R6M"]
    return c == res


@constraint_udf
def cfs_03_2399(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_할부_유이자_R6M = M12_이용금액_할부_유이자_R3M + M09_이용금액_할부_유이자_R3M
    """
    dd = df[["M12_이용금액_할부_유이자_R3M", "M09_이용금액_할부_유이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_할부_유이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_2400(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_할부_무이자_R6M = M12_이용금액_할부_무이자_R3M + M09_이용금액_할부_무이자_R3M
    """
    dd = df[["M12_이용금액_할부_무이자_R3M", "M09_이용금액_할부_무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_할부_무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_2401(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_부분무이자_R6M = M12_이용금액_부분무이자_R3M +  M09_이용금액_부분무이자_R3M
    """
    dd = df[["M12_이용금액_부분무이자_R3M", "M09_이용금액_부분무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_부분무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_2402(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_CA_R6M = M12_이용금액_CA_R3M + M09_이용금액_CA_R3M
    """
    dd = df[["M12_이용금액_CA_R3M", "M09_이용금액_CA_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_CA_R6M"]
    return c == res


@constraint_udf
def cfs_03_2403(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_체크_R6M = M12_이용금액_체크_R3M + M09_이용금액_체크_R3M
    """
    dd = df[["M12_이용금액_체크_R3M", "M09_이용금액_체크_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_체크_R6M"]
    return c == res


@constraint_udf
def cfs_03_2404(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_카드론_R6M = M12_이용금액_카드론_R3M + M09_이용금액_카드론_R3M
    """
    dd = df[["M12_이용금액_카드론_R3M", "M09_이용금액_카드론_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_카드론_R6M"]
    return c == res


@constraint_udf
def cfs_03_2405(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_신용_R6M = M12_이용개월수_신용_R3M + M09_이용개월수_신용_R3M
    """
    dd = df[["M12_이용개월수_신용_R3M", "M09_이용개월수_신용_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용개월수_신용_R6M"]
    return c == res


@constraint_udf
def cfs_03_2406(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_신판_R6M = M12_이용개월수_신판_R3M + M09_이용개월수_신판_R3M
    """
    dd = df[["M12_이용개월수_신판_R3M", "M09_이용개월수_신판_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용개월수_신판_R6M"]
    return c == res


@constraint_udf
def cfs_03_2407(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_일시불_R6M = M12_이용개월수_일시불_R3M + M09_이용개월수_일시불_R3M
    """
    dd = df[["M12_이용개월수_일시불_R3M", "M09_이용개월수_일시불_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용개월수_일시불_R6M"]
    return c == res


@constraint_udf
def cfs_03_2408(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_할부_R6M = M12_이용개월수_할부_R3M + M09_이용개월수_할부_R3M
    """
    dd = df[["M12_이용개월수_할부_R3M", "M09_이용개월수_할부_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용개월수_할부_R6M"]
    return c == res


@constraint_udf
def cfs_03_2409(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_할부_유이자_R6M = M12_이용개월수_할부_유이자_R3M + M09_이용개월수_할부_유이자_R3M
    """
    dd = df[["M12_이용개월수_할부_유이자_R3M", "M09_이용개월수_할부_유이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용개월수_할부_유이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_2410(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_할부_무이자_R6M = M12_이용개월수_할부_무이자_R3M + M09_이용개월수_할부_무이자_R3M
    """
    dd = df[["M12_이용개월수_할부_무이자_R3M", "M09_이용개월수_할부_무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용개월수_할부_무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_2411(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_부분무이자_R6M = M12_이용개월수_부분무이자_R3M + M09_이용개월수_부분무이자_R3M
    """
    dd = df[["M12_이용개월수_부분무이자_R3M", "M09_이용개월수_부분무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용개월수_부분무이자_R6M"]
    return c == res


@constraint_udf
def cfs_03_2412(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_CA_R6M = M12_이용개월수_CA_R3M + M09_이용개월수_CA_R3M
    """
    dd = df[["M12_이용개월수_CA_R3M", "M09_이용개월수_CA_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용개월수_CA_R6M"]
    return c == res


@constraint_udf
def cfs_03_2413(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_체크_R6M = M12_이용개월수_체크_R3M + M09_이용개월수_체크_R3M
    """
    dd = df[["M12_이용개월수_체크_R3M", "M09_이용개월수_체크_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용개월수_체크_R6M"]
    return c == res


@constraint_udf
def cfs_03_2414(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_카드론_R6M = M12_이용개월수_카드론_R3M + M09_이용개월수_카드론_R3M
    """
    dd = df[["M12_이용개월수_카드론_R3M", "M09_이용개월수_카드론_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용개월수_카드론_R6M"]
    return c == res


@constraint_udf
def cfs_03_2415(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_신용_R3M = M12_이용건수_신용_B0M + M11_이용건수_신용_B0M + M10_이용건수_신용_B0M
    """
    dd = df[["M12_이용건수_신용_B0M", "M11_이용건수_신용_B0M", "M10_이용건수_신용_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_신용_R3M"]
    return c == res


@constraint_udf
def cfs_03_2416(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_신판_R3M = M12_이용건수_신판_B0M + M11_이용건수_신판_B0M + M10_이용건수_신판_B0M
    """
    dd = df[["M12_이용건수_신판_B0M", "M11_이용건수_신판_B0M", "M10_이용건수_신판_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_신판_R3M"]
    return c == res


@constraint_udf
def cfs_03_2417(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_일시불_R3M = M12_이용건수_일시불_B0M + M11_이용건수_일시불_B0M + M10_이용건수_일시불_B0M
    """
    dd = df[["M12_이용건수_일시불_B0M", "M11_이용건수_일시불_B0M", "M10_이용건수_일시불_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_일시불_R3M"]
    return c == res


@constraint_udf
def cfs_03_2418(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_할부_R3M = M12_이용건수_할부_B0M + M11_이용건수_할부_B0M + M10_이용건수_할부_B0M
    """
    dd = df[["M12_이용건수_할부_B0M", "M11_이용건수_할부_B0M", "M10_이용건수_할부_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_할부_R3M"]
    return c == res


@constraint_udf
def cfs_03_2419(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_할부_유이자_R3M = M12_이용건수_할부_유이자_B0M + M11_이용건수_할부_유이자_B0M + M10_이용건수_할부_유이자_B0M
    """
    dd = df[["M12_이용건수_할부_유이자_B0M", "M11_이용건수_할부_유이자_B0M", "M10_이용건수_할부_유이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_할부_유이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_2420(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_할부_무이자_R3M = M12_이용건수_할부_무이자_B0M + M11_이용건수_할부_무이자_B0M + M10_이용건수_할부_무이자_B0M
    """
    dd = df[["M12_이용건수_할부_무이자_B0M", "M11_이용건수_할부_무이자_B0M", "M10_이용건수_할부_무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_할부_무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_2421(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_부분무이자_R3M = M12_이용건수_부분무이자_B0M + M11_이용건수_부분무이자_B0M + M10_이용건수_부분무이자_B0M
    """
    dd = df[["M12_이용건수_부분무이자_B0M", "M11_이용건수_부분무이자_B0M", "M10_이용건수_부분무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_부분무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_2422(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_CA_R3M = M12_이용건수_CA_B0M + M11_이용건수_CA_B0M + M10_이용건수_CA_B0M
    """
    dd = df[["M12_이용건수_CA_B0M", "M11_이용건수_CA_B0M", "M10_이용건수_CA_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_CA_R3M"]
    return c == res


@constraint_udf
def cfs_03_2423(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_체크_R3M = M12_이용건수_체크_B0M + M11_이용건수_체크_B0M + M10_이용건수_체크_B0M
    """
    dd = df[["M12_이용건수_체크_B0M", "M11_이용건수_체크_B0M", "M10_이용건수_체크_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_체크_R3M"]
    return c == res


@constraint_udf
def cfs_03_2424(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_카드론_R3M = M12_이용건수_카드론_B0M + M11_이용건수_카드론_B0M + M10_이용건수_카드론_B0M
    """
    dd = df[["M12_이용건수_카드론_B0M", "M11_이용건수_카드론_B0M", "M10_이용건수_카드론_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_카드론_R3M"]
    return c == res


@constraint_udf
def cfs_03_2425(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_신용_R3M = M12_이용금액_신용_B0M + M11_이용금액_신용_B0M + M10_이용금액_신용_B0M
    """
    dd = df[["M12_이용금액_신용_B0M", "M11_이용금액_신용_B0M", "M10_이용금액_신용_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_신용_R3M"]
    return c == res


@constraint_udf
def cfs_03_2426(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_신판_R3M = M12_이용금액_신판_B0M + M11_이용금액_신판_B0M + M10_이용금액_신판_B0M
    """
    dd = df[["M12_이용금액_신판_B0M", "M11_이용금액_신판_B0M", "M10_이용금액_신판_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_신판_R3M"]
    return c == res


@constraint_udf
def cfs_03_2427(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_일시불_R3M = M12_이용금액_일시불_B0M + M11_이용금액_일시불_B0M + M10_이용금액_일시불_B0M
    """
    dd = df[["M12_이용금액_일시불_B0M", "M11_이용금액_일시불_B0M", "M10_이용금액_일시불_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_일시불_R3M"]
    return c == res


@constraint_udf
def cfs_03_2428(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_할부_R3M = M12_이용금액_할부_B0M + M11_이용금액_할부_B0M + M10_이용금액_할부_B0M
    """
    dd = df[["M12_이용금액_할부_B0M", "M11_이용금액_할부_B0M", "M10_이용금액_할부_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_할부_R3M"]
    return c == res


@constraint_udf
def cfs_03_2429(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_할부_유이자_R3M = M12_이용금액_할부_유이자_B0M + M11_이용금액_할부_유이자_B0M + M10_이용금액_할부_유이자_B0M
    """
    dd = df[["M12_이용금액_할부_유이자_B0M", "M11_이용금액_할부_유이자_B0M", "M10_이용금액_할부_유이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_할부_유이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_2430(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_할부_무이자_R3M = M12_이용금액_할부_무이자_B0M + M11_이용금액_할부_무이자_B0M + M10_이용금액_할부_무이자_B0M
    """
    dd = df[["M12_이용금액_할부_무이자_B0M", "M11_이용금액_할부_무이자_B0M", "M10_이용금액_할부_무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_할부_무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_2431(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_부분무이자_R3M = M12_이용금액_부분무이자_B0M + M11_이용금액_부분무이자_B0M + M10_이용금액_부분무이자_B0M
    """
    dd = df[["M12_이용금액_부분무이자_B0M", "M11_이용금액_부분무이자_B0M", "M10_이용금액_부분무이자_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_부분무이자_R3M"]
    return c == res


@constraint_udf
def cfs_03_2432(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_CA_R3M = M12_이용금액_CA_B0M + M11_이용금액_CA_B0M + M10_이용금액_CA_B0M
    """
    dd = df[["M12_이용금액_CA_B0M", "M11_이용금액_CA_B0M", "M10_이용금액_CA_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_CA_R3M"]
    return c == res


@constraint_udf
def cfs_03_2433(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_체크_R3M = M12_이용금액_체크_B0M + M11_이용금액_체크_B0M + M10_이용금액_체크_B0M
    """
    dd = df[["M12_이용금액_체크_B0M", "M11_이용금액_체크_B0M", "M10_이용금액_체크_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_체크_R3M"]
    return c == res


@constraint_udf
def cfs_03_2434(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_카드론_R3M = M12_이용금액_카드론_B0M + M11_이용금액_카드론_B0M + M10_이용금액_카드론_B0M
    """
    dd = df[["M12_이용금액_카드론_B0M", "M11_이용금액_카드론_B0M", "M10_이용금액_카드론_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_카드론_R3M"]
    return c == res


@constraint_udf
def cfs_03_2449(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_건수_할부전환_R6M = M12_건수_할부전환_R3M + M09_건수_할부전환_R3M
    """
    dd = df[["M12_건수_할부전환_R3M", "M09_건수_할부전환_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_건수_할부전환_R6M"]
    return c == res


@constraint_udf
def cfs_03_2450(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_금액_할부전환_R6M = M12_금액_할부전환_R3M + M09_금액_할부전환_R3M
    """
    dd = df[["M12_금액_할부전환_R3M", "M09_금액_할부전환_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_금액_할부전환_R6M"]
    return c == res


@constraint_udf
def cfs_03_2567(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_RP건수_B0M > 0 THEN M12_RP후경과월 = 0
        ELSE M12_RP후경과월 = min(M11_RP후경과월 + 1, 6)
    """
    dd = df[["M12_RP건수_B0M", "M11_RP후경과월"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_RP후경과월"]
    return c == res


@constraint_udf
def cfs_03_2568(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_RP건수_통신_B0M > 0 THEN M12_RP후경과월_통신 = 0
        ELSE M12_RP후경과월_통신 = min(M11_RP후경과월_통신 + 1, 6)
    """
    dd = df[["M12_RP건수_통신_B0M", "M11_RP후경과월_통신"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_RP후경과월_통신"]
    return c == res


@constraint_udf
def cfs_03_2569(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_RP건수_아파트_B0M > 0 THEN M12_RP후경과월_아파트 = 0
        ELSE M12_RP후경과월_아파트 = min(M11_RP후경과월_아파트 + 1, 6)
    """
    dd = df[["M12_RP건수_아파트_B0M", "M11_RP후경과월_아파트"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_RP후경과월_아파트"]
    return c == res


@constraint_udf
def cfs_03_2570(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_RP건수_제휴사서비스직접판매_B0M > 0 THEN M12_RP후경과월_제휴사서비스직접판매 = 0
        ELSE M12_RP후경과월_제휴사서비스직접판매 = min(M11_RP후경과월_제휴사서비스직접판매 + 1, 6)
    """
    dd = df[["M12_RP건수_제휴사서비스직접판매_B0M", "M11_RP후경과월_제휴사서비스직접판매"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_RP후경과월_제휴사서비스직접판매"]
    return c == res


@constraint_udf
def cfs_03_2571(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_RP건수_렌탈_B0M > 0 THEN M12_RP후경과월_렌탈 = 0
        ELSE M12_RP후경과월_렌탈 = min(M11_RP후경과월_렌탈 + 1, 6)
    """
    dd = df[["M12_RP건수_렌탈_B0M", "M11_RP후경과월_렌탈"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_RP후경과월_렌탈"]
    return c == res


@constraint_udf
def cfs_03_2572(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_RP건수_가스_B0M > 0 THEN M12_RP후경과월_가스 = 0
        ELSE M12_RP후경과월_가스 = min(M11_RP후경과월_가스 + 1, 6)
    """
    dd = df[["M12_RP건수_가스_B0M", "M11_RP후경과월_가스"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_RP후경과월_가스"]
    return c == res


@constraint_udf
def cfs_03_2573(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_RP건수_전기_B0M > 0 THEN M12_RP후경과월_전기 = 0
        ELSE M12_RP후경과월_전기 = min(M11_RP후경과월_전기 + 1, 6)
    """
    dd = df[["M12_RP건수_전기_B0M", "M11_RP후경과월_전기"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_RP후경과월_전기"]
    return c == res


@constraint_udf
def cfs_03_2574(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_RP건수_보험_B0M > 0 THEN M12_RP후경과월_보험 = 0
        ELSE M12_RP후경과월_보험 = min(M11_RP후경과월_보험 + 1, 6)
    """
    dd = df[["M12_RP건수_보험_B0M", "M11_RP후경과월_보험"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_RP후경과월_보험"]
    return c == res


@constraint_udf
def cfs_03_2575(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_RP건수_학습비_B0M > 0 THEN M12_RP후경과월_학습비 = 0
        ELSE M12_RP후경과월_학습비 = min(M11_RP후경과월_학습비 + 1, 6)
    """
    dd = df[["M12_RP건수_학습비_B0M", "M11_RP후경과월_학습비"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_RP후경과월_학습비"]
    return c == res


@constraint_udf
def cfs_03_2576(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_RP건수_유선방송_B0M > 0 THEN M12_RP후경과월_유선방송 = 0
        ELSE M12_RP후경과월_유선방송 = min(M11_RP후경과월_유선방송 + 1, 6)
    """
    dd = df[["M12_RP건수_유선방송_B0M", "M11_RP후경과월_유선방송"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_RP후경과월_유선방송"]
    return c == res


@constraint_udf
def cfs_03_2577(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_RP건수_건강_B0M > 0 THEN M12_RP후경과월_건강 = 0
        ELSE M12_RP후경과월_건강 = min(M11_RP후경과월_건강 + 1, 6)
    """
    dd = df[["M12_RP건수_건강_B0M", "M11_RP후경과월_건강"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_RP후경과월_건강"]
    return c == res


@constraint_udf
def cfs_03_2578(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_RP건수_교통_B0M > 0 THEN M12_RP후경과월_교통 = 0
        ELSE M12_RP후경과월_교통 = min(M11_RP후경과월_교통 + 1, 6)
    """
    dd = df[["M12_RP건수_교통_B0M", "M11_RP후경과월_교통"]]
    res = dd.apply(lambda x: 0 if x[0] > 0 else min(x[1] + 1, 6), axis=1)

    c = df["M12_RP후경과월_교통"]
    return c == res


@constraint_udf
def cfs_03_2752(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_승인거절건수_R3M = M10_승인거절건수_B0M + M11_승인거절건수_B0M + M12_승인거절건수_B0M
    """
    dd = df[["M10_승인거절건수_B0M", "M11_승인거절건수_B0M", "M12_승인거절건수_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_승인거절건수_R3M"]
    return c == res


@constraint_udf
def cfs_03_2753(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_승인거절건수_한도초과_R3M = M10_승인거절건수_한도초과_B0M + M11_승인거절건수_한도초과_B0M + M12_승인거절건수_한도초과_B0M
    """
    dd = df[["M10_승인거절건수_한도초과_B0M", "M11_승인거절건수_한도초과_B0M", "M12_승인거절건수_한도초과_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_승인거절건수_한도초과_R3M"]
    return c == res


@constraint_udf
def cfs_03_2754(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_승인거절건수_BL_R3M = M10_승인거절건수_BL_B0M + M11_승인거절건수_BL_B0M + M12_승인거절건수_BL_B0M
    """
    dd = df[["M10_승인거절건수_BL_B0M", "M11_승인거절건수_BL_B0M", "M12_승인거절건수_BL_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_승인거절건수_BL_R3M"]
    return c == res


@constraint_udf
def cfs_03_2755(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_승인거절건수_입력오류_R3M = M10_승인거절건수_입력오류_B0M + M11_승인거절건수_입력오류_B0M + M12_승인거절건수_입력오류_B0M
    """
    dd = df[["M10_승인거절건수_입력오류_B0M", "M11_승인거절건수_입력오류_B0M", "M12_승인거절건수_입력오류_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_승인거절건수_입력오류_R3M"]
    return c == res


@constraint_udf
def cfs_03_2756(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_승인거절건수_기타_R3M = M10_승인거절건수_기타_B0M + M11_승인거절건수_기타_B0M + M12_승인거절건수_기타_B0M
    """
    dd = df[["M10_승인거절건수_기타_B0M", "M11_승인거절건수_기타_B0M", "M12_승인거절건수_기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_승인거절건수_기타_R3M"]
    return c == res


@constraint_udf
def cfs_03_2580(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF M12_최종카드론_대출일자 IS NULL
        THEN M12_최종카드론이용경과월 = 999
        ELIF M12_최종카드론_대출일자 = M11_최종카드론_대출일자
        THEN M12_최종카드론이용경과월 = M11_최종카드론이용경과월 + 1
        ELSE M12_최종카드론이용경과월 = 0
    """
    dd = df[["M12_최종카드론_대출일자", "M11_최종카드론_대출일자", "M11_최종카드론이용경과월"]]
    res = dd.apply(lambda x: 999 if pd.isna(x[0]) else (x[2] + 1 if x[0] == x[1] else 0), axis=1)

    c = df["M12_최종카드론이용경과월"]
    return c == res


@constraint_udf
def cfs_03_2596(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_신청건수_ATM_CA_R6M = SUM(M07~M12_신청건수_ATM_CA_B0)
    """
    dd = df[["M12_신청건수_ATM_CA_B0",
             "M11_신청건수_ATM_CA_B0",
             "M10_신청건수_ATM_CA_B0",
             "M09_신청건수_ATM_CA_B0",
             "M08_신청건수_ATM_CA_B0",
             "M07_신청건수_ATM_CA_B0"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_신청건수_ATM_CA_R6M"]
    return c == res


@constraint_udf
def cfs_03_2597(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_신청건수_ATM_CL_R6M = SUM(M07~M12_신청건수_ATM_CL_B0)
    """
    dd = df[["M12_신청건수_ATM_CL_B0",
             "M11_신청건수_ATM_CL_B0",
             "M10_신청건수_ATM_CL_B0",
             "M09_신청건수_ATM_CL_B0",
             "M08_신청건수_ATM_CL_B0",
             "M07_신청건수_ATM_CL_B0"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_신청건수_ATM_CL_R6M"]
    return c == res


@constraint_udf
def cfs_03_2599(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_온라인_R6M = SUM(1 IF M0X_이용건수_온라인_B0M > 0 ELSE 0)
    """
    dd = df[["M12_이용건수_온라인_B0M",
             "M11_이용건수_온라인_B0M",
             "M10_이용건수_온라인_B0M",
             "M09_이용건수_온라인_B0M",
             "M08_이용건수_온라인_B0M",
             "M07_이용건수_온라인_B0M"]]
    res = dd.apply(lambda x: sum(x > 0), axis=1)

    c = df["M12_이용개월수_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2600(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_오프라인_R6M = SUM(1 IF M0X_이용건수_오프라인_B0M > 0 ELSE 0)
    """
    dd = df[["M12_이용건수_오프라인_B0M",
             "M11_이용건수_오프라인_B0M",
             "M10_이용건수_오프라인_B0M",
             "M09_이용건수_오프라인_B0M",
             "M08_이용건수_오프라인_B0M",
             "M07_이용건수_오프라인_B0M"]]
    res = dd.apply(lambda x: sum(x > 0), axis=1)

    c = df["M12_이용개월수_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2601(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_온라인_R6M = M12_이용금액_온라인_R3M + M09_이용금액_온라인_R3M
    """
    dd = df[["M12_이용금액_온라인_R3M", "M09_이용금액_온라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2602(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_오프라인_R6M = M12_이용금액_오프라인_R3M + M09_이용금액_오프라인_R3M
    """
    dd = df[["M12_이용금액_오프라인_R3M", "M09_이용금액_오프라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2603(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_온라인_R6M = M12_이용건수_온라인_R3M + M09_이용건수_온라인_R3M
    """
    dd = df[["M12_이용건수_온라인_R3M", "M09_이용건수_온라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2604(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_오프라인_R6M = M12_이용건수_오프라인_R3M + M09_이용건수_오프라인_R3M
    """
    dd = df[["M12_이용건수_오프라인_R3M", "M09_이용건수_오프라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2605(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_온라인_R3M = M12_이용금액_온라인_B0M + M11_이용금액_온라인_B0M + M10_이용금액_온라인_B0M
    """
    dd = df[["M12_이용금액_온라인_B0M", "M11_이용금액_온라인_B0M", "M10_이용금액_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2606(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_오프라인_R3M = M12_이용금액_오프라인_B0M + M11_이용금액_오프라인_B0M + M10_이용금액_오프라인_B0M
    """
    dd = df[["M12_이용금액_오프라인_B0M", "M11_이용금액_오프라인_B0M", "M10_이용금액_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2607(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_온라인_R3M = M12_이용건수_온라인_B0M + M11_이용건수_온라인_B0M + M10_이용건수_온라인_B0M
    """
    dd = df[["M12_이용건수_온라인_B0M", "M11_이용건수_온라인_B0M", "M10_이용건수_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2608(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_오프라인_R3M = M12_이용건수_오프라인_B0M + M11_이용건수_오프라인_B0M + M10_이용건수_오프라인_B0M
    """
    dd = df[["M12_이용건수_오프라인_B0M", "M11_이용건수_오프라인_B0M", "M10_이용건수_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2613(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_페이_온라인_R6M = SUM(1 IF M0X_이용건수_페이_온라인_B0M > 0 ELSE 0)
    """
    dd = df[["M12_이용건수_페이_온라인_B0M",
             "M11_이용건수_페이_온라인_B0M",
             "M10_이용건수_페이_온라인_B0M",
             "M09_이용건수_페이_온라인_B0M",
             "M08_이용건수_페이_온라인_B0M",
             "M07_이용건수_페이_온라인_B0M"]]
    res = dd.apply(lambda x: sum(x > 0), axis=1)

    c = df["M12_이용개월수_페이_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2614(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_페이_오프라인_R6M = SUM(1 IF M0X_이용건수_페이_오프라인_B0M > 0 ELSE 0)
    """
    dd = df[["M12_이용건수_페이_오프라인_B0M",
             "M11_이용건수_페이_오프라인_B0M",
             "M10_이용건수_페이_오프라인_B0M",
             "M09_이용건수_페이_오프라인_B0M",
             "M08_이용건수_페이_오프라인_B0M",
             "M07_이용건수_페이_오프라인_B0M"]]
    res = dd.apply(lambda x: sum(x > 0), axis=1)

    c = df["M12_이용개월수_페이_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2615(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_페이_온라인_R6M = M12_이용금액_페이_온라인_R3M + M09_이용금액_페이_온라인_R3M
    """
    dd = df[["M12_이용금액_페이_온라인_R3M", "M09_이용금액_페이_온라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_페이_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2616(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_페이_오프라인_R6M = M12_이용금액_페이_오프라인_R3M + M09_이용금액_페이_오프라인_R3M
    """
    dd = df[["M12_이용금액_페이_오프라인_R3M", "M09_이용금액_페이_오프라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_페이_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2617(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_페이_온라인_R6M = M12_이용건수_페이_온라인_R3M + M09_이용건수_페이_온라인_R3M
    """
    dd = df[["M12_이용건수_페이_온라인_R3M", "M09_이용건수_페이_온라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_페이_온라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2618(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_페이_오프라인_R6M = M12_이용건수_페이_오프라인_R3M + M09_이용건수_페이_오프라인_R3M
    """
    dd = df[["M12_이용건수_페이_오프라인_R3M", "M09_이용건수_페이_오프라인_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_페이_오프라인_R6M"]
    return c == res


@constraint_udf
def cfs_03_2619(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_페이_온라인_R3M = M12_이용금액_페이_온라인_B0M + M11_이용금액_페이_온라인_B0M + M10_이용금액_페이_온라인_B0M
    """
    dd = df[["M12_이용금액_페이_온라인_B0M", "M11_이용금액_페이_온라인_B0M", "M10_이용금액_페이_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_페이_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2620(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_페이_오프라인_R3M = M12_이용금액_페이_오프라인_B0M + M11_이용금액_페이_오프라인_B0M + M10_이용금액_페이_오프라인_B0M
    """
    dd = df[["M12_이용금액_페이_오프라인_B0M", "M11_이용금액_페이_오프라인_B0M", "M10_이용금액_페이_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_페이_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2621(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_페이_온라인_R3M = M12_이용건수_페이_온라인_B0M + M11_이용건수_페이_온라인_B0M + M10_이용건수_페이_온라인_B0M
    """
    dd = df[["M12_이용건수_페이_온라인_B0M", "M11_이용건수_페이_온라인_B0M", "M10_이용건수_페이_온라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_페이_온라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2622(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_페이_오프라인_R3M = M12_이용건수_페이_오프라인_B0M + M11_이용건수_페이_오프라인_B0M + M10_이용건수_페이_오프라인_B0M
    """
    dd = df[["M12_이용건수_페이_오프라인_B0M", "M11_이용건수_페이_오프라인_B0M", "M10_이용건수_페이_오프라인_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_페이_오프라인_R3M"]
    return c == res


@constraint_udf
def cfs_03_2627(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_간편결제_R6M = SUM(1 IF M0X_이용건수_간편결제_B0M > 0 ELSE 0)
    """
    dd = df[["M12_이용건수_간편결제_B0M",
             "M11_이용건수_간편결제_B0M",
             "M10_이용건수_간편결제_B0M",
             "M09_이용건수_간편결제_B0M",
             "M08_이용건수_간편결제_B0M",
             "M07_이용건수_간편결제_B0M"]]
    res = dd.apply(lambda x: sum(x > 0), axis=1)

    c = df["M12_이용개월수_간편결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_2628(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_당사페이_R6M = SUM(1 IF M0X_이용건수_당사페이_B0M > 0 ELSE 0)
    """
    dd = df[["M12_이용건수_당사페이_B0M",
             "M11_이용건수_당사페이_B0M",
             "M10_이용건수_당사페이_B0M",
             "M09_이용건수_당사페이_B0M",
             "M08_이용건수_당사페이_B0M",
             "M07_이용건수_당사페이_B0M"]]
    res = dd.apply(lambda x: sum(x > 0), axis=1)

    c = df["M12_이용개월수_당사페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2629(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_당사기타_R6M = SUM(1 IF M0X_이용건수_당사기타_B0M > 0 ELSE 0)
    """
    dd = df[["M12_이용건수_당사기타_B0M",
             "M11_이용건수_당사기타_B0M",
             "M10_이용건수_당사기타_B0M",
             "M09_이용건수_당사기타_B0M",
             "M08_이용건수_당사기타_B0M",
             "M07_이용건수_당사기타_B0M"]]
    res = dd.apply(lambda x: sum(x > 0), axis=1)

    c = df["M12_이용개월수_당사기타_R6M"]
    return c == res


@constraint_udf
def cfs_03_2630(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_A페이_R6M = SUM(1 IF M0X_이용건수_A페이_B0M > 0 ELSE 0)
    """
    dd = df[["M12_이용건수_A페이_B0M",
             "M11_이용건수_A페이_B0M",
             "M10_이용건수_A페이_B0M",
             "M09_이용건수_A페이_B0M",
             "M08_이용건수_A페이_B0M",
             "M07_이용건수_A페이_B0M"]]
    res = dd.apply(lambda x: sum(x > 0), axis=1)

    c = df["M12_이용개월수_A페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2631(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_B페이_R6M = SUM(1 IF M0X_이용건수_B페이_B0M > 0 ELSE 0)
    """
    dd = df[["M12_이용건수_B페이_B0M",
             "M11_이용건수_B페이_B0M",
             "M10_이용건수_B페이_B0M",
             "M09_이용건수_B페이_B0M",
             "M08_이용건수_B페이_B0M",
             "M07_이용건수_B페이_B0M"]]
    res = dd.apply(lambda x: sum(x > 0), axis=1)

    c = df["M12_이용개월수_B페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2632(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_C페이_R6M = SUM(1 IF M0X_이용건수_C페이_B0M > 0 ELSE 0)
    """
    dd = df[["M12_이용건수_C페이_B0M",
             "M11_이용건수_C페이_B0M",
             "M10_이용건수_C페이_B0M",
             "M09_이용건수_C페이_B0M",
             "M08_이용건수_C페이_B0M",
             "M07_이용건수_C페이_B0M"]]
    res = dd.apply(lambda x: sum(x > 0), axis=1)

    c = df["M12_이용개월수_C페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2633(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용개월수_D페이_R6M = SUM(1 IF M0X_이용건수_D페이_B0M > 0 ELSE 0)
    """
    dd = df[["M12_이용건수_D페이_B0M",
             "M11_이용건수_D페이_B0M",
             "M10_이용건수_D페이_B0M",
             "M09_이용건수_D페이_B0M",
             "M08_이용건수_D페이_B0M",
             "M07_이용건수_D페이_B0M"]]
    res = dd.apply(lambda x: sum(x > 0), axis=1)

    c = df["M12_이용개월수_D페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2634(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_간편결제_R6M = M12_이용금액_간편결제_R3M + M09_이용금액_간편결제_R3M
    """
    dd = df[["M12_이용금액_간편결제_R3M", "M09_이용금액_간편결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_간편결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_2635(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_당사페이_R6M = M12_이용금액_당사페이_R3M + M09_이용금액_당사페이_R3M
    """
    dd = df[["M12_이용금액_당사페이_R3M", "M09_이용금액_당사페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_당사페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2636(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_당사기타_R6M = M12_이용금액_당사기타_R3M + M09_이용금액_당사기타_R3M
    """
    dd = df[["M12_이용금액_당사기타_R3M", "M09_이용금액_당사기타_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_당사기타_R6M"]
    return c == res


@constraint_udf
def cfs_03_2637(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_A페이_R6M = M12_이용금액_A페이_R3M + M09_이용금액_A페이_R3M
    """
    dd = df[["M12_이용금액_A페이_R3M", "M09_이용금액_A페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_A페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2638(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_B페이_R6M = M12_이용금액_B페이_R3M + M09_이용금액_B페이_R3M
    """
    dd = df[["M12_이용금액_B페이_R3M", "M09_이용금액_B페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_B페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2639(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_C페이_R6M = M12_이용금액_C페이_R3M + M09_이용금액_C페이_R3M
    """
    dd = df[["M12_이용금액_C페이_R3M", "M09_이용금액_C페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_C페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2640(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_D페이_R6M = M12_이용금액_D페이_R3M + M09_이용금액_D페이_R3M
    """
    dd = df[["M12_이용금액_D페이_R3M", "M09_이용금액_D페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_D페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2641(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_간편결제_R6M = M12_이용건수_간편결제_R3M + M09_이용건수_간편결제_R3M
    """
    dd = df[["M12_이용건수_간편결제_R3M", "M09_이용건수_간편결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_간편결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_2642(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_당사페이_R6M = M12_이용건수_당사페이_R3M + M09_이용건수_당사페이_R3M
    """
    dd = df[["M12_이용건수_당사페이_R3M", "M09_이용건수_당사페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_당사페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2643(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_당사기타_R6M = M12_이용건수_당사기타_R3M + M09_이용건수_당사기타_R3M
    """
    dd = df[["M12_이용건수_당사기타_R3M", "M09_이용건수_당사기타_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_당사기타_R6M"]
    return c == res


@constraint_udf
def cfs_03_2644(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_A페이_R6M = M12_이용건수_A페이_R3M + M09_이용건수_A페이_R3M
    """
    dd = df[["M12_이용건수_A페이_R3M", "M09_이용건수_A페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_A페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2645(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_B페이_R6M = M12_이용건수_B페이_R3M + M09_이용건수_B페이_R3M
    """
    dd = df[["M12_이용건수_B페이_R3M", "M09_이용건수_B페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_B페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2646(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_C페이_R6M = M12_이용건수_C페이_R3M + M09_이용건수_C페이_R3M
    """
    dd = df[["M12_이용건수_C페이_R3M", "M09_이용건수_C페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_C페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2647(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_D페이_R6M = M12_이용건수_D페이_R3M + M09_이용건수_D페이_R3M
    """
    dd = df[["M12_이용건수_D페이_R3M", "M09_이용건수_D페이_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_D페이_R6M"]
    return c == res


@constraint_udf
def cfs_03_2648(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_간편결제_R3M = M12_이용금액_간편결제_B0M + M11_이용금액_간편결제_B0M + M10_이용금액_간편결제_B0M
    """
    dd = df[["M12_이용금액_간편결제_B0M", "M11_이용금액_간편결제_B0M", "M10_이용금액_간편결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_간편결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_2649(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_당사페이_R3M = M12_이용금액_당사페이_B0M + M11_이용금액_당사페이_B0M + M10_이용금액_당사페이_B0M
    """
    dd = df[["M12_이용금액_당사페이_B0M", "M11_이용금액_당사페이_B0M", "M10_이용금액_당사페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_당사페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2650(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_당사기타_R3M = M12_이용금액_당사기타_B0M + M11_이용금액_당사기타_B0M + M10_이용금액_당사기타_B0M
    """
    dd = df[["M12_이용금액_당사기타_B0M", "M11_이용금액_당사기타_B0M", "M10_이용금액_당사기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_당사기타_R3M"]
    return c == res


@constraint_udf
def cfs_03_2651(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_A페이_R3M = M12_이용금액_A페이_B0M + M11_이용금액_A페이_B0M + M10_이용금액_A페이_B0M
    """
    dd = df[["M12_이용금액_A페이_B0M", "M11_이용금액_A페이_B0M", "M10_이용금액_A페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_A페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2652(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_B페이_R3M = M12_이용금액_B페이_B0M + M11_이용금액_B페이_B0M + M10_이용금액_B페이_B0M
    """
    dd = df[["M12_이용금액_B페이_B0M", "M11_이용금액_B페이_B0M", "M10_이용금액_B페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_B페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2653(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_C페이_R3M = M12_이용금액_C페이_B0M + M11_이용금액_C페이_B0M + M10_이용금액_C페이_B0M
    """
    dd = df[["M12_이용금액_C페이_B0M", "M11_이용금액_C페이_B0M", "M10_이용금액_C페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_C페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2654(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_D페이_R3M = M12_이용금액_D페이_B0M + M11_이용금액_D페이_B0M + M10_이용금액_D페이_B0M
    """
    dd = df[["M12_이용금액_D페이_B0M", "M11_이용금액_D페이_B0M", "M10_이용금액_D페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_D페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2655(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_간편결제_R3M = M12_이용건수_간편결제_B0M + M11_이용건수_간편결제_B0M + M10_이용건수_간편결제_B0M
    """
    dd = df[["M12_이용건수_간편결제_B0M", "M11_이용건수_간편결제_B0M", "M10_이용건수_간편결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_간편결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_2656(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_당사페이_R3M = M12_이용건수_당사페이_B0M + M11_이용건수_당사페이_B0M + M10_이용건수_당사페이_B0M
    """
    dd = df[["M12_이용건수_당사페이_B0M", "M11_이용건수_당사페이_B0M", "M10_이용건수_당사페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_당사페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2657(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_당사기타_R3M = M12_이용건수_당사기타_B0M + M11_이용건수_당사기타_B0M + M10_이용건수_당사기타_B0M
    """
    dd = df[["M12_이용건수_당사기타_B0M", "M11_이용건수_당사기타_B0M", "M10_이용건수_당사기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_당사기타_R3M"]
    return c == res


@constraint_udf
def cfs_03_2658(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_A페이_R3M = M12_이용건수_A페이_B0M + M11_이용건수_A페이_B0M + M10_이용건수_A페이_B0M
    """
    dd = df[["M12_이용건수_A페이_B0M", "M11_이용건수_A페이_B0M", "M10_이용건수_A페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_A페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2659(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_B페이_R3M = M12_이용건수_B페이_B0M + M11_이용건수_B페이_B0M + M10_이용건수_B페이_B0M
    """
    dd = df[["M12_이용건수_B페이_B0M", "M11_이용건수_B페이_B0M", "M10_이용건수_B페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_B페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2660(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_C페이_R3M = M12_이용건수_C페이_B0M + M11_이용건수_C페이_B0M + M10_이용건수_C페이_B0M
    """
    dd = df[["M12_이용건수_C페이_B0M", "M11_이용건수_C페이_B0M", "M10_이용건수_C페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_C페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2661(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_D페이_R3M = M12_이용건수_D페이_B0M + M11_이용건수_D페이_B0M + M10_이용건수_D페이_B0M
    """
    dd = df[["M12_이용건수_D페이_B0M", "M11_이용건수_D페이_B0M", "M10_이용건수_D페이_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_D페이_R3M"]
    return c == res


@constraint_udf
def cfs_03_2676(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       M12_이용개월수_선결제_R6M = SUM(1 IF M0X_이용건수_선결제_B0M > 0 ELSE 0)
    """
    dd = df[["M12_이용건수_선결제_B0M",
             "M11_이용건수_선결제_B0M",
             "M10_이용건수_선결제_B0M",
             "M09_이용건수_선결제_B0M",
             "M08_이용건수_선결제_B0M",
             "M07_이용건수_선결제_B0M"]]
    res = dd.apply(lambda x: sum(x > 0), axis=1)

    c = df["M12_이용개월수_선결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_2677(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용횟수_선결제_R6M = M12_이용횟수_선결제_R3M + M09_이용횟수_선결제_R3M
    """
    dd = df[["M12_이용횟수_선결제_R3M", "M09_이용횟수_선결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용횟수_선결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_2678(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_선결제_R6M = M12_이용금액_선결제_R3M + M09_이용금액_선결제_R3M
    """
    dd = df[["M12_이용금액_선결제_R3M", "M09_이용금액_선결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_선결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_2679(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_선결제_R6M = M12_이용건수_선결제_R3M + M09_이용건수_선결제_R3M
    """
    dd = df[["M12_이용건수_선결제_R3M", "M09_이용건수_선결제_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_선결제_R6M"]
    return c == res


@constraint_udf
def cfs_03_2680(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용횟수_선결제_R3M = M12_이용횟수_선결제_B0M + M11_이용횟수_선결제_B0M + M10_이용횟수_선결제_B0M
    """
    dd = df[["M12_이용횟수_선결제_B0M", "M11_이용횟수_선결제_B0M", "M10_이용횟수_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용횟수_선결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_2681(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_선결제_R3M = M12_이용금액_선결제_B0M + M11_이용금액_선결제_B0M + M10_이용금액_선결제_B0M
    """
    dd = df[["M12_이용금액_선결제_B0M", "M11_이용금액_선결제_B0M", "M10_이용금액_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_선결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_2682(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용건수_선결제_R3M = M12_이용건수_선결제_B0M + M11_이용건수_선결제_B0M + M10_이용건수_선결제_B0M
    """
    dd = df[["M12_이용건수_선결제_B0M", "M11_이용건수_선결제_B0M", "M10_이용건수_선결제_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용건수_선결제_R3M"]
    return c == res


@constraint_udf
def cfs_03_2687(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_가맹점매출금액_B2M = M11_가맹점매출금액_B1M
    """
    res = df["M11_가맹점매출금액_B1M"]

    c = df["M12_가맹점매출금액_B2M"]
    return c == res


@constraint_udf
def cfs_03_2689(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_정상청구원금_B2M = M10_정상청구원금_B0M
    """
    res = df["M10_정상청구원금_B0M"]

    c = df["M12_정상청구원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_2690(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_정상청구원금_B5M = M07_정상청구원금_B0M
    """
    res = df["M07_정상청구원금_B0M"]

    c = df["M12_정상청구원금_B5M"]
    return c == res


@constraint_udf
def cfs_03_2692(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_선입금원금_B2M = M10_선입금원금_B0M
    """
    res = df["M10_선입금원금_B0M"]

    c = df["M12_선입금원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_2693(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_선입금원금_B5M = M07_선입금원금_B0M
    """
    res = df["M07_선입금원금_B0M"]

    c = df["M12_선입금원금_B5M"]
    return c == res


@constraint_udf
def cfs_03_2695(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_정상입금원금_B2M = M10_정상입금원금_B0M
    """
    res = df["M10_정상입금원금_B0M"]

    c = df["M12_정상입금원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_2696(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_정상입금원금_B5M = M07_정상입금원금_B0M
    """
    res = df["M07_정상입금원금_B0M"]

    c = df["M12_정상입금원금_B5M"]
    return c == res


@constraint_udf
def cfs_03_2698(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_연체입금원금_B2M = M10_연체입금원금_B0M
    """
    res = df["M10_연체입금원금_B0M"]

    c = df["M12_연체입금원금_B2M"]
    return c == res


@constraint_udf
def cfs_03_2699(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_연체입금원금_B5M = M07_연체입금원금_B0M
    """
    res = df["M07_연체입금원금_B0M"]

    c = df["M12_연체입금원금_B5M"]
    return c == res


@constraint_udf
def cfs_03_2700(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       M12_이용개월수_전체_R6M = M12_이용개월수_전체_R3M + M09_이용개월수_전체_R3M
    """
    dd = df[["M12_이용개월수_전체_R3M", "M09_이용개월수_전체_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용개월수_전체_R6M"]
    return c == res


@constraint_udf
def cfs_03_2704(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용횟수_연체_R6M = M12_이용횟수_연체_R3M + M09_이용횟수_연체_R3M
    """
    dd = df[["M12_이용횟수_연체_R3M", "M09_이용횟수_연체_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용횟수_연체_R6M"]
    return c == res


@constraint_udf
def cfs_03_2705(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_연체_R6M = M12_이용금액_연체_R3M + M09_이용금액_연체_R3M
    """
    dd = df[["M12_이용금액_연체_R3M", "M09_이용금액_연체_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_연체_R6M"]
    return c == res


@constraint_udf
def cfs_03_2706(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용횟수_연체_R3M = M12_이용횟수_연체_B0M + M11_이용횟수_연체_B0M + M10_이용횟수_연체_B0M
    """
    dd = df[["M12_이용횟수_연체_B0M", "M11_이용횟수_연체_B0M", "M10_이용횟수_연체_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용횟수_연체_R3M"]
    return c == res


@constraint_udf
def cfs_03_2707(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_이용금액_연체_R3M = M12_이용금액_연체_B0M + M11_이용금액_연체_B0M + M10_이용금액_연체_B0M
    """
    dd = df[["M12_이용금액_연체_B0M", "M11_이용금액_연체_B0M", "M10_이용금액_연체_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["M12_이용금액_연체_R3M"]
    return c == res


@constraint_udf
def cfs_03_2720(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_증감_RP건수_전월 = M11_RP건수_B0M - M12_RP건수_B0M
    """
    dd = df[["M11_RP건수_B0M", "M12_RP건수_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M12_증감_RP건수_전월"]
    return c == res


@constraint_udf
def cfs_03_2722(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_증감_RP건수_통신_전월 = M11_RP건수_통신_B0M - M12_RP건수_통신_B0M
    """
    dd = df[["M11_RP건수_통신_B0M", "M12_RP건수_통신_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M12_증감_RP건수_통신_전월"]
    return c == res


@constraint_udf
def cfs_03_2723(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_증감_RP건수_아파트_전월 = M11_RP건수_아파트_B0M - M12_RP건수_아파트_B0M
    """
    dd = df[["M11_RP건수_아파트_B0M", "M12_RP건수_아파트_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M12_증감_RP건수_아파트_전월"]
    return c == res


@constraint_udf
def cfs_03_2724(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_증감_RP건수_제휴사서비스직접판매_전월 = M11_RP건수_제휴사서비스직접판매_B0M - M12_RP건수_제휴사서비스직접판매_B0M
    """
    dd = df[["M11_RP건수_제휴사서비스직접판매_B0M", "M12_RP건수_제휴사서비스직접판매_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M12_증감_RP건수_제휴사서비스직접판매_전월"]
    return c == res


@constraint_udf
def cfs_03_2725(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_증감_RP건수_렌탈_전월 = M11_RP건수_렌탈_B0M - M12_RP건수_렌탈_B0M
    """
    dd = df[["M11_RP건수_렌탈_B0M", "M12_RP건수_렌탈_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M12_증감_RP건수_렌탈_전월"]
    return c == res


@constraint_udf
def cfs_03_2726(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_증감_RP건수_가스_전월 = M11_RP건수_가스_B0M - M12_RP건수_가스_B0M
    """
    dd = df[["M11_RP건수_가스_B0M", "M12_RP건수_가스_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M12_증감_RP건수_가스_전월"]
    return c == res


@constraint_udf
def cfs_03_2727(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_증감_RP건수_전기_전월 = M11_RP건수_전기_B0M - M12_RP건수_전기_B0M
    """
    dd = df[["M11_RP건수_전기_B0M", "M12_RP건수_전기_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M12_증감_RP건수_전기_전월"]
    return c == res


@constraint_udf
def cfs_03_2728(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_증감_RP건수_보험_전월 = M11_RP건수_보험_B0M - M12_RP건수_보험_B0M
    """
    dd = df[["M11_RP건수_보험_B0M", "M12_RP건수_보험_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M12_증감_RP건수_보험_전월"]
    return c == res


@constraint_udf
def cfs_03_2729(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_증감_RP건수_학습비_전월 = M11_RP건수_학습비_B0M - M12_RP건수_학습비_B0M
    """
    dd = df[["M11_RP건수_학습비_B0M", "M12_RP건수_학습비_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M12_증감_RP건수_학습비_전월"]
    return c == res


@constraint_udf
def cfs_03_2730(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_증감_RP건수_유선방송_전월 = M11_RP건수_유선방송_B0M - M12_RP건수_유선방송_B0M
    """
    dd = df[["M11_RP건수_유선방송_B0M", "M12_RP건수_유선방송_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M12_증감_RP건수_유선방송_전월"]
    return c == res


@constraint_udf
def cfs_03_2731(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_증감_RP건수_건강_전월 = M11_RP건수_건강_B0M - M12_RP건수_건강_B0M
    """
    dd = df[["M11_RP건수_건강_B0M", "M12_RP건수_건강_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M12_증감_RP건수_건강_전월"]
    return c == res


@constraint_udf
def cfs_03_2732(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        M12_증감_RP건수_교통_전월 = M11_RP건수_교통_B0M - M12_RP건수_교통_B0M
    """
    dd = df[["M11_RP건수_교통_B0M", "M12_RP건수_교통_B0M"]]
    res = dd.apply(lambda x: x[0] - x[1], axis=1)

    c = df["M12_증감_RP건수_교통_전월"]
    return c == res
