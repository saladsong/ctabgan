# 컬럼 간 제약조건 또는 파생수식 만족 여부 검증용
# (v.1) 1개월치 데이터 내에서의 파생 관계만 고려되어 있음
# (v.1.5) validity 검증 후 수식>조건 수정 및 중복 컬럼 반영
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
    return val != val


constraints = [
    # 1.회원 테이블 컬럼 Constraints
    {
        "columns": ["회원여부_이용가능", "회원여부_이용가능_CA", "회원여부_이용가능_카드론"],
        "fname": "cc_01_0001",
        "type": "constraint",
        "content": "IF (회원여부_이용가능_CA == 1) OR (회원여부_이용가능_카드론 == 1): 회원여부_이용가능 == 1",
    },
    {
        "columns": ["대표BL코드_ACCOUNT", "BL여부"],
        "fname": "cc_01_0002",
        "type": "constraint",
        "content": "IF 대표BL코드_ACCOUNT IS NOT NULL: BL여부 == 1",
    },
    {
        "columns": ["탈회횟수_누적", "최종탈회후경과월"],
        "fname": "cc_01_0003",
        "type": "constraint",
        "content": "IF 탈회횟수_누적 > 0: 최종탈회후경과월 IS NOT NULL",
    },
    {
        "columns": ["탈회횟수_발급6개월이내", "탈회횟수_발급1년이내", "탈회횟수_누적"],
        "fname": "cc_01_0004",
        "type": "constraint",
        "content": "탈회횟수_발급6개월이내 <= 탈회횟수_발급1년이내 <= 탈회횟수_누적",
    },
    {
        "columns": ["소지여부_신용", "유효카드수_신용"],
        "fname": "cc_01_0005",
        "type": "constraint",
        "content": "IF 소지여부_신용 == 1: 유효카드수_신용 > 0",
    },
    {
        "columns": ["이용카드수_신용_가족", "이용가능카드수_신용_가족", "유효카드수_신용_가족", "유효카드수_신용"],
        "fname": "cc_01_0006",
        "type": "constraint",
        "content": "이용카드수_신용_가족 <= 이용가능카드수_신용_가족 <= 유효카드수_신용_가족 <= 유효카드수_신용",
    },
    {
        "columns": ["이용카드수_체크_가족", "이용가능카드수_체크_가족", "유효카드수_체크_가족", "유효카드수_체크"],
        "fname": "cc_01_0007",
        "type": "constraint",
        "content": "이용카드수_체크_가족 <= 이용가능카드수_체크_가족 <= 유효카드수_체크_가족 <= 유효카드수_체크",
    },
    {
        "columns": ["이용카드수_신용체크", "이용가능카드수_신용체크", "유효카드수_신용체크"],
        "fname": "cc_01_0008",
        "type": "constraint",
        "content": "이용카드수_신용체크 <= 이용가능카드수_신용체크 <= 유효카드수_신용체크",
    },
    {
        "columns": ["이용카드수_신용", "이용가능카드수_신용", "유효카드수_신용"],
        "fname": "cc_01_0009",
        "type": "constraint",
        "content": "이용카드수_신용 <= 이용가능카드수_신용 <= 유효카드수_신용",
    },
    {
        "columns": ["이용카드수_체크", "이용가능카드수_체크", "유효카드수_체크"],
        "fname": "cc_01_0010",
        "type": "constraint",
        "content": "이용카드수_체크 <= 이용가능카드수_체크 <= 유효카드수_체크",
    },
    {
        "columns": ["이용금액_R3M_신용체크", "_1순위카드이용금액", "_2순위카드이용금액"],
        "fname": "cc_01_0011",
        "type": "constraint",
        "content": "이용금액_R3M_신용체크 >= (_1순위카드이용금액 + _2순위카드이용금액)",
    },
    {
        "columns": ["이용금액_R3M_신용_가족", "이용금액_R3M_신용"],
        "fname": "cc_01_0012",
        "type": "constraint",
        "content": "이용금액_R3M_신용_가족 <= 이용금액_R3M_신용",
    },
    {
        "columns": ["이용금액_R3M_체크_가족", "이용금액_R3M_체크"],
        "fname": "cc_01_0013",
        "type": "constraint",
        "content": "이용금액_R3M_체크_가족 <= 이용금액_R3M_체크",
    },
    {
        "columns": ["_2순위카드이용금액", "_1순위카드이용금액", "이용금액_R3M_신용체크"],
        "fname": "cc_01_0014",
        "type": "constraint",
        "content": "_2순위카드이용금액 <= _1순위카드이용금액 <= 이용금액_R3M_신용체크",
    },
    {
        "columns": ["_1순위카드이용금액", "_1순위카드이용건수"],
        "fname": "cc_01_0015",
        "type": "constraint",
        "content": "IF _1순위카드이용금액 > 0: _1순위카드이용건수 > 0",
    },
    {
        "columns": ["_2순위카드이용금액", "_2순위카드이용건수"],
        "fname": "cc_01_0016",
        "type": "constraint",
        "content": "IF _2순위카드이용금액 > 0: _2순위카드이용건수 > 0",
    },
    {
        "columns": ["최종카드발급일자", "기준년월"],
        "fname": "cc_01_0017",
        "type": "constraint",
        "content": "최종카드발급일자 <= LAST_DAY(기준년월)",
    },
    {
        "columns": ["유효카드수_신용체크", "보유여부_해외겸용_본인"],
        "fname": "cc_01_0018",
        "type": "constraint",
        "content": "IF 유효카드수_신용체크 == 0: 보유여부_해외겸용_본인 == '0'",
    },
    {
        "columns": ["보유여부_해외겸용_본인", "이용가능여부_해외겸용_본인"],
        "fname": "cc_01_0019",
        "type": "constraint",
        "content": "IF 보유여부_해외겸용_본인 == '0': 이용가능여부_해외겸용_본인 == '0'",
    },
    {
        "columns": ["보유여부_해외겸용_본인", "보유여부_해외겸용_신용_본인"],
        "fname": "cc_01_0020",
        "type": "constraint",
        "content": "IF 보유여부_해외겸용_본인 == '0': 보유여부_해외겸용_신용_본인 == '0'",
    },
    {
        "columns": ["보유여부_해외겸용_신용_본인", "이용가능여부_해외겸용_신용_본인"],
        "fname": "cc_01_0021",
        "type": "constraint",
        "content": "IF 보유여부_해외겸용_신용_본인 == '0': 이용가능여부_해외겸용_신용_본인 == '0'",
    },
    {
        "columns": ["연회비발생카드수_B0M", "유효카드수_신용"],
        "fname": "cc_01_0022",
        "type": "constraint",
        "content": "연회비발생카드수_B0M <= 유효카드수_신용",
    },
    {
        "columns": ["연회비할인카드수_B0M", "유효카드수_신용"],
        "fname": "cc_01_0023",
        "type": "constraint",
        "content": "연회비할인카드수_B0M <= 유효카드수_신용",
    },
    {
        "columns": ["상품관련면제카드수_B0M", "연회비할인카드수_B0M"],
        "fname": "cc_01_0024",
        "type": "constraint",
        "content": "상품관련면제카드수_B0M <= 연회비할인카드수_B0M",
    },
    {
        "columns": ["임직원면제카드수_B0M", "연회비할인카드수_B0M"],
        "fname": "cc_01_0025",
        "type": "constraint",
        "content": "임직원면제카드수_B0M <= 연회비할인카드수_B0M",
    },
    {
        "columns": ["우수회원면제카드수_B0M", "연회비할인카드수_B0M"],
        "fname": "cc_01_0026",
        "type": "constraint",
        "content": "우수회원면제카드수_B0M <= 연회비할인카드수_B0M",
    },
    {
        "columns": ["기타면제카드수_B0M", "연회비할인카드수_B0M"],
        "fname": "cc_01_0027",
        "type": "constraint",
        "content": "기타면제카드수_B0M <= 연회비할인카드수_B0M",
    },
    {
        "columns": ["소지카드수_유효_신용", "유효카드수_신용"],
        "fname": "cc_01_0028",
        "type": "constraint",
        "content": "소지카드수_유효_신용 <= 유효카드수_신용",
    },
    {
        "columns": ["소지카드수_이용가능_신용", "이용가능카드수_신용"],
        "fname": "cc_01_0029",
        "type": "constraint",
        "content": "소지카드수_이용가능_신용 <= 이용가능카드수_신용",
    },
    {
        "columns": ["소지여부_신용", "소지카드수_유효_신용"],
        "fname": "cc_01_0030",
        "type": "constraint",
        "content": "IF 소지여부_신용=='1' THEN 소지카드수_유효_신용>0 ELSE 소지카드수_유효_신용==0",
    },
    {
        "columns": ["소지카드수_이용가능_신용", "소지카드수_유효_신용"],
        "fname": "cc_01_0031",
        "type": "constraint",
        "content": "소지카드수_이용가능_신용 <= 소지카드수_유효_신용",
    },

    # 1.회원 테이블 컬럼 Formula
    {
        "columns": ["기준년월", "입회일자_신용"],
        "output": "입회경과개월수_신용",
        "fname": "cf_01_0018",
        "type": "formula",
        "content": "입회경과개월수_신용 = MONTHS_BETWEEN(LAST_DAY(기준년월), 입회일자_신용)",
    },
    {
        "columns": ["이용횟수_연체_B0M"],
        "output": "회원여부_연체",
        "fname": "cf_01_0023",
        "type": "formula",
        "content": "회원여부_연체 = IF `이용횟수_연체_B0M` > 0 THEN '1' ELSE '0'",
    },
    {
        "columns": ["유효카드수_신용", "유효카드수_체크"],
        "output": "유효카드수_신용체크",
        "fname": "cf_01_0039",
        "type": "formula",
        "content": "유효카드수_신용체크 = 유효카드수_신용 + 유효카드수_체크",
    },
    {
        "columns": ["이용가능카드수_신용", "이용가능카드수_체크"],
        "output": "이용가능카드수_신용체크",
        "fname": "cf_01_0044",
        "type": "formula",
        "content": "이용가능카드수_신용체크 = 이용가능카드수_신용 + 이용가능카드수_체크",
    },
    {
        "columns": ["이용카드수_신용", "이용카드수_체크"],
        "output": "이용카드수_신용체크",
        "fname": "cf_01_0049",
        "type": "formula",
        "content": "이용카드수_신용체크 = 이용카드수_신용 + 이용카드수_체크",
    },
    {
        "columns": ["이용금액_R3M_신용", "이용금액_R3M_체크"],
        "output": "이용금액_R3M_신용체크",
        "fname": "cf_01_0054",
        "type": "formula",
        "content": "이용금액_R3M_신용체크 = 이용금액_R3M_신용 + 이용금액_R3M_체크",
    },
    {
        "columns": ["할인금액_기본연회비_B0M", "청구금액_기본연회비_B0M"],
        "output": "기본연회비_B0M",
        "fname": "cf_01_0083",
        "type": "formula",
        "content": "기본연회비_B0M = 할인금액_기본연회비_B0M+청구금액_기본연회비_B0M",
    },
    {
        "columns": ["할인금액_제휴연회비_B0M", "청구금액_제휴연회비_B0M"],
        "output": "제휴연회비_B0M",
        "fname": "cf_01_0084",
        "type": "formula",
        "content": "제휴연회비_B0M = 할인금액_제휴연회비_B0M+청구금액_제휴연회비_B0M",
    },
    {
        "columns": ["기준년월"],
        "output": "기타면제카드수_B0M",
        "fname": "cf_01_0092",
        "type": "formula",
        "content": "기타면제카드수_B0M = 0",
    },
    {
        "columns": ["기준년월", "최종카드발급일자"],
        "output": "최종카드발급경과월",
        "fname": "cf_01_0133",
        "type": "formula",
        "content": "최종카드발급경과월 = MONTHS_BETWEEN(LAST_DAY(기준년월), 최종카드발급일자)",
    },

    # 2.신용 테이블 컬럼 Constraints
    {
        "columns": ["CA한도금액", "카드이용한도금액"],
        "fname": "cc_02_0001",
        "type": "constraint",
        "content": "CA한도금액 <= 카드이용한도금액*0.4",
    },
    {
        "columns": ["일시상환론한도금액"],
        "fname": "cc_02_0002",
        "type": "constraint",
        "content": "일시상환론한도금액 <= 5000만원",
    },
    {
        "columns": ["월상환론한도금액"],
        "fname": "cc_02_0003",
        "type": "constraint",
        "content": "월상환론한도금액 <= 5000만원",
    },
    {
        "columns": ["CA이자율_할인전"],
        "fname": "cc_02_0004",
        "type": "constraint",
        "content": "CA이자율_할인전 <= 20%",
    },
    {
        "columns": ["CL이자율_할인전"],
        "fname": "cc_02_0005",
        "type": "constraint",
        "content": "CL이자율_할인전 <= 20%",
    },
    {
        "columns": ["RV일시불이자율_할인전"],
        "fname": "cc_02_0006",
        "type": "constraint",
        "content": "RV일시불이자율_할인전 <= 20%",
    },
    {
        "columns": ["RV현금서비스이자율_할인전"],
        "fname": "cc_02_0007",
        "type": "constraint",
        "content": "RV현금서비스이자율_할인전 <= 20%",
    },
    {
        "columns": ["RV신청일자", "RV약정청구율", "RV최소결제비율"],
        "fname": "cc_02_0008",
        "type": "constraint",
        "content": "IF RV신청일자 IS NULL: RV약정청구율 == 0",
    },
    {
        "columns": ["자발한도감액횟수_R12M"],
        "fname": "cc_02_0009",
        "type": "constraint",
        "content": "0 <= 자발한도감액횟수_R12M <= 12",
    },
    {
        "columns": ["자발한도감액횟수_R12M", "자발한도감액금액_R12M", "자발한도감액후경과월"],
        "fname": "cc_02_0010",
        "type": "constraint",
        "content": "IF 자발한도감액횟수_R12M >0: (자발한도감액금액_R12M >0) & (자발한도감액후경과월 <12)",
    },
    {
        "columns": ["연체감액여부_R3M", "강제한도감액횟수_R12M"],
        "fname": "cc_02_0011",
        "type": "constraint",
        "content": "IF 연체감액여부_R3M == '1': 강제한도감액횟수_R12M >0",
    },
    {
        "columns": ["강제한도감액횟수_R12M"],
        "fname": "cc_02_0012",
        "type": "constraint",
        "content": "0 <= 강제한도감액횟수_R12M <= 12",
    },
    {
        "columns": ["강제한도감액횟수_R12M", "강제한도감액금액_R12M", "강제한도감액후경과월"],
        "fname": "cc_02_0013",
        "type": "constraint",
        "content": "IF 강제한도감액횟수_R12M >0: (강제한도감액금액_R12M >0) & (강제한도감액후경과월 <12)",
    },
    {
        "columns": ["한도증액횟수_R12M"],
        "fname": "cc_02_0014",
        "type": "constraint",
        "content": "0 <= 한도증액횟수_R12M <= 12",
    },
    {
        "columns": ["한도증액횟수_R12M", "한도증액금액_R12M", "한도증액후경과월"],
        "fname": "cc_02_0015",
        "type": "constraint",
        "content": "IF 한도증액횟수_R12M >0: (한도증액금액_R12M >0) & (한도증액후경과월 <12)",
    },
    {
        "columns": ["상향가능CA한도금액", "카드이용한도금액", "상향가능한도금액", "CA한도금액"],
        "fname": "cc_02_0016",
        "type": "constraint",
        "content": "상향가능CA한도금액 <= (카드이용한도금액+상향가능한도금액)*0.4 - CA한도금액",
    },
    {
        "columns": ["월상환론상향가능한도금액", "월상환론한도금액"],
        "fname": "cc_02_0017",
        "type": "constraint",
        "content": "월상환론상향가능한도금액 <= 5000만원-월상환론한도금액",
    },
    {
        "columns": ["한도심사요청건수", "한도심사요청후경과월"],
        "fname": "cc_02_0018",
        "type": "constraint",
        "content": "IF 한도심사요청건수=0: 한도심사요청후경과월 = 3",
    },
    {
        "columns": ["한도요청거절건수", "한도심사거절후경과월"],
        "fname": "cc_02_0019",
        "type": "constraint",
        "content": "IF 한도요청거절건수=0: 한도심사거절후경과월 = 3",
    },
    {
        "columns": ["시장단기연체여부_R3M", "시장단기연체여부_R6M"],
        "fname": "cc_02_0020",
        "type": "constraint",
        "content": "IF 시장단기연체여부_R3M == '1' THEN 시장단기연체여부_R6M = '1'",
    },
    {
        "columns": ["시장연체상환여부_R3M", "시장연체상환여부_R6M"],
        "fname": "cc_02_0021",
        "type": "constraint",
        "content": "IF 시장연체상환여부_R3M == '1' THEN 시장연체상환여부_R6M = '1'",
    },
    {
        "columns": ["한도요청거절건수", "한도심사요청건수"],
        "fname": "cc_02_0022",
        "type": "constraint",
        "content": "한도요청거절건수 <= 한도심사요청건수",
    },


    # 2.신용 테이블 컬럼 Formula
    {
        "columns": ["이용거절여부_카드론"],
        "output": "카드론동의여부",
        "fname": "cf_02_0030",
        "type": "formula",
        "content": "IF 이용거절여부_카드론=='1' THEN 카드론동의여부='N' ELSE 카드론동의여부='Y'",
    },
    # {
    #     "columns": ["RV신청일자"],
    #     "output": "rv최초시작일자",
    #     "fname": "cf_02_0038",
    #     "type": "formula",
    #     "content": "IF RV신청일자 IS NOT NULL THEN rv최초시작일자=RV신청일자 ELSE rv최초시작일자 IS NULL",
    # },
    # {
    #     "columns": ["RV신청일자"],
    #     "output": "rv등록일자",
    #     "fname": "cf_02_0039",
    #     "type": "formula",
    #     "content": "IF RV신청일자 IS NOT NULL THEN rv등록일자=RV신청일자 ELSE rv등록일자 IS NULL",
    # },
    {
        "columns": ["한도요청거절건수", "한도요청승인건수"],
        "output": "한도심사요청건수",
        "fname": "cf_02_0040",
        "type": "formula",
        "content": "한도심사요청건수 = 한도요청거절건수 + 한도요청승인건수",
    },
    {
        "columns": ["기준년월", "rv최초시작일자"],
        "output": "rv최초시작후경과일",
        "fname": "cf_02_0060",
        "type": "formula",
        "content": "rv최초시작후경과일 = DATEDIFF(LAST_DAY(기준년월), rv최초시작일자)",
    },

    # 4.청구 테이블 컬럼 Constraints
    {
        "columns": ["청구서발송여부_B0", "청구서발송여부_R3M"],
        "fname": "cc_04_0001",
        "type": "constraint",
        "content": "IF 청구서발송여부_B0 =='1': 청구서발송여부_R3M ='1'",
    },
    {
        "columns": ["청구서발송여부_R3M", "청구서발송여부_R6M"],
        "fname": "cc_04_0002",
        "type": "constraint",
        "content": "IF 청구서발송여부_R3M =='1': 청구서발송여부_R6M =='1'",
    },
    {
        "columns": ["청구금액_B0", "청구금액_R3M", "청구금액_R6M"],
        "fname": "cc_04_0003",
        "type": "constraint",
        "content": "청구금액_B0 <= 청구금액_R3M <= 청구금액_R6M",
    },
    {
        "columns": ["포인트_마일리지_건별_B0M", "포인트_마일리지_건별_R3M"],
        "fname": "cc_04_0004",
        "type": "constraint",
        "content": "포인트_마일리지_건별_B0M <= 포인트_마일리지_건별_R3M",
    },
    {
        "columns": ["포인트_포인트_건별_B0M", "포인트_포인트_건별_R3M"],
        "fname": "cc_04_0005",
        "type": "constraint",
        "content": "포인트_포인트_건별_B0M <= 포인트_포인트_건별_R3M",
    },
    {
        "columns": ["포인트_마일리지_월적립_B0M", "포인트_마일리지_월적립_R3M"],
        "fname": "cc_04_0006",
        "type": "constraint",
        "content": "포인트_마일리지_월적립_B0M <= 포인트_마일리지_월적립_R3M",
    },
    {
        "columns": ["포인트_포인트_월적립_B0M", "포인트_포인트_월적립_R3M"],
        "fname": "cc_04_0007",
        "type": "constraint",
        "content": "포인트_포인트_월적립_B0M <= 포인트_포인트_월적립_R3M",
    },
    {
        "columns": ["포인트_적립포인트_R3M", "포인트_적립포인트_R12M"],
        "fname": "cc_04_0008",
        "type": "constraint",
        "content": "포인트_적립포인트_R3M <= 포인트_적립포인트_R12M",
    },
    {
        "columns": ["포인트_이용포인트_R3M", "포인트_이용포인트_R12M"],
        "fname": "cc_04_0009",
        "type": "constraint",
        "content": "포인트_이용포인트_R3M <= 포인트_이용포인트_R12M",
    },
    {
        "columns": ["마일_적립포인트_R3M", "마일_적립포인트_R12M"],
        "fname": "cc_04_0010",
        "type": "constraint",
        "content": "마일_적립포인트_R3M <= 마일_적립포인트_R12M",
    },
    {
        "columns": ["마일_이용포인트_R3M", "마일_이용포인트_R12M"],
        "fname": "cc_04_0011",
        "type": "constraint",
        "content": "마일_이용포인트_R3M <= 마일_이용포인트_R12M",
    },
    {
        "columns": ["할인건수_B0M", "할인건수_R3M"],
        "fname": "cc_04_0012",
        "type": "constraint",
        "content": "할인건수_B0M <= 할인건수_R3M",
    },
    {
        "columns": ["할인금액_R3M", "할인건수_R3M"],
        "fname": "cc_04_0013",
        "type": "constraint",
        "content": "IF 할인금액_R3M >0: 할인건수_R3M >0",
    },
    {
        "columns": ["할인금액_B0M", "할인금액_R3M"],
        "fname": "cc_04_0014",
        "type": "constraint",
        "content": "할인금액_B0M <= 할인금액_R3M",
    },
    {
        "columns": ["할인금액_B0M", "할인건수_B0M"],
        "fname": "cc_04_0015",
        "type": "constraint",
        "content": "IF 할인금액_B0M >0: 할인건수_B0M >0",
    },
    {
        "columns": ["할인금액_B0M", "이용금액_신판_B0M"],
        "fname": "cc_04_0016",
        "type": "constraint",
        "content": "할인금액_B0M <= 이용금액_신판_B0M",
    },
    {
        "columns": ["할인금액_청구서_B0M", "할인금액_청구서_R3M"],
        "fname": "cc_04_0017",
        "type": "constraint",
        "content": "할인금액_청구서_B0M <= 할인금액_청구서_R3M",
    },
    {
        "columns": ["혜택수혜금액", "혜택수혜금액_R3M"],
        "fname": "cc_04_0018",
        "type": "constraint",
        "content": "혜택수혜금액 <= 혜택수혜금액_R3M",
    },
    {
        "columns": ["상환개월수_결제일_R3M", "상환개월수_결제일_R6M", "상환개월수_결제일_R12M"],
        "fname": "cc_04_0019",
        "type": "constraint",
        "content": "상환개월수_결제일_R3M <= 상환개월수_결제일_R6M <= 상환개월수_결제일_R12M",
    },
    {
        "columns": ["선결제건수_R3M", "선결제건수_R6M", "선결제건수_R12M"],
        "fname": "cc_04_0020",
        "type": "constraint",
        "content": "선결제건수_R3M <= 선결제건수_R6M <= 선결제건수_R12M",
    },
    {
        "columns": ["연체건수_R3M", "연체건수_R6M", "연체건수_R12M"],
        "fname": "cc_04_0021",
        "type": "constraint",
        "content": "연체건수_R3M <= 연체건수_R6M <= 연체건수_R12M",
    },

    # 4.청구 테이블 컬럼 Formula
    {
        "columns": ["기준년월"],
        "output": "대표결제방법코드",
        "fname": "cf_04_0008",
        "type": "formula",
        "content": "대표결제방법코드 = 2",
    },
    {
        "columns": ["대표청구서수령지구분코드"],
        "output": "청구서수령방법",
        "fname": "cf_04_0011",
        "type": "formula",
        "content": """IF 구분코드 IN ('1','3') THEN 청구서수령방법 = 01.우편, ELIF 구분코드 IN ('2') THEN 02.이메일,
                      ELIF 구분코드 IN ('L','S') THEN 03.LMS, ELIF 구분코드 IN ('K') THEN 04.카카오,
                      ELIF 구분코드 IN ('H') THEN 05.당사멤버십, ELIF 구분코드 IN ('T') THEN 07.기타,
                      ELIF 구분코드 IN ('0') THEN 99.미수령
                   """,
    },
    {
        "columns": ["청구금액_B0"],
        "output": "청구서발송여부_B0",
        "fname": "cf_04_0012",
        "type": "formula",
        "content": "IF 청구금액_B0 > 0 THEN 청구서발송여부_B0 = '1' ELSE '0'",
    },
    {
        "columns": ["포인트_포인트_건별_R3M", "포인트_포인트_월적립_R3M"],
        "output": "포인트_적립포인트_R3M",
        "fname": "cf_04_0027",
        "type": "formula",
        "content": "포인트_적립포인트_R3M = 포인트_포인트_건별_R3M + 포인트_포인트_월적립_R3M",
    },
    {
        "columns": ["포인트_마일리지_건별_R3M", "포인트_마일리지_월적립_R3M"],
        "output": "마일_적립포인트_R3M",
        "fname": "cf_04_0032",
        "type": "formula",
        "content": "마일_적립포인트_R3M = 포인트_마일리지_건별_R3M + 포인트_마일리지_월적립_R3M",
    },

    # 5.잔액 테이블 컬럼 Constraints
    {
        "columns": ["잔액_B0M", "카드이용한도금액"],
        "fname": "cc_05_0001",
        "type": "constraint",
        "content": "잔액_B0M <= 카드이용한도금액",
    },
    {
        "columns": ["잔액_일시불_B0M", "잔액_할부_B0M", "잔액_리볼빙일시불이월_B0M", "카드이용한도금액"],
        "fname": "cc_05_0002",
        "type": "constraint",
        "content": "잔액_일시불_B0M + 잔액_할부_B0M + 잔액_리볼빙일시불이월_B0M <= 카드이용한도금액",
    },
    {
        "columns": ["잔액_현금서비스_B0M", "잔액_리볼빙CA이월_B0M", "CA한도금액"],
        "fname": "cc_05_0003",
        "type": "constraint",
        "content": "잔액_현금서비스_B0M + 잔액_리볼빙CA이월_B0M <= CA한도금액",
    },
    {
        "columns": ["잔액_카드론_B0M", "월상환론한도금액"],
        "fname": "cc_05_0004",
        "type": "constraint",
        "content": "잔액_카드론_B0M <= 월상환론한도금액",
    },
    {
        "columns": ["잔액_카드론_B0M", "카드론잔액_최종경과월"],
        "fname": "cc_05_0005",
        "type": "constraint",
        "content": "IF 잔액_카드론_B0M >0: 카드론잔액_최종경과월 IS NOT NULL",
    },
    {
        "columns": ["연체일자_B0M", "연체잔액_B0M"],
        "fname": "cc_05_0006",
        "type": "constraint",
        "content": "IF 연체일자_B0M IS NOT NULL: 연체잔액_B0M >0",
    },
    {
        "columns": ["회원여부_이용가능_CA", "연체잔액_현금서비스_B0M"],
        "fname": "cc_05_0007",
        "type": "constraint",
        "content": "IF 회원여부_이용가능_CA == 0: 연체잔액_현금서비스_B0M = 0",
    },
    {
        "columns": ["회원여부_이용가능_카드론", "연체잔액_카드론_B0M"],
        "fname": "cc_05_0008",
        "type": "constraint",
        "content": "IF 회원여부_이용가능_카드론 == 0: 연체잔액_카드론_B0M = 0",
    },
    {
        "columns": ["RV_최대잔액_R12M", "RV_평균잔액_R12M"],
        "fname": "cc_05_0009",
        "type": "constraint",
        "content": "RV_최대잔액_R12M >= RV_평균잔액_R12M",
    },
    {
        "columns": ["RV_최대잔액_R6M", "RV_평균잔액_R6M"],
        "fname": "cc_05_0010",
        "type": "constraint",
        "content": "RV_최대잔액_R6M >= RV_평균잔액_R6M",
    },
    {
        "columns": ["RV_최대잔액_R3M", "RV_평균잔액_R3M"],
        "fname": "cc_05_0011",
        "type": "constraint",
        "content": "RV_최대잔액_R3M >= RV_평균잔액_R3M",
    },
    {
        "columns": ["RV잔액이월횟수_R3M", "RV잔액이월횟수_R6M", "RV잔액이월횟수_R12M"],
        "fname": "cc_05_0016",
        "type": "constraint",
        "content": "RV잔액이월횟수_R3M <= RV잔액이월횟수_R6M <= RV잔액이월횟수_R12M",
    },
    {
        "columns": ["잔액_할부_해외_B0M", "잔액_할부_B0M"],
        "fname": "cc_05_0017",
        "type": "constraint",
        "content": "잔액_할부_해외_B0M <= 잔액_할부_B0M",
    },
    {
        "columns": ["연체잔액_일시불_B0M", "연체잔액_일시불_해외_B0M"],
        "fname": "cc_05_0018",
        "type": "constraint",
        "content": "연체잔액_일시불_B0M <= 연체잔액_일시불_해외_B0M",
    },
    {
        "columns": ["연체잔액_일시불_해외_B0M", "연체잔액_RV일시불_해외_B0M"],
        "fname": "cc_05_0019",
        "type": "constraint",
        "content": "연체잔액_일시불_해외_B0M <= 연체잔액_RV일시불_해외_B0M",
    },
    {
        "columns": ["연체잔액_RV일시불_B0M", "연체잔액_일시불_B0M"],
        "fname": "cc_05_0020",
        "type": "constraint",
        "content": "연체잔액_RV일시불_B0M <= 연체잔액_일시불_B0M",
    },
    {
        "columns": ["연체잔액_RV일시불_해외_B0M", "연체잔액_RV일시불_B0M"],
        "fname": "cc_05_0021",
        "type": "constraint",
        "content": "연체잔액_RV일시불_해외_B0M <= 연체잔액_RV일시불_B0M",
    },
    {
        "columns": ["연체잔액_할부_해외_B0M", "연체잔액_할부_B0M"],
        "fname": "cc_05_0022",
        "type": "constraint",
        "content": "연체잔액_할부_해외_B0M <= 연체잔액_할부_B0M",
    },
    {
        "columns": ["연체잔액_CA_해외_B0M", "연체잔액_CA_B0M"],
        "fname": "cc_05_0023",
        "type": "constraint",
        "content": "연체잔액_CA_해외_B0M <= 연체잔액_CA_B0M",
    },
    {
        "columns": ["평잔_일시불_해외_3M", "평잔_일시불_3M"],
        "fname": "cc_05_0024",
        "type": "constraint",
        "content": "평잔_일시불_해외_3M <= 평잔_일시불_3M",
    },
    {
        "columns": ["평잔_RV일시불_해외_3M", "평잔_RV일시불_3M"],
        "fname": "cc_05_0025",
        "type": "constraint",
        "content": "평잔_RV일시불_해외_3M <= 평잔_RV일시불_3M",
    },
    {
        "columns": ["평잔_할부_해외_3M", "평잔_할부_3M"],
        "fname": "cc_05_0026",
        "type": "constraint",
        "content": "평잔_할부_해외_3M <= 평잔_할부_3M",
    },
    {
        "columns": ["평잔_CA_해외_3M", "평잔_CA_3M"],
        "fname": "cc_05_0027",
        "type": "constraint",
        "content": "평잔_CA_해외_3M <= 평잔_CA_3M",
    },
    {
        "columns": ["평잔_일시불_해외_6M", "평잔_일시불_6M"],
        "fname": "cc_05_0028",
        "type": "constraint",
        "content": "평잔_일시불_해외_6M <= 평잔_일시불_6M",
    },
    {
        "columns": ["평잔_RV일시불_해외_6M", "평잔_RV일시불_6M"],
        "fname": "cc_05_0029",
        "type": "constraint",
        "content": "평잔_RV일시불_해외_6M <= 평잔_RV일시불_6M",
    },
    {
        "columns": ["평잔_할부_해외_6M", "평잔_할부_6M"],
        "fname": "cc_05_0030",
        "type": "constraint",
        "content": "평잔_할부_해외_6M <= 평잔_할부_6M",
    },
    {
        "columns": ["평잔_CA_해외_6M", "평잔_CA_6M"],
        "fname": "cc_05_0031",
        "type": "constraint",
        "content": "평잔_CA_해외_6M <= 평잔_CA_6M",
    },

    # 5.잔액 테이블 컬럼 Formula
    {
        "columns": [
            "잔액_일시불_B0M",
            "잔액_할부_B0M",
            "잔액_현금서비스_B0M",
            "잔액_리볼빙일시불이월_B0M",
            "잔액_리볼빙CA이월_B0M",
            "잔액_카드론_B0M",
        ],
        "output": "잔액_B0M",
        "fname": "cf_05_0006",
        "type": "formula",
        "content": "잔액_B0M = SUM(잔액_일시불_B0M, 할부, 현금서비스, 리볼빙일시불이월, 리볼빙CA이월, 카드론)",
    },
    {
        "columns": [
            "잔액_할부_유이자_B0M",
            "잔액_할부_무이자_B0M",
        ],
        "output": "잔액_할부_B0M",
        "fname": "cf_05_0008",
        "type": "formula",
        "content": "잔액_할부_B0M = SUM(잔액_할부_유이자_B0M, 잔액_할부_무이자_B0M)",
    },
    {
        "columns": [
            "연체잔액_일시불_B0M",
            "연체잔액_할부_B0M",
            "연체잔액_현금서비스_B0M",
            "연체잔액_카드론_B0M",
            "연체잔액_대환론_B0M",
        ],
        "output": "연체잔액_B0M",
        "fname": "cf_05_0018",
        "type": "formula",
        "content": "연체잔액_B0M = SUM(연체잔액_일시불_B0M, 할부, 현금서비스, 카드론, 대환론)",
    },

    # 6.채널활동 테이블 컬럼 Constraints
    {
        "columns": ["인입횟수_ARS_B0M", "인입횟수_ARS_R6M"],
        "fname": "cc_06_0001",
        "type": "constraint",
        "content": "인입횟수_ARS_B0M <= 인입횟수_ARS_R6M",
    },
    {
        "columns": ["이용메뉴건수_ARS_B0M", "이용메뉴건수_ARS_R6M"],
        "fname": "cc_06_0002",
        "type": "constraint",
        "content": "이용메뉴건수_ARS_B0M <= 이용메뉴건수_ARS_R6M",
    },
    {
        "columns": ["이용메뉴건수_ARS_R6M", "인입횟수_ARS_R6M"],
        "fname": "cc_06_0064",
        "type": "constraint",
        "content": "이용메뉴건수_ARS_R6M >= 인입횟수_ARS_R6M",
    },
    {
        "columns": ["인입일수_ARS_B0M", "인입일수_ARS_R6M"],
        "fname": "cc_06_0003",
        "type": "constraint",
        "content": "인입일수_ARS_B0M <= 인입일수_ARS_R6M",
    },
    {
        "columns": ["인입월수_ARS_R6M", "인입일수_ARS_R6M", "인입횟수_ARS_R6M"],
        "fname": "cc_06_0004",
        "type": "constraint",
        "content": "인입월수_ARS_R6M <= 인입일수_ARS_R6M <= 인입횟수_ARS_R6M",
    },
    {
        "columns": ["인입횟수_ARS_R6M", "인입월수_ARS_R6M", "인입후경과월_ARS"],
        "fname": "cc_06_0005",
        "type": "constraint",
        "content": "IF 인입횟수_ARS_R6M >0: (0 < 인입월수_ARS_R6M <= 6) & (인입후경과월_< 6)",
    },
    {
        "columns": ["이용메뉴건수_ARS_B0M", "인입횟수_ARS_B0M"],
        "fname": "cc_06_0006",
        "type": "constraint",
        "content": "이용메뉴건수_ARS_B0M >= 인입횟수_ARS_B0M",
    },
    {
        "columns": ["인입일수_ARS_B0M", "인입횟수_ARS_B0M"],
        "fname": "cc_06_0007",
        "type": "constraint",
        "content": "인입일수_ARS_B0M <= 인입횟수_ARS_B0M",
    },
    {
        "columns": ["방문횟수_PC_B0M", "방문횟수_PC_R6M"],
        "fname": "cc_06_0008",
        "type": "constraint",
        "content": "방문횟수_PC_B0M <= 방문횟수_PC_R6M",
    },
    {
        "columns": ["방문일수_PC_B0M", "방문일수_PC_R6M"],
        "fname": "cc_06_0009",
        "type": "constraint",
        "content": "방문일수_PC_B0M <= 방문일수_PC_R6M",
    },
    {
        "columns": ["방문월수_PC_R6M", "방문일수_PC_R6M", "방문횟수_PC_R6M"],
        "fname": "cc_06_0010",
        "type": "constraint",
        "content": "방문월수_PC_R6M <= 방문일수_PC_R6M <= 방문횟수_PC_R6M",
    },
    {
        "columns": ["방문횟수_PC_R6M", "방문월수_PC_R6M", "방문후경과월_PC_R6M"],
        "fname": "cc_06_0011",
        "type": "constraint",
        "content": "IF 방문횟수_PC_R6M >0:  (0 < 방문월수_PC_R6M <= 6) & (방문후경과월_PC_R6M < 6)",
    },
    {
        "columns": ["방문횟수_앱_B0M", "방문횟수_앱_R6M"],
        "fname": "cc_06_0012",
        "type": "constraint",
        "content": "방문횟수_앱_B0M <= 방문횟수_앱_R6M",
    },
    {
        "columns": ["방문일수_앱_B0M", "방문일수_앱_R6M"],
        "fname": "cc_06_0013",
        "type": "constraint",
        "content": "방문일수_앱_B0M <= 방문일수_앱_R6M",
    },
    {
        "columns": ["방문월수_앱_R6M", "방문일수_앱_R6M", "방문횟수_앱_R6M"],
        "fname": "cc_06_0014",
        "type": "constraint",
        "content": "방문월수_앱_R6M <= 방문일수_앱_R6M <= 방문횟수_앱_R6M",
    },
    {
        "columns": ["방문횟수_앱_R6M", "방문월수_앱_R6M", "방문후경과월_앱_R6M"],
        "fname": "cc_06_0015",
        "type": "constraint",
        "content": "IF 방문횟수_앱_R6M >0:  (0 < 방문월수_앱_R6M <= 6) & (방문후경과월_앱_R6M < 6)",
    },
    {
        "columns": ["방문횟수_모바일웹_B0M", "방문횟수_모바일웹_R6M"],
        "fname": "cc_06_0016",
        "type": "constraint",
        "content": "방문횟수_모바일웹_B0M <= 방문횟수_모바일웹_R6M",
    },
    {
        "columns": ["방문일수_모바일웹_B0M", "방문일수_모바일웹_R6M"],
        "fname": "cc_06_0017",
        "type": "constraint",
        "content": "방문일수_모바일웹_B0M <= 방문일수_모바일웹_R6M",
    },
    {
        "columns": ["방문월수_모바일웹_R6M", "방문일수_모바일웹_R6M", "방문횟수_모바일웹_R6M"],
        "fname": "cc_06_0018",
        "type": "constraint",
        "content": "방문월수_모바일웹_R6M <= 방문일수_모바일웹_R6M <= 방문횟수_모바일웹_R6M",
    },
    {
        "columns": ["방문횟수_모바일웹_R6M", "방문월수_모바일웹_R6M", "방문후경과월_모바일웹_R6M"],
        "fname": "cc_06_0019",
        "type": "constraint",
        "content": "IF 방문횟수_모바일웹_R6M >0:  (0 < 방문월수_모바일웹_R6M <= 6) & (방문후경과월_모바일웹_R6M < 6)",
    },
    {
        "columns": ["방문일수_PC_B0M", "방문횟수_PC_B0M"],
        "fname": "cc_06_0020",
        "type": "constraint",
        "content": "방문일수_PC_B0M <= 방문횟수_PC_B0M",
    },
    {
        "columns": ["방문일수_앱_B0M", "방문횟수_앱_B0M"],
        "fname": "cc_06_0021",
        "type": "constraint",
        "content": "방문일수_앱_B0M <= 방문횟수_앱_B0M",
    },
    {
        "columns": ["방문일수_모바일웹_B0M", "방문횟수_모바일웹_B0M"],
        "fname": "cc_06_0022",
        "type": "constraint",
        "content": "방문일수_모바일웹_B0M <= 방문횟수_모바일웹_B0M",
    },
    {
        "columns": ["인입횟수_IB_B0M", "인입횟수_IB_R6M"],
        "fname": "cc_06_0023",
        "type": "constraint",
        "content": "인입횟수_IB_B0M <= 인입횟수_IB_R6M",
    },
    {
        "columns": ["인입일수_IB_B0M", "인입일수_IB_R6M"],
        "fname": "cc_06_0024",
        "type": "constraint",
        "content": "인입일수_IB_B0M <= 인입일수_IB_R6M",
    },
    {
        "columns": ["인입월수_IB_R6M", "인입일수_IB_R6M", "인입횟수_IB_R6M"],
        "fname": "cc_06_0025",
        "type": "constraint",
        "content": "인입월수_IB_R6M <= 인입일수_IB_R6M <= 인입횟수_IB_R6M",
    },
    {
        "columns": ["인입횟수_IB_R6M", "인입월수_IB_R6M", "인입후경과월_IB_R6M"],
        "fname": "cc_06_0026",
        "type": "constraint",
        "content": "IF 인입횟수_IB_R6M >0:  (0 < 인입월수_IB_R6M <= 6) & (인입후경과월_IB_R6M < 6)",
    },
    {
        "columns": ["이용메뉴건수_IB_B0M", "이용메뉴건수_IB_R6M", "인입횟수_IB_R6M"],
        "fname": "cc_06_0027",
        "type": "constraint",
        "content": "이용메뉴건수_IB_B0M <= 이용메뉴건수_IB_R6M <= 인입횟수_IB_R6M",
    },
    {
        "columns": ["인입일수_IB_B0M", "인입횟수_IB_B0M"],
        "fname": "cc_06_0028",
        "type": "constraint",
        "content": "인입일수_IB_B0M <= 인입횟수_IB_B0M",
    },
    {
        "columns": ["이용메뉴건수_IB_B0M", "인입횟수_IB_B0M"],
        "fname": "cc_06_0029",
        "type": "constraint",
        "content": "이용메뉴건수_IB_B0M <= 인입횟수_IB_B0M",
    },
    {
        "columns": ["인입불만횟수_IB_B0M", "인입불만횟수_IB_R6M"],
        "fname": "cc_06_0030",
        "type": "constraint",
        "content": "인입불만횟수_IB_B0M <= 인입불만횟수_IB_R6M",
    },
    {
        "columns": ["인입불만일수_IB_B0M", "인입불만일수_IB_R6M"],
        "fname": "cc_06_0031",
        "type": "constraint",
        "content": "인입불만일수_IB_B0M <= 인입불만일수_IB_R6M",
    },
    {
        "columns": ["인입불만월수_IB_R6M", "인입불만일수_IB_R6M", "인입불만횟수_IB_R6M"],
        "fname": "cc_06_0032",
        "type": "constraint",
        "content": "인입불만월수_IB_R6M <= 인입불만일수_IB_R6M <= 인입불만횟수_IB_R6M",
    },
    {
        "columns": ["인입불만횟수_IB_R6M", "인입불만월수_IB_R6M", "인입불만후경과월_IB_R6M"],
        "fname": "cc_06_0033",
        "type": "constraint",
        "content": "IF 인입불만횟수_IB_R6M >0: (0 < 인입불만월수_IB_R6M <= 6) & (인입불만후경과월_IB_R6M < 6)",
    },
    {
        "columns": ["인입불만일수_IB_B0M", "인입불만횟수_IB_B0M"],
        "fname": "cc_06_0034",
        "type": "constraint",
        "content": "인입불만일수_IB_B0M <= 인입불만횟수_IB_B0M",
    },
    {
        "columns": ["IB문의건수_사용승인내역_B0M", "IB문의건수_사용승인내역_R6M"],
        "fname": "cc_06_0035",
        "type": "constraint",
        "content": "IB문의건수_사용승인내역_B0M <= IB문의건수_사용승인내역_R6M",
    },
    {
        "columns": ["IB문의건수_한도_B0M", "IB문의건수_한도_R6M"],
        "fname": "cc_06_0036",
        "type": "constraint",
        "content": "IB문의건수_한도_B0M <= IB문의건수_한도_R6M",
    },
    {
        "columns": ["IB문의건수_선결제_B0M", "IB문의건수_선결제_R6M"],
        "fname": "cc_06_0037",
        "type": "constraint",
        "content": "IB문의건수_선결제_B0M <= IB문의건수_선결제_R6M",
    },
    {
        "columns": ["IB문의건수_결제_B0M", "IB문의건수_결제_R6M"],
        "fname": "cc_06_0038",
        "type": "constraint",
        "content": "IB문의건수_결제_B0M <= IB문의건수_결제_R6M",
    },
    {
        "columns": ["IB문의건수_할부_B0M", "IB문의건수_할부_R6M"],
        "fname": "cc_06_0039",
        "type": "constraint",
        "content": "IB문의건수_할부_B0M <= IB문의건수_할부_R6M",
    },
    {
        "columns": ["IB문의건수_정보변경_B0M", "IB문의건수_정보변경_R6M"],
        "fname": "cc_06_0040",
        "type": "constraint",
        "content": "IB문의건수_정보변경_B0M <= IB문의건수_정보변경_R6M",
    },
    {
        "columns": ["IB문의건수_결제일변경_B0M", "IB문의건수_결제일변경_R6M"],
        "fname": "cc_06_0041",
        "type": "constraint",
        "content": "IB문의건수_결제일변경_B0M <= IB문의건수_결제일변경_R6M",
    },
    {
        "columns": ["IB문의건수_명세서_B0M", "IB문의건수_명세서_R6M"],
        "fname": "cc_06_0042",
        "type": "constraint",
        "content": "IB문의건수_명세서_B0M <= IB문의건수_명세서_R6M",
    },
    {
        "columns": ["IB문의건수_비밀번호_B0M", "IB문의건수_비밀번호_R6M"],
        "fname": "cc_06_0043",
        "type": "constraint",
        "content": "IB문의건수_비밀번호_B0M <= IB문의건수_비밀번호_R6M",
    },
    {
        "columns": ["IB문의건수_SMS_B0M", "IB문의건수_SMS_R6M"],
        "fname": "cc_06_0044",
        "type": "constraint",
        "content": "IB문의건수_SMS_B0M <= IB문의건수_SMS_R6M",
    },
    {
        "columns": ["IB문의건수_APP_B0M", "IB문의건수_APP_R6M"],
        "fname": "cc_06_0045",
        "type": "constraint",
        "content": "IB문의건수_APP_B0M <= IB문의건수_APP_R6M",
    },
    {
        "columns": ["IB문의건수_부대서비스_B0M", "IB문의건수_부대서비스_R6M"],
        "fname": "cc_06_0046",
        "type": "constraint",
        "content": "IB문의건수_부대서비스_B0M <= IB문의건수_부대서비스_R6M",
    },
    {
        "columns": ["IB문의건수_포인트_B0M", "IB문의건수_포인트_R6M"],
        "fname": "cc_06_0047",
        "type": "constraint",
        "content": "IB문의건수_포인트_B0M <= IB문의건수_포인트_R6M",
    },
    {
        "columns": ["IB문의건수_카드발급_B0M", "IB문의건수_카드발급_R6M"],
        "fname": "cc_06_0048",
        "type": "constraint",
        "content": "IB문의건수_카드발급_B0M <= IB문의건수_카드발급_R6M",
    },
    {
        "columns": ["IB문의건수_BL_B0M", "IB문의건수_BL_R6M"],
        "fname": "cc_06_0049",
        "type": "constraint",
        "content": "IB문의건수_BL_B0M <= IB문의건수_BL_R6M",
    },
    {
        "columns": ["IB문의건수_분실도난_B0M", "IB문의건수_분실도난_R6M"],
        "fname": "cc_06_0050",
        "type": "constraint",
        "content": "IB문의건수_분실도난_B0M <= IB문의건수_분실도난_R6M",
    },
    {
        "columns": ["IB문의건수_CA_B0M", "IB문의건수_CA_R6M"],
        "fname": "cc_06_0051",
        "type": "constraint",
        "content": "IB문의건수_CA_B0M <= IB문의건수_CA_R6M",
    },
    {
        "columns": ["IB문의건수_CL_RV_B0M", "IB문의건수_CL_RV_R6M"],
        "fname": "cc_06_0052",
        "type": "constraint",
        "content": "IB문의건수_CL_RV_B0M <= IB문의건수_CL_RV_R6M",
    },
    {
        "columns": ["IB문의건수_CS_B0M", "IB문의건수_CS_R6M"],
        "fname": "cc_06_0053",
        "type": "constraint",
        "content": "IB문의건수_CS_B0M <= IB문의건수_CS_R6M",
    },
    {
        "columns": ["IB상담건수_VOC_B0M", "IB상담건수_VOC_R6M"],
        "fname": "cc_06_0054",
        "type": "constraint",
        "content": "IB상담건수_VOC_B0M <= IB상담건수_VOC_R6M",
    },
    {
        "columns": ["IB상담건수_VOC민원_B0M", "IB상담건수_VOC민원_R6M"],
        "fname": "cc_06_0055",
        "type": "constraint",
        "content": "IB상담건수_VOC민원_B0M <= IB상담건수_VOC민원_R6M",
    },
    {
        "columns": ["IB상담건수_VOC불만_B0M", "IB상담건수_VOC불만_R6M"],
        "fname": "cc_06_0056",
        "type": "constraint",
        "content": "IB상담건수_VOC불만_B0M <= IB상담건수_VOC불만_R6M",
    },
    {
        "columns": ["IB상담건수_금감원_B0M", "IB상담건수_금감원_R6M"],
        "fname": "cc_06_0057",
        "type": "constraint",
        "content": "IB상담건수_금감원_B0M <= IB상담건수_금감원_R6M",
    },
    {
        "columns": ["불만제기건수_B0M", "불만제기건수_R12M"],
        "fname": "cc_06_0058",
        "type": "constraint",
        "content": "불만제기건수_B0M <= 불만제기건수_R12M",
    },
    {
        "columns": ["불만제기건수_R12M", "불만제기후경과월_R12M"],
        "fname": "cc_06_0059",
        "type": "constraint",
        "content": "IF 불만제기건수_R12M >0: 0 <= 불만제기후경과월_R12M < 12",
    },
    {
        "columns": ["당사멤버쉽_방문횟수_B0M", "당사멤버쉽_방문횟수_R6M"],
        "fname": "cc_06_0060",
        "type": "constraint",
        "content": "당사멤버쉽_방문횟수_B0M <= 당사멤버쉽_방문횟수_R6M",
    },
    {
        "columns": ["당사멤버쉽_방문횟수_R6M", "당사멤버쉽_방문월수_R6M"],
        "fname": "cc_06_0061",
        "type": "constraint",
        "content": "IF 당사멤버쉽_방문횟수_R6M >0: 0 < 당사멤버쉽_방문월수_R6M <= 6",
    },
    {
        "columns": ["당사멤버쉽_방문월수_R6M", "당사멤버쉽_방문횟수_R6M"],
        "fname": "cc_06_0062",
        "type": "constraint",
        "content": "당사멤버쉽_방문월수_R6M <= 당사멤버쉽_방문횟수_R6M",
    },
    {
        "columns": ["상담건수_B0M", "IB상담건수_VOC_B0M", "IB상담건수_금감원_B0M"],
        "fname": "cc_06_0063",
        "type": "constraint",
        "content": "상담건수_B0M >= SUM(IB상담건수_VOC_B0M, IB상담건수_금감원_B0M)",
    },
    {
        "columns": ["홈페이지_금융건수_R3M", "홈페이지_금융건수_R6M"],
        "fname": "cc_06_0065",
        "type": "constraint",
        "content": "홈페이지_금융건수_R3M <= 홈페이지_금융건수_R6M",
    },
    {
        "columns": ["홈페이지_선결제건수_R3M", "홈페이지_선결제건수_R6M"],
        "fname": "cc_06_0066",
        "type": "constraint",
        "content": "홈페이지_선결제건수_R3M <= 홈페이지_선결제건수_R6M",
    },
    {
        "columns": ["상담건수_B0M", "상담건수_R6M"],
        "fname": "cc_06_0067",
        "type": "constraint",
        "content": "상담건수_B0M <= 상담건수_R6M",
    },

    # 6.채널활동 테이블 컬럼 Formula
    {
        "columns": ["IB상담건수_VOC민원_B0M", "IB상담건수_VOC불만_B0M"],
        "output": "IB상담건수_VOC_B0M",
        "fname": "cf_06_0066",
        "type": "formula",
        "content": "IB상담건수_VOC_B0M = SUM(IB상담건수_VOC민원_B0M, IB상담건수_VOC불만_B0M)",
    },
    {
        "columns": ["IB상담건수_VOC민원_R6M", "IB상담건수_VOC불만_R6M"],
        "output": "IB상담건수_VOC_R6M",
        "fname": "cf_06_0089",
        "type": "formula",
        "content": "IB상담건수_VOC_R6M = SUM(IB상담건수_VOC민원_R6M, IB상담건수_VOC불만_R6M)",
    },
    {
        "columns": ["기준년월"],
        "output": "당사PAY_방문횟수_B0M",
        "fname": "cf_06_0096",
        "type": "formula",
        "content": "당사PAY_방문횟수_B0M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "당사PAY_방문횟수_R6M",
        "fname": "cf_06_0097",
        "type": "formula",
        "content": "당사PAY_방문횟수_R6M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "당사PAY_방문월수_R6M",
        "fname": "cf_06_0098",
        "type": "formula",
        "content": "당사PAY_방문월수_R6M = 0",
    },

    # 7.마케팅 테이블 컬럼 Constraint
    {
        "columns": ["컨택건수_카드론_TM_B0M", "컨택건수_카드론_TM_R6M"],
        "fname": "cc_07_0001",
        "type": "constraint",
        "content": "컨택건수_카드론_TM_B0M <= 컨택건수_카드론_TM_R6M",
    },
    {
        "columns": ["컨택건수_리볼빙_TM_B0M", "컨택건수_리볼빙_TM_R6M"],
        "fname": "cc_07_0002",
        "type": "constraint",
        "content": "컨택건수_리볼빙_TM_B0M <= 컨택건수_리볼빙_TM_R6M",
    },
    {
        "columns": ["컨택건수_CA_TM_B0M", "컨택건수_CA_TM_R6M"],
        "fname": "cc_07_0003",
        "type": "constraint",
        "content": "컨택건수_CA_TM_B0M <= 컨택건수_CA_TM_R6M",
    },
    {
        "columns": ["컨택건수_이용유도_TM_B0M", "컨택건수_이용유도_TM_R6M"],
        "fname": "cc_07_0004",
        "type": "constraint",
        "content": "컨택건수_이용유도_TM_B0M <= 컨택건수_이용유도_TM_R6M",
    },
    {
        "columns": ["컨택건수_신용발급_TM_B0M", "컨택건수_신용발급_TM_R6M"],
        "fname": "cc_07_0005",
        "type": "constraint",
        "content": "컨택건수_신용발급_TM_B0M <= 컨택건수_신용발급_TM_R6M",
    },
    {
        "columns": ["컨택건수_부대서비스_TM_B0M", "컨택건수_부대서비스_TM_R6M"],
        "fname": "cc_07_0006",
        "type": "constraint",
        "content": "컨택건수_부대서비스_TM_B0M <= 컨택건수_부대서비스_TM_R6M",
    },
    {
        "columns": ["컨택건수_포인트소진_TM_B0M", "컨택건수_포인트소진_TM_R6M"],
        "fname": "cc_07_0007",
        "type": "constraint",
        "content": "컨택건수_포인트소진_TM_B0M <= 컨택건수_포인트소진_TM_R6M",
    },
    {
        "columns": ["컨택건수_보험_TM_B0M", "컨택건수_보험_TM_R6M"],
        "fname": "cc_07_0008",
        "type": "constraint",
        "content": "컨택건수_보험_TM_B0M <= 컨택건수_보험_TM_R6M",
    },
    {
        "columns": ["컨택건수_카드론_LMS_B0M", "컨택건수_카드론_LMS_R6M"],
        "fname": "cc_07_0009",
        "type": "constraint",
        "content": "컨택건수_카드론_LMS_B0M <= 컨택건수_카드론_LMS_R6M",
    },
    {
        "columns": ["컨택건수_CA_LMS_B0M", "컨택건수_CA_LMS_R6M"],
        "fname": "cc_07_0010",
        "type": "constraint",
        "content": "컨택건수_CA_LMS_B0M <= 컨택건수_CA_LMS_R6M",
    },
    {
        "columns": ["컨택건수_리볼빙_LMS_B0M", "컨택건수_리볼빙_LMS_R6M"],
        "fname": "cc_07_0011",
        "type": "constraint",
        "content": "컨택건수_리볼빙_LMS_B0M <= 컨택건수_리볼빙_LMS_R6M",
    },
    {
        "columns": ["컨택건수_이용유도_LMS_B0M", "컨택건수_이용유도_LMS_R6M"],
        "fname": "cc_07_0012",
        "type": "constraint",
        "content": "컨택건수_이용유도_LMS_B0M <= 컨택건수_이용유도_LMS_R6M",
    },
    {
        "columns": ["컨택건수_카드론_EM_B0M", "컨택건수_카드론_EM_R6M"],
        "fname": "cc_07_0013",
        "type": "constraint",
        "content": "컨택건수_카드론_EM_B0M <= 컨택건수_카드론_EM_R6M",
    },
    {
        "columns": ["컨택건수_CA_EM_B0M", "컨택건수_CA_EM_R6M"],
        "fname": "cc_07_0014",
        "type": "constraint",
        "content": "컨택건수_CA_EM_B0M <= 컨택건수_CA_EM_R6M",
    },
    {
        "columns": ["컨택건수_리볼빙_EM_B0M", "컨택건수_리볼빙_EM_R6M"],
        "fname": "cc_07_0015",
        "type": "constraint",
        "content": "컨택건수_리볼빙_EM_B0M <= 컨택건수_리볼빙_EM_R6M",
    },
    {
        "columns": ["컨택건수_이용유도_EM_B0M", "컨택건수_이용유도_EM_R6M"],
        "fname": "cc_07_0016",
        "type": "constraint",
        "content": "컨택건수_이용유도_EM_B0M <= 컨택건수_이용유도_EM_R6M",
    },
    {
        "columns": ["컨택건수_카드론_청구서_B0M", "컨택건수_카드론_청구서_R6M"],
        "fname": "cc_07_0017",
        "type": "constraint",
        "content": "컨택건수_카드론_청구서_B0M <= 컨택건수_카드론_청구서_R6M",
    },
    {
        "columns": ["컨택건수_CA_청구서_B0M", "컨택건수_CA_청구서_R6M"],
        "fname": "cc_07_0018",
        "type": "constraint",
        "content": "컨택건수_CA_청구서_B0M <= 컨택건수_CA_청구서_R6M",
    },
    {
        "columns": ["컨택건수_리볼빙_청구서_B0M", "컨택건수_리볼빙_청구서_R6M"],
        "fname": "cc_07_0019",
        "type": "constraint",
        "content": "컨택건수_리볼빙_청구서_B0M <= 컨택건수_리볼빙_청구서_R6M",
    },
    {
        "columns": ["컨택건수_이용유도_청구서_B0M", "컨택건수_이용유도_청구서_R6M"],
        "fname": "cc_07_0020",
        "type": "constraint",
        "content": "컨택건수_이용유도_청구서_B0M <= 컨택건수_이용유도_청구서_R6M",
    },
    {
        "columns": ["컨택건수_카드론_인터넷_B0M", "컨택건수_카드론_인터넷_R6M"],
        "fname": "cc_07_0021",
        "type": "constraint",
        "content": "컨택건수_카드론_인터넷_B0M <= 컨택건수_카드론_인터넷_R6M",
    },
    {
        "columns": ["컨택건수_CA_인터넷_B0M", "컨택건수_CA_인터넷_R6M"],
        "fname": "cc_07_0022",
        "type": "constraint",
        "content": "컨택건수_CA_인터넷_B0M <= 컨택건수_CA_인터넷_R6M",
    },
    {
        "columns": ["컨택건수_리볼빙_인터넷_B0M", "컨택건수_리볼빙_인터넷_R6M"],
        "fname": "cc_07_0023",
        "type": "constraint",
        "content": "컨택건수_리볼빙_인터넷_B0M <= 컨택건수_리볼빙_인터넷_R6M",
    },
    {
        "columns": ["컨택건수_이용유도_인터넷_B0M", "컨택건수_이용유도_인터넷_R6M"],
        "fname": "cc_07_0024",
        "type": "constraint",
        "content": "컨택건수_이용유도_인터넷_B0M <= 컨택건수_이용유도_인터넷_R6M",
    },
    {
        "columns": ["컨택건수_카드론_당사앱_B0M", "컨택건수_카드론_당사앱_R6M"],
        "fname": "cc_07_0025",
        "type": "constraint",
        "content": "컨택건수_카드론_당사앱_B0M <= 컨택건수_카드론_당사앱_R6M",
    },
    {
        "columns": ["컨택건수_CA_당사앱_B0M", "컨택건수_CA_당사앱_R6M"],
        "fname": "cc_07_0026",
        "type": "constraint",
        "content": "컨택건수_CA_당사앱_B0M <= 컨택건수_CA_당사앱_R6M",
    },
    {
        "columns": ["컨택건수_리볼빙_당사앱_B0M", "컨택건수_리볼빙_당사앱_R6M"],
        "fname": "cc_07_0027",
        "type": "constraint",
        "content": "컨택건수_리볼빙_당사앱_B0M <= 컨택건수_리볼빙_당사앱_R6M",
    },
    {
        "columns": ["컨택건수_이용유도_당사앱_B0M", "컨택건수_이용유도_당사앱_R6M"],
        "fname": "cc_07_0028",
        "type": "constraint",
        "content": "컨택건수_이용유도_당사앱_B0M <= 컨택건수_이용유도_당사앱_R6M",
    },
    {
        "columns": ["컨택건수_채권_B0M", "컨택건수_채권_R6M"],
        "fname": "cc_07_0029",
        "type": "constraint",
        "content": "컨택건수_채권_B0M <= 컨택건수_채권_R6M",
    },
    {
        "columns": ["컨택건수_FDS_B0M", "컨택건수_FDS_R6M"],
        "fname": "cc_07_0030",
        "type": "constraint",
        "content": "컨택건수_FDS_B0M <= 컨택건수_FDS_R6M",
    },
    {
        "columns": ["캠페인접촉건수_R12M", "캠페인접촉일수_R12M"],
        "fname": "cc_07_0031",
        "type": "constraint",
        "content": "캠페인접촉건수_R12M >= 캠페인접촉일수_R12M",
    },

    # 7.마케팅 테이블 컬럼 Formula
    {
        "columns": ["기준년월"],
        "output": "컨택건수_CA_EM_B0M",
        "fname": "cf_07_0019",
        "type": "formula",
        "content": "컨택건수_CA_EM_B0M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_리볼빙_EM_B0M",
        "fname": "cf_07_0020",
        "type": "formula",
        "content": "컨택건수_리볼빙_EM_B0M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_리볼빙_청구서_B0M",
        "fname": "cf_07_0024",
        "type": "formula",
        "content": "컨택건수_리볼빙_청구서_B0M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_카드론_인터넷_B0M",
        "fname": "cf_07_0026",
        "type": "formula",
        "content": "컨택건수_카드론_인터넷_B0M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_CA_인터넷_B0M",
        "fname": "cf_07_0027",
        "type": "formula",
        "content": "컨택건수_CA_인터넷_B0M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_리볼빙_인터넷_B0M",
        "fname": "cf_07_0028",
        "type": "formula",
        "content": "컨택건수_리볼빙_인터넷_B0M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_리볼빙_당사앱_B0M",
        "fname": "cf_07_0032",
        "type": "formula",
        "content": "컨택건수_리볼빙_당사앱_B0M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_CA_EM_R6M",
        "fname": "cf_07_0047",
        "type": "formula",
        "content": "컨택건수_CA_EM_R6M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_리볼빙_EM_R6M",
        "fname": "cf_07_0048",
        "type": "formula",
        "content": "컨택건수_리볼빙_EM_R6M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_리볼빙_청구서_R6M",
        "fname": "cf_07_0052",
        "type": "formula",
        "content": "컨택건수_리볼빙_청구서_R6M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_카드론_인터넷_R6M",
        "fname": "cf_07_0054",
        "type": "formula",
        "content": "컨택건수_카드론_인터넷_R6M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_CA_인터넷_R6M",
        "fname": "cf_07_0055",
        "type": "formula",
        "content": "컨택건수_CA_인터넷_R6M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_리볼빙_인터넷_R6M",
        "fname": "cf_07_0056",
        "type": "formula",
        "content": "컨택건수_리볼빙_인터넷_R6M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "컨택건수_리볼빙_당사앱_R6M",
        "fname": "cf_07_0060",
        "type": "formula",
        "content": "컨택건수_리볼빙_당사앱_R6M = 0",
    },

    # 8.성과 테이블 컬럼 Constraints
    {
        "columns": ["증감율_이용건수_신판_전월", "증감율_이용건수_CA_전월", "증감율_이용건수_신용_전월"],
        "fname": "cc_08_0001",
        "type": "constraint",
        "content": """min(증감율_이용건수_신판_전월, 증감율_이용건수_CA_전월) <= 증감율_이용건수_신용_전월
                      <= max(증감율_이용건수_신판_전월, 증감율_이용건수_CA_전월)""",
    },
    {
        "columns": ["증감율_이용건수_일시불_전월", "증감율_이용건수_할부_전월", "증감율_이용건수_신판_전월"],
        "fname": "cc_08_0002",
        "type": "constraint",
        "content": """min(증감율_이용건수_일시불_전월, 증감율_이용건수_할부_전월) <= 증감율_이용건수_신판_전월
                      <= max(증감율_이용건수_일시불_전월, 증감율_이용건수_할부_전월)""",
    },
    {
        "columns": ["증감율_이용금액_신판_전월", "증감율_이용금액_CA_전월", "증감율_이용금액_신용_전월"],
        "fname": "cc_08_0003",
        "type": "constraint",
        "content": """min(증감율_이용금액_신판_전월, 증감율_이용금액_CA_전월) <= 증감율_이용금액_신용_전월
                      <= max(증감율_이용금액_신판_전월, 증감율_이용금액_CA_전월)""",
    },
    {
        "columns": ["증감율_이용금액_일시불_전월", "증감율_이용금액_할부_전월", "증감율_이용금액_신판_전월"],
        "fname": "cc_08_0004",
        "type": "constraint",
        "content": """min(증감율_이용금액_일시불_전월, 증감율_이용금액_할부_전월) <= 증감율_이용금액_신판_전월
                      <= max(증감율_이용금액_일시불_전월, 증감율_이용금액_할부_전월)""",
    },
    {
        "columns": ["증감율_이용건수_신판_분기", "증감율_이용건수_CA_분기", "증감율_이용건수_신용_분기"],
        "fname": "cc_08_0005",
        "type": "constraint",
        "content": """min(증감율_이용건수_신판_분기, 증감율_이용건수_CA_분기) <= 증감율_이용건수_신용_분기
                      <= max(증감율_이용건수_신판_분기, 증감율_이용건수_CA_분기)""",
    },
    {
        "columns": ["증감율_이용건수_일시불_분기", "증감율_이용건수_할부_분기", "증감율_이용건수_신판_분기"],
        "fname": "cc_08_0006",
        "type": "constraint",
        "content": """min(증감율_이용건수_일시불_분기, 증감율_이용건수_할부_분기) <= 증감율_이용건수_신판_분기
                      <= max(증감율_이용건수_일시불_분기, 증감율_이용건수_할부_분기)""",
    },
    {
        "columns": ["증감율_이용금액_신판_분기", "증감율_이용금액_CA_분기", "증감율_이용금액_신용_분기"],
        "fname": "cc_08_0007",
        "type": "constraint",
        "content": """min(증감율_이용금액_신판_분기, 증감율_이용금액_CA_분기) <= 증감율_이용금액_신용_분기
                      <= max(증감율_이용금액_신판_분기, 증감율_이용금액_CA_분기)""",
    },
    {
        "columns": ["증감율_이용금액_일시불_분기", "증감율_이용금액_할부_분기", "증감율_이용금액_신판_분기"],
        "fname": "cc_08_0008",
        "type": "constraint",
        "content": """min(증감율_이용금액_일시불_분기, 증감율_이용금액_할부_분기) <= 증감율_이용금액_신판_분기
                      <= max(증감율_이용금액_일시불_분기, 증감율_이용금액_할부_분기)""",
    },
    {
        "columns": ["잔액_신판최대한도소진율_r3m", "잔액_신판평균한도소진율_r3m"],
        "fname": "cc_08_0009",
        "type": "constraint",
        "content": "잔액_신판최대한도소진율_r3m >= 잔액_신판평균한도소진율_r3m",
    },
    {
        "columns": ["잔액_신판최대한도소진율_r6m", "잔액_신판평균한도소진율_r6m"],
        "fname": "cc_08_0010",
        "type": "constraint",
        "content": "잔액_신판최대한도소진율_r6m >= 잔액_신판평균한도소진율_r6m",
    },
    {
        "columns": ["잔액_신판ca최대한도소진율_r3m", "잔액_신판ca평균한도소진율_r3m"],
        "fname": "cc_08_0011",
        "type": "constraint",
        "content": "잔액_신판ca최대한도소진율_r3m >= 잔액_신판ca평균한도소진율_r3m",
    },
    {
        "columns": ["잔액_신판ca최대한도소진율_r6m", "잔액_신판ca평균한도소진율_r6m"],
        "fname": "cc_08_0012",
        "type": "constraint",
        "content": "잔액_신판ca최대한도소진율_r6m >= 잔액_신판ca평균한도소진율_r6m",
    },

    # 8.성과 테이블 컬럼 Formula
    {
        "columns": ["증감율_이용건수_CA_전월", "증감율_이용건수_신판_전월", "증감율_이용건수_신용_전월"],
        "output": "증감율_이용건수_신용_전월",
        "fname": "cf_08_0005",
        "type": "formula",
        "content": """IF 증감율_이용건수_CA_전월 == 0 THEN 증감율_이용건수_신용_전월 = 증감율_이용건수_신판_전월
                       ELIF 증감율_이용건수_신판_전월 == 0 THEN 증감율_이용건수_신용_전월 = 증감율_이용건수_CA_전월
                       ELSE PASS""",
    },
    {
        "columns": ["증감율_이용건수_할부_전월", "증감율_이용건수_일시불_전월", "증감율_이용건수_신판_전월"],
        "output": "증감율_이용건수_신판_전월",
        "fname": "cf_08_0006",
        "type": "formula",
        "content": """IF 증감율_이용건수_할부_전월 == 0 THEN 증감율_이용건수_신판_전월 = 증감율_이용건수_일시불_전월
                      ELIF 증감율_이용건수_일시불_전월 == 0 THEN 증감율_이용건수_신판_전월 = 증감율_이용건수_할부_전월
                      ELSE PASS""",
    },
    {
        "columns": ["증감율_이용금액_CA_전월", "증감율_이용금액_신판_전월", "증감율_이용금액_신용_전월"],
        "output": "증감율_이용금액_신용_전월",
        "fname": "cf_08_0012",
        "type": "formula",
        "content": """IF 증감율_이용금액_CA_전월 == 0 THEN 증감율_이용금액_신용_전월 = 증감율_이용금액_신판_전월
                       ELIF 증감율_이용금액_신판_전월 == 0 THEN 증감율_이용금액_신용_전월 = 증감율_이용금액_CA_전월
                       ELSE PASS""",
    },
    {
        "columns": ["증감율_이용금액_할부_전월", "증감율_이용금액_일시불_전월", "증감율_이용금액_신판_전월"],
        "output": "증감율_이용금액_신판_전월",
        "fname": "cf_08_0013",
        "type": "formula",
        "content": """IF 증감율_이용금액_할부_전월 == 0 THEN 증감율_이용금액_신판_전월 = 증감율_이용금액_일시불_전월
                      ELIF 증감율_이용금액_일시불_전월 == 0 THEN 증감율_이용금액_신판_전월 = 증감율_이용금액_할부_전월
                      ELSE PASS""",
    },
    {
        "columns": ["증감율_이용건수_CA_분기", "증감율_이용건수_신판_분기", "증감율_이용건수_신용_분기"],
        "output": "증감율_이용건수_신용_분기",
        "fname": "cf_08_0033",
        "type": "formula",
        "content": """IF 증감율_이용건수_CA_분기 == 0 THEN 증감율_이용건수_신용_분기 = 증감율_이용건수_신판_분기
                       ELIF 증감율_이용건수_신판_분기 == 0 THEN 증감율_이용건수_신용_분기 = 증감율_이용건수_CA_분기
                       ELSE PASS""",
    },
    {
        "columns": ["증감율_이용건수_할부_분기", "증감율_이용건수_일시불_분기", "증감율_이용건수_신판_분기"],
        "output": "증감율_이용건수_신판_분기",
        "fname": "cf_08_0034",
        "type": "formula",
        "content": """IF 증감율_이용건수_할부_분기 == 0 THEN 증감율_이용건수_신판_분기 = 증감율_이용건수_일시불_분기
                      ELIF 증감율_이용건수_일시불_분기 == 0 THEN 증감율_이용건수_신판_분기 = 증감율_이용건수_할부_분기
                      ELSE PASS""",
    },
    {
        "columns": ["증감율_이용금액_CA_분기", "증감율_이용금액_신판_분기", "증감율_이용금액_신용_분기"],
        "output": "증감율_이용금액_신용_분기",
        "fname": "cf_08_0040",
        "type": "formula",
        "content": """IF 증감율_이용금액_CA_분기 == 0 THEN 증감율_이용금액_신용_분기 = 증감율_이용금액_신판_분기
                       ELIF 증감율_이용금액_신판_분기 == 0 THEN 증감율_이용금액_신용_분기 = 증감율_이용금액_CA_분기
                       ELSE PASS""",
    },
    {
        "columns": ["증감율_이용금액_할부_분기", "증감율_이용금액_일시불_분기", "증감율_이용금액_신판_분기"],
        "output": "증감율_이용금액_신판_분기",
        "fname": "cf_08_0041",
        "type": "formula",
        "content": """IF 증감율_이용금액_할부_분기 == 0 THEN 증감율_이용금액_신판_분기 = 증감율_이용금액_일시불_분기
                      ELIF 증감율_이용금액_일시불_분기 == 0 THEN 증감율_이용금액_신판_분기 = 증감율_이용금액_할부_분기
                      ELSE PASS""",
    },

    # 3.승인.매출 테이블 컬럼 Constraint
    {
        "columns": ["기준년월", "최종이용일자_할부", "이용후경과월_할부"],
        "fname": "cc_03_0147",
        "type": "constraint",
        "content": "MONTHS_BETWEEN(최종이용일자_할부, 기준년월) == 이용후경과월_할부",
    },
    {
        "columns": ["이용금액_신용_B0M", "카드이용한도금액"],
        "fname": "cc_03_0001",
        "type": "constraint",
        "content": "이용금액_신용_B0M <= 카드이용한도금액",
    },
    {
        "columns": ["이용금액_CA_B0M", "CA한도금액"],
        "fname": "cc_03_0002",
        "type": "constraint",
        "content": "이용금액_CA_B0M <= CA한도금액",
    },
    {
        "columns": ["이용금액_카드론_B0M", "월상환론한도금액"],
        "fname": "cc_03_0003",
        "type": "constraint",
        "content": "이용금액_카드론_B0M <= 월상환론한도금액",
    },
    {
        "columns": ["이용건수_일시불_R12M", "이용후경과월_일시불"],
        "fname": "cc_03_0004",
        "type": "constraint",
        "content": "IF 이용건수_일시불_R12M >0: 이용후경과월_일시불 < 12",
    },
    {
        "columns": ["이용건수_할부_R12M", "이용후경과월_할부"],
        "fname": "cc_03_0005",
        "type": "constraint",
        "content": "IF 이용건수_할부_R12M >0: 이용후경과월_할부 < 12",
    },
    {
        "columns": ["이용건수_할부_유이자_R12M", "이용후경과월_할부_유이자"],
        "fname": "cc_03_0006",
        "type": "constraint",
        "content": "IF 이용건수_할부_유이자_R12M >0: 이용후경과월_할부_유이자 < 12",
    },
    {
        "columns": ["이용건수_할부_무이자_R12M", "이용후경과월_할부_무이자"],
        "fname": "cc_03_0007",
        "type": "constraint",
        "content": "IF 이용건수_할부_무이자_R12M >0: 이용후경과월_할부_무이자 < 12",
    },
    {
        "columns": ["이용건수_부분무이자_R12M", "이용후경과월_부분무이자"],
        "fname": "cc_03_0008",
        "type": "constraint",
        "content": "IF 이용건수_부분무이자_R12M >0: 이용후경과월_부분무이자 < 12",
    },
    {
        "columns": ["이용건수_CA_R12M", "이용후경과월_CA"],
        "fname": "cc_03_0009",
        "type": "constraint",
        "content": "IF 이용건수_CA_R12M >0: 이용후경과월_CA < 12",
    },
    {
        "columns": ["이용건수_체크_R12M", "이용후경과월_체크"],
        "fname": "cc_03_0010",
        "type": "constraint",
        "content": "IF 이용건수_체크_R12M >0: 이용후경과월_체크 < 12",
    },
    {
        "columns": ["이용건수_카드론_R12M", "이용후경과월_카드론"],
        "fname": "cc_03_0011",
        "type": "constraint",
        "content": "IF 이용건수_카드론_R12M >0: 이용후경과월_카드론 < 12",
    },
    {
        "columns": ["이용건수_신용_B0M", "이용건수_신용_R3M", "이용건수_신용_R6M", "이용건수_신용_R12M"],
        "fname": "cc_03_0012",
        "type": "constraint",
        "content": "이용건수_신용_B0M <= 이용건수_신용_R3M <= 이용건수_신용_R6M <= 이용건수_신용_R12M",
    },
    {
        "columns": ["이용건수_신판_B0M", "이용건수_신판_R3M", "이용건수_신판_R6M", "이용건수_신판_R12M"],
        "fname": "cc_03_0013",
        "type": "constraint",
        "content": "이용건수_신판_B0M <= 이용건수_신판_R3M <= 이용건수_신판_R6M <= 이용건수_신판_R12M",
    },
    {
        "columns": ["이용건수_일시불_B0M", "이용건수_일시불_R3M", "이용건수_일시불_R6M", "이용건수_일시불_R12M"],
        "fname": "cc_03_0014",
        "type": "constraint",
        "content": "이용건수_일시불_B0M <= 이용건수_일시불_R3M <= 이용건수_일시불_R6M <= 이용건수_일시불_R12M",
    },
    {
        "columns": ["이용건수_할부_B0M", "이용건수_할부_R3M", "이용건수_할부_R6M", "이용건수_할부_R12M"],
        "fname": "cc_03_0015",
        "type": "constraint",
        "content": "이용건수_할부_B0M <= 이용건수_할부_R3M <= 이용건수_할부_R6M <= 이용건수_할부_R12M",
    },
    {
        "columns": [
            "이용건수_할부_유이자_B0M",
            "이용건수_할부_유이자_R3M",
            "이용건수_할부_유이자_R6M",
            "이용건수_할부_유이자_R12M",
        ],
        "fname": "cc_03_0016",
        "type": "constraint",
        "content": "이용건수_할부_유이자_B0M <= 이용건수_할부_유이자_R3M <= 이용건수_할부_유이자_R6M <= 이용건수_할부_유이자_R12M",
    },
    {
        "columns": [
            "이용건수_할부_무이자_B0M",
            "이용건수_할부_무이자_R3M",
            "이용건수_할부_무이자_R6M",
            "이용건수_할부_무이자_R12M",
        ],
        "fname": "cc_03_0017",
        "type": "constraint",
        "content": "이용건수_할부_무이자_B0M <= 이용건수_할부_무이자_R3M <= 이용건수_할부_무이자_R6M <= 이용건수_할부_무이자_R12M",
    },
    {
        "columns": [
            "이용건수_부분무이자_B0M",
            "이용건수_부분무이자_R3M",
            "이용건수_부분무이자_R6M",
            "이용건수_부분무이자_R12M",
        ],
        "fname": "cc_03_0018",
        "type": "constraint",
        "content": "이용건수_부분무이자_B0M <= 이용건수_부분무이자_R3M <= 이용건수_부분무이자_R6M <= 이용건수_부분무이자_R12M",
    },
    {
        "columns": ["이용건수_CA_B0M", "이용건수_CA_R3M", "이용건수_CA_R6M", "이용건수_CA_R12M"],
        "fname": "cc_03_0019",
        "type": "constraint",
        "content": "이용건수_CA_B0M <= 이용건수_CA_R3M <= 이용건수_CA_R6M <= 이용건수_CA_R12M",
    },
    {
        "columns": ["이용건수_체크_B0M", "이용건수_체크_R3M", "이용건수_체크_R6M", "이용건수_체크_R12M"],
        "fname": "cc_03_0020",
        "type": "constraint",
        "content": "이용건수_체크_B0M <= 이용건수_체크_R3M <= 이용건수_체크_R6M <= 이용건수_체크_R12M",
    },
    {
        "columns": ["이용건수_카드론_B0M", "이용건수_카드론_R3M", "이용건수_카드론_R6M", "이용건수_카드론_R12M"],
        "fname": "cc_03_0021",
        "type": "constraint",
        "content": "이용건수_카드론_B0M <= 이용건수_카드론_R3M <= 이용건수_카드론_R6M <= 이용건수_카드론_R12M",
    },
    {
        "columns": ["이용금액_신용_B0M", "이용금액_신용_R3M", "이용금액_신용_R6M", "이용금액_신용_R12M"],
        "fname": "cc_03_0022",
        "type": "constraint",
        "content": "이용건수_금액_B0M <= 이용건수_금액_R3M <= 이용건수_금액_R6M <= 이용건수_금액_R12M",
    },
    {
        "columns": ["이용금액_신판_B0M", "이용금액_신판_R3M", "이용금액_신판_R6M", "이용금액_신판_R12M"],
        "fname": "cc_03_0023",
        "type": "constraint",
        "content": "이용금액_신판_B0M <= 이용금액_신판_R3M <= 이용금액_신판_R6M <= 이용금액_신판_R12M",
    },
    {
        "columns": ["이용금액_일시불_B0M", "이용금액_일시불_R3M", "이용금액_일시불_R6M", "이용금액_일시불_R12M"],
        "fname": "cc_03_0024",
        "type": "constraint",
        "content": "이용금액_일시불_B0M <= 이용금액_일시불_R3M <= 이용금액_일시불_R6M <= 이용금액_일시불_R12M",
    },
    {
        "columns": ["이용금액_할부_B0M", "이용금액_할부_R3M", "이용금액_할부_R6M", "이용금액_할부_R12M"],
        "fname": "cc_03_0025",
        "type": "constraint",
        "content": "이용금액_할부_B0M <= 이용금액_할부_R3M <= 이용금액_할부_R6M <= 이용금액_할부_R12M",
    },
    {
        "columns": [
            "이용금액_할부_유이자_B0M",
            "이용금액_할부_유이자_R3M",
            "이용금액_할부_유이자_R6M",
            "이용금액_할부_유이자_R12M",
        ],
        "fname": "cc_03_0026",
        "type": "constraint",
        "content": "이용금액_할부_유이자_B0M <= 이용금액_할부_유이자_R3M <= 이용금액_할부_유이자_R6M <= 이용금액_할부_유이자_R12M",
    },
    {
        "columns": [
            "이용금액_할부_무이자_B0M",
            "이용금액_할부_무이자_R3M",
            "이용금액_할부_무이자_R6M",
            "이용금액_할부_무이자_R12M",
        ],
        "fname": "cc_03_0027",
        "type": "constraint",
        "content": "이용금액_할부_무이자_B0M <= 이용금액_할부_무이자_R3M <= 이용금액_할부_무이자_R6M <= 이용금액_할부_무이자_R12M",
    },
    {
        "columns": [
            "이용금액_부분무이자_B0M",
            "이용금액_부분무이자_R3M",
            "이용금액_부분무이자_R6M",
            "이용금액_부분무이자_R12M",
        ],
        "fname": "cc_03_0028",
        "type": "constraint",
        "content": "이용금액_부분무이자_B0M <= 이용금액_부분무이자_R3M <= 이용금액_부분무이자_R6M <= 이용금액_부분무이자_R12M",
    },
    {
        "columns": ["이용금액_CA_B0M", "이용금액_CA_R3M", "이용금액_CA_R6M", "이용금액_CA_R12M"],
        "fname": "cc_03_0029",
        "type": "constraint",
        "content": "이용금액_CA_B0M <= 이용금액_CA_R3M <= 이용금액_CA_R6M <= 이용금액_CA_R12M",
    },
    {
        "columns": ["이용금액_체크_B0M", "이용금액_체크_R3M", "이용금액_체크_R6M", "이용금액_체크_R12M"],
        "fname": "cc_03_0030",
        "type": "constraint",
        "content": "이용금액_체크_B0M <= 이용금액_체크_R3M <= 이용금액_체크_R6M <= 이용금액_체크_R12M",
    },
    {
        "columns": ["이용금액_카드론_B0M", "이용금액_카드론_R3M", "이용금액_카드론_R6M", "이용금액_카드론_R12M"],
        "fname": "cc_03_0031",
        "type": "constraint",
        "content": "이용금액_카드론_B0M <= 이용금액_카드론_R3M <= 이용금액_카드론_R6M <= 이용금액_카드론_R12M",
    },
    {
        "columns": ["최대이용금액_신용_R12M", "최대이용금액_신판_R12M", "최대이용금액_CA_R12M"],
        "fname": "cc_03_0156",
        "type": "constraint",
        "content": "최대이용금액_신용_R12M >= MAX(최대이용금액_신판_R12M, 최대이용금액_CA_R12M)",
    },
    {
        "columns": ["최대이용금액_신판_R12M", "최대이용금액_일시불_R12M", "최대이용금액_할부_R12M"],
        "fname": "cc_03_0157",
        "type": "constraint",
        "content": "최대이용금액_신판_R12M >= MAX(최대이용금액_일시불_R12M, 최대이용금액_할부_R12M)",
    },
    # {
    #     "columns": ["최근6개월 이용금액_일시불_B0M", "최대이용금액_일시불_R12M"],
    #     "fname": "cc_03_0032"
    #     "type": "constraint",
    #     "content": "MAX(최근6개월 이용금액_일시불_B0M) <= 최대이용금액_일시불_R12M",
    # },
    # {
    #     "columns": ["최근6개월 이용금액_할부_유이자_B0M", "최대이용금액_할부_유이자_R12M"],
    #     "fname": "cc_03_0033"
    #     "type": "constraint",
    #     "content": "MAX(최근6개월 이용금액_할부_유이자_B0M) <= 최대이용금액_할부_유이자_R12M",
    # },
    # {
    #     "columns": ["최근6개월 이용금액_할부_무이자_B0M", "최대이용금액_할부_무이자_R12M"],
    #     "fname": "cc_03_0034"
    #     "type": "constraint",
    #     "content": "MAX(최근6개월 이용금액_할부_무이자_B0M <= 최대이용금액_할부_무이자_R12M",
    # },
    # {
    #     "columns": ["최근6개월 이용금액_할부_무이자_B0M", "최대이용금액_할부_무이자_R12M"],
    #     "fname": "cc_03_0034"
    #     "type": "constraint",
    #     "content": "MAX(최근6개월 이용금액_할부_무이자_B0M <= 최대이용금액_할부_무이자_R12M",
    # },
    # {
    #     "columns": ["최근6개월 이용금액_부분무이자_B0M", "최대이용금액_부분무이자_R12M"],
    #     "fname": "cc_03_0035"
    #     "type": "constraint",
    #     "content": "MAX(최근6개월 이용금액_부분무이자_B0M <= 최대이용금액_부분무이자_R12M",
    # },
    # {
    #     "columns": ["최근6개월 이용금액_CA_B0M", "최대이용금액_CA_R12M"],
    #     "fname": "cc_03_0036"
    #     "type": "constraint",
    #     "content": "MAX(최근6개월 이용금액_CA_B0M) <= 최대이용금액_CA_R12M",
    # },
    # {
    #     "columns": ["최근6개월 이용금액_체크_B0M", "최대이용금액_체크_R12M"],
    #     "fname": "cc_03_0037"
    #     "type": "constraint",
    #     "content": "MAX(최근6개월 이용금액_체크_B0M) <= 최대이용금액_체크_R12M",
    # },
    # {
    #     "columns": ["최근6개월 이용금액_카드론_B0M", "최대이용금액_카드론_R12M"],
    #     "fname": "cc_03_0038"
    #     "type": "constraint",
    #     "content": "MAX(최근6개월 이용금액_카드론_B0M) <= 최대이용금액_카드론_R12M",
    # },
    {
        "columns": ["이용개월수_신용_R3M", "이용개월수_신용_R6M", "이용개월수_신용_R12M"],
        "fname": "cc_03_0039",
        "type": "constraint",
        "content": "이용개월수_신용_R3M <= 이용개월수_신용_R6M <= 이용개월수_신용_R12M  <= 12",
    },
    {
        "columns": ["이용개월수_신판_R12M", "이용개월수_CA_R12M", "이용개월수_신용_R12M"],
        "fname": "cc_03_0040",
        "type": "constraint",
        "content": "MAX(이용개월수_신판_R12M, 이용개월수_CA_R12M) <= 이용개월수_신용_R12M",
    },
    {
        "columns": ["이용개월수_신판_R3M", "이용개월수_신판_R6M", "이용개월수_신판_R12M"],
        "fname": "cc_03_0041",
        "type": "constraint",
        "content": "이용개월수_신판_R3M <= 이용개월수_신판_R6M <= 이용개월수_신판_R12M <= 12",
    },
    {
        "columns": ["이용개월수_일시불_R12M", "이용개월수_할부_R12M", "이용개월수_신판_R12M"],
        "fname": "cc_03_0042",
        "type": "constraint",
        "content": "MAX(이용개월수_일시불_R12M, 이용개월수_할부_R12M) <= 이용개월수_신판_R12M",
    },
    {
        "columns": ["이용개월수_일시불_R3M", "이용개월수_일시불_R6M", "이용개월수_일시불_R12M"],
        "fname": "cc_03_0043",
        "type": "constraint",
        "content": "이용개월수_일시불_R3M <= 이용개월수_일시불_R6M <= 이용개월수_일시불_R12M <= 12",
    },
    {
        "columns": ["이용개월수_할부_R3M", "이용개월수_할부_유이자_R6M", "이용개월수_할부_유이자_R12M"],
        "fname": "cc_03_0044",
        "type": "constraint",
        "content": "이용개월수_할부_R3M <= 이용개월수_할부_유이자_R6M <= 이용개월수_할부_유이자_R12M <= 12",
    },
    {
        "columns": [
            "이용개월수_할부_유이자_R12M",
            "이용개월수_할부_무이자_R12M",
            "이용개월수_부분무이자_R12M",
            "이용개월수_할부_R12M",
        ],
        "fname": "cc_03_0045",
        "type": "constraint",
        "content": "MAX(이용개월수_할부_유이자_R12M, 이용개월수_할부_무이자_R12M, 이용개월수_부분무이자_R12M) <= 이용개월수_할부_R12M",
    },
    {
        "columns": ["이용개월수_할부_유이자_R3M", "이용개월수_할부_유이자_R6M", "이용개월수_할부_유이자_R12M"],
        "fname": "cc_03_0046",
        "type": "constraint",
        "content": "이용개월수_할부_유이자_R3M <= 이용개월수_할부_유이자_R6M <= 이용개월수_할부_유이자_R12M <= 12",
    },
    {
        "columns": ["이용개월수_할부_무이자_R3M", "이용개월수_할부_무이자_R6M", "이용개월수_할부_무이자_R12M"],
        "fname": "cc_03_0047",
        "type": "constraint",
        "content": "이용개월수_할부_무이자_R3M <= 이용개월수_할부_무이자_R6M <= 이용개월수_할부_무이자_R12M <= 12",
    },
    {
        "columns": ["이용개월수_부분무이자_R3M", "이용개월수_부분무이자_R6M", "이용개월수_부분무이자_R12M"],
        "fname": "cc_03_0048",
        "type": "constraint",
        "content": "이용개월수_부분무이자_R3M <=  이용개월수_부분무이자_R6M <= 이용개월수_부분무이자_R12M <= 12",
    },
    {
        "columns": ["이용개월수_CA_R3M", "이용개월수_CA_R6M", "이용개월수_CA_R12M"],
        "fname": "cc_03_0049",
        "type": "constraint",
        "content": "이용개월수_CA_R3M <= 이용개월수_CA_R6M <= 이용개월수_CA_R12M <= 12",
    },
    {
        "columns": ["이용개월수_체크_R3M", "이용개월수_체크_R6M", "이용개월수_체크_R12M"],
        "fname": "cc_03_0050",
        "type": "constraint",
        "content": "이용개월수_체크_R3M <= 이용개월수_체크_R6M <= 이용개월수_체크_R12M <= 12",
    },
    {
        "columns": ["이용개월수_카드론_R3M", "이용개월수_카드론_R6M", "이용개월수_카드론_R12M"],
        "fname": "cc_03_0051",
        "type": "constraint",
        "content": "이용개월수_카드론_R3M <= 이용개월수_카드론_R6M <= 이용개월수_카드론_R12M <= 12",
    },
    {
        "columns": ["이용금액_신용_R3M", "이용건수_신용_R3M"],
        "fname": "cc_03_0052",
        "type": "constraint",
        "content": "IF 이용금액_신용_R3M>0: 이용건수_신용_R3M >0",
    },
    {
        "columns": ["이용금액_신판_R3M", "이용건수_신판_R3M"],
        "fname": "cc_03_0053",
        "type": "constraint",
        "content": "IF 이용금액_신판_R3M>0: 이용건수_신판_R3M >0",
    },
    {
        "columns": ["이용금액_일시불_R3M", "이용건수_일시불_R3M"],
        "fname": "cc_03_0054",
        "type": "constraint",
        "content": "IF 이용금액_일시불_R3M>0: 이용건수_일시불_R3M >0",
    },
    {
        "columns": ["이용금액_할부_R3M", "이용건수_할부_R3M"],
        "fname": "cc_03_0055",
        "type": "constraint",
        "content": "IF 이용금액_할부_R3M>0: 이용건수_할부_R3M >0",
    },
    {
        "columns": ["이용금액_할부_유이자_R3M", "이용건수_할부_유이자_R3M"],
        "fname": "cc_03_0056",
        "type": "constraint",
        "content": "IF 이용금액_할부_유이자_R3M>0: 이용건수_할부_유이자_R3M >0",
    },
    {
        "columns": ["이용금액_할부_무이자_R3M", "이용건수_할부_무이자_R3M"],
        "fname": "cc_03_0057",
        "type": "constraint",
        "content": "IF 이용금액_할부_무이자_R3M>0: 이용건수_할부_무이자_R3M >0",
    },
    {
        "columns": ["이용금액_부분무이자_R3M", "이용건수_부분무이자_R3M"],
        "fname": "cc_03_0058",
        "type": "constraint",
        "content": "IF 이용금액_부분무이자_R3M>0: 이용건수_부분무이자_R3M >0",
    },
    {
        "columns": ["이용금액_CA_R3M", "이용건수_CA_R3M"],
        "fname": "cc_03_0059",
        "type": "constraint",
        "content": "IF 이용금액_CA_R3M>0: 이용건수_CA_R3M >0",
    },
    {
        "columns": ["이용금액_체크_R3M", "이용건수_체크_R3M"],
        "fname": "cc_03_0060",
        "type": "constraint",
        "content": "IF 이용금액_체크_R3M>0: 이용건수_체크_R3M >0",
    },
    {
        "columns": ["이용금액_카드론_R3M", "이용건수_카드론_R3M"],
        "fname": "cc_03_0061",
        "type": "constraint",
        "content": "IF 이용금액_카드론_R3M>0: 이용건수_카드론_R3M >0",
    },
    {
        "columns": ["이용개월수_신판_R3M", "이용개월수_CA_R3M", "이용개월수_신용_R3M"],
        "fname": "cc_03_0062",
        "type": "constraint",
        "content": "MAX(이용개월수_신판_R3M, 이용개월수_CA_R3M) <= 이용개월수_신용_R3M",
    },
    {
        "columns": ["이용건수_신용_R3M", "이용개월수_신용_R3M"],
        "fname": "cc_03_0063",
        "type": "constraint",
        "content": "IF 이용건수_신용_R3M>0: 이용개월수_신용_R3M >0",
    },
    {
        "columns": ["이용개월수_일시불_R3M", "이용개월수_할부_R3M", "이용개월수_신판_R3M"],
        "fname": "cc_03_0064",
        "type": "constraint",
        "content": "MAX(이용개월수_일시불_R3M, 이용개월수_할부_R3M) <= 이용개월수_신판_R3M",
    },
    {
        "columns": ["이용건수_신판_R3M", "이용개월수_신판_R3M"],
        "fname": "cc_03_0065",
        "type": "constraint",
        "content": "IF 이용건수_신판_R3M>0: 이용개월수_신판_R3M >0",
    },
    {
        "columns": ["이용건수_일시불_R3M", "이용개월수_일시불_R3M"],
        "fname": "cc_03_0066",
        "type": "constraint",
        "content": "IF 이용건수_일시불_R3M>0: 이용개월수_일시불_R3M >0",
    },
    {
        "columns": [
            "이용개월수_할부_유이자_R3M",
            "이용개월수_할부_무이자_R3M",
            "이용개월수_부분무이자_R3M",
            "이용개월수_할부_R3M",
        ],
        "fname": "cc_03_0067",
        "type": "constraint",
        "content": "MAX(이용개월수_할부_유이자_R3M, 이용개월수_할부_무이자_R3M, 이용개월수_부분무이자_R3M) <= 이용개월수_할부_R3M",
    },
    {
        "columns": ["이용건수_할부_R3M", "이용개월수_할부_R3M"],
        "fname": "cc_03_0068",
        "type": "constraint",
        "content": "IF 이용건수_할부_R3M>0: 이용개월수_할부_R3M >0",
    },
    {
        "columns": ["이용건수_할부_유이자_R3M", "이용개월수_할부_유이자_R3M"],
        "fname": "cc_03_0069",
        "type": "constraint",
        "content": "IF 이용건수_할부_유이자_R3M>0: 이용개월수_할부_유이자_R3M >0",
    },
    {
        "columns": ["이용건수_할부_무이자_R3M", "이용개월수_할부_무이자_R3M"],
        "fname": "cc_03_0070",
        "type": "constraint",
        "content": "IF 이용건수_할부_무이자_R3M>0: 이용개월수_할부_무이자_R3M >0",
    },
    {
        "columns": ["이용건수_부분무이자_R3M", "이용개월수_부분무이자_R3M"],
        "fname": "cc_03_0071",
        "type": "constraint",
        "content": "IF 이용건수_부분무이자_R3M>0: 이용개월수_부분무이자_R3M >0",
    },
    {
        "columns": ["이용건수_CA_R3M", "이용개월수_CA_R3M"],
        "fname": "cc_03_0072",
        "type": "constraint",
        "content": "IF 이용건수_CA_R3M>0: 이용개월수_CA_R3M >0",
    },
    {
        "columns": ["이용건수_체크_R3M", "이용개월수_체크_R3M"],
        "fname": "cc_03_0073",
        "type": "constraint",
        "content": "IF 이용건수_체크_R3M>0: 이용개월수_체크_R3M >0",
    },
    {
        "columns": ["이용건수_카드론_R3M", "이용개월수_카드론_R3M"],
        "fname": "cc_03_0074",
        "type": "constraint",
        "content": "IF 이용건수_카드론_R3M>0: 이용개월수_카드론_R3M >0",
    },
    {
        "columns": ["건수_할부전환_R3M", "건수_할부전환_R6M", "건수_할부전환_R12M"],
        "fname": "cc_03_0075",
        "type": "constraint",
        "content": "건수_할부전환_R3M <= 건수_할부전환_R6M <= 건수_할부전환_R12M",
    },
    {
        "columns": ["금액_할부전환_R3M", "금액_할부전환_R6M", "금액_할부전환_R12M"],
        "fname": "cc_03_0076",
        "type": "constraint",
        "content": "금액_할부전환_R3M <= 금액_할부전환_R6M <= 금액_할부전환_R12M",
    },
    {
        "columns": ["이용개월수_할부전환_R3M", "이용개월수_할부전환_R6M", "이용개월수_할부전환_R12M"],
        "fname": "cc_03_0077",
        "type": "constraint",
        "content": "이용개월수_할부전환_R3M <= 이용개월수_할부전환_R6M <= 이용개월수_할부전환_R12M",
    },
    {
        "columns": ["가맹점매출금액_B1M", "가맹점매출금액_B2M", "이용가맹점수"],
        "fname": "cc_03_0078",
        "type": "constraint",
        "content": "IF (가맹점매출금액_B1M + 가맹점매출금액_B2M) >0: 이용가맹점수 >0",
    },
    {
        "columns": [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
            "이용가맹점수",
        ],
        "fname": "cc_03_0079",
        "type": "constraint",
        "content": "이용가맹점수 >= SUM(IF 이용금액_업종*>0 THEN 1 ELSE 0)",
    },
    {
        "columns": [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
            "이용금액_업종기준",
        ],
        "fname": "cc_03_0080",
        "type": "constraint",
        "content": "이용금액_쇼핑 + 이용금액_요식 + 이용금액_교통 + 이용금액_의료 + 이용금액_납부 + 이용금액_교육 + 이용금액_여유생활 + 이용금액_사교활동 + 이용금액_일상생활 + 이용금액_해외 <= 이용금액_업종기준",
    },
    {
        "columns": [
            "이용금액_쇼핑",
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ],
        "fname": "cc_03_0155",
        "type": "constraint",
        "content": "이용금액_쇼핑 >= SUM(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)",
    },
    {
        "columns": ["_3순위업종_이용금액", "_1순위업종_이용금액"],
        "fname": "cc_03_0081",
        "type": "constraint",
        "content": "_3순위업종_이용금액 <= _1순위업종_이용금액",
    },
    {
        "columns": ["_3순위쇼핑업종_이용금액", "_1순위쇼핑업종_이용금액"],
        "fname": "cc_03_0082",
        "type": "constraint",
        "content": "_3순위쇼핑업종_이용금액 <= _1순위쇼핑업종_이용금액",
    },
    {
        "columns": ["_3순위교통업종_이용금액", "_1순위교통업종_이용금액"],
        "fname": "cc_03_0083",
        "type": "constraint",
        "content": "_3순위교통업종_이용금액 <= _1순위교통업종_이용금액",
    },
    {
        "columns": ["_3순위여유업종_이용금액", "_1순위여유업종_이용금액"],
        "fname": "cc_03_0084",
        "type": "constraint",
        "content": "_3순위여유업종_이용금액 <= _1순위여유업종_이용금액",
    },
    {
        "columns": ["_3순위납부업종_이용금액", "_1순위납부업종_이용금액"],
        "fname": "cc_03_0085",
        "type": "constraint",
        "content": "_3순위납부업종_이용금액 <= _1순위납부업종_이용금액",
    },
    {
        "columns": ["RP건수_B0M", "RP금액_B0M"],
        "fname": "cc_03_0086",
        "type": "constraint",
        "content": "IF RP건수_B0M >0: RP금액_B0M >0",
    },
    {
        "columns": [
            "RP건수_통신_B0M",
            "RP건수_아파트_B0M",
            "RP건수_제휴사서비스직접판매_B0M",
            "RP건수_렌탈_B0M",
            "RP건수_가스_B0M",
            "RP건수_전기_B0M",
            "RP건수_보험_B0M",
            "RP건수_학습비_B0M",
            "RP건수_유선방송_B0M",
            "RP건수_건강_B0M",
            "RP건수_교통_B0M",
            "RP유형건수_B0M",
            "RP건수_B0M"
        ],
        "fname": "cc_03_0148",
        "type": "constraint",
        "content": "SUM(IF RP건수_*_B0M > 0 THEN 1 ELSE 0) <= RP유형건수_B0M <= RP건수_B0M",
    },
    {
        "columns": ["RP건수_통신_B0M", "RP후경과월_통신"],
        "fname": "cc_03_0087",
        "type": "constraint",
        "content": "IF RP건수_통신_B0M > 0: RP후경과월_통신 = 0 ELSE RP후경과월_통신 >0",
    },
    {
        "columns": ["RP건수_아파트_B0M", "RP후경과월_아파트"],
        "fname": "cc_03_0088",
        "type": "constraint",
        "content": "IF RP건수_아파트_B0M > 0: RP후경과월_아파트 = 0 ELSE RP후경과월_아파트 >0",
    },
    {
        "columns": ["RP건수_제휴사서비스직접판매_B0M", "RP후경과월_제휴사서비스직접판매"],
        "fname": "cc_03_0089",
        "type": "constraint",
        "content": "IF RP건수_제휴사서비스직접판매_B0M > 0: RP후경과월_제휴사서비스직접판매 = 0 ELSE RP후경과월_제휴사서비스직접판매 >0",
    },
    {
        "columns": ["RP건수_렌탈_B0M", "RP후경과월_렌탈"],
        "fname": "cc_03_0090",
        "type": "constraint",
        "content": "IF RP건수_렌탈_B0M > 0: RP후경과월_렌탈 = 0 ELSE RP후경과월_렌탈 >0",
    },
    {
        "columns": ["RP건수_가스_B0M", "RP후경과월_가스"],
        "fname": "cc_03_0091",
        "type": "constraint",
        "content": "IF RP건수_가스_B0M > 0: RP후경과월_가스 = 0 ELSE RP후경과월_가스 >0",
    },
    {
        "columns": ["RP건수_전기_B0M", "RP후경과월_전기"],
        "fname": "cc_03_0092",
        "type": "constraint",
        "content": "IF RP건수_전기_B0M > 0: RP후경과월_전기 = 0 ELSE RP후경과월_전기 >0",
    },
    {
        "columns": ["RP건수_보험_B0M", "RP후경과월_보험"],
        "fname": "cc_03_0093",
        "type": "constraint",
        "content": "IF RP건수_보험_B0M > 0: RP후경과월_보험 = 0 ELSE RP후경과월_보험 >0",
    },
    {
        "columns": ["RP건수_학습비_B0M", "RP후경과월_학습비"],
        "fname": "cc_03_0094",
        "type": "constraint",
        "content": "IF RP건수_학습비_B0M > 0: RP후경과월_학습비 = 0 ELSE RP후경과월_학습비 >0",
    },
    {
        "columns": ["RP건수_유선방송_B0M", "RP후경과월_유선방송"],
        "fname": "cc_03_0095",
        "type": "constraint",
        "content": "IF RP건수_유선방송_B0M > 0: RP건수_유선방송 = 0 ELSE RP후경과월_유선방송 >0",
    },
    {
        "columns": ["RP건수_건강_B0M", "RP후경과월_건강"],
        "fname": "cc_03_0096",
        "type": "constraint",
        "content": "IF RP건수_건강_B0M > 0: RP후경과월_건강 = 0 ELSE RP후경과월_건강 >0",
    },
    {
        "columns": ["RP건수_교통_B0M", "RP후경과월_교통"],
        "fname": "cc_03_0097",
        "type": "constraint",
        "content": "IF RP건수_교통_B0M > 0: RP후경과월_교통 = 0 ELSE RP후경과월_교통 >0",
    },
    {
        "columns": ["최초카드론이용경과월", "최종카드론이용경과월"],
        "fname": "cc_03_0098",
        "type": "constraint",
        "content": "최초카드론이용경과월 >= 최종카드론이용경과월",
    },
    {
        "columns": ["이용건수_카드론_R12M", "카드론이용건수_누적"],
        "fname": "cc_03_0099",
        "type": "constraint",
        "content": "이용건수_카드론_R12M <= 카드론이용건수_누적",
    },
    {
        "columns": ["이용개월수_카드론_R12M", "카드론이용월수_누적"],
        "fname": "cc_03_0100",
        "type": "constraint",
        "content": "이용개월수_카드론_R12M <= 카드론이용월수_누적",
    },
    {
        "columns": ["이용금액_카드론_R12M", "카드론이용금액_누적"],
        "fname": "cc_03_0101",
        "type": "constraint",
        "content": "이용금액_카드론_R12M <= 카드론이용금액_누적",
    },
    {
        "columns": ["연속무실적개월수_기본_24M_카드"],
        "fname": "cc_03_0102",
        "type": "constraint",
        "content": "0 <= 연속무실적개월수_기본_24M_카드 <= 24",
    },
    {
        "columns": ["연속유실적개월수_기본_24M_카드", "연속무실적개월수_기본_24M_카드"],
        "fname": "cc_03_0103",
        "type": "constraint",
        "content": "연속유실적개월수_기본_24M_카드 <= 24 - 연속무실적개월수_기본_24M_카드",
    },
    {
        "columns": ["신청건수_ATM_CA_B0", "신청건수_ATM_CA_R6M"],
        "fname": "cc_03_0104",
        "type": "constraint",
        "content": "신청건수_ATM_CA_B0 <= 신청건수_ATM_CA_R6M",
    },
    {
        "columns": ["신청건수_ATM_CA_B0", "이용건수_CA_B0M"],
        "fname": "cc_03_0105",
        "type": "constraint",
        "content": "신청건수_ATM_CA_B0 <= 이용건수_CA_B0M",
    },
    {
        "columns": ["신청건수_ATM_CL_B0", "신청건수_ATM_CL_R6M"],
        "fname": "cc_03_0106",
        "type": "constraint",
        "content": "신청건수_ATM_CL_B0 <= 신청건수_ATM_CL_R6M",
    },
    {
        "columns": ["신청건수_ATM_CL_B0", "이용건수_카드론_B0M"],
        "fname": "cc_03_0107",
        "type": "constraint",
        "content": "신청건수_ATM_CL_B0 <= 이용건수_카드론_B0M",
    },
    {
        "columns": ["이용개월수_페이_온라인_R6M", "이용개월수_온라인_R6M"],
        "fname": "cc_03_0108",
        "type": "constraint",
        "content": "이용개월수_페이_온라인_R6M <= 이용개월수_온라인_R6M",
    },
    {
        "columns": ["이용개월수_페이_오프라인_R6M", "이용개월수_오프라인_R6M"],
        "fname": "cc_03_0109",
        "type": "constraint",
        "content": "이용개월수_페이_오프라인_R6M <= 이용개월수_오프라인_R6M",
    },
    {
        "columns": ["이용금액_온라인_B0M", "이용금액_온라인_R3M", "이용금액_온라인_R6M"],
        "fname": "cc_03_0110",
        "type": "constraint",
        "content": "이용금액_온라인_B0M <= 이용금액_온라인_R3M <= 이용금액_온라인_R6M",
    },
    {
        "columns": ["이용금액_오프라인_B0M", "이용금액_오프라인_R3M", "이용금액_오프라인_R6M"],
        "fname": "cc_03_0111",
        "type": "constraint",
        "content": "이용금액_오프라인_B0M <= 이용금액_오프라인_R3M <= 이용금액_오프라인_R6M",
    },
    {
        "columns": ["이용건수_온라인_B0M", "이용건수_온라인_R3M", "이용건수_온라인_R6M"],
        "fname": "cc_03_0112",
        "type": "constraint",
        "content": "이용건수_온라인_B0M <= 이용건수_온라인_R3M <= 이용건수_온라인_R6M",
    },
    {
        "columns": ["이용건수_오프라인_B0M", "이용건수_오프라인_R3M", "이용건수_오프라인_R6M"],
        "fname": "cc_03_0113",
        "type": "constraint",
        "content": "이용건수_오프라인_B0M <= 이용건수_오프라인_R3M <= 이용건수_오프라인_R6M",
    },
    {
        "columns": ["이용금액_페이_온라인_B0M", "이용금액_페이_온라인_R3M", "이용금액_페이_온라인_R6M"],
        "fname": "cc_03_0114",
        "type": "constraint",
        "content": "이용금액_페이_온라인_B0M <= 이용금액_페이_온라인_R3M <= 이용금액_페이_온라인_R6M",
    },
    {
        "columns": ["이용금액_페이_오프라인_B0M", "이용금액_페이_오프라인_R3M", "이용금액_페이_오프라인_R6M"],
        "fname": "cc_03_0115",
        "type": "constraint",
        "content": "이용금액_페이_오프라인_B0M <= 이용금액_페이_오프라인_R3M <= 이용금액_페이_오프라인_R6M",
    },
    {
        "columns": ["이용건수_페이_온라인_B0M", "이용건수_페이_온라인_R3M", "이용건수_페이_온라인_R6M"],
        "fname": "cc_03_0116",
        "type": "constraint",
        "content": "이용건수_페이_온라인_B0M <= 이용건수_페이_온라인_R3M <= 이용건수_페이_온라인_R6M",
    },
    {
        "columns": ["이용건수_페이_오프라인_B0M", "이용건수_페이_오프라인_R3M", "이용건수_페이_오프라인_R6M"],
        "fname": "cc_03_0117",
        "type": "constraint",
        "content": "이용건수_페이_오프라인_B0M <= 이용건수_페이_오프라인_R3M <= 이용건수_페이_오프라인_R6M",
    },
    {
        "columns": [
            "이용개월수_당사페이_R6M",
            "이용개월수_당사기타_R6M",
            "이용개월수_A페이_R6M",
            "이용개월수_B페이_R6M",
            "이용개월수_C페이_R6M",
            "이용개월수_D페이_R6M",
            "이용개월수_간편결제_R6M",
        ],
        "fname": "cc_03_0118",
        "type": "constraint",
        "content": "MAX(이용개월수_당사페이_R6M, 이용개월수_당사기타_R6M, 이용개월수_A페이_R6M, 이용개월수_B페이_R6M, 이용개월수_C페이_R6M, 이용개월수_D페이_R6M) <= 이용개월수_간편결제_R6M",
    },
    {
        "columns": ["이용개월수_페이_온라인_R6M", "이용개월수_페이_오프라인_R6M", "이용개월수_간편결제_R6M"],
        "fname": "cc_03_0119",
        "type": "constraint",
        "content": "MAX(이용개월수_페이_온라인_R6M, 이용개월수_페이_오프라인_R6M) <= 이용개월수_간편결제_R6M",
    },
    {
        "columns": ["이용금액_간편결제_B0M", "이용금액_간편결제_R3M", "이용금액_간편결제_R6M"],
        "fname": "cc_03_0120",
        "type": "constraint",
        "content": "이용금액_간편결제_B0M <= 이용금액_간편결제_R3M <= 이용금액_간편결제_R6M",
    },
    {
        "columns": [
            "이용금액_간편결제_R6M",
            "이용금액_당사페이_R6M",
            "이용금액_당사기타_R6M",
            "이용금액_A페이_R6M",
            "이용금액_B페이_R6M",
            "이용금액_C페이_R6M",
            "이용금액_D페이_R6M",
        ],
        "fname": "cc_03_0149",
        "type": "constraint",
        "content": "이용금액_간편결제_R6M >= SUM(이용금액_당사페이_R6M, 당사기타, A페이, B페이, C페이, D페이)",
    },
    {
        "columns": ["이용금액_당사페이_B0M", "이용금액_당사페이_R3M", "이용금액_당사페이_R6M"],
        "fname": "cc_03_0121",
        "type": "constraint",
        "content": "이용금액_당사페이_B0M <= 이용금액_당사페이_R3M <= 이용금액_당사페이_R6M",
    },
    {
        "columns": ["이용금액_당사기타_B0M", "이용금액_당사기타_R3M", "이용금액_당사기타_R6M"],
        "fname": "cc_03_0122",
        "type": "constraint",
        "content": "이용금액_당사기타_B0M <=이용금액_당사기타_R3M <= 이용금액_당사기타_R6M",
    },
    {
        "columns": ["이용금액_A페이_B0M", "이용금액_A페이_R3M", "이용금액_A페이_R6M"],
        "fname": "cc_03_0123",
        "type": "constraint",
        "content": "이용금액_A페이_B0M <=이용금액_A페이_R3M <= 이용금액_A페이_R6M",
    },
    {
        "columns": ["이용금액_B페이_B0M", "이용금액_B페이_R3M", "이용금액_B페이_R6M"],
        "fname": "cc_03_0124",
        "type": "constraint",
        "content": "이용금액_B페이_B0M  <=이용금액_B페이_R3M <= 이용금액_B페이_R6M",
    },
    {
        "columns": ["이용금액_C페이_B0M", "이용금액_C페이_R3M", "이용금액_C페이_R6M"],
        "fname": "cc_03_0125",
        "type": "constraint",
        "content": "이용금액_C페이_B0M  <=이용금액_C페이_R3M <= 이용금액_C페이_R6M",
    },
    {
        "columns": ["이용금액_D페이_B0M", "이용금액_D페이_R3M", "이용금액_D페이_R6M"],
        "fname": "cc_03_0126",
        "type": "constraint",
        "content": "이용금액_D페이_B0M  <=이용금액_D페이_R3M <= 이용금액_D페이_R6M",
    },
    {
        "columns": ["이용건수_간편결제_B0M", "이용건수_간편결제_R3M", "이용건수_간편결제_R6M"],
        "fname": "cc_03_0127",
        "type": "constraint",
        "content": "이용건수_간편결제_B0M <= 이용건수_간편결제_R3M <= 이용건수_간편결제_R6M",
    },
    {
        "columns": [
            "이용건수_간편결제_R6M",
            "이용건수_당사페이_R6M",
            "이용건수_당사기타_R6M",
            "이용건수_A페이_R6M",
            "이용건수_B페이_R6M",
            "이용건수_C페이_R6M",
            "이용건수_D페이_R6M",
        ],
        "fname": "cc_03_0150",
        "type": "constraint",
        "content": "이용건수_간편결제_R6M >= SUM(이용건수_당사페이_R6M, 당사기타, A페이, B페이, C페이, D페이)",
    },
    {
        "columns": ["이용건수_당사페이_B0M", "이용건수_당사페이_R3M", "이용건수_당사페이_R6M"],
        "fname": "cc_03_0128",
        "type": "constraint",
        "content": "이용건수_당사페이_B0M <= 이용건수_당사페이_R3M <= 이용건수_당사페이_R6M",
    },
    {
        "columns": ["이용건수_당사기타_B0M", "이용건수_당사기타_R3M", "이용건수_당사기타_R6M"],
        "fname": "cc_03_0129",
        "type": "constraint",
        "content": "이용건수_당사기타_B0M <=이용건수_당사기타_R3M <= 이용건수_당사기타_R6M",
    },
    {
        "columns": ["이용건수_A페이_B0M", "이용건수_A페이_R3M", "이용건수_A페이_R6M"],
        "fname": "cc_03_0130",
        "type": "constraint",
        "content": "이용건수_A페이_B0M <=이용건수_A페이_R3M <= 이용건수_A페이_R6M",
    },
    {
        "columns": ["이용건수_B페이_B0M", "이용건수_B페이_R3M", "이용건수_B페이_R6M"],
        "fname": "cc_03_0131",
        "type": "constraint",
        "content": "이용건수_B페이_B0M  <=이용건수_B페이_R3M <= 이용건수_B페이_R6M",
    },
    {
        "columns": ["이용건수_C페이_B0M", "이용건수_C페이_R3M", "이용건수_C페이_R6M"],
        "fname": "cc_03_0132",
        "type": "constraint",
        "content": "이용건수_C페이_B0M  <=이용건수_C페이_R3M <= 이용건수_C페이_R6M",
    },
    {
        "columns": ["이용건수_D페이_B0M", "이용건수_D페이_R3M", "이용건수_D페이_R6M"],
        "fname": "cc_03_0133",
        "type": "constraint",
        "content": "이용건수_D페이_B0M  <=이용건수_D페이_R3M <= 이용건수_D페이_R6M",
    },
    {
        "columns": [
            "이용금액_간편결제_R3M",
            "이용금액_당사페이_R3M",
            "이용금액_당사기타_R3M",
            "이용금액_A페이_R3M",
            "이용금액_B페이_R3M",
            "이용금액_C페이_R3M",
            "이용금액_D페이_R3M",
        ],
        "fname": "cc_03_0151",
        "type": "constraint",
        "content": "이용금액_간편결제_R3M >= SUM(이용금액_당사페이_R3M, 당사기타, A페이, B페이, C페이, D페이)",
    },
    {
        "columns": [
            "이용건수_간편결제_R3M",
            "이용건수_당사페이_R3M",
            "이용건수_당사기타_R3M",
            "이용건수_A페이_R3M",
            "이용건수_B페이_R3M",
            "이용건수_C페이_R3M",
            "이용건수_D페이_R3M",
        ],
        "fname": "cc_03_0152",
        "type": "constraint",
        "content": "이용건수_간편결제_R3M >= SUM(이용건수_당사페이_R3M, 당사기타, A페이, B페이, C페이, D페이)",
    },
    {
        "columns": [
            "이용금액_간편결제_B0M",
            "이용금액_당사페이_B0M",
            "이용금액_당사기타_B0M",
            "이용금액_A페이_B0M",
            "이용금액_B페이_B0M",
            "이용금액_C페이_B0M",
            "이용금액_D페이_B0M",
        ],
        "fname": "cc_03_0153",
        "type": "constraint",
        "content": "이용금액_간편결제_B0M >= SUM(이용금액_당사페이_B0M, 당사기타, A페이, B페이, C페이, D페이)",
    },
    {
        "columns": [
            "이용건수_간편결제_B0M",
            "이용건수_당사페이_B0M",
            "이용건수_당사기타_B0M",
            "이용건수_A페이_B0M",
            "이용건수_B페이_B0M",
            "이용건수_C페이_B0M",
            "이용건수_D페이_B0M",
        ],
        "fname": "cc_03_0154",
        "type": "constraint",
        "content": "이용건수_간편결제_B0M >= SUM(이용건수_당사페이_B0M, 당사기타, A페이, B페이, C페이, D페이)",
    },
    {
        "columns": ["이용횟수_선결제_R6M", "이용개월수_선결제_R6M"],
        "fname": "cc_03_0134",
        "type": "constraint",
        "content": "IF 이용횟수_선결제_R6M >0: 이용개월수_선결제_R6M >0",
    },
    {
        "columns": ["이용횟수_선결제_B0M", "이용횟수_선결제_R3M", "이용횟수_선결제_R6M"],
        "fname": "cc_03_0135",
        "type": "constraint",
        "content": "이용횟수_선결제_B0M <= 이용횟수_선결제_R3M <= 이용횟수_선결제_R6M",
    },
    {
        "columns": ["이용금액_선결제_B0M", "이용금액_선결제_R3M", "이용금액_선결제_R6M"],
        "fname": "cc_03_0136",
        "type": "constraint",
        "content": "이용금액_선결제_B0M <= 이용금액_선결제_R3M <= 이용금액_선결제_R6M",
    },
    {
        "columns": ["이용건수_선결제_B0M", "이용건수_선결제_R3M", "이용건수_선결제_R6M"],
        "fname": "cc_03_0137",
        "type": "constraint",
        "content": "이용건수_선결제_B0M <= 이용건수_선결제_R3M <= 이용건수_선결제_R6M",
    },
    {
        "columns": ["이용개월수_전체_R3M", "이용개월수_전체_R6M"],
        "fname": "cc_03_0141",
        "type": "constraint",
        "content": "이용개월수_전체_R3M <= 이용개월수_전체_R6M",
    },
    {
        "columns": ["이용개월수_신용_R6M", "이용개월수_카드론_R6M", "이용개월수_전체_R6M"],
        "fname": "cc_03_0142",
        "type": "constraint",
        "content": "MAX(이용개월수_신용_R6M, 이용개월수_카드론_R6M) <= 이용개월수_전체_R6M",
    },
    {
        "columns": ["이용개월수_신용_R3M", "이용개월수_카드론_R3M", "이용개월수_전체_R3M"],
        "fname": "cc_03_0143",
        "type": "constraint",
        "content": "MAX(이용개월수_신용_R3M, 이용개월수_카드론_R3M) <= 이용개월수_전체_R3M",
    },
    {
        "columns": ["이용개월수_결제일_R3M", "이용개월수_결제일_R6M"],
        "fname": "cc_03_0144",
        "type": "constraint",
        "content": "이용개월수_결제일_R3M <= 이용개월수_결제일_R6M",
    },
    {
        "columns": ["이용횟수_연체_B0M", "이용횟수_연체_R3M", "이용횟수_연체_R6M"],
        "fname": "cc_03_0145",
        "type": "constraint",
        "content": "이용횟수_연체_B0M <= 이용횟수_연체_R3M <= 이용횟수_연체_R6M",
    },
    {
        "columns": ["이용금액_연체_B0M", "이용금액_연체_R3M", "이용금액_연체_R6M"],
        "fname": "cc_03_0146",
        "type": "constraint",
        "content": "이용금액_연체_B0M <= 이용금액_연체_R3M <= 이용금액_연체_R6M",
    },
    {
        "columns": ["승인거절건수_한도초과_B0M", "승인거절건수_한도초과_R3M"],
        "fname": "cc_03_0158",
        "type": "constraint",
        "content": "승인거절건수_한도초과_B0M <= 승인거절건수_한도초과_R3M",
    },
    {
        "columns": ["승인거절건수_BL_B0M", "승인거절건수_BL_R3M"],
        "fname": "cc_03_0159",
        "type": "constraint",
        "content": "승인거절건수_BL_B0M <= 승인거절건수_BL_R3M",
    },
    {
        "columns": ["승인거절건수_입력오류_B0M", "승인거절건수_입력오류_R3M"],
        "fname": "cc_03_0160",
        "type": "constraint",
        "content": "승인거절건수_입력오류_B0M <= 승인거절건수_입력오류_R3M",
    },
    {
        "columns": ["승인거절건수_기타_B0M", "승인거절건수_기타_R3M"],
        "fname": "cc_03_0161",
        "type": "constraint",
        "content": "승인거절건수_기타_B0M <= 승인거절건수_기타_R3M",
    },

    # 3.승인.매출 테이블 컬럼 Formula
    {
        "columns": ["최종이용일자_신판", "최종이용일자_CA", "최종이용일자_카드론"],
        "output": "최종이용일자_기본",
        "fname": "cf_03_0006",
        "type": "formula",
        "content": "최종이용일자_기본 = MAX(최종이용일자_신판, 최종이용일자_CA, 최종이용일자_카드론)",
    },
    {
        "columns": ["최종이용일자_일시불", "최종이용일자_할부"],
        "output": "최종이용일자_신판",
        "fname": "cf_03_0007",
        "type": "formula",
        "content": "최종이용일자_신판 = MAX(최종이용일자_일시불, 최종이용일자_할부)",
    },
    {
        "columns": ["이용건수_신판_B0M", "이용건수_CA_B0M"],
        "output": "이용건수_신용_B0M",
        "fname": "cf_03_0013",
        "type": "formula",
        "content": "이용건수_신용_B0M = 이용건수_신판_B0M + 이용건수_CA_B0M",
    },
    {
        "columns": ["이용건수_일시불_B0M", "이용건수_할부_B0M"],
        "output": "이용건수_신판_B0M",
        "fname": "cf_03_0014",
        "type": "formula",
        "content": "이용건수_신판_B0M = 이용건수_일시불_B0M + 이용건수_할부_B0M",
    },
    {
        "columns": ["이용건수_할부_유이자_B0M", "이용건수_할부_무이자_B0M", "이용건수_부분무이자_B0M"],
        "output": "이용건수_할부_B0M",
        "fname": "cf_03_0016",
        "type": "formula",
        "content": "이용건수_할부_B0M = 이용건수_할부_유이자_B0M + 이용건수_할부_무이자_B0M + 이용건수_부분무이자_B0M",
    },
    {
        "columns": ["이용금액_신판_B0M", "이용금액_CA_B0M"],
        "output": "이용금액_신용_B0M",
        "fname": "cf_03_0023",
        "type": "formula",
        "content": "이용금액_신용_B0M = 이용금액_신판_B0M + 이용금액_CA_B0M",
    },
    {
        "columns": ["이용금액_일시불_B0M", "이용금액_할부_B0M"],
        "output": "이용금액_신판_B0M",
        "fname": "cf_03_0024",
        "type": "formula",
        "content": "이용금액_신판_B0M = 이용금액_일시불_B0M + 이용금액_할부_B0M",
    },
    {
        "columns": ["이용금액_할부_유이자_B0M", "이용금액_할부_무이자_B0M", "이용금액_부분무이자_B0M"],
        "output": "이용금액_할부_B0M",
        "fname": "cf_03_0026",
        "type": "formula",
        "content": "이용금액_할부_B0M = 이용금액_할부_유이자_B0M + 이용금액_할부_무이자_B0M + 이용금액_부분무이자_B0M",
    },
    {
        "columns": ["이용후경과월_신판", "이용후경과월_CA"],
        "output": "이용후경과월_신용",
        "fname": "cf_03_0033",
        "type": "formula",
        "content": "이용후경과월_신용 = MIN(이용후경과월_신판, 이용후경과월_CA)",
    },
    {
        "columns": [
            "이용후경과월_일시불",
            "이용후경과월_할부",
        ],
        "output": "이용후경과월_신판",
        "fname": "cf_03_0034",
        "type": "formula",
        "content": "이용후경과월_신판 = MIN(이용후경과월_일시불, 이용후경과월_할부)",
    },
    {
        "columns": ["기준년월", "최종이용일자_일시불"],
        "output": "이용후경과월_일시불",
        "fname": "cf_03_0035",
        "type": "formula",
        "content": "이용후경과월_일시불 = MONTHS_BETWEEN(LAST_DAY(기준년월), 최종이용일자_일시불)",
    },
    {
        "columns": ["이용후경과월_할부_유이자", "이용후경과월_할부_무이자", "이용후경과월_부분무이자"],
        "output": "이용후경과월_할부",
        "fname": "cf_03_0036",
        "type": "formula",
        "content": "이용후경과월_할부 = MIN(이용후경과월_할부_유이자, 이용후경과월_할부_무이자, 이용후경과월_부분무이자)",
    },
    {
        "columns": ["기준년월", "최종이용일자_CA"],
        "output": "이용후경과월_CA",
        "fname": "cf_03_0040",
        "type": "formula",
        "content": "이용후경과월_CA = MONTHS_BETWEEN(LAST_DAY(기준년월), 최종이용일자_CA)",
    },
    {
        "columns": ["기준년월", "최종이용일자_체크"],
        "output": "이용후경과월_체크",
        "fname": "cf_03_0041",
        "type": "formula",
        "content": "이용후경과월_체크 = MONTHS_BETWEEN(LAST_DAY(기준년월), 최종이용일자_체크)",
    },
    {
        "columns": ["기준년월", "최종이용일자_카드론"],
        "output": "이용후경과월_카드론",
        "fname": "cf_03_0042",
        "type": "formula",
        "content": "이용후경과월_카드론 = MONTHS_BETWEEN(LAST_DAY(기준년월), 최종이용일자_카드론)",
    },
    {
        "columns": ["이용건수_신판_R12M", "이용건수_CA_R12M"],
        "output": "이용건수_신용_R12M",
        "fname": "cf_03_0043",
        "type": "formula",
        "content": "이용건수_신용_R12M = SUM(이용건수_신판_R12M, 이용건수_CA_R12M)",
    },
    {
        "columns": ["이용건수_일시불_R12M", "이용건수_할부_R12M"],
        "output": "이용건수_신판_R12M",
        "fname": "cf_03_0044",
        "type": "formula",
        "content": "이용건수_신판_R12M = SUM(이용건수_일시불_R12M, 이용건수_할부_R12M)",
    },
    {
        "columns": ["이용건수_할부_유이자_R12M", "이용건수_할부_무이자_R12M", "이용건수_부분무이자_R12M"],
        "output": "이용건수_할부_R12M",
        "fname": "cf_03_0046",
        "type": "formula",
        "content": "이용건수_할부_R12M = SUM(이용건수_할부_유이자_R12M, 이용건수_할부_무이자_R12M, 이용건수_부분무이자_R12M)",
    },
    {
        "columns": [
            "할부건수_유이자_3M_R12M",
            "할부건수_유이자_6M_R12M",
            "할부건수_유이자_12M_R12M",
            "할부건수_유이자_14M_R12M",
        ],
        "output": "이용건수_할부_유이자_R12M",
        "fname": "cf_03_0047",
        "type": "formula",
        "content": "이용건수_할부_유이자_R12M = SUM(할부건수_유이자_3M_R12M, 할부건수_유이자_6M_R12M, 할부건수_유이자_12M_R12M, 할부건수_유이자_14M_R12M)",
    },
    {
        "columns": [
            "할부건수_무이자_3M_R12M",
            "할부건수_무이자_6M_R12M",
            "할부건수_무이자_12M_R12M",
            "할부건수_무이자_14M_R12M",
        ],
        "output": "이용건수_할부_무이자_R12M",
        "fname": "cf_03_0048",
        "type": "formula",
        "content": "이용건수_할부_무이자_R12M = SUM(할부건수_무이자_3M_R12M, 할부건수_무이자_6M_R12M, 할부건수_무이자_12M_R12M, 할부건수_무이자_14M_R12M)",
    },
    {
        "columns": [
            "할부건수_부분_3M_R12M",
            "할부건수_부분_6M_R12M",
            "할부건수_부분_12M_R12M",
            "할부건수_부분_14M_R12M",
        ],
        "output": "이용건수_부분무이자_R12M",
        "fname": "cf_03_0049",
        "type": "formula",
        "content": "이용건수_부분무이자_R12M = SUM(할부건수_부분_3M_R12M, 할부건수_부분_6M_R12M, 할부건수_부분_12M_R12M, 할부건수_부분_14M_R12M)",
    },
    {
        "columns": ["이용금액_신판_R12M", "이용금액_CA_R12M"],
        "output": "이용금액_신용_R12M",
        "fname": "cf_03_0053",
        "type": "formula",
        "content": "이용금액_신용_R12M = SUM(이용금액_신판_R12M, 이용금액_CA_R12M)",
    },
    {
        "columns": ["이용금액_일시불_R12M", "이용금액_할부_R12M"],
        "output": "이용금액_신판_R12M",
        "fname": "cf_03_0054",
        "type": "formula",
        "content": "이용금액_신판_R12M = SUM(이용금액_일시불_R12M, 이용금액_할부_R12M)",
    },
    {
        "columns": ["이용금액_할부_유이자_R12M", "이용금액_할부_무이자_R12M", "이용금액_부분무이자_R12M"],
        "output": "이용금액_할부_R12M",
        "fname": "cf_03_0056",
        "type": "formula",
        "content": "이용금액_할부_R12M = SUM(이용금액_할부_유이자_R12M, 이용금액_할부_무이자_R12M, 이용금액_부분무이자_R12M)",
    },
    {
        "columns": [
            "할부금액_유이자_3M_R12M",
            "할부금액_유이자_6M_R12M",
            "할부금액_유이자_12M_R12M",
            "할부금액_유이자_14M_R12M",
        ],
        "output": "이용금액_할부_유이자_R12M",
        "fname": "cf_03_0057",
        "type": "formula",
        "content": "이용금액_할부_유이자_R12M = SUM(할부금액_유이자_3M_R12M, 할부금액_유이자_6M_R12M, 할부금액_유이자_12M_R12M, 할부금액_유이자_14M_R12M)",
    },
    {
        "columns": [
            "할부금액_무이자_3M_R12M",
            "할부금액_무이자_6M_R12M",
            "할부금액_무이자_12M_R12M",
            "할부금액_무이자_14M_R12M",
        ],
        "output": "이용금액_할부_무이자_R12M",
        "fname": "cf_03_0058",
        "type": "formula",
        "content": "이용금액_할부_무이자_R12M = SUM(할부금액_무이자_3M_R12M, 할부금액_무이자_6M_R12M, 할부금액_무이자_12M_R12M, 할부금액_무이자_14M_R12M)",
    },
    {
        "columns": [
            "할부금액_부분_3M_R12M",
            "할부금액_부분_6M_R12M",
            "할부금액_부분_12M_R12M",
            "할부금액_부분_14M_R12M",
        ],
        "output": "이용금액_부분무이자_R12M",
        "fname": "cf_03_0059",
        "type": "formula",
        "content": "이용금액_부분무이자_R12M = SUM(할부금액_부분_3M_R12M, 할부금액_부분_6M_R12M, 할부금액_부분_12M_R12M, 할부금액_부분_14M_R12M)",
    },
    {
        "columns": ["최대이용금액_할부_유이자_R12M", "최대이용금액_할부_무이자_R12M", "최대이용금액_부분무이자_R12M"],
        "output": "최대이용금액_할부_R12M",
        "fname": "cf_03_0066",
        "type": "formula",
        "content": "최대이용금액_할부_R12M = MAX(최대이용금액_할부_유이자_R12M, 할부_무이자, 부분무이자)",
    },
    {
        "columns": ["이용건수_신판_R6M", "이용건수_CA_R6M"],
        "output": "이용건수_신용_R6M",
        "fname": "cf_03_0083",
        "type": "formula",
        "content": "이용건수_신용_R6M = SUM(이용건수_신판_R6M, 이용건수_CA_R6M)",
    },
    {
        "columns": ["이용건수_일시불_R6M", "이용건수_할부_R6M"],
        "output": "이용건수_신판_R6M",
        "fname": "cf_03_0084",
        "type": "formula",
        "content": "이용건수_신판_R6M = SUM(이용건수_일시불_R6M, 이용건수_할부_R6M)",
    },
    {
        "columns": ["이용건수_할부_유이자_R6M", "이용건수_할부_무이자_R6M", "이용건수_부분무이자_R6M"],
        "output": "이용건수_할부_R6M",
        "fname": "cf_03_0086",
        "type": "formula",
        "content": "이용건수_할부_R6M = SUM(이용건수_할부_유이자_R6M, 이용건수_할부_무이자_R6M, 이용건수_부분무이자_R6M)",
    },
    {
        "columns": ["이용금액_신판_R6M", "이용금액_CA_R6M"],
        "output": "이용금액_신용_R6M",
        "fname": "cf_03_0093",
        "type": "formula",
        "content": "이용금액_신용_R6M = SUM(이용금액_신판_R6M, 이용금액_CA_R6M)",
    },
    {
        "columns": ["이용금액_일시불_R6M", "이용금액_할부_R6M"],
        "output": "이용금액_신판_R6M",
        "fname": "cf_03_0094",
        "type": "formula",
        "content": "이용금액_신판_R6M = SUM(이용금액_일시불_R6M, 이용금액_할부_R6M)",
    },
    {
        "columns": ["이용금액_할부_유이자_R6M", "이용금액_할부_무이자_R6M", "이용금액_부분무이자_R6M"],
        "output": "이용금액_할부_R6M",
        "fname": "cf_03_0096",
        "type": "formula",
        "content": "이용금액_할부_R6M = SUM(이용금액_할부_유이자_R6M, 이용금액_할부_무이자_R6M, 이용금액_부분무이자_R6M)",
    },
    {
        "columns": ["이용건수_신판_R3M", "이용건수_CA_R3M"],
        "output": "이용건수_신용_R3M",
        "fname": "cf_03_0113",
        "type": "formula",
        "content": "이용건수_신용_R3M = SUM(이용건수_신판_R3M, 이용건수_CA_R3M)",
    },
    {
        "columns": ["이용건수_일시불_R3M", "이용건수_할부_R3M"],
        "output": "이용건수_신판_R3M",
        "fname": "cf_03_0114",
        "type": "formula",
        "content": "이용건수_신판_R3M = SUM(이용건수_일시불_R3M, 이용건수_할부_R3M)",
    },
    {
        "columns": ["이용건수_할부_유이자_R3M", "이용건수_할부_무이자_R3M", "이용건수_부분무이자_R3M"],
        "output": "이용건수_할부_R3M",
        "fname": "cf_03_0116",
        "type": "formula",
        "content": "이용건수_할부_R3M = SUM(이용건수_할부_유이자_R3M, 이용건수_할부_무이자_R3M, 이용건수_부분무이자_R3M)",
    },
    {
        "columns": ["이용금액_신판_R3M", "이용금액_CA_R3M"],
        "output": "이용금액_신용_R3M",
        "fname": "cf_03_0123",
        "type": "formula",
        "content": "이용금액_신용_R3M = SUM(이용금액_신판_R3M, 이용금액_CA_R3M)",
    },
    {
        "columns": ["이용금액_일시불_R3M", "이용금액_할부_R3M"],
        "output": "이용금액_신판_R3M",
        "fname": "cf_03_0124",
        "type": "formula",
        "content": "이용금액_신판_R3M = SUM(이용금액_일시불_R3M, 이용금액_할부_R3M)",
    },
    {
        "columns": ["이용금액_할부_유이자_R3M", "이용금액_할부_무이자_R3M", "이용금액_부분무이자_R3M"],
        "output": "이용금액_할부_R3M",
        "fname": "cf_03_0126",
        "type": "formula",
        "content": "이용금액_할부_R3M = SUM(이용금액_할부_유이자_R3M, 이용금액_할부_무이자_R3M, 이용금액_부분무이자_R3M)",
    },
    {
        "columns": [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ],
        "output": "이용금액_교통",
        "fname": "cf_03_0160",
        "type": "formula",
        "content": "이용금액_교통 = SUM(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)",
    },
    {
        "columns": [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ],
        "output": "이용금액_납부",
        "fname": "cf_03_0162",
        "type": "formula",
        "content": "이용금액_납부 = SUM(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)",
    },
    {
        "columns": [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ],
        "output": "이용금액_여유생활",
        "fname": "cf_03_0164",
        "type": "formula",
        "content": "이용금액_여유생활 = SUM(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)",
    },
    {
        "columns": ["이용금액_쇼핑"],
        "output": "쇼핑_전체_이용금액",
        "fname": "cf_03_0176",
        "type": "formula",
        "content": "쇼핑_전체_이용금액 = 이용금액_쇼핑",
    },
    {
        "columns": ["이용금액_교통"],
        "output": "교통_전체이용금액",
        "fname": "cf_03_0183",
        "type": "formula",
        "content": "교통_전체이용금액 = 이용금액_교통",
    },
    {
        "columns": ["이용금액_여유생활"],
        "output": "여유_전체이용금액",
        "fname": "cf_03_0192",
        "type": "formula",
        "content": "여유_전체이용금액 = 이용금액_여유생활",
    },
    {
        "columns": ["기준년월"],
        "output": "납부_렌탈료이용금액",
        "fname": "cf_03_0195",
        "type": "formula",
        "content": "납부_렌탈료이용금액 = 0",
    },
    {
        "columns": ["이용금액_납부"],
        "output": "납부_전체이용금액",
        "fname": "cf_03_0201",
        "type": "formula",
        "content": "납부_전체이용금액 = 이용금액_납부",
    },
    {
        "columns": [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ],
        "output": "_1순위업종",
        "fname": "cf_03_0202",
        "type": "formula",
        "content": "_1순위업종 = ARGMAX(이용금액_쇼핑, 요식, 교통, 의료, 납부, 교육, 여유생활, 사교활동, 일상생활, 해외)",
    },
    {
        "columns": [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ],
        "output": "_1순위업종_이용금액",
        "fname": "cf_03_0203",
        "type": "formula",
        "content": "_1순위업종_이용금액 = MAX(이용금액_쇼핑, 요식, 교통, 의료, 납부, 교육, 여유생활, 사교활동, 일상생활, 해외)",
    },
    {
        "columns": [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ],
        "output": "_3순위업종",
        "fname": "cf_03_0204",
        "type": "formula",
        "content": "_3순위업종 = ARG3rd(이용금액_쇼핑, 요식, 교통, 의료, 납부, 교육, 여유생활, 사교활동, 일상생활, 해외)",
    },
    {
        "columns": [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ],
        "output": "_3순위업종_이용금액",
        "fname": "cf_03_0205",
        "type": "formula",
        "content": "_3순위업종_이용금액 = 3rd(이용금액_쇼핑, 요식, 교통, 의료, 납부, 교육, 여유생활, 사교활동, 일상생활, 해외)",
    },
    {
        "columns": [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ],
        "output": "_1순위쇼핑업종",
        "fname": "cf_03_0206",
        "type": "formula",
        "content": "_1순위쇼핑업종 = ARGMAX(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)",
    },
    {
        "columns": [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ],
        "output": "_1순위쇼핑업종_이용금액",
        "fname": "cf_03_0207",
        "type": "formula",
        "content": "_1순위쇼핑업종_이용금액 = MAX(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)",
    },
    {
        "columns": [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ],
        "output": "_3순위쇼핑업종",
        "fname": "cf_03_0208",
        "type": "formula",
        "content": "_3순위쇼핑업종 = ARG3rd(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)",
    },
    {
        "columns": [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ],
        "output": "_3순위쇼핑업종_이용금액",
        "fname": "cf_03_0209",
        "type": "formula",
        "content": "_3순위쇼핑업종_이용금액 = 3rd(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)",
    },
    {
        "columns": [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ],
        "output": "_1순위교통업종",
        "fname": "cf_03_0210",
        "type": "formula",
        "content": "_1순위교통업종 = ARGMAX(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)",
    },
    {
        "columns": [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ],
        "output": "_1순위교통업종_이용금액",
        "fname": "cf_03_0211",
        "type": "formula",
        "content": "_1순위교통업종_이용금액 = MAX(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)",
    },
    {
        "columns": [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ],
        "output": "_3순위교통업종",
        "fname": "cf_03_0212",
        "type": "formula",
        "content": "_3순위교통업종 = ARG3rd(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)",
    },
    {
        "columns": [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ],
        "output": "_3순위교통업종_이용금액",
        "fname": "cf_03_0213",
        "type": "formula",
        "content": "_3순위교통업종_이용금액 = 3rd(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)",
    },
    {
        "columns": [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ],
        "output": "_1순위여유업종",
        "fname": "cf_03_0214",
        "type": "formula",
        "content": "_1순위여유업종 = ARGMAX(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)",
    },
    {
        "columns": [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ],
        "output": "_1순위여유업종_이용금액",
        "fname": "cf_03_0215",
        "type": "formula",
        "content": "_1순위여유업종_이용금액 = MAX(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)",
    },
    {
        "columns": [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ],
        "output": "_3순위여유업종",
        "fname": "cf_03_0216",
        "type": "formula",
        "content": "_3순위여유업종 = ARG3rd(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)",
    },
    {
        "columns": [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ],
        "output": "_3순위여유업종_이용금액",
        "fname": "cf_03_0217",
        "type": "formula",
        "content": "_3순위여유업종_이용금액 = 3rd(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)",
    },
    {
        "columns": [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ],
        "output": "_1순위납부업종",
        "fname": "cf_03_0218",
        "type": "formula",
        "content": "_1순위납부업종 = ARGMAX(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)",
    },
    {
        "columns": [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ],
        "output": "_1순위납부업종_이용금액",
        "fname": "cf_03_0219",
        "type": "formula",
        "content": "_1순위납부업종_이용금액 = MAX(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)",
    },
    {
        "columns": [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ],
        "output": "_3순위납부업종",
        "fname": "cf_03_0220",
        "type": "formula",
        "content": "_3순위납부업종 = ARG3rd(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)",
    },
    {
        "columns": [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ],
        "output": "_3순위납부업종_이용금액",
        "fname": "cf_03_0221",
        "type": "formula",
        "content": "_3순위납부업종_이용금액 = 3rd(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)",
    },
    {
        "columns": ["할부건수_유이자_3M_R12M", "할부건수_무이자_3M_R12M", "할부건수_부분_3M_R12M"],
        "output": "할부건수_3M_R12M",
        "fname": "cf_03_0222",
        "type": "formula",
        "content": "할부건수_3M_R12M = SUM(할부건수_유이자_3M_R12M, 할부건수_무이자_3M_R12M, 할부건수_부분_3M_R12M)",
    },
    {
        "columns": ["할부건수_유이자_6M_R12M", "할부건수_무이자_6M_R12M", "할부건수_부분_6M_R12M"],
        "output": "할부건수_6M_R12M",
        "fname": "cf_03_0223",
        "type": "formula",
        "content": "할부건수_6M_R12M = SUM(할부건수_유이자_6M_R12M, 할부건수_무이자_6M_R12M, 할부건수_부분_6M_R12M)",
    },
    {
        "columns": ["할부건수_유이자_12M_R12M", "할부건수_무이자_12M_R12M", "할부건수_부분_12M_R12M"],
        "output": "할부건수_12M_R12M",
        "fname": "cf_03_0224",
        "type": "formula",
        "content": "할부건수_12M_R12M = SUM(할부건수_유이자_12M_R12M, 할부건수_무이자_12M_R12M, 할부건수_부분_12M_R12M)",
    },
    {
        "columns": ["할부건수_유이자_14M_R12M", "할부건수_무이자_14M_R12M", "할부건수_부분_14M_R12M"],
        "output": "할부건수_14M_R12M",
        "fname": "cf_03_0225",
        "type": "formula",
        "content": "할부건수_14M_R12M = SUM(할부건수_유이자_14M_R12M, 할부건수_무이자_14M_R12M, 할부건수_부분_14M_R12M)",
    },
    {
        "columns": ["할부금액_유이자_3M_R12M", "할부금액_무이자_3M_R12M", "할부금액_부분_3M_R12M"],
        "output": "할부금액_3M_R12M",
        "fname": "cf_03_0226",
        "type": "formula",
        "content": "할부금액_3M_R12M = SUM(할부금액_유이자_3M_R12M, 할부금액_무이자_3M_R12M, 할부금액_부분_3M_R12M)",
    },
    {
        "columns": ["할부금액_유이자_6M_R12M", "할부금액_무이자_6M_R12M", "할부금액_부분_6M_R12M"],
        "output": "할부금액_6M_R12M",
        "fname": "cf_03_0227",
        "type": "formula",
        "content": "할부금액_6M_R12M = SUM(할부금액_유이자_6M_R12M, 할부금액_무이자_6M_R12M, 할부금액_부분_6M_R12M)",
    },
    {
        "columns": ["할부금액_유이자_12M_R12M", "할부금액_무이자_12M_R12M", "할부금액_부분_12M_R12M"],
        "output": "할부금액_12M_R12M",
        "fname": "cf_03_0228",
        "type": "formula",
        "content": "할부금액_12M_R12M = SUM(할부금액_유이자_12M_R12M, 할부금액_무이자_12M_R12M, 할부금액_부분_12M_R12M)",
    },
    {
        "columns": ["할부금액_유이자_14M_R12M", "할부금액_무이자_14M_R12M", "할부금액_부분_14M_R12M"],
        "output": "할부금액_14M_R12M",
        "fname": "cf_03_0229",
        "type": "formula",
        "content": "할부금액_14M_R12M = SUM(할부금액_유이자_14M_R12M, 할부금액_무이자_14M_R12M, 할부금액_부분_14M_R12M)",
    },
    {
        "columns": ["기준년월"],
        "output": "할부건수_부분_3M_R12M",
        "fname": "cf_03_0246",
        "type": "formula",
        "content": "할부건수_부분_3M_R12M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "할부금액_부분_3M_R12M",
        "fname": "cf_03_0250",
        "type": "formula",
        "content": "할부금액_부분_3M_R12M = 0",
    },
    {
        "columns": [
            "RP건수_통신_B0M",
            "RP건수_아파트_B0M",
            "RP건수_제휴사서비스직접판매_B0M",
            "RP건수_렌탈_B0M",
            "RP건수_가스_B0M",
            "RP건수_전기_B0M",
            "RP건수_보험_B0M",
            "RP건수_학습비_B0M",
            "RP건수_유선방송_B0M",
            "RP건수_건강_B0M",
            "RP건수_교통_B0M",
        ],
        "output": "RP건수_B0M",
        "fname": "cf_03_0254",
        "type": "formula",
        "content": "RP건수_B0M = SUM(RP건수_통신_B0M, 아파트, 제휴사서비스직접판매, 렌탈, 가스, 전기, 보험, 학습비, 유선방송, 건강, 교통)",
    },
    {
        "columns": [
            "RP후경과월_통신",
            "RP후경과월_아파트",
            "RP후경과월_제휴사서비스직접판매",
            "RP후경과월_렌탈",
            "RP후경과월_가스",
            "RP후경과월_전기",
            "RP후경과월_보험",
            "RP후경과월_학습비",
            "RP후경과월_유선방송",
            "RP후경과월_건강",
            "RP후경과월_교통",
        ],
        "output": "RP후경과월",
        "fname": "cf_03_0268",
        "type": "formula",
        "content": "RP후경과월 = MIN(RP후경과월_통신, 아파트, 제휴사서비스직접판매, 렌탈, 가스, 전기, 보험, 학습비, 유선방송, 건강, 교통)",
    },
    {
        "columns": ["기준년월", "최종이용일자_카드론"],
        "output": "최종카드론이용경과월",
        "fname": "cf_03_0281",
        "type": "formula",
        "content": "최종카드론이용경과월 = MONTHS_BETWEEN(LAST_DAY(기준년월), 최종이용일자_카드론)",
    },
    {
        "columns": ["최종이용일자_카드론"],
        "output": "최종카드론_대출일자",
        "fname": "cf_03_0289",
        "type": "formula",
        "content": "최종카드론_대출일자 == 최종이용일자_카드론",
    },
    {
        "columns": ["기준년월"],
        "output": "이용개월수_당사페이_R6M",
        "fname": "cf_03_0338",
        "type": "formula",
        "content": "이용개월수_당사페이_R6M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "이용금액_당사페이_R6M",
        "fname": "cf_03_0345",
        "type": "formula",
        "content": "이용금액_당사페이_R6M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "이용금액_당사기타_R6M",
        "fname": "cf_03_0346",
        "type": "formula",
        "content": "이용금액_당사기타_R6M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "이용건수_당사페이_R6M",
        "fname": "cf_03_0352",
        "type": "formula",
        "content": "이용건수_당사페이_R6M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "이용건수_당사기타_R6M",
        "fname": "cf_03_0353",
        "type": "formula",
        "content": "이용건수_당사기타_R6M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "이용금액_당사페이_R3M",
        "fname": "cf_03_0359",
        "type": "formula",
        "content": "이용금액_당사페이_R3M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "이용금액_당사기타_R3M",
        "fname": "cf_03_0360",
        "type": "formula",
        "content": "이용금액_당사기타_R3M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "이용건수_당사페이_R3M",
        "fname": "cf_03_0366",
        "type": "formula",
        "content": "이용건수_당사페이_R3M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "이용건수_당사기타_R3M",
        "fname": "cf_03_0367",
        "type": "formula",
        "content": "이용건수_당사기타_R3M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "이용금액_당사페이_B0M",
        "fname": "cf_03_0373",
        "type": "formula",
        "content": "이용금액_당사페이_B0M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "이용금액_당사기타_B0M",
        "fname": "cf_03_0374",
        "type": "formula",
        "content": "이용금액_당사기타_B0M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "이용건수_당사페이_B0M",
        "fname": "cf_03_0380",
        "type": "formula",
        "content": "이용건수_당사페이_B0M = 0",
    },
    {
        "columns": ["기준년월"],
        "output": "이용건수_당사기타_B0M",
        "fname": "cf_03_0381",
        "type": "formula",
        "content": "이용건수_당사기타_B0M = 0",
    },
    {
        "columns": ["정상청구원금_B0M", "선입금원금_B0M", "정상입금원금_B0M"],
        "output": "연체입금원금_B0M",
        "fname": "cf_03_0408",
        "type": "formula",
        "content": "연체입금원금_B0M = 정상청구원금_B0M - (선입금원금_B0M + 정상입금원금_B0M)",
    },
    {
        "columns": ["정상청구원금_B2M", "선입금원금_B2M", "정상입금원금_B2M"],
        "output": "연체입금원금_B2M",
        "fname": "cf_03_0409",
        "type": "formula",
        "content": "연체입금원금_B2M = 정상청구원금_B2M - (선입금원금_B2M + 정상입금원금_B2M)",
    },
    {
        "columns": ["정상청구원금_B5M", "선입금원금_B5M", "정상입금원금_B5M"],
        "output": "연체입금원금_B5M",
        "fname": "cf_03_0410",
        "type": "formula",
        "content": "연체입금원금_B5M = 정상청구원금_B5M - (선입금원금_B5M + 정상입금원금_B5M)",
    },
    {
        "columns": [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ],
        "output": "_2순위업종",
        "fname": "cf_03_0425",
        "type": "formula",
        "content": "_2순위업종 = ARG2nd(이용금액_쇼핑, 요식, 교통, 의료, 납부, 교육, 여유생활, 사교활동, 일상생활, 해외)",
    },
    {
        "columns": [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ],
        "output": "_2순위업종_이용금액",
        "fname": "cf_03_0426",
        "type": "formula",
        "content": "_2순위업종_이용금액 = 2nd(이용금액_쇼핑, 요식, 교통, 의료, 납부, 교육, 여유생활, 사교활동, 일상생활, 해외)",
    },
    {
        "columns": [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ],
        "output": "_2순위쇼핑업종",
        "fname": "cf_03_0427",
        "type": "formula",
        "content": "_2순위쇼핑업종 = ARG2nd(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)",
    },
    {
        "columns": [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ],
        "output": "_2순위쇼핑업종_이용금액",
        "fname": "cf_03_0428",
        "type": "formula",
        "content": "_2순위쇼핑업종_이용금액 = 2nd(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)",
    },
    {
        "columns": [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ],
        "output": "_2순위교통업종",
        "fname": "cf_03_0429",
        "type": "formula",
        "content": "_2순위교통업종 = ARG2nd(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)",
    },
    {
        "columns": [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ],
        "output": "_2순위교통업종_이용금액",
        "fname": "cf_03_0430",
        "type": "formula",
        "content": "_2순위교통업종_이용금액 = 2nd(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)",
    },
    {
        "columns": [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ],
        "output": "_2순위여유업종",
        "fname": "cf_03_0431",
        "type": "formula",
        "content": "_2순위여유업종 = ARG2nd(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)",
    },
    {
        "columns": [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ],
        "output": "_2순위여유업종_이용금액",
        "fname": "cf_03_0432",
        "type": "formula",
        "content": "_2순위여유업종_이용금액 = 2nd(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)",
    },
    {
        "columns": [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ],
        "output": "_2순위납부업종",
        "fname": "cf_03_0433",
        "type": "formula",
        "content": "_2순위납부업종 = ARG2nd(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)",
    },
    {
        "columns": [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ],
        "output": "_2순위납부업종_이용금액",
        "fname": "cf_03_0434",
        "type": "formula",
        "content": "_2순위납부업종_이용금액 = 2nd(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)",
    },
    {
        "columns": ["승인거절건수_한도초과_B0M", "승인거절건수_BL_B0M", "승인거절건수_입력오류_B0M", "승인거절건수_기타_B0M"],
        "output": "승인거절건수_B0M",
        "fname": "cf_03_0462",
        "type": "formula",
        "content": "승인거절건수_B0M = SUM(승인거절건수_한도초과_B0M, BL_B0M, 입력오류_B0M, 기타_B0M)",
    },
    {
        "columns": ["승인거절건수_한도초과_R3M", "승인거절건수_BL_R3M", "승인거절건수_입력오류_R3M", "승인거절건수_기타_R3M"],
        "output": "승인거절건수_R3M",
        "fname": "cf_03_0467",
        "type": "formula",
        "content": "승인거절건수_R3M = SUM(승인거절건수_한도초과_R3M, BL_R3M, 입력오류_R3M, 기타_R3M)",
    },
]

# --------- constraint/formula 함수 정의 ---------
# cc: check constraint
# cf: check formula


@constraint_udf
def cc_01_0001(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF (회원여부_이용가능_CA == 1) OR (회원여부_이용가능_카드론 == 1): 회원여부_이용가능 = 1
    """
    dd = df[["회원여부_이용가능", "회원여부_이용가능_CA", "회원여부_이용가능_카드론"]]  # pd.Series
    ret = dd.apply(lambda x: x[0] == 1 if (x[1] == 1 or x[2] == 1) else True, axis=1)
    return ret


@constraint_udf
def cc_01_0002(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF 대표BL코드_ACCOUNT IS NOT NULL: BL여부 == 1
    """
    dd = df[["대표BL코드_ACCOUNT", "BL여부"]]
    ret = dd.apply(lambda x: x[1] == "0" if x[0] == "_" else x[1] == "1", axis=1)
    return ret


@constraint_udf
def cc_01_0003(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF 탈회횟수_누적 > 0: 최종탈회후경과월 IS NOT NULL
    """
    dd = df[["탈회횟수_누적", "최종탈회후경과월"]]
    ret = dd.apply(lambda x: not pd.isna(x[1]) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_01_0004(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        탈회횟수_발급6개월이내 <= 탈회횟수_발급1년이내 <= 탈회횟수_누적
    """
    r6m, r12m, tot = df["탈회횟수_발급6개월이내"], df["탈회횟수_발급1년이내"], df["탈회횟수_누적"]
    return (r6m <= r12m) * (r12m <= tot)


@constraint_udf
def cc_01_0005(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF 소지여부_신용 == 1: 유효카드수_신용 > 0
    """
    dd = df[["소지여부_신용", "유효카드수_신용"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] == 1 else True, axis=1)
    return ret


@constraint_udf
def cc_01_0006(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용카드수_신용_가족 <= 이용가능카드수_신용_가족 <= 유효카드수_신용_가족 <= 유효카드수_신용
    """
    c1, c2, c3, c4 = (
        df["이용카드수_신용_가족"],
        df["이용가능카드수_신용_가족"],
        df["유효카드수_신용_가족"],
        df["유효카드수_신용"],
    )
    return (c1 <= c2) * (c2 <= c3) * (c3 <= c4)


@constraint_udf
def cc_01_0007(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용카드수_체크_가족 <= 이용가능카드수_체크_가족 <= 유효카드수_체크_가족 <= 유효카드수_체크
    """
    c1, c2, c3, c4 = (
        df["이용카드수_체크_가족"],
        df["이용가능카드수_체크_가족"],
        df["유효카드수_체크_가족"],
        df["유효카드수_체크"],
    )
    return (c1 <= c2) * (c2 <= c3) * (c3 <= c4)


@constraint_udf
def cc_01_0008(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용카드수_신용체크 <= 이용가능카드수_신용체크 <= 유효카드수_신용체크
    """
    c1, c2, c3 = df["이용카드수_신용체크"], df["이용가능카드수_신용체크"], df["유효카드수_신용체크"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_01_0009(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용카드수_신용 <= 이용가능카드수_신용 <= 유효카드수_신용
    """
    c1, c2, c3 = df["이용카드수_신용"], df["이용가능카드수_신용"], df["유효카드수_신용"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_01_0010(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용카드수_체크 <= 이용가능카드수_체크 <= 유효카드수_체크
    """
    c1, c2, c3 = df["이용카드수_체크"], df["이용가능카드수_체크"], df["유효카드수_체크"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_01_0011(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용금액_R3M_신용체크 >= (_1순위카드이용금액 + _2순위카드이용금액)
    """
    c1, c2, c3 = df["이용금액_R3M_신용체크"], df["_1순위카드이용금액"], df["_2순위카드이용금액"]
    return c1 >= (c2 + c3)


@constraint_udf
def cc_01_0012(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용금액_R3M_신용_가족 <= 이용금액_R3M_신용
    """
    c1, c2 = df["이용금액_R3M_신용_가족"], df["이용금액_R3M_신용"]
    return c1 <= c2


@constraint_udf
def cc_01_0013(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용금액_R3M_체크_가족 <= 이용금액_R3M_체크
    """
    c1, c2 = df["이용금액_R3M_체크_가족"], df["이용금액_R3M_체크"]
    return c1 <= c2


@constraint_udf
def cc_01_0014(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        _2순위카드이용금액 <= _1순위카드이용금액 <= 이용금액_R3M_신용체크
    """
    c1, c2, c3 = df["_2순위카드이용금액"], df["_1순위카드이용금액"], df["이용금액_R3M_신용체크"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_01_0015(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF _1순위카드이용금액 > 0: _1순위카드이용건수 > 0
    """
    dd = df[["_1순위카드이용금액", "_1순위카드이용건수"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_01_0016(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF _2순위카드이용금액 > 0: _2순위카드이용건수 > 0
    """
    dd = df[["_2순위카드이용금액", "_2순위카드이용건수"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_01_0017(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        최종카드발급일자 <= LAST_DAY(기준년월)
    """
    dd = df[["기준년월", "최종카드발급일자"]]
    ret = dd.apply(
        lambda x: datetime(year=int(x[0][:4]), month=int(x[0][4:6]), day=1)
        + relativedelta(months=1, days=-1)
        >= datetime.strptime(x[1], "%Y%m%d")
        if not pd.isna(x[1])
        else True,
        axis=1,
    )
    return ret


@constraint_udf
def cc_01_0018(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF 유효카드수_신용체크 == 0: 보유여부_해외겸용_본인 == '0'
    """
    dd = df[["유효카드수_신용체크", "보유여부_해외겸용_본인"]]
    ret = dd.apply(lambda x: x[1] == "0" if x[0] == 0 else True, axis=1)
    return ret


@constraint_udf
def cc_01_0019(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF 보유여부_해외겸용_본인 == '0': 이용가능여부_해외겸용_본인 == '0'
    """
    dd = df[["보유여부_해외겸용_본인", "이용가능여부_해외겸용_본인"]]
    ret = dd.apply(lambda x: x[1] == "0" if x[0] == "0" else True, axis=1)
    return ret


@constraint_udf
def cc_01_0020(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF 보유여부_해외겸용_본인 == '0': 보유여부_해외겸용_신용_본인 == '0'
    """
    dd = df[["보유여부_해외겸용_본인", "보유여부_해외겸용_신용_본인"]]
    ret = dd.apply(lambda x: x[1] == "0" if x[0] == "0" else True, axis=1)
    return ret


@constraint_udf
def cc_01_0021(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF 보유여부_해외겸용_신용_본인 == '0': 이용가능여부_해외겸용_신용_본인 == '0'
    """
    dd = df[["보유여부_해외겸용_신용_본인", "이용가능여부_해외겸용_신용_본인"]]
    ret = dd.apply(lambda x: x[1] == "0" if x[0] == "0" else True, axis=1)
    return ret


@constraint_udf
def cc_01_0022(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        연회비발생카드수_B0M <= 유효카드수_신용
    """
    c1, c2 = df["연회비발생카드수_B0M"], df["유효카드수_신용"]
    return c1 <= c2


@constraint_udf
def cc_01_0023(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        연회비할인카드수_B0M <= 유효카드수_신용
    """
    c1, c2 = df["연회비할인카드수_B0M"], df["유효카드수_신용"]
    return c1 <= c2


@constraint_udf
def cc_01_0024(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        상품관련면제카드수_B0M <= 연회비할인카드수_B0M
    """
    c1, c2 = df["상품관련면제카드수_B0M"], df["연회비할인카드수_B0M"]
    return c1 <= c2


@constraint_udf
def cc_01_0025(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        임직원면제카드수_B0M <= 연회비할인카드수_B0M
    """
    c1, c2 = df["임직원면제카드수_B0M"], df["연회비할인카드수_B0M"]
    return c1 <= c2


@constraint_udf
def cc_01_0026(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        우수회원면제카드수_B0M <= 연회비할인카드수_B0M
    """
    c1, c2 = df["우수회원면제카드수_B0M"], df["연회비할인카드수_B0M"]
    return c1 <= c2


@constraint_udf
def cc_01_0027(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        기타면제카드수_B0M <= 연회비할인카드수_B0M
    """
    c1, c2 = df["기타면제카드수_B0M"], df["연회비할인카드수_B0M"]
    return c1 <= c2


@constraint_udf
def cc_01_0028(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        소지카드수_유효_신용 <= 유효카드수_신용
    """
    c1, c2 = df["소지카드수_유효_신용"], df["유효카드수_신용"]
    return c1 <= c2


@constraint_udf
def cc_01_0029(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        소지카드수_이용가능_신용 <= 이용가능카드수_신용
    """
    c1, c2 = df["소지카드수_이용가능_신용"], df["이용가능카드수_신용"]
    return c1 <= c2


@constraint_udf
def cc_01_0030(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF 소지여부_신용=='1' THEN 소지카드수_유효_신용>0 ELSE 소지카드수_유효_신용==0
    """
    dd = df[["소지여부_신용", "소지카드수_유효_신용"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] == "1" else x[1] == 0, axis=1)
    return ret


@constraint_udf
def cc_01_0031(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        소지카드수_이용가능_신용 <= 소지카드수_유효_신용
    """
    c1, c2 = df["소지카드수_이용가능_신용"], df["소지카드수_유효_신용"]
    return c1 <= c2


@constraint_udf
def cf_01_0018(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        입회경과개월수_신용 = MONTHS_BETWEEN(LAST_DAY(기준년월), 입회일자_신용)
    """
    dd = df[["기준년월", "입회일자_신용"]]
    tmp_res = dd.apply(
        lambda x: relativedelta(
            datetime(year=int(x[0][:4]), month=int(x[0][4:6]), day=1)
            + relativedelta(months=1, days=-1),
            datetime.strptime(x[1], "%Y%m%d"),
        )
        if not pd.isna(x[1])
        else 999,
        axis=1,
    )
    res = tmp_res.apply(
        lambda x: x if x == 999 else x.years * 12 + x.months + int(x.days > 0)
    )

    c = df["입회경과개월수_신용"]
    return c == res


@constraint_udf
def cf_01_0023(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        회원여부_연체 = CASE WHEN `이용횟수_연체_B0M` > 0 THEN '1' ELSE '0'
    """
    dd = df[["이용횟수_연체_B0M"]]
    res = dd.apply(lambda x: "1" if x[0] > 0 else "0", axis=1)

    c = df["회원여부_연체"]
    return c == res


@constraint_udf
def cf_01_0039(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        유효카드수_신용체크 = 유효카드수_신용 + 유효카드수_체크
    """
    c1, c2 = df["유효카드수_신용"], df["유효카드수_체크"]
    res = c1 + c2

    c = df["유효카드수_신용체크"]
    return c == res


@constraint_udf
def cf_01_0044(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용가능카드수_신용체크 = 이용가능카드수_신용 + 이용가능카드수_체크
    """
    c1, c2 = df["이용가능카드수_신용"], df["이용가능카드수_체크"]
    res = c1 + c2

    c = df["이용가능카드수_신용체크"]
    return c == res


@constraint_udf
def cf_01_0049(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용카드수_신용체크 = 이용카드수_신용 + 이용카드수_체크
    """
    c1, c2 = df["이용카드수_신용"], df["이용카드수_체크"]
    res = c1 + c2

    c = df["이용카드수_신용체크"]
    return c == res


@constraint_udf
def cf_01_0054(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_R3M_신용체크 = 이용금액_R3M_신용 + 이용금액_R3M_체크
    """
    c1, c2 = df["이용금액_R3M_신용"], df["이용금액_R3M_체크"]
    res = c1 + c2

    c = df["이용금액_R3M_신용체크"]
    return c == res


@constraint_udf
def cf_01_0083(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        기본연회비_B0M = 할인금액_기본연회비_B0M+청구금액_기본연회비_B0M
    """
    c1, c2 = df["할인금액_기본연회비_B0M"], df["청구금액_기본연회비_B0M"]
    res = c1 + c2

    c = df["기본연회비_B0M"]
    return c == res


@constraint_udf
def cf_01_0084(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        제휴연회비_B0M = 할인금액_제휴연회비_B0M+청구금액_제휴연회비_B0M
    """
    c1, c2 = df["할인금액_제휴연회비_B0M"], df["청구금액_제휴연회비_B0M"]
    res = c1 + c2

    c = df["제휴연회비_B0M"]
    return c == res


@constraint_udf
def cf_01_0092(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        기타면제카드수_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["기타면제카드수_B0M"]
    return c == res


@constraint_udf
def cf_01_0133(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        최종카드발급경과월 = MONTHS_BETWEEN(LAST_DAY(기준년월), 최종카드발급일자)
    """
    dd = df[["기준년월", "최종카드발급일자"]]
    tmp_res = dd.apply(
        lambda x: relativedelta(
            datetime(year=int(x[0][:4]), month=int(x[0][4:6]), day=1)
            + relativedelta(months=1, days=-1),
            datetime.strptime(x[1], "%Y%m%d"),
        )
        if (not pd.isna(x[1])) * (x[1] != "10101")
        else 999,
        axis=1,
    )
    res = tmp_res.apply(lambda x: 12 if x == 999 else x.years * 12 + x.months)

    c = df["최종카드발급경과월"]
    return c == res


@constraint_udf
def cc_02_0001(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        CA한도금액 <= 카드이용한도금액*0.4
    """
    c1, c2 = df["CA한도금액"], df["카드이용한도금액"]
    return c1 <= c2 * 0.4


@constraint_udf
def cc_02_0002(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        일시상환론한도금액 <= 5000만원
    """
    c1 = df["일시상환론한도금액"]
    return c1 <= 50000000


@constraint_udf
def cc_02_0003(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        월상환론한도금액 <= 5000만원
    """
    c1 = df["월상환론한도금액"]
    return c1 <= 50000000


@constraint_udf
def cc_02_0004(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        CA이자율_할인전 <= 24%
    """
    c1 = df["CA이자율_할인전"]
    return c1 <= 24


@constraint_udf
def cc_02_0005(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        CL이자율_할인전 <= 24%
    """
    c1 = df["CL이자율_할인전"]
    return c1 <= 24


@constraint_udf
def cc_02_0006(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        RV일시불이자율_할인전 <= 24%
    """
    c1 = df["RV일시불이자율_할인전"]
    return c1 <= 24


@constraint_udf
def cc_02_0007(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        RV현금서비스이자율_할인전 <= 24%
    """
    c1 = df["RV현금서비스이자율_할인전"]
    return c1 <= 24


@constraint_udf
def cc_02_0008(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF RV신청일자 IS NULL: RV약정청구율 == 0
    """
    dd = df[["RV신청일자", "RV약정청구율"]]
    ret = dd.apply(lambda x: x[1] == 0 if pd.isna(x[0]) else True, axis=1)
    return ret


@constraint_udf
def cc_02_0009(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        0 <= 자발한도감액횟수_R12M <= 12
    """
    c1 = df["자발한도감액횟수_R12M"]
    return (c1 >= 0) * (c1 <= 12)


@constraint_udf
def cc_02_0010(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 자발한도감액횟수_R12M >0: (자발한도감액금액_R12M >0) & (자발한도감액후경과월 <12)
    """
    dd = df[["자발한도감액횟수_R12M", "자발한도감액금액_R12M", "자발한도감액후경과월"]]
    ret = dd.apply(lambda x: (x[1] > 0) * (x[2] < 12) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_02_0011(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 연체감액여부_R3M == '1': 강제한도감액횟수_R12M >0
    """
    dd = df[["연체감액여부_R3M", "강제한도감액횟수_R12M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] == "1" else True, axis=1)
    return ret


@constraint_udf
def cc_02_0012(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        0 <= 강제한도감액횟수_R12M <= 12
    """
    c1 = df["강제한도감액횟수_R12M"]
    return (c1 >= 0) * (c1 <= 12)


@constraint_udf
def cc_02_0013(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 강제한도감액횟수_R12M >0: (강제한도감액금액_R12M >0) & (강제한도감액후경과월 <12)
    """
    dd = df[["강제한도감액횟수_R12M", "강제한도감액금액_R12M", "강제한도감액후경과월"]]
    ret = dd.apply(lambda x: (x[1] > 0) * (x[2] < 12) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_02_0014(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        0 <= 한도증액횟수_R12M <= 12
    """
    c1 = df["한도증액횟수_R12M"]
    return (c1 >= 0) * (c1 <= 12)


@constraint_udf
def cc_02_0015(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 한도증액횟수_R12M >0: (한도증액금액_R12M >0) & (한도증액후경과월 <12)
    """
    dd = df[["한도증액횟수_R12M", "한도증액금액_R12M", "한도증액후경과월"]]
    ret = dd.apply(lambda x: (x[1] > 0) * (x[2] < 12) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_02_0016(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        상향가능CA한도금액 <= (카드이용한도금액+상향가능한도금액)*0.4 - CA한도금액
    """
    c1, c2, c3, c4 = df["상향가능CA한도금액"], df["카드이용한도금액"], df["상향가능한도금액"], df["CA한도금액"]
    return (c1 + c4) <= (c2 + c3) * 0.4


@constraint_udf
def cc_02_0017(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        월상환론상향가능한도금액 <= 5000만원-월상환론한도금액
    """
    c1, c2 = df["월상환론상향가능한도금액"], df["월상환론한도금액"]
    return (c1 + c2) <= 50000000


@constraint_udf
def cc_02_0018(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 한도심사요청건수=0: 한도심사요청후경과월 = 3
    """
    dd = df[["한도심사요청건수", "한도심사요청후경과월"]]
    ret = dd.apply(lambda x: x[1] == 3 if x[0] == 0 else True, axis=1)
    return ret


@constraint_udf
def cc_02_0019(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 한도요청거절건수=0: 한도심사거절후경과월 = 3
    """
    dd = df[["한도요청거절건수", "한도심사거절후경과월"]]
    ret = dd.apply(lambda x: x[1] == 3 if x[0] == 0 else True, axis=1)
    return ret


@constraint_udf
def cc_02_0020(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 시장단기연체여부_R3M == '1' THEN 시장단기연체여부_R6M = '1'
    """
    dd = df[["시장단기연체여부_R3M", "시장단기연체여부_R6M"]]
    ret = dd.apply(lambda x: x[1] == '1' if x[0] == '1' else True, axis=1)
    return ret


@constraint_udf
def cc_02_0021(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 시장연체상환여부_R3M == '1' THEN 시장연체상환여부_R6M = '1'
    """
    dd = df[["시장연체상환여부_R3M", "시장연체상환여부_R6M"]]
    ret = dd.apply(lambda x: x[1] == '1' if x[0] == '1' else True, axis=1)
    return ret


@constraint_udf
def cc_02_0022(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       한도요청거절건수 <= 한도심사요청건수
    """
    c1, c2 = df["한도요청거절건수"], df["한도심사요청건수"]
    return c1 <= c2


@constraint_udf
def cf_02_0030(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 이용거절여부_카드론=='1' THEN 카드론동의여부='N' ELSE 카드론동의여부='Y'
    """
    dd = df[["이용거절여부_카드론"]]
    res = dd.apply(lambda x: "N" if x[0] == '1' else "Y", axis=1)

    c = df["카드론동의여부"]
    return c == res


# @constraint_udf
# def cf_02_0038(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         IF RV신청일자 IS NOT NULL THEN rv최초시작일자=RV신청일자 ELSE rv최초시작일자 IS NULL
#     """
#     dd = df[["RV신청일자"]]
#     res = dd.apply(lambda x: x[0] if not pd.isna(x[0]) else 'nan', axis=1)

#     c = df["rv최초시작일자"]
#     return c == res


# @constraint_udf
# def cf_02_0039(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         IF RV신청일자 IS NOT NULL THEN rv등록일자=RV신청일자 ELSE rv등록일자 IS NULL
#     """
#     dd = df[["RV신청일자"]]
#     res = dd.apply(lambda x: x[0] if not pd.isna(x[0]) else 'nan', axis=1)

#     c = df["rv등록일자"]
#     return c == res


@constraint_udf
def cf_02_0040(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        한도심사요청건수 = 한도요청거절건수 + 한도요청승인건수
    """
    c1, c2 = df["한도요청거절건수"], df["한도요청승인건수"]
    res = c1 + c2

    c = df["한도심사요청건수"]
    return c == res


@constraint_udf
def cf_02_0060(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        rv최초시작후경과일 = DATEDIFF(LAST_DAY(기준년월), rv최초시작일자)
    """
    dd = df[["기준년월", "rv최초시작일자"]]
    res = dd.apply(
        lambda x: (datetime(year=int(x[0][:4]), month=int(x[0][4:6]), day=1) + relativedelta(months=1, days=-1) - datetime.strptime(x[1], "%Y%m%d")).days
        if (not pd.isna(x[1])) & (x[1] != '10101')
        else 99999999,
        axis=1,
    )

    c = df["rv최초시작후경과일"]
    return c == res


@constraint_udf
def cc_04_0001(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 청구서발송여부_B0 =='1': 청구서발송여부_R3M ='1'
    """
    dd = df[["청구서발송여부_B0", "청구서발송여부_R3M"]]
    ret = dd.apply(lambda x: x[1] == "1" if x[0] == "1" else True, axis=1)
    return ret


@constraint_udf
def cc_04_0002(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 청구서발송여부_R3M =='1': 청구서발송여부_R6M =='1'
    """
    dd = df[["청구서발송여부_R3M", "청구서발송여부_R6M"]]
    ret = dd.apply(lambda x: x[1] == "1" if x[0] == "1" else True, axis=1)
    return ret


@constraint_udf
def cc_04_0003(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        청구금액_B0 <= 청구금액_R3M <= 청구금액_R6M
    """
    c1, c2, c3 = df["청구금액_B0"], df["청구금액_R3M"], df["청구금액_R6M"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_04_0004(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        포인트_마일리지_건별_B0M <= 포인트_마일리지_건별_R3M
    """
    c1, c2 = df["포인트_마일리지_건별_B0M"], df["포인트_마일리지_건별_R3M"]
    return c1 <= c2


@constraint_udf
def cc_04_0005(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        포인트_포인트_건별_B0M <= 포인트_포인트_건별_R3M
    """
    c1, c2 = df["포인트_포인트_건별_B0M"], df["포인트_포인트_건별_R3M"]
    return c1 <= c2


@constraint_udf
def cc_04_0006(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        포인트_마일리지_월적립_B0M <= 포인트_마일리지_월적립_R3M
    """
    c1, c2 = df["포인트_마일리지_월적립_B0M"], df["포인트_마일리지_월적립_R3M"]
    return c1 <= c2


@constraint_udf
def cc_04_0007(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        포인트_포인트_월적립_B0M <= 포인트_포인트_월적립_R3M"
    """
    c1, c2 = df["포인트_포인트_월적립_B0M"], df["포인트_포인트_월적립_R3M"]
    return c1 <= c2


@constraint_udf
def cc_04_0008(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        포인트_적립포인트_R3M <= 포인트_적립포인트_R12M
    """
    c1, c2 = df["포인트_적립포인트_R3M"], df["포인트_적립포인트_R12M"]
    return c1 <= c2


@constraint_udf
def cc_04_0009(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        포인트_이용포인트_R3M <= 포인트_이용포인트_R12M
    """
    c1, c2 = df["포인트_이용포인트_R3M"], df["포인트_이용포인트_R12M"]
    return c1 <= c2


@constraint_udf
def cc_04_0010(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        마일_적립포인트_R3M <= 마일_적립포인트_R12M
    """
    c1, c2 = df["마일_적립포인트_R3M"], df["마일_적립포인트_R12M"]
    return c1 <= c2


@constraint_udf
def cc_04_0011(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        마일_이용포인트_R3M <= 마일_이용포인트_R12M
    """
    c1, c2 = df["마일_이용포인트_R3M"], df["마일_이용포인트_R12M"]
    return c1 <= c2


@constraint_udf
def cc_04_0012(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        할인건수_B0M <= 할인건수_R3M
    """
    c1, c2 = df["할인건수_B0M"], df["할인건수_R3M"]
    return c1 <= c2


@constraint_udf
def cc_04_0013(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 할인금액_R3M >0: 할인건수_R3M >0
    """
    dd = df[["할인금액_R3M", "할인건수_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_04_0014(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        할인금액_B0M <= 할인금액_R3M
    """
    c1, c2 = df["할인금액_B0M"], df["할인금액_R3M"]
    return c1 <= c2


@constraint_udf
def cc_04_0015(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 할인금액_B0M >0: 할인건수_B0M >0
    """
    dd = df[["할인금액_B0M", "할인건수_B0M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_04_0016(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        할인금액_B0M <= 이용금액_신판_B0M
    """
    c1, c2 = df["할인금액_B0M"], df["이용금액_신판_B0M"]
    return c1 <= c2


@constraint_udf
def cc_04_0017(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        할인금액_청구서_B0M <= 할인금액_청구서_R3M
    """
    c1, c2 = df["할인금액_청구서_B0M"], df["할인금액_청구서_R3M"]
    return c1 <= c2


@constraint_udf
def cc_04_0018(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        혜택수혜금액 <= 혜택수혜금액_R3M
    """
    c1, c2 = df["혜택수혜금액"], df["혜택수혜금액_R3M"]
    return c1 <= c2


@constraint_udf
def cc_04_0019(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        상환개월수_결제일_R3M <= 상환개월수_결제일_R6M <= 상환개월수_결제일_R12M
    """
    c1, c2, c3 = df["상환개월수_결제일_R3M"], df["상환개월수_결제일_R6M"], df["상환개월수_결제일_R12M"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_04_0020(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        선결제건수_R3M <= 선결제건수_R6M <= 선결제건수_R12M
    """
    c1, c2, c3 = df["선결제건수_R3M"], df["선결제건수_R6M"], df["선결제건수_R12M"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_04_0021(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        연체건수_R3M <= 연체건수_R6M <= 연체건수_R12M
    """
    c1, c2, c3 = df["연체건수_R3M"], df["연체건수_R6M"], df["연체건수_R12M"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cf_04_0008(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        대표결제방법코드 = 2
    """
    c1 = df["기준년월"]
    res = pd.Series(['2'] * len(c1))

    c = df["대표결제방법코드"]
    return c == res


@constraint_udf
def cf_04_0011(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 구분코드 IN ('1','3') THEN 청구서수령방법 = 01.우편, ELIF 구분코드 IN ('2') THEN 02.이메일,
        ELIF 구분코드 IN ('L','S') THEN 03.LMS, ELIF 구분코드 IN ('K') THEN 04.카카오,
        ELIF 구분코드 IN ('H') THEN 05.당사멤버십, ELIF 구분코드 IN ('T') THEN 07.기타,
        ELIF 구분코드 IN ('0') THEN 99.미수령
    """
    code_map = {
        "1": "01.우편",
        "3": "01.우편",
        "2": "02.이메일",
        "L": "03.LMS",
        "S": "03.LMS",
        "K": "04.카카오",
        "H": "05.당사멤버십",
        "T": "07.기타",
        "_": "07.기타",
        "0": "99.미수령",
    }
    c1 = df["대표청구서수령지구분코드"]
    res = list(map(lambda x: code_map[x], c1))

    c = df["청구서수령방법"]
    return c == res


@constraint_udf
def cf_04_0012(df: pd.DataFrame) -> Union[pd.Series, List[str]]:
    """
    formula:
        IF 청구금액_B0 > 0 THEN 청구서발송여부_B0 = '1' ELSE '0'
    """
    dd = df[["청구금액_B0"]]
    res = dd.apply(lambda x: "1" if x[0] > 0 else "0", axis=1)

    c = df["청구서발송여부_B0"]
    return c == res


@constraint_udf
def cf_04_0027(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        포인트_적립포인트_R3M = 포인트_포인트_건별_R3M + 포인트_포인트_월적립_R3M
    """
    c1, c2 = df["포인트_포인트_건별_R3M"], df["포인트_포인트_월적립_R3M"]
    res = c1 + c2

    c = df["포인트_적립포인트_R3M"]
    return c == res


@constraint_udf
def cf_04_0032(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        마일_적립포인트_R3M = 포인트_마일리지_건별_R3M + 포인트_마일리지_월적립_R3M
    """
    c1, c2 = df["포인트_마일리지_건별_R3M"], df["포인트_마일리지_월적립_R3M"]
    res = c1 + c2

    c = df["마일_적립포인트_R3M"]
    return c == res


@constraint_udf
def cc_05_0001(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        잔액_B0M <= 카드이용한도금액
    """
    c1, c2 = df["잔액_B0M"], df["카드이용한도금액"]
    return c1 <= c2


@constraint_udf
def cc_05_0002(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        잔액_일시불_B0M + 잔액_할부_B0M + 잔액_리볼빙일시불이월_B0M <= 카드이용한도금액
    """
    c1, c2, c3, c4 = (
        df["잔액_일시불_B0M"],
        df["잔액_할부_B0M"],
        df["잔액_리볼빙일시불이월_B0M"],
        df["카드이용한도금액"],
    )
    return (c1 + c2 + c3) <= c4


@constraint_udf
def cc_05_0003(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        잔액_현금서비스_B0M + 잔액_리볼빙CA이월_B0M <= CA한도금액
    """
    c1, c2, c3 = df["잔액_현금서비스_B0M"], df["잔액_리볼빙CA이월_B0M"], df["CA한도금액"]
    return (c1 + c2) <= c3


@constraint_udf
def cc_05_0004(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        잔액_카드론_B0M <= 월상환론한도금액
    """
    c1, c2 = df["잔액_카드론_B0M"], df["월상환론한도금액"]
    return c1 <= c2


@constraint_udf
def cc_05_0005(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 잔액_카드론_B0M >0: 카드론잔액_최종경과월 IS NOT NULL
    """
    dd = df[["잔액_카드론_B0M", "카드론잔액_최종경과월"]]
    ret = dd.apply(lambda x: not pd.isna(x[1]) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_05_0006(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 연체일자_B0M IS NOT NULL: 연체잔액_B0M >0
    """
    dd = df[["연체일자_B0M", "연체잔액_B0M"]]
    ret = dd.apply(lambda x: x[1] > 0 if not pd.isna(x[0]) else True, axis=1)
    return ret


@constraint_udf
def cc_05_0007(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 회원여부_이용가능_CA == '0': 연체잔액_현금서비스_B0M == 0
    """
    dd = df[["회원여부_이용가능_CA", "연체잔액_현금서비스_B0M"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] == "0" else True, axis=1)
    return ret


@constraint_udf
def cc_05_0008(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 회원여부_이용가능_카드론 == '0': 연체잔액_카드론_B0M == 0
    """
    dd = df[["회원여부_이용가능_카드론", "연체잔액_카드론_B0M"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] == "0" else True, axis=1)
    return ret


@constraint_udf
def cc_05_0009(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        RV_최대잔액_R12M >= RV_평균잔액_R12M
    """
    c1, c2 = df["RV_최대잔액_R12M"], df["RV_평균잔액_R12M"]
    return c1 >= c2


@constraint_udf
def cc_05_0010(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        RV_최대잔액_R6M >= RV_평균잔액_R6M
    """
    c1, c2 = df["RV_최대잔액_R6M"], df["RV_평균잔액_R6M"]
    return c1 >= c2


@constraint_udf
def cc_05_0011(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        RV_최대잔액_R3M >= RV_평균잔액_R3M
    """
    c1, c2 = df["RV_최대잔액_R3M"], df["RV_평균잔액_R3M"]
    return c1 >= c2


@constraint_udf
def cc_05_0016(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        RV잔액이월횟수_R3M <= RV잔액이월횟수_R6M <= RV잔액이월횟수_R12M
    """
    c1, c2, c3 = df["RV잔액이월횟수_R3M"], df["RV잔액이월횟수_R6M"], df["RV잔액이월횟수_R12M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_05_0017(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        잔액_할부_해외_B0M <= 잔액_할부_B0M
    """
    c1, c2 = df["잔액_할부_해외_B0M"], df["잔액_할부_B0M"]
    return c1 <= c2


@constraint_udf
def cc_05_0018(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        연체잔액_일시불_B0M <= 연체잔액_일시불_해외_B0M
    """
    c1, c2 = df["연체잔액_일시불_B0M"], df["연체잔액_일시불_해외_B0M"]
    return c1 <= c2


@constraint_udf
def cc_05_0019(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        연체잔액_일시불_해외_B0M <= 연체잔액_RV일시불_해외_B0M
    """
    c1, c2 = df["연체잔액_일시불_해외_B0M"], df["연체잔액_RV일시불_해외_B0M"]
    return c1 <= c2


@constraint_udf
def cc_05_0020(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        연체잔액_RV일시불_B0M <= 연체잔액_일시불_B0M
    """
    c1, c2 = df["연체잔액_RV일시불_B0M"], df["연체잔액_일시불_B0M"]
    return c1 <= c2


@constraint_udf
def cc_05_0021(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        연체잔액_RV일시불_해외_B0M <= 연체잔액_RV일시불_B0M
    """
    c1, c2 = df["연체잔액_RV일시불_해외_B0M"], df["연체잔액_RV일시불_B0M"]
    return c1 <= c2


@constraint_udf
def cc_05_0022(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        연체잔액_할부_해외_B0M <= 연체잔액_할부_B0M
    """
    c1, c2 = df["연체잔액_할부_해외_B0M"], df["연체잔액_할부_B0M"]
    return c1 <= c2


@constraint_udf
def cc_05_0023(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        연체잔액_CA_해외_B0M <= 연체잔액_CA_B0M
    """
    c1, c2 = df["연체잔액_CA_해외_B0M"], df["연체잔액_CA_B0M"]
    return c1 <= c2


@constraint_udf
def cc_05_0024(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        평잔_일시불_해외_3M <= 평잔_일시불_3M
    """
    c1, c2 = df["평잔_일시불_해외_3M"], df["평잔_일시불_3M"]
    return c1 <= c2


@constraint_udf
def cc_05_0025(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        평잔_RV일시불_해외_3M <= 평잔_RV일시불_3M
    """
    c1, c2 = df["평잔_RV일시불_해외_3M"], df["평잔_RV일시불_3M"]
    return c1 <= c2


@constraint_udf
def cc_05_0026(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        평잔_할부_해외_3M <= 평잔_할부_3M
    """
    c1, c2 = df["평잔_할부_해외_3M"], df["평잔_할부_3M"]
    return c1 <= c2


@constraint_udf
def cc_05_0027(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        평잔_CA_해외_3M <= 평잔_CA_3M
    """
    c1, c2 = df["평잔_CA_해외_3M"], df["평잔_CA_3M"]
    return c1 <= c2


@constraint_udf
def cc_05_0028(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        평잔_일시불_해외_6M <= 평잔_일시불_6M
    """
    c1, c2 = df["평잔_일시불_해외_6M"], df["평잔_일시불_6M"]
    return c1 <= c2


@constraint_udf
def cc_05_0029(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        평잔_RV일시불_해외_6M <= 평잔_RV일시불_6M
    """
    c1, c2 = df["평잔_RV일시불_해외_6M"], df["평잔_RV일시불_6M"]
    return c1 <= c2


@constraint_udf
def cc_05_0030(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        평잔_할부_해외_6M <= 평잔_할부_6M
    """
    c1, c2 = df["평잔_할부_해외_6M"], df["평잔_할부_6M"]
    return c1 <= c2


@constraint_udf
def cc_05_0031(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        평잔_CA_해외_6M <= 평잔_CA_6M
    """
    c1, c2 = df["평잔_CA_해외_6M"], df["평잔_CA_6M"]
    return c1 <= c2


@constraint_udf
def cf_05_0006(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        잔액_B0M = SUM(잔액_일시불_B0M, 할부, 현금서비스, 리볼빙일시불이월, 리볼빙CA이월, 카드론)
    """
    c1, c2, c3 = df["잔액_일시불_B0M"], df["잔액_할부_B0M"], df["잔액_현금서비스_B0M"]
    c4, c5, c6 = df["잔액_리볼빙일시불이월_B0M"], df["잔액_리볼빙CA이월_B0M"], df["잔액_카드론_B0M"]
    res = c1 + c2 + c3 + c4 + c5 + c6

    c = df["잔액_B0M"]
    return c == res


@constraint_udf
def cf_05_0008(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        잔액_할부_B0M = SUM(잔액_할부_유이자_B0M, 잔액_할부_무이자_B0M)
    """
    c1, c2 = df["잔액_할부_유이자_B0M"], df["잔액_할부_무이자_B0M"]
    res = c1 + c2

    c = df["잔액_할부_B0M"]
    return c == res


@constraint_udf
def cf_05_0018(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        연체잔액_B0M = SUM(연체잔액_일시불_B0M, 할부, 현금서비스, 카드론, 대환론)
    """
    c1, c2, c3 = df["연체잔액_일시불_B0M"], df["연체잔액_할부_B0M"], df["연체잔액_현금서비스_B0M"]
    c4, c5 = df["연체잔액_카드론_B0M"], df["연체잔액_대환론_B0M"]
    res = c1 + c2 + c3 + c4 + c5

    c = df["연체잔액_B0M"]
    return c == res


@constraint_udf
def cc_06_0001(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        인입횟수_ARS_B0M <= 인입횟수_ARS_R6M
    """
    c1, c2 = df["인입횟수_ARS_B0M"], df["인입횟수_ARS_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0002(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용메뉴건수_ARS_B0M <= 이용메뉴건수_ARS_R6M
    """
    c1, c2 = df["이용메뉴건수_ARS_B0M"], df["이용메뉴건수_ARS_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0064(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용메뉴건수_ARS_R6M >= 인입횟수_ARS_R6M
    """
    c1, c2 = df["이용메뉴건수_ARS_R6M"], df["인입횟수_ARS_R6M"]
    return c1 >= c2


@constraint_udf
def cc_06_0003(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        인입일수_ARS_B0M <= 인입일수_ARS_R6M
    """
    c1, c2 = df["인입일수_ARS_B0M"], df["인입일수_ARS_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0004(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        인입월수_ARS_R6M <= 인입일수_ARS_R6M <= 인입횟수_ARS_R6M
    """
    c1, c2, c3 = df["인입월수_ARS_R6M"], df["인입일수_ARS_R6M"], df["인입횟수_ARS_R6M"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_06_0005(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 인입횟수_ARS_R6M >0: (0 < 인입월수_ARS_R6M <= 6) & (인입후경과월_ARS < 6)
    """
    dd = df[["인입횟수_ARS_R6M", "인입월수_ARS_R6M", "인입후경과월_ARS"]]
    ret = dd.apply(lambda x: (0 < x[1] <= 6) * (x[2] < 6) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_06_0006(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용메뉴건수_ARS_B0M >= 인입횟수_ARS_B0M
    """
    c1, c2 = df["이용메뉴건수_ARS_B0M"], df["인입횟수_ARS_B0M"]
    return c1 >= c2


@constraint_udf
def cc_06_0007(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        인입일수_ARS_B0M <= 인입횟수_ARS_B0M
    """
    c1, c2 = df["인입일수_ARS_B0M"], df["인입횟수_ARS_B0M"]
    return c1 <= c2


@constraint_udf
def cc_06_0008(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        방문횟수_PC_B0M <= 방문횟수_PC_R6M
    """
    c1, c2 = df["방문횟수_PC_B0M"], df["방문횟수_PC_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0009(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        방문일수_PC_B0M <= 방문일수_PC_R6M
    """
    c1, c2 = df["방문일수_PC_B0M"], df["방문일수_PC_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0010(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        방문월수_PC_R6M <= 방문일수_PC_R6M <= 방문횟수_PC_R6M
    """
    c1, c2, c3 = df["방문월수_PC_R6M"], df["방문일수_PC_R6M"], df["방문횟수_PC_R6M"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_06_0011(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 방문횟수_PC_R6M >0:  (0 < 방문월수_PC_R6M <= 6) & (방문후경과월_PC_R6M < 6)
    """
    dd = df[["방문횟수_PC_R6M", "방문월수_PC_R6M", "방문후경과월_PC_R6M"]]
    ret = dd.apply(lambda x: (0 < x[1] <= 6) * (x[2] < 6) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_06_0012(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        방문횟수_앱_B0M <= 방문횟수_앱_R6M
    """
    c1, c2 = df["방문횟수_앱_B0M"], df["방문횟수_앱_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0013(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        방문일수_앱_B0M <= 방문일수_앱_R6M
    """
    c1, c2 = df["방문일수_앱_B0M"], df["방문일수_앱_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0014(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        방문월수_앱_R6M <= 방문일수_앱_R6M <= 방문횟수_앱_R6M
    """
    c1, c2, c3 = df["방문월수_앱_R6M"], df["방문일수_앱_R6M"], df["방문횟수_앱_R6M"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_06_0015(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 방문횟수_앱_R6M >0:  (0 < 방문월수_앱_R6M <= 6) & (방문후경과월_앱_R6M < 6)
    """
    dd = df[["방문횟수_앱_R6M", "방문월수_앱_R6M", "방문후경과월_앱_R6M"]]
    ret = dd.apply(lambda x: (0 < x[1] <= 6) * (x[2] < 6) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_06_0016(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        방문횟수_모바일웹_B0M <= 방문횟수_모바일웹_R6M
    """
    c1, c2 = df["방문횟수_모바일웹_B0M"], df["방문횟수_모바일웹_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0017(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        방문일수_모바일웹_B0M <= 방문일수_모바일웹_R6M
    """
    c1, c2 = df["방문일수_모바일웹_B0M"], df["방문일수_모바일웹_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0018(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        방문월수_모바일웹_R6M <= 방문일수_모바일웹_R6M <= 방문횟수_모바일웹_R6M
    """
    c1, c2, c3 = df["방문월수_모바일웹_R6M"], df["방문일수_모바일웹_R6M"], df["방문횟수_모바일웹_R6M"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_06_0019(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 방문횟수_모바일웹_R6M >0:  (0 < 방문월수_모바일웹_R6M <= 6) & (방문후경과월_모바일웹_R6M < 6
    """
    dd = df[["방문횟수_모바일웹_R6M", "방문월수_모바일웹_R6M", "방문후경과월_모바일웹_R6M"]]
    ret = dd.apply(lambda x: (0 < x[1] <= 6) * (x[2] < 6) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_06_0020(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        방문일수_PC_B0M <= 방문횟수_PC_B0M
    """
    c1, c2 = df["방문일수_PC_B0M"], df["방문횟수_PC_B0M"]
    return c1 <= c2


@constraint_udf
def cc_06_0021(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        방문일수_앱_B0M <= 방문횟수_앱_B0M
    """
    c1, c2 = df["방문일수_앱_B0M"], df["방문횟수_앱_B0M"]
    return c1 <= c2


@constraint_udf
def cc_06_0022(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        방문일수_모바일웹_B0M <= 방문횟수_모바일웹_B0M
    """
    c1, c2 = df["방문일수_모바일웹_B0M"], df["방문횟수_모바일웹_B0M"]
    return c1 <= c2


@constraint_udf
def cc_06_0023(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        인입횟수_IB_B0M <= 인입횟수_IB_R6M
    """
    c1, c2 = df["인입횟수_IB_B0M"], df["인입횟수_IB_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0024(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        인입일수_IB_B0M <= 인입일수_IB_R6M
    """
    c1, c2 = df["인입일수_IB_B0M"], df["인입일수_IB_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0025(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        인입월수_IB_R6M <= 인입일수_IB_R6M <= 인입횟수_IB_R6M
    """
    c1, c2, c3 = df["인입월수_IB_R6M"], df["인입일수_IB_R6M"], df["인입횟수_IB_R6M"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_06_0026(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 인입횟수_IB_R6M >0:  (0 < 인입월수_IB_R6M <= 6) & (인입후경과월_IB_R6M < 6)
    """
    dd = df[["인입횟수_IB_R6M", "인입월수_IB_R6M", "인입후경과월_IB_R6M"]]
    ret = dd.apply(lambda x: (0 < x[1] <= 6) * (x[2] < 6) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_06_0027(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용메뉴건수_IB_B0M <= 이용메뉴건수_IB_R6M <= 인입횟수_IB_R6M
    """
    c1, c2, c3 = df["이용메뉴건수_IB_B0M"], df["이용메뉴건수_IB_R6M"], df["인입횟수_IB_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0028(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        인입일수_IB_B0M <= 인입횟수_IB_B0M
    """
    c1, c2 = df["인입일수_IB_B0M"], df["인입횟수_IB_B0M"]
    return c1 <= c2


@constraint_udf
def cc_06_0029(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용메뉴건수_IB_B0M <= 인입횟수_IB_B0M
    """
    c1, c2 = df["이용메뉴건수_IB_B0M"], df["인입횟수_IB_B0M"]
    return c1 <= c2


@constraint_udf
def cc_06_0030(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        인입불만횟수_IB_B0M <= 인입불만횟수_IB_R6M
    """
    c1, c2 = df["인입불만횟수_IB_B0M"], df["인입불만횟수_IB_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0031(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        인입불만일수_IB_B0M <= 인입불만일수_IB_R6M
    """
    c1, c2 = df["인입불만일수_IB_B0M"], df["인입불만일수_IB_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0032(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        인입불만월수_IB_R6M <= 인입불만일수_IB_R6M <= 인입불만횟수_IB_R6M
    """
    c1, c2, c3 = df["인입불만월수_IB_R6M"], df["인입불만일수_IB_R6M"], df["인입불만횟수_IB_R6M"]
    return (c1 <= c2) * (c2 <= c3)


@constraint_udf
def cc_06_0033(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 인입불만횟수_IB_R6M >0:  (0 < 인입불만월수_IB_R6M <= 6) & (인입불만후경과월_IB_R6M < 6)
    """
    dd = df[["인입불만횟수_IB_R6M", "인입불만월수_IB_R6M", "인입불만후경과월_IB_R6M"]]
    ret = dd.apply(lambda x: (0 < x[1] <= 6) * (x[2] < 6) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_06_0034(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        인입불만일수_IB_B0M <= 인입불만횟수_IB_B0M
    """
    c1, c2 = df["인입불만일수_IB_B0M"], df["인입불만횟수_IB_B0M"]
    return c1 <= c2


@constraint_udf
def cc_06_0035(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_사용승인내역_B0M <= IB문의건수_사용승인내역_R6M
    """
    c1, c2 = df["IB문의건수_사용승인내역_B0M"], df["IB문의건수_사용승인내역_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0036(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_한도_B0M <= IB문의건수_한도_R6M
    """
    c1, c2 = df["IB문의건수_한도_B0M"], df["IB문의건수_한도_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0037(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_선결제_B0M <= IB문의건수_선결제_R6M
    """
    c1, c2 = df["IB문의건수_선결제_B0M"], df["IB문의건수_선결제_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0038(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_결제_B0M <= IB문의건수_결제_R6M
    """
    c1, c2 = df["IB문의건수_결제_B0M"], df["IB문의건수_결제_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0039(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_할부_B0M <= IB문의건수_할부_R6M
    """
    c1, c2 = df["IB문의건수_할부_B0M"], df["IB문의건수_할부_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0040(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_정보변경_B0M <= IB문의건수_정보변경_R6M
    """
    c1, c2 = df["IB문의건수_정보변경_B0M"], df["IB문의건수_정보변경_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0041(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_결제일변경_B0M <= IB문의건수_결제일변경_R6M
    """
    c1, c2 = df["IB문의건수_결제일변경_B0M"], df["IB문의건수_결제일변경_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0042(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_명세서_B0M <= IB문의건수_명세서_R6M
    """
    c1, c2 = df["IB문의건수_명세서_B0M"], df["IB문의건수_명세서_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0043(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_비밀번호_B0M <= IB문의건수_비밀번호_R6M
    """
    c1, c2 = df["IB문의건수_비밀번호_B0M"], df["IB문의건수_비밀번호_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0044(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_SMS_B0M <= IB문의건수_SMS_R6M
    """
    c1, c2 = df["IB문의건수_SMS_B0M"], df["IB문의건수_SMS_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0045(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_APP_B0M <= IB문의건수_APP_R6M
    """
    c1, c2 = df["IB문의건수_APP_B0M"], df["IB문의건수_APP_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0046(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_부대서비스_B0M <= IB문의건수_부대서비스_R6M
    """
    c1, c2 = df["IB문의건수_부대서비스_B0M"], df["IB문의건수_부대서비스_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0047(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_포인트_B0M <= IB문의건수_포인트_R6M
    """
    c1, c2 = df["IB문의건수_포인트_B0M"], df["IB문의건수_포인트_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0048(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_카드발급_B0M <= IB문의건수_카드발급_R6M
    """
    c1, c2 = df["IB문의건수_카드발급_B0M"], df["IB문의건수_카드발급_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0049(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_BL_B0M <= IB문의건수_BL_R6M
    """
    c1, c2 = df["IB문의건수_BL_B0M"], df["IB문의건수_BL_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0050(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_분실도난_B0M <= IB문의건수_분실도난_R6M
    """
    c1, c2 = df["IB문의건수_분실도난_B0M"], df["IB문의건수_분실도난_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0051(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_CA_B0M <= IB문의건수_CA_R6M
    """
    c1, c2 = df["IB문의건수_CA_B0M"], df["IB문의건수_CA_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0052(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_CL_RV_B0M <= IB문의건수_CL_RV_R6M
    """
    c1, c2 = df["IB문의건수_CL_RV_B0M"], df["IB문의건수_CL_RV_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0053(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB문의건수_CS_B0M <= IB문의건수_CS_R6M
    """
    c1, c2 = df["IB문의건수_CS_B0M"], df["IB문의건수_CS_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0054(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB상담건수_VOC_B0M <= IB상담건수_VOC_R6M
    """
    c1, c2 = df["IB상담건수_VOC_B0M"], df["IB상담건수_VOC_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0055(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB상담건수_VOC민원_B0M <= IB상담건수_VOC민원_R6M
    """
    c1, c2 = df["IB상담건수_VOC민원_B0M"], df["IB상담건수_VOC민원_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0056(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB상담건수_VOC불만_B0M <= IB상담건수_VOC불만_R6M
    """
    c1, c2 = df["IB상담건수_VOC불만_B0M"], df["IB상담건수_VOC불만_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0057(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IB상담건수_금감원_B0M <= IB상담건수_금감원_R6M
    """
    c1, c2 = df["IB상담건수_금감원_B0M"], df["IB상담건수_금감원_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0058(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        불만제기건수_B0M <= 불만제기건수_R12M
    """
    c1, c2 = df["불만제기건수_B0M"], df["불만제기건수_R12M"]
    return c1 <= c2


@constraint_udf
def cc_06_0059(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 불만제기건수_R12M >0: 0 <= 불만제기후경과월_R12M < 12
    """
    dd = df[["불만제기건수_R12M", "불만제기후경과월_R12M"]]
    ret = dd.apply(lambda x: (0 <= x[1] < 12) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_06_0060(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        당사멤버쉽_방문횟수_B0M <= 당사멤버쉽_방문횟수_R6M
    """
    c1, c2 = df["당사멤버쉽_방문횟수_B0M"], df["당사멤버쉽_방문횟수_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0061(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 당사멤버쉽_방문횟수_R6M >0: 0 < 당사멤버쉽_방문월수_R6M <= 6
    """
    dd = df[["당사멤버쉽_방문횟수_R6M", "당사멤버쉽_방문월수_R6M"]]
    ret = dd.apply(lambda x: (0 < x[1] <= 6) if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_06_0062(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        당사멤버쉽_방문월수_R6M <= 당사멤버쉽_방문횟수_R6M
    """
    c1, c2 = df["당사멤버쉽_방문월수_R6M"], df["당사멤버쉽_방문횟수_R6M"]
    return c1 <= c2


## 수식 -> 제약조건으로 변경
@constraint_udf
def cc_06_0063(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        상담건수_B0M >= SUM(IB상담건수_VOC_B0M, IB상담건수_금감원_B0M)
    """
    c1, c2 = df["IB상담건수_VOC_B0M"], df["IB상담건수_금감원_B0M"]
    res = c1 + c2

    c = df["상담건수_B0M"]
    return c >= res


@constraint_udf
def cc_06_0065(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        홈페이지_금융건수_R3M <= 홈페이지_금융건수_R6M
    """
    c1, c2 = df["홈페이지_금융건수_R3M"], df["홈페이지_금융건수_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0066(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        홈페이지_선결제건수_R3M <= 홈페이지_선결제건수_R6M
    """
    c1, c2 = df["홈페이지_선결제건수_R3M"], df["홈페이지_선결제건수_R6M"]
    return c1 <= c2


@constraint_udf
def cc_06_0067(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        상담건수_B0M <= 상담건수_R6M
    """
    c1, c2 = df["상담건수_B0M"], df["상담건수_R6M"]
    return c1 <= c2


@constraint_udf
def cf_06_0066(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IB상담건수_VOC_B0M = SUM(IB상담건수_VOC민원_B0M, IB상담건수_VOC불만_B0M)
    """
    c1, c2 = df["IB상담건수_VOC민원_B0M"], df["IB상담건수_VOC불만_B0M"]
    res = c1 + c2

    c = df["IB상담건수_VOC_B0M"]
    return c == res


@constraint_udf
def cf_06_0089(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IB상담건수_VOC_R6M = SUM(IB상담건수_VOC민원_R6M, IB상담건수_VOC불만_R6M)
    """
    c1, c2 = df["IB상담건수_VOC민원_R6M"], df["IB상담건수_VOC불만_R6M"]
    res = c1 + c2

    c = df["IB상담건수_VOC_R6M"]
    return c == res


@constraint_udf
def cf_06_0096(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        당사PAY_방문횟수_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["당사PAY_방문횟수_B0M"]
    return c == res


@constraint_udf
def cf_06_0097(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        당사PAY_방문횟수_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["당사PAY_방문횟수_R6M"]
    return c == res


@constraint_udf
def cf_06_0098(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        당사PAY_방문월수_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["당사PAY_방문월수_R6M"]
    return c == res


@constraint_udf
def cc_07_0001(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_카드론_TM_B0M <= 컨택건수_카드론_TM_R6M
    """
    c1, c2 = df["컨택건수_카드론_TM_B0M"], df["컨택건수_카드론_TM_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0002(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_리볼빙_TM_B0M <= 컨택건수_리볼빙_TM_R6M
    """
    c1, c2 = df["컨택건수_리볼빙_TM_B0M"], df["컨택건수_리볼빙_TM_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0003(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_CA_TM_B0M <= 컨택건수_CA_TM_R6M
    """
    c1, c2 = df["컨택건수_CA_TM_B0M"], df["컨택건수_CA_TM_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0004(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_이용유도_TM_B0M <= 컨택건수_이용유도_TM_R6M
    """
    c1, c2 = df["컨택건수_이용유도_TM_B0M"], df["컨택건수_이용유도_TM_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0005(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_신용발급_TM_B0M <= 컨택건수_신용발급_TM_R6M
    """
    c1, c2 = df["컨택건수_신용발급_TM_B0M"], df["컨택건수_신용발급_TM_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0006(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_부대서비스_TM_B0M <= 컨택건수_부대서비스_TM_R6M
    """
    c1, c2 = df["컨택건수_부대서비스_TM_B0M"], df["컨택건수_부대서비스_TM_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0007(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_포인트소진_TM_B0M <= 컨택건수_포인트소진_TM_R6M
    """
    c1, c2 = df["컨택건수_포인트소진_TM_B0M"], df["컨택건수_포인트소진_TM_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0008(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_보험_TM_B0M <= 컨택건수_보험_TM_R6M
    """
    c1, c2 = df["컨택건수_보험_TM_B0M"], df["컨택건수_보험_TM_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0009(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_카드론_LMS_B0M <= 컨택건수_카드론_LMS_R6M
    """
    c1, c2 = df["컨택건수_카드론_LMS_B0M"], df["컨택건수_카드론_LMS_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0010(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_CA_LMS_B0M <= 컨택건수_CA_LMS_R6M
    """
    c1, c2 = df["컨택건수_CA_LMS_B0M"], df["컨택건수_CA_LMS_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0011(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_리볼빙_LMS_B0M <= 컨택건수_리볼빙_LMS_R6M
    """
    c1, c2 = df["컨택건수_리볼빙_LMS_B0M"], df["컨택건수_리볼빙_LMS_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0012(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_이용유도_LMS_B0M <= 컨택건수_이용유도_LMS_R6M
    """
    c1, c2 = df["컨택건수_이용유도_LMS_B0M"], df["컨택건수_이용유도_LMS_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0013(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_카드론_EM_B0M <= 컨택건수_카드론_EM_R6M
    """
    c1, c2 = df["컨택건수_카드론_EM_B0M"], df["컨택건수_카드론_EM_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0014(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_CA_EM_B0M <= 컨택건수_CA_EM_R6M
    """
    c1, c2 = df["컨택건수_CA_EM_B0M"], df["컨택건수_CA_EM_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0015(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_리볼빙_EM_B0M <= 컨택건수_리볼빙_EM_R6M
    """
    c1, c2 = df["컨택건수_리볼빙_EM_B0M"], df["컨택건수_리볼빙_EM_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0016(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_이용유도_EM_B0M <= 컨택건수_이용유도_EM_R6M
    """
    c1, c2 = df["컨택건수_이용유도_EM_B0M"], df["컨택건수_이용유도_EM_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0017(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_카드론_청구서_B0M <= 컨택건수_카드론_청구서_R6M
    """
    c1, c2 = df["컨택건수_카드론_청구서_B0M"], df["컨택건수_카드론_청구서_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0018(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_CA_청구서_B0M <= 컨택건수_CA_청구서_R6M
    """
    c1, c2 = df["컨택건수_CA_청구서_B0M"], df["컨택건수_CA_청구서_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0019(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_리볼빙_청구서_B0M <= 컨택건수_리볼빙_청구서_R6M
    """
    c1, c2 = df["컨택건수_리볼빙_청구서_B0M"], df["컨택건수_리볼빙_청구서_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0020(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_이용유도_청구서_B0M <= 컨택건수_이용유도_청구서_R6M
    """
    c1, c2 = df["컨택건수_이용유도_청구서_B0M"], df["컨택건수_이용유도_청구서_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0021(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_카드론_인터넷_B0M <= 컨택건수_카드론_인터넷_R6M
    """
    c1, c2 = df["컨택건수_카드론_인터넷_B0M"], df["컨택건수_카드론_인터넷_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0022(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_CA_인터넷_B0M <= 컨택건수_CA_인터넷_R6M
    """
    c1, c2 = df["컨택건수_CA_인터넷_B0M"], df["컨택건수_CA_인터넷_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0023(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_리볼빙_인터넷_B0M <= 컨택건수_리볼빙_인터넷_R6M
    """
    c1, c2 = df["컨택건수_리볼빙_인터넷_B0M"], df["컨택건수_리볼빙_인터넷_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0024(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_이용유도_인터넷_B0M <= 컨택건수_이용유도_인터넷_R6M
    """
    c1, c2 = df["컨택건수_이용유도_인터넷_B0M"], df["컨택건수_이용유도_인터넷_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0025(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_카드론_당사앱_B0M <= 컨택건수_카드론_당사앱_R6M
    """
    c1, c2 = df["컨택건수_카드론_당사앱_B0M"], df["컨택건수_카드론_당사앱_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0026(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_CA_당사앱_B0M <= 컨택건수_CA_당사앱_R6M
    """
    c1, c2 = df["컨택건수_CA_당사앱_B0M"], df["컨택건수_CA_당사앱_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0027(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_리볼빙_당사앱_B0M <= 컨택건수_리볼빙_당사앱_R6M
    """
    c1, c2 = df["컨택건수_리볼빙_당사앱_B0M"], df["컨택건수_리볼빙_당사앱_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0028(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_이용유도_당사앱_B0M <= 컨택건수_이용유도_당사앱_R6M
    """
    c1, c2 = df["컨택건수_이용유도_당사앱_B0M"], df["컨택건수_이용유도_당사앱_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0029(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_채권_B0M <= 컨택건수_채권_R6M
    """
    c1, c2 = df["컨택건수_채권_B0M"], df["컨택건수_채권_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0030(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        컨택건수_FDS_B0M <= 컨택건수_FDS_R6M
    """
    c1, c2 = df["컨택건수_FDS_B0M"], df["컨택건수_FDS_R6M"]
    return c1 <= c2


@constraint_udf
def cc_07_0031(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        캠페인접촉건수_R12M >= 캠페인접촉일수_R12M
    """
    c1, c2 = df["캠페인접촉건수_R12M"], df["캠페인접촉일수_R12M"]
    return c1 >= c2


@constraint_udf
def cf_07_0019(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_CA_EM_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_CA_EM_B0M"]
    return c == res


@constraint_udf
def cf_07_0020(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_리볼빙_EM_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_리볼빙_EM_B0M"]
    return c == res


@constraint_udf
def cf_07_0024(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_리볼빙_청구서_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_리볼빙_청구서_B0M"]
    return c == res


@constraint_udf
def cf_07_0026(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_카드론_인터넷_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_카드론_인터넷_B0M"]
    return c == res


@constraint_udf
def cf_07_0027(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_CA_인터넷_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_CA_인터넷_B0M"]
    return c == res


@constraint_udf
def cf_07_0028(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_리볼빙_인터넷_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_리볼빙_인터넷_B0M"]
    return c == res


@constraint_udf
def cf_07_0032(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_리볼빙_당사앱_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_리볼빙_당사앱_B0M"]
    return c == res


@constraint_udf
def cf_07_0047(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_CA_EM_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_CA_EM_R6M"]
    return c == res


@constraint_udf
def cf_07_0048(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_리볼빙_EM_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_리볼빙_EM_R6M"]
    return c == res


@constraint_udf
def cf_07_0052(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_리볼빙_청구서_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_리볼빙_청구서_R6M"]
    return c == res


@constraint_udf
def cf_07_0054(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_카드론_인터넷_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_카드론_인터넷_R6M"]
    return c == res


@constraint_udf
def cf_07_0055(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_CA_인터넷_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_CA_인터넷_R6M"]
    return c == res


@constraint_udf
def cf_07_0056(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_리볼빙_인터넷_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_리볼빙_인터넷_R6M"]
    return c == res


@constraint_udf
def cf_07_0060(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        컨택건수_리볼빙_당사앱_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["컨택건수_리볼빙_당사앱_R6M"]
    return c == res


@constraint_udf
def cc_08_0001(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        min(증감율_이용건수_신판_전월, 증감율_이용건수_CA_전월) <= 증감율_이용건수_신용_전월
        <= max(증감율_이용건수_신판_전월, 증감율_이용건수_CA_전월)
    """
    dd = df[["증감율_이용건수_신판_전월", "증감율_이용건수_CA_전월"]]
    _min, _max = dd.min(axis=1), dd.max(axis=1)

    c = df["증감율_이용건수_신용_전월"]
    return (_min <= c) & (c <= _max)


@constraint_udf
def cc_08_0002(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        min(증감율_이용건수_일시불_전월, 증감율_이용건수_할부_전월) <= 증감율_이용건수_신판_전월
        <= max(증감율_이용건수_일시불_전월, 증감율_이용건수_할부_전월)
    """
    dd = df[["증감율_이용건수_일시불_전월", "증감율_이용건수_할부_전월"]]
    _min, _max = dd.min(axis=1), dd.max(axis=1)

    c = df["증감율_이용건수_신판_전월"]
    return (_min <= c) & (c <= _max)


@constraint_udf
def cc_08_0003(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        min(증감율_이용금액_신판_전월, 증감율_이용금액_CA_전월) <= 증감율_이용금액_신용_전월
        <= max(증감율_이용금액_신판_전월, 증감율_이용금액_CA_전월)
    """
    dd = df[["증감율_이용금액_신판_전월", "증감율_이용금액_CA_전월"]]
    _min, _max = dd.min(axis=1), dd.max(axis=1)

    c = df["증감율_이용금액_신용_전월"]
    return (_min <= c) & (c <= _max)


@constraint_udf
def cc_08_0004(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        min(증감율_이용금액_일시불_전월, 증감율_이용금액_할부_전월) <= 증감율_이용금액_신판_전월
        <= max(증감율_이용금액_일시불_전월, 증감율_이용금액_할부_전월)
    """
    dd = df[["증감율_이용금액_일시불_전월", "증감율_이용금액_할부_전월"]]
    _min, _max = dd.min(axis=1), dd.max(axis=1)

    c = df["증감율_이용금액_신판_전월"]
    return (_min <= c) & (c <= _max)


@constraint_udf
def cc_08_0005(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        min(증감율_이용건수_신판_분기, 증감율_이용건수_CA_분기) <= 증감율_이용건수_신용_분기
        <= max(증감율_이용건수_신판_분기, 증감율_이용건수_CA_분기)
    """
    dd = df[["증감율_이용건수_신판_분기", "증감율_이용건수_CA_분기"]]
    _min, _max = dd.min(axis=1), dd.max(axis=1)

    c = df["증감율_이용건수_신용_분기"]
    return (_min <= c) & (c <= _max)


@constraint_udf
def cc_08_0006(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        min(증감율_이용건수_일시불_분기, 증감율_이용건수_할부_분기) <= 증감율_이용건수_신판_분기
        <= max(증감율_이용건수_일시불_분기, 증감율_이용건수_할부_분기)
    """
    dd = df[["증감율_이용건수_일시불_분기", "증감율_이용건수_할부_분기"]]
    _min, _max = dd.min(axis=1), dd.max(axis=1)

    c = df["증감율_이용건수_신판_분기"]
    return (_min <= c) & (c <= _max)


@constraint_udf
def cc_08_0007(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        min(증감율_이용금액_신판_분기, 증감율_이용금액_CA_분기) <= 증감율_이용금액_신용_분기
        <= max(증감율_이용금액_신판_분기, 증감율_이용금액_CA_분기)
    """
    dd = df[["증감율_이용금액_신판_분기", "증감율_이용금액_CA_분기"]]
    _min, _max = dd.min(axis=1), dd.max(axis=1)

    c = df["증감율_이용금액_신용_분기"]
    return (_min <= c) & (c <= _max)


@constraint_udf
def cc_08_0008(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        min(증감율_이용금액_일시불_분기, 증감율_이용금액_할부_분기) <= 증감율_이용금액_신판_분기
        <= max(증감율_이용금액_일시불_분기, 증감율_이용금액_할부_분기)
    """
    dd = df[["증감율_이용금액_일시불_분기", "증감율_이용금액_할부_분기"]]
    _min, _max = dd.min(axis=1), dd.max(axis=1)

    c = df["증감율_이용금액_신판_분기"]
    return (_min <= c) & (c <= _max)


@constraint_udf
def cc_08_0009(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        잔액_신판최대한도소진율_r3m >= 잔액_신판평균한도소진율_r3m
    """
    c1, c2 = df["잔액_신판최대한도소진율_r3m"], df["잔액_신판평균한도소진율_r3m"]
    return c1 >= c2


@constraint_udf
def cc_08_0010(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        잔액_신판최대한도소진율_r6m >= 잔액_신판평균한도소진율_r6m
    """
    c1, c2 = df["잔액_신판최대한도소진율_r6m"], df["잔액_신판평균한도소진율_r6m"]
    return c1 >= c2


@constraint_udf
def cc_08_0011(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        잔액_신판ca최대한도소진율_r3m >= 잔액_신판ca평균한도소진율_r3m
    """
    c1, c2 = df["잔액_신판ca최대한도소진율_r3m"], df["잔액_신판ca평균한도소진율_r3m"]
    return c1 >= c2


@constraint_udf
def cc_08_0012(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        잔액_신판ca최대한도소진율_r6m >= 잔액_신판ca평균한도소진율_r6m
    """
    c1, c2 = df["잔액_신판ca최대한도소진율_r6m"], df["잔액_신판ca평균한도소진율_r6m"]
    return c1 >= c2


@constraint_udf
def cf_08_0005(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 증감율_이용건수_CA_전월 == 0 THEN 증감율_이용건수_신용_전월 = 증감율_이용건수_신판_전월
        ELIF 증감율_이용건수_신판_전월 == 0 THEN 증감율_이용건수_신용_전월 = 증감율_이용건수_CA_전월
        ELSE PASS
    """
    dd = df[["증감율_이용건수_CA_전월", "증감율_이용건수_신판_전월", "증감율_이용건수_신용_전월"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)

    c = df["증감율_이용건수_신용_전월"]
    return c == res


@constraint_udf
def cf_08_0006(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 증감율_이용건수_할부_전월 == 0 THEN 증감율_이용건수_신판_전월 = 증감율_이용건수_일시불_전월
        ELIF 증감율_이용건수_일시불_전월 == 0 THEN 증감율_이용건수_신판_전월 = 증감율_이용건수_할부_전월
        ELSE PASS
    """
    dd = df[["증감율_이용건수_할부_전월", "증감율_이용건수_일시불_전월", "증감율_이용건수_신판_전월"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)

    c = df["증감율_이용건수_신판_전월"]
    return c == res


@constraint_udf
def cf_08_0012(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 증감율_이용금액_CA_전월 == 0 THEN 증감율_이용금액_신용_전월 = 증감율_이용금액_신판_전월
        ELIF 증감율_이용금액_신판_전월 == 0 THEN 증감율_이용금액_신용_전월 = 증감율_이용금액_CA_전월
        ELSE PASS
    """
    dd = df[["증감율_이용금액_CA_전월", "증감율_이용금액_신판_전월", "증감율_이용금액_신용_전월"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)

    c = df["증감율_이용금액_신용_전월"]
    return c == res


@constraint_udf
def cf_08_0013(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 증감율_이용금액_할부_전월 == 0 THEN 증감율_이용금액_신판_전월 = 증감율_이용금액_일시불_전월
        ELIF 증감율_이용금액_일시불_전월 == 0 THEN 증감율_이용금액_신판_전월 = 증감율_이용금액_할부_전월
        ELSE PASS
    """
    dd = df[["증감율_이용금액_할부_전월", "증감율_이용금액_일시불_전월", "증감율_이용금액_신판_전월"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)

    c = df["증감율_이용금액_신판_전월"]
    return c == res


@constraint_udf
def cf_08_0033(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 증감율_이용건수_CA_분기 == 0 THEN 증감율_이용건수_신용_분기 = 증감율_이용건수_신판_분기
        ELIF 증감율_이용건수_신판_분기 == 0 THEN 증감율_이용건수_신용_분기 = 증감율_이용건수_CA_분기
        ELSE PASS
    """
    dd = df[["증감율_이용건수_CA_분기", "증감율_이용건수_신판_분기", "증감율_이용건수_신용_분기"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)

    c = df["증감율_이용건수_신용_분기"]
    return c == res


@constraint_udf
def cf_08_0034(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 증감율_이용건수_할부_분기 == 0 THEN 증감율_이용건수_신판_분기 = 증감율_이용건수_일시불_분기
        ELIF 증감율_이용건수_일시불_분기 == 0 THEN 증감율_이용건수_신판_분기 = 증감율_이용건수_할부_분기
        ELSE PASS
    """
    dd = df[["증감율_이용건수_할부_분기", "증감율_이용건수_일시불_분기", "증감율_이용건수_신판_분기"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)

    c = df["증감율_이용건수_신판_분기"]
    return c == res


@constraint_udf
def cf_08_0040(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 증감율_이용금액_CA_분기 == 0 THEN 증감율_이용금액_신용_분기 = 증감율_이용금액_신판_분기
        ELIF 증감율_이용금액_신판_분기 == 0 THEN 증감율_이용금액_신용_분기 = 증감율_이용금액_CA_분기
        ELSE PASS
    """
    dd = df[["증감율_이용금액_CA_분기", "증감율_이용금액_신판_분기", "증감율_이용금액_신용_분기"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)

    c = df["증감율_이용금액_신용_분기"]
    return c == res


@constraint_udf
def cf_08_0041(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        IF 증감율_이용금액_할부_분기 == 0 THEN 증감율_이용금액_신판_분기 = 증감율_이용금액_일시불_분기
        ELIF 증감율_이용금액_일시불_분기 == 0 THEN 증감율_이용금액_신판_분기 = 증감율_이용금액_할부_분기
        ELSE PASS
    """
    dd = df[["증감율_이용금액_할부_분기", "증감율_이용금액_일시불_분기", "증감율_이용금액_신판_분기"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)

    c = df["증감율_이용금액_신판_분기"]
    return c == res


# 추가됨 마지막 번호 달아줌
@constraint_udf
def cc_03_0001(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용금액_신용_B0M <= 카드이용한도금액
    """
    c1, c2 = df["이용금액_신용_B0M"], df["카드이용한도금액"]  # pd.Series
    return c1 <= c2


@constraint_udf
def cc_03_0002(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용금액_CA_B0M <= CA한도금액
    """
    c1, c2 = df["이용금액_CA_B0M"], df["CA한도금액"]  # pd.Series
    return c1 <= c2


@constraint_udf
def cc_03_0003(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용금액_카드론_B0M <= 월상환론한도금액
    """
    c1, c2 = df["이용금액_카드론_B0M"], df["월상환론한도금액"]  # pd.Series
    return c1 <= c2


@constraint_udf
def cc_03_0004(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF 이용건수_일시불_R12M >0: 이용후경과월_일시불 < 12
    """
    dd = df[["이용건수_일시불_R12M", "이용후경과월_일시불"]]
    ret = dd.apply(lambda x: x[1] < 12 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0005(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        IF 이용건수_할부_R12M >0: 이용후경과월_할부 < 12
    """
    dd = df[["이용건수_할부_R12M", "이용후경과월_할부"]]
    ret = dd.apply(lambda x: x[1] < 12 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0006(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_할부_유이자_R12M >0: 이용후경과월_할부_유이자 < 12
    """
    dd = df[["이용건수_할부_유이자_R12M", "이용후경과월_할부_유이자"]]
    ret = dd.apply(lambda x: x[1] < 12 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0007(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_할부_무이자_R12M >0: 이용후경과월_할부_무이자 < 12
    """
    dd = df[["이용건수_할부_무이자_R12M", "이용후경과월_할부_무이자"]]
    ret = dd.apply(lambda x: x[1] < 12 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0008(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_부분무이자_R12M >0: 이용후경과월_부분무이자 < 12
    """
    dd = df[["이용건수_부분무이자_R12M", "이용후경과월_부분무이자"]]
    ret = dd.apply(lambda x: x[1] < 12 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0009(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_CA_R12M >0: 이용후경과월_CA < 12
    """
    dd = df[["이용건수_CA_R12M", "이용후경과월_CA"]]
    ret = dd.apply(lambda x: x[1] < 12 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0010(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_체크_R12M >0: 이용후경과월_체크 < 12
    """
    dd = df[["이용건수_체크_R12M", "이용후경과월_체크"]]
    ret = dd.apply(lambda x: x[1] < 12 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0011(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_카드론_R12M >0: 이용후경과월_카드론 < 12
    """
    dd = df[["이용건수_카드론_R12M", "이용후경과월_카드론"]]
    ret = dd.apply(lambda x: x[1] < 12 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0012(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_신용_B0M <= 이용건수_신용_R3M <= 이용건수_신용_R6M <= 이용건수_신용_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용건수_신용_B0M"],
        df["이용건수_신용_R3M"],
        df["이용건수_신용_R6M"],
        df["이용건수_신용_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0013(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_신판_B0M <= 이용건수_신판_R3M <= 이용건수_신판_R6M <= 이용건수_신판_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용건수_신판_B0M"],
        df["이용건수_신판_R3M"],
        df["이용건수_신판_R6M"],
        df["이용건수_신판_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0014(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_일시불_B0M <= 이용건수_일시불_R3M <= 이용건수_일시불_R6M <= 이용건수_일시불_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용건수_일시불_B0M"],
        df["이용건수_일시불_R3M"],
        df["이용건수_일시불_R6M"],
        df["이용건수_일시불_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0015(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_할부_B0M <= 이용건수_할부_R3M <= 이용건수_할부_R6M <= 이용건수_할부_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용건수_할부_B0M"],
        df["이용건수_할부_R3M"],
        df["이용건수_할부_R6M"],
        df["이용건수_할부_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0016(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_할부_유이자_B0M <= 이용건수_할부_유이자_R3M <= 이용건수_할부_유이자_R6M <= 이용건수_할부_유이자_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용건수_할부_유이자_B0M"],
        df["이용건수_할부_유이자_R3M"],
        df["이용건수_할부_유이자_R6M"],
        df["이용건수_할부_유이자_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0017(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_할부_무이자_B0M <= 이용건수_할부_무이자_R3M <= 이용건수_할부_무이자_R6M <= 이용건수_할부_무이자_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용건수_할부_무이자_B0M"],
        df["이용건수_할부_무이자_R3M"],
        df["이용건수_할부_무이자_R6M"],
        df["이용건수_할부_무이자_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0018(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_부분무이자_B0M <= 이용건수_부문무이자_R3M <= 이용건수_부분무이자_R6M <= 이용건수_부분무이자_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용건수_부분무이자_B0M"],
        df["이용건수_부분무이자_R3M"],
        df["이용건수_부분무이자_R6M"],
        df["이용건수_부분무이자_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0019(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_CA_B0M <= 이용건수_CA_R3M <= 이용건수_CA_R6M <= 이용건수_CA_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용건수_CA_B0M"],
        df["이용건수_CA_R3M"],
        df["이용건수_CA_R6M"],
        df["이용건수_CA_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0020(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_체크_B0M <= 이용건수_체크_R3M <= 이용건수_체크_R6M <= 이용건수_체크_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용건수_체크_B0M"],
        df["이용건수_체크_R3M"],
        df["이용건수_체크_R6M"],
        df["이용건수_체크_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0021(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_카드론_B0M <= 이용건수_카드론_R3M <= 이용건수_카드론_R6M <= 이용건수_카드론_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용건수_카드론_B0M"],
        df["이용건수_카드론_R3M"],
        df["이용건수_카드론_R6M"],
        df["이용건수_카드론_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0022(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_신용_B0M <= 이용금액_신용_R3M <= 이용금액_신용_R6M <= 이용금액_신용_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용금액_신용_B0M"],
        df["이용금액_신용_R3M"],
        df["이용금액_신용_R6M"],
        df["이용금액_신용_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0023(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_신판_B0M <= 이용금액_신판_R3M <= 이용금액_신판_R6M <= 이용금액_신판_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용금액_신판_B0M"],
        df["이용금액_신판_R3M"],
        df["이용금액_신판_R6M"],
        df["이용금액_신판_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0024(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_일시불_B0M <= 이용금액_일시불_R3M <= 이용금액_일시불_R6M <= 이용금액_일시불_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용금액_일시불_B0M"],
        df["이용금액_일시불_R3M"],
        df["이용금액_일시불_R6M"],
        df["이용금액_일시불_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0025(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_할부_B0M <= 이용금액_할부_R3M <= 이용금액_할부_R6M <= 이용금액_할부_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용금액_할부_B0M"],
        df["이용금액_할부_R3M"],
        df["이용금액_할부_R6M"],
        df["이용금액_할부_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0026(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_할부_유이자_B0M <= 이용금액_할부_유이자_R3M <= 이용금액_할부_유이자_R6M <= 이용금액_할부_유이자_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용금액_할부_유이자_B0M"],
        df["이용금액_할부_유이자_R3M"],
        df["이용금액_할부_유이자_R6M"],
        df["이용금액_할부_유이자_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0027(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_할부_무이자_B0M <= 이용금액_할부_무이자_R3M <= 이용금액_할부_무이자_R6M <= 이용금액_할부_무이자_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용금액_할부_무이자_B0M"],
        df["이용금액_할부_무이자_R3M"],
        df["이용금액_할부_무이자_R6M"],
        df["이용금액_할부_무이자_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0028(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_부분무이자_B0M <= 이용금액_부분무이자_R3M <= 이용금액_부분무이자_R6M <= 이용금액_부분무이자_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용금액_부분무이자_B0M"],
        df["이용금액_부분무이자_R3M"],
        df["이용금액_부분무이자_R6M"],
        df["이용금액_부분무이자_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0029(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_CA_B0M <= 이용금액_CA_R3M <= 이용금액_CA_R6M <= 이용금액_CA_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용금액_CA_B0M"],
        df["이용금액_CA_R3M"],
        df["이용금액_CA_R6M"],
        df["이용금액_CA_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0030(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_체크_B0M <= 이용금액_체크_R3M <= 이용금액_체크_R6M <= 이용금액_체크_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용금액_체크_B0M"],
        df["이용금액_체크_R3M"],
        df["이용금액_체크_R6M"],
        df["이용금액_체크_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


@constraint_udf
def cc_03_0031(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_카드론_B0M <= 이용금액_카드론_R3M <= 이용금액_카드론_R6M <= 이용금액_카드론_R12M
    """
    b0m, r3m, r6m, r12m = (
        df["이용금액_카드론_B0M"],
        df["이용금액_카드론_R3M"],
        df["이용금액_카드론_R6M"],
        df["이용금액_카드론_R12M"],
    )  # pd.Series
    return (b0m <= r3m) & (r3m <= r6m) & (r6m <= r12m)


# 수식 -> 제약조건으로 변경됨
@constraint_udf
def cc_03_0156(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        최대이용금액_신용_R12M >= MAX(최대이용금액_신판_R12M, 최대이용금액_CA_R12M)
    """
    dd = df[["최대이용금액_신판_R12M", "최대이용금액_CA_R12M"]]
    res = dd.max(axis=1).astype(int)

    c = df["최대이용금액_신용_R12M"]
    return c >= res


# 수식 -> 제약조건으로 변경됨
@constraint_udf
def cc_03_0157(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        최대이용금액_신판_R12M >= MAX(최대이용금액_일시불_R12M, 최대이용금액_할부_R12M)
    """
    dd = df[["최대이용금액_일시불_R12M", "최대이용금액_할부_R12M"]]
    res = dd.max(axis=1).astype(int)

    c = df["최대이용금액_신판_R12M"]
    return c >= res


@constraint_udf
def cc_03_0039(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_신용_R3M <= 이용개월수_신용_R6M <= 이용개월수_신용_R12M <= 12
    """
    c1, c2, c3 = df["이용개월수_신용_R3M"], df["이용개월수_신용_R6M"], df["이용개월수_신용_R12M"]
    return (c1 <= c2) & (c2 <= c3) & (c3 <= 12)


@constraint_udf
def cc_03_0040(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       MAX(이용개월수_신판_R12M, 이용개월수_CA_R12M) <= 이용개월수_신용_R12M
    """
    c1, c2, c3 = df["이용개월수_신판_R12M"], df["이용개월수_CA_R12M"], df["이용개월수_신용_R12M"]
    return (c3 >= c1) & (c3 >= c2)


@constraint_udf
def cc_03_0041(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_신판_R3M <= 이용개월수_신판_R6M <= 이용개월수_신판_R12M <= 12
    """
    c1, c2, c3 = df["이용개월수_신판_R3M"], df["이용개월수_신판_R6M"], df["이용개월수_신판_R12M"]
    return (c1 <= c2) & (c2 <= c3) & (c3 <= 12)


@constraint_udf
def cc_03_0042(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       MAX(이용개월수_일시불_R12M, 이용개월수_할부_R12M) <= 이용개월수_신판_R12M
    """
    c1, c2, c3 = df["이용개월수_일시불_R12M"], df["이용개월수_할부_R12M"], df["이용개월수_신판_R12M"]
    return (c3 >= c1) & (c3 >= c2)


@constraint_udf
def cc_03_0043(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_일시불_R3M <= 이용개월수_일시불_R6M <= 이용개월수_일시불_R12M <= 12
    """
    c1, c2, c3 = df["이용개월수_일시불_R3M"], df["이용개월수_일시불_R6M"], df["이용개월수_일시불_R12M"]
    return (c1 <= c2) & (c2 <= c3) & (c3 <= 12)


@constraint_udf
def cc_03_0044(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_할부_R3M <= 이용개월수_할부_유이자_R6M <= 이용개월수_할부_유이자_R12M <= 12
    """
    c1, c2, c3 = df["이용개월수_할부_R3M"], df["이용개월수_할부_유이자_R6M"], df["이용개월수_할부_유이자_R12M"]
    return (c1 <= c2) & (c2 <= c3) & (c3 <= 12)


@constraint_udf
def cc_03_0045(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       MAX(이용개월수_할부_유이자_R12M, 이용개월수_할부_무이자_R12M, 이용개월수_부분무이자_R12M) <= 이용개월수_할부_R12M
    """
    dd = df[["이용개월수_할부_유이자_R12M", "이용개월수_할부_무이자_R12M", "이용개월수_부분무이자_R12M"]]
    c_m = dd.max(axis=1)
    c1 = df["이용개월수_할부_R12M"]
    return c_m <= c1


@constraint_udf
def cc_03_0046(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_할부_유이자_R3M <= 이용개월수_할부_유이자_R6M <= 이용개월수_할부_유이자_R12M <= 12
    """
    c1, c2, c3 = df["이용개월수_할부_유이자_R3M"], df["이용개월수_할부_유이자_R6M"], df["이용개월수_할부_유이자_R12M"]
    return (c1 <= c2) & (c2 <= c3) & (c3 <= 12)


@constraint_udf
def cc_03_0047(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_할부_무이자_R3M <= 이용개월수_할부_무이자_R6M <= 이용개월수_할부_무이자_R12M <= 12
    """
    c1, c2, c3 = df["이용개월수_할부_무이자_R3M"], df["이용개월수_할부_무이자_R6M"], df["이용개월수_할부_무이자_R12M"]
    return (c1 <= c2) & (c2 <= c3) & (c3 <= 12)


@constraint_udf
def cc_03_0048(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_부분무이자_R3M <= 이용개월수_부분무이자_R6M <= 이용개월수_부분무이자_R12M <= 12
    """
    c1, c2, c3 = df["이용개월수_부분무이자_R3M"], df["이용개월수_부분무이자_R6M"], df["이용개월수_부분무이자_R12M"]
    return (c1 <= c2) & (c2 <= c3) & (c3 <= 12)


@constraint_udf
def cc_03_0049(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_CA_R3M <= 이용개월수_CA_R6M <= 이용개월수_CA_R12M <= 12
    """
    c1, c2, c3 = df["이용개월수_CA_R3M"], df["이용개월수_CA_R6M"], df["이용개월수_CA_R12M"]
    return (c1 <= c2) & (c2 <= c3) & (c3 <= 12)


@constraint_udf
def cc_03_0050(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_체크_R3M <= 이용개월수_체크_R6M <= 이용개월수_체크_R12M <= 12
    """
    c1, c2, c3 = df["이용개월수_체크_R3M"], df["이용개월수_체크_R6M"], df["이용개월수_체크_R12M"]
    return (c1 <= c2) & (c2 <= c3) & (c3 <= 12)


@constraint_udf
def cc_03_0051(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_카드론_R3M <= 이용개월수_카드론_R6M <= 이용개월수_카드론_R12M <= 12
    """
    c1, c2, c3 = df["이용개월수_카드론_R3M"], df["이용개월수_카드론_R6M"], df["이용개월수_카드론_R12M"]
    return (c1 <= c2) & (c2 <= c3) & (c3 <= 12)


@constraint_udf
def cc_03_0052(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용금액_신용_R3M>0: 이용건수_신용_R3M >0
    """
    dd = df[["이용금액_신용_R3M", "이용건수_신용_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0053(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용금액_신판_R3M>0: 이용건수_신판_R3M >0
    """
    dd = df[["이용금액_신판_R3M", "이용건수_신판_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0054(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용금액_일시불_R3M>0: 이용건수_일시불_R3M >0
    """
    dd = df[["이용금액_일시불_R3M", "이용건수_일시불_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0055(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용금액_할부_R3M>0: 이용건수_할부_R3M >0
    """
    dd = df[["이용금액_할부_R3M", "이용건수_할부_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0056(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용금액_할부_유이자_R3M>0: 이용건수_할부_유이자_R3M >0
    """
    dd = df[["이용금액_할부_유이자_R3M", "이용건수_할부_유이자_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0057(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용금액_할부_무이자_R3M>0: 이용건수_할부_무이자_R3M >0
    """
    dd = df[["이용금액_할부_무이자_R3M", "이용건수_할부_무이자_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0058(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용금액_부분무이자_R3M>0: 이용건수_부분무이자_R3M >0
    """
    dd = df[["이용금액_부분무이자_R3M", "이용건수_부분무이자_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0059(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용금액_CA_R3M>0: 이용건수_CA_R3M >0
    """
    dd = df[["이용금액_CA_R3M", "이용건수_CA_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0060(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용금액_체크_R3M>0: 이용건수_체크_R3M >0
    """
    dd = df[["이용금액_체크_R3M", "이용건수_체크_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0061(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용금액_카드론_R3M>0: 이용건수_카드론_R3M >0
    """
    dd = df[["이용금액_카드론_R3M", "이용건수_카드론_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0062(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       MAX(이용개월수_신판_R3M, 이용개월수_CA_R3M) <= 이용개월수_신용_R3M
    """
    dd = df[["이용개월수_신판_R3M", "이용개월수_CA_R3M"]]
    c_m = dd.max(axis=1)
    c1 = df["이용개월수_신용_R3M"]
    return c_m <= c1


@constraint_udf
def cc_03_0063(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_신용_R3M>0: 이용개월수_신용_R3M >0
    """
    dd = df[["이용건수_신용_R3M", "이용개월수_신용_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0064(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       MAX(이용개월수_일시불_R3M, 이용개월수_할부_R3M) <= 이용개월수_신판_R3M
    """
    dd = df[["이용개월수_일시불_R3M", "이용개월수_할부_R3M"]]
    c_m = dd.max(axis=1)
    c1 = df["이용개월수_신판_R3M"]
    return c_m <= c1


@constraint_udf
def cc_03_0065(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_신판_R3M>0: 이용개월수_신판_R3M >0
    """
    dd = df[["이용건수_신판_R3M", "이용개월수_신판_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0066(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_일시불_R3M>0: 이용개월수_일시불_R3M >0
    """
    dd = df[["이용건수_일시불_R3M", "이용개월수_일시불_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0067(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       MAX(이용개월수_할부_유이자_R3M, 이용개월수_할부_무이자_R3M, 이용개월수_부분무이자_R3M) <= 이용개월수_할부_R3M
    """
    dd = df[["이용개월수_할부_유이자_R3M", "이용개월수_할부_무이자_R3M", "이용개월수_부분무이자_R3M"]]
    c_m = dd.max(axis=1)
    c1 = df["이용개월수_할부_R3M"]
    return c_m <= c1


@constraint_udf
def cc_03_0068(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_할부_R3M>0: 이용개월수_할부_R3M >0
    """
    dd = df[["이용건수_할부_R3M", "이용개월수_할부_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0069(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_할부_유이자_R3M>0: 이용개월수_할부_유이자_R3M >0
    """
    dd = df[["이용건수_할부_유이자_R3M", "이용개월수_할부_유이자_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0070(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_할부_무이자_R3M>0: 이용개월수_할부_무이자_R3M >0
    """
    dd = df[["이용건수_할부_무이자_R3M", "이용개월수_할부_무이자_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0071(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_부분무이자_R3M>0: 이용개월수_부분무이자_R3M >0
    """
    dd = df[["이용건수_부분무이자_R3M", "이용개월수_부분무이자_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0072(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_CA_R3M>0: 이용개월수_CA_R3M >0
    """
    dd = df[["이용건수_CA_R3M", "이용개월수_CA_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0073(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_체크_R3M>0: 이용개월수_체크_R3M >0
    """
    dd = df[["이용건수_체크_R3M", "이용개월수_체크_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0074(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용건수_카드론_R3M>0: 이용개월수_카드론_R3M >0
    """
    dd = df[["이용건수_카드론_R3M", "이용개월수_카드론_R3M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0075(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       건수_할부전환_R3M <= 건수_할부전환_R6M <= 건수_할부전환_R12M
    """
    c1, c2, c3 = df["건수_할부전환_R3M"], df["건수_할부전환_R6M"], df["건수_할부전환_R12M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0076(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       금액_할부전환_R3M <= 금액_할부전환_R6M <= 금액_할부전환_R12M
    """
    c1, c2, c3 = df["금액_할부전환_R3M"], df["금액_할부전환_R6M"], df["금액_할부전환_R12M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0077(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_할부전환_R3M <= 이용개월수_할부전환_R6M <= 이용개월수_할부전환_R12M
    """
    c1, c2, c3 = df["이용개월수_할부전환_R3M"], df["이용개월수_할부전환_R6M"], df["이용개월수_할부전환_R12M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0078(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF (가맹점매출금액_B1M + 가맹점매출금액_B2M) >0: 이용가맹점수 >0
    """
    dd = df[["가맹점매출금액_B1M", "가맹점매출금액_B2M", "이용가맹점수"]]
    ret = dd.apply(lambda x: x[2] > 0 if (x[0] + x[1]) > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0079(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용가맹점수 >= SUM(IF 이용금액_업종*>0 THEN 1 ELSE 0)
    """
    c1 = df["이용가맹점수"]
    dd = df[
        [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ]
    ]
    c2 = dd.apply(lambda x: sum(x > 0), axis=1)
    return c1 >= c2


@constraint_udf
def cc_03_0080(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_쇼핑 + _요식 + _교통 + _의료 + _납부 + _교육 + _여유생활 + _사교활동 + _일상생활 + _해외 <= 이용금액_업종기준
    """
    dd = df[
        [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ]
    ]
    c1 = df["이용금액_업종기준"]
    return c1 >= dd.sum(axis=1)


# 수식 -> 제약조건으로 변경됨
@constraint_udf
def cc_03_0155(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        이용금액_쇼핑 >= SUM(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)
    """
    dd = df[
        [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ]
    ]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_쇼핑"]
    return c >= res


@constraint_udf
def cc_03_0081(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       _3순위업종_이용금액 <= _1순위업종_이용금액
    """
    c1, c2 = df["_3순위업종_이용금액"], df["_1순위업종_이용금액"]
    return c1 <= c2


@constraint_udf
def cc_03_0082(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       _3순위쇼핑업종_이용금액 <= _1순위쇼핑업종_이용금액
    """
    c1, c2 = df["_3순위쇼핑업종_이용금액"], df["_1순위쇼핑업종_이용금액"]
    return c1 <= c2


@constraint_udf
def cc_03_0083(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       _3순위교통업종_이용금액 <= _1순위교통업종_이용금액
    """
    c1, c2 = df["_3순위교통업종_이용금액"], df["_1순위교통업종_이용금액"]
    return c1 <= c2


@constraint_udf
def cc_03_0084(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       _3순위여유업종_이용금액 <= _1순위여유업종_이용금액
    """
    c1, c2 = df["_3순위여유업종_이용금액"], df["_1순위여유업종_이용금액"]
    return c1 <= c2


@constraint_udf
def cc_03_0085(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       _3순위납부업종_이용금액 <= _1순위납부업종_이용금액
    """
    c1, c2 = df["_3순위납부업종_이용금액"], df["_1순위납부업종_이용금액"]
    return c1 <= c2


@constraint_udf
def cc_03_0086(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF RP건수_B0M >0: RP금액_B0M >0
    """
    dd = df[["RP건수_B0M", "RP금액_B0M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0087(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF RP건수_통신_B0M > 0: RP후경과월_통신 = 0 ELSE RP후경과월_통신 >0
    """
    dd = df[["RP건수_통신_B0M", "RP후경과월_통신"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] > 0 else x[1] > 0, axis=1)
    return ret


@constraint_udf
def cc_03_0088(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF RP건수_아파트_B0M > 0: RP후경과월_아파트 = 0 ELSE RP후경과월_아파트 >0
    """
    dd = df[["RP건수_아파트_B0M", "RP후경과월_아파트"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] > 0 else x[1] > 0, axis=1)
    return ret


@constraint_udf
def cc_03_0089(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF RP건수_제휴사서비스직접판매_B0M > 0: RP후경과월_제휴사서비스직접판매 = 0 ELSE RP후경과월_제휴사서비스직접판매 >0
    """
    dd = df[["RP건수_제휴사서비스직접판매_B0M", "RP후경과월_제휴사서비스직접판매"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] > 0 else x[1] > 0, axis=1)
    return ret


@constraint_udf
def cc_03_0090(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF RP건수_렌탈_B0M > 0: RP후경과월_렌탈 = 0 ELSE RP후경과월_렌탈 >0
    """
    dd = df[["RP건수_렌탈_B0M", "RP후경과월_렌탈"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] > 0 else x[1] > 0, axis=1)
    return ret


@constraint_udf
def cc_03_0091(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF RP건수_가스_B0M > 0: RP후경과월_가스 = 0 ELSE RP후경과월_가스 >0
    """
    dd = df[["RP건수_가스_B0M", "RP후경과월_가스"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] > 0 else x[1] > 0, axis=1)
    return ret


@constraint_udf
def cc_03_0092(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF RP건수_전기_B0M > 0: RP후경과월_전기 = 0 ELSE RP후경과월_전기 >0
    """
    dd = df[["RP건수_전기_B0M", "RP후경과월_전기"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] > 0 else x[1] > 0, axis=1)
    return ret


@constraint_udf
def cc_03_0093(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF RP건수_보험_B0M > 0: RP후경과월_보험 = 0 ELSE RP후경과월_보험 >0
    """
    dd = df[["RP건수_보험_B0M", "RP후경과월_보험"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] > 0 else x[1] > 0, axis=1)
    return ret


@constraint_udf
def cc_03_0094(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF RP건수_학습비_B0M > 0: RP후경과월_학습비 = 0 ELSE RP후경과월_학습비 >0
    """
    dd = df[["RP건수_학습비_B0M", "RP후경과월_학습비"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] > 0 else x[1] > 0, axis=1)
    return ret


@constraint_udf
def cc_03_0095(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF RP건수_유선방송_B0M > 0: RP후경과월_유선방송 = 0 ELSE RP후경과월_유선방송 >0
    """
    dd = df[["RP건수_유선방송_B0M", "RP후경과월_유선방송"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] > 0 else x[1] > 0, axis=1)
    return ret


@constraint_udf
def cc_03_0096(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF RP건수_건강_B0M > 0: RP후경과월_건강 = 0 ELSE RP후경과월_건강 >0
    """
    dd = df[["RP건수_건강_B0M", "RP후경과월_건강"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] > 0 else x[1] > 0, axis=1)
    return ret


@constraint_udf
def cc_03_0097(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF RP건수_교통_B0M > 0: RP후경과월_교통 = 0 ELSE RP후경과월_교통 >0
    """
    dd = df[["RP건수_교통_B0M", "RP후경과월_교통"]]
    ret = dd.apply(lambda x: x[1] == 0 if x[0] > 0 else x[1] > 0, axis=1)
    return ret


@constraint_udf
def cc_03_0098(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       최초카드론이용경과월 >= 최종카드론이용경과월
    """
    c1, c2 = df["최초카드론이용경과월"], df["최종카드론이용경과월"]
    return c1 >= c2


@constraint_udf
def cc_03_0099(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_카드론_R12M <= 카드론이용건수_누적
    """
    c1, c2 = df["이용건수_카드론_R12M"], df["카드론이용건수_누적"]
    return c1 <= c2


@constraint_udf
def cc_03_0100(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_카드론_R12M <= 카드론이용월수_누적
    """
    c1, c2 = df["이용개월수_카드론_R12M"], df["카드론이용월수_누적"]
    return c1 <= c2


@constraint_udf
def cc_03_0101(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_카드론_R12M <= 카드론이용금액_누적
    """
    c1, c2 = df["이용금액_카드론_R12M"], df["카드론이용금액_누적"]
    return c1 <= c2


@constraint_udf
def cc_03_0102(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       0 <= 연속무실적개월수_기본_24M_카드 <= 24
    """
    c1 = df["연속무실적개월수_기본_24M_카드"]
    return (0 <= c1) & (c1 <= 24)


@constraint_udf
def cc_03_0103(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       연속유실적개월수_기본_24M_카드 <= 24 - 연속무실적개월수_기본_24M_카드
    """
    c1, c2 = df["연속유실적개월수_기본_24M_카드"], df["연속무실적개월수_기본_24M_카드"]
    return c1 <= 24 - c2


@constraint_udf
def cc_03_0104(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       신청건수_ATM_CA_B0 <= 신청건수_ATM_CA_R6M
    """
    c1, c2 = df["신청건수_ATM_CA_B0"], df["신청건수_ATM_CA_R6M"]
    return c1 <= c2


@constraint_udf
def cc_03_0105(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       신청건수_ATM_CA_B0 <= 이용건수_CA_B0M
    """
    c1, c2 = df["신청건수_ATM_CA_B0"], df["이용건수_CA_B0M"]
    return c1 <= c2


@constraint_udf
def cc_03_0106(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       신청건수_ATM_CL_B0 <= 신청건수_ATM_CL_R6M
    """
    c1, c2 = df["신청건수_ATM_CL_B0"], df["신청건수_ATM_CL_R6M"]
    return c1 <= c2


@constraint_udf
def cc_03_0107(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       신청건수_ATM_CL_B0 <= 이용건수_카드론_B0M
    """
    c1, c2 = df["신청건수_ATM_CL_B0"], df["이용건수_카드론_B0M"]
    return c1 <= c2


@constraint_udf
def cc_03_0108(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_페이_온라인_R6M <= 이용개월수_온라인_R6M
    """
    c1, c2 = df["이용개월수_페이_온라인_R6M"], df["이용개월수_온라인_R6M"]
    return c1 <= c2


@constraint_udf
def cc_03_0109(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_페이_오프라인_R6M <= 이용개월수_오프라인_R6M
    """
    c1, c2 = df["이용개월수_페이_오프라인_R6M"], df["이용개월수_오프라인_R6M"]
    return c1 <= c2


@constraint_udf
def cc_03_0110(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_온라인_B0M <= 이용금액_온라인_R3M <= 이용금액_온라인_R6M
    """
    c1, c2, c3 = df["이용금액_온라인_B0M"], df["이용금액_온라인_R3M"], df["이용금액_온라인_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0111(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_오프라인_B0M <= 이용금액_오프라인_R3M <= 이용금액_오프라인_R6M
    """
    c1, c2, c3 = df["이용금액_오프라인_B0M"], df["이용금액_오프라인_R3M"], df["이용금액_오프라인_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0112(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_온라인_B0M <= 이용건수_온라인_R3M <= 이용건수_온라인_R6M
    """
    c1, c2, c3 = df["이용건수_온라인_B0M"], df["이용건수_온라인_R3M"], df["이용건수_온라인_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0113(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_오프라인_B0M <= 이용건수_오프라인_R3M <= 이용건수_오프라인_R6M
    """
    c1, c2, c3 = df["이용건수_오프라인_B0M"], df["이용건수_오프라인_R3M"], df["이용건수_오프라인_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0114(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_페이_온라인_B0M <= 이용금액_페이_온라인_R3M <= 이용금액_페이_온라인_R6M
    """
    c1, c2, c3 = df["이용금액_페이_온라인_B0M"], df["이용금액_페이_온라인_R3M"], df["이용금액_페이_온라인_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0115(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_페이_오프라인_B0M <= 이용금액_페이_오프라인_R3M <= 이용금액_페이_오프라인_R6M
    """
    c1, c2, c3 = df["이용금액_페이_오프라인_B0M"], df["이용금액_페이_오프라인_R3M"], df["이용금액_페이_오프라인_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0116(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_페이_온라인_B0M <= 이용건수_페이_온라인_R3M <= 이용건수_페이_온라인_R6M
    """
    c1, c2, c3 = df["이용건수_페이_온라인_B0M"], df["이용건수_페이_온라인_R3M"], df["이용건수_페이_온라인_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0117(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_페이_오프라인_B0M <= 이용건수_페이_오프라인_R3M <= 이용건수_페이_오프라인_R6M
    """
    c1, c2, c3 = df["이용건수_페이_오프라인_B0M"], df["이용건수_페이_오프라인_R3M"], df["이용건수_페이_오프라인_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0118(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       MAX(이용개월수_당사페이_R6M, 이용개월수_당사기타_R6M, 이용개월수_A페이_R6M, 이용개월수_B페이_R6M, 이용개월수_C페이_R6M, 이용개월수_D페이_R6M) <= 이용개월수_간편결제_R6M
    """
    dd = df[
        [
            "이용개월수_당사페이_R6M",
            "이용개월수_당사기타_R6M",
            "이용개월수_A페이_R6M",
            "이용개월수_B페이_R6M",
            "이용개월수_C페이_R6M",
            "이용개월수_D페이_R6M",
        ]
    ]
    c_m = dd.max(axis=1)
    c1 = df["이용개월수_간편결제_R6M"]

    return c_m <= c1


@constraint_udf
def cc_03_0119(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       MAX(이용개월수_페이_온라인_R6M, 이용개월수_페이_오프라인_R6M) <= 이용개월수_간편결제_R6M
    """
    dd = df[["이용개월수_페이_온라인_R6M", "이용개월수_페이_오프라인_R6M"]]
    c_m = dd.max(axis=1)
    c1 = df["이용개월수_간편결제_R6M"]
    return c_m <= c1


@constraint_udf
def cc_03_0120(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_간편결제_B0M <= 이용금액_간편결제_R3M <= 이용금액_간편결제_R6M
    """
    c1, c2, c3 = df["이용금액_간편결제_B0M"], df["이용금액_간편결제_R3M"], df["이용금액_간편결제_R6M"]
    return (c1 <= c2) & (c2 <= c3)


### 수식 -> 제약조건으로 convert
@constraint_udf
def cc_03_0149(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_간편결제_R6M >= SUM(이용금액_당사페이_R6M, 당사기타, A페이, B페이, C페이, D페이)
    """
    dd = df[
        [
            "이용금액_당사페이_R6M",
            "이용금액_당사기타_R6M",
            "이용금액_A페이_R6M",
            "이용금액_B페이_R6M",
            "이용금액_C페이_R6M",
            "이용금액_D페이_R6M",
        ]
    ]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_간편결제_R6M"]
    return c >= res


@constraint_udf
def cc_03_0121(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_당사페이_B0M <= 이용금액_당사페이_R3M <= 이용금액_당사페이_R6M
    """
    c1, c2, c3 = df["이용금액_당사페이_B0M"], df["이용금액_당사페이_R3M"], df["이용금액_당사페이_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0122(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_당사기타_B0M <=이용금액_당사기타_R3M <= 이용금액_당사기타_R6M
    """
    c1, c2, c3 = df["이용금액_당사기타_B0M"], df["이용금액_당사기타_R3M"], df["이용금액_당사기타_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0123(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_A페이_B0M <=이용금액_A페이_R3M <= 이용금액_A페이_R6M
    """
    c1, c2, c3 = df["이용금액_A페이_B0M"], df["이용금액_A페이_R3M"], df["이용금액_A페이_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0124(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_B페이_B0M <=이용금액_B페이_R3M <= 이용금액_B페이_R6M
    """
    c1, c2, c3 = df["이용금액_B페이_B0M"], df["이용금액_B페이_R3M"], df["이용금액_B페이_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0125(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_C페이_B0M <=이용금액_C페이_R3M <= 이용금액_C페이_R6M
    """
    c1, c2, c3 = df["이용금액_C페이_B0M"], df["이용금액_C페이_R3M"], df["이용금액_C페이_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0126(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_D페이_B0M <=이용금액_D페이_R3M <= 이용금액_D페이_R6M
    """
    c1, c2, c3 = df["이용금액_D페이_B0M"], df["이용금액_D페이_R3M"], df["이용금액_D페이_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0127(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_간편결제_B0M <= 이용건수_간편결제_R3M <= 이용건수_간편결제_R6M
    """
    c1, c2, c3 = df["이용건수_간편결제_B0M"], df["이용건수_간편결제_R3M"], df["이용건수_간편결제_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0150(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_간편결제_R6M >= SUM(이용건수_당사페이_R6M, 당사기타, A페이, B페이, C페이, D페이)
    """
    dd = df[
        [
            "이용건수_당사페이_R6M",
            "이용건수_당사기타_R6M",
            "이용건수_A페이_R6M",
            "이용건수_B페이_R6M",
            "이용건수_C페이_R6M",
            "이용건수_D페이_R6M",
        ]
    ]
    res = dd.sum(axis=1).astype(int)

    c = df["이용건수_간편결제_R6M"]
    return c >= res


@constraint_udf
def cc_03_0128(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_당사페이_B0M <= 이용건수_당사페이_R3M <= 이용건수_당사페이_R6M
    """
    c1, c2, c3 = df["이용건수_당사페이_B0M"], df["이용건수_당사페이_R3M"], df["이용건수_당사페이_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0129(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_당사기타_B0M <=이용건수_당사기타_R3M <= 이용건수_당사기타_R6M
    """
    c1, c2, c3 = df["이용건수_당사기타_B0M"], df["이용건수_당사기타_R3M"], df["이용건수_당사기타_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0130(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_A페이_B0M <=이용건수_A페이_R3M <= 이용건수_A페이_R6M
    """
    c1, c2, c3 = df["이용건수_A페이_B0M"], df["이용건수_A페이_R3M"], df["이용건수_A페이_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0131(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_B페이_B0M <=이용건수_B페이_R3M <= 이용건수_B페이_R6M
    """
    c1, c2, c3 = df["이용건수_B페이_B0M"], df["이용건수_B페이_R3M"], df["이용건수_B페이_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0132(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_C페이_B0M <=이용건수_C페이_R3M <= 이용건수_C페이_R6M
    """
    c1, c2, c3 = df["이용건수_C페이_B0M"], df["이용건수_C페이_R3M"], df["이용건수_C페이_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0133(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_D페이_B0M <=이용건수_D페이_R3M <= 이용건수_D페이_R6M
    """
    c1, c2, c3 = df["이용건수_D페이_B0M"], df["이용건수_D페이_R3M"], df["이용건수_D페이_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0151(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_간편결제_R3M >= SUM(이용금액_당사페이_R3M, 당사기타, A페이, B페이, C페이, D페이)
    """
    dd = df[
        [
            "이용금액_당사페이_R3M",
            "이용금액_당사기타_R3M",
            "이용금액_A페이_R3M",
            "이용금액_B페이_R3M",
            "이용금액_C페이_R3M",
            "이용금액_D페이_R3M",
        ]
    ]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_간편결제_R3M"]
    return c >= res


@constraint_udf
def cc_03_0152(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_간편결제_R3M >= SUM(이용건수_당사페이_R3M, 당사기타, A페이, B페이, C페이, D페이)
    """
    dd = df[
        [
            "이용건수_당사페이_R3M",
            "이용건수_당사기타_R3M",
            "이용건수_A페이_R3M",
            "이용건수_B페이_R3M",
            "이용건수_C페이_R3M",
            "이용건수_D페이_R3M",
        ]
    ]
    res = dd.sum(axis=1).astype(int)

    c = df["이용건수_간편결제_R3M"]
    return c >= res


@constraint_udf
def cc_03_0153(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_간편결제_B0M >= SUM(이용금액_당사페이_B0M, 당사기타, A페이, B페이, C페이, D페이)
    """
    dd = df[
        [
            "이용금액_당사페이_B0M",
            "이용금액_당사기타_B0M",
            "이용금액_A페이_B0M",
            "이용금액_B페이_B0M",
            "이용금액_C페이_B0M",
            "이용금액_D페이_B0M",
        ]
    ]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_간편결제_B0M"]
    return c >= res


@constraint_udf
def cc_03_0154(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_간편결제_B0M >= SUM(이용건수_당사페이_B0M, 당사기타, A페이, B페이, C페이, D페이)
    """
    dd = df[
        [
            "이용건수_당사페이_B0M",
            "이용건수_당사기타_B0M",
            "이용건수_A페이_B0M",
            "이용건수_B페이_B0M",
            "이용건수_C페이_B0M",
            "이용건수_D페이_B0M",
        ]
    ]
    res = dd.sum(axis=1).astype(int)

    c = df["이용건수_간편결제_B0M"]
    return c >= res


@constraint_udf
def cc_03_0134(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       IF 이용횟수_선결제_R6M >0: 이용개월수_선결제_R6M >0
    """
    dd = df[["이용횟수_선결제_R6M", "이용개월수_선결제_R6M"]]
    ret = dd.apply(lambda x: x[1] > 0 if x[0] > 0 else True, axis=1)
    return ret


@constraint_udf
def cc_03_0135(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용횟수_선결제_B0M <= 이용횟수_선결제_R3M <= 이용횟수_선결제_R6M
    """
    c1, c2, c3 = df["이용횟수_선결제_B0M"], df["이용횟수_선결제_R3M"], df["이용횟수_선결제_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0136(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_선결제_B0M <= 이용금액_선결제_R3M <= 이용금액_선결제_R6M
    """
    c1, c2, c3 = df["이용금액_선결제_B0M"], df["이용금액_선결제_R3M"], df["이용금액_선결제_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0137(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용건수_선결제_B0M <= 이용건수_선결제_R3M <= 이용건수_선결제_R6M
    """
    c1, c2, c3 = df["이용건수_선결제_B0M"], df["이용건수_선결제_R3M"], df["이용건수_선결제_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0141(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_전체_R3M <= 이용개월수_전체_R6M
    """
    c1, c2 = df["이용개월수_전체_R3M"], df["이용개월수_전체_R6M"]
    return c1 <= c2


@constraint_udf
def cc_03_0142(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       MAX(이용개월수_신용_R6M, 이용개월수_카드론_R6M) <= 이용개월수_전체_R6M
    """
    dd = df[["이용개월수_신용_R6M", "이용개월수_카드론_R6M"]]
    c_m = dd.max(axis=1)
    c1 = df["이용개월수_전체_R6M"]
    return c_m <= c1


@constraint_udf
def cc_03_0143(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       MAX(이용개월수_신용_R3M, 이용개월수_카드론_R3M) <= 이용개월수_전체_R3M
    """
    dd = df[["이용개월수_신용_R3M", "이용개월수_카드론_R3M"]]
    c_m = dd.max(axis=1)
    c1 = df["이용개월수_전체_R3M"]
    return c_m <= c1


@constraint_udf
def cc_03_0144(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용개월수_결제일_R3M <= 이용개월수_결제일_R6M
    """
    c1, c2 = df["이용개월수_결제일_R3M"], df["이용개월수_결제일_R6M"]
    return c1 <= c2


@constraint_udf
def cc_03_0145(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용횟수_연체_B0M <= 이용횟수_연체_R3M <= 이용횟수_연체_R6M
    """
    c1, c2, c3 = df["이용횟수_연체_B0M"], df["이용횟수_연체_R3M"], df["이용횟수_연체_R6M"]
    return (c1 <= c2) & (c2 <= c3)


@constraint_udf
def cc_03_0146(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       이용금액_연체_B0M <= 이용금액_연체_R3M <= 이용금액_연체_R6M
    """
    c1, c2, c3 = df["이용금액_연체_B0M"], df["이용금액_연체_R3M"], df["이용금액_연체_R6M"]
    return (c1 <= c2) & (c2 <= c3)


## 추가됨 마지막 번호 달아줌
@constraint_udf
def cc_03_0147(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
        MONTHS_BETWEEN(최종이용일자_할부, 기준년월) == 이용후경과월_할부
    """
    dd = df[["기준년월", "최종이용일자_할부"]]
    c = df["이용후경과월_할부"]
    tmp_res = dd.apply(
        lambda x: relativedelta(
            datetime(year=int(x[0][:4]), month=int(x[0][4:6]), day=1)
            + relativedelta(months=1, days=-1),
            datetime.strptime(x[1], "%Y%m%d"),
        )
        if (not pd.isna(x[1])) & (x[1] != "10101")
        else 12,
        axis=1,
    )
    res = tmp_res.apply(lambda x: 12 if x == 12 else min(x.years * 12 + x.months, 12))
    return c == res


# 수식 -> 제약조건으로 변경/추가됨
@constraint_udf
def cc_03_0148(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       SUM(IF RP건수_*_B0M > 0 THEN 1 ELSE 0) <= RP유형건수_B0M <= RP건수_B0M
    """
    c1, c2 = df["RP유형건수_B0M"], df["RP건수_B0M"]
    dd = df[
        [
            "RP건수_통신_B0M",
            "RP건수_아파트_B0M",
            "RP건수_제휴사서비스직접판매_B0M",
            "RP건수_렌탈_B0M",
            "RP건수_가스_B0M",
            "RP건수_전기_B0M",
            "RP건수_보험_B0M",
            "RP건수_학습비_B0M",
            "RP건수_유선방송_B0M",
            "RP건수_건강_B0M",
            "RP건수_교통_B0M",
        ]
    ]
    c0 = dd.apply(lambda x: sum(x > 0), axis=1)
    return (c0 <= c1) & (c1 <= c2)


@constraint_udf
def cc_03_0158(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       승인거절건수_한도초과_B0M <= 승인거절건수_한도초과_R3M
    """
    c1, c2 = df["승인거절건수_한도초과_B0M"], df["승인거절건수_한도초과_R3M"]
    return c1 <= c2


@constraint_udf
def cc_03_0159(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       승인거절건수_BL_B0M <= 승인거절건수_BL_R3M
    """
    c1, c2 = df["승인거절건수_BL_B0M"], df["승인거절건수_BL_R3M"]
    return c1 <= c2


@constraint_udf
def cc_03_0160(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       승인거절건수_입력오류_B0M <= 승인거절건수_입력오류_R3M
    """
    c1, c2 = df["승인거절건수_입력오류_B0M"], df["승인거절건수_입력오류_R3M"]
    return c1 <= c2


@constraint_udf
def cc_03_0161(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    Constraint:
       승인거절건수_기타_B0M <= 승인거절건수_기타_R3M
    """
    c1, c2 = df["승인거절건수_기타_B0M"], df["승인거절건수_기타_R3M"]
    return c1 <= c2


@constraint_udf
def cf_03_0006(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        최종이용일자_기본 = MAX(최종이용일자_신판, 최종이용일자_CA, 최종이용일자_카드론)
    """
    dd = df[["최종이용일자_신판", "최종이용일자_CA", "최종이용일자_카드론"]]
    res = dd.max(axis=1).astype(int).astype(str)

    c = df["최종이용일자_기본"]
    return c == res


@constraint_udf
def cf_03_0007(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        최종이용일자_신판 = MAX(최종이용일자_일시불, 최종이용일자_할부)
    """
    dd = df[["최종이용일자_일시불", "최종이용일자_할부"]]
    res = dd.max(axis=1).astype(int).astype(str)

    c = df["최종이용일자_신판"]
    return c == res


@constraint_udf
def cf_03_0013(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_신용_B0M = 이용건수_신판_B0M + 이용건수_CA_B0M
    """
    c1, c2 = df["이용건수_신판_B0M"], df["이용건수_CA_B0M"]
    res = c1 + c2

    c = df["이용건수_신용_B0M"]
    return c == res


@constraint_udf
def cf_03_0014(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_신판_B0M = 이용건수_일시불_B0M + 이용건수_할부_B0M
    """
    c1, c2 = df["이용건수_일시불_B0M"], df["이용건수_할부_B0M"]
    res = c1 + c2

    c = df["이용건수_신판_B0M"]
    return c == res


@constraint_udf
def cf_03_0016(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_할부_B0M = 이용건수_할부_유이자_B0M + 이용건수_할부_무이자_B0M + 이용건수_부분무이자_B0M
    """
    c1, c2, c3 = df["이용건수_할부_유이자_B0M"], df["이용건수_할부_무이자_B0M"], df["이용건수_부분무이자_B0M"]
    res = c1 + c2 + c3

    c = df["이용건수_할부_B0M"]
    return c == res


@constraint_udf
def cf_03_0023(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_신용_B0M = 이용금액_신판_B0M + 이용금액_CA_B0M
    """
    c1, c2 = df["이용금액_신판_B0M"], df["이용금액_CA_B0M"]
    res = c1 + c2

    c = df["이용금액_신용_B0M"]
    return c == res


@constraint_udf
def cf_03_0024(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_신판_B0M = 이용금액_일시불_B0M + 이용금액_할부_B0M
    """
    c1, c2 = df["이용금액_일시불_B0M"], df["이용금액_할부_B0M"]
    res = c1 + c2

    c = df["이용금액_신판_B0M"]
    return c == res


@constraint_udf
def cf_03_0026(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_할부_B0M = 이용금액_할부_유이자_B0M + 이용금액_할부_무이자_B0M + 이용금액_부분무이자_B0M
    """
    c1, c2, c3 = df["이용금액_할부_유이자_B0M"], df["이용금액_할부_무이자_B0M"], df["이용금액_부분무이자_B0M"]
    res = c1 + c2 + c3

    c = df["이용금액_할부_B0M"]
    return c == res


@constraint_udf
def cf_03_0033(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용후경과월_신용 = MIN(이용후경과월_신판, 이용후경과월_CA)
    """
    dd = df[["이용후경과월_신판", "이용후경과월_CA"]]
    res = dd.min(axis=1).astype(int)

    c = df["이용후경과월_신용"]
    return c == res


@constraint_udf
def cf_03_0034(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용후경과월_신판 = MIN(이용후경과월_일시불, 이용후경과월_할부)
    """
    dd = df[["이용후경과월_일시불", "이용후경과월_할부"]]
    res = dd.min(axis=1).astype(int)

    c = df["이용후경과월_신판"]
    return c == res


@constraint_udf
def cf_03_0035(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용후경과월_일시불 = MONTHS_BETWEEN(LAST_DAY(기준년월), 최종이용일자_일시불)
    """
    dd = df[["기준년월", "최종이용일자_일시불"]]
    tmp_res = dd.apply(
        lambda x: relativedelta(
            datetime(year=int(x[0][:4]), month=int(x[0][4:6]), day=1)
            + relativedelta(months=1, days=-1),
            datetime.strptime(x[1], "%Y%m%d"),
        )
        if (not pd.isna(x[1])) * (x[1] != "10101")
        else 999,
        axis=1,
    )
    res = tmp_res.apply(lambda x: 12 if x == 999 else x.years * 12 + x.months)
    res = res.apply(lambda x: 12 if x > 12 else x)

    c = df["이용후경과월_일시불"]
    return c == res


@constraint_udf
def cf_03_0036(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       이용후경과월_할부 = MIN(이용후경과월_할부_유이자, 이용후경과월_할부_무이자, 이용후경과월_부분무이자)
    """
    dd = df[["이용후경과월_할부_유이자", "이용후경과월_할부_무이자", "이용후경과월_부분무이자"]]
    res = dd.min(axis=1).astype(int)

    c = df["이용후경과월_할부"]
    return c == res


@constraint_udf
def cf_03_0040(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용후경과월_CA = MONTHS_BETWEEN(LAST_DAY(기준년월), 최종이용일자_CA)
    """
    dd = df[["기준년월", "최종이용일자_CA"]]
    tmp_res = dd.apply(
        lambda x: relativedelta(
            datetime(year=int(x[0][:4]), month=int(x[0][4:6]), day=1)
            + relativedelta(months=1, days=-1),
            datetime.strptime(x[1], "%Y%m%d"),
        )
        if (not pd.isna(x[1])) * (x[1] != "10101")
        else 999,
        axis=1,
    )
    res = tmp_res.apply(lambda x: 12 if x == 999 else x.years * 12 + x.months)
    res = res.apply(lambda x: 12 if x > 12 else x)

    c = df["이용후경과월_CA"]
    return c == res


@constraint_udf
def cf_03_0041(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용후경과월_체크 = MONTHS_BETWEEN(LAST_DAY(기준년월), 최종이용일자_체크)
    """
    dd = df[["기준년월", "최종이용일자_체크"]]
    tmp_res = dd.apply(
        lambda x: relativedelta(
            datetime(year=int(x[0][:4]), month=int(x[0][4:6]), day=1)
            + relativedelta(months=1, days=-1),
            datetime.strptime(x[1], "%Y%m%d"),
        )
        if (not pd.isna(x[1])) * (x[1] != "10101")
        else 999,
        axis=1,
    )
    res = tmp_res.apply(lambda x: 12 if x == 999 else x.years * 12 + x.months)
    res = res.apply(lambda x: 12 if x > 12 else x)

    c = df["이용후경과월_체크"]
    return c == res


@constraint_udf
def cf_03_0042(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용후경과월_카드론 = MONTHS_BETWEEN(LAST_DAY(기준년월), 최종이용일자_카드론)
    """
    dd = df[["기준년월", "최종이용일자_카드론"]]
    tmp_res = dd.apply(
        lambda x: relativedelta(
            datetime(year=int(x[0][:4]), month=int(x[0][4:6]), day=1)
            + relativedelta(months=1, days=-1),
            datetime.strptime(x[1], "%Y%m%d"),
        )
        if (not pd.isna(x[1])) * (x[1] != "10101")
        else 999,
        axis=1,
    )
    res = tmp_res.apply(lambda x: 12 if x == 999 else x.years * 12 + x.months)
    res = res.apply(lambda x: 12 if x > 12 else x)

    c = df["이용후경과월_카드론"]
    return c == res


@constraint_udf
def cf_03_0043(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_신용_R12M = SUM(이용건수_신판_R12M, 이용건수_CA_R12M)
    """
    c1, c2 = df["이용건수_신판_R12M"], df["이용건수_CA_R12M"]
    res = c1 + c2

    c = df["이용건수_신용_R12M"]
    return c == res


@constraint_udf
def cf_03_0044(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       이용건수_신판_R12M = SUM(이용건수_일시불_R12M, 이용건수_할부_R12M)
    """
    c1, c2 = df["이용건수_일시불_R12M"], df["이용건수_할부_R12M"]
    res = c1 + c2

    c = df["이용건수_신판_R12M"]
    return c == res


@constraint_udf
def cf_03_0046(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       이용건수_할부_R12M = SUM(이용건수_할부_유이자_R12M, 이용건수_할부_무이자_R12M, 이용건수_부분무이자_R12M)
    """
    c1, c2, c3 = df["이용건수_할부_유이자_R12M"], df["이용건수_할부_무이자_R12M"], df["이용건수_부분무이자_R12M"]
    res = c1 + c2 + c3

    c = df["이용건수_할부_R12M"]
    return c == res


@constraint_udf
def cf_03_0047(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       이용건수_할부_유이자_R12M = SUM(할부건수_유이자_3M_R12M, 할부건수_유이자_6M_R12M, 할부건수_유이자_12M_R12M, 할부건수_유이자_14M_R12M)
    """
    c1, c2, c3, c4 = (
        df["할부건수_유이자_3M_R12M"],
        df["할부건수_유이자_6M_R12M"],
        df["할부건수_유이자_12M_R12M"],
        df["할부건수_유이자_14M_R12M"],
    )
    res = c1 + c2 + c3 + c4

    c = df["이용건수_할부_유이자_R12M"]
    return c == res


@constraint_udf
def cf_03_0048(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       이용건수_할부_무이자_R12M = SUM(할부건수_무이자_3M_R12M, 할부건수_무이자_6M_R12M, 할부건수_무이자_12M_R12M, 할부건수_무이자_14M_R12M)
    """
    c1, c2, c3, c4 = (
        df["할부건수_무이자_3M_R12M"],
        df["할부건수_무이자_6M_R12M"],
        df["할부건수_무이자_12M_R12M"],
        df["할부건수_무이자_14M_R12M"],
    )
    res = c1 + c2 + c3 + c4

    c = df["이용건수_할부_무이자_R12M"]
    return c == res


@constraint_udf
def cf_03_0049(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       이용건수_부분무이자_R12M = SUM(할부건수_부분_3M_R12M, 할부건수_부분_6M_R12M, 할부건수_부분_12M_R12M, 할부건수_부분_14M_R12M)
    """
    c1, c2, c3, c4 = (
        df["할부건수_부분_3M_R12M"],
        df["할부건수_부분_6M_R12M"],
        df["할부건수_부분_12M_R12M"],
        df["할부건수_부분_14M_R12M"],
    )
    res = c1 + c2 + c3 + c4

    c = df["이용건수_부분무이자_R12M"]
    return c == res


@constraint_udf
def cf_03_0053(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       이용금액_신용_R12M = SUM(이용금액_신판_R12M, 이용금액_CA_R12M)
    """
    c1, c2 = df["이용금액_신판_R12M"], df["이용금액_CA_R12M"]
    res = c1 + c2

    c = df["이용금액_신용_R12M"]
    return c == res


@constraint_udf
def cf_03_0054(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       이용금액_신판_R12M = SUM(이용금액_일시불_R12M, 이용금액_할부_R12M)
    """
    c1, c2 = df["이용금액_일시불_R12M"], df["이용금액_할부_R12M"]
    res = c1 + c2

    c = df["이용금액_신판_R12M"]
    return c == res


@constraint_udf
def cf_03_0056(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       이용금액_할부_R12M = SUM(이용금액_할부_유이자_R12M, 이용금액_할부_무이자_R12M, 이용금액_부분무이자_R12M)
    """
    c1, c2, c3 = df["이용금액_할부_유이자_R12M"], df["이용금액_할부_무이자_R12M"], df["이용금액_부분무이자_R12M"]
    res = c1 + c2 + c3

    c = df["이용금액_할부_R12M"]
    return c == res


@constraint_udf
def cf_03_0057(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       이용금액_할부_유이자_R12M = SUM(할부금액_유이자_3M_R12M, 할부금액_유이자_6M_R12M, 할부금액_유이자_12M_R12M, 할부금액_유이자_14M_R12M)
    """
    c1, c2, c3, c4 = (
        df["할부금액_유이자_3M_R12M"],
        df["할부금액_유이자_6M_R12M"],
        df["할부금액_유이자_12M_R12M"],
        df["할부금액_유이자_14M_R12M"],
    )
    res = c1 + c2 + c3 + c4

    c = df["이용금액_할부_유이자_R12M"]
    return c == res


@constraint_udf
def cf_03_0058(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       이용금액_할부_무이자_R12M = SUM(할부금액_무이자_3M_R12M, 할부금액_무이자_6M_R12M, 할부금액_무이자_12M_R12M, 할부금액_무이자_14M_R12M)
    """
    c1, c2, c3, c4 = (
        df["할부금액_무이자_3M_R12M"],
        df["할부금액_무이자_6M_R12M"],
        df["할부금액_무이자_12M_R12M"],
        df["할부금액_무이자_14M_R12M"],
    )
    res = c1 + c2 + c3 + c4

    c = df["이용금액_할부_무이자_R12M"]
    return c == res


@constraint_udf
def cf_03_0059(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
       이용금액_부분무이자_R12M = SUM(할부금액_부분_3M_R12M, 할부금액_부분_6M_R12M, 할부금액_부분_12M_R12M, 할부금액_부분_14M_R12M) =
    """
    c1, c2, c3, c4 = (
        df["할부금액_부분_3M_R12M"],
        df["할부금액_부분_6M_R12M"],
        df["할부금액_부분_12M_R12M"],
        df["할부금액_부분_14M_R12M"],
    )
    res = c1 + c2 + c3 + c4

    c = df["이용금액_부분무이자_R12M"]
    return c == res


@constraint_udf
def cf_03_0066(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        최대이용금액_할부_R12M = MAX(최대이용금액_할부_유이자_R12M, 할부_무이자, 부분무이자)
    """
    dd = df[["최대이용금액_할부_유이자_R12M", "최대이용금액_할부_무이자_R12M", "최대이용금액_부분무이자_R12M"]]
    res = dd.max(axis=1).astype(int)

    c = df["최대이용금액_할부_R12M"]
    return c == res


@constraint_udf
def cf_03_0083(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_신용_R6M = SUM(이용건수_신판_R6M, 이용건수_CA_R6M)
    """
    dd = df[["이용건수_신판_R6M", "이용건수_CA_R6M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["이용건수_신용_R6M"]
    return c == res


@constraint_udf
def cf_03_0084(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_신판_R6M = SUM(이용건수_일시불_R6M, 이용건수_할부_R6M)
    """
    dd = df[["이용건수_일시불_R6M", "이용건수_할부_R6M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["이용건수_신판_R6M"]
    return c == res


@constraint_udf
def cf_03_0086(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_할부_R6M = SUM(이용건수_할부_유이자_R6M, 이용건수_할부_무이자_R6M, 이용건수_부분무이자_R6M)
    """
    dd = df[["이용건수_할부_유이자_R6M", "이용건수_할부_무이자_R6M", "이용건수_부분무이자_R6M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["이용건수_할부_R6M"]
    return c == res


@constraint_udf
def cf_03_0093(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_신용_R6M = SUM(이용금액_신판_R6M, 이용금액_CA_R6M)
    """
    dd = df[["이용금액_신판_R6M", "이용금액_CA_R6M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_신용_R6M"]
    return c == res


@constraint_udf
def cf_03_0094(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_신판_R6M = SUM(이용금액_일시불_R6M, 이용금액_할부_R6M)
    """
    dd = df[["이용금액_일시불_R6M", "이용금액_할부_R6M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_신판_R6M"]
    return c == res


@constraint_udf
def cf_03_0096(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_할부_R6M = SUM(이용금액_할부_유이자_R6M, 이용금액_할부_무이자_R6M, 이용금액_부분무이자_R6M)
    """
    dd = df[["이용금액_할부_유이자_R6M", "이용금액_할부_무이자_R6M", "이용금액_부분무이자_R6M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_할부_R6M"]
    return c == res


@constraint_udf
def cf_03_0113(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_신용_R3M = SUM(이용건수_신판_R3M, 이용건수_CA_R3M)
    """
    dd = df[["이용건수_신판_R3M", "이용건수_CA_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["이용건수_신용_R3M"]
    return c == res


@constraint_udf
def cf_03_0114(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_신판_R3M = SUM(이용건수_일시불_R3M, 이용건수_할부_R3M)
    """
    dd = df[["이용건수_일시불_R3M", "이용건수_할부_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["이용건수_신판_R3M"]
    return c == res


@constraint_udf
def cf_03_0116(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_할부_R3M = SUM(이용건수_할부_유이자_R3M, 이용건수_할부_무이자_R3M, 이용건수_부분무이자_R3M)
    """
    dd = df[["이용건수_할부_유이자_R3M", "이용건수_할부_무이자_R3M", "이용건수_부분무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["이용건수_할부_R3M"]
    return c == res


@constraint_udf
def cf_03_0123(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_신용_R3M = SUM(이용금액_신판_R3M, 이용금액_CA_R3M)
    """
    dd = df[["이용금액_신판_R3M", "이용금액_CA_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_신용_R3M"]
    return c == res


@constraint_udf
def cf_03_0124(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_신판_R3M = SUM(이용금액_일시불_R3M, 이용금액_할부_R3M)
    """
    dd = df[["이용금액_일시불_R3M", "이용금액_할부_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_신판_R3M"]
    return c == res


@constraint_udf
def cf_03_0126(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_할부_R3M = SUM(이용금액_할부_유이자_R3M, 이용금액_할부_무이자_R3M, 이용금액_부분무이자_R3M)
    """
    dd = df[["이용금액_할부_유이자_R3M", "이용금액_할부_무이자_R3M", "이용금액_부분무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_할부_R3M"]
    return c == res


@constraint_udf
def cf_03_0160(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_교통 = SUM(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)
    """
    dd = df[
        [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ]
    ]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_교통"]
    return c == res


@constraint_udf
def cf_03_0162(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_납부 = SUM(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)
    """
    dd = df[
        [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ]
    ]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_납부"]
    return c == res


@constraint_udf
def cf_03_0164(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_여유생활 = SUM(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)
    """
    dd = df[
        [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ]
    ]
    res = dd.sum(axis=1).astype(int)

    c = df["이용금액_여유생활"]
    return c == res


@constraint_udf
def cf_03_0176(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_쇼핑 = 쇼핑_전체_이용금액
    """
    res = df["쇼핑_전체_이용금액"]
    c = df["이용금액_쇼핑"]
    return c == res


@constraint_udf
def cf_03_0183(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        교통_전체이용금액 = 이용금액_교통
    """
    res = df["이용금액_교통"]
    c = df["교통_전체이용금액"]
    return c == res


@constraint_udf
def cf_03_0192(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        여유_전체이용금액 = 이용금액_여유생활
    """
    res = df["이용금액_여유생활"]
    c = df["여유_전체이용금액"]
    return c == res


@constraint_udf
def cf_03_0195(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        납부_렌탈료이용금액 = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["납부_렌탈료이용금액"]
    return c == res


@constraint_udf
def cf_03_0201(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        납부_전체이용금액 = 이용금액_납부
    """
    res = df["이용금액_납부"]
    c = df["납부_전체이용금액"]
    return c == res


@constraint_udf
def cf_03_0202(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _1순위업종 = ARGMAX(이용금액_쇼핑, 요식, 교통, 의료, 납부, 교육, 여유생활, 사교활동, 일상생활, 해외)
    """
    dd = df[
        [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ]
    ]
    dd["_max"] = dd.max(axis=1).astype(int)

    code_map = {
        0: "쇼핑",
        1: "요식",
        2: "교통",
        3: "의료",
        4: "납부",
        5: "교육",
        6: "여유생활",
        7: "사교활동",
        8: "일상생활",
        9: "해외",
    }
    tmp_res = dd.apply(
        lambda x: np.where(x[:-1] == x[-1])[0][0] if x[-1] > 0 else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_1순위업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == 'nan' else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0203(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _1순위업종_이용금액 = MAX(이용금액_쇼핑, 요식, 교통, 의료, 납부, 교육, 여유생활, 사교활동, 일상생활, 해외)
    """
    dd = df[
        [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ]
    ]
    res = dd.max(axis=1).astype(int)

    c = df["_1순위업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0204(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _3순위업종 = ARG3rd(이용금액_쇼핑, 요식, 교통, 의료, 납부, 교육, 여유생활, 사교활동, 일상생활, 해외)
    """
    dd = df[
        [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ]
    ]

    # 3순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 2, axis=1)

    code_map = {
        0: "쇼핑",
        1: "요식",
        2: "교통",
        3: "의료",
        4: "납부",
        5: "교육",
        6: "여유생활",
        7: "사교활동",
        8: "일상생활",
        9: "해외",
    }
    tmp_res = dd.apply(
        lambda x: np.argsort(x[:-1])[-3] if x[-1] else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_3순위업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == 'nan' else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0205(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _3순위업종_이용금액 = 3rd(이용금액_쇼핑, 요식, 교통, 의료, 납부, 교육, 여유생활, 사교활동, 일상생활, 해외)
    """
    dd = df[
        [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ]
    ]

    # 3순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 2, axis=1)
    res = dd.apply(lambda x: np.sort(x[:-1])[-3] if x[-1] else 0, axis=1)

    c = df["_3순위업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0206(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _1순위쇼핑업종 = ARGMAX(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)
    """
    dd = df[
        [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ]
    ]
    dd["_max"] = dd.max(axis=1).astype(int)

    code_map = {
        0: "도소매",
        1: "백화점",
        2: "마트",
        3: "슈퍼마켓",
        4: "편의점",
        5: "아울렛",
        6: "온라인",
        7: "쇼핑기타",
    }
    tmp_res = dd.apply(
        lambda x: np.where(x[:-1] == x[-1])[0][0] if x[-1] > 0 else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_1순위쇼핑업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == 'nan' else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0207(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _1순위쇼핑업종_이용금액 = MAX(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)
    """
    dd = df[
        [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ]
    ]
    res = dd.max(axis=1).astype(int)

    c = df["_1순위쇼핑업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0208(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _3순위쇼핑업종 = ARG3rd(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)
    """
    dd = df[
        [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ]
    ]

    # 3순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 2, axis=1)

    code_map = {
        0: "도소매",
        1: "백화점",
        2: "마트",
        3: "슈퍼마켓",
        4: "편의점",
        5: "아울렛",
        6: "온라인",
        7: "쇼핑기타",
    }
    tmp_res = dd.apply(
        lambda x: np.argsort(x[:-1])[-3] if x[-1] else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_3순위쇼핑업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == "nan" else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0209(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _3순위쇼핑업종_이용금액 = 3rd(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)
    """
    dd = df[
        [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ]
    ]

    # 3순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 2, axis=1)
    res = dd.apply(lambda x: np.sort(x[:-1])[-3] if x[-1] else 0, axis=1)

    c = df["_3순위쇼핑업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0210(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _1순위교통업종 = ARGMAX(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)
    """
    dd = df[
        [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ]
    ]
    dd["_max"] = dd.max(axis=1).astype(int)

    code_map = {0: "주유", 1: "정비", 2: "통행료", 3: "버스지하철", 4: "택시", 5: "철도버스"}
    tmp_res = dd.apply(
        lambda x: np.where(x[:-1] == x[-1])[0][0] if x[-1] > 0 else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_1순위교통업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == "nan" else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0211(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _1순위교통업종_이용금액 = MAX(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)
    """
    dd = df[
        [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ]
    ]
    res = dd.max(axis=1).astype(int)

    c = df["_1순위교통업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0212(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _3순위교통업종 = ARG3rd(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)
    """
    dd = df[
        [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ]
    ]

    # 3순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 2, axis=1)

    code_map = {0: "주유", 1: "정비", 2: "통행료", 3: "버스지하철", 4: "택시", 5: "철도버스"}
    tmp_res = dd.apply(
        lambda x: np.argsort(x[:-1])[-3] if x[-1] else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_3순위교통업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == "nan" else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0213(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _3순위교통업종_이용금액 = 3rd(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)
    """
    dd = df[
        [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ]
    ]

    # 3순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 2, axis=1)
    res = dd.apply(lambda x: np.sort(x[:-1])[-3] if x[-1] else 0, axis=1)

    c = df["_3순위교통업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0214(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _1순위여유업종 = ARGMAX(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)
    """
    dd = df[
        [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ]
    ]
    dd["_max"] = dd.max(axis=1).astype(int)

    code_map = {
        0: "운동",
        1: "Pet",
        2: "공연",
        3: "공원",
        4: "숙박",
        5: "여행",
        6: "항공",
        7: "여유기타",
    }
    tmp_res = dd.apply(
        lambda x: np.where(x[:-1] == x[-1])[0][0] if x[-1] > 0 else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_1순위여유업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == "nan" else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0215(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _1순위여유업종_이용금액 = MAX(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)
    """
    dd = df[
        [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ]
    ]
    res = dd.max(axis=1).astype(int)

    c = df["_1순위여유업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0216(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _3순위여유업종 = ARG3rd(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)
    """
    dd = df[
        [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ]
    ]

    # 3순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 2, axis=1)

    code_map = {
        0: "운동",
        1: "Pet",
        2: "공연",
        3: "공원",
        4: "숙박",
        5: "여행",
        6: "항공",
        7: "여유기타",
    }
    tmp_res = dd.apply(
        lambda x: np.argsort(x[:-1])[-3] if x[-1] else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_3순위여유업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == "nan" else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0217(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _3순위여유업종_이용금액 = 3rd(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)
    """
    dd = df[
        [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ]
    ]

    # 3순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 2, axis=1)

    res = dd.apply(lambda x: np.sort(x[:-1])[-3] if x[-1] else 0, axis=1)

    c = df["_3순위여유업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0218(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _1순위납부업종 = ARGMAX(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)
    """
    dd = df[
        [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ]
    ]
    dd["_max"] = dd.max(axis=1).astype(int)

    code_map = {
        0: "통신비",
        1: "관리비",
        2: "렌탈료",
        3: "가스/전기료",
        4: "보험료",
        5: "유선료",
        6: "건강/연금",
        7: "납부기타",
    }
    tmp_res = dd.apply(
        lambda x: np.where(x[:-1] == x[-1])[0][0] if x[-1] > 0 else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_1순위납부업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == "nan" else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0219(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _1순위납부업종_이용금액 = MAX(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)
    """
    dd = df[
        [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ]
    ]
    res = dd.max(axis=1).astype(int)

    c = df["_1순위납부업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0220(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _3순위납부업종 = ARG3rd(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)
    """
    dd = df[
        [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ]
    ]

    # 3순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 2, axis=1)

    code_map = {
        0: "통신비",
        1: "관리비",
        2: "렌탈료",
        3: "가스/전기료",
        4: "보험료",
        5: "유선료",
        6: "건강/연금",
        7: "납부기타",
    }
    tmp_res = dd.apply(
        lambda x: np.argsort(x[:-1])[-3] if x[-1] else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_3순위납부업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == "nan" else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0221(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _3순위납부업종_이용금액 = 3rd(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)
    """
    dd = df[
        [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ]
    ]

    # 3순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 2, axis=1)
    res = dd.apply(lambda x: np.sort(x[:-1])[-3] if x[-1] else 0, axis=1)

    c = df["_3순위납부업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0222(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        할부건수_3M_R12M = SUM(할부건수_유이자_3M_R12M, 할부건수_무이자_3M_R12M, 할부건수_부분_3M_R12M)
    """
    dd = df[["할부건수_유이자_3M_R12M", "할부건수_무이자_3M_R12M", "할부건수_부분_3M_R12M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["할부건수_3M_R12M"]
    return c == res


@constraint_udf
def cf_03_0223(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        할부건수_6M_R12M = SUM(할부건수_유이자_6M_R12M, 할부건수_무이자_6M_R12M, 할부건수_부분_6M_R12M)
    """
    dd = df[["할부건수_유이자_6M_R12M", "할부건수_무이자_6M_R12M", "할부건수_부분_6M_R12M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["할부건수_6M_R12M"]
    return c == res


@constraint_udf
def cf_03_0224(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        할부건수_12M_R12M = SUM(할부건수_유이자_12M_R12M, 할부건수_무이자_12M_R12M, 할부건수_부분_12M_R12M)
    """
    dd = df[["할부건수_유이자_12M_R12M", "할부건수_무이자_12M_R12M", "할부건수_부분_12M_R12M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["할부건수_12M_R12M"]
    return c == res


@constraint_udf
def cf_03_0225(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        할부건수_14M_R12M = SUM(할부건수_유이자_14M_R12M, 할부건수_무이자_14M_R12M, 할부건수_부분_14M_R12M)
    """
    dd = df[["할부건수_유이자_14M_R12M", "할부건수_무이자_14M_R12M", "할부건수_부분_14M_R12M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["할부건수_14M_R12M"]
    return c == res


@constraint_udf
def cf_03_0226(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        할부금액_3M_R12M = SUM(할부금액_유이자_3M_R12M, 할부금액_무이자_3M_R12M, 할부금액_부분_3M_R12M)
    """
    dd = df[["할부금액_유이자_3M_R12M", "할부금액_무이자_3M_R12M", "할부금액_부분_3M_R12M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["할부금액_3M_R12M"]
    return c == res


@constraint_udf
def cf_03_0227(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        할부금액_6M_R12M = SUM(할부금액_유이자_6M_R12M, 할부금액_무이자_6M_R12M, 할부금액_부분_6M_R12M)
    """
    dd = df[["할부금액_유이자_6M_R12M", "할부금액_무이자_6M_R12M", "할부금액_부분_6M_R12M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["할부금액_6M_R12M"]
    return c == res


@constraint_udf
def cf_03_0228(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        할부금액_12M_R12M = SUM(할부금액_유이자_12M_R12M, 할부금액_무이자_12M_R12M, 할부금액_부분_12M_R12M)
    """
    dd = df[["할부금액_유이자_12M_R12M", "할부금액_무이자_12M_R12M", "할부금액_부분_12M_R12M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["할부금액_12M_R12M"]
    return c == res


@constraint_udf
def cf_03_0229(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        할부금액_14M_R12M = SUM(할부금액_유이자_14M_R12M, 할부금액_무이자_14M_R12M, 할부금액_부분_14M_R12M)
    """
    dd = df[["할부금액_유이자_14M_R12M", "할부금액_무이자_14M_R12M", "할부금액_부분_14M_R12M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["할부금액_14M_R12M"]
    return c == res


@constraint_udf
def cf_03_0246(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        할부건수_부분_3M_R12M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["할부건수_부분_3M_R12M"]
    return c == res


@constraint_udf
def cf_03_0250(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        할부금액_부분_3M_R12M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["할부금액_부분_3M_R12M"]
    return c == res


@constraint_udf
def cf_03_0254(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        RP건수_B0M = SUM(RP건수_통신_B0M, 아파트, 제휴사서비스직접판매, 렌탈, 가스, 전기, 보험, 학습비, 유선방송, 건강, 교통)
    """
    dd = df[
        [
            "RP건수_통신_B0M",
            "RP건수_아파트_B0M",
            "RP건수_제휴사서비스직접판매_B0M",
            "RP건수_렌탈_B0M",
            "RP건수_가스_B0M",
            "RP건수_전기_B0M",
            "RP건수_보험_B0M",
            "RP건수_학습비_B0M",
            "RP건수_유선방송_B0M",
            "RP건수_건강_B0M",
            "RP건수_교통_B0M",
        ]
    ]
    res = dd.sum(axis=1).astype(int)

    c = df["RP건수_B0M"]
    return c == res


@constraint_udf
def cf_03_0268(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        RP후경과월 = MIN(RP후경과월_통신, 아파트, 제휴사서비스직접판매, 렌탈, 가스, 전기, 보험, 학습비, 유선방송, 건강, 교통)
    """
    dd = df[
        [
            "RP후경과월_통신",
            "RP후경과월_아파트",
            "RP후경과월_제휴사서비스직접판매",
            "RP후경과월_렌탈",
            "RP후경과월_가스",
            "RP후경과월_전기",
            "RP후경과월_보험",
            "RP후경과월_학습비",
            "RP후경과월_유선방송",
            "RP후경과월_건강",
            "RP후경과월_교통",
        ]
    ]
    res = dd.min(axis=1).astype(int)

    c = df["RP후경과월"]
    return c == res


@constraint_udf
def cf_03_0281(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        최종카드론이용경과월 = MONTHS_BETWEEN(LAST_DAY(기준년월), 최종이용일자_카드론)
    """
    dd = df[["기준년월", "최종이용일자_카드론"]]
    tmp_res = dd.apply(
        lambda x: relativedelta(
            datetime(year=int(x[0][:4]), month=int(x[0][4:6]), day=1)
            + relativedelta(months=1, days=-1),
            datetime.strptime(x[1], "%Y%m%d"),
        )
        if (not pd.isna(x[1])) * (x[1] != "10101")
        else 999,
        axis=1,
    )
    res = tmp_res.apply(lambda x: 999 if x == 999 else x.years * 12 + x.months)

    c = df["최종카드론이용경과월"]
    return c == res


@constraint_udf
def cf_03_0289(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        최종카드론_대출일자 == 최종이용일자_카드론
    """
    dd = df[["최종카드론_대출일자", "최종이용일자_카드론"]]
    res = dd.apply(lambda x: x[0] == x[1] if (not isNaN(x[1])) & (not x[1] == '10101')
                   else isNaN(x[0]), axis=1)
    return res


@constraint_udf
def cf_03_0338(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용개월수_당사페이_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용개월수_당사페이_R6M"]
    return c == res


@constraint_udf
def cf_03_0345(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_당사페이_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용금액_당사페이_R6M"]
    return c == res


@constraint_udf
def cf_03_0346(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_당사기타_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용금액_당사기타_R6M"]
    return c == res


@constraint_udf
def cf_03_0352(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_당사페이_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용건수_당사페이_R6M"]
    return c == res


@constraint_udf
def cf_03_0353(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_당사기타_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용건수_당사기타_R6M"]
    return c == res


@constraint_udf
def cf_03_0359(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_당사페이_R3M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용금액_당사페이_R3M"]
    return c == res


@constraint_udf
def cf_03_0360(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_당사기타_R3M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용금액_당사기타_R3M"]
    return c == res


@constraint_udf
def cf_03_0366(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_당사페이_R3M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용건수_당사페이_R3M"]
    return c == res


@constraint_udf
def cf_03_0367(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_당사기타_R3M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용건수_당사기타_R3M"]
    return c == res


@constraint_udf
def cf_03_0373(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_당사페이_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용금액_당사페이_B0M"]
    return c == res


@constraint_udf
def cf_03_0374(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용금액_당사기타_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용금액_당사기타_B0M"]
    return c == res


@constraint_udf
def cf_03_0380(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_당사페이_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용건수_당사페이_B0M"]
    return c == res


@constraint_udf
def cf_03_0381(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        이용건수_당사기타_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))

    c = df["이용건수_당사기타_B0M"]
    return c == res


# @constraint_udf
# def cf_03_0344(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         이용금액_간편결제_R6M = SUM(이용금액_당사페이_R6M, 당사기타, A페이, B페이, C페이, D페이)
#     """
#     dd = df[["이용금액_당사페이_R6M", "이용금액_당사기타_R6M", "이용금액_A페이_R6M", "이용금액_B페이_R6M", \
#              "이용금액_C페이_R6M", "이용금액_D페이_R6M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df['이용금액_간편결제_R6M']
#     return c == res

# @constraint_udf
# def cf_03_0351(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         이용건수_간편결제_R6M = SUM(이용건수_당사페이_R6M, 당사기타, A페이, B페이, C페이, D페이)
#     """
#     dd = df[["이용건수_당사페이_R6M", "이용건수_당사기타_R6M", "이용건수_A페이_R6M", "이용건수_B페이_R6M", \
#                     "이용건수_C페이_R6M", "이용건수_D페이_R6M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df['이용건수_간편결제_R6M']
#     return c == res

# @constraint_udf
# def cf_03_0358(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         이용금액_간편결제_R3M = SUM(이용금액_당사페이_R3M, 당사기타, A페이, B페이, C페이, D페이)
#     """
#     dd = df[["이용금액_당사페이_R3M", "이용금액_당사기타_R3M", "이용금액_A페이_R3M", "이용금액_B페이_R3M", \
#              "이용금액_C페이_R3M", "이용금액_D페이_R3M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df['이용금액_간편결제_R3M']
#     return c == res

# @constraint_udf
# def cf_03_0365(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         이용건수_간편결제_R3M = SUM(이용건수_당사페이_R3M, 당사기타, A페이, B페이, C페이, D페이)
#     """
#     dd = df[["이용건수_당사페이_R3M", "이용건수_당사기타_R3M", "이용건수_A페이_R3M", "이용건수_B페이_R3M", \
#                     "이용건수_C페이_R3M", "이용건수_D페이_R3M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df['이용건수_간편결제_R3M']
#     return c == res

# @constraint_udf
# def cf_03_0372(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         이용금액_간편결제_B0M = SUM(이용금액_당사페이_B0M, 당사기타, A페이, B페이, C페이, D페이)
#     """
#     dd = df[["이용금액_당사페이_B0M", "이용금액_당사기타_B0M", "이용금액_A페이_B0M", "이용금액_B페이_B0M", \
#              "이용금액_C페이_B0M", "이용금액_D페이_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df['이용금액_간편결제_B0M']
#     return c == res

# @constraint_udf
# def cf_03_0379(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
#     """
#     formula:
#         이용건수_간편결제_B0M = SUM(이용건수_당사페이_B0M, 당사기타, A페이, B페이, C페이, D페이)
#     """
#     dd = df[["이용건수_당사페이_B0M", "이용건수_당사기타_B0M", "이용건수_A페이_B0M", "이용건수_B페이_B0M", \
#                     "이용건수_C페이_B0M", "이용건수_D페이_B0M"]]
#     res = dd.sum(axis=1).astype(int)

#     c = df['이용건수_간편결제_B0M']
#     return c == res


@constraint_udf
def cf_03_0408(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        연체입금원금_B0M = 정상청구원금_B0M - (선입금원금_B0M + 정상입금원금_B0M)
    """
    c1, c2, c3 = df["정상청구원금_B0M"], df["선입금원금_B0M"], df["정상입금원금_B0M"]
    res = c1 - (c2 + c3)

    c = df["연체입금원금_B0M"]
    return c == res


@constraint_udf
def cf_03_0409(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        연체입금원금_B2M = 정상청구원금_B2M - (선입금원금_B2M + 정상입금원금_B2M)
    """
    c1, c2, c3 = df["정상청구원금_B2M"], df["선입금원금_B2M"], df["정상입금원금_B2M"]
    res = c1 - (c2 + c3)

    c = df["연체입금원금_B2M"]
    return c == res


@constraint_udf
def cf_03_0410(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        연체입금원금_B5M = 정상청구원금_B5M - (선입금원금_B5M + 정상입금원금_B5M)
    """
    c1, c2, c3 = df["정상청구원금_B5M"], df["선입금원금_B5M"], df["정상입금원금_B5M"]
    res = c1 - (c2 + c3)

    c = df["연체입금원금_B5M"]
    return c == res


@constraint_udf
def cf_03_0425(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _2순위업종 = ARG2nd(이용금액_쇼핑, 요식, 교통, 의료, 납부, 교육, 여유생활, 사교활동, 일상생활, 해외)
    """
    dd = df[
        [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ]
    ]

    # 2순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 1, axis=1)

    code_map = {
        0: "쇼핑",
        1: "요식",
        2: "교통",
        3: "의료",
        4: "납부",
        5: "교육",
        6: "여유생활",
        7: "사교활동",
        8: "일상생활",
        9: "해외",
    }
    tmp_res = dd.apply(
        lambda x: np.argsort(x[:-1])[-2] if x[-1] else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_2순위업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == "nan" else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0426(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _2순위업종_이용금액 = 2nd(이용금액_쇼핑, 요식, 교통, 의료, 납부, 교육, 여유생활, 사교활동, 일상생활, 해외)
    """
    dd = df[
        [
            "이용금액_쇼핑",
            "이용금액_요식",
            "이용금액_교통",
            "이용금액_의료",
            "이용금액_납부",
            "이용금액_교육",
            "이용금액_여유생활",
            "이용금액_사교활동",
            "이용금액_일상생활",
            "이용금액_해외",
        ]
    ]

    # 2순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 1, axis=1)
    res = dd.apply(lambda x: np.sort(x[:-1])[-2] if x[-1] else 0, axis=1)

    c = df["_2순위업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0427(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _2순위쇼핑업종 = ARG2nd(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)
    """
    dd = df[
        [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ]
    ]

    # 2순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 1, axis=1)

    code_map = {
        0: "도소매",
        1: "백화점",
        2: "마트",
        3: "슈퍼마켓",
        4: "편의점",
        5: "아울렛",
        6: "온라인",
        7: "쇼핑기타",
    }
    tmp_res = dd.apply(
        lambda x: np.argsort(x[:-1])[-2] if x[-1] else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_2순위쇼핑업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == "nan" else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0428(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _2순위쇼핑업종_이용금액 = 2nd(쇼핑_도소매_이용금액, 백화점, 마트, 슈퍼마켓, 편의점, 아울렛, 온라인, 기타)
    """
    dd = df[
        [
            "쇼핑_도소매_이용금액",
            "쇼핑_백화점_이용금액",
            "쇼핑_마트_이용금액",
            "쇼핑_슈퍼마켓_이용금액",
            "쇼핑_편의점_이용금액",
            "쇼핑_아울렛_이용금액",
            "쇼핑_온라인_이용금액",
            "쇼핑_기타_이용금액",
        ]
    ]

    # 2순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 1, axis=1)
    res = dd.apply(lambda x: np.sort(x[:-1])[-2] if x[-1] else 0, axis=1)

    c = df["_2순위쇼핑업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0429(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _2순위교통업종 = ARG2nd(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)
    """
    dd = df[
        [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ]
    ]

    # 2순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 1, axis=1)

    code_map = {0: "주유", 1: "정비", 2: "통행료", 3: "버스지하철", 4: "택시", 5: "철도버스"}
    tmp_res = dd.apply(
        lambda x: np.argsort(x[:-1])[-2] if x[-1] else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_2순위교통업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == "nan" else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0430(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _2순위교통업종_이용금액 = 2nd(교통_주유이용금액, 정비, 통행료, 버스지하철, 택시, 철도버스)
    """
    dd = df[
        [
            "교통_주유이용금액",
            "교통_정비이용금액",
            "교통_통행료이용금액",
            "교통_버스지하철이용금액",
            "교통_택시이용금액",
            "교통_철도버스이용금액",
        ]
    ]

    # 2순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 1, axis=1)
    res = dd.apply(lambda x: np.sort(x[:-1])[-2] if x[-1] else 0, axis=1)

    c = df["_2순위교통업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0431(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _2순위여유업종 = ARG2nd(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)
    """
    dd = df[
        [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ]
    ]

    # 2순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 1, axis=1)

    code_map = {
        0: "운동",
        1: "Pet",
        2: "공연",
        3: "공원",
        4: "숙박",
        5: "여행",
        6: "항공",
        7: "여유기타",
    }
    tmp_res = dd.apply(
        lambda x: np.argsort(x[:-1])[-2] if x[-1] else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_2순위여유업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == "nan" else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0432(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _2순위여유업종_이용금액 = 2nd(여유_운동이용금액, Pet, 공연, 공원, 숙박, 여행, 항공, 기타)
    """
    dd = df[
        [
            "여유_운동이용금액",
            "여유_Pet이용금액",
            "여유_공연이용금액",
            "여유_공원이용금액",
            "여유_숙박이용금액",
            "여유_여행이용금액",
            "여유_항공이용금액",
            "여유_기타이용금액",
        ]
    ]

    # 2순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 1, axis=1)

    res = dd.apply(lambda x: np.sort(x[:-1])[-2] if x[-1] else 0, axis=1)

    c = df["_2순위여유업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0433(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _2순위납부업종 = ARG2nd(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)
    """
    dd = df[
        [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ]
    ]

    # 2순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 1, axis=1)

    code_map = {
        0: "통신비",
        1: "관리비",
        2: "렌탈료",
        3: "가스/전기료",
        4: "보험료",
        5: "유선료",
        6: "건강/연금",
        7: "납부기타",
    }
    tmp_res = dd.apply(
        lambda x: np.argsort(x[:-1])[-2] if x[-1] else "nan", axis=1
    ).replace(code_map)

    res_df = tmp_res.to_frame()
    res_df.columns = ['res']
    res_df['y'] = df["_2순위납부업종"]
    res = res_df.apply(lambda x: isNaN(x[1]) if x[0] == "nan" else x[0] == x[1], axis=1)
    return res


@constraint_udf
def cf_03_0434(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        _2순위납부업종_이용금액 = 2rd(납부_통신비이용금액, 관리비, 렌탈료, 가스전기료, 보험료, 유선방송, 건강연금, 기타)
    """
    dd = df[
        [
            "납부_통신비이용금액",
            "납부_관리비이용금액",
            "납부_렌탈료이용금액",
            "납부_가스전기료이용금액",
            "납부_보험료이용금액",
            "납부_유선방송이용금액",
            "납부_건강연금이용금액",
            "납부_기타이용금액",
        ]
    ]

    # 2순위 업종 존재여부
    dd["is_valid"] = dd.apply(lambda x: sum(x > 0) > 1, axis=1)
    res = dd.apply(lambda x: np.sort(x[:-1])[-2] if x[-1] else 0, axis=1)

    c = df["_2순위납부업종_이용금액"]
    return c == res


@constraint_udf
def cf_03_0462(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        승인거절건수_B0M = SUM(승인거절건수_한도초과_B0M, BL_B0M, 입력오류_B0M, 기타_B0M)
    """
    dd = df[["승인거절건수_한도초과_B0M", "승인거절건수_BL_B0M", "승인거절건수_입력오류_B0M", "승인거절건수_기타_B0M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["승인거절건수_B0M"]
    return c == res


@constraint_udf
def cf_03_0467(df: pd.DataFrame) -> Union[pd.Series, List[bool]]:
    """
    formula:
        승인거절건수_R3M = SUM(승인거절건수_한도초과_R3M, BL_R3M, 입력오류_R3M, 기타_R3M)
    """
    dd = df[["승인거절건수_한도초과_R3M", "승인거절건수_BL_R3M", "승인거절건수_입력오류_R3M", "승인거절건수_기타_R3M"]]
    res = dd.sum(axis=1).astype(int)

    c = df["승인거절건수_R3M"]
    return c == res


# cc_01_0001(df)
