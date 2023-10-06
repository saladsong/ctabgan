# 수식으로 합성 가능한 파생 컬럼 생성용
# (v.1) 1개월치 데이터 내에서의 파생 관계만 고려되어 있음
# (v.1.5) validity 검증 후 수식>조건 수정 및 중복 컬럼 반영
# for each formula fx,
# I: 전체 데이터프레임 (CTAB-GAN+ 알고리즘 수행 결과 생성되는 데이터)
# O: 파생 컬럼

import pandas as pd
import numpy as np
from datetime import datetimewm
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
    # {
    #     "columns": ["한도요청거절건수", "한도요청승인건수"],
    #     "output": "한도심사요청건수",
    #     "fname": "cf_02_0040",
    #     "type": "formula",
    #     "content": "한도심사요청건수 = 한도요청거절건수 + 한도요청승인건수",
    # },
    # {
    #     "columns": ["기준년월", "rv최초시작일자"],
    #     "output": "rv최초시작후경과일",
    #     "fname": "cf_02_0060",
    #     "type": "formula",
    #     "content": "rv최초시작후경과일 = DATEDIFF(LAST_DAY(기준년월), rv최초시작일자)",
    # },

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
    # 8.성과 테이블 컬럼 Formula
    {
        "columns": ["증감율_이용건수_CA_전월", "증감율_이용건수_신판_전월"],
        "output": "증감율_이용건수_신용_전월",
        "fname": "cf_08_0005",
        "type": "cond_formula",
        "content": """IF 증감율_이용건수_CA_전월 == 0 THEN 증감율_이용건수_신용_전월 = 증감율_이용건수_신판_전월
                       ELIF 증감율_이용건수_신판_전월 == 0 THEN 증감율_이용건수_신용_전월 = 증감율_이용건수_CA_전월
                       ELSE PASS""",
    },
    {
        "columns": ["증감율_이용건수_할부_전월", "증감율_이용건수_일시불_전월"],
        "output": "증감율_이용건수_신판_전월",
        "fname": "cf_08_0006",
        "type": "cond_formula",
        "content": """IF 증감율_이용건수_할부_전월 == 0 THEN 증감율_이용건수_신판_전월 = 증감율_이용건수_일시불_전월
                      ELIF 증감율_이용건수_일시불_전월 == 0 THEN 증감율_이용건수_신판_전월 = 증감율_이용건수_할부_전월
                      ELSE PASS""",
    },
    {
        "columns": ["증감율_이용금액_CA_전월", "증감율_이용금액_신판_전월"],
        "output": "증감율_이용금액_신용_전월",
        "fname": "cf_08_0012",
        "type": "cond_formula",
        "content": """IF 증감율_이용금액_CA_전월 == 0 THEN 증감율_이용금액_신용_전월 = 증감율_이용금액_신판_전월
                       ELIF 증감율_이용금액_신판_전월 == 0 THEN 증감율_이용금액_신용_전월 = 증감율_이용금액_CA_전월
                       ELSE PASS""",
    },
    {
        "columns": ["증감율_이용금액_할부_전월", "증감율_이용금액_일시불_전월"],
        "output": "증감율_이용금액_신판_전월",
        "fname": "cf_08_0013",
        "type": "cond_formula",
        "content": """IF 증감율_이용금액_할부_전월 == 0 THEN 증감율_이용금액_신판_전월 = 증감율_이용금액_일시불_전월
                      ELIF 증감율_이용금액_일시불_전월 == 0 THEN 증감율_이용금액_신판_전월 = 증감율_이용금액_할부_전월
                      ELSE PASS""",
    },
    {
        "columns": ["증감율_이용건수_CA_분기", "증감율_이용건수_신판_분기"],
        "output": "증감율_이용건수_신용_분기",
        "fname": "cf_08_0033",
        "type": "cond_formula",
        "content": """IF 증감율_이용건수_CA_분기 == 0 THEN 증감율_이용건수_신용_분기 = 증감율_이용건수_신판_분기
                       ELIF 증감율_이용건수_신판_분기 == 0 THEN 증감율_이용건수_신용_분기 = 증감율_이용건수_CA_분기
                       ELSE PASS""",
    },
    {
        "columns": ["증감율_이용건수_할부_분기", "증감율_이용건수_일시불_분기"],
        "output": "증감율_이용건수_신판_분기",
        "fname": "cf_08_0034",
        "type": "cond_formula",
        "content": """IF 증감율_이용건수_할부_분기 == 0 THEN 증감율_이용건수_신판_분기 = 증감율_이용건수_일시불_분기
                      ELIF 증감율_이용건수_일시불_분기 == 0 THEN 증감율_이용건수_신판_분기 = 증감율_이용건수_할부_분기
                      ELSE PASS""",
    },
    {
        "columns": ["증감율_이용금액_CA_분기", "증감율_이용금액_신판_분기"],
        "output": "증감율_이용금액_신용_분기",
        "fname": "cf_08_0040",
        "type": "cond_formula",
        "content": """IF 증감율_이용금액_CA_분기 == 0 THEN 증감율_이용금액_신용_분기 = 증감율_이용금액_신판_분기
                       ELIF 증감율_이용금액_신판_분기 == 0 THEN 증감율_이용금액_신용_분기 = 증감율_이용금액_CA_분기
                       ELSE PASS""",
    },
    {
        "columns": ["증감율_이용금액_할부_분기", "증감율_이용금액_일시불_분기"],
        "output": "증감율_이용금액_신판_분기",
        "fname": "cf_08_0041",
        "type": "cond_formula",
        "content": """IF 증감율_이용금액_할부_분기 == 0 THEN 증감율_이용금액_신판_분기 = 증감율_이용금액_일시불_분기
                      ELIF 증감율_이용금액_일시불_분기 == 0 THEN 증감율_이용금액_신판_분기 = 증감율_이용금액_할부_분기
                      ELSE PASS""",
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
    # {
    #     "columns": ["이용후경과월_신판", "이용후경과월_CA"],
    #     "output": "이용후경과월_신용",
    #     "fname": "cf_03_0033",
    #     "type": "formula",
    #     "content": "이용후경과월_신용 = MIN(이용후경과월_신판, 이용후경과월_CA)",
    # },
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
        "columns": ["이용개월수_B페이_R6M"],
        "output": "이용건수_B페이_R6M",
        "fname": "cf_03_0355",
        "type": "formula",
        "content": "이용건수_B페이_R6M = 이용개월수_B페이_R6M",
    },
    {
        "columns": ["이용개월수_C페이_R6M"],
        "output": "이용건수_C페이_R6M",
        "fname": "cf_03_0356",
        "type": "formula",
        "content": "이용건수_C페이_R6M = 이용개월수_C페이_R6M",
    },
    {
        "columns": ["이용개월수_D페이_R6M"],
        "output": "이용건수_D페이_R6M",
        "fname": "cf_03_0357",
        "type": "formula",
        "content": "이용건수_D페이_R6M = 이용개월수_D페이_R6M",
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
def cf_01_0018(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_01_0023(df: pd.DataFrame) -> Union[pd.Series, List[str]]:
    """
    formula:
        회원여부_연체 = CASE WHEN `이용횟수_연체_B0M` > 0 THEN '1' ELSE '0'
    """
    dd = df[["이용횟수_연체_B0M"]]
    res = dd.apply(lambda x: "1" if x[0] > 0 else "0", axis=1)
    return res


@constraint_udf
def cf_01_0039(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        유효카드수_신용체크 = 유효카드수_신용 + 유효카드수_체크
    """
    c1, c2 = df["유효카드수_신용"], df["유효카드수_체크"]
    res = c1 + c2
    return res


@constraint_udf
def cf_01_0044(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용가능카드수_신용체크 = 이용가능카드수_신용 + 이용가능카드수_체크
    """
    c1, c2 = df["이용가능카드수_신용"], df["이용가능카드수_체크"]
    res = c1 + c2
    return res


@constraint_udf
def cf_01_0049(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용카드수_신용체크 = 이용카드수_신용 + 이용카드수_체크
    """
    c1, c2 = df["이용카드수_신용"], df["이용카드수_체크"]
    res = c1 + c2
    return res


@constraint_udf
def cf_01_0054(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_R3M_신용체크 = 이용금액_R3M_신용 + 이용금액_R3M_체크
    """
    c1, c2 = df["이용금액_R3M_신용"], df["이용금액_R3M_체크"]
    res = c1 + c2
    return res


@constraint_udf
def cf_01_0083(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        기본연회비_B0M = 할인금액_기본연회비_B0M+청구금액_기본연회비_B0M
    """
    c1, c2 = df["할인금액_기본연회비_B0M"], df["청구금액_기본연회비_B0M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_01_0084(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        제휴연회비_B0M = 할인금액_제휴연회비_B0M+청구금액_제휴연회비_B0M
    """
    c1, c2 = df["할인금액_제휴연회비_B0M"], df["청구금액_제휴연회비_B0M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_01_0092(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        기타면제카드수_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


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
    return res


@constraint_udf
def cf_02_0030(df: pd.DataFrame) -> Union[pd.Series, List[str]]:
    """
    formula:
        IF 이용거절여부_카드론=='1' THEN 카드론동의여부='N' ELSE 카드론동의여부='Y'
    """
    dd = df[["이용거절여부_카드론"]]
    res = dd.apply(lambda x: "N" if x[0] == "1" else "Y", axis=1)
    return res


# @constraint_udf
# def cf_02_0038(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
#     """
#     formula:
#         IF RV신청일자 IS NOT NULL THEN rv최초시작일자=RV신청일자 ELSE rv최초시작일자 IS NULL
#     """
#     dd = df[["RV신청일자"]]
#     res = dd.apply(lambda x: x[0] if not pd.isna(x[0]) else "nan", axis=1)
#     return res


# @constraint_udf
# def cf_02_0039(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
#     """
#     formula:
#         IF RV신청일자 IS NOT NULL THEN rv등록일자=RV신청일자 ELSE rv등록일자 IS NULL
#     """
#     dd = df[["RV신청일자"]]
#     res = dd.apply(lambda x: x[0] if not pd.isna(x[0]) else "nan", axis=1)
#     return res


# @constraint_udf
# def cf_02_0040(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
#     """
#     formula:
#         한도심사요청건수 = 한도요청거절건수 + 한도요청승인건수
#     """
#     c1, c2 = df["한도요청거절건수"], df["한도요청승인건수"]
#     res = c1 + c2
#     return res


# @constraint_udf
# def cf_02_0060(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
#     """
#     formula:
#         rv최초시작후경과일 = DATEDIFF(LAST_DAY(기준년월), rv최초시작일자)
#     """
#     dd = df[["기준년월", "rv최초시작일자"]]
#     res = dd.apply(
#         lambda x: (datetime(year=int(x[0][:4]), month=int(x[0][4:6]), day=1) + relativedelta(months=1, days=-1) - datetime.strptime(x[1], "%Y%m%d")).days
#         if (not pd.isna(x[1])) & (x[1] != '10101')
#         else 99999999,
#         axis=1,
#     )
#     return res


@constraint_udf
def cf_04_0008(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        대표결제방법코드 = 2
    """
    c1 = df["기준년월"]
    res = pd.Series(['2'] * len(c1))
    return res


@constraint_udf
def cf_04_0011(df: pd.DataFrame) -> Union[pd.Series, List[str]]:
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
    return res


@constraint_udf
def cf_04_0012(df: pd.DataFrame) -> Union[pd.Series, List[str]]:
    """
    formula:
        IF 청구금액_B0 > 0 THEN 청구서발송여부_B0 = '1' ELSE '0'
    """
    dd = df[["청구금액_B0"]]
    res = dd.apply(lambda x: "1" if x[0] > 0 else "0", axis=1)
    return res


@constraint_udf
def cf_04_0027(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        포인트_적립포인트_R3M = 포인트_포인트_건별_R3M + 포인트_포인트_월적립_R3M
    """
    c1, c2 = df["포인트_포인트_건별_R3M"], df["포인트_포인트_월적립_R3M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_04_0032(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        마일_적립포인트_R3M = 포인트_마일리지_건별_R3M + 포인트_마일리지_월적립_R3M
    """
    c1, c2 = df["포인트_마일리지_건별_R3M"], df["포인트_마일리지_월적립_R3M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_05_0006(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        잔액_B0M = SUM(잔액_일시불_B0M, 할부, 현금서비스, 리볼빙일시불이월, 리볼빙CA이월, 카드론)
    """
    c1, c2, c3 = df["잔액_일시불_B0M"], df["잔액_할부_B0M"], df["잔액_현금서비스_B0M"]
    c4, c5, c6 = df["잔액_리볼빙일시불이월_B0M"], df["잔액_리볼빙CA이월_B0M"], df["잔액_카드론_B0M"]
    res = c1 + c2 + c3 + c4 + c5 + c6
    return res


@constraint_udf
def cf_05_0008(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        잔액_할부_B0M = SUM(잔액_할부_유이자_B0M, 잔액_할부_무이자_B0M)
    """
    c1, c2 = df["잔액_할부_유이자_B0M"], df["잔액_할부_무이자_B0M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_05_0018(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        연체잔액_B0M = SUM(연체잔액_일시불_B0M, 할부, 현금서비스, 카드론, 대환론)
    """
    c1, c2, c3 = df["연체잔액_일시불_B0M"], df["연체잔액_할부_B0M"], df["연체잔액_현금서비스_B0M"]
    c4, c5 = df["연체잔액_카드론_B0M"], df["연체잔액_대환론_B0M"]
    res = c1 + c2 + c3 + c4 + c5
    return res


@constraint_udf
def cf_06_0066(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        IB상담건수_VOC_B0M = SUM(IB상담건수_VOC민원_B0M, IB상담건수_VOC불만_B0M)
    """
    c1, c2 = df["IB상담건수_VOC민원_B0M"], df["IB상담건수_VOC불만_B0M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_06_0089(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        IB상담건수_VOC_R6M = SUM(IB상담건수_VOC민원_R6M, IB상담건수_VOC불만_R6M)
    """
    c1, c2 = df["IB상담건수_VOC민원_R6M"], df["IB상담건수_VOC불만_R6M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_06_0096(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        당사PAY_방문횟수_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_06_0097(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        당사PAY_방문횟수_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_06_0098(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        당사PAY_방문월수_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0019(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_CA_EM_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0020(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_리볼빙_EM_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0024(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_리볼빙_청구서_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0026(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_카드론_인터넷_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0027(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_CA_인터넷_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0028(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_리볼빙_인터넷_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0032(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_리볼빙_당사앱_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0047(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_CA_EM_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0048(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_리볼빙_EM_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0052(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_리볼빙_청구서_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0054(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_카드론_인터넷_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0055(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_CA_인터넷_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0056(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_리볼빙_인터넷_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_07_0060(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        컨택건수_리볼빙_당사앱_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_08_0005(df: pd.DataFrame) -> Union[pd.Series, List[float]]:
    """
    formula:
        IF 증감율_이용건수_CA_전월 == 0 THEN 증감율_이용건수_신용_전월 = 증감율_이용건수_신판_전월
        ELIF 증감율_이용건수_신판_전월 == 0 THEN 증감율_이용건수_신용_전월 = 증감율_이용건수_CA_전월
        ELSE PASS
    """
    dd = df[["증감율_이용건수_CA_전월", "증감율_이용건수_신판_전월", "증감율_이용건수_신용_전월"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)
    return res


@constraint_udf
def cf_08_0006(df: pd.DataFrame) -> Union[pd.Series, List[float]]:
    """
    formula:
        IF 증감율_이용건수_할부_전월 == 0 THEN 증감율_이용건수_신판_전월 = 증감율_이용건수_일시불_전월
        ELIF 증감율_이용건수_일시불_전월 == 0 THEN 증감율_이용건수_신판_전월 = 증감율_이용건수_할부_전월
        ELSE PASS
    """
    dd = df[["증감율_이용건수_할부_전월", "증감율_이용건수_일시불_전월", "증감율_이용건수_신판_전월"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)
    return res


@constraint_udf
def cf_08_0012(df: pd.DataFrame) -> Union[pd.Series, List[float]]:
    """
    formula:
        IF 증감율_이용금액_CA_전월 == 0 THEN 증감율_이용금액_신용_전월 = 증감율_이용금액_신판_전월
        ELIF 증감율_이용금액_신판_전월 == 0 THEN 증감율_이용금액_신용_전월 = 증감율_이용금액_CA_전월
        ELSE PASS
    """
    dd = df[["증감율_이용금액_CA_전월", "증감율_이용금액_신판_전월", "증감율_이용금액_신용_전월"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)
    return res


@constraint_udf
def cf_08_0013(df: pd.DataFrame) -> Union[pd.Series, List[float]]:
    """
    formula:
        IF 증감율_이용금액_할부_전월 == 0 THEN 증감율_이용금액_신판_전월 = 증감율_이용금액_일시불_전월
        ELIF 증감율_이용금액_일시불_전월 == 0 THEN 증감율_이용금액_신판_전월 = 증감율_이용금액_할부_전월
        ELSE PASS
    """
    dd = df[["증감율_이용금액_할부_전월", "증감율_이용금액_일시불_전월", "증감율_이용금액_신판_전월"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)
    return res


@constraint_udf
def cf_08_0033(df: pd.DataFrame) -> Union[pd.Series, List[float]]:
    """
    formula:
        IF 증감율_이용건수_CA_분기 == 0 THEN 증감율_이용건수_신용_분기 = 증감율_이용건수_신판_분기
        ELIF 증감율_이용건수_신판_분기 == 0 THEN 증감율_이용건수_신용_분기 = 증감율_이용건수_CA_분기
        ELSE PASS
    """
    dd = df[["증감율_이용건수_CA_분기", "증감율_이용건수_신판_분기", "증감율_이용건수_신용_분기"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)
    return res


@constraint_udf
def cf_08_0034(df: pd.DataFrame) -> Union[pd.Series, List[float]]:
    """
    formula:
        IF 증감율_이용건수_할부_분기 == 0 THEN 증감율_이용건수_신판_분기 = 증감율_이용건수_일시불_분기
        ELIF 증감율_이용건수_일시불_분기 == 0 THEN 증감율_이용건수_신판_분기 = 증감율_이용건수_할부_분기
        ELSE PASS
    """
    dd = df[["증감율_이용건수_할부_분기", "증감율_이용건수_일시불_분기", "증감율_이용건수_신판_분기"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)
    return res


@constraint_udf
def cf_08_0040(df: pd.DataFrame) -> Union[pd.Series, List[float]]:
    """
    formula:
        IF 증감율_이용금액_CA_분기 == 0 THEN 증감율_이용금액_신용_분기 = 증감율_이용금액_신판_분기
        ELIF 증감율_이용금액_신판_분기 == 0 THEN 증감율_이용금액_신용_분기 = 증감율_이용금액_CA_분기
        ELSE PASS
    """
    dd = df[["증감율_이용금액_CA_분기", "증감율_이용금액_신판_분기", "증감율_이용금액_신용_분기"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)
    return res


@constraint_udf
def cf_08_0041(df: pd.DataFrame) -> Union[pd.Series, List[float]]:
    """
    formula:
        IF 증감율_이용금액_할부_분기 == 0 THEN 증감율_이용금액_신판_분기 = 증감율_이용금액_일시불_분기
        ELIF 증감율_이용금액_일시불_분기 == 0 THEN 증감율_이용금액_신판_분기 = 증감율_이용금액_할부_분기
        ELSE PASS
    """
    dd = df[["증감율_이용금액_할부_분기", "증감율_이용금액_일시불_분기", "증감율_이용금액_신판_분기"]]
    res = dd.apply(lambda x: x[1] if x[0] == 0 else (x[0] if x[1] == 0 else x[2]), axis=1)
    return res


@constraint_udf
def cf_03_0006(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        최종이용일자_기본 = MAX(최종이용일자_신판, 최종이용일자_CA, 최종이용일자_카드론)
    """
    dd = df[["최종이용일자_신판", "최종이용일자_CA", "최종이용일자_카드론"]]
    res = dd.max(axis=1).astype(int).astype(str)
    return res


@constraint_udf
def cf_03_0007(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        최종이용일자_신판 = MAX(최종이용일자_일시불, 최종이용일자_할부)
    """
    dd = df[["최종이용일자_일시불", "최종이용일자_할부"]]
    res = dd.max(axis=1).astype(int).astype(str)
    return res


@constraint_udf
def cf_03_0013(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_신용_B0M = 이용건수_신판_B0M + 이용건수_CA_B0M
    """
    c1, c2 = df["이용건수_신판_B0M"], df["이용건수_CA_B0M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_03_0014(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_신판_B0M = 이용건수_일시불_B0M + 이용건수_할부_B0M
    """
    c1, c2 = df["이용건수_일시불_B0M"], df["이용건수_할부_B0M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_03_0016(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_할부_B0M = 이용건수_할부_유이자_B0M + 이용건수_할부_무이자_B0M + 이용건수_부분무이자_B0M
    """
    c1, c2, c3 = df["이용건수_할부_유이자_B0M"], df["이용건수_할부_무이자_B0M"], df["이용건수_부분무이자_B0M"]
    res = c1 + c2 + c3
    return res


@constraint_udf
def cf_03_0023(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_신용_B0M = 이용금액_신판_B0M + 이용금액_CA_B0M
    """
    c1, c2 = df["이용금액_신판_B0M"], df["이용금액_CA_B0M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_03_0024(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_신판_B0M = 이용금액_일시불_B0M + 이용금액_할부_B0M
    """
    c1, c2 = df["이용금액_일시불_B0M"], df["이용금액_할부_B0M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_03_0026(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_할부_B0M = 이용금액_할부_유이자_B0M + 이용금액_할부_무이자_B0M + 이용금액_부분무이자_B0M
    """
    c1, c2, c3 = df["이용금액_할부_유이자_B0M"], df["이용금액_할부_무이자_B0M"], df["이용금액_부분무이자_B0M"]
    res = c1 + c2 + c3
    return res


# @constraint_udf
# def cf_03_0033(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
#     """
#     formula:
#         이용후경과월_신용 = MIN(이용후경과월_신판, 이용후경과월_CA)
#     """
#     dd = df[["이용후경과월_신판", "이용후경과월_CA"]]
#     res = dd.min(axis=1).astype(int)
#     return res


@constraint_udf
def cf_03_0034(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용후경과월_신판 = MIN(이용후경과월_일시불, 이용후경과월_할부)
    """
    dd = df[["이용후경과월_일시불", "이용후경과월_할부"]]
    res = dd.min(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0035(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0036(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
       이용후경과월_할부 = MIN(이용후경과월_할부_유이자, 이용후경과월_할부_무이자, 이용후경과월_부분무이자)
    """
    dd = df[["이용후경과월_할부_유이자", "이용후경과월_할부_무이자", "이용후경과월_부분무이자"]]
    res = dd.min(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0040(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0041(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0042(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0043(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_신용_R12M = SUM(이용건수_신판_R12M, 이용건수_CA_R12M)
    """
    c1, c2 = df["이용건수_신판_R12M"], df["이용건수_CA_R12M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_03_0044(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
       이용건수_신판_R12M = SUM(이용건수_일시불_R12M, 이용건수_할부_R12M)
    """
    c1, c2 = df["이용건수_일시불_R12M"], df["이용건수_할부_R12M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_03_0046(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
       이용건수_할부_R12M = SUM(이용건수_할부_유이자_R12M, 이용건수_할부_무이자_R12M, 이용건수_부분무이자_R12M)
    """
    c1, c2, c3 = df["이용건수_할부_유이자_R12M"], df["이용건수_할부_무이자_R12M"], df["이용건수_부분무이자_R12M"]
    res = c1 + c2 + c3
    return res


@constraint_udf
def cf_03_0047(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0048(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0049(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0053(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
       이용금액_신용_R12M = SUM(이용금액_신판_R12M, 이용금액_CA_R12M)
    """
    c1, c2 = df["이용금액_신판_R12M"], df["이용금액_CA_R12M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_03_0054(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
       이용금액_신판_R12M = SUM(이용금액_일시불_R12M, 이용금액_할부_R12M)
    """
    c1, c2 = df["이용금액_일시불_R12M"], df["이용금액_할부_R12M"]
    res = c1 + c2
    return res


@constraint_udf
def cf_03_0056(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
       이용금액_할부_R12M = SUM(이용금액_할부_유이자_R12M, 이용금액_할부_무이자_R12M, 이용금액_부분무이자_R12M)
    """
    c1, c2, c3 = df["이용금액_할부_유이자_R12M"], df["이용금액_할부_무이자_R12M"], df["이용금액_부분무이자_R12M"]
    res = c1 + c2 + c3
    return res


@constraint_udf
def cf_03_0057(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0058(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0059(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0066(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        최대이용금액_할부_R12M = MAX(최대이용금액_할부_유이자_R12M, 할부_무이자, 부분무이자)
    """
    dd = df[["최대이용금액_할부_유이자_R12M", "최대이용금액_할부_무이자_R12M", "최대이용금액_부분무이자_R12M"]]
    res = dd.max(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0083(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_신용_R6M = SUM(이용건수_신판_R6M, 이용건수_CA_R6M)
    """
    dd = df[["이용건수_신판_R6M", "이용건수_CA_R6M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0084(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_신판_R6M = SUM(이용건수_일시불_R6M, 이용건수_할부_R6M)
    """
    dd = df[["이용건수_일시불_R6M", "이용건수_할부_R6M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0086(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_할부_R6M = SUM(이용건수_할부_유이자_R6M, 이용건수_할부_무이자_R6M, 이용건수_부분무이자_R6M)
    """
    dd = df[["이용건수_할부_유이자_R6M", "이용건수_할부_무이자_R6M", "이용건수_부분무이자_R6M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0093(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_신용_R6M = SUM(이용금액_신판_R6M, 이용금액_CA_R6M)
    """
    dd = df[["이용금액_신판_R6M", "이용금액_CA_R6M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0094(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_신판_R6M = SUM(이용금액_일시불_R6M, 이용금액_할부_R6M)
    """
    dd = df[["이용금액_일시불_R6M", "이용금액_할부_R6M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0096(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_할부_R6M = SUM(이용금액_할부_유이자_R6M, 이용금액_할부_무이자_R6M, 이용금액_부분무이자_R6M)
    """
    dd = df[["이용금액_할부_유이자_R6M", "이용금액_할부_무이자_R6M", "이용금액_부분무이자_R6M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0113(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_신용_R3M = SUM(이용건수_신판_R3M, 이용건수_CA_R3M)
    """
    dd = df[["이용건수_신판_R3M", "이용건수_CA_R3M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0114(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_신판_R3M = SUM(이용건수_일시불_R3M, 이용건수_할부_R3M)
    """
    dd = df[["이용건수_일시불_R3M", "이용건수_할부_R3M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0116(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_할부_R3M = SUM(이용건수_할부_유이자_R3M, 이용건수_할부_무이자_R3M, 이용건수_부분무이자_R3M)
    """
    dd = df[["이용건수_할부_유이자_R3M", "이용건수_할부_무이자_R3M", "이용건수_부분무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0123(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_신용_R3M = SUM(이용금액_신판_R3M, 이용금액_CA_R3M)
    """
    dd = df[["이용금액_신판_R3M", "이용금액_CA_R3M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0124(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_신판_R3M = SUM(이용금액_일시불_R3M, 이용금액_할부_R3M)
    """
    dd = df[["이용금액_일시불_R3M", "이용금액_할부_R3M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0126(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_할부_R3M = SUM(이용금액_할부_유이자_R3M, 이용금액_할부_무이자_R3M, 이용금액_부분무이자_R3M)
    """
    dd = df[["이용금액_할부_유이자_R3M", "이용금액_할부_무이자_R3M", "이용금액_부분무이자_R3M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0160(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0162(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0164(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0176(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        쇼핑_전체_이용금액 = 이용금액_쇼핑
    """
    res = df["이용금액_쇼핑"]
    return res


@constraint_udf
def cf_03_0183(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        교통_전체이용금액 = 이용금액_교통
    """
    res = df["이용금액_교통"]
    return res


@constraint_udf
def cf_03_0192(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        여유_전체이용금액 = 이용금액_여유생활
    """
    res = df["이용금액_여유생활"]
    return res


@constraint_udf
def cf_03_0195(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        납부_렌탈료이용금액 = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0201(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        납부_전체이용금액 = 이용금액_납부
    """
    res = df["이용금액_납부"]
    return res


@constraint_udf
def cf_03_0202(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    dd["_max"] = dd.max(axis=1)

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
    res = dd.apply(
        lambda x: np.where(x[:-1] == x[-1])[0][0] if x[-1] > 0 else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0203(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0204(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    res = dd.apply(
        lambda x: np.argsort(x[:-1])[-3] if x[-1] else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0205(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0206(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    res = dd.apply(
        lambda x: np.where(x[:-1] == x[-1])[0][0] if x[-1] > 0 else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0207(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0208(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    res = dd.apply(
        lambda x: np.argsort(x[:-1])[-3] if x[-1] else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0209(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0210(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    res = dd.apply(
        lambda x: np.where(x[:-1] == x[-1])[0][0] if x[-1] > 0 else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0211(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0212(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    res = dd.apply(
        lambda x: np.argsort(x[:-1])[-3] if x[-1] else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0213(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0214(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    res = dd.apply(
        lambda x: np.where(x[:-1] == x[-1])[0][0] if x[-1] > 0 else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0215(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0216(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    res = dd.apply(
        lambda x: np.argsort(x[:-1])[-3] if x[-1] else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0217(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0218(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
        3: "가스전기료",
        4: "보험료",
        5: "유선방송",
        6: "건강연금",
        7: "납부기타",
    }
    res = dd.apply(
        lambda x: np.where(x[:-1] == x[-1])[0][0] if x[-1] > 0 else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0219(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0220(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
        3: "가스전기료",
        4: "보험료",
        5: "유선방송",
        6: "건강연금",
        7: "납부기타",
    }
    res = dd.apply(
        lambda x: np.argsort(x[:-1])[-3] if x[-1] else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0221(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0222(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        할부건수_3M_R12M = SUM(할부건수_유이자_3M_R12M, 할부건수_무이자_3M_R12M, 할부건수_부분_3M_R12M)
    """
    dd = df[["할부건수_유이자_3M_R12M", "할부건수_무이자_3M_R12M", "할부건수_부분_3M_R12M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0223(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        할부건수_6M_R12M = SUM(할부건수_유이자_6M_R12M, 할부건수_무이자_6M_R12M, 할부건수_부분_6M_R12M)
    """
    dd = df[["할부건수_유이자_6M_R12M", "할부건수_무이자_6M_R12M", "할부건수_부분_6M_R12M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0224(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        할부건수_12M_R12M = SUM(할부건수_유이자_12M_R12M, 할부건수_무이자_12M_R12M, 할부건수_부분_12M_R12M)
    """
    dd = df[["할부건수_유이자_12M_R12M", "할부건수_무이자_12M_R12M", "할부건수_부분_12M_R12M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0225(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        할부건수_14M_R12M = SUM(할부건수_유이자_14M_R12M, 할부건수_무이자_14M_R12M, 할부건수_부분_14M_R12M)
    """
    dd = df[["할부건수_유이자_14M_R12M", "할부건수_무이자_14M_R12M", "할부건수_부분_14M_R12M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0226(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        할부금액_3M_R12M = SUM(할부금액_유이자_3M_R12M, 할부금액_무이자_3M_R12M, 할부금액_부분_3M_R12M)
    """
    dd = df[["할부금액_유이자_3M_R12M", "할부금액_무이자_3M_R12M", "할부금액_부분_3M_R12M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0227(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        할부금액_6M_R12M = SUM(할부금액_유이자_6M_R12M, 할부금액_무이자_6M_R12M, 할부금액_부분_6M_R12M)
    """
    dd = df[["할부금액_유이자_6M_R12M", "할부금액_무이자_6M_R12M", "할부금액_부분_6M_R12M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0228(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        할부금액_12M_R12M = SUM(할부금액_유이자_12M_R12M, 할부금액_무이자_12M_R12M, 할부금액_부분_12M_R12M)
    """
    dd = df[["할부금액_유이자_12M_R12M", "할부금액_무이자_12M_R12M", "할부금액_부분_12M_R12M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0229(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        할부금액_14M_R12M = SUM(할부금액_유이자_14M_R12M, 할부금액_무이자_14M_R12M, 할부금액_부분_14M_R12M)
    """
    dd = df[["할부금액_유이자_14M_R12M", "할부금액_무이자_14M_R12M", "할부금액_부분_14M_R12M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0246(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        할부건수_부분_3M_R12M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0250(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        할부금액_부분_3M_R12M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0254(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0268(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0281(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0289(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        최종카드론_대출일자 == 최종이용일자_카드론
    """
    dd = df[["최종이용일자_카드론"]]
    res = dd.apply(lambda x: x[0] if (not isNaN(x[0])) & (not x[0] == '10101')
                   else float('nan'), axis=1)
    return res


@constraint_udf
def cf_03_0338(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용개월수_당사페이_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0345(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_당사페이_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0346(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_당사기타_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0352(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_당사페이_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0353(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_당사기타_R6M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0355(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_B페이_R6M = 이용개월수_B페이_R6M
    """
    res = df["이용개월수_B페이_R6M"]
    return res


@constraint_udf
def cf_03_0356(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_C페이_R6M = 이용개월수_C페이_R6M
    """
    res = df["이용개월수_C페이_R6M"]
    return res


@constraint_udf
def cf_03_0357(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_D페이_R6M = 이용개월수_D페이_R6M
    """
    res = df["이용개월수_D페이_R6M"]
    return res


@constraint_udf
def cf_03_0359(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_당사페이_R3M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0360(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_당사기타_R3M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0366(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_당사페이_R3M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0367(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_당사기타_R3M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0373(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_당사페이_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0374(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용금액_당사기타_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0380(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_당사페이_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0381(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        이용건수_당사기타_B0M = 0
    """
    c1 = df["기준년월"]
    res = pd.Series([0] * len(c1))
    return res


@constraint_udf
def cf_03_0408(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        연체입금원금_B0M = 정상청구원금_B0M - (선입금원금_B0M + 정상입금원금_B0M)
    """
    c1, c2, c3 = df["정상청구원금_B0M"], df["선입금원금_B0M"], df["정상입금원금_B0M"]
    res = c1 - (c2 + c3)
    return res


@constraint_udf
def cf_03_0409(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        연체입금원금_B2M = 정상청구원금_B2M - (선입금원금_B2M + 정상입금원금_B2M)
    """
    c1, c2, c3 = df["정상청구원금_B2M"], df["선입금원금_B2M"], df["정상입금원금_B2M"]
    res = c1 - (c2 + c3)
    return res


@constraint_udf
def cf_03_0410(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        연체입금원금_B5M = 정상청구원금_B5M - (선입금원금_B5M + 정상입금원금_B5M)
    """
    c1, c2, c3 = df["정상청구원금_B5M"], df["선입금원금_B5M"], df["정상입금원금_B5M"]
    res = c1 - (c2 + c3)
    return res


@constraint_udf
def cf_03_0425(df: pd.DataFrame) -> Union[pd.Series, List[str]]:
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
    res = dd.apply(
        lambda x: np.argsort(x[:-1])[-2] if x[-1] else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0426(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0427(df: pd.DataFrame) -> Union[pd.Series, List[str]]:
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
    res = dd.apply(
        lambda x: np.argsort(x[:-1])[-2] if x[-1] else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0428(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0429(df: pd.DataFrame) -> Union[pd.Series, List[str]]:
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
    res = dd.apply(
        lambda x: np.argsort(x[:-1])[-2] if x[-1] else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0430(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0431(df: pd.DataFrame) -> Union[pd.Series, List[str]]:
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
    res = dd.apply(
        lambda x: np.argsort(x[:-1])[-2] if x[-1] else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0432(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0433(df: pd.DataFrame) -> Union[pd.Series, List[str]]:
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
        3: "가스전기료",
        4: "보험료",
        5: "유선방송",
        6: "건강연금",
        7: "납부기타",
    }
    res = dd.apply(
        lambda x: np.argsort(x[:-1])[-2] if x[-1] else float('nan'), axis=1
    ).replace(code_map)
    return res


@constraint_udf
def cf_03_0434(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
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
    return res


@constraint_udf
def cf_03_0462(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        승인거절건수_B0M = SUM(승인거절건수_한도초과_B0M, BL_B0M, 입력오류_B0M, 기타_B0M)
    """
    dd = df[["승인거절건수_한도초과_B0M", "승인거절건수_BL_B0M", "승인거절건수_입력오류_B0M", "승인거절건수_기타_B0M"]]
    res = dd.sum(axis=1).astype(int)
    return res


@constraint_udf
def cf_03_0467(df: pd.DataFrame) -> Union[pd.Series, List[int]]:
    """
    formula:
        승인거절건수_R3M = SUM(승인거절건수_한도초과_R3M, BL_R3M, 입력오류_R3M, 기타_R3M)
    """
    dd = df[["승인거절건수_한도초과_R3M", "승인거절건수_BL_R3M", "승인거절건수_입력오류_R3M", "승인거절건수_기타_R3M"]]
    res = dd.sum(axis=1).astype(int)
    return res