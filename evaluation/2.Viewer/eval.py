import pandas as pd
import numpy as np
import os
import warnings
from tqdm.auto import tqdm
from utils import get_jsd, get_pmse, get_corrdiff

warnings.filterwarnings("ignore")

topic_list = [
    "01.회원정보",
    "02.신용정보",
    "03.승인매출정보",
    "04.청구정보",
    "05.잔액정보",
    "06.채널정보",
    "07.마케팅정보",
    "08.성과정보",
]
baseym_list = ["201807", "201808", "201809", "201810", "201811", "201812"]
high_cardinality_cols = [
    # 코드 필드
    "_1순위카드상품종류코드",
    "최종카드상품종류코드",
    "_2순위카드상품종류코드",
    "단말기모델명",
    "유치경로코드_신용",
    "자사카드자격코드",
    "OS버전값",
    "대표결제은행코드",
    "최종카드론_금융상품코드",
    "대표BL코드_ACCOUNT",
    # 날짜 필드 추가
    "입회일자_신용",
    "최종카드발급일자",
    "RV신청일자",
    "최종이용일자_기본",
    "최종이용일자_신판",
    "최종이용일자_CA",
    "최종이용일자_카드론",
    "최종이용일자_체크",
    "최종이용일자_일시불",
    "최종이용일자_할부",
    "최종카드론_대출일자",
    "연체일자_B0M",
]

KEY_COL = "발급회원번호"
BASE_YM_COL = "기준년월"
PARTITION_COL = "is_syn"


# 컬럼별 타입 데이터 로드
df_types = pd.read_csv("../1.Dataset/Other/메타데이터/col_types_orig.csv")

meta = {}
for k, v in df_types.values:
    _type = None
    if v.lower() == "char":
        _type = "str"
        # _type = "object"
    elif v.lower() == "int":
        _type = "int"
    elif v.lower() == "num":
        _type = "float"
    else:
        raise Exception(f"현재 지원하지 않는 타입입니다: {v}")
    meta.update({k: _type})


from typing import Tuple, List, Union


def load_data(
    topic: str,
    baseym: str,
    # origdata: Union[str, pd.DataFrame] = None,
    syndata: Union[str, pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    base_data_dir = "../1.Dataset"
    idx, tname = topic.split(".")

    df_orig = pd.read_csv(
        os.path.join(base_data_dir, f"원천데이터/{idx}.카드 {tname}/{baseym}_{tname}.csv")
    )
    if syndata is None:
        # for 08 tta
        df_syn = pd.read_csv(
            os.path.join(base_data_dir, f"합성데이터/{idx}.카드 {tname}/{baseym}_{tname}.csv")
        )
    elif isinstance(syndata, pd.DataFrame):
        df_syn = syndata[syndata["기준년월"] == baseym][df_orig.columns]
    else:
        pass
        # assert isinstance(syndata, str)
        # df_syn = pd.read_parquet(syndata)

    assert (
        len(set(df_orig.columns).difference(df_syn.columns)) == 0
    ), "원천데이터와 합성데이터의 컬럼 구성이 다릅니다."

    curr_columns = list(df_orig.columns)
    print(f"데이터 모양: [df_orig]: {df_orig.shape}, [df_syn]: {df_syn.shape}")

    # 유효성 체크
    unseened_cols = list(set(df_orig.columns).difference(df_types["데이터필드명"]))
    assert len(unseened_cols) == 0, f"메타정보에 존재하지 않는 컬럼이 있습니다: {unseened_cols}"

    curr_high_cardinality_cols = list(
        set(curr_columns).intersection(high_cardinality_cols)
    )
    print(f"현재 데이터는 다음의 high_cardinality_cols를 갖고 있습니다: {curr_high_cardinality_cols}")

    return df_orig, df_syn, curr_high_cardinality_cols


def calc_eval(
    topic,
    baseym,
    syndata: Union[str, pd.DataFrame] = None,
    BASE_YM_COL: str = None,
    KEY_COL: str = None,
    PARTITION_COL: str = None,
):
    print(f"수행 정보: [주제영역]: {topic}, [기준일자]: {baseym}")
    df_orig, df_syn, curr_high_cardinality_cols = load_data(topic, baseym, syndata)

    # 데이터 타입을 메타에 맞게 변환
    curr_columns = list(df_orig.columns)
    included_meta = {k: v for k, v in meta.items() if k in curr_columns}
    df_orig_pps = df_orig.astype(included_meta)
    df_syn_pps = df_syn.astype(included_meta)
    assert all(df_orig_pps.dtypes.values == df_syn_pps.dtypes.values)

    # merge orig, syn data
    df_merge = pd.concat(
        [
            pd.concat(
                [df_orig_pps, pd.Series([0] * len(df_orig_pps), name=PARTITION_COL)],
                axis=1,
            ),
            pd.concat(
                [
                    df_syn_pps,
                    pd.Series([1] * len(df_syn_pps), name=PARTITION_COL),
                ],
                axis=1,
            ),
        ]
    ).reset_index(drop=True)

    ### JSD 계산
    jsd = get_jsd(
        df_merge.drop(columns=[BASE_YM_COL]),
        key_col=KEY_COL,
        partition_col=PARTITION_COL,
    )

    ### pMSE 계산
    pmse = get_pmse(
        df_merge.drop(columns=[BASE_YM_COL, KEY_COL]),
        high_cardinality_cols=curr_high_cardinality_cols,
        partition_col=PARTITION_COL,
    )

    ### Corr.diff 계산
    cate_cols = list(df_orig_pps.select_dtypes(include=["object"]).columns)
    corr_diff = get_corrdiff(
        real=df_orig_pps.drop(columns=[BASE_YM_COL, KEY_COL]),
        fake=df_syn_pps.drop(columns=[BASE_YM_COL, KEY_COL]),
        categorical_columns=cate_cols,
    )

    res = {
        "baseym": baseym,
        "topic": topic,
        "js_divergence": jsd,
        "pmse": pmse,
        "corr_diff": corr_diff,
    }
    print(res)
    return res


from multiprocessing import Pool, Manager, cpu_count
from datetime import datetime
import json
import argparse
from itertools import product

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpath", "-d", type=str, help="합성데이터 파케이파일 패스", required=True)
    parser.add_argument(
        "--parallel",
        "-p",
        type=str,
        choices=["y", "n"],
        default="n",
        help="병렬처리여부",
        required=False,
    )
    parser.add_argument(
        "--n_jobs", "-n", type=int, default=1, help="병렬처리 job개수", required=False
    )
    config = parser.parse_args()
    # parse parmas
    parallel = True if config.parallel == "y" else False
    n_jobs = config.n_jobs
    dpath = config.dpath

    conditions = list(product(topic_list, baseym_list))
    res_all = []

    date_str = datetime.now().strftime("%y%m%d-%H%M%S")
    fname = f"eval_{date_str}"

    def callback(result):
        """tqdm 업데이트용 콜백함수"""
        pbar.update(1)

    if parallel:
        with Manager() as manager:
            with Pool(n_jobs) as pool:
                with tqdm(total=len(conditions)) as pbar:
                    results = []
                    # run multi processes
                    for topic, baseym in conditions:
                        results.append(
                            pool.apply_async(
                                calc_eval,
                                args=(
                                    topic,
                                    baseym,
                                    BASE_YM_COL,
                                    KEY_COL,
                                    PARTITION_COL,
                                ),
                                callback=callback,
                            )
                        )
                    # gather results
                    for id_, r in enumerate(results):
                        res_dict = r.get()
                        res_all.append(res_dict)
    else:
        syn = pd.read_parquet(dpath)
        for topic, baseym in tqdm(conditions):
            res_dict = calc_eval(
                topic, baseym, syn, BASE_YM_COL, KEY_COL, PARTITION_COL
            )
            res_all.append(res_dict)

    # 결과지표 확인
    df_res = pd.DataFrame(res_all)
    # df_res

    # csv 형태로 결과 저장
    df_res.to_csv(f"{fname}.csv", index=False)

    # # json 형태로 결과 저장
    # with open(f"{fname}.json", "w") as f:
    #     json.dump(res_all, f, ensure_ascii=False, indent=4)
