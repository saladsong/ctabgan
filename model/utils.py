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

    # integer
    tmp = list(
        filter(lambda x: x[1]["properties"]["is_integer"] == 1, col_info.items())
    )
    integer_list = [k for k, v in tmp]
    print(f"[integer_list]: {len(integer_list)}")

    return {
        "categorical_columns": cate_list,
        "log_columns": log_list,
        "mixed_columns": mixed_dict,
        "general_columns": general_list,
        "non_categorical_columns": non_cate_list,
        "integer_columns": integer_list,
    }
