import numpy as np
import pandas as pd
import torch
from sklearn.mixture import BayesianGaussianMixture
import logging
from tqdm.auto import tqdm
from typing import List, Union, Tuple
import pickle
import os
from multiprocessing import Process, Queue, Pool, cpu_count

# from model.synthesizer.utils import get_tcol_idx_st_ed_tuple

RANDOM_SEED = 777


def encode_column(
    transformer: "DataTransformer",
    arr: np.ndarray,
    info: dict,
    ispositive=False,
    positive_list=None,
) -> Union[np.array, List[np.array]]:
    """encode single column"""
    len_data = len(arr)
    id_ = info["name"]
    ret = None

    if info["type"] == "continuous":
        # MSN 적용 대상 컬럼인 경우: get alpha_i, beta_i
        if id_ not in transformer.general_columns:
            arr = arr.reshape([-1, 1])
            means = transformer.model[id_].means_.reshape((1, transformer.n_clusters))
            stds = np.sqrt(transformer.model[id_].covariances_).reshape(
                (1, transformer.n_clusters)
            )
            # features: 각 mode 별 정규화 (alpha_i candidates)
            features = np.empty(shape=(len(arr), transformer.n_clusters))
            if ispositive is True:
                if id_ in positive_list:
                    features = np.abs(arr - means) / (
                        4 * stds
                    )  # lsw: 0으로 자르는 것도 아니고 절대값? 약간 애매하지만 넘어가자
            else:
                features = (arr - means) / (4 * stds)

            probs = transformer.model[id_].predict_proba(arr.reshape([-1, 1]))
            n_opts = sum(transformer.valid_mode_flags[id_])  # n_opts: 해당 컬럼의 유효 mode 개수
            features = features[
                :, transformer.valid_mode_flags[id_]
            ]  # features: 유효 mode 의 alpha_i 만 필터링 (N * #valid_mode)
            probs = probs[
                :, transformer.valid_mode_flags[id_]
            ]  # probs: 유효 mode 의 확률만 필터링 (N * #valid_mode)

            # 해당 확률 기반 최적 mode 선택
            opt_sel = np.zeros(len_data, dtype="int")
            for i in range(len_data):
                pp = probs[i] + 1e-6
                pp = pp / sum(pp)
                # lsw: 왜 랜덤이 들어감???????????? 논문에선 확률 가장높은 모드로 쓰는데?? 샘플링은 ctgan 방법임
                opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

            idx = np.arange((len(features)))
            features = features[idx, opt_sel].reshape(
                [-1, 1]
            )  # (optimal) alpha_i list (N*1)
            # lsw: 클리핑 부분은 논문엔 없었음, 그래도 얘도 tanh랑 맞출려고 사용하는 듯
            features = np.clip(features, -0.99, 0.99)
            probs_onehot = np.zeros_like(probs)  # (N * #valid_mode)
            probs_onehot[np.arange(len(probs)), opt_sel] = 1  # mode-indicator beta_i

            # lsw: re-ordering 하는 이유????
            # 앞에서 확률 가장높은 모드로 쓰지않고 ctgan 방법으로 beta 샘플링해서 무조건 맨 처음값이 1은 아니지만
            # 그래도 굳이 높은 확률 모드 순으로 리오더링 하는 이유는 없는듯
            # 그냥 연구자가 어느 오더의 모드에서 beta 샘플링 되었는지 쉽게 확인하기 위함일지도
            re_ordered_phot = np.zeros_like(probs_onehot)  # (N * #valid_mode)
            col_sums = probs_onehot.sum(axis=0)  # (#valid_mode,)
            largest_indices = np.argsort(-1 * col_sums)  # 큰값부터 인덱싱
            # lsw: 불필요
            # n = probs_onehot.shape[1]  # #valid_mode
            # largest_indices = np.argsort(-1 * col_sums)[:n]  # 큰값부터 인덱싱
            transformer.ordering[id_] = largest_indices  # (#valid_mode,)

            for id, val in enumerate(largest_indices):
                re_ordered_phot[:, id] = probs_onehot[:, val]

            ret = [
                features,
                re_ordered_phot,
            ]  # alpha_i (N * 1), beta_i (N * #valid_mode)

        # GT 적용 대상 컬럼인 경우: transform to x_t
        # lsw: 추후 데이터 미니 배치로 넣으면 이부분도 최소, 최대값 저장했다가 다시 쓰도록 리팩토링 해야할지도 ...
        else:
            transformer.ordering[id_] = None

            if id_ in transformer.non_categorical_columns:
                info["min"] = -1e-3
                info["max"] = info["max"] + 1e-3

            arr = (arr - (info["min"])) / (info["max"] - info["min"])
            arr = arr * 2 - 1
            arr = arr.reshape([-1, 1])
            ret = arr  # alpha_i (N * 1)

    # MSN 적용 대상 mixed 컬럼인 경우: get alpha_i, beta_i
    elif info["type"] == "mixed":
        means_0 = transformer.model[id_][0].means_.reshape([-1])
        stds_0 = np.sqrt(transformer.model[id_][0].covariances_).reshape([-1])

        zero_std_list = []
        means_needed = []
        stds_needed = []

        # -9999999 외의 modal 값에 대한 mean, sqrt 계산 (normalize 에 활용)
        # gm1 모델의 10개 mode 중, modal 값과 mean 값이 가장 가까운 mode 의 mean, sqrt 를 취함
        for mode in info["modal"]:
            if mode != -9999999:
                dist = []
                for idx, val in enumerate(list(means_0.flatten())):
                    dist.append(abs(mode - val))
                index_min = np.argmin(np.array(dist))
                zero_std_list.append(index_min)
            else:
                continue

        for idx in zero_std_list:
            means_needed.append(means_0[idx])
            stds_needed.append(stds_0[idx])

        # mode 별 normalized alpha_i 를 저장
        mode_vals = []
        # lsw: 이 코드는 -9999999 가 info["modal"]의 맨 마지막에 추가되므로 가능한 것임 - 안좋은 코드
        # modal 값에 대한 alpha_i 계산
        for i, j, k in zip(info["modal"], means_needed, stds_needed):
            this_val = np.abs(i - j) / (4 * k)
            mode_vals.append(this_val)
        if -9999999 in info["modal"]:
            mode_vals.append(0)

        # modal 이 아닌 continuous 값의 mode 에 대한 alpha_i, beta_i 계산
        # gm2 모델의 mean, std 활용
        arr = arr.reshape([-1, 1])
        filter_arr = info["filter_arr"]
        arr = arr[filter_arr]

        means = transformer.model[id_][1].means_.reshape((1, transformer.n_clusters))
        stds = np.sqrt(transformer.model[id_][1].covariances_).reshape(
            (1, transformer.n_clusters)
        )
        features = np.empty(shape=(len(arr), transformer.n_clusters))
        if ispositive is True:
            if id_ in positive_list:
                features = np.abs(arr - means) / (4 * stds)
        else:
            features = (arr - means) / (4 * stds)

        probs = transformer.model[id_][1].predict_proba(arr.reshape([-1, 1]))
        n_opts = sum(transformer.valid_mode_flags[id_])
        features = features[:, transformer.valid_mode_flags[id_]]
        probs = probs[:, transformer.valid_mode_flags[id_]]

        opt_sel = np.zeros(len(arr), dtype="int")
        for i in range(len(arr)):
            pp = probs[i] + 1e-6
            pp = pp / sum(pp)
            opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

        idx = np.arange((len(features)))
        features = features[idx, opt_sel].reshape([-1, 1])
        features = np.clip(features, -0.99, 0.99)
        probs_onehot = np.zeros_like(probs)
        probs_onehot[np.arange(len(probs)), opt_sel] = 1

        # modal 값 포함 전체 mode 에 대한 최종 concat vector (final) 생성
        # final shape: (n * (1 for alpha_i + one-hot for modal + one-hot for modes))
        extra_bits = np.zeros([len(arr), len(info["modal"])])
        temp_probs_onehot = np.concatenate(
            [extra_bits, probs_onehot], axis=1
        )  # modal 을 MSN 앞에 붙이네
        final = np.zeros(
            [
                len_data,
                1 + probs_onehot.shape[1] + len(info["modal"]),
            ]  # (N * 1+ #vaild_mode + #modal)  // +1 은 alpha 위한 것
        )

        # final 내에 alpha, beta 채우는 과정
        features_curser = 0
        for idx, val in enumerate(arr):
            if val in info["modal"]:
                # category_ = list(map(info["modal"].index, [val]))[0]
                category_ = info["modal"].index(val)
                final[idx, 0] = mode_vals[category_]  # alpha_i
                final[idx, (category_ + 1)] = 1  # beta_i

            else:
                final[idx, 0] = features[features_curser]
                final[idx, (1 + len(info["modal"])) :] = temp_probs_onehot[
                    features_curser
                ][len(info["modal"]) :]
                features_curser = features_curser + 1

        # one-hot mode 순서 정렬 (빈도 수 높은 mode 순)
        just_onehot = final[:, 1:]
        re_ordered_jhot = np.zeros_like(just_onehot)
        col_sums = just_onehot.sum(axis=0)
        largest_indices = np.argsort(-1 * col_sums)
        # n = just_onehot.shape[1]
        # largest_indices = np.argsort(-1 * col_sums)[:n]
        transformer.ordering[id_] = largest_indices

        for id, val in enumerate(largest_indices):
            re_ordered_jhot[:, id] = just_onehot[:, val]

        final_features = final[:, 0].reshape([-1, 1])
        ret = [final_features, re_ordered_jhot]  # alpha_i, beta_i

    # categorical 컬럼인 경우: get one-hot
    else:
        transformer.ordering[id_] = None
        col_t = np.zeros([len_data, info["size"]])
        idx = list(map(info["i2s"].index, arr))
        col_t[np.arange(len_data), idx] = 1
        ret = col_t  # gamma_i

    return ret


def decode_column(
    transformer: "DataTransformer", arr: np.ndarray, info: dict
) -> Tuple[np.ndarray, list]:
    """decode single column"""
    len_data = len(arr)
    id_ = info["name"]
    invalid_ids = []  # fake 를 decode 해보니 컬럼 조건(min, max) 에 위배되는 경우
    ret = None
    if info["type"] == "continuous":
        # MSN 역변환
        if id_ not in transformer.general_columns:
            u = arr[:, 0]  # alphas
            v = arr[:, 1:]  # betas
            order = transformer.ordering[id_]
            v_re_ordered = np.zeros_like(v)

            for id, val in enumerate(order):
                v_re_ordered[:, val] = v[:, id]

            v = v_re_ordered

            u = np.clip(u, -1, 1)
            v_t = np.ones((len_data, transformer.n_clusters)) * -100
            v_t[:, transformer.valid_mode_flags[id_]] = v
            v = v_t
            means = transformer.model[id_].means_.reshape([-1])
            stds = np.sqrt(transformer.model[id_].covariances_).reshape([-1])
            p_argmax = np.argmax(v, axis=1)
            std_t = stds[p_argmax]
            mean_t = means[p_argmax]
            tmp = u * 4 * std_t + mean_t

            for idx, val in enumerate(tmp):
                if (val < info["min"]) | (val > info["max"]):
                    invalid_ids.append(idx)

            if id_ in transformer.non_categorical_columns:
                tmp = np.round(tmp)

            ret = tmp

        # GT 역변환
        else:
            u = arr[:, 0]  # alphas
            u = (u + 1) / 2
            u = np.clip(u, 0, 1)
            u = u * (info["max"] - info["min"]) + info["min"]
            if id_ in transformer.non_categorical_columns:
                ret = np.round(u)
            else:
                ret = u

    # mixed MSN 역변환
    elif info["type"] == "mixed":
        u = arr[:, 0]  # alphas
        full_v = arr[:, 1:]  # betas
        order = transformer.ordering[id_]
        full_v_re_ordered = np.zeros_like(full_v)

        for id, val in enumerate(order):
            full_v_re_ordered[:, val] = full_v[:, id]

        full_v = full_v_re_ordered

        mixed_v = full_v[:, : len(info["modal"])]
        v = full_v[:, -np.sum(transformer.valid_mode_flags[id_]) :]

        u = np.clip(u, -1, 1)
        v_t = np.ones((len_data, transformer.n_clusters)) * -100
        v_t[:, transformer.valid_mode_flags[id_]] = v
        v = np.concatenate([mixed_v, v_t], axis=1)

        means = transformer.model[id_][1].means_.reshape([-1])
        stds = np.sqrt(transformer.model[id_][1].covariances_).reshape([-1])
        p_argmax = np.argmax(v, axis=1)

        result = np.zeros_like(u)

        for idx in range(len_data):
            if p_argmax[idx] < len(info["modal"]):
                argmax_value = p_argmax[idx]
                result[idx] = float(
                    list(map(info["modal"].__getitem__, [argmax_value]))[0]
                )
            else:
                std_t = stds[(p_argmax[idx] - len(info["modal"]))]
                mean_t = means[(p_argmax[idx] - len(info["modal"]))]
                result[idx] = u[idx] * 4 * std_t + mean_t

        for idx, val in enumerate(result):
            if (val < info["min"]) | (val > info["max"]):
                invalid_ids.append(idx)

        ret = result

    # one-hot 역변환
    else:
        # arr <- gammas
        idx = np.argmax(arr, axis=1)
        ret = np.array(list(map(info["i2s"].__getitem__, idx)))

    return ret, invalid_ids


class DataTransformer:
    def __init__(
        self,
        categorical_list: list = None,
        mixed_dict: dict = None,
        general_list: list = None,
        non_categorical_list: list = None,
        n_clusters: int = 10,
        eps: float = 0.005,
    ):
        # set initial params
        if categorical_list is None:
            categorical_list = []
        if mixed_dict is None:
            mixed_dict = {}
        if general_list is None:
            general_list = []
        if non_categorical_list is None:
            non_categorical_list = []

        self.logger = logging.getLogger()
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps
        self.categorical_columns = categorical_list
        self.mixed_columns = mixed_dict
        self.general_columns = general_list
        self.non_categorical_columns = non_categorical_list
        self.is_fit_ = False

    def get_metadata(self, df: pd.DataFrame) -> List[dict]:
        meta = []
        self.logger.info("[Transformer]: get metadata ...")
        for index in tqdm(range(df.shape[1])):
            column = df.iloc[:, index]
            # 범주형 컬럼
            if index in self.categorical_columns:
                # 범주형 컬럼 중 연속형처럼 GT, MSN 적용할 컬럼
                if index in self.non_categorical_columns:
                    meta.append(
                        {
                            "name": index,
                            "type": "continuous",
                            "min": column.min(),
                            "max": column.max(),
                        }
                    )
                else:
                    # mapper = column.value_counts().index.tolist()  # lsw: 인덱싱만 필요하므로 구태여 무겁게 모든 카운트 셀필요 없음
                    mapper = column.unique().tolist()
                    meta.append(
                        {
                            "name": index,
                            "type": "categorical",
                            "size": len(mapper),  # 유효 카테고리 개수
                            "i2s": mapper,
                        }
                    )
            # mixed 컬럼
            elif index in self.mixed_columns.keys():
                meta.append(
                    {
                        "name": index,
                        "type": "mixed",
                        "min": column.min(),
                        "max": column.max(),
                        "modal": self.mixed_columns[index],  # given -9999999 for nan
                    }
                )
            # 연속형 컬럼
            else:
                meta.append(
                    {
                        "name": index,
                        "type": "continuous",
                        "min": column.min(),
                        "max": column.max(),
                    }
                )

        return meta

    def fit(self, train_data: pd.DataFrame):
        """VGM(MSN) 모델 피팅"""
        data = train_data.values
        self.meta = self.get_metadata(train_data)
        model = []  # VGM Model 저장 영역
        self.output_info = []  # 데이터 인코딩 후 출력 정보 List[tuple]
        self.output_dim = 0  # 데이터 인코딩 후 차원
        self.valid_mode_flags = []  # 컬럼별 MSN 모드별 유효여부 저장 영역 List[bool]
        self.filter_arr = []

        self.logger.info("[Transformer]: fitting start ...")
        st = 0  # encoding 후 인덱스 추적용
        for id_, info in enumerate(tqdm(self.meta)):
            train_columns = train_data.columns
            colname = train_columns[id_]
            current_arr = data[:, id_]
            if info["type"] == "continuous":
                # num. type & single Gaussian 이 아닌 경우: MSN (VGM)
                if id_ not in self.general_columns:
                    gm = BayesianGaussianMixture(
                        n_components=self.n_clusters,
                        weight_concentration_prior_type="dirichlet_process",
                        weight_concentration_prior=0.001,
                        max_iter=100,
                        n_init=1,
                        random_state=RANDOM_SEED,
                    )
                    gm.fit(current_arr.reshape([-1, 1]))

                    # 유효한 모드 인디케이팅
                    # lsw: 원본코드에서 mode_freq 찾는 부분 불필요... 심지어 freq 는 사용도 안함
                    model.append(gm)
                    comp = (
                        gm.weights_ > self.eps
                    )  # weight 가 epsilon 보다 크고 데이터 상 존재하는 mode(comp) 만 True
                    self.valid_mode_flags.append(comp)

                    self.output_info += [
                        (
                            1,
                            "tanh",
                            colname,
                            "msn",
                        ),  # for alpha_i  (len(alpha_i), activaton_fn, col_name, GT_indicator) // 'msn' 이것 'get_tcol_idx_st_ed_tuple' 에 쓰임
                        (
                            np.sum(comp),
                            "softmax",
                            colname,
                        ),  # for beta_i  (len(beta_i), activaton_fn, col_name)
                    ]
                    st_delta = 1 + np.sum(comp)
                    self.output_dim += st_delta
                    info.update({"st": st, "end": st + st_delta})
                    st += st_delta
                else:  # single Gaussian 또는 large num cate 인 경우: GT
                    model.append(None)
                    self.valid_mode_flags.append(None)
                    self.output_info += [
                        (1, "tanh", colname, "gt")
                    ]  # for alpha_i // gt는 beta_i 불필요
                    st_delta = 1
                    self.output_dim += st_delta
                    info.update({"st": st, "end": st + st_delta})
                    st += st_delta

            # mixed type 인 경우: MSN (VGM)
            elif info["type"] == "mixed":
                # modal(범주/Nan/null...) 포함 피팅
                gm1 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=RANDOM_SEED,
                )
                # modal(범주/Nan/null...) 제거 후 피팅
                gm2 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=RANDOM_SEED,
                )

                gm1.fit(current_arr.reshape([-1, 1]))

                # modal값이 아닌 데이터 샘플만 필터링 (T/F indicating)
                #  -> mixed 애서 continuous 부분만
                filter_arr = ~np.isin(current_arr, info["modal"])

                gm2.fit(current_arr[filter_arr].reshape([-1, 1]))
                info["filter_arr"] = filter_arr
                model.append((gm1, gm2))
                comp = (
                    gm2.weights_ > self.eps
                )  # weight 가 epsilon 보다 크고 데이터 상 존재하는 mode(comp) 만 True
                self.valid_mode_flags.append(comp)

                self.output_info += [
                    (1, "tanh", colname, "msn"),  # for alpha_i
                    (
                        np.sum(comp) + len(info["modal"]),
                        "softmax",
                        colname,
                    ),  # for beta_i
                ]
                st_delta = 1 + np.sum(comp) + len(info["modal"])
                self.output_dim += st_delta
                info.update({"st": st, "end": st + st_delta})
                st += st_delta

            # categorical type 인 경우: one-hot
            else:
                model.append(None)
                self.valid_mode_flags.append(None)
                self.output_info += [(info["size"], "softmax", colname)]  # for gamma_i

                st_delta = info["size"]
                self.output_dim += st_delta
                info.update({"st": st, "end": st + st_delta})
                st += st_delta

        self.model = model  # VGM Model 저장 영역
        self.is_fit_ = True
        self.logger.info("[Transformer]: fitting end ...")

    def transform(
        self, data: np.ndarray, ispositive=False, positive_list=None
    ) -> np.array:
        """encode data row"""
        self.use_parallel_transfrom = False
        if self.use_parallel_transfrom:
            return self._parallel_transform(data, ispositive, positive_list)
        else:
            return self._transform(data, ispositive, positive_list)

    def _parallel_transform(self, data):
        pass

    # def _parallel_transform(self, data):
    #     # worker 정의
    #     def worker(data_partition, queue, *args):
    #         # 여기에 처리 로직의 일부를 넣는다.
    #         result = None  # 해당 작업자의 처리 결과
    #         queue.put(result)

    #     num_workers = int(cpu_count() * 0.9)
    #     queue = Queue()

    #     # processes = []
    #     # for partition in partitions:
    #     #     p = Process(target=worker, args=(partition, queue))
    #     #     processes.append(p)
    #     #     p.start()

    #     # results = [queue.get() for _ in processes]

    #     # for p in processes:
    #     #     p.join()

    #     # Pool을 사용하여 병렬 처리
    #     with Pool(num_workers) as pool:
    #         results = pool.starmap(
    #             worker, [(partition, queue) for partition in partitions]
    #         )

    #     return np.concatenate(results, axis=0)

    def _transform(
        self, data: np.ndarray, ispositive=False, positive_list=None
    ) -> np.ndarray:
        self.ordering = {}  # 높은 확률 모드 순으로 리오더링 하기 위해 활용,
        values = []
        for info in tqdm(self.meta):
            id_ = info["name"]
            arr = data[:, id_]
            encoded = encode_column(self, arr, info, ispositive, positive_list)
            if isinstance(encoded, list):
                values += encoded
            else:
                values.append(encoded)
        return np.concatenate(values, axis=1)

    def invrese_transform(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """decode data row"""
        self.use_parallel_inverse_transfrom = False
        if self.use_parallel_inverse_transfrom:
            return self._parallel_inverse_transform(data)
        else:
            return self._inverse_transform(data)

    def _parallel_inverse_transform(self, data: np.ndarray) -> Tuple[np.ndarray, list]:
        """generated data 를 원본 데이터 형태로 parallel decode"""
        pass

    def _inverse_transform(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """generated data 를 원본 데이터 형태로 decode"""
        values = []
        invalid_ids_merged = []  # fake 를 decode 해보니 컬럼 조건(min, max) 에 위배되는 경우
        st = 0
        for info in self.meta:
            id_ = info["name"]  # 0, 1, 2 ... 순서대로 들어있음
            st, end = info["st"], info["end"]
            arr = data[:, st:end]
            decoded, _invalid_ids = decode_column(self, arr, info)
            values.append(decoded)  # decoded (N,) 1d-array
            invalid_ids_merged += _invalid_ids

        values = np.column_stack(values)  # make values (N, #columns) 2d-array
        invalid_ids_merged = np.unique(invalid_ids_merged)
        all_ids = np.arange(0, len(data))
        valid_ids = list(set(all_ids) - set(invalid_ids_merged))

        # return values, len(invalid_ids_merged)
        return values[valid_ids], len(invalid_ids_merged)

    def save(self, mpath: str):
        assert self.is_fit_, "only fitted model could be saved, fit first please..."
        os.makedirs(mpath, exist_ok=True)
        mpath = os.path.join(mpath, "transformer.pickle")

        with open(mpath, "wb") as f:
            pickle.dump(self, f)
            self.logger.info(f"[DataTransformer]: Model saved at {mpath}")
        return

    @staticmethod
    def load(mpath: str) -> "DataTransformer":
        if not os.path.exists(mpath):
            raise FileNotFoundError(f"[DataTransformer]: Model not exists at {mpath}")
        with open(mpath, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model


class ImageTransformer:
    def __init__(self, side):
        self.height = side

    # zero-padding 후 sqaure matrix 형태로 변환
    def transform(self, data):
        """Transform to shape (#batch, C, H, W)"""
        if self.height * self.height > len(data[0]):
            padding = torch.zeros(
                (len(data), self.height * self.height - len(data[0]))
            ).to(data.device)
            data = torch.cat([data, padding], axis=1)

        return data.view(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        data = data.view(-1, self.height * self.height)

        return data
