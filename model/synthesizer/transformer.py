import numpy as np
import pandas as pd
import torch
from sklearn.mixture import BayesianGaussianMixture
import logging
from tqdm.auto import tqdm
from typing import List


class DataTransformer:
    def __init__(
        self,
        train_data: pd.DataFrame,
        categorical_list: list = [],
        mixed_dict: dict = {},
        general_list: list = [],
        non_categorical_list: list = [],
        n_clusters: int = 10,
        eps: float = 0.005,
    ):
        self.logger = logging.getLogger()
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps
        self.train_data = train_data
        self.categorical_columns = categorical_list
        self.mixed_columns = mixed_dict
        self.general_columns = general_list
        self.non_categorical_columns = non_categorical_list

    def get_metadata(self) -> List[dict]:
        meta = []
        self.logger.info("[Transformer]: get metadata ...")
        for index in tqdm(range(self.train_data.shape[1])):
            column = self.train_data.iloc[:, index]
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
                        "modal": self.mixed_columns[
                            index
                        ],  # given 0.0 or -9999999 for nan
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

    def fit(self):
        """VGM(MSN) 모델 피팅"""
        data = self.train_data.values
        self.meta = self.get_metadata()
        model = []  # VGM Model 저장 영역
        self.ordering = []  #
        self.output_info = []  # 데이터 인코딩 후 출력 정보 List[tuple]
        self.output_dim = 0  # 데이터 인코딩 후 차원
        self.components = []  # 컬럼별 MSN 모드별 유효여부 저장 영역 List[bool]
        self.filter_arr = []

        self.logger.info("[Transformer]: fitting start ...")
        for id_, info in enumerate(tqdm(self.meta)):
            if info["type"] == "continuous":
                # num. type & single Gaussian 이 아닌 경우: MSN (VGM)
                if id_ not in self.general_columns:
                    gm = BayesianGaussianMixture(
                        n_components=self.n_clusters,
                        weight_concentration_prior_type="dirichlet_process",
                        weight_concentration_prior=0.001,
                        max_iter=100,
                        n_init=1,
                        random_state=42,
                    )
                    gm.fit(data[:, id_].reshape([-1, 1]))

                    # 유효한 모드 인디케이팅
                    # lsw: 원본코드에서 mode_freq 찾는 부분 불필요... 심지어 freq 는 사용도 안함
                    model.append(gm)
                    comp = (
                        gm.weights_ > self.eps
                    )  # weight 가 epsilon 보다 크고 데이터 상 존재하는 mode(comp) 만 True
                    self.components.append(comp)
                    # 원본 코드
                    # mode_freq = (
                    #     pd.Series(gm.predict(data[:, id_].reshape([-1, 1])))
                    #     .value_counts()
                    #     .keys()
                    # )
                    # model.append(gm)
                    # old_comp = gm.weights_ > self.eps
                    # comp = []  # weight 가 epsilon 보다 크고 데이터 상 존재하는 mode(comp) 만 True
                    # for i in range(self.n_clusters):
                    #     if (i in (mode_freq)) & old_comp[i]:  # lsw: 이거는 당연히 돼야하는거 아님? i in (mode_freq)
                    #         comp.append(True)
                    #     else:
                    #         comp.append(False)
                    # self.components.append(comp)

                    self.output_info += [
                        (
                            1,
                            "tanh",
                            "no_g",
                        ),  # for alpha_i  (len(alpha_i), activaton_fn, GT_indicator) // 'no_g' 이것 'get_tcol_idx_st_ed_tuple' 에 쓰임
                        (
                            np.sum(comp),
                            "softmax",
                        ),  # for beta_i  (len(beta_i), activaton_fn)
                    ]
                    self.output_dim += 1 + np.sum(comp)
                else:  # single Gaussian 또는 large num cate 인 경우: GT
                    model.append(None)
                    self.components.append(None)
                    # "yes_g"는 GT 수행의 의미인듯
                    self.output_info += [
                        (1, "tanh", "yes_g")
                    ]  # for alpha_i // gt는 beta_i 불필요
                    self.output_dim += 1

            # mixed type 인 경우: MSN (VGM)
            elif info["type"] == "mixed":
                # modal(범주/Nan/null...) 포함 피팅
                gm1 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42,
                )
                # modal(범주/Nan/null...) 제거 후 피팅
                gm2 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42,
                )

                gm1.fit(data[:, id_].reshape([-1, 1]))

                # modal값이 아닌 데이터 샘플만 필터링 (T/F indicating)
                #  -> mixed 애서 continuous 부분만
                filter_arr = []
                for element in data[:, id_]:
                    if element not in info["modal"]:
                        filter_arr.append(True)
                    else:
                        filter_arr.append(False)

                gm2.fit(data[:, id_][filter_arr].reshape([-1, 1]))
                self.filter_arr.append(filter_arr)
                model.append((gm1, gm2))
                comp = (
                    gm2.weights_ > self.eps
                )  # weight 가 epsilon 보다 크고 데이터 상 존재하는 mode(comp) 만 True
                self.components.append(comp)
                # mode_freq = (
                #     pd.Series(gm2.predict(data[:, id_][filter_arr].reshape([-1, 1])))
                #     .value_counts()
                #     .keys()
                # )
                # old_comp = gm2.weights_ > self.eps
                # comp = []
                # for i in range(self.n_clusters):
                #     if (i in (mode_freq)) & old_comp[i]:
                #         comp.append(True)
                #     else:
                #         comp.append(False)
                # self.components.append(comp)

                self.output_info += [
                    (1, "tanh", "no_g"),  # for alpha_i
                    (np.sum(comp) + len(info["modal"]), "softmax"),  # for beta_i
                ]
                self.output_dim += 1 + np.sum(comp) + len(info["modal"])

            # categorical type 인 경우: one-hot
            else:
                model.append(None)
                self.components.append(None)
                self.output_info += [(info["size"], "softmax")]  # for gamma_i
                self.output_dim += info["size"]
        self.model = model  # VGM Model 저장 영역
        self.logger.info("[Transformer]: fitting end ...")

    def transform(
        self, data: np.ndarray, ispositive=False, positive_list=None
    ) -> np.ndarray:
        """encode data row"""
        values = []
        mixed_counter = 0
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info["type"] == "continuous":
                # MSN 적용 대상 컬럼인 경우: get alpha_i, beta_i
                if id_ not in self.general_columns:
                    current = current.reshape([-1, 1])
                    means = self.model[id_].means_.reshape((1, self.n_clusters))
                    stds = np.sqrt(self.model[id_].covariances_).reshape(
                        (1, self.n_clusters)
                    )
                    # features: 각 mode 별 정규화 (alpha_i candidates)
                    features = np.empty(shape=(len(current), self.n_clusters))
                    if ispositive is True:
                        if id_ in positive_list:
                            features = np.abs(current - means) / (
                                4 * stds
                            )  # lsw: 0으로 자르는 것도 아니고 절대값? 약간 애매하지만 넘어가자
                    else:
                        features = (current - means) / (4 * stds)

                    probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                    n_opts = sum(self.components[id_])  # n_opts: 해당 컬럼의 유효 mode 개수
                    features = features[
                        :, self.components[id_]
                    ]  # features: 유효 mode 의 alpha_i 만 필터링 (N * #valid_mode)
                    probs = probs[
                        :, self.components[id_]
                    ]  # probs: 유효 mode 의 확률만 필터링 (N * #valid_mode)

                    # 해당 확률 기반 최적 mode 선택
                    opt_sel = np.zeros(len(data), dtype="int")
                    for i in range(len(data)):
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
                    probs_onehot[
                        np.arange(len(probs)), opt_sel
                    ] = 1  # mode-indicator beta_i

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
                    self.ordering.append(largest_indices)  # (#valid_mode,)

                    for id, val in enumerate(largest_indices):
                        re_ordered_phot[:, id] = probs_onehot[:, val]

                    values += [
                        features,
                        re_ordered_phot,
                    ]  # alpha_i (N * 1), beta_i (N * #valid_mode)

                # GT 적용 대상 컬럼인 경우: transform to x_t
                # lsw: 추후 데이터 미니 배치로 넣으면 이부분도 최소, 최대값 저장했다가 다시 쓰도록 리팩토링 해야할지도 ...
                else:
                    self.ordering.append(None)

                    if id_ in self.non_categorical_columns:
                        info["min"] = -1e-3
                        info["max"] = info["max"] + 1e-3

                    current = (current - (info["min"])) / (info["max"] - info["min"])
                    current = current * 2 - 1
                    current = current.reshape([-1, 1])
                    values.append(current)  # alpha_i (N * 1)

            # MSN 적용 대상 mixed 컬럼인 경우: get alpha_i, beta_i
            elif info["type"] == "mixed":
                means_0 = self.model[id_][0].means_.reshape([-1])
                stds_0 = np.sqrt(self.model[id_][0].covariances_).reshape([-1])

                zero_std_list = []
                means_needed = []
                stds_needed = []

                # -9999999 외의 modal 값에 대한 mean, sqrt 계산
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
                for i, j, k in zip(info["modal"], means_needed, stds_needed):
                    this_val = np.abs(i - j) / (4 * k)
                    mode_vals.append(this_val)
                if -9999999 in info["modal"]:
                    mode_vals.append(0)

                # modal 이 아닌 continuous 값의 mode 에 대한 alpha_i, beta_i 계산
                current = current.reshape([-1, 1])
                filter_arr = self.filter_arr[mixed_counter]
                current = current[filter_arr]

                means = self.model[id_][1].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_][1].covariances_).reshape(
                    (1, self.n_clusters)
                )
                features = np.empty(shape=(len(current), self.n_clusters))
                if ispositive is True:
                    if id_ in positive_list:
                        features = np.abs(current - means) / (4 * stds)
                else:
                    features = (current - means) / (4 * stds)

                probs = self.model[id_][1].predict_proba(current.reshape([-1, 1]))
                n_opts = sum(self.components[id_])
                features = features[:, self.components[id_]]
                probs = probs[:, self.components[id_]]

                opt_sel = np.zeros(len(current), dtype="int")
                for i in range(len(current)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                idx = np.arange((len(features)))
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -0.99, 0.99)
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1

                extra_bits = np.zeros([len(current), len(info["modal"])])
                temp_probs_onehot = np.concatenate(
                    [extra_bits, probs_onehot], axis=1
                )  # modal 을 MSN 앞에 붙이네
                final = np.zeros(
                    [
                        len(data),
                        1 + probs_onehot.shape[1] + len(info["modal"]),
                    ]  # (N * 1+ #vaild_mode + #modal)  // +1 은 alpha 위한 것
                )

                # sjy: 얘는 뭐지...???
                #  -> lsw: final 내에 alpha, beta 채우는 과정
                features_curser = 0
                for idx, val in enumerate(data[:, id_]):
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

                just_onehot = final[:, 1:]
                re_ordered_jhot = np.zeros_like(just_onehot)
                col_sums = just_onehot.sum(axis=0)
                largest_indices = np.argsort(-1 * col_sums)
                # n = just_onehot.shape[1]
                # largest_indices = np.argsort(-1 * col_sums)[:n]
                self.ordering.append(largest_indices)

                for id, val in enumerate(largest_indices):
                    re_ordered_jhot[:, id] = just_onehot[:, val]

                final_features = final[:, 0].reshape([-1, 1])
                values += [final_features, re_ordered_jhot]  # alpha_i, beta_i
                mixed_counter = mixed_counter + 1

            # categorical 컬럼인 경우: get one-hot
            else:
                self.ordering.append(None)
                col_t = np.zeros([len(data), info["size"]])
                idx = list(map(info["i2s"].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)  # gamma_i

        return np.concatenate(values, axis=1)

    # fake (generated) 를 원본 데이터 형태로 decode
    def inverse_transform(self, data):
        data_t = np.zeros([len(data), len(self.meta)])
        invalid_ids = []  # fake 를 decode 해보니 컬럼 조건(min, max) 에 위배되는 경우
        st = 0
        for id_, info in enumerate(self.meta):
            if info["type"] == "continuous":
                # not mixed MSN 역변환
                if id_ not in self.general_columns:
                    u = data[:, st]
                    v = data[:, st + 1 : st + 1 + np.sum(self.components[id_])]
                    order = self.ordering[id_]
                    v_re_ordered = np.zeros_like(v)

                    for id, val in enumerate(order):
                        v_re_ordered[:, val] = v[:, id]

                    v = v_re_ordered

                    u = np.clip(u, -1, 1)
                    v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                    v_t[:, self.components[id_]] = v
                    v = v_t
                    st += 1 + np.sum(self.components[id_])
                    means = self.model[id_].means_.reshape([-1])
                    stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                    p_argmax = np.argmax(v, axis=1)
                    std_t = stds[p_argmax]
                    mean_t = means[p_argmax]
                    tmp = u * 4 * std_t + mean_t

                    for idx, val in enumerate(tmp):
                        if (val < info["min"]) | (val > info["max"]):
                            invalid_ids.append(idx)

                    if id_ in self.non_categorical_columns:
                        tmp = np.round(tmp)

                    data_t[:, id_] = tmp

                # GT 역변환
                else:
                    u = data[:, st]
                    u = (u + 1) / 2
                    u = np.clip(u, 0, 1)
                    u = u * (info["max"] - info["min"]) + info["min"]
                    if id_ in self.non_categorical_columns:
                        data_t[:, id_] = np.round(u)
                    else:
                        data_t[:, id_] = u

                    st += 1

            # mixed MSN 역변환
            elif info["type"] == "mixed":
                u = data[:, st]
                full_v = data[
                    :,
                    (st + 1) : (st + 1)
                    + len(info["modal"])
                    + np.sum(self.components[id_]),
                ]
                order = self.ordering[id_]
                full_v_re_ordered = np.zeros_like(full_v)

                for id, val in enumerate(order):
                    full_v_re_ordered[:, val] = full_v[:, id]

                full_v = full_v_re_ordered

                mixed_v = full_v[:, : len(info["modal"])]
                v = full_v[:, -np.sum(self.components[id_]) :]

                u = np.clip(u, -1, 1)
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v
                v = np.concatenate([mixed_v, v_t], axis=1)

                st += 1 + np.sum(self.components[id_]) + len(info["modal"])
                means = self.model[id_][1].means_.reshape([-1])
                stds = np.sqrt(self.model[id_][1].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)

                result = np.zeros_like(u)

                for idx in range(len(data)):
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

                data_t[:, id_] = result

            # one-hot 역변환
            else:
                current = data[:, st : st + info["size"]]
                st += info["size"]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info["i2s"].__getitem__, idx))

        invalid_ids = np.unique(np.array(invalid_ids))
        all_ids = np.arange(0, len(data))
        valid_ids = list(set(all_ids) - set(invalid_ids))

        return data_t[valid_ids], len(invalid_ids)


class ImageTransformer:
    def __init__(self, side):
        self.height = side

    # zero-padding 후 sqaure matrix 형태로 변환
    def transform(self, data):
        if self.height * self.height > len(data[0]):
            padding = torch.zeros(
                (len(data), self.height * self.height - len(data[0]))
            ).to(data.device)
            data = torch.cat([data, padding], axis=1)

        return data.view(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        data = data.view(-1, self.height * self.height)

        return data
