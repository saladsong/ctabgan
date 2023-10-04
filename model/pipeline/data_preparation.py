import numpy as np
import pandas as pd
import pickle
import os
import logging
from sklearn import preprocessing


class DataPrep(object):
    def __init__(
        self,
        categorical: list,
        log: list,
        mixed: dict,
        general: list,
        skewed: list,  # jys: added
        non_categorical: list,
        integer: list,
        ptype: dict = None,
    ):
        self.categorical_columns = categorical  # cate type
        self.log_columns = log  # num type 중 long-tail dist.
        self.mixed_columns = mixed  # num + cate/null
        self.general_columns = (
            general  # num type 중 single-mode gaussian | cate type 중 high class num
        )
        self.skewed_columns = skewed  # jys: added
        self.non_categorical_columns = (
            non_categorical  # categorical 중 one-hot 안하고 MSN 적용할 컬럼들
        )
        self.integer_columns = integer
        # 컬럼 인덱스 저장
        self.column_types = dict()
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}
        self.column_types["general"] = []
        self.column_types["skewed"] = []
        self.column_types["non_categorical"] = []
        self.lower_bounds = {}
        self.label_encoder_list = []
        self.ptype = ptype
        self.is_fit_ = False
        self.logger = logging.getLogger()
        super().__init__()

    def prep(
        self,
        raw_df: pd.DataFrame,
    ) -> pd.DataFrame:
        ptype = self.ptype
        if ptype is not None:
            target_col = list(ptype.values())[0]  # ptype - {"Classification": "income"}
            if target_col is not None:
                # target(y) 맨 뒤로 보내주기
                y_real = raw_df[target_col]
                X_real = raw_df.drop(columns=[target_col])
                X_real[target_col] = y_real
                df = X_real
            else:
                df = raw_df

        # data imputation
        # lsw: 이 부분은 데이터셋에 따라 변경 필요
        # df = df.replace(r" ", np.nan)
        df = df.fillna("empty")
        # # 데이터 타입에 따른 fillna 수행
        # for col in df.columns:
        #     if df[col].dtype == "object":  # 문자열 타입
        #         df[col].fillna("empty", inplace=True)
        #     elif df[col].dtype in ["int", "float"]:  # 숫자 타입
        #         df[col].fillna(-9999999, inplace=True)

        all_columns = set(df.columns)
        irrelevant_missing_columns = set(self.categorical_columns)
        # numeric, mixed type
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)

        # numeric, mixed type 컬럼 중 null 값 존재시에 -9999999 등록/추가하고 mixed 타입으로 변경
        for i in relevant_missing_columns:
            if "empty" in df[i].values:
                df[i] = df[i].replace(
                    "empty", -9999999
                )  # numeric, mixed type 은 null 값 -9999999 로 변경
                if i in self.mixed_columns.keys():
                    self.mixed_columns[i].append(-9999999)
                else:
                    self.mixed_columns[i] = [-9999999]

        # log 분포 컬럼 전처리
        if len(self.log_columns) > 0:
            for log_column in self.log_columns:
                # 유효한 값만 탐색하며 최소값 뽑기
                valid_indices = []
                for idx, val in enumerate(df[log_column].values):
                    if val != -9999999 and val not in self.mixed_columns.get(
                        log_column, []
                    ):
                        valid_indices.append(idx)
                eps = 1
                lower = np.min(df[log_column].iloc[valid_indices].values)
                self.lower_bounds[log_column] = lower
                # 그 후 로그변환 수행 (유효값들만)
                if lower > 0:
                    df.loc[valid_indices, log_column] = df.loc[
                        valid_indices, log_column
                    ].apply(lambda x: np.log(x))
                elif lower == 0:
                    df.loc[valid_indices, log_column] = df.loc[
                        valid_indices, log_column
                    ].apply(lambda x: np.log(x + eps))
                else:
                    df.loc[valid_indices, log_column] = df.loc[
                        valid_indices, log_column
                    ].apply(lambda x: np.log(x - lower + eps))

        for column_index, column in enumerate(df.columns):
            # 카테고리 컬럼인경우 더미화
            # 저자의 저서는 이미 수치형으로 인코딩된 카테고리 형식의 컬럼이 들어옴
            if column in self.categorical_columns:
                label_encoder = preprocessing.LabelEncoder()
                label_encoder.fit(df[column])
                # label_encoder.fit(df[column].astype(str))
                df[column] = label_encoder.transform(df[column])
                current_label_encoder = {
                    "column": column,
                    "label_encoder": label_encoder,
                }
                self.label_encoder_list.append(current_label_encoder)
                self.column_types["categorical"].append(column_index)

                # sjy: 실제로는 cate. type 이지만 num. type 으로 간주하고 MSN 처리할 수 있게
                if column in self.non_categorical_columns:
                    self.column_types["non_categorical"].append(column_index)
                    # sjy: class 수가 너무 많은 경우 general 로 처리할 수 있게
                    if column in self.general_columns:
                        self.column_types["general"].append(column_index)

            elif column in self.mixed_columns:
                self.column_types["mixed"][column_index] = self.mixed_columns[column]

                if column in self.skewed_columns:  # jys: mixed & skewed
                    self.column_types["skewed"].append(column_index)

            elif column in self.general_columns:
                self.column_types["general"].append(column_index)

            elif column in self.skewed_columns:  # jys: continuous & skewed
                self.column_types["skewed"].append(column_index)

        self.columns = df.columns
        self.is_fit_ = True
        return df

    def inverse_prep(self, data, eps=1):
        df_sample = pd.DataFrame(data, columns=self.columns)

        # 카테고리 컬럼 역변환
        for le_dict in self.label_encoder_list:
            le = le_dict["label_encoder"]
            df_sample[le_dict["column"]] = le.inverse_transform(
                df_sample[le_dict["column"]].astype(int)
            )

        # 로그분포 역변환
        if self.log_columns:
            for column in df_sample:
                if column in self.log_columns:
                    lower_bound = self.lower_bounds[column]

                    # 유효한 값만 탐색하며 역변환
                    valid_indices = []
                    for idx, val in enumerate(df_sample[column].values):
                        if val != -9999999 and val not in self.mixed_columns.get(
                            column, []
                        ):
                            valid_indices.append(idx)

                    if lower_bound > 0:
                        df_sample.loc[valid_indices, column] = df_sample.loc[
                            valid_indices, column
                        ].apply(lambda x: np.exp(x))
                    elif lower_bound == 0:
                        df_sample.loc[valid_indices, column] = df_sample.loc[
                            valid_indices, column
                        ].apply(
                            lambda x: np.ceil(np.exp(x) - eps)
                            if (np.exp(x) - eps) < 0
                            else (np.exp(x) - eps)
                        )
                    else:
                        df_sample.loc[valid_indices, column] = df_sample.loc[
                            valid_indices, column
                        ].apply(lambda x: np.exp(x) - eps + lower_bound)

        # 정수형 타입 반올림
        if self.integer_columns:
            for column in self.integer_columns:
                df_sample[column] = np.round(df_sample[column].values)
                df_sample[column] = df_sample[column].astype(int)

        df_sample.replace(-9999999, np.nan, inplace=True)
        df_sample.replace("empty", np.nan, inplace=True)

        return df_sample

    def save(self, mpath: str):
        """확장자는 *.pickle 로"""
        assert self.is_fit_, "only fitted model could be saved, fit first please..."
        os.makedirs(os.path.dirname(mpath), exist_ok=True)

        with open(mpath, "wb") as f:
            pickle.dump(self, f)
            self.logger.info(f"[DataPrep]: Model saved at {mpath}")
        return

    @staticmethod
    def load(mpath: str) -> "DataPrep":
        if not os.path.exists(mpath):
            raise FileNotFoundError(f"[DataPrep]: Model not exists at {mpath}")
        with open(mpath, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model
