import numpy as np
import pandas as pd
from sklearn import preprocessing


class DataPrep(object):
    def __init__(
        self,
        raw_df: pd.DataFrame,
        categorical: list,
        log: list,
        mixed: dict,
        general: list,
        non_categorical: list,
        integer: list,
        ptype: dict,
    ):
        self.categorical_columns = categorical  # cate type
        self.log_columns = log  # num type 중 long-tail dist.
        self.mixed_columns = mixed  # num + cate/null
        self.general_columns = (
            general  # num type 중 single-mode gaussian | cate type 중 high class num
        )
        self.non_categorical_columns = (
            non_categorical  # categorical 중 one-hot 안하고 MSN 적용할 컬럼들
        )
        self.integer_columns = integer
        # 컬럼 인덱스 저장
        self.column_types = dict()
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}
        self.column_types["general"] = []
        self.column_types["non_categorical"] = []
        self.lower_bounds = {}
        self.label_encoder_list = []

        target_col = list(ptype.values())[0]  # ptype - {"Classification": "income"}
        if target_col is not None:
            # target(y) 맨 뒤로 보내주기
            y_real = raw_df[target_col]
            X_real = raw_df.drop(columns=[target_col])
            X_real[target_col] = y_real
            self.df = X_real
        else:
            self.df = raw_df

        # data imputation
        # lsw: 이 부분은 데이터셋에 따라 변경 필요
        # self.df = self.df.replace(r" ", np.nan)
        self.df = self.df.fillna("empty")
        # # 데이터 타입에 따른 fillna 수행
        # for col in self.df.columns:
        #     if self.df[col].dtype == "object":  # 문자열 타입
        #         self.df[col].fillna("empty", inplace=True)
        #     elif self.df[col].dtype in ["int", "float"]:  # 숫자 타입
        #         self.df[col].fillna(-9999999, inplace=True)

        all_columns = set(self.df.columns)
        irrelevant_missing_columns = set(self.categorical_columns)
        # numeric, mixed type
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)

        # numeric, mixed type 컬럼 중 null 값 존재시에 -9999999 등록/추가하고 mixed 타입으로 변경
        for i in relevant_missing_columns:
            if "empty" in self.df[i].values:
                self.df[i] = self.df[i].replace(
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
                for idx, val in enumerate(self.df[log_column].values):
                    if val != -9999999:
                        valid_indices.append(idx)
                eps = 1
                lower = np.min(self.df[log_column].iloc[valid_indices].values)
                self.lower_bounds[log_column] = lower
                # 그 후 로그변환 수행 (유효값들만)
                if lower > 0:

                    def apply_log(x):
                        return np.log(x) if x != -9999999 else -9999999

                    self.df[log_column] = self.df[log_column].apply(apply_log)
                    # mixed_columns 이면서 log 분포인경우 모드들도 로그변환 필요해서 추가
                    # lsw: load/save 타 환경에서 하는 경우 역시 floating point error 발생...
                    # mixed-log 를 강건하게 처리할 코드 필요 - 모드는 로그변환하면 안됨
                    if log_column in self.mixed_columns.keys():
                        self.mixed_columns[log_column] = [
                            apply_log(x) for x in self.mixed_columns[log_column]
                        ]
                elif lower == 0:

                    def apply_log(x):
                        return np.log(x + eps) if x != -9999999 else -9999999

                    self.df[log_column] = self.df[log_column].apply(apply_log)
                    if log_column in self.mixed_columns.keys():
                        self.mixed_columns[log_column] = [
                            apply_log(x) for x in self.mixed_columns[log_column]
                        ]
                else:

                    def apply_log(x):
                        return np.log(x - lower + eps) if x != -9999999 else -9999999

                    self.df[log_column] = self.df[log_column].apply(apply_log)
                    if log_column in self.mixed_columns.keys():
                        self.mixed_columns[log_column] = [
                            apply_log(x) for x in self.mixed_columns[log_column]
                        ]

        for column_index, column in enumerate(self.df.columns):
            # 카테고리 컬럼인경우 더미화
            # 저자의 저서는 이미 수치형으로 인코딩된 카테공리형식의 컬럼이 들어옴
            if column in self.categorical_columns:
                label_encoder = preprocessing.LabelEncoder()
                label_encoder.fit(self.df[column])
                # label_encoder.fit(self.df[column].astype(str))
                self.df[column] = label_encoder.transform(self.df[column])
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

            elif column in self.general_columns:
                self.column_types["general"].append(column_index)

        super().__init__()

    def inverse_prep(self, data, eps=1):
        df_sample = pd.DataFrame(data, columns=self.df.columns)

        # 카테고리 컬럼 역변환
        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"]
            df_sample[self.label_encoder_list[i]["column"]] = le.inverse_transform(
                df_sample[self.label_encoder_list[i]["column"]].astype(int)
            )

        # 로그분포 역변환
        if self.log_columns:
            for column in df_sample:
                if column in self.log_columns:
                    lower_bound = self.lower_bounds[column]
                    if lower_bound > 0:
                        df_sample[column].apply(lambda x: np.exp(x))
                    elif lower_bound == 0:
                        df_sample[column] = df_sample[column].apply(
                            lambda x: np.ceil(np.exp(x) - eps)
                            if (np.exp(x) - eps) < 0
                            else (np.exp(x) - eps)
                        )
                    else:
                        df_sample[column] = df_sample[column].apply(
                            lambda x: np.exp(x) - eps + lower_bound
                        )

        # 정수형 타입 반올림
        if self.integer_columns:
            for column in self.integer_columns:
                df_sample[column] = np.round(df_sample[column].values)
                df_sample[column] = df_sample[column].astype(int)

        df_sample.replace(-9999999, np.nan, inplace=True)
        df_sample.replace("empty", np.nan, inplace=True)

        return df_sample
