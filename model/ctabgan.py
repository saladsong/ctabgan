"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import warnings
import logging

warnings.filterwarnings("ignore")


class CTABGAN:
    def __init__(
        self,
        raw_csv_path="Real_Datasets/Adult.csv",
        test_ratio=0.20,
        categorical_columns=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "native-country",
            "income",
        ],
        log_columns=[],
        mixed_columns={"capital-loss": [0.0], "capital-gain": [0.0]},
        general_columns=["age"],  # categorical 중 one-hot 안하고 GT 적용할 컬럼들도 같이 적어줘야 함
        non_categorical_columns=[],  # categorical 중 one-hot 안하고 MSN 적용할 컬럼들...  lsw: 헷갈림. 이름 변경필요
        integer_columns=[
            "age",
            "fnlwgt",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ],
        problem_type={"Classification": "income"},
    ):
        self.__name__ = "CTABGAN"
        # for logger
        self.logger = logging.getLogger()

        self.synthesizer = CTABGANSynthesizer()
        self.raw_df = pd.read_csv(raw_csv_path)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type

    def fit(self):
        start_time = time.time()
        self.logger.info("[CTABGAN]: data preprocessor ready start")
        # DataPrep: 데이터 전처리
        self.data_prep = DataPrep(
            self.raw_df,
            self.categorical_columns,
            self.log_columns,
            self.mixed_columns,
            self.general_columns,
            self.non_categorical_columns,
            self.integer_columns,
            self.problem_type,
            self.test_ratio,
        )
        self.logger.info("[CTABGAN]: data preprocessor ready end")
        self.logger.info("[CTABGAN]: synthesizer fit start")
        # print(self.data_prep.df)
        # print(self.data_prep.column_types)
        self.synthesizer.fit(
            train_data=self.data_prep.df,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            general=self.data_prep.column_types["general"],
            non_categorical=self.data_prep.column_types["non_categorical"],
            ptype=self.problem_type,
        )
        self.logger.info("[CTABGAN]: synthesizer fit end")
        end_time = time.time()
        self.logger.info(f"Finished training in {end_time - start_time} seconds.")

    def generate_samples(self):
        sample = self.synthesizer.sample(len(self.raw_df))
        sample_df = self.data_prep.inverse_prep(sample)

        return sample_df
