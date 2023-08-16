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
        raw_df: pd.DataFrame,
        *,
        categorical_columns: list = None,
        log_columns: list = None,
        mixed_columns: dict = None,  # {"capital-loss": [0.0], "capital-gain": [0.0]} 포맷으로 입력
        general_columns: list = None,  # categorical 중 one-hot 안하고 GT 적용할 컬럼들도 같이 적어줘야 함
        non_categorical_columns: list = None,  # categorical 중 one-hot 안하고 MSN 적용할 컬럼들...  lsw: 헷갈림. 이름 변경필요
        integer_columns: list = None,
        problem_type: dict = None,  # {"Classification": "income"} 포맷으로 입력
        params_ctabgan: dict = None,  # CTABGANSynthesizer 에 적용할 파라메터 딕셔너리
    ):
        self.__name__ = "CTABGAN"
        # set initial params
        if categorical_columns is None:
            categorical_columns = []
        if log_columns is None:
            log_columns = []
        if mixed_columns is None:
            mixed_columns = {}
        if general_columns is None:
            general_columns = []
        if non_categorical_columns is None:
            non_categorical_columns = []
        if integer_columns is None:
            integer_columns = []
        if problem_type is None:
            problem_type = {}
        if params_ctabgan is None:
            params_ctabgan = {}

        # for logger
        self.logger = logging.getLogger()

        self.synthesizer = CTABGANSynthesizer(**params_ctabgan)
        self.raw_df = raw_df
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type

    def fit(self):
        start_time = time.time()
        self.logger.info("[CTABGAN]: build data preprocessor start")
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
        )
        self.logger.info("[CTABGAN]: build data preprocessor end")
        self.logger.info("[CTABGAN]: fit synthesizer start")
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
        self.logger.info("[CTABGAN]: fit synthesizer end")
        end_time = time.time()
        self.logger.info(f"Finished training in {end_time - start_time} seconds.")

    def generate_samples(self):
        sample = self.synthesizer.sample(len(self.raw_df))
        sample_df = self.data_prep.inverse_prep(sample)

        return sample_df
