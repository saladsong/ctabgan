"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer
from model.synthesizer.transformer import DataTransformer

import warnings
import logging
import wandb

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
        transformer: DataTransformer = None,
        project: str = "ctabgan-project",  # wandb config
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

        # for logger
        self.logger = logging.getLogger()

        self.transformer = (
            transformer  # transformer 의 경우 VGM 학습에 오랜 시간이 걸리므로 저장 후 재사용 가능토록 함.
        )
        self.raw_df = raw_df
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        # for wandb
        self.project = project

        self.is_fit_ = False

    def pps(self):
        """본격적인 GAN 모델 학습에 앞서 입력 데이터의 인코딩에 사용되는 DataPrep, DataTransformer 를 적합 시키고 그 객체를 준비"""
        self.logger.info("[CTABGAN]: build data preprocessor start")
        # DataPrep: 데이터 전처리 (오래 걸리는 작업은 아님)
        #   - missing value 처리
        #   - mixed column modal 값 처리
        #   - log 변환
        #   - label encoding
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

        # set data transformer
        if self.transformer is None or not self.transformer.is_fit_:
            self.logger.info("[CTABGAN]: fit data transformer start")
            self.transformer = DataTransformer(
                categorical_list=self.data_prep.column_types["categorical"],
                mixed_dict=self.data_prep.column_types["mixed"],
                general_list=self.data_prep.column_types["general"],
                non_categorical_list=self.data_prep.column_types["non_categorical"],
            )
            self.transformer.fit(train_data=self.data_prep.df)
            self.logger.info("[CTABGAN]: fit data transformer end")
        else:
            self.logger.info("[CTABGAN]: use already fitted transformer")

    def fit(self, use_parallel_transfrom: bool = False, **kwargs):
        """CTABGAN 모델 학습"""
        start_time = time.time()
        self.synthesizer = CTABGANSynthesizer(**kwargs)
        self.params_ctabgan = kwargs

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=self.project,
            # track hyperparameters and run metadata
            config={
                "architecture": "orignal",
                **self.params_ctabgan,
            },
        )

        self.logger.info("[CTABGAN]: fit synthesizer start")
        # print(self.data_prep.df)
        # print(self.data_prep.column_types)
        self.synthesizer.fit(
            train_data=self.data_prep.df,
            data_transformer=self.transformer,
            ptype=self.problem_type,
            use_parallel_transfrom=use_parallel_transfrom,
        )
        self.logger.info("[CTABGAN]: fit synthesizer end")
        end_time = time.time()
        self.logger.info(f"Finished training in {end_time - start_time} seconds.")

        self.is_fit_ = True
        wandb.finish()

    def generate_samples(
        self,
        n: int = None,
        *,
        transformer: DataTransformer = None,
        use_parallel_inverse_transfrom: bool = False,
        resample_invalid: bool = True,
    ):
        assert self.is_fit_, "must fit the model first!!"

        if isinstance(transformer, DataTransformer):
            assert transformer.is_fit_, "you must use fitted data_transformer!!"
            _transformer = transformer
        else:
            _transformer = self.transformer

        if n is None:
            n = len(self.raw_df)
        sample = self.synthesizer.sample(
            n,
            _transformer,
            use_parallel_inverse_transfrom=use_parallel_inverse_transfrom,
            resample_invalid=resample_invalid,
        )
        sample_df = self.data_prep.inverse_prep(sample)

        return sample_df
