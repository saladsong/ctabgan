"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import numpy as np
import time
from typing import Union, Optional
from tqdm.auto import tqdm
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer, Cond
from model.synthesizer.encoder import DataEncoder, ImageTransformer

import warnings
import logging
import wandb

warnings.filterwarnings("ignore")


class CTABGAN:
    def __init__(
        self,
        *,
        categorical_columns: list = None,
        log_columns: list = None,
        mixed_columns: dict = None,  # {"capital-loss": [0.0], "capital-gain": [0.0]} 포맷으로 입력
        general_columns: list = None,  # categorical 중 one-hot 안하고 GT 적용할 컬럼들도 같이 적어줘야 함
        skewed_columns: list = None,  # jys: contiunous / mixed 중 MSN 안하고 skew-norm 적용할 컬럼
        non_categorical_columns: list = None,  # categorical 중 one-hot 안하고 MSN 적용할 컬럼들...  lsw: 헷갈림. 이름 변경필요
        integer_columns: list = None,
        problem_type: dict = None,  # {"Classification": "income"} 포맷으로 입력
        encoder: DataEncoder = None,
        data_prep: DataPrep = None,
        project: str = "synthe",  # wandb config
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
        if skewed_columns is None:
            skewed_columns = []
        if non_categorical_columns is None:
            non_categorical_columns = []
        if integer_columns is None:
            integer_columns = []
        if problem_type is None:
            problem_type = {}

        # for logger
        self.logger = logging.getLogger()

        self.encoder = encoder  # encoder 의 경우 VGM 학습에 오랜 시간이 걸리므로 저장 후 재사용 가능토록 함.
        self.data_prep = data_prep  # data_prep 도 재사용
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.skewed_columns = skewed_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        # for wandb
        self.project = project

        self.is_fit_ = False

    def _prep(
        self,
        raw_df: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """본격적인 GAN 모델 학습에 앞서 입력 데이터의 인코딩에 사용되는 DataPrep 를 적합 시키고 그 객체를 준비
        Returns:
            Optional[pd.DataFrame]: data_prep 이후 데이터프레임. df_prep = self.data_prep.prep(raw_df)
        """
        # DataPrep: 데이터 전처리 (encoder 만큼 오래 걸리는 작업은 아님)
        #   - missing value 처리
        #   - mixed column modal 값 처리
        #   - log 변환
        #   - label encoding
        if self.data_prep is None or not self.data_prep.is_fit_:
            self.logger.info("[CTABGAN]: fit data_prep start")
            self.data_prep = DataPrep(
                self.categorical_columns,
                self.log_columns,
                self.mixed_columns,
                self.general_columns,
                self.skewed_columns,
                self.non_categorical_columns,
                self.integer_columns,
                self.problem_type,
            )
            df_prep = self.data_prep.prep(raw_df)
            self.logger.info("[CTABGAN]: fit data_prep end")
            return df_prep
        else:
            self.logger.info("[CTABGAN]: use already fitted data_prep")
            return None

    def _fit_encoder(self, df_prep=pd.DataFrame):
        """본격적인 GAN 모델 학습에 앞서 입력 데이터의 인코딩에 사용되는 DataEncoder 를 적합 시키고 그 객체를 준비"""
        # set data encoder
        if self.encoder is None or not self.encoder.is_fit_:
            self.logger.info("[CTABGAN]: fit data encoder start")
            self.encoder = DataEncoder(
                categorical_list=self.data_prep.column_types["categorical"],
                mixed_dict=self.data_prep.column_types["mixed"],
                general_list=self.data_prep.column_types["general"],
                skew_norm_list=self.data_prep.column_types["skewed"],
                non_categorical_list=self.data_prep.column_types["non_categorical"],
            )
            self.encoder.fit(train_data=df_prep)
            self.logger.info("[CTABGAN]: fit data encoder end")
        else:
            self.logger.info("[CTABGAN]: use already fitted encoder")

    def pps(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """본격적인 GAN 모델 학습에 앞서 입력 데이터의 인코딩에 사용되는 DataPrep, DataEncoder 를 적합 시키고 그 객체를 준비
        df_prep은 encoder를 학습하고 encoded 를 준비하기 위한 임시 중간 데이터. 처음부터 fitting 하는 경우만 활용하고, 저장된 모델을 불러와 재사용할 시에는 불필요
        """
        # data_prep 학습/재사용. 재사용시에는 df_prep가 None 값을 가짐
        df_prep: Optional[pd.DataFrame] = self._prep(raw_df)
        # encoder 학습/재사용
        self._fit_encoder(df_prep)
        return df_prep

    def fit(
        self,
        *,
        df_prep: pd.DataFrame = None,
        encoded_data: np.ndarray = None,
        n_jobs: Union[float, int] = None,
        wandb_exp_name: str = None,
        **kwargs,
    ):
        """CTABGAN 모델 학습"""
        self.logger.info("[CTABGAN]: fit synthesizer start")
        start_time = time.time()
        self.synthesizer = CTABGANSynthesizer(**kwargs)
        self.params_ctabgan = kwargs
        self.wandb_exp_name = wandb_exp_name

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
        # wandb 실행 이름 설정
        if wandb_exp_name is not None:
            wandb.run.name = wandb_exp_name
            wandb.run.save()

        # auxiliary classifier 타겟 컬럼 인덱스 찾기
        target_index = None
        if self.problem_type is not None:  # ex) {"Classification": "income"}
            pkind = list(self.problem_type.keys())[0]  # 일단은 맨 처음것만 사용중
            target_index = self.data_prep.columns.get_loc(
                self.problem_type[pkind]
            )  # data_prep 에서 target_col 맨 마지막으로 밀었음. lsw: 굳이 밀어야 하나?

        # 데이터 전처리(인코딩)하는 부분 (prep -> encoded)
        if encoded_data is None:
            assert isinstance(
                df_prep, pd.DataFrame
            ), "encoded_data 가 None 면 반드시 df_prep 을 입력해야 합니다."
            encoded_data = self.encoder.transform(df_prep.values, n_jobs=n_jobs)
        else:
            self.logger.info("[CTABGAN]: use input encoded data")

        self.synthesizer.fit(
            encoded_data=encoded_data,
            encoder=self.encoder,
            target_index=target_index,
        )
        self.logger.info("[CTABGAN]: fit synthesizer end")
        end_time = time.time()
        self.logger.info(f"Finished training in {end_time - start_time} seconds.")

        self.is_fit_ = True
        wandb.finish()

    def generate_samples(
        self,
        n: int,
        encoder: DataEncoder = None,
        *,
        n_jobs: Union[float, int] = None,
        resample_invalid: bool = True,
        times_resample: int = 10,  # 리샘플링 무한정 하지 않으려고
        minmax_clip: bool = True,
    ):
        assert self.is_fit_, "must fit the model first!!"

        if isinstance(encoder, DataEncoder):
            assert encoder.is_fit_, "you must use fitted encoder!!"
        else:
            encoder = self.encoder
        len_encoded = encoder.output_dim

        # smaple encode data
        sample = self.synthesizer.sample(n, encoder)  # (n, M, #encode)
        # inverse_transform전에 월별로 정렬 후 3lank 텐서를 2lank 로 변환
        sample = sample.transpose(1, 0, 2).reshape(-1, len_encoded)  # (n*M, #encode)
        # inverse transform by DataEncoder
        result, invalid_ids = encoder.inverse_transform(
            sample, n_jobs=n_jobs, minmax_clip=minmax_clip
        )  # (n*M, n_col)
        self.logger.info(
            f"[CTABGAN]: sythesized data has {len(invalid_ids)}/{len(result)} invalid rows."
        )

        if resample_invalid:
            num_for_resample = len(invalid_ids)
            all_ids = np.arange(0, len(result))
            valid_ids = list(set(all_ids) - set(invalid_ids))
            result = result[valid_ids]  # valid_result

            # 원하는 n 개 데이터가 다 만들어지지 않은 경우 (invalid id 존재)
            resample_cnt = 1
            while len(result) < n and resample_cnt <= times_resample:
                self.logger.info(
                    f"[CTABGAN]: resample count ({resample_cnt}/{times_resample})"
                )

                self.logger.info("[CTABGAN]: generate raw encode vectors start")
                # resample invalid data
                data_resample = self.synthesizer.sample(
                    num_for_resample, encoder
                )  # (n, M, #encode)
                # inverse_transform전에 월별로 정렬 후 3lank 텐서를 2lank 로 변환
                data_resample = data_resample.transpose(1, 0, 2).reshape(
                    -1, len_encoded
                )  # (n*M, #encode)
                new_result, invalid_ids = encoder.inverse_transform(
                    data_resample,
                    n_jobs=n_jobs,
                )  # (n*M, n_col)
                self.logger.info(
                    f"[CTABGAN]: sythesized data has {len(invalid_ids)}/{len(new_result)} invalid rows."
                )
                # lsw: invalid_ids 평가가 월 종합으로 이뤄져야함 .... 쉽지않네
                # num_for_resample 도 월 종합으로 계산 필요 ...
                num_for_resample = len(invalid_ids)
                all_ids = np.arange(0, len(new_result))
                valid_ids = list(set(all_ids) - set(invalid_ids))
                new_result = new_result[valid_ids]  # valid_result
                # merge previous result
                result = np.concatenate([result, new_result], axis=0)
                resample_cnt += 1

        # inverse prep by DataPrep
        sample_df = self.data_prep.inverse_prep(result)  # (n*M, #encode)

        self.logger.info("[CTABGAN]: data sampling end")
        return sample_df

    def load_generator(
        self,
        *,
        gpath: str,
        gside: int,
        params_ctabgan: dict,
        encoded_data: np.ndarray = None,
        n_jobs: Union[float, int] = None,
    ):
        # generator 로드
        self.synthesizer = CTABGANSynthesizer(**params_ctabgan)
        self.synthesizer.load_generator(gpath)
        self.synthesizer.is_fit_ = True
        self.is_fit_ = True

        # 이미지 트랜스포머 빌드
        self.synthesizer.gside = gside
        self.synthesizer.Gtransformer = ImageTransformer(
            self.synthesizer.gside, encoded_data.shape[1]
        )

        # 컨디션 벡터 생성기 빌드
        encoder = self.encoder
        assert encoder.is_fit_, "encoder should already be fitted!"
        if encoded_data is None:
            encoded_data = encoder.transform(self.data_prep.df.values, n_jobs=n_jobs)
        self.synthesizer.cond_generator = Cond(encoded_data, encoder.output_info)
