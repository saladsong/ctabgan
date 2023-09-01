# 금융 합성 데이터 데이터 품질 지표 계산
- 작성일: 2023/08/31

## workspace 구조
```sh
2.Viewer
├── drift_utils.py  // 평가 코드 수행 위한 유틸코드들
├── utils.py  // 평가 코드 수행 위한 유틸코드들
├── readme.md
├── requirements.txt  // 평가 코드 수행 위한 디펜던시
├── synthe-eval_230831-153735.csv  // 평가 코드 수행 후 결과 (csv 포맷)
├── synthe-eval_230831-153735.json  // 평가 코드 수행 후 결과 (json 포맷)
└── synthetic_data_evaluation.ipynb  // // 평가 코드 수행용 노트북
```

## prereuisites
- python3
  - requirements.txt 내의 패키지 설치 필요
  - `pip install -r requirements.txt`

## 품질 지표 계산 메인 코드
- synthetic_data_evaluation.ipynb

### 데이터 포맷
- *.csv
- 원본과 합성 데이터셋 모두 주제영역별, 기준년월 별로 저장

### 품질지표
- JSD
    - 원천데이터와 합성데이터를 구성하는 데이터 필드별 분포의 유사성을 판단
- pMSE
    - 원천데이터와 합성데이터의가 기계학습 모델에 의해 구분 가능한 정도를 평가
- Corr.diff
    - 원자료의 상관도 분석 결과와 합성데이터의 상관도 분석 결과를 비교하여 합성데이터와 원천데이터의 상관관계와 유사한 정도를 평가


## snipet
```sh
# 가상환경 생성
# virtualenv --python python3.10 {가상환경이름}
virtualenv --python python3.10 tmpenv

# 가상환경 커널 주피터 노트북에 등록
# python -m ipykernel install --user --name {가상환경이름}
python -m ipykernel install --user --name tmpenv
```