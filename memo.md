- 주석이 너무 없다
- 프로세스별 로그도 너무 없다
- 코드 병렬체리 최적화 필요


- GAN 모델 로스 시각화 위한 저장
- 데이터 패스 대신에 데이터프레임 입력 받도록 수정
- VGM jobs 100% 안쓰도록 세팅
- inital value 에서 [], {} 제거 [x]
- gan params 설정 (epochs, ...) 밖으로 빼기 [x]
- 판다스 로드 데이터 자동 컬럼 타입 재고해보기
- csv 대신 parquet 사용 고려, FILE IO 시간 단축 고려
- VGM 모델 피클링 해서 외부 저장 고려
- transformer.transform 싱글 스레드로 되는데 컬럼 유닛모델별로 병렬처리

- 현재 1000 -> 6400 정도 뻥튀기
  - OOM 남
  - mini batch 사이즈 때문인가??
- classifier 너무 히든 레이어 많음