class diagram 뽑는법
```
// pyreverse -o "확장자명" "폴더 혹은 경로명"
pyreverse -o png model 

// 단 모듈 내에 __init__.py  필요
```

- 주석이 너무 없다
- 프로세스별 로그도 너무 없다
- 코드 병렬체리 최적화 필요

코드 변경 내용
진행중 [...], 완료 [x]

- GAN 모델 로스 시각화 위한 저장 [x]
  - WANDB 써보자
- 데이터 패스 대신에 데이터프레임 입력 받도록 수정 [x]
- VGM jobs 100% 안쓰도록 세팅 [...]
  - 이거 왜 n_jobs 옵션 없나?
- inital value 에서 [], {} 제거 [x]
- gan params 설정 (epochs, ...) 밖으로 빼기 [x]
- 판다스 로드 데이터 자동 컬럼 타입 재고해보기
- csv 대신 parquet 사용 고려, FILE IO 시간 단축 고려 [x]
- VGM 모델 피클링 해서 외부 저장 고려 [...]
  - data transformer 최상위로 올리기 [x]
  - data transformer 내에 train data 저장하는 부분 제거 [x]
  - vgm 모델 저장/로드 구현 [x]
- transformer.transform, inverse_transform 싱글 스레드로 되는데 컬럼 유닛모델별로 병렬처리 [x]
- generator, discriminator side 자유롭도록 변경 [x]
- gan 모델 포함 전체 저장/로드 구현 [...]
- data 2d -> 3d 변경, gan channel 여러개 변경 [...]
  - 시계열 차원 → CNN 채널 차원으로 적용
  - CNN 피처 중 파생 컬럼 마스킹
- wandb 프로젝트명, cpu core 수 등 파라메터 밖으로 꺼내기 [x]
- 추후 gpu 메모리 모자랄 경우 대비해서 adam 대신 rmsprop 도 고려 해야함... WAS GAN 이랑 상충하는 부분은 없나?
- 배치 사이즈 늘리기 위해 discriminator, generator 각각 다른 gpu에 올릴 수 있도록 [...]
- inv_transform 에서 invalid resample 로직 밖으로 빼내기 [x]
- eval 코드도 병렬 최적화 필요
- encoded vec 도 저장/로드 가능케하자 [x]
- generator 저장 시에 state_dict 만 저장토록... 모델 전체는 용량이 너무 큼
  - build 메서드 구현 필요할 듯
- G, D 케파늘리기 [x]
  - 일단은 G만 늘림, D는 지금도 충분
- lr 스케줄러  lambda 로 바꿔보기
- gradient accumulation 적용해보기 [x]
- 모드 체크 , 가장 가까운 곳으로 넣기 [x]
    - 99999999->100000000 변환 되는것 float32 정밀도 문제임
    - 마지막 샘플링만 float64로 되도록 변경??
- 모드 역변환 레이블 디코딩 안되는 문제 [x]
    - non-cate 애들이 con으로 디코딩 된후 min-max 범위 벗어나서 발생하는 문제
- 제너레이터 state-dict 저장/로드로 바꾸기
- transformer VGM 모델만 저장/로드로 바꾸기 [x]
  - 얘는 불필요할듯.. 어차피 피클 로드시 내부 로직은 업데이트된 코드로 교체됨
- formula 체크 [...]
  - 우선순위 알고리즘
- discriminator 마지막 relu 레이어 없애보기 -> avg-pooling 으로 대체


---
8월말까지
- 월별로 4만건
- 


mixed & log & 음수 존재
log 와 mixed 는 사실 같이 안쓰엿다

disciriminator 훈련 중 만든 fake 들은 generator 업데이트에 안쓰네?? 왜 재활용 안함? 