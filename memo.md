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
- 99999999 -> 100000000 or 99999998 로 변경되는거 체크
  - mixed 경우 gm1 에서 모드로 선정된 것은 추후 meta참고해 가장 가까운 모드값으로 변경 해야함
- min-max 벗어나는 부분이 모드에 있을경우 inv prep 안돰 이건 해결해야 함
  - 이거 위의 모드 문제랑도 겹치는듯
- encoded vec 도 저장/로드 가능케하자 [x]
- transformer save시 VGM 모델만 저장토록하자
  - build, fit 분리 필요
- generator 저장 시에 state_dict 만 저장토록... 모델 전체는 용량이 너무 큼
  - build 메서드 구현 필요할 듯
- G, D 케파늘리기
- lr 스케줄러  lambda 로 바꿔보기

---


mixed & log & 음수 존재
log 와 mixed 는 사실 같이 안쓰엿다

disciriminator 훈련 중 만든 fake 들은 generator 업데이트에 안쓰네?? 왜 재활용 안함? 