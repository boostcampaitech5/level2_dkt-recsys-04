# level2_dkt-recsys-04

level2_dkt-recsys-04 created by GitHub Classroom

## 1. 프로젝트 개요

### 1-1. 프로젝트 주제

![프로젝트 주제](./docs/%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8_%EC%A3%BC%EC%A0%9C.png)

사용자의 "지식 상태"를 추적하는 딥러닝 방법론인 DKT(Deep Knowledge Tracing) 모델을 구축하고, 사용자가 최종 문제를 맞출지 틀릴지 예측하는 것

### 1-2. 활용 장비 및 재료

```bash
ai stage server : V100 GPU x 5

python==3.10.1
torch==1.7.1
CUDA==11.0
```

### 1-3. 프로젝트 구조 및 사용 데이터셋의 구조도(연관도)

```bash
:package:level2_dkt-recsys-04
 ┣ :open_file_folder:code
 ┃ ┣ :open_file_folder:FM
 ┃ ┃ ┣ :open_file_folder:src
 ┃ ┃ ┃ ┣ :open_file_folder:data
 ┃ ┃ ┃ ┃ ┣ :scroll:__init__.py
 ┃ ┃ ┃ ┃ ┣ :scroll:context_data.py
 ┃ ┃ ┃ ┃ ┗ :scroll:dataset.py
 ┃ ┃ ┃ ┣ :open_file_folder:train
 ┃ ┃ ┃ ┃ ┗ :scroll:trainer.py
 ┃ ┃ ┃ ┗ :scroll:utils.py
 ┃ ┃ ┗ :scroll:main.py
 ┃ ┣ :open_file_folder:MF
 ┃ ┃ ┣ :open_file_folder:MF_util
 ┃ ┃ ┣ :scroll:README.md
 ┃ ┃ ┗ :scroll:main.py
 ┃ ┣ :open_file_folder:boosting
 ┃ ┃ ┣ :open_file_folder:boosting_util
 ┃ ┃ ┣ :scroll:inference.py
 ┃ ┃ ┣ :scroll:inference_cv.py
 ┃ ┃ ┗ :scroll:train.py
 ┃ ┣ :open_file_folder:ensembles
 ┃ ┣ :open_file_folder:light-gcn
 ┃ ┃ ┣ :open_file_folder:lightgcn_utils
 ┃ ┃ ┣ :scroll:inference.py
 ┃ ┃ ┗ :scroll:train.py
 ┃ ┣ :open_file_folder:oof-stacking
 ┃ ┃ ┣ :open_file_folder:oof_stacking_util
 ┃ ┃ ┣ :scroll:args_list.json
 ┃ ┃ ┗ :scroll:main.py
 ┃ ┣ :open_file_folder:sequence
 ┃ ┃ ┣ :open_file_folder:sequence_utils
 ┃ ┃ ┣ :scroll:inference.py
 ┃ ┃ ┣ :scroll:train.py
 ┃ ┃ ┗ :scroll:tune.py
 ┃ ┣ :open_file_folder:utils
 ┃ ┃ ┗ :scroll:feature_selection.py
 ┃ ┗ :scroll:__init__.py
 ┗ :scroll:.gitignore
```

## 2. 프로젝트 팀 구성 및 역할

| 이름          | 역할                                                                               |
| ------------- | ---------------------------------------------------------------------------------- |
| 김수민\_T5040 | Catboost, LightGCN 모델 구현, 하이퍼 파라미터 튜닝, OOF-Stacking                   |
| 박예림\_T5088 | Cross-Validation 구현, 하이퍼 파라미터 튜닝, XGBoost 모델 구현                     |
| 임도현\_T5170 | Sequential 모델, 변수 선택, 하이퍼 파라미터 튜닝, Stacking 메타 모델 구현          |
| 임우열\_T5173 | Boosting 모델 구현, Feature Engineering, OOF-Stacking, Visualization Output        |
| 임지수\_T5176 | Latent Vector 모델 구현, Feature Engineering, Error Analysis, Visualization Output |

## 3. 프로젝트 수행 절차 및 방법

![수행절차_1](./docs/%EC%88%98%ED%96%89%EC%A0%88%EC%B0%A8_1.png)

![수행절차_2](./docs/%EC%88%98%ED%96%89%EC%A0%88%EC%B0%A8_2.png)

## 4. 단일 모델 결과

| Model                      | valid_auc |
| -------------------------- | --------- |
| CatBoost                   | 0.8588    |
| XGBoost                    | 0.8012    |
| LGBM                       | 0.6944    |
| LightGCN                   | 0.7661    |
| LSTM                       | 0.7998    |
| LSTMATTN                   | 0.8278    |
| Last Query Transformer RNN | 0.8256    |
| BERT                       | 0.8327    |
| SAINT                      | 0.8351    |
| SAINT+                     | 0.6107    |

## 5. 최종 제출

| Models                   | public_auc | public_acc |
| ------------------------ | ---------- | ---------- |
| CatBoost, LGBM, LightGCN | 0.8225     | 0.7446     |
| CatBoost, LGBM           | 0.8131     | 0.7527     |

## 6. 최종 순위

### public score

![public score](./docs/public_score.png)

### private score

![private score](./docs/private.png)
