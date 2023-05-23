# Sequential Model Baseline

## How To Run?

```bash
poetry install
poetry shell

python train.py
python inference.py --load_path ".../saved/{}/best_model.pt"

# Hyperparameter Tuning
python tune.py
```

### Directory Structure

```bash
.
|-- README.md
|-- inference.py
|-- poetry.lock
|-- pyproject.toml
|-- run.log
|-- saved
|   |-- 0523_032831
|   |-- 0523_033234
|   `-- asset
|-- sequence_utils
|   |-- __pycache__
|   |-- args.py
|   |-- config.py
|   |-- datasets.py
|   |-- models.py
|   |-- trainer.py
|   `-- utils.py
|-- train.py
`-- wandb
```

### Configuration

```python
import os
import torch
import numpy as np

class CONFIG:    
    def __init__(self):
        # 설정
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_dir = "/opt/ml/input/data/"
        self.save_dir = os.path.join(os.getcwd(), "saved")
        # self.model_dir = os.path.join(self.save_dir, "models")

        # 데이터
        self.max_seq_len = 500

        # 데이터 증강 (Data Augmentation)
        self.window = False
        self.stride = self.max_seq_len
        self.shuffle = False
        self.shuffle_n  = 2

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.n_questions = len(np.load(os.path.join(f'{self.save_dir}/asset', "assessmentItemID_classes.npy")))
        self.n_test = len(np.load(os.path.join(f'{self.save_dir}/asset', "testId_classes.npy")))
        self.n_tag = len(np.load(os.path.join(f'{self.save_dir}/asset', "KnowledgeTag_classes.npy")))
        self.n_elapsed_question = len(np.load(os.path.join(f'{self.save_dir}/asset', "elapsed_question_classes.npy")))
        # self.n_elapsed_tests = len(np.load(os.path.join(f'{self.save_dir}/asset', "elapsed_test_classes.npy")))

        # 모델
        self.hidden_dim = 256
        self.n_layers = 1
        self.dropout = 0.1
        self.n_heads = 1

        # T Fixup
        self.Tfixup = False
        self.layer_norm = True

        # 훈련
        self.n_epochs = 20
        self.batch_size = 128
        self.lr = 0.0001
        self.clip_grad = 10

        ### 중요 ###
        self.model = "bert"
        self.optimizer = "adam"
        self.scheduler = "plateau"
```

# Feature Engineering

- 전처리
    - 유저별 시퀀스 고려를 위한 UserID, TimeStamp기반 데이터 정렬
    - 임베딩을 위한 Label Encoding
    - Timestamp datetime 변환
    - Data augmentation
- 사용 피처
    - `userID` : 유저 ID
    - `assessmentItemID` : 문제 번호
    - `KnowledgeTag` : 문제 고유 태그
    - `mask` : answerCode 를 max_seq_len 에 맞춰 패딩 변환
    - `elapsed_question` : 문제를 푸는데 걸린 시간