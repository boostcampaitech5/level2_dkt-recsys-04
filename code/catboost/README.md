# CatBoost Baseline

## How To Run?
```bash
    cd code/catboost & poetry install
    python main.py # --seed=42 --data_dir=/opt/ml/input/data --output_dir=./outputs/ --num_iterations=1000 --lr=0.1
```

### Directory Structure
```bash
    .
    |-- README.md
    |-- catboost_util
    |   |-- __init__.py
    |   |-- args.py         # Argument
    |   |-- datasets.py     # Datasets for CatBoost
    |   |-- models.py       # CatBoost Model Class
    |   `-- utils.py        # Common Utils
    |-- main.py             # train/inference code
    |-- poetry.lock         # poetry
    |-- pyproject.toml      # poetry
```

### Run OutPut
```bash
    --------------- CatBoost Set Random Seed ---------------
    --------------- CatBoost Set WandB ---------------
    --------------- CatBoost Load Data ---------------
    --------------- CatBoost Data Split   ---------------
    --------------- CatBoost Data Loader   ---------------
    --------------- CatBoost Train   ---------------
    Default metric period is 5 because AUC is/are not implemented for GPU
    0:      test: 0.7258499 best: 0.7258499 (0)     total: 46.9ms   remaining: 46.9s
    100:    test: 0.7436019 best: 0.7436305 (98)    total: 6.03s    remaining: 53.7s
    200:    test: 0.7440670 best: 0.7443389 (168)   total: 12.1s    remaining: 48s
    bestTest = 0.7443389297
    bestIteration = 168
    Shrink model to first 169 iterations.
    --------------- CatBoost Valid   ---------------
    VALID AUC : 0.7443389191086156 ACC : 0.6839378238341969

    BEST VALIDATION : {'Logloss': 0.5958350892423436, 'AUC': 0.7443411350250244}

    Feature Importance : 
    [('assessmentItemID', 46.81424646566238),
    ('user_acc', 32.47341283995978),
    ('userID', 4.446694435416015),
    ('testId', 4.220605327065889),
    ('test_mean', 3.509694472157899),
    ('user_correct_answer', 2.6702036778561857),
    ('user_total_answer', 2.2787890541669227),
    ('tag_mean', 1.8963748487486443),
    ('KnowledgeTag', 0.999328079763184),
    ('tag_sum', 0.3383336007269159),
    ('test_sum', 0.2598353902503896),
    ('Timestamp', 0.09248180822587801)]
    --------------- CatBoost Predict   ---------------

    --------------- Save Output Predict   ---------------
    writing prediction : ./outputs/{date_time}_{auc}_catboost.csv
```