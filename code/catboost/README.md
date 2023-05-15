# CatBoost Baseline

## How To Run?
```bash
    cd code/catboost & poetry install
    python train.py # --seed=42 --data_dir=/opt/ml/input/data --output_dir=./outputs/ --num_iterations=1000 --lr=0.1 --depth=6 --model_dir=./models/
    python inference.py --model_name={prefix}_catboost.cbm # --seed=42 --data_dir=/opt/ml/input/data --output_dir=./outputs/ --model_dir=./models/
```

### Directory Structure
```bash
    .
    |-- README.md
    |-- catboost_util
    |   |-- __init__.py
    |   |-- args.py                                        # Argument
    |   |-- datasets.py                                    # Datasets for CatBoost
    |   |-- models.py                                      # CatBoost Model Class
    |   `-- utils.py                                       # Common Utils
    |-- configs                                            # Config Save Directory
    |   `-- {prefix}_catboost.json
    |-- models                                             # Model Save Directory
    |   `-- {prefix}_catboost.cbm
    |-- outputs                                            # Predict Ouput Directory
    |   |-- {prefix}_catboost.csv
    |   `-- {prefix}_catboost_inference.csv
    |-- poetry.lock                                        # poetry
    |-- pyproject.toml                                     # poetry
    |-- train.py                                           # train model
    `-- inference.py                                       # inference model
```

### Run OutPut
#### Train Model
```bash
    --------------- CatBoost Set Random Seed ---------------
    --------------- CatBoost Set WandB ---------------
    --------------- CatBoost Load Data ---------------
    --------------- CatBoost Data Split   ---------------
    --------------- CatBoost Data Loader   ---------------
    --------------- CatBoost Train   ---------------
    Default metric period is 5 because AUC is/are not implemented for GPU
    0:      test: 0.7258499 best: 0.7258499 (0)     total: 48.3ms   remaining: 48.3s
    100:    test: 0.7436019 best: 0.7436305 (98)    total: 6.16s    remaining: 54.8s
    200:    test: 0.7441152 best: 0.7442468 (197)   total: 12.4s    remaining: 49.3s
    300:    test: 0.7443740 best: 0.7444003 (298)   total: 18.8s    remaining: 43.6s
    bestTest = 0.7444003224
    bestIteration = 298
    Shrink model to first 299 iterations.
    --------------- CatBoost Valid   ---------------
    VALID AUC : 0.7444003333918232 ACC : 0.6824574389341229

    BEST VALIDATION : {'Logloss': 0.5958337339094884, 'AUC': 0.7444003224372864}

    Feature Importance : 
    [('assessmentItemID', 45.38904044230427),
    ('user_acc', 31.889092871452355),
    ('userID', 5.132668252821983),
    ('testId', 4.636163875245036),
    ('test_mean', 3.6584272758696597),
    ('user_correct_answer', 2.9528448190841257),
    ('user_total_answer', 2.536130978974387),
    ('tag_mean', 1.9793931701734564),
    ('KnowledgeTag', 1.0906650373308275),
    ('tag_sum', 0.3572590814340498),
    ('test_sum', 0.27473139531210844),
    ('Timestamp', 0.10358279999767406)]
    --------------- CatBoost Predict   ---------------

    --------------- Save Output Predict   ---------------
    writing prediction : ./outputs/{prefix}_catboost.csv

    --------------- Save Model   ---------------
    saving model : ./models/{prefix}_catboost.cbm

    --------------- Save Config   ---------------
    saving config : ./configs/{prefix}_catboost.json
```

#### Inference Model
```bash
    --------------- CatBoost Set Random Seed ---------------
    --------------- CatBoost Load Data ---------------
    --------------- CatBoost Load Model ---------------
    --------------- CatBoost Predict   ---------------

    --------------- Save Output Predict   ---------------
    writing prediction : ./outputs/{prefix}_catboost_inference.csv
```