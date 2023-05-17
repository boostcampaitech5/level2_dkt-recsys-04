# Lightgbm Baseline

## How To Run?
```bash
    cd code/lgbm & poetry install
    python train.py # --seed=42 --data_dir=/opt/ml/input/data --output_dir=./outputs/ --num_iterations=1000 --lr=0.1 --num_boost_round=500 --model_dir=./models/ --num_leaves=31
    python inference.py --model_name={prefix}_lgbm.txt # --seed=42 --data_dir=/opt/ml/input/data --output_dir=./outputs/ --model_dir=./models/
```

### Directory Structure
```bash
    .
    |-- README.md
    |-- lgbm_util
    |   |-- __init__.py
    |   |-- args.py                                        # Argument
    |   |-- datasets.py                                    # Datasets for LGBM
    |   |-- models.py                                      # LGBM Model Class
    |   `-- utils.py                                       # Common Utils
    |-- configs                                            # Config Save Directory
    |   `-- {prefix}_lgbm.json
    |-- models                                             # Model Save Directory
    |   `-- {prefix}_lgbm.cbm
    |-- outputs                                            # Predict Ouput Directory
    |   |-- {prefix}_lgbm.csv
    |   `-- {prefix}_lgbm_inference.csv
    |-- poetry.lock                                        # poetry
    |-- pyproject.toml                                     # poetry
    |-- train.py                                           # train model
    `-- inference.py                                       # inference model
```

### Run OutPut
#### Train Model
```bash
    --------------- LGBM Set Random Seed ---------------
    --------------- LGBM Set WandB ---------------
    --------------- LGBM Load Data ---------------
    --------------- LGBM Data Split   ---------------
    --------------- LGBM Data Loader   ---------------
    --------------- LGBM Train   ---------------
    [LightGBM] [Info] Number of positive: 1187785, number of negative: 624671
    [LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.013923 seconds.
    [LightGBM] [Info] Total Bins 3052
    [LightGBM] [Info] Number of data points in the train set: 1812456, number of used features: 12
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.655346 -> initscore=0.642620
    [LightGBM] [Info] Start training from score 0.642620
    Training until validation scores don't improve for 100 rounds
    [50]    training's auc: 0.738507        training's binary_logloss: 0.559965     valid_1's auc: 0.693519 valid_1's binary_logloss: 0.670838
    [100]   training's auc: 0.742643        training's binary_logloss: 0.556552     valid_1's auc: 0.694313 valid_1's binary_logloss: 0.667549
    [150]   training's auc: 0.74557 training's binary_logloss: 0.554275     valid_1's auc: 0.693973 valid_1's binary_logloss: 0.667136
    Early stopping, best iteration is:
    [72]    training's auc: 0.740665        training's binary_logloss: 0.558153     valid_1's auc: 0.694409 valid_1's binary_logloss: 0.668714
    Evaluated only: auc
    --------------- LGBM Valid   ---------------
    VALID AUC : 0.6944091068608527 ACC : 0.622501850481125

    BEST VALIDATION : 0.6944091068608528

    Feature Importance : 
    [('user_acc', 419),
    ('userID', 326),
    ('test_mean', 295),
    ('assessmentItemID', 252),
    ('tag_mean', 242),
    ('testId', 142),
    ('user_correct_answer', 122),
    ('user_total_answer', 103),
    ('Timestamp', 100),
    ('KnowledgeTag', 64),
    ('tag_sum', 56),
    ('test_sum', 39)]
    --------------- LGBM Predict   ---------------

    --------------- Save Output Predict   ---------------
    writing prediction : ./outputs/{prefix}_lgbm.csv

    --------------- Save Model   ---------------
    saving model : ./models/{prefix}_lgbm.txt

    --------------- Save Config   ---------------
    saving config : ./configs/{prefix}_lgbm.json
```

#### Inference Model
```bash
    --------------- LGBM Set Random Seed ---------------
    --------------- LGBM Load Data ---------------
    --------------- LGBM Load Model ---------------
    --------------- LGBM Predict   ---------------

    --------------- Save Output Predict   ---------------
    writing prediction : ./outputs/{prefix}_lgbm_inference.csv
```