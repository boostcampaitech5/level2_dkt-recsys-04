# Boosting Baseline
1. CatBoost
2. LGBM

## How To Run?
```bash
    cd code/boosting & poetry install
    python train.py --model=CatBoost # --seed=42 --data_dir=/opt/ml/input/data --output_dir=./outputs/ --num_iterations=1000 --lr=0.1 --depth=6 --model_dir=./models/
    python inference.py --model=CatBoost --model_name={prefix}_catboost.cbm # --seed=42 --data_dir=/opt/ml/input/data --output_dir=./outputs/ --model_dir=./models/
```

```bash
    python train_cv.py --model=CatBoost 
    
    python inference_cv.py --model=CatBoost --model_name={Sample}
```

```bash
    python train_oof.py --model=CatBoost --user_id_dir=/opt/ml/level2_dkt-recsys-04/code/oof-stacking/userid/0.7

    python inference_cv.py --model=CatBoost --model_name=20230525_115217
```

```bash
    cd code/boosting & poetry install
    python train_cv.py --model=CatBoost # --seed=42 --data_dir=/opt/ml/input/data --output_dir=./outputs/ --num_iterations=1000 --lr=0.1 --depth=6 --model_dir=./models/
    python inference_cv.py --model=CatBoost --model_name={save_time} # --seed=42 --data_dir=/opt/ml/input/data --output_dir=./outputs/ --model_dir=./models/
```

### Directory Structure
```bash
    .
    |-- README.md
    |-- boosting_util
    |   |-- __init__.py
    |   |-- args.py                                        # Argument
    |   |-- datasets.py                                    # Datasets for Model
    |   |-- models.py                                      # Boosting Model Class
    |   `-- utils.py                                       # Common Utils
    |-- configs                                            # Config Save Directory
    |   |-- CatBoost    
    |       `-- {prefix}_{model}.json
    |   |-- LGBM
    |       `-- {prefix}_{model}.json
    |-- models                                             # Model Save Directory
    |   |-- CatBoost    
    |       `-- {prefix}_{model}.cbm
    |   |-- LGBM
    |       `-- {prefix}_{model}.txt
    |-- outputs                                            # Predict Ouput Directory
    |   |-- CatBoost    
    |       |-- {prefix}_{model}_inference.csv
    |       `-- {prefix}_{model}.csv
    |   |-- LGBM
    |       |-- {prefix}_{model}_inference.csv
    |       `-- {prefix}_{model}.csv
    |-- poetry.lock                                        # poetry
    |-- pyproject.toml                                     # poetry
    |-- train_cv.py                                        # train_cv model
    |-- train.py                                           # train model
    |-- inference_cv.py                                    # inference_cv model
    `-- inference.py                                       # inference model
```

### Run OutPut
#### Set args
```bash
    # 원하는 FE 기법 선택 or 주석처리
    FEATS = [
        "calculate_cumulative_stats_by_time",
        "calculate_overall_accuracy_by_testID",
        "calculate_overall_accuracy_by_KnowledgeTag",
        # # 시간 칼럼을 사용하는 FE
        "calculate_solve_time_column",  # Time 관련 Feature Engineering할 때 필수!
        "check_answer_at_time",
        "calculate_total_time_per_user",
        "calculate_past_correct_answers_per_user",
        "calculate_future_correct_answers_per_user",
        "calculate_past_correct_attempts_per_user",
        "calculate_past_solved_problems_per_user",
        "calculate_past_average_accuracy_per_user",
        "calculate_past_average_accuracy_current_problem_per_user",
        "calculate_rolling_mean_time_last_3_problems_per_user",
        # "calculate_mean_and_stddev_per_user", # 오류가 많아서 스킵
        "calculate_median_time_per_user",
        "calculate_problem_solving_time_per_user",
        "calculate_accuracy_by_time_of_day",
        "calculate_user_activity_time_preference",
        "calculate_normalized_time_per_user",
        "calculate_relative_time_spent_per_user",
        "calculate_time_cut_column",
        # "calculate_items_svd_latent",
        # "calculate_times_nmf_latent",
        # "calculate_users_pca_latent",
        # "calculate_items_pca_latent",
        # "calculate_times_pca_latent",
        # "calculate_times_lda_latent",
        # "caculate_item_latent_dirichlet_allocation",  # 50초 걸림
        # "caculate_user_latent_dirichlet_allocation",  # 50초 걸림
    ]
```

#### Train Model
1. CatBoost
```bash
    --------------- CatBoost Set Random Seed ---------------
    --------------- CatBoost Set WandB ---------------
    --------------- CatBoost Load Data ---------------
    FE Success : 20 / 20
    After Train/Valid DataSet Feature Engineering Columns : ['userID', 'assessmentItemID', 'testId', 'Timestamp', 'KnowledgeTag', 'user_correct_answer', 'user_total_answer', 'user_acc', 'test_mean', 'test_sum', 'tag_mean', 'tag_sum', 'time', 'correct_shift_-2', 'correct_shift_-1', 'correct_shift_1', 'correct_shift_2', 'total_used_time', 'future_correct', 'past_count', 'average_correct', 'past_content_count', 'average_content_correct', 'mean_time', 'time_median', 'hour', 'correct_per_hour', 'is_night', 'normalized_time', 'relative_time', 'time_cut_enc', 'answerCode']
    FE Success : 20 / 20
    After Test DataSet Feature Engineering Columns : ['userID', 'assessmentItemID', 'testId', 'Timestamp', 'KnowledgeTag', 'user_correct_answer', 'user_total_answer', 'user_acc', 'test_mean', 'test_sum', 'tag_mean', 'tag_sum', 'time', 'correct_shift_-2', 'correct_shift_-1', 'correct_shift_1', 'correct_shift_2', 'total_used_time', 'future_correct', 'past_count', 'average_correct', 'past_content_count', 'average_content_correct', 'mean_time', 'time_median', 'hour', 'correct_per_hour', 'is_night', 'normalized_time', 'relative_time', 'time_cut_enc', 'answerCode']
    --------------- CatBoost Data Split   ---------------
    --------------- CatBoost Data Loader   ---------------
    --------------- CatBoost Train   ---------------
    Train Feature Engineering Columns : ['userID', 'assessmentItemID', 'testId', 'Timestamp', 'KnowledgeTag', 'user_correct_answer', 'user_total_answer', 'user_acc', 'test_mean', 'test_sum', 'tag_mean', 'tag_sum', 'time', 'correct_shift_-2', 'correct_shift_-1', 'correct_shift_1', 'correct_shift_2', 'total_used_time', 'future_correct', 'past_count', 'average_correct', 'past_content_count', 'average_content_correct', 'mean_time', 'time_median', 'hour', 'correct_per_hour', 'is_night', 'normalized_time', 'relative_time', 'time_cut_enc']
    Categoy : ['assessmentItemID', 'testId', 'Timestamp', 'time_cut_enc']
    Default metric period is 5 because AUC is/are not implemented for GPU
    0:      test: 0.7645296 best: 0.7645296 (0)     total: 72ms     remaining: 1m 11s
    100:    test: 0.8270878 best: 0.8270878 (100)   total: 6.66s    remaining: 59.3s
    200:    test: 0.8295902 best: 0.8297704 (192)   total: 13.4s    remaining: 53.1s
    300:    test: 0.8307310 best: 0.8307617 (293)   total: 20s      remaining: 46.4s
    400:    test: 0.8314013 best: 0.8314211 (393)   total: 26.5s    remaining: 39.6s
    500:    test: 0.8317310 best: 0.8318497 (496)   total: 33s      remaining: 32.9s
    600:    test: 0.8321882 best: 0.8322453 (593)   total: 39.6s    remaining: 26.3s
    700:    test: 0.8326388 best: 0.8327729 (692)   total: 46.2s    remaining: 19.7s
    bestTest = 0.832772851
    bestIteration = 692
    Shrink model to first 693 iterations.
    --------------- CatBoost Valid   ---------------
    VALID AUC : 0.8327728543545919 ACC : 0.7542561065877128

    BEST VALIDATION : 0.8327728509902954

    Feature Importance : 
    [('assessmentItemID', 27.45978257191047),
    ('time', 19.194359338369487),
    ('correct_shift_-1', 9.145462278980352),
    ('correct_shift_1', 9.057939122059675),
    ('correct_shift_-2', 5.321674183619508),
    ('mean_time', 4.710470563906527),
    ('correct_shift_2', 4.484470571606457),
    ('user_acc', 3.2808586669567124),
    ('testId', 3.2335494798926496),
    ('average_correct', 3.2216515130403405),
    ('test_mean', 2.0749459492288778),
    ('future_correct', 1.3116463511319014),
    ('user_correct_answer', 0.981434574237881),
    ('time_median', 0.9791652502502822),
    ('tag_mean', 0.8087052858096802),
    ('relative_time', 0.7697576524690524),
    ('time_cut_enc', 0.5811882400846062),
    ('total_used_time', 0.472617075630057),
    ('normalized_time', 0.4569988984893562),
    ('userID', 0.4306606503782087),
    ('KnowledgeTag', 0.42650426272794106),
    ('hour', 0.415908107303737),
    ('tag_sum', 0.3657497137338018),
    ('user_total_answer', 0.2693594926246566),
    ('correct_per_hour', 0.17329981013619686),
    ('test_sum', 0.14536515040021933),
    ('past_count', 0.14272519194116015),
    ('is_night', 0.05366109335111704),
    ('Timestamp', 0.030088959728908772),
    ('past_content_count', 0.0),
    ('average_content_correct', 0.0)]
    --------------- CatBoost Predict   ---------------

    --------------- Save Output Predict   ---------------
    writing prediction : ./outputs/CatBoost/{prefix}_catboost.csv

    --------------- Save Model   ---------------
    saving model : ./models/CatBoost/{prefix}_catboost.cbm

    --------------- Save Config   ---------------
    saving config : ./configs/CatBoost/{prefix}_catboost.json
```

#### Inference Model
```bash
    --------------- CatBoost Set Random Seed ---------------
    --------------- CatBoost Load Data ---------------
    FE Success : 20 / 20
    --------------- CatBoost Load Model ---------------
    Categoy : ['assessmentItemID', 'testId', 'Timestamp', 'time_cut_enc']
    --------------- CatBoost Predict   ---------------

    --------------- Save Output Predict   ---------------
    writing prediction : ./outputs/CatBoost/{prefix}_catboost_inference.csv
```

2. LGBM
#### Train Model
```bash
    --------------- LGBM Set Random Seed ---------------
    --------------- LGBM Set WandB ---------------
    --------------- LGBM Load Data ---------------
    FE Success : 20 / 20
    After Train/Valid DataSet Feature Engineering Columns : ['userID', 'assessmentItemID', 'testId', 'Timestamp', 'KnowledgeTag', 'user_correct_answer', 'user_total_answer', 'user_acc', 'test_mean', 'test_sum', 'tag_mean', 'tag_sum', 'time', 'correct_shift_-2', 'correct_shift_-1', 'correct_shift_1', 'correct_shift_2', 'total_used_time', 'future_correct', 'past_count', 'average_correct', 'past_content_count', 'average_content_correct', 'mean_time', 'time_median', 'hour', 'correct_per_hour', 'is_night', 'normalized_time', 'relative_time', 'time_cut_enc', 'answerCode']
    FE Success : 20 / 20
    After Test DataSet Feature Engineering Columns : ['userID', 'assessmentItemID', 'testId', 'Timestamp', 'KnowledgeTag', 'user_correct_answer', 'user_total_answer', 'user_acc', 'test_mean', 'test_sum', 'tag_mean', 'tag_sum', 'time', 'correct_shift_-2', 'correct_shift_-1', 'correct_shift_1', 'correct_shift_2', 'total_used_time', 'future_correct', 'past_count', 'average_correct', 'past_content_count', 'average_content_correct', 'mean_time', 'time_median', 'hour', 'correct_per_hour', 'is_night', 'normalized_time', 'relative_time', 'time_cut_enc', 'answerCode']
    --------------- LGBM Data Split   ---------------
    --------------- LGBM Data Loader   ---------------
    --------------- LGBM Train   ---------------
    Train Feature Engineering Columns : ['userID', 'assessmentItemID', 'testId', 'Timestamp', 'KnowledgeTag', 'user_correct_answer', 'user_total_answer', 'user_acc', 'test_mean', 'test_sum', 'tag_mean', 'tag_sum', 'time', 'correct_shift_-2', 'correct_shift_-1', 'correct_shift_1', 'correct_shift_2', 'total_used_time', 'future_correct', 'past_count', 'average_correct', 'past_content_count', 'average_content_correct', 'mean_time', 'time_median', 'hour', 'correct_per_hour', 'is_night', 'normalized_time', 'relative_time', 'time_cut_enc']
    Categoy : ['assessmentItemID', 'testId', 'Timestamp', 'time_cut_enc']
    [LightGBM] [Warning] Met categorical feature which contains sparse values. Consider renumbering to consecutive integers started from zero
    [LightGBM] [Info] Number of positive: 1158779, number of negative: 616759
    [LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.028051 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 41612
    [LightGBM] [Info] Number of data points in the train set: 1775538, number of used features: 28

    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.652635 -> initscore=0.630644
    [LightGBM] [Info] Start training from score 0.630644
    Training until validation scores don't improve for 50 rounds
    [50]    training's auc: 0.845952        training's binary_logloss: 0.458743     valid_1's auc: 0.793221 valid_1's binary_logloss: 0.554536
    [100]   training's auc: 0.861307        training's binary_logloss: 0.437302     valid_1's auc: 0.810556 valid_1's binary_logloss: 0.536039
    Did not meet early stopping. Best iteration is:
    [100]   training's auc: 0.861307        training's binary_logloss: 0.437302     valid_1's auc: 0.810556 valid_1's binary_logloss: 0.536039
    Evaluated only: auc
    --------------- LGBM Valid   ---------------
    VALID AUC : 0.8105559072286402 ACC : 0.7313101406365655

    BEST VALIDATION : 0.8105559072286402

    Feature Importance : 
    [('assessmentItemID', 2355),
    ('testId', 170),
    ('Timestamp', 56),
    ('time', 49),
    ('correct_shift_1', 46),
    ('correct_shift_-2', 45),
    ('user_acc', 43),
    ('correct_shift_-1', 42),
    ('correct_shift_2', 41),
    ('test_mean', 39),
    ('mean_time', 25),
    ('average_correct', 24),
    ('tag_mean', 19),
    ('user_correct_answer', 15),
    ('future_correct', 14),
    ('normalized_time', 5),
    ('test_sum', 4),
    ('tag_sum', 3),
    ('hour', 3),
    ('relative_time', 2),
    ('userID', 0),
    ('KnowledgeTag', 0),
    ('user_total_answer', 0),
    ('total_used_time', 0),
    ('past_count', 0),
    ('past_content_count', 0),
    ('average_content_correct', 0),
    ('time_median', 0),
    ('correct_per_hour', 0),
    ('is_night', 0),
    ('time_cut_enc', 0)]
    --------------- LGBM Predict   ---------------

    --------------- Save Output Predict   ---------------
    writing prediction : ./outputs/LGBM/{prefix}_lgbm.csv

    --------------- Save Model   ---------------
    saving model : ./models/LGBM/{prefix}_lgbm.txt

    --------------- Save Config   ---------------
    saving config : ./configs/LGBM/{prefix}_lgbm.json
```

#### Inference Model
```bash
    --------------- LGBM Set Random Seed ---------------
    --------------- LGBM Load Data ---------------
    FE Success : 20 / 20
    --------------- LGBM Load Model ---------------
    Categoy : ['assessmentItemID', 'testId', 'Timestamp', 'time_cut_enc']
    --------------- LGBM Predict   ---------------

    --------------- Save Output Predict   ---------------
    writing prediction : ./outputs/LGBM/{prefix}_lgbm_inference.csv
```