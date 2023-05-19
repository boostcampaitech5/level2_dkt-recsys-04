# LightGCN Baseline

## How To Run?
```bash
    ## Setup
    cd code/light-gcn & poetry install & poetry shell
    python train.py
    python inference.py
```

### Directory Structure
```bash
    .
    |-- README.md
    |-- lightgcn_utils
    |   |-- args.py                                        # Argument
    |   |-- datasets.py                                    # Datasets for LightGCN
    |   |-- models.py                                      # LightGCN Model Class
    |   `-- utils.py                                       # Common Utils
    |-- poetry.lock                                        # poetry
    |-- pyproject.toml                                     # poetry
    |-- train.py                                           # train model
    `-- inference.py                                       # inference model
```

### Run OutPut
#### Train Model
```bash
    --------------- LightGCN Load Data ---------------
    --------------- LightGCN Data Split ---------------
    --------------- LightGCN Set Random Seed ---------------
    --------------- LightGCN Set WandB ---------------
    wandb: (1) Create a W&B account
    wandb: (2) Use an existing W&B account
    wandb: (3) Don't visualize my results
    wandb: Enter your choice: 3
    wandb: You chose "Don't visualize my results"
    wandb: Tracking run with wandb version 0.15.2
    wandb: W&B syncing is set to `offline` in this directory.  
    wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
    --------------- LightGCN Data Loader   ---------------
    2023-05-18 18:46:40,068 - root - INFO - Preparing data ...
    2023-05-18 18:46:52,028 - root - INFO - Train Dataset Info
    2023-05-18 18:46:52,029 - root - INFO -  * Num. Users    : 7442
    2023-05-18 18:46:52,029 - root - INFO -  * Max. UserID   : 7441
    2023-05-18 18:46:52,029 - root - INFO -  * Num. Items    : 9454
    2023-05-18 18:46:52,030 - root - INFO -  * Num. Records  : 2475962
    2023-05-18 18:46:52,030 - root - INFO - Test Dataset Info
    2023-05-18 18:46:52,031 - root - INFO -  * Num. Users    : 744
    2023-05-18 18:46:52,031 - root - INFO -  * Max. UserID   : 7439
    2023-05-18 18:46:52,031 - root - INFO -  * Num. Items    : 444
    2023-05-18 18:46:52,031 - root - INFO -  * Num. Records  : 744
    --------------- LightGCN Model Build   ---------------
    2023-05-18 18:46:52,123 - root - INFO - Building Model ...
    2023-05-18 18:46:52,140 - root - INFO - No load model
    --------------- LightGCN Train   ---------------
    2023-05-18 18:46:52,143 - root - INFO - Start Training ...
    2023-05-18 18:46:52,223 - root - INFO - Training Started : n_epochs=10
    2023-05-18 18:46:52,224 - root - INFO - Epoch: 0
    2023-05-18 18:46:53,359 - root - INFO - TRAIN AUC : 0.4995 ACC : 0.4998 LOSS : 0.6931
    2023-05-18 18:46:53,363 - root - INFO - VALID AUC : 0.4813 ACC : 0.4790
    2023-05-18 18:46:53,364 - root - INFO - Best model updated AUC from 0.0000 to 0.4813
    2023-05-18 18:46:53,384 - root - INFO - Epoch: 1
    2023-05-18 18:46:54,533 - root - INFO - TRAIN AUC : 0.5020 ACC : 0.5015 LOSS : 0.6931
    2023-05-18 18:46:54,538 - root - INFO - VALID AUC : 0.4831 ACC : 0.4750
    2023-05-18 18:46:54,538 - root - INFO - Best model updated AUC from 0.4813 to 0.4831
    2023-05-18 18:46:54,555 - root - INFO - Epoch: 2
    2023-05-18 18:46:55,705 - root - INFO - TRAIN AUC : 0.5047 ACC : 0.5036 LOSS : 0.6931
    2023-05-18 18:46:55,709 - root - INFO - VALID AUC : 0.4855 ACC : 0.4780
    2023-05-18 18:46:55,710 - root - INFO - Best model updated AUC from 0.4831 to 0.4855
    2023-05-18 18:46:55,725 - root - INFO - Epoch: 3
    2023-05-18 18:46:56,863 - root - INFO - TRAIN AUC : 0.5076 ACC : 0.5056 LOSS : 0.6931
    2023-05-18 18:46:56,868 - root - INFO - VALID AUC : 0.4882 ACC : 0.4810
    2023-05-18 18:46:56,868 - root - INFO - Best model updated AUC from 0.4855 to 0.4882
    2023-05-18 18:46:56,885 - root - INFO - Epoch: 4
    2023-05-18 18:46:58,021 - root - INFO - TRAIN AUC : 0.5108 ACC : 0.5078 LOSS : 0.6931
    2023-05-18 18:46:58,025 - root - INFO - VALID AUC : 0.4915 ACC : 0.4780
    2023-05-18 18:46:58,026 - root - INFO - Best model updated AUC from 0.4882 to 0.4915
    2023-05-18 18:46:58,045 - root - INFO - Epoch: 5
    2023-05-18 18:46:59,166 - root - INFO - TRAIN AUC : 0.5141 ACC : 0.5101 LOSS : 0.6931
    2023-05-18 18:46:59,170 - root - INFO - VALID AUC : 0.4949 ACC : 0.4860
    2023-05-18 18:46:59,171 - root - INFO - Best model updated AUC from 0.4915 to 0.4949
    2023-05-18 18:46:59,189 - root - INFO - Epoch: 6
    2023-05-18 18:47:00,312 - root - INFO - TRAIN AUC : 0.5177 ACC : 0.5125 LOSS : 0.6931
    2023-05-18 18:47:00,317 - root - INFO - VALID AUC : 0.4988 ACC : 0.4950
    2023-05-18 18:47:00,318 - root - INFO - Best model updated AUC from 0.4949 to 0.4988
    2023-05-18 18:47:00,333 - root - INFO - Epoch: 7
    2023-05-18 18:47:01,459 - root - INFO - TRAIN AUC : 0.5216 ACC : 0.5153 LOSS : 0.6931
    2023-05-18 18:47:01,463 - root - INFO - VALID AUC : 0.5037 ACC : 0.5040
    2023-05-18 18:47:01,464 - root - INFO - Best model updated AUC from 0.4988 to 0.5037
    2023-05-18 18:47:01,482 - root - INFO - Epoch: 8
    2023-05-18 18:47:02,594 - root - INFO - TRAIN AUC : 0.5257 ACC : 0.5183 LOSS : 0.6931
    2023-05-18 18:47:02,598 - root - INFO - VALID AUC : 0.5086 ACC : 0.5020
    2023-05-18 18:47:02,599 - root - INFO - Best model updated AUC from 0.5037 to 0.5086
    2023-05-18 18:47:02,619 - root - INFO - Epoch: 9
    2023-05-18 18:47:03,739 - root - INFO - TRAIN AUC : 0.5302 ACC : 0.5216 LOSS : 0.6931
    2023-05-18 18:47:03,743 - root - INFO - VALID AUC : 0.5139 ACC : 0.5060
    2023-05-18 18:47:03,744 - root - INFO - Best model updated AUC from 0.5086 to 0.5139
    2023-05-18 18:47:03,779 - root - INFO - Best Weight Confirmed : 10'th epoch
    wandb: Waiting for W&B process to finish... (success).
    wandb: 
    wandb: Run history:
    wandb:  train_acc_epoch ▁▂▂▃▄▄▅▆▇█
    wandb:  train_auc_epoch ▁▂▂▃▄▄▅▆▇█
    wandb: train_loss_epoch █▆▄▃▂▁▁▁▁▁
    wandb:  valid_acc_epoch ▂▁▂▂▂▃▆█▇█
    wandb:  valid_auc_epoch ▁▁▂▂▃▄▅▆▇█
    wandb: 
    wandb: Run summary:
    wandb:  train_acc_epoch 0.5216
    wandb:  train_auc_epoch 0.53019
    wandb: train_loss_epoch 0.69315
    wandb:  valid_acc_epoch 0.506
    wandb:  valid_auc_epoch 0.51388
    wandb: 
    wandb: You can sync this run to the cloud by running:
    wandb: wandb sync /opt/ml/level2_dkt-recsys-04/code/light-gcn/wandb/offline-run-20230518_184640-prlqt6kr
    wandb: Find logs at: ./wandb/offline-run-20230518_184640-prlqt6kr/logs
```

#### Inference Model
```bash
    --------------- LightGCN Load Data ---------------
    --------------- LightGCN Data Split ---------------
    --------------- LightGCN Set Random Seed ---------------
    --------------- LightGCN Data Loader   ---------------
    2023-05-18 18:09:48,940 - root - INFO - Preparing data ...
    /opt/ml/.cache/pypoetry/virtualenvs/lightgcn-model-Xgg5at35-py3.10/lib/python3.10/site-packages/torch/cuda/__init__.py:497: UserWarning: Can't initialize NVML
    warnings.warn("Can't initialize NVML")
    2023-05-18 18:10:00,854 - root - INFO - Train Dataset Info
    2023-05-18 18:10:00,854 - root - INFO -  * Num. Users    : 7442
    2023-05-18 18:10:00,855 - root - INFO -  * Max. UserID   : 7441
    2023-05-18 18:10:00,855 - root - INFO -  * Num. Items    : 9454
    2023-05-18 18:10:00,855 - root - INFO -  * Num. Records  : 2475962
    2023-05-18 18:10:00,855 - root - INFO - Test Dataset Info
    2023-05-18 18:10:00,856 - root - INFO -  * Num. Users    : 744
    2023-05-18 18:10:00,856 - root - INFO -  * Max. UserID   : 7439
    2023-05-18 18:10:00,856 - root - INFO -  * Num. Items    : 444
    2023-05-18 18:10:00,856 - root - INFO -  * Num. Records  : 744
    --------------- LightGCN Load Model ---------------
    2023-05-18 18:10:00,942 - root - INFO - Loading Model ...
    2023-05-18 18:10:00,960 - root - INFO - Load model
    --------------- LightGCN Predict   ---------------
    2023-05-18 18:10:00,966 - root - INFO - Make Predictions & Save Submission ...
    2023-05-18 18:10:00,970 - root - INFO - Saving Result ...

    --------------- Save Output Predict   ---------------
    2023-05-18 18:10:00,977 - root - INFO - Successfully saved submission as ./outputs/lightgcn.csv
```