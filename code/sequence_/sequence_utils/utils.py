import os
import random

import numpy as np
import pandas as pd

import torch
import hyperopt


class process:
    def __init__(self, logger, name):
        self.logger = logger
        self.name = name

    def __enter__(self):
        self.logger.info(f"{self.name} - Started")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info(f"{self.name} - Complete")


def seed_everything(seed=42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 로거 객체 생성
def get_logger(logger_conf: dict):
    import logging
    import logging.config

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {"basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}

# 하이퍼파라미터튜닝 결과 저장 함수
def trials_to_df(trials, space, best):
    # 전체 결과
    rows = []
    keys = list(trials.trials[0]["misc"]["vals"].keys())

    # 전체 실험결과 저장
    for trial in trials:
        row = {}

        # tid
        tid = trial["tid"]
        row["experiment"] = str(tid)

        # hyperparameter 값 저장
        vals = trial["misc"]["vals"]
        hparam = {key: value[0] for key, value in vals.items()}

        # space가 1개 - 값을 바로 반환
        # space가 다수 - dict에 값을 반환
        hparam = hyperopt.space_eval(space, hparam)

        if len(keys) == 1:
            row[keys[0]] = hparam
        else:
            for key in keys:
                row[key] = hparam[key]

        # metric
        row["metric"] = abs(trial["result"]["loss"])

        # 소요 시간
        row["time"] = (trial["refresh_time"] - trial["book_time"]).total_seconds()

        rows.append(row)

    experiment_df = pd.DataFrame(rows)

    # best 실험
    row = {}
    best_hparam = hyperopt.space_eval(space, best)

    if len(keys) == 1:
        row[keys[0]] = best_hparam
    else:
        for key in keys:
            row[key] = best_hparam[key]
    row["experiment"] = "best"

    best_df = pd.DataFrame([row])

    # best 결과의 auc / time searching 하여 찾기
    search_df = pd.merge(best_df, experiment_df, on=keys)

    # column명 변경
    search_df = search_df.drop(columns=["experiment_y"])
    search_df = search_df.rename(columns={"experiment_x": "experiment"})

    # 가장 좋은 metric 결과 중 가장 짧은 시간을 가진 결과를 가져옴
    best_time = search_df.time.min()
    search_df = search_df.query("time == @best_time")

    df = pd.concat([experiment_df, search_df], axis=0)

    return df