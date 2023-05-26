import os
import re
import json
import random
import time
import argparse
import pandas as pd
import numpy as np

from typing import List

import logging
import logging.config

import hyperopt


class process:
    def __init__(self, logger, name):
        self.logger = logger
        self.name = name

    def __enter__(self):
        self.logger.info(f"{self.name} - Started")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info(f"{self.name} - Complete")


class Setting:
    @staticmethod
    def set_seeds(seed: int = 42):
        """_summary_
        랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
        Args:
            seed (int, optional): _description_. Defaults to 4.
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    def __init__(self, args) -> None:
        now = time.localtime()
        now_date = time.strftime("%Y%m%d", now)
        now_hour = time.strftime("%X", now)
        save_time = now_date + "_" + now_hour.replace(":", "")
        self.save_time = save_time
        self.args = args

    def save_predict(self, filename: str, predict: pd.DataFrame) -> bool:
        """_summary_
        예측값을 파일에 작성하기
        Args:
            filename (str): FileName
            predict (pd.DataFrame): Predic 결과

        Returns:
            bool: Save 결과
        """
        with open(filename, "w", encoding="utf8") as w:
            print("writing prediction : {}".format(filename))
            w.write("id,prediction\n")
            for id, p in enumerate(predict):
                w.write("{},{}\n".format(id, p))
        return True

    def make_dir(self, path: str) -> str:
        """
        [description]
        경로가 존재하지 않을 경우 해당 경로를 생성하며, 존재할 경우 pass를 하는 함수입니다.

        [arguments]
        path : 경로

        [return]
        path : 경로
        """
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass
        return path

    def get_submit_filename(
        self,
        output_dir: str,
        auc_score: float,
        cv_info: str = "basic",
        format_name: str = "csv",
    ) -> str:
        """
        [description]
        submit file을 저장할 경로를 반환하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

        [return]
        filename : submit file을 저장할 경로를 반환합니다.
        이 때, 파일명은 submit/날짜_시간_모델명.csv 입니다.
        """

        if self.args.model == "CatBoost":
            self.make_dir(output_dir)
            output_dir = os.path.join(output_dir, "CatBoost")
            self.make_dir(output_dir)
            filename = f"{output_dir}/{self.save_time}_{auc_score:.5f}_catboost_{cv_info}.{format_name}"
        elif self.args.model == "LGBM":
            self.make_dir(output_dir)
            output_dir = os.path.join(output_dir, "LGBM")
            self.make_dir(output_dir)
            filename = f"{output_dir}/{self.save_time}_{auc_score:.5f}_lgbm_{cv_info}.{format_name}"
        elif self.args.model == "XGBoost":
            self.make_dir(output_dir)
            output_dir = os.path.join(output_dir, "XGBoost")
            self.make_dir(output_dir)
            filename = f"{output_dir}/{self.save_time}_{auc_score:.5f}_xgboost_{cv_info}.{format_name}"

        return filename

    def save_config(
        self,
        args: argparse.Namespace,
        auc_score: float,
        cv_info: str,
        save_path: str = "./configs/",
    ) -> str:
        """_summary_

        Argparse의 내용들을 저장하기 위한 Config 파일 저장

        Args:
            args (argparse.Namespace): Argment : hyperparameter
            auc_score (float): AUC score
            save_path (str, optional): 저장 경로. Defaults to "./configs/".

        Returns:
            str: 저장된 경로 반환
        """
        file_path = self.get_submit_filename(
            output_dir=save_path,
            auc_score=auc_score,
            cv_info=cv_info,
            format_name="json",
        )
        args_dict = vars(args)
        with open(file_path, "w", encoding="utf-8") as f:
            print(f"saving config : {file_path}")
            json.dump(args_dict, f, indent="\t")
        return file_path


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
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


def get_logger(logger_conf: dict) -> logging.Logger:
    """
    Return Logger
    Args:
        logger_conf (dict): Logger Config Dict

    Returns:
        logging.Logger: Logger
    """
    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


def find_files_with_string(path: str, search_string: str) -> list:
    """

    해당 경로에서 해당 문자열을 가지는 파일명을 모두 반환하는 함수

    Args:
        path (str): 탐색할 디렉토리
        search_string (str): 탐색할 문자열

    Returns:
        list: 파일 리스트
    """
    files = []
    for file_name in os.listdir(path):
        if file_name.startswith(search_string):
            files.append(file_name)
    return files


def calculate_average_from_list(predicts: list) -> np.ndarray:
    """

    predicts 리스트의 평균을 구하여 최종 predict를 구하는 함수

    Args:
        predicts (list): np.ndarray 리스트

    Returns:
        np.ndarray: average_arr
    """
    sum_arr = predicts[0]
    for arr in predicts[1:]:
        sum_arr += arr
    average_arr = sum_arr / len(predicts)
    return average_arr


def calculate_average_score_from_extract_numbers_from_strings(
    string_list: List[str],
) -> float:
    """

    AUC Score를 반환하는 함수
    20230521_231338_0.85161_catboost_tscv_1.cbm -> 0.85161

    Args:
        string_list (List[str]): 모델의 이름중

    Returns:
        float: AUC Score의 평균을 반환
    """
    numbers = 0.0
    pattern = r"[-+]?\d*\.?\d+|\d+"  # 숫자 또는 실수에 매칭되는 정규 표현식 패턴
    for string in string_list:
        matches = re.findall(pattern, string)
        for match in matches:
            if "." in match:  # 소수점이 있는 경우만 실수로 추출
                numbers += float(match)
    return numbers / len(string_list)


# hyperopt (tuning) 결과 저장 함수
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
