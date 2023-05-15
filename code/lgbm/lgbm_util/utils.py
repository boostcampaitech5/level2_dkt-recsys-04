import os
import json
import random
import time
import argparse
import pandas as pd
import numpy as np

import logging
import logging.config


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
    def set_seeds(seed: int = 4):
        """_summary_
        랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
        Args:
            seed (int, optional): _description_. Defaults to 4.
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    def __init__(self) -> None:
        now = time.localtime()
        now_date = time.strftime("%Y%m%d", now)
        now_hour = time.strftime("%X", now)
        save_time = now_date + "_" + now_hour.replace(":", "")
        self.save_time = save_time

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
        self, output_dir: str, auc_score: float, format_name: str = "csv"
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
        self.make_dir(output_dir)
        filename = (
            f"{output_dir}{self.save_time}_{auc_score:.5f}_lgbm.{format_name}"
        )
        return filename

    def save_config(
        self,
        args: argparse.Namespace,
        auc_score: float,
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
            output_dir=save_path, auc_score=auc_score, format_name="json"
        )
        args_dict = vars(args)
        with open(file_path, "w", encoding="utf-8") as f:
            print(f"saving config : {file_path}")
            json.dump(args_dict, f, indent="\t")
        return file_path


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
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
