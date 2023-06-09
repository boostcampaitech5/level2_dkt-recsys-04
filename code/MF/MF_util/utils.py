import os
import random
import time
import numpy as np
import pandas as pd


class Setting:
    @staticmethod
    def set_seeds(seed: int = 42):
        """### [summary]
        랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
        Args:
            - seed (int, optional): _description_. Defaults to 4.
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
        """### [SUMMARY]
        - 예측값을 파일에 작성하기
        Args:
            - filename (str): FileName
            - predict (pd.DataFrame): Predic 결과
        Returns:
            - bool: Save 결과
        """
        with open(filename, "w", encoding="utf8") as w:
            print("writing prediction : {}".format(filename))
            w.write("id,prediction\n")
            for id, p in enumerate(predict):
                w.write("{},{}\n".format(id, p))
        return True

    def make_dir(self, path: str) -> str:
        """
        ### [description]
        - 경로가 존재하지 않을 경우 해당 경로를 생성하며, 존재할 경우 pass를 하는 함수입니다.
        ### [arguments]
        - path : 경로
        ### [return]
        - path : 경로
        """
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass
        return path

    def get_submit_filename(self, args, auc_score: float) -> str:
        """
        ### [description]
        - submit file을 저장할 경로를 반환하는 함수입니다.
        ### [arguments]
        - args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.
        ### [return]
        - filename : submit file을 저장할 경로를 반환합니다.
        - 이 때, 파일명은 submit/날짜_시간_모델명.csv 입니다.
        """
        self.make_dir(args.output_dir)
        filename = f"{args.output_dir}{self.save_time}_{auc_score:.5f}_{args.model}.csv"
        return filename
