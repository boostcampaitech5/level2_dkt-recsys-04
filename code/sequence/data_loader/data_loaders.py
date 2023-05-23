import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import default_collate

from sklearn.preprocessing import LabelEncoder

from base import BaseDataLoader


from typing import Tuple


class Preprocess:
    def __init__(self, data_dir, asset_dir, is_train):
        self.data_dir = data_dir
        self.asset_dir = asset_dir
        self.is_train = is_train

    def get_data(self):
        if self.is_train:
            csv_file_path = os.path.join(self.data_dir, "train_data.csv")
        else:
            csv_file_path = os.path.join(self.data_dir, "test_data.csv")

        df = pd.read_csv(csv_file_path)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, self.is_train)

        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # 문자 -> 날짜 형식으로 변환
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # 문항별 걸린 시간 (초)
        df["elapsed_question"] = df.groupby(["userID", "testId"])["Timestamp"].diff()
        df["elapsed_question"] = df["elapsed_question"].map(lambda x: x.seconds)  # 초 단위로 저장
        df["elapsed_question"].fillna(df["elapsed_question"].mode()[0], inplace=True)  # 최빈값으로 결측치 대체
        df["elapsed_question"].map(lambda x: 259200 if x > 259200 else x)  # 최대 3일(=259200초)로 clip

        # 시험지별 걸린 시간 (초)
        df["elapsed_test"] = df.groupby(["userID", "testId"])["elapsed_question"].transform("sum")  # 문항별 걸린 시간을 합침

        df["Timestamp"] = df["Timestamp"].map(lambda x: str(x))  # 다시 문자열로 변환

        return df

    def __preprocessing(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", "elapsed_question", "elapsed_test"]

        # LabelEncoding
        if not os.path.exists(self.asset_dir):
            os.makedirs(self.asset_dir)

        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(lambda x: x if str(x) in le.classes_ else "unknown")

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s: str):
            timestamp = time.mktime(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)


class SequentialDataset(torch.utils.data.Dataset):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        shuffle: bool,
        validation_split: float,
        num_workers: int,
        shuffle_n: int,
        augmentation: bool,
        max_seq_len: int,
        is_train: bool = True,
        asset_dir: str = "asset/",
        stride: int = 10,
    ):
        self.data_dir = data_dir
        self.asset_dir = asset_dir
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.shuffle = shuffle
        self.shuffle_n = shuffle_n

        preprocess = Preprocess(data_dir, asset_dir, is_train)
        df = preprocess.get_data()

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)

        columns = [
            "userID",
            "answerCode",
            "assessmentItemID",
            "testId",
            "KnowledgeTag",
            "elapsed_question",
            "elapsed_test",
        ]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["answerCode"].values,
                    r["assessmentItemID"].values,
                    r["testId"].values,
                    r["KnowledgeTag"].values,
                    r["elapsed_question"].values,
                    r["elapsed_test"].values,
                )
            )
        )

        self.data = group.values

        # super().__init__(self.data, batch_size, shuffle, validation_split, num_workers)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]

        correct, question, test, tag,  elapsed_question, elapsed_test = row[0], row[1], row[2], row[3], row[4], row[5]
        # cate_cols = [col for col in row]
        cate_cols = [correct, question, test, tag,  elapsed_question, elapsed_test]

        # Generate mask: max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        seq_len = len(row[0])
        
        if seq_len > self.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.max_seq_len:]
            mask = np.ones(self.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.max_seq_len, dtype=np.int16)
            mask[:seq_len] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols
    
    def slidding_window(self, data):
        window_size = self.max_seq_len
        stride = self.stride

        def shuffle(shuffle_n, win_data, win_data_size):
            shuffle_datas = []
            for i in range(shuffle_n):
                # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
                shuffle_data = []
                random_index = np.random.permutation(win_data_size)
                for col in win_data:
                    shuffle_data.append(col[random_index])
                shuffle_datas.append(tuple(shuffle_data))
            return shuffle_datas

        augmented_datas = []
        for row in data:
            seq_len = len(row[0])

            # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
            if seq_len <= window_size:
                augmented_datas.append(row)
            else:
                total_window = ((seq_len - window_size) // stride) + 1
                
                # 앞에서부터 slidding window 적용
                for window_i in range(total_window):
                    # window로 잘린 데이터를 모으는 리스트
                    window_data = []
                    for col in row:
                        window_data.append(col[window_i*stride:window_i*stride + window_size])

                    # Shuffle
                    # 마지막 데이터의 경우 shuffle을 하지 않는다
                    if self.shuffle and window_i + 1 != total_window:
                        shuffle_datas = shuffle(self.shuffle_n, window_data, window_size)
                        augmented_datas += shuffle_datas
                    else:
                        augmented_datas.append(tuple(window_data))

                # slidding window에서 뒷부분이 누락될 경우 추가
                total_len = window_size + (stride * (total_window - 1))
                if seq_len != total_len:
                    window_data = []
                    for col in row:
                        window_data.append(col[-window_size:])
                    augmented_datas.append(tuple(window_data))

        return augmented_datas        

                