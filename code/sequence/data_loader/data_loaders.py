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
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        max_seq_len: int,
        is_train: bool = True,
        asset_dir: str = "asset/",
    ):
        self.data_dir = data_dir
        self.asset_dir = asset_dir
        self.max_seq_len = max_seq_len

        preprocess = Preprocess(data_dir, asset_dir, is_train)
        df = preprocess.get_data()

        self.get_npy()

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)

        columns = [
            "userID",
            "assessmentItemID",
            "testId",
            "KnowledgeTag",
            "answerCode",
            "elapsed_question",
            "elapsed_test",
        ]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["assessmentItemID"].values,
                    r["testId"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
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

        question, test, tag, correct, elapsed_question, elapsed_test = row[0], row[1], row[2], row[3], row[4], row[5]
        data = {
            "question": torch.tensor(question + 1, dtype=torch.int),
            "test": torch.tensor(test + 1, dtype=torch.int),
            "tag": torch.tensor(tag + 1, dtype=torch.int),
            "correct": torch.tensor(correct, dtype=torch.int),
            "elapsed_question": torch.tensor(elapsed_question + 1, dtype=torch.int),
            "elapsed_test": torch.tensor(elapsed_test + 1, dtype=torch.int),
        }

        # Generate mask: max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len :]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len - seq_len :] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        data["mask"] = mask

        # Generate interaction
        interaction = data["correct"] + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1)
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        data["interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}

        return data

    def get_npy(self):
        # self.n_questions = n_questions
        self.n_questions = len(np.load(os.path.join(self.asset_dir, "assessmentItemID_classes.npy")))
        # self.n_tests = n_tests
        self.n_tests = len(np.load(os.path.join(self.asset_dir, "testId_classes.npy")))
        # self.n_tags = n_tags
        self.n_tags = len(np.load(os.path.join(self.asset_dir, "KnowledgeTag_classes.npy")))
        # self.n_elapsed_question = n_elapsed_question
        self.n_elapsed_questions = len(np.load(os.path.join(self.asset_dir, "elapsed_question_classes.npy")))
        # self.n_elapsed_tests = n_elapsed_tests
        self.n_elapsed_tests = len(np.load(os.path.join(self.asset_dir, "elapsed_test_classes.npy")))
