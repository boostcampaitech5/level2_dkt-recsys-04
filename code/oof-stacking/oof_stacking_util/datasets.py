import os
import time
import random

import numpy as np
import pandas as pd

import torch
from datetime import datetime

from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None

    def get_train_data(self):
        return self.train_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.data_dir, name + "_classes.npy")
        print(le_path)
        np.save(le_path, encoder.classes_)  # 클래스에 대한 라벨 인코딩된 값

    def __preprocessing(self, df):  # 전처리
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]  # 카테고리형 변수
        for col in cate_cols:
            # For UNKNOWN class
            a = df[col].unique().tolist() + [
                np.nan
            ]  # 결측치 추가 -> 결측치도 라벨 인코딩 과정에서 unique 정수 인덱스로 매핑되게 하기 위해서

            le = LabelEncoder()  # 레이블 인코딩(카테고리형 변수 -> 숫자값)
            le.fit(a)
            df[col] = le.transform(df[col])
            self.__save_labels(le, col)

        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def load_data_from_file(self, file_name):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        print(self.args.data_dir)
        df = pd.read_csv(csv_file_path)
        df = self.__preprocessing(df)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = df["assessmentItemID"].nunique()  # 고유 문항 수: 총 9454개
        self.args.n_test = df["testId"].nunique()  # 고유 시험지의 수: 총 1537개
        self.args.n_tag = df["KnowledgeTag"].nunique()  # 고유 태그의 수: 총 912개

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)  # user별 시간순으로 정렬
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                )
            )
        )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct = row[0], row[1], row[2], row[3]

        cate_cols = [test, question, tag, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[:seq_len] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)
