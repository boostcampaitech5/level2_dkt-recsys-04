import pandas as pd
import numpy as np
import random

import os

from sklearn.preprocessing import LabelEncoder

import time
from datetime import datetime

import torch
from torch.nn.utils.rnn import pad_sequence


class Preprocess:
    def __init__(self, args, is_train=True):
        self.args = args
        self.train_data = None
        self.is_train = is_train

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
        os.makedirs(f"{self.args.save_dir}/asset", exist_ok=True)
        le_path = os.path.join(f"{self.args.save_dir}/asset", name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df):
        # cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", "elapsed_question", "elapsed_test"]
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", "elapsed_question"]
        for col in cate_cols:
            le = LabelEncoder()
            if self.is_train == True:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(f"{self.args.save_dir}/asset", col + "_classes.npy")
                le.classes_ = np.load(label_path)
                df[col] = df[col].apply(lambda x: x if str(x) in le.classes_ else "unknown")

            df[col] = df[col].astype(str)
            df[col] = le.transform(df[col])

        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

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
        # df["elapsed_test"] = df.groupby(["userID", "testId"])["elapsed_question"].transform("sum")  # 문항별 걸린 시간을 합침

        df["Timestamp"] = df["Timestamp"].map(lambda x: str(x))  # 다시 문자열로 변환

        return df

    def load_data_from_file(self, file_name):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df)

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        # columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag", "elapsed_question", "elapsed_test"]
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag", "elapsed_question"]

        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                    r["elapsed_question"].values,
                    # r["elapsed_test"].values
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

        # test, question, tag, correct, elapsed_question, elapsed_test = row[0], row[1], row[2], row[3], row[4], row[5]
        test, question, tag, correct, elapsed_question = row[0], row[1], row[2], row[3], row[4]

        # cate_cols = [test, question, tag, correct, elapsed_question, elapsed_test]
        cate_cols = [test, question, tag, correct, elapsed_question]

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


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            col_list[i].append(col)

    # 각 column의 값들을 대상으로 padding 진행
    # pad_sequence([[1, 2, 3], [3, 4]]) -> [[1, 2, 3],
    #                                       [3, 4, 0]]
    for i, col_batch in enumerate(col_list):
        col_list[i] = pad_sequence(col_batch, batch_first=True)

    # mask의 경우 max_seq_len을 기준으로 길이가 설정되어있다.
    # 만약 다른 column들의 seq_len이 max_seq_len보다 작다면
    # 이 길이에 맞추어 mask의 길이도 조절해준다
    col_seq_len = col_list[0].size(1)
    mask_seq_len = col_list[-1].size(1)
    if col_seq_len < mask_seq_len:
        col_list[-1] = col_list[-1][:, :col_seq_len]

    return tuple(col_list)


def get_loaders(args, train, valid, inference: bool = False):
    pin_memory = False
    trainset = DKTDataset(train, args)

    if inference == True:
        train_loader = torch.utils.data.DataLoader(
            trainset, shuffle=False, batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate
        )
        valid_loader = None

    else:
        train_loader = torch.utils.data.DataLoader(
            trainset, shuffle=True, batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate
        )

        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset, shuffle=False, batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate
        )

    return train_loader, valid_loader


def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = args.stride

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
                    window_data.append(col[window_i * stride : window_i * stride + window_size])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
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


def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas


def data_augmentation(data, args):
    if args.window == True:
        data = slidding_window(data, args)

    return data
