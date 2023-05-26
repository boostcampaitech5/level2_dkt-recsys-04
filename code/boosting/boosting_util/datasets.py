import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Tuple
import random
from .feature_engineering import FeatureEnginnering
from sklearn.preprocessing import LabelEncoder


def preprocessing(dataframe: pd.DataFrame) -> Tuple[list, pd.DataFrame]:
    cate_cols = ["assessmentItemID", "testId"]
    # LabelEncoding
    for col in cate_cols:
        le = LabelEncoder()
        # For UNKNOWN class
        a = dataframe[col].unique().tolist() + ["unknown"]
        le.fit(a)

        # cate_cols 는 범주형이라고 가정
        dataframe[col] = dataframe[col].astype(str)
        encoded_values = le.transform(dataframe[col])
        dataframe[col] = encoded_values

    def convert_time(s: str):
        timestamp = time.mktime(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())
        return int(timestamp)

    dataframe["Timestamp"] = dataframe["Timestamp"].map(lambda x: str(x))
    dataframe["Timestamp"] = dataframe["Timestamp"].apply(convert_time)
    cate_cols.append("Timestamp")
    try:
        dataframe["time_cut_enc"] = dataframe["time_cut_enc"].astype(int)
        cate_cols.append("time_cut_enc")
    except Exception:
        print("'time_cut_enc' column not in dataframe.")
        pass

    return cate_cols, dataframe


def feature_engineering(feats: list, df: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    Feature Engineering을 하는 함수

    Args:
        df (pd.DataFrame): FE를 진행할 Dataframe
        feats (list): 진행FE

    Returns:
        pd.DataFrame: FE가 진행된 DataFrame
    """
    # 유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df = FeatureEnginnering(df, feats).df
    return df


def custom_train_test_split(
    df: pd.DataFrame, ratio: float = 0.7, user_id_dir: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        - df (pd.DataFrame): Split 할 Train Set
        - ratio (float, optional): train / valid 셋 중 train 셋 비율. Defaults to 0.7.

    Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: Train , Valid Data Set
    """
    if user_id_dir:
        train_user_id = pd.read_csv(
            os.path.join(user_id_dir, "userid_train.csv")
        )
        train = df[df["userID"].isin(train_user_id["userID"])]
        test = df[df["userID"].isin(train_user_id["userID"]) == False]
    else:
        users = list(
            zip(df["userID"].value_counts().index, df["userID"].value_counts())
        )
        random.shuffle(users)

        max_train_data_len = ratio * len(df)
        sum_of_train_data = 0
        user_ids = []

        for user_id, count in users:
            sum_of_train_data += count
            if max_train_data_len < sum_of_train_data:
                break
            user_ids.append(user_id)

        train = df[df["userID"].isin(user_ids)]
        test = df[df["userID"].isin(user_ids) == False]

        # test데이터셋은 각 유저의 마지막 interaction만 추출
        test = test[test["userID"] != test["userID"].shift(-1)]
    return (train, test)


def custom_label_split(df: pd.DataFrame) -> Tuple[list, pd.DataFrame]:
    """_summary_

    Args:
        - df (pd.DataFrame): label을 분리할 데이터 셋

    Returns:
        - Tuple[list, pd.DataFrame]: label, Data 셋 반환
    """
    y = df["answerCode"]
    X = df.drop(["answerCode"], axis=1)
    return y, X


class BlockingTimeSeriesSplit:
    def __init__(self, n_splits=None):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start:mid], indices[mid + margin : stop]
