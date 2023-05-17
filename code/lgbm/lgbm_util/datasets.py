import pandas as pd
from typing import Tuple
import random

from sklearn.preprocessing import LabelEncoder

# 사용할 Feature 선택
FEATS = [
    "userID",
    "assessmentItemID",
    "testId",
    "Timestamp",
    "KnowledgeTag",
    "user_correct_answer",
    "user_total_answer",
    "user_acc",
    "test_mean",
    "test_sum",
    "tag_mean",
    "tag_sum",
]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    Feature Engineering을 하는 함수
    1. 유저의 문제 풀이 수, 정답 수, 정답률을 시간순으로 누적한 칼럼 추가
    2. 시험지 별로 전체 유저에 대한 정답률 칼럼 추가
    3. 문제 카테고리 별로 전체 유저에 대한 정답률 칼럼 추가

    Args:
        df (pd.DataFrame): FE를 진행할 Dataframe

    Returns:
        pd.DataFrame: FE가 진행된 DataFrame
    """
    # 유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=["userID", "Timestamp"], inplace=True)

    # 유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df["user_correct_answer"] = df.groupby("userID")["answerCode"].transform(
        lambda x: x.cumsum().shift(1)
    )
    df["user_total_answer"] = df.groupby("userID")["answerCode"].cumcount()
    df["user_acc"] = df["user_correct_answer"] / df["user_total_answer"]

    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    correct_t = df.groupby(["testId"])["answerCode"].agg(["mean", "sum"])
    correct_t.columns = ["test_mean", "test_sum"]
    correct_k = df.groupby(["KnowledgeTag"])["answerCode"].agg(["mean", "sum"])
    correct_k.columns = ["tag_mean", "tag_sum"]

    df = pd.merge(df, correct_t, on=["testId"], how="left")
    df = pd.merge(df, correct_k, on=["KnowledgeTag"], how="left")

    # 범주형 데이터 라벨 인코딩
    label_encoder = LabelEncoder()
    df["assessmentItemID"] = label_encoder.fit_transform(df["assessmentItemID"])
    df["testId"] = label_encoder.fit_transform(df["testId"])
    df["Timestamp"] = label_encoder.fit_transform(df["Timestamp"])

    return df


def custom_train_test_split(
    df: pd.DataFrame, ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        - df (pd.DataFrame): Split 할 Train Set
        - ratio (float, optional): train / valid 셋 중 train 셋 비율. Defaults to 0.8.

    Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: Train , Valid Data Set
    """
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
    return y, X[FEATS]
