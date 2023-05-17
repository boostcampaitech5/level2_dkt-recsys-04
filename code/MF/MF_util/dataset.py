import pandas as pd
from typing import Tuple


def train_test_eval_split(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """### [SUMMARY]
    - 훈련 데이터 셋, 테스트 데이터 셋을 입력으로 받아 MF Task에 맞게 정제 후 Train, Test, Eval 데이터 셋으로 나눠 반환합니다.
    - 협업 필터링은 시간에 따른 변화를 고려하지 않기 때문에 Train, Test 데이터 셋에서는 최종 성적만을 바탕으로 평가하도록 중복이 제거되며, userID와 assessmentItemID, answerCode만 활용됩니다.
    - 테스트 데이터 셋에서 answerCode가 -1인 항목은 최종 평가시 사용되는 항목으로 분리하여 Eval 데이터 셋으로 추출합니다.

    Args :
        - train_data (pd.DataFrame) : 훈련 데이터 셋
        - test_data (pd.DataFrame) : 예측 해야할 값(-1)이 포함되어 있는 테스트 데이터 셋

    Returns :
        - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] : MF Task에 맞게 정제된 Train, Test, Eval 데이터 셋


    """

    userid, itemid = list(set(train_data.userID)), list(
        set(train_data.assessmentItemID)
    )
    n_user, n_item = len(userid), len(itemid)

    # 최종 성적만을 바탕으로 평가 -> 사용자 별 각 문제 풀이의 마지막 기록만 남기도록 중복 제거
    train_data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )
    test_data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )

    train_data.drop(
        ["Timestamp", "testId", "KnowledgeTag"], axis=1, inplace=True, errors="ignore"
    )

    # 값이 -1 경우 평가를 위해 제거 (Num Of Records 256073 -> 255329)
    test_data = test_data[test_data.answerCode >= 0].copy()

    userid, itemid = list(set(test_data.userID)), list(set(test_data.assessmentItemID))

    # 평가 항목 신규 생성 -> 남은 테스트 항목 중 각 사용자 별 최종 레코드를 새로운 평가 항목으로 정한다.
    eval_data = test_data.copy()
    eval_data.drop_duplicates(subset=["userID"], keep="last", inplace=True)

    # 평가 항목을 테스트 항목에서 제거
    test_data.drop(index=eval_data.index, inplace=True, errors="ignore")
    print(f" Num. Trains  : {len(train_data)}")
    print(f" Num. Tests  : {len(test_data)}")
    print(f" Num. Predicts  : {len(eval_data)}")

    return train_data, test_data, eval_data
