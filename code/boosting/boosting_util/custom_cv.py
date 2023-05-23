import numpy as np
from sklearn.model_selection import BaseCrossValidator


class UserBasedKFold(BaseCrossValidator):
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def _iter_test_indices(self, X, y=None, groups=None):
        n_users = X.userID.nunique()
        user_cnt = X.groupby(["userID"]).count()["assessmentItemID"].values
        user_cnt_tup_list = np.array(
            sorted(
                [(user, cnt) for user, cnt in enumerate(user_cnt)], key=lambda x: x[1]
            )
        )

        for i in range(self.n_splits):
            indices = (
                np.arange(n_users, step=self.n_splits) + i
            )  # sequence 수로 정렬된 리스트에서 가져올 index
            indices = indices[indices <= n_users - 1]  # 범위를 초과하는 부분 예외 처리
            userID_list = np.array([user for user, _ in user_cnt_tup_list[indices]])
            fold_indices = X.loc[X["userID"].isin(userID_list)].index.values

            yield fold_indices
