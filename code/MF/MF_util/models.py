import numpy as np  # linear algebra
import pandas as pd
import random
from typing import Tuple
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, roc_auc_score
from abc import *

_eval = eval


class Latent_factors(metaclass=ABCMeta):
    """## [SUMMARY]
    - Latent factor 모델의 추상 클래스입니다. NMF와 SVD 모델을 구현하기 위해 활용됩니다.
    - SVD와 NMF는 predict하는 코드가 상이하므로 하위 클래스에서 정의하도록 predict 메서드는 추상 메서드로 정의하였습니다.
    - evaluate 로직은 정확하게 동일하기 때문에 상위 클래스에 구현하였습니다.

    Attributes:
        - self.train (pd.DataFrame), self.test (pd.DataFrame), self.eval (pd.DataFrame) : MF 테스크에 맞게 정제된 데이터 셋(dataset 모듈의 train_test_eval_split 메서드의 output)
        - self.matrix_train (pd.DataFrame) : user - item(문제) pivot matrix
        - self.user_id2idx(dict), self.user_idx2id(dict), self.item_id2idx(dict), self.item_idx2id(dict) : user, item to idx mapping dictionary

        - self.X (numpy.ndarray) : user - item(문제) pivot 테이블(`self.matrix_train`)의 values
        - self.XM (numpy.ndarray) : user - item(문제) pivot 테이블을 normalized matrix로 변경
    """

    def __init__(
        self, train: pd.DataFrame, test: pd.DataFrame, eval: pd.DataFrame
    ) -> None:
        self.train = train
        self.test = test
        self.eval = eval

        # user - item(문제)를 pivot 테이블로 변경 (사용자 별로 해당 문제를 맞췄는지)
        self.matrix_train = train.pivot_table(
            "answerCode", index="userID", columns="assessmentItemID"
        )
        self.matrix_train.fillna(0.5, inplace=True)

        self.user_id2idx = {v: i for i, v in enumerate(self.matrix_train.index)}
        self.user_idx2id = {i: v for i, v in enumerate(self.matrix_train.index)}

        self.item_id2idx = {v: i for i, v in enumerate(self.matrix_train.columns)}
        self.item_idx2id = {i: v for i, v in enumerate(self.matrix_train.columns)}

        # user - item(문제) pivot 테이블을 normalized matrix로 변경
        self.X = self.matrix_train.values
        self.XM = self.X - np.mean(self.X, axis=1).reshape(-1, 1)

    @abstractmethod
    def predict(self) -> list:
        """
        하위 클래스에서 정의
        """
        pass

    def evaluate(self) -> Tuple[list, float]:
        """### [SUMMARY]
        - Train, Test, Eval 데이터 셋에 대해 타겟 값 추론 결과를 평가하는 메서드입니다.
        - Test 데이터의 User는 Train 데이터 셋에 존재하지 않기 때문에 별도의 과정으로 추론하게 됩니다(강의 자료 참고)

        Returns
            - pred : 추론한 타겟 값 (after predict eval dataset)
            - score : 추론한 값을 포함한 test set에 대한 AUROC
        """

        ######### train_data ########
        true_values = self.train.answerCode
        prob = self.predict(self.X, mode="train")
        pred = [round(v) for v in prob]

        print("Train data prediction")
        print(f" - Accuracy = {100*accuracy_score(true_values, pred):.2f}%")
        print(f" - ROC-AUC  = {100*roc_auc_score(true_values, prob):.2f}%")

        ######### test_data ########
        # test user의 history에 대한 vector를 train_data의 pivot table에서 가져왔으므로
        # item_id2idx는 train에서 사용한 것을 다시 사용한다.

        userid = sorted(list(set([u for u in self.test.userID])))
        self.user_id2idx_test = {v: i for i, v in enumerate(userid)}

        matrix_test = 0.5 * np.ones((len(userid), len(self.item_id2idx)))
        for user, item, answer in zip(
            self.test.userID, self.test.assessmentItemID, self.test.answerCode
        ):
            user, item = self.user_id2idx_test[user], self.item_id2idx[item]
            matrix_test[user, item] = answer

        true_values = self.test.answerCode
        prob = self.predict(matrix_test, mode="test")
        pred = [round(v) for v in prob]

        print("Test data prediction")
        print(f" - Accuracy = {100*accuracy_score(true_values, pred):.2f}%")
        print(f" - ROC-AUC  = {100*roc_auc_score(true_values, prob):.2f}%")

        # ######## eval_data ########
        true_values = self.eval.answerCode
        prob = self.predict(matrix_test, mode="eval")
        pred = [round(v) for v in prob]
        print("eval data prediction")
        print(f" - Accuracy = {100*accuracy_score(true_values, pred):.2f}%")
        print(f" - ROC-AUC  = {100*roc_auc_score(true_values, prob):.2f}%")
        score = 100 * roc_auc_score(true_values, prob)  # auroc
        return pred, score


class _SVD(Latent_factors):
    """### [SUMMARY]
    #### SVD를 통한 MF
    - SVD는 행렬의 특이값 (Singular Value)를 기반으로 행렬을 분해하는 기법 [참조](https://angeloyeo.github.io/2019/08/01/SVD.html)
    - SVD는 행렬 X에 대해 다음 식이 성립하는 행렬 값을 계산
    - X@V = U@Sigma
    - U 는 각 축이 서로 직교하는 백터로 구성된 행렬
    - Sigma 는 각 대각 원소 sigma_i가 특이값을 나타내는 대각 행렬
    - 이를 통해, 직교행렬(orthogonal matrix) V의 열벡터에 대해, A로 선형 변환할 때 크기가 각각 sigma_i 만큼 변하지만 여전히 직교하는 행렬 U를 찾는 문제로 나타낼 수 있음
    ### SVD 협업 필터링을 통한 DKT 문제
    - DKT 에서의 '지식'은 문제를 맞출 수 있을지 없을지에 대한 hidden factor
    - DKT에서의 '지식'은 협업 필터링에서의 사용자의 '취향'과, 문제의 정답 여부는 협업 필터링에서의 아이템의 '선호도'로 매핑될 수 있음
    - 따라서 선호를 통해 사용자의 취향을 추론하듯, 문제의 정답 여부를 통해 문제에 대한 지식 정도를 추론 가능
    - 단, DKT에서는 주로 시간에 따른 지식 변화를 '추적'하는 데에 초점을 맞추고 있으나, 협업 필터링은 시간에 따른 취향의 변화가 없음을 전제로 하고 있음. 따라서 시간 요소는 협업 필터링에서 보통 고려되지 않으며, 이에 대한 추론에 한계가 있음을 감안할 필요가 있음.

    Attributes:
        - `self.U(np.ndarray)`, `self.sigma(np.ndarray`), `self.V(np.ndarray)` : nomalized user-item matrix를 svd로 분해한 U, sigma, v matrix


    """

    def __init__(
        self, train: pd.DataFrame, test: pd.DataFrame, eval: pd.DataFrame, k: int = 12
    ) -> None:
        super().__init__(train, test, eval)

        self.U, self.sigma, self.V = svds(self.XM, k=k)
        print(f"U={self.U.shape}, sigma={self.sigma.shape}, V={self.V.shape}")
        print(f"Singular Vlaues : {self.sigma}")

    def predict(self, B_matrix: pd.DataFrame, mode: str = "eval") -> list:
        """### [SUMMARY]
        학습 및 학습되지 않은 사용자가 포함된 target matrix에 대해서도 SVD를 활용해 추론하는 메서드

        Args:
            - B_matrix (pd.DataFrame) : 학습 or 학습되지 않은 사용자가 포함된 target matrix
            - mode (str):
                - train : B_matrix로 train data set을 지정한 경우, 클래스 객체 생성 시 생성되는 `user_id2idx`가 예측 행렬의 row로 들어감
                - test : B_matrix로 test_data_set을 지정한 경우, evaluate에서 생성하는 `user_id2idx_test`가 예측 행렬의 row로 들어감
                - eval : B_matrix로 eval_data_set을 지정한 경우, evaluate에서 생성하는 `user_id2idx_test`가 예측 행렬의 row로 들어감
        Returns:
            - ret (List) : 추론한 값이 저장된 list
        """
        Sigma = np.diag(self.sigma)
        Sigma_i = np.diag(1 / self.sigma)
        pred_matrix = self.V.T @ Sigma_i @ Sigma @ self.V

        B_mean = np.mean(B_matrix, axis=1)
        BM = B_matrix - B_mean.reshape(-1, 1)  # BM = B Matrix

        B_pred = BM @ pred_matrix + B_mean.reshape(-1, 1)

        if mode == "train":
            return [
                B_pred[self.user_id2idx[u], self.item_id2idx[i]]
                for u, i in zip(self.train.userID, self.train.assessmentItemID)
            ]
        else:
            return [
                B_pred[self.user_id2idx_test[u], self.item_id2idx[i]]
                for u, i in zip(
                    _eval(f"self.{mode}").userID, _eval(f"self.{mode}").assessmentItemID
                )
            ]


class _NMF(Latent_factors):
    """### [SUMMARY]
    #### 비음수 행렬 분해 (Non-negative matrix factorization: NMF) 를 통한 MF
    - 비음수 행렬 분해(Non-negative matrix factorization, NMF)는 음수를 포함하지 않은 행렬 V를 음수를 포함하지 않은 행렬 W와 H의 곱으로 분해하는 알고리즘이다.
    - 행렬이 음수를 포함하지 않는 성질은 분해 결과 행렬을 찾기 쉽게 만든다.
    - 일반적으로 행렬 분해는 정확한 해가 없기 때문에 이 알고리즘은 대략적인 해를 구하게 된다.
    - 비음수 행렬 분해는 컴퓨터 시각 처리, 문서 분류, 음파 분석, 계량분석화학, 추천 시스템 등에 쓰인다.[(Wikipedia)](https://ko.wikipedia.org/wiki/%EB%B9%84%EC%9D%8C%EC%88%98_%ED%96%89%EB%A0%AC_%EB%B6%84%ED%95%B4)
    #### NMF 협업 필터링을 통한 DKT 문제
    - DKT 에서의 '지식'은 문제를 맞출 수 있을지 없을지에 대한 hidden factor
    - DKT에서의 '지식'은 협업 필터링에서의 사용자의 '취향'과, 문제의 정답 여부는 협업 필터링에서의 아이템의 '선호도'로 매핑될 수 있음
    - 따라서 선호를 통해 사용자의 취향을 추론하듯, 문제의 정답 여부를 통해 문제에 대한 지식 정도를 추론 가능
    - 단, DKT에서는 주로 시간에 따른 지식 변화를 '추적'하는 데에 초점을 맞추고 있으나, 협업 필터링은 시간에 따른 취향의 변화가 없음을 전제로 하고 있음. 따라서 시간 요소는 협업 필터링에서 보통 고려되지 않으며, 이에 대한 추론에 한계가 있음을 감안할 필요가 있음.

    Attributes:
        - self.nmf : NMF모델(클래스)
    """

    def __init__(
        self, train: pd.DataFrame, test: pd.DataFrame, eval: pd.DataFrame, k: int = 12
    ) -> None:
        super().__init__(train, test, eval)

        self.nmf = NMF(n_components=k)
        self.nmf.fit(self.X)

    def predict(self, B_matrix: pd.DataFrame, mode: str = "eval") -> list:
        """### [SUMMARY]
        학습 및 학습되지 않은 사용자가 포함된 target matrix에 대해서도 NMF를 활용해 추론하는 메서드

        Args:
            - B_matrix (pd.DataFrame) : 학습 or 학습되지 않은 사용자가 포함된 target matrix
            - mode (str):
                - train : B_matrix로 train data set을 지정한 경우, 클래스 객체 생성 시 생성되는 `user_id2idx`가 예측 행렬의 row로 들어감
                - test : B_matrix로 test_data_set을 지정한 경우, evaluate에서 생성하는 `user_id2idx_test`가 예측 행렬의 row로 들어감
                - eval : B_matrix로 eval_data_set을 지정한 경우, evaluate에서 생성하는 `user_id2idx_test`가 예측 행렬의 row로 들어감
        Returns:
            - ret (List) : 추론한 값이 저장된 list
        """
        X = B_matrix
        X_pred = self.nmf.inverse_transform(self.nmf.transform(X))

        if mode == "train":
            return [
                X_pred[self.user_id2idx[u], self.item_id2idx[i]]
                for u, i in zip(self.train.userID, self.train.assessmentItemID)
            ]
        else:
            return [
                X_pred[self.user_id2idx_test[u], self.item_id2idx[i]]
                for u, i in zip(
                    _eval(f"self.{mode}").userID, _eval(f"self.{mode}").assessmentItemID
                )
            ]
