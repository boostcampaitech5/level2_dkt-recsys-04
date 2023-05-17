import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class FeatureEnginnering:
    """
    이산화 (Binning)
        문제 풀이 시간 구간으로 나누기
    상호작용 (Interaction)
        범주형 / 범주형
        수치형 / 수치형
        범주형 / 수치형
    행렬 분해(Matrix Factorization)
        SVD
        Truncated SVD
        NMF
    차원 축소(Dimension Reduction)
        PCA
            유저들에 대한 문제별 정답 횟수 행렬 PCA
            시간과 연관된 feature들의 PCA
        LDA (Linear Discriminant Analysis)
            시간과 연관된 feature들의 LDA
        LDA(Latent Dirichlet Allocation)
            문제들에 대한 유저별 정답 횟수 행렬 LDA
    임베딩 (Embedding)
        카테고리 임베딩
        Word2Vec
            문제를 word2vec을 적용하여 임베딩
    """

    def __init__(self, df: pd.DataFrame, feats: list) -> None:
        self.df = df
        self.feats = feats

        self.df = self.df.drop_duplicates(
            subset=["userID", "assessmentItemID"], keep="last"
        )
        self.df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    def sample(self) -> pd.DataFrame:
        pass

    def calculate_solve_time_column(self, threshold: float = 700) -> pd.DataFrame:
        df = self.df.copy()

        if "time" not in df.columns:
            df["time"] = (
                df.groupby(["userID", "testId"])
                .Timestamp.diff()
                .map(lambda x: x.total_seconds())
                .shift(-1)
                .fillna(method="ffill")
            )
            df.loc[df["time"] > threshold]["time"] = threshold

        return df

    def calculate_time_cut_column(self, bins: int = 3) -> pd.DataFrame:
        df = self.df.copy()

        if "time" not in df.columns:
            self.df = self.calculate_solve_time_column()
        df["time_cut"] = pd.cut(df["time"], bins=bins)

        return df

    def concat_column_name(self) -> pd.DataFrame:
        # 범주형 featrue 합치기
        df = self.df.copy()
        df["user|item"] = df.userID.map(str) + "|" + df.assessmentItemID.map(str)

        # 원핫 인코딩 -> 메모리 이슈
        # X = df['user|item'].values.reshape(-1, 1)
        # enc = OneHotEncoder()
        # encoded = enc.fit_transform(X).toarray()

        # interaction_df = pd.DataFrame(encoded, columns=enc.categories_[0], dtype=np.int8)

        # return pd.concat([df, interaction_df], axis=1)
        return df


if __name__ == "__main__":
    df = pd.read_csv("/opt/ml/input/data/train_data.csv")
    FE = FeatureEnginnering(df=df, feats=[])
    print(FE.concat_column_name().head())
