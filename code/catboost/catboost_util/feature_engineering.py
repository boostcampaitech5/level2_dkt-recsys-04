import pandas as pd


class FeatureEnginnering:
    def __init__(self, df: pd.DataFrame, feats: list) -> None:
        self.df = df
        self.feats = feats

    def sample(self) -> pd.DataFrame:
        pass
