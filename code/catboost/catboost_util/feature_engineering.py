import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import LatentDirichletAllocation


class FeatureEnginnering:
    """
    {메소드 이름} : {추가되는 칼럼명}
    calculate_cumulative_stats_by_time : user_correct_answer, user_total_answer, user_acc
    calculate_overall_accuracy_by_testID : test_mean, test_sum
    calculate_overall_accuracy_by_KnowledgeTag : tag_mean, tag_sum

    calculate_solve_time_column : time
    check_answer_at_time : correct_shift_-2, correct_shift_-1, correct_shift_1, correct_shift_2
    calculate_total_time_per_user : total_used_time
    calculate_past_correct_answers_per_user : past_correct
    calculate_future_correct_answers_per_user : future_correct

    calculate_past_correct_attempts_per_user : past_content_correct
    calculate_past_solved_problems_per_user : past_count
    calculate_past_average_accuracy_per_user : average_correct
    calculate_past_average_accuracy_current_problem_per_user : average_content_correct

    calculate_rolling_mean_time_last_3_problems_per_user : mean_time
    # calculate_mean_and_stddev_per_user : {모든 수치형 데이터 칼럼}_mean, {모든 수치형 데이터 칼럼}_std
    calculate_median_time_per_user : time_median
    calculate_problem_solving_time_per_user : hour

    calculate_accuracy_by_time_of_day : correct_per_hour
    calculate_user_activity_time_preference : is_night
    calculate_normalized_time_per_user : normalized_time
    calculate_relative_time_spent_per_user : relative_time

    calculate_time_cut_column : 소요 시간 column에 대한 구간 bins개로 나누기
    calculate_items_svd_latent : 유저에 대한 각 문제의 svd latent 구하기
    calculate_times_nmf_latent : 각 문제 풀이 시간에 대한 nmf latent 구하기
    calculate_users_pca_latent : 각 문제의 대한 유저 latent를 저차원으로 pca
    calculate_items_pca_latent : 각 유저에 대한 아이템 latent를 저차원으로 pca
    calculate_times_pca_latent : 각 문제 풀이 시간에 대한 latent를 저차원으로 pca
    calculate_times_lda_latent : 각 문제 풀이 시간에 대한 latent를 저차원으로 lda
    caculate_item_latent_dirichlet_allocation : 각 고차원 데이터의 문제에 대한 K개의 주제(Topic) 비율을 알아내어 이를 저차원 데이터로 사용
    caculate_user_latent_dirichlet_allocation : 각 고차원 데이터의 유저에 대한 K개의 주제(Topic) 비율을 알아내어 이를 저차원 데이터로 사용
    """

    def __init__(self, df: pd.DataFrame, feats: list) -> None:
        self.df = df
        self.df.sort_values(by=["userID", "Timestamp"], inplace=True)
        self.df = self.df.drop_duplicates(
            subset=["userID", "assessmentItemID"], keep="last"
        )
        self.df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        self.feats = feats
        self.call_methods()

    def call_methods(self):
        success = 0
        for name in self.feats:
            feat = getattr(self, name)
            condition = feat()
            success += condition
            if not condition:
                print(f"Fail : {name}")
        COLUMNS = list(self.df.columns)
        COLUMNS.remove("answerCode")
        COLUMNS.append("answerCode")
        self.df = self.df[COLUMNS]
        print(f"FE Success : {success} / {len(self.feats)}")

    def calculate_solve_time_column(self, threshold: float = 700) -> bool:
        """
        문제 풀이시간을 Feature로 추가하기
        EDA 결과 650초를 넘어가면 정답률이 낮아지기에
        만약 700초가 넘어가면 700초로 통일 될 수 있게 변경

        Args:
            threshold (float, optional): _description_. Defaults to 700.

        Returns:
            pd.DataFrame: _description_
        """
        try:
            if "time" not in self.df.columns:
                self.df["time"] = (
                    self.df.groupby(["userID", "testId"])
                    .Timestamp.diff()
                    .map(lambda x: x.total_seconds())
                    .shift(-1)
                    .fillna(method="ffill")
                )
                self.df.loc[self.df["time"] > threshold]["time"] = threshold
        except Exception:
            return False
        return True

    def calculate_cumulative_stats_by_time(self) -> bool:
        """
        유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산

        Returns:
            bool: 성공여부
        """
        try:
            # 유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
            self.df["user_correct_answer"] = self.df.groupby("userID")[
                "answerCode"
            ].transform(lambda x: x.cumsum().shift(1))
            self.df["user_total_answer"] = self.df.groupby("userID")[
                "answerCode"
            ].cumcount()
            self.df["user_acc"] = (
                self.df["user_correct_answer"] / self.df["user_total_answer"]
            )
        except Exception:
            return False
        return True

    def calculate_overall_accuracy_by_testID(self) -> bool:
        """
        시험지 별로 전체 유저에 대한 정답률 칼럼 추가

        Returns:
            bool: 성공여부
        """
        try:
            correct_t = self.df.groupby(["testId"])["answerCode"].agg(
                ["mean", "sum"]
            )
            correct_t.columns = ["test_mean", "test_sum"]
            self.df = pd.merge(self.df, correct_t, on=["testId"], how="left")

        except Exception:
            return False
        return True

    def calculate_overall_accuracy_by_KnowledgeTag(self) -> bool:
        """
        문제 카테고리 별로 전체 유저에 대한 정답률 칼럼 추가

        Returns:
            bool: 성공여부
        """
        try:
            correct_k = self.df.groupby(["KnowledgeTag"])["answerCode"].agg(
                ["mean", "sum"]
            )
            correct_k.columns = ["tag_mean", "tag_sum"]
            self.df = pd.merge(
                self.df, correct_k, on=["KnowledgeTag"], how="left"
            )
        except Exception:
            return False
        return True

    def check_answer_at_time(self) -> bool:
        """
        유저별로 미래/과거의 특정 시점에 문제 정답을 맞췄는지 못 맞췄는지에 대한 여부를 feature로 추가
        미래나 과거 정보가 없을 경우 NaN이 채워짐

        Returns:
            bool: 성공여부
        """
        try:
            # 미래 정보
            self.df["correct_shift_-2"] = self.df.groupby("userID")[
                "answerCode"
            ].shift(-2)
            self.df["correct_shift_-1"] = self.df.groupby("userID")[
                "answerCode"
            ].shift(-1)

            # 과거 정보
            self.df["correct_shift_1"] = self.df.groupby("userID")[
                "answerCode"
            ].shift(1)
            self.df["correct_shift_2"] = self.df.groupby("userID")[
                "answerCode"
            ].shift(2)
        except Exception:
            return False
        return True

    def calculate_total_time_per_user(self) -> bool:
        """
        누적합을 이용하여 유저별로 문제풀이에 사용한 총 시간을 featTure로 추가

        Returns:
            bool: 성공여부
        """
        try:
            self.df["total_used_time"] = self.df.groupby("userID")[
                "time"
            ].cumsum()
        except Exception:
            return False
        return True

    def calculate_past_correct_answers_per_user(self) -> bool:
        """
        유저별 과거에 맞춘 문제 수를 feature로 추가

        Returns:
            bool: 성공여부
        """
        try:
            # 과거에 맞춘 문제 수
            self.df["shift"] = (
                self.df.groupby("userID")["answerCode"].shift().fillna(0)
            )
            self.df["past_correct"] = self.df.groupby("userID")[
                "shift"
            ].cumsum()
            self.df.drop("shift", axis=1, inplace=True)
        except Exception:
            return False
        return True

    def calculate_future_correct_answers_per_user(self) -> bool:
        """
        유저별 미래에 맞출 문제 수를 feature로 추가

        Returns:
            bool: 성공여부
        """
        try:
            reversed_df = self.df.iloc[::-1].copy()
            # 미래에 맞출 문제 수
            reversed_df["shift"] = (
                reversed_df.groupby("userID")["answerCode"].shift().fillna(0)
            )
            reversed_df["future_correct"] = reversed_df.groupby("userID")[
                "shift"
            ].cumsum()
            self.df = reversed_df.iloc[::-1]
            self.df.drop("shift", axis=1, inplace=True)
        except Exception:
            return False
        return True

    def calculate_past_correct_attempts_per_user(self) -> bool:
        """
        유저별 현재 풀고 있는 문제를 과거에 맞춘 횟수를 feature로 추가
        유저가 해당 문제를 과거에 풀었다면 다시 마주할 경우 더욱 쉽게 풀 것이라는 가정이 깔려있다.

        Returns:
            bool: 성공여부
        """
        try:
            # 과거에 해당 문제를 맞춘 횟수
            self.df["shift"] = (
                self.df.groupby(["userID", "assessmentItemID"])["answerCode"]
                .shift()
                .fillna(0)
            )
            self.df["past_content_correct"] = self.df.groupby(
                ["userID", "assessmentItemID"]
            )["shift"].cumsum()
            self.df.drop("shift", axis=1, inplace=True)
        except Exception:
            return False
        return True

    def calculate_past_solved_problems_per_user(self) -> bool:
        """
        유저별 과거에 푼 문제수를 feature로 추가

        Returns:
            bool: 성공여부
        """
        try:
            # 과거에 푼 문제 수
            self.df["past_count"] = self.df.groupby("userID").cumcount()
        except Exception:
            return False
        return True

    def calculate_past_average_accuracy_per_user(self) -> bool:
        """
        유저별 과거 평균 정답률을 feature로 추가해보자.

        다음의 2가지 feature를 만든 후 이를 나눈다.
        - 과거에 맞춘 문제 수
        - 과거에 푼 문제 수

        과거 평균 정답률 = 과거에 맞춘 문제 수 % 과거에 푼 문제 수

        Returns:
            bool: 성공여부
        """
        try:
            # 과거에 푼 문제 수
            self.df["past_count"] = self.df.groupby("userID").cumcount()

            # 과거에 맞춘 문제 수
            self.df["shift"] = (
                self.df.groupby("userID")["answerCode"].shift().fillna(0)
            )
            self.df["past_correct"] = self.df.groupby("userID")[
                "shift"
            ].cumsum()

            # 과거 평균 정답률
            self.df["average_correct"] = (
                self.df["past_correct"] / self.df["past_count"]
            ).fillna(0)
            self.df.drop("shift", axis=1, inplace=True)
            self.df.drop("past_correct", axis=1, inplace=True)

        except Exception:
            return False
        return True

    def calculate_past_average_accuracy_current_problem_per_user(self) -> bool:
        """
        유저별 현재 풀고 있는 문제의 과거 평균 정답률을 feature로 추가해보자.

        다음의 2가지 feature를 만든 후 이를 나눈다.
        - 과거에 해당 문제를 맞춘 수
        - 과거에 해당 문제를 푼 수

        과거 해당 문제 평균 정답률 = 과거에 해당 문제를 맞춘 수 % 과거에 해당 문제를 푼 수

        Returns:
            bool: 성공여부
        """
        try:
            # 과거에 해당 문제를 푼 수
            self.df["past_content_count"] = self.df.groupby(
                ["userID", "assessmentItemID"]
            ).cumcount()

            # 과거에 해당 문제를 맞춘 수
            self.df["shift"] = (
                self.df.groupby(["userID", "assessmentItemID"])["answerCode"]
                .shift()
                .fillna(0)
            )
            self.df["past_content_correct"] = self.df.groupby(
                ["userID", "assessmentItemID"]
            )["shift"].cumsum()

            # 과거 해당 문제 평균 정답률
            self.df["average_content_correct"] = (
                self.df["past_content_correct"] / self.df["past_content_count"]
            ).fillna(0)

            self.df.drop("shift", axis=1, inplace=True)
            self.df.drop("past_content_correct", axis=1, inplace=True)

        except Exception:
            return False
        return True

    def calculate_rolling_mean_time_last_3_problems_per_user(self) -> bool:
        """
        이동 평균(Rolling Mean)을 사용해 현재 푸는 문제를 포함해서
        최근 3개 문제의 평균 풀이 시간을 feature로 추가해보자.

        최근 문제를 얼마나 빨리 푸느냐에 따라 학생의 최근 컨디션을 추측해볼 수 있을지도 모른다.

        Returns:
            bool: 성공여부
        """
        try:
            self.df["mean_time"] = (
                self.df.groupby(["userID"])["time"].rolling(3).mean().values
            )
        except Exception:
            return False
        return True

    def calculate_mean_and_stddev_per_user(self) -> bool:
        """
        데이터셋의 모든 수치형 feature의 유저별
        평균(mean)과 표준편차(standard deviation)을 새로운 feature로 추가해보자.

        Returns:
            bool: 성공여부
        """
        try:
            # 평균 (mean) / 표준 편차 (std)
            agg_df = self.df.groupby("userID").agg(["mean", "std"])

            # mapping을 위해 pandas DataFrame을 dictionary형태로 변환
            agg_dict = agg_df.to_dict()

            # 구한 통계량을 각 사용자에게 mapping
            for k, v in agg_dict.items():
                # feature 이름
                feature_name = "_".join(k)

                # mapping이후 새로운 feature 추가
                self.df[feature_name] = self.df["userID"].map(v)
        except Exception:
            return False
        return True

    def calculate_median_time_per_user(self) -> bool:
        """
        유저별 문제 풀이에 사용한 시간의 중간값을 feature로 추가해보자.

        이 경우 각 문제를 풀 때 상대적으로 빨리 풀었는지 늦게 풀었는지 알 수 있다.
        상대적인 풀이 시간 비교를 통해 해당 사용자가 문제를 쉽게 느끼는지 어렵게 느끼는지 추론해 볼 수 있을지도 모른다.

        Returns:
            bool: 성공여부
        """
        try:
            # 중간값 (median)
            agg_df = self.df.groupby("userID")["time"].agg(["median"])

            # mapping을 위해 pandas DataFrame을 dictionary형태로 변환
            agg_dict = agg_df.to_dict()

            # 구한 통계량을 각 사용자에게 mapping
            self.df["time_median"] = self.df["userID"].map(agg_dict["median"])

        except Exception:
            return False
        return True

    def calculate_problem_solving_time_per_user(self) -> bool:
        """
        유저가 문제를 몇 시에 푸는지를 feature로 추가해보자.
        여기서 더 응용한다면 특정 시간대의 문제 정답률을 확인해봄으로서 특정 시간대의 사용자들의 능률을 알 수 있을지도 모른다.

        Returns:
            bool: 성공여부
        """
        try:
            # custom 함수 적용
            self.df["hour"] = self.df["Timestamp"].transform(
                lambda x: pd.to_datetime(x, unit="s").dt.hour
            )

        except Exception:
            return False
        return True

    def calculate_accuracy_by_time_of_day(self) -> bool:
        """
        유저 상관 없이 시간대별 정답률을 feature로 추가해보자.
        특정 시간대의 정답률에서 패턴을 발견할 수 있을지도 모른다.

        Returns:
            bool: 성공여부
        """
        try:
            # 문제를 푸는 시간대
            self.df["hour_temp"] = self.df["Timestamp"].transform(
                lambda x: pd.to_datetime(x, unit="s").dt.hour
            )
            # 시간대별 정답률
            hour_dict = (
                self.df.groupby(["hour_temp"])["answerCode"].mean().to_dict()
            )
            self.df["correct_per_hour"] = self.df["hour"].map(hour_dict)

            self.df.drop("hour_temp", axis=1, inplace=True)
        except Exception:
            return False
        return True

    def calculate_user_activity_time_preference(self) -> bool:
        """
        사용자가 밤에 주로 활동하는 사람인지 낮에 주로 활동하는 사람인지 여부를 feature로 추가해보자.
        이를 통해 사용자가 문제를 풀 때 어떤 시간대인지에 따라 효율을 추정해볼 수 있을지도 모른다.

        Returns:
            bool: 성공여부
        """
        try:
            # 문제를 푸는 시간대
            self.df["hour_temp"] = self.df["Timestamp"].transform(
                lambda x: pd.to_datetime(x, unit="s").dt.hour
            )
            # 사용자의 주 활동 시간
            mode_dict = (
                self.df.groupby(["userID"])["hour_temp"]
                .agg(lambda x: pd.Series.mode(x)[0])
                .to_dict()
            )
            self.df["hour_mode"] = self.df["userID"].map(mode_dict)

            # 사용자의 야행성 여부
            # 시간이 10 ~ 15시 사이에 분포되어 있어 여기에서는 임의로 12로 분리하였다
            self.df["is_night"] = self.df["hour_mode"] > 12

            self.df.drop("hour_temp", axis=1, inplace=True)
            self.df.drop("hour_mode", axis=1, inplace=True)
        except Exception:
            return False
        return True

    def calculate_normalized_time_per_user(self) -> bool:
        """
        문제 풀이에 사용한 시간을 정규화하여 feature로 추가해보자.

        Returns:
            bool: 성공여부
        """
        try:
            # custom 함수 적용
            # time만 transform하였다
            self.df["normalized_time"] = self.df.groupby("userID")[
                "time"
            ].transform(lambda x: (x - x.mean()) / x.std())
        except Exception:
            return False
        return True

    def calculate_relative_time_spent_per_user(self) -> bool:
        """
        문제 풀이에 사용한 시간과 이를 바탕으로 구한 중간값과의 차이를 통해
        문제 풀이에 시간을 얼마나 사용하였는지를 상대적으로 비교하는 feature를 추가해보자.

        Returns:
            bool: 성공여부
        """
        try:
            # custom 함수 적용
            # apply를 사용해 time column을 직접 지정할 수 있다
            self.df["relative_time"] = (
                self.df.groupby("userID")
                .apply(lambda x: x["time"] - x["time"].median())
                .values
            )
        except Exception:
            return False
        return True

    def calculate_time_cut_column(self, bins: int = 3) -> bool:
        """
        각 문제 풀이에 사용한 시간들을 최대-최소 값을 활용해 bins개의 구간으로 나눠 할당

        [Args]
        - bins : int (default=3)
            - 나눌 구간 수

        [FEATS]
        - time_cut : 구간 값
        - time_cut_enc : 구간 값을 ordered encoding 한 값

        [Returns]
            bool : 성공 여부
        """
        try:
            df = self.df.copy()

            if "time" not in df.columns:
                self.calculate_solve_time_column()
                df = self.df.copy()
            df["time_cut"] = pd.cut(df["time"], bins=bins).astype("str")

            cut2idx = {v: i for i, v in enumerate(df["time_cut"].unique())}
            df["time_cut_enc"] = df["time_cut"].map(cut2idx).astype("str")
            self.df = df.drop(["time_cut"], axis=1)
        except Exception:
            return False
        return True

    def calculate_items_svd_latent(self, n_component: int = 3) -> bool:
        """
        유저가 각 문제를 맞춘 횟수를 활용하여 Truncated SVD 적용, 각 문제들의 잠재 요인을 구해 feature로 추가

        [Args]
        - n_component : int (default = 3)
            - 얻어낼 잠재요인의 수

        [FEAT]
        - 각 문제의 잠재요인을 표현한 n_component 개의 column 추가

        [Returns]
        - bool : 성공 여부

        """

        try:
            edu_svd_df = self.df.copy()

            edu_correct_df = (
                edu_svd_df.groupby(["userID", "assessmentItemID"])
                .answerCode.sum()
                .reset_index()
            )
            edu_pivot_df = edu_correct_df.pivot(
                index="userID", columns="assessmentItemID", values="answerCode"
            ).fillna(0)

            svd = TruncatedSVD(n_components=n_component)

            # 문제의 잠재 요인 행렬
            svd.fit(edu_pivot_df.T)

            index = edu_pivot_df.columns.values
            columns = [f"items_svd_{i}" for i in range(n_component)]
            svd_df = pd.DataFrame(
                svd.transform(edu_pivot_df.T), index=index, columns=columns
            ).reset_index()
            self.df = (
                edu_svd_df.merge(
                    right=svd_df, left_on="assessmentItemID", right_on="index"
                )
                .sort_values(by=["userID", "Timestamp"])
                .reset_index(drop=True)
            )
        except Exception:
            return False
        return True

    def calculate_times_nmf_latent(self, n_component: int = 3) -> bool:
        """
        유저가 각 문제룰 푼 시간을 활용하여 NMF 적용, 각 문제들의 잠재 요인을 구해 featrue로 추가

        [Args]
            - n_component : int (default = 3)
            - 얻어낼 잠재요인의 수

        [FEAT]
            - 각 문제 풀이 시간의 잠재요인을 표현한 n_component 개의 column 추가

        [Returns]
            - bool : 성공 여부

        """
        try:
            edu_nmf_df = self.df.copy()

            if "time" not in edu_nmf_df.columns:
                self.calculate_solve_time_column()
                edu_nmf_df = self.df.copy()

            edu_time_df = (
                edu_nmf_df.groupby(["userID", "assessmentItemID"])
                .time.sum()
                .reset_index()
            )
            edu_pivot_df = edu_time_df.pivot(
                index="userID", columns="assessmentItemID", values="time"
            ).fillna(0)

            nmf = NMF(n_components=n_component)
            nmf.fit(edu_pivot_df.T)
            index = edu_pivot_df.columns.values
            columns = [f"times_nmf_{i}" for i in range(n_component)]

            nmf_df = pd.DataFrame(
                nmf.transform(edu_pivot_df.T), index=index, columns=columns
            ).reset_index()
            self.df = (
                edu_nmf_df.merge(
                    right=nmf_df, left_on="assessmentItemID", right_on="index"
                )
                .sort_values(by=["userID", "Timestamp"])
                .reset_index(drop=True)
            )
        except Exception:
            return False
        return True

    def calculate_users_pca_latent(self, n_component: int = 3) -> bool:
        """
        유저의 각 문제 별 정답횟수를 활용하여 PCA 적용, 각 유저의 문제에 대한 잠재 요인을 구해 featrue로 추가

        [Args]
            - n_component : int (default = 3)
            - 얻어낼 잠재요인의 수

        [FEAT]
            - 각 문제에 대한 유저의 잠재 요인을 구해 feature로 추가

        [Returns]
            - bool : 성공 여부
        """

        try:
            edu_pca_df = self.df.copy()

            # 유저들에 대한 문제별 정답 횟수 행렬
            edu_pca_df = edu_pca_df.groupby(["userID", "assessmentItemID"])[
                "answerCode"
            ].sum()
            edu_pca_df = edu_pca_df.astype(np.uint8)
            edu_pca_df = edu_pca_df.unstack(fill_value=0)

            pca = PCA(n_components=n_component)

            # 유저의 문제별 정답 횟수에 대한 차원 축소
            pca.fit(edu_pca_df)
            columns = [f"user_pca_{i}" for i in range(n_component)]
            index = edu_pca_df.index.values

            user_pca_df = pd.DataFrame(
                pca.transform(edu_pca_df), index=index, columns=columns
            ).reset_index()
            self.df = (
                self.df.merge(
                    right=user_pca_df, left_on="userID", right_on="index"
                )
                .sort_values(by=["userID", "Timestamp"])
                .reset_index(drop=True)
                .drop(["index"], axis=1)
            )
        except Exception:
            return False
        return True

    def calculate_items_pca_latent(self, n_component: int = 3) -> bool:
        """
        유저의 각 문제 별 정답횟수를 활용하여 PCA 적용, 각 문제의 유저에 대한 잠재 요인을 구해 featrue로 추가

        [Args]
            - n_component : int (default = 3)
            - 얻어낼 잠재요인의 수

        [FEAT]
            - 각 유저에 대한 문제의 잠재 요인을 구해 feature로 추가

        [Returns]
            - bool : 성공 여부
        """
        try:
            edu_pca_df = self.df.copy()
            # 유저들에 대한 문제별 정답 횟수 행렬
            edu_pca_df = edu_pca_df.groupby(["userID", "assessmentItemID"])[
                "answerCode"
            ].sum()
            edu_pca_df = edu_pca_df.astype(np.uint8)
            edu_pca_df = edu_pca_df.unstack(fill_value=0)

            pca = PCA(n_components=n_component)
            # 문제의 유저별 정답 횟수에 대한 차원 축소
            pca.fit(edu_pca_df.T)

            columns = [f"item_pca_{i}" for i in range(n_component)]
            index = edu_pca_df.columns.values
            item_pca_df = pd.DataFrame(
                pca.transform(edu_pca_df.T), index=index, columns=columns
            ).reset_index()
            self.df = (
                self.df.merge(
                    right=item_pca_df,
                    left_on="assessmentItemID",
                    right_on="index",
                )
                .sort_values(by=["userID", "Timestamp"])
                .reset_index(drop=True)
                .drop(["index"], axis=1)
            )

        except Exception:
            return False
        return True

    def calculate_times_pca_latent(self, n_component: int = 2) -> bool:
        """
        시간과 연관된 feature들의 PCA, 2개의 feature만 이용했지만 feature가 많을 때 효과적이다.

        [Args]
            - n_component : int (default = 3)
            - 얻어낼 잠재요인의 수

        [FEAT]
            - 시간과 연관된 feature들의 잠재 요인을 구해 feature로 추가

        [Returns]
            - bool : 성공 여부
        """
        try:
            edu_time_df = self.df.copy()

            # 유저별 문제를 푼 시간대를 feature로 추가

            if "time" not in edu_time_df.columns:
                self.calculate_solve_time_column()
            if "hour" not in edu_time_df.columns:
                edu_time_df["hour"] = edu_time_df["Timestamp"].transform(
                    lambda x: pd.to_datetime(x, unit="s").dt.hour
                )

            # 시간 관련 feature
            X = edu_time_df[["time", "hour"]]
            pca = PCA(n_components=n_component)
            columns = [f"time_pca_{i}" for i in range(n_component)]
            edu_time_df[columns] = pca.fit_transform(X)

            self.df = edu_time_df
        except Exception:
            return False
        return True

    def calculate_times_lda_latent(self):
        """
        시간과 연관된 feature들의 LDA

        - PCA가 전체 데이터 셋의 분산이 최대가 되는 것을 목적으로 한다면, LDA는 각 클래스로 데이터셋을 나눠 분산을 고려한다.
        - 각 클래스 내 분산은 최소가 되고, 클래스 간의 거리가 최대가 되도록 고차원 데이터를 저차원 데이터로 변환한다.

        [FEAT]
            - 시간과 연관된 feature들의 잠재 요인을 구해 feature로 추가
            - 현재 데이터 셋에서 n_component를 1개 보다 크게 지정하면 에러가 발생한다.

        [Returns]
            - bool : 성공 여부
        """

        try:
            edu_time_df = self.df.copy()
            # 유저별 문제를 푼 시간대를 feature로 추가
            if "time" not in edu_time_df.columns:
                self.calculate_solve_time_column()
            if "hour" not in edu_time_df.columns:
                edu_time_df["hour"] = edu_time_df["Timestamp"].transform(
                    lambda x: pd.to_datetime(x, unit="s").dt.hour
                )

            # 시간 관련 feature
            X = edu_time_df[["time", "hour"]]
            y = edu_time_df[["answerCode"]]

            lda = LDA(n_components=1)
            edu_time_df["time_lda"] = lda.fit_transform(X, y)
            self.df = edu_time_df
        except Exception:
            return False
        return True

    def caculate_item_latent_dirichlet_allocation(
        self, n_component: int = 2
    ) -> bool:
        """(50초 정도 걸림)
        - 토픽 모델링(Topic Modeling)에 속하는 비지도 학습 모델로 문서를 자신이 지정한 K개의 주제(Topic) 중 하나로 분류하기 위해 사용된다.
        - 여기서는 각 고차원 데이터의 K개의 주제(Topic) 비율을 알아내어 이를 저차원 데이터로 사용한다.
            - 문서 : 문제
            - 단어 : 유저
            - 단어 출현 횟수 : 정답 횟수

        [Args]
            - n_component : int (default = 2)
            - 얻어낼 잠재요인의 수

        [FEAT]
            - 각 유저에 대한 문제의 lda를 구해 feature로 추가

        [Returns]
            - bool : 성공 여부
        """

        try:
            edu_lda_df = self.df.copy()

            ### 문제들에 대한 유저별 정답 횟수 행렬 만들기
            edu_lda_df = edu_lda_df.groupby(["assessmentItemID", "userID"])[
                "answerCode"
            ].sum()
            edu_lda_df = edu_lda_df.astype(np.uint8)
            edu_lda_df = edu_lda_df.unstack(fill_value=0)

            # 유저에 대한 각 문제의 lda
            lda = LatentDirichletAllocation(n_components=2)
            lda.fit(edu_lda_df)

            index = edu_lda_df.index.values
            columns = [f"item_lda_{i}" for i in range(n_component)]
            lda_df = pd.DataFrame(
                lda.transform(edu_lda_df), index=index, columns=columns
            ).reset_index()
            self.df = (
                self.df.merge(
                    right=lda_df, left_on="assessmentItemID", right_on="index"
                )
                .sort_values(by=["userID", "Timestamp"])
                .reset_index(drop=True)
            )
            if "index" in self.df.columns.values:
                self.df.drop(["index"], axis=1, inplace=True)
        except Exception:
            return False
        return True

    def caculate_user_latent_dirichlet_allocation(
        self, n_component: int = 2
    ) -> bool:
        """(50초 정도 걸림)
        - 토픽 모델링(Topic Modeling)에 속하는 비지도 학습 모델로 문서를 자신이 지정한 K개의 주제(Topic) 중 하나로 분류하기 위해 사용된다.
        - 여기서는 각 고차원 데이터의 K개의 주제(Topic) 비율을 알아내어 이를 저차원 데이터로 사용한다.
            - 문서 : 문제
            - 단어 : 유저
            - 단어 출현 횟수 : 정답 횟수

        [Args]
            - n_component : int (default = 2)
            - 얻어낼 잠재요인의 수

        [FEAT]
            - 각 문제에 대한 유저의 lda를 구해 feature로 추가

        [Returns]
            - bool : 성공 여부
        """
        try:
            edu_lda_df = self.df.copy()

            ### 문제들에 대한 유저별 정답 횟수 행렬 만들기
            edu_lda_df = edu_lda_df.groupby(["assessmentItemID", "userID"])[
                "answerCode"
            ].sum()
            edu_lda_df = edu_lda_df.astype(np.uint8)
            edu_lda_df = edu_lda_df.unstack(fill_value=0)

            # 문제에 대한 각 유저의 lda
            lda = LatentDirichletAllocation(n_components=2)
            lda.fit(edu_lda_df.T)
            index = edu_lda_df.columns.values
            columns = [f"user_lda_{i}" for i in range(n_component)]
            lda_df = pd.DataFrame(
                lda.transform(edu_lda_df.T), index=index, columns=columns
            ).reset_index()

            if "index_x" in self.df.columns.values:
                self.df.drop(["index_x"], axis=1, inplace=True)
            self.df = (
                self.df.merge(right=lda_df, left_on="userID", right_on="index")
                .sort_values(by=["userID", "Timestamp"])
                .reset_index(drop=True)
            )

            if "index" in self.df.columns.values:
                self.df.drop(["index"], axis=1, inplace=True)
        except Exception:
            return False
        return True
