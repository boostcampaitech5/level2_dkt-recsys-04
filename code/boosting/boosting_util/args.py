import argparse

# answerCode 사용하지 않는 FE
# FEATS = [
#     # "calculate_cumulative_stats_by_time",
#     "calculate_overall_accuracy_by_testID",
#     "calculate_overall_accuracy_by_KnowledgeTag",
#     # # 시간 칼럼을 사용하는 FE
#     "calculate_solve_time_column",  # Time 관련 Feature Engineering할 때 필수!
#     # "check_answer_at_time",
#     "calculate_total_time_per_user",
#     # "calculate_past_correct_answers_per_user",
#     # "calculate_future_correct_answers_per_user",
#     # "calculate_past_correct_attempts_per_user",
#     "calculate_past_solved_problems_per_user",
#     # "calculate_past_average_accuracy_per_user",
#     # "calculate_past_average_accuracy_current_problem_per_user",
#     "calculate_rolling_mean_time_last_3_problems_per_user",
#     # "calculate_mean_and_stddev_per_user", # 오류가 많아서 스킵
#     # "calculate_median_time_per_user",
#     "calculate_problem_solving_time_per_user",
#     # "calculate_accuracy_by_time_of_day",
#     # "calculate_user_activity_time_preference",
#     "calculate_normalized_time_per_user",
#     "calculate_relative_time_spent_per_user",
#     "calculate_time_cut_column",
#     # "calculate_items_svd_latent",
#     # "calculate_times_nmf_latent",
#     # "calculate_users_pca_latent",
#     # "calculate_items_pca_latent",
#     # "calculate_times_pca_latent",
#     # "calculate_times_lda_latent",
#     # "caculate_item_latent_dirichlet_allocation",  # 50초 걸림
#     # "caculate_user_latent_dirichlet_allocation",  # 50초 걸림
# ]
FEATS = [
    "calculate_cumulative_stats_by_time",
    "calculate_overall_accuracy_by_testID",
    "calculate_overall_accuracy_by_KnowledgeTag",
    # # 시간 칼럼을 사용하는 FE
    "calculate_solve_time_column",  # Time 관련 Feature Engineering할 때 필수!
    "check_answer_at_time",
    "calculate_total_time_per_user",
    "calculate_past_correct_answers_per_user",
    "calculate_future_correct_answers_per_user",
    "calculate_past_correct_attempts_per_user",
    "calculate_past_solved_problems_per_user",
    "calculate_past_average_accuracy_per_user",
    "calculate_past_average_accuracy_current_problem_per_user",
    "calculate_rolling_mean_time_last_3_problems_per_user",
    # "calculate_mean_and_stddev_per_user",  # 오류가 많아서 스킵
    "calculate_median_time_per_user",
    "calculate_problem_solving_time_per_user",
    "calculate_accuracy_by_time_of_day",
    "calculate_user_activity_time_preference",
    "calculate_normalized_time_per_user",
    "calculate_relative_time_spent_per_user",
    "calculate_time_cut_column",
    # "calculate_items_svd_latent",
    # "calculate_times_nmf_latent",
    # "calculate_users_pca_latent",
    # "calculate_items_pca_latent",
    # "calculate_times_pca_latent",
    # "calculate_times_lda_latent",
    # "caculate_item_latent_dirichlet_allocation",  # 50초 걸림
    # "caculate_user_latent_dirichlet_allocation",  # 50초 걸림
]


def train_parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "CatBoost",
            "LGBM",
        ],
        help="학습 및 예측할 모델을 선택할 수 있습니다.",
    )
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument(
        "--use_cuda_if_available", default=True, type=bool, help="Use GPU"
    )

    parser.add_argument(
        "--data_dir", default="/opt/ml/input/data", type=str, help=""
    )
    parser.add_argument(
        "--model_dir", default="./models/", type=str, help="model dir"
    )
    parser.add_argument(
        "--user_id_dir",
        default=None,
        type=str,
        help="OOF용 user_id가 작성된 user_id_dir",
    )
    parser.add_argument(
        "--output_dir", default="./outputs/", type=str, help="output dir"
    )
    parser.add_argument(
        "--num_iterations", default=1000, type=int, help="num_iterations"
    )
    parser.add_argument(
        "--num_boost_round", default=100, type=int, help="num_boost_round"
    )
    parser.add_argument("--num_leaves", default=31, type=int, help="num_leaves")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--depth", default=6, type=int, help="depth")
    parser.add_argument(
        "--feats",
        default=FEATS,
        type=list,
        help="feats",
    )

    args = parser.parse_args()

    return args


def cv_parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["CatBoost", "LGBM", "XGBoost"],
        help="학습 및 예측할 모델을 선택할 수 있습니다.",
    )
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument(
        "--use_cuda_if_available", default=True, type=bool, help="Use GPU"
    )

    parser.add_argument(
        "--data_dir", default="/opt/ml/input/data", type=str, help=""
    )
    parser.add_argument(
        "--model_dir", default="./models/", type=str, help="model dir"
    )
    parser.add_argument(
        "--output_dir", default="./outputs/", type=str, help="output dir"
    )

    parser.add_argument(
        "--user_id_dir",
        default=None,
        type=str,
        help="OOF용 user_id가 작성된 user_id_dir",
    )

    parser.add_argument(
        "--num_iterations", default=1000, type=int, help="num_iterations"
    )
    parser.add_argument(
        "--num_boost_round", default=1000, type=int, help="num_boost_round"
    )
    parser.add_argument(
        "--use_kfold", default=True, type=bool, help="Use K-Fold CV"
    )
    parser.add_argument(
        "--use_skfold",
        default=False,
        type=bool,
        help="Use Stratified K-Fold CV",
    )
    parser.add_argument(
        "--use_tscv", default=False, type=bool, help="Use Time Series CV"
    )
    parser.add_argument(
        "--use_btscv",
        default=False,
        type=bool,
        help="Use Blocking Time Series CV",
    )
    parser.add_argument(
        "--user_based_kfold",
        default=False,
        type=bool,
        help="User User Based K-Fold Crossvalidation Strategy",
    )
    parser.add_argument(
        "--n_splits", default=10, type=int, help="n_splits in Cross-Validation"
    )
    parser.add_argument("--num_leaves", default=31, type=int, help="num_leaves")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--depth", default=6, type=int, help="depth")
    parser.add_argument(
        "--feats",
        default=FEATS,
        type=list,
        help="feats",
    )

    args = parser.parse_args()

    return args


def inference_parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["CatBoost", "LGBM", "XGBoost"],
        help="학습 및 예측할 모델을 선택할 수 있습니다.",
    )
    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument(
        "--data_dir", default="/opt/ml/input/data", type=str, help=""
    )
    parser.add_argument(
        "--output_dir", default="./outputs/", type=str, help="output dir"
    )
    parser.add_argument(
        "--model_dir", default="./models/", type=str, help="model dir"
    )
    parser.add_argument(
        "--model_name", default="catboost.cbm", type=str, help="output dir"
    )
    parser.add_argument(
        "--feats",
        default=FEATS,
        type=list,
        help="feats",
    )
    args = parser.parse_args()

    return args
