import torch
import easydict
import os
from .config import CONFIG
from datetime import datetime

configuration = CONFIG()


def load_args(configuration=configuration):
    config = {}

    # 설정
    config["seed"] = configuration.seed
    config["device"] = configuration.device
    config["data_dir"] = configuration.data_dir
    config["save_dir"] = configuration.save_dir
    config["model_dir"] = os.path.join(configuration.save_dir, datetime.now().strftime(r"%m%d_%H%M%S"))

    # 데이터
    config["max_seq_len"] = configuration.max_seq_len

    # 데이터 증강 (Data Augmentation)
    config["window"] = configuration.window
    config["stride"] = configuration.stride
    config["shuffle"] = configuration.shuffle
    config["shuffle_n"] = configuration.shuffle_n

    # LabelEncoder
    config["n_questions"] = configuration.n_questions
    config["n_test"] = configuration.n_test
    config["n_tag"] = configuration.n_tag
    config["n_elapsed_question"] = configuration.n_elapsed_question

    # 모델
    config["hidden_dim"] = configuration.hidden_dim
    config["n_layers"] = configuration.n_layers
    config["dropout"] = configuration.dropout
    config["n_heads"] = configuration.n_heads

    # T Fixup
    config["Tfixup"] = configuration.Tfixup
    config["layer_norm"] = configuration.layer_norm

    # 훈련
    config["n_epochs"] = configuration.n_epochs
    config["batch_size"] = configuration.batch_size
    config["lr"] = configuration.lr
    config["clip_grad"] = configuration.clip_grad

    ### 중요 ###
    config["model"] = configuration.model
    config["optimizer"] = configuration.optimizer
    config["scheduler"] = configuration.scheduler

    args = easydict.EasyDict(config)

    return args
