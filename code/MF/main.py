import os
import argparse
import random
from pprint import pprint


import numpy as np
import pandas as pd

from MF_util.dataset import train_test_eval_split
from MF_util.models import _SVD, _NMF
from MF_util.args import parse_args
from MF_util.utils import Setting

setting = Setting()


def main(args: argparse.Namespace):
    random.seed(42)
    np.random.seed(42)

    print("--------------- Data Load ---------------")
    train_data_path = os.path.join(args.data_dir, "train_data.csv")
    test_data_path = os.path.join(args.data_dir, "test_data.csv")
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    print("--------------- Data Split ---------------")
    train, test, eval = train_test_eval_split(train_data, test_data)
    print("--------------- Model Load   ---------------")
    # mySVD = _SVD(train, test, eval)
    myNMF = _NMF(train, test, eval)
    print("--------------- Predict & Evaluate ---------------")
    # pred, score = mySVD.evaluate()
    pred, score = myNMF.evaluate()
    print("--------------- Save Predict ---------------")
    filename = setting.get_submit_filename(args, score)
    setting.save_predict(filename=filename, predict=pred)


if __name__ == "__main__":
    args = parse_args()
    main(args)
