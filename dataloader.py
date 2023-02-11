import numpy as np
import pandas as pd


def loader(dataset="SMAP"):
    if dataset == "SMAP":
        return _SMAP_loader()
    elif dataset == "UCR":
        return _UCR_loader()


def _SMAP_loader():
    np_train = np.load("./datasets/SMAP/SMAP_train.npy")
    np_test = np.load("./datasets/SMAP/SMAP_test.npy")
    np_test_label = np.load("./datasets/SMAP/SMAP_test_label.npy")

    df_train = pd.DataFrame(data=np_train, columns=[f"feature_{i+1}" for i in range(np_train.shape[1])], dtype="float32")
    df_test = pd.DataFrame(data=np_test, columns=[f"feature_{i+1}" for i in range(np_test.shape[1])], dtype="float32")
    df_test_label = pd.DataFrame(data=np_test_label, columns=["label"], dtype="float32")

    return df_train, df_test, df_test_label


def _UCR_loader():
    csv_train = pd.read_csv("./datasets/UCR/train/001_UCR_Anomaly_train.csv", dtype="float32")
    csv_test = pd.read_csv("./datasets/UCR/test/001_UCR_Anomaly_test.csv", dtype="float32")
    df_train = csv_train.drop(columns=["label", "timestamp"])
    df_test_label = pd.DataFrame(data=csv_test["label"], dtype="int")
    df_test = csv_test.drop(columns=["label", "timestamp"])

    return df_train, df_test, df_test_label