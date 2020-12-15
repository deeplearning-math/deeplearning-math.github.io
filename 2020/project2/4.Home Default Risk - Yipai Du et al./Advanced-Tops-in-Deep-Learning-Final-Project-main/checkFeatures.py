import featuretools as ft
import pandas as pd
import numpy as np

if __name__ == "__main__":
    X_train = np.load('data/train_X.npy')
    # X_test = pd.read_csv("data/test.csv")
    # X_test = X_test.drop(columns=['SK_ID_CURR', 'TARGET'])
    # print(X_test.columns[864])
    # print(X_test.columns[978])
    # X_train = np.concatenate([X_train, X_test], axis=0)
    X_train = pd.DataFrame(X_train)
    # X_train = pd.read_csv('data/train.csv', memory_map=True, low_memory=False)
    # X_test = pd.read_csv("data/test.csv", memory_map=True, low_memory=False)
    # X_train = X_train.append(X_test, ignore_index=True)
    filtered_X_train = ft.selection.remove_low_information_features(X_train)
    print("Difference: ", X_train.shape[1] - filtered_X_train.shape[1])
    for c in X_train.columns:
        if c not in filtered_X_train.columns:
            print(c, " is not in filtered frame")