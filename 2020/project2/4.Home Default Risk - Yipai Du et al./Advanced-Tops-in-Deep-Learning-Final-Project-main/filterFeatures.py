import time

import featuretools as ft
import numpy as np
import pandas as pd

if __name__ == "__main__":
    feature_matrix = pd.read_csv('data/feature_matrix.csv',
                                 low_memory=False,
                                 memory_map=True)
    train = feature_matrix[feature_matrix['set'] == 'train']
    test = feature_matrix[feature_matrix['set'] == 'test']

    train = train.drop(
        columns=['MODE(bureau.CREDIT_ACTIVE)', 'MODE(bureau.CREDIT_TYPE)']
    )  # the number of categories are different for these two features in training and testing set, thus removed
    print("Initial removed two features")
    start_time = time.time()
    filtered_train = ft.selection.remove_low_information_features(train)
    print("Before ", train.shape, " after ", filtered_train.shape)
    print("Removed ", train.shape[1] - filtered_train.shape[1])
    print("Used " + str(time.time() - start_time) + "s")

    start_time = time.time()
    train = filtered_train
    filtered_train = ft.selection.remove_single_value_features(train)
    filtered_train = ft.selection.remove_highly_null_features(filtered_train)
    filtered_train = ft.selection.remove_highly_correlated_features(
        filtered_train)
    print("Before ", train.shape, " after ", filtered_train.shape)
    print("Removed ", train.shape[1] - filtered_train.shape[1])
    print("Used " + str(time.time() - start_time) + "s")

    filtered_train.to_csv('data/filtered_train_feature_matrix.csv',
                          index=False,
                          header=True)
