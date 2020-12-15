import gc
import time
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
import scipy
from sklearn.impute import SimpleImputer
import featuretools as ft
if __name__ == "__main__":
    feature_matrix = pd.read_csv('data/kaggle_notebook/feature_matrix.csv',
                                 low_memory=False,
                                 memory_map=True)
    train = pd.read_csv('data/filtered_train_feature_matrix.csv',
                        low_memory=False,
                        memory_map=True)
    train['set'] = 'train'
    test = feature_matrix[feature_matrix['set'] == 'test']
    print("train shape, test shape ", train.shape, test.shape)
    train, test = train.align(test, join='inner', axis=1)
    app = train.append(test, ignore_index=True)
    print("app shape: ", app.shape)
    gc.enable()
    del feature_matrix, train, test
    gc.collect()
    start_time = time.time()

    app_array = app.to_numpy(copy=True)
    imputer = SimpleImputer(verbose=1, copy=False, strategy='most_frequent')
    app_array = imputer.fit_transform(app_array)
    # app_array = np.load("imputed.npy", allow_pickle=True)
    app = pd.DataFrame(app_array, index=app.index,
                       columns=app.columns).astype(app.dtypes.to_dict())
    print(app.dtypes)
    print("Imputing takes: ", time.time() - start_time)

    print("App contain nan is: ", str(app.isnull().values.any()))
    print("App shape: ", app.shape)

    train = app[app['set'] == 'train'].drop(columns=['set', 'SK_ID_CURR'])
    test = app[app['set'] == 'test'].drop(
        columns=['set', 'SK_ID_CURR', 'TARGET'])
    test_id = app[app['set'] == 'test']['SK_ID_CURR']
    test.to_csv('data/test.csv', index=False, header=True)
    test_id.to_csv('data/test_id.csv', index=False, header=True)
    train.to_csv('data/train.csv', index=False, header=True)

    train_y = app[app['set'] == 'train']['TARGET'].to_numpy()
    np.save("data/train_y.npy", train_y)

    app = app.drop(columns=['SK_ID_CURR', 'TARGET'])
    app = pd.get_dummies(app)

    train_X = app[app['set_train'] == 1].drop(
        columns=['set_train', 'set_test'])
    np.save("data/feature_columns.npy", np.array(train_X.columns))
    train_X = train_X.to_numpy()
    test_X = app[app['set_test'] == 1].drop(
        columns=['set_train', 'set_test']).to_numpy()
    print("Train X and y shape: ", train_X.shape, train_y.shape)
    np.save("data/train_X.npy", train_X)
    np.save("data/test_X.npy", test_X)