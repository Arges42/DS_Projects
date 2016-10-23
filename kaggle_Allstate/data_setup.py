import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, boxcox
from math import exp, log
import numpy as np
from datetime import datetime
from os.path import join

def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

#Load the data and do some preprocessing
def load_data(path_train='data/train.csv', path_test='data/test.csv'):
    train_loader = pd.read_csv(path_train, dtype={'id': np.int32})
    train = train_loader.drop(['id', 'loss'], axis=1)
    test_loader = pd.read_csv(path_test, dtype={'id': np.int32})
    test = test_loader.drop(['id'], axis=1)
    ntrain = train.shape[0]
    ntest = test.shape[0]
    train_test = pd.concat((train, test)).reset_index(drop=True)
    numeric_feats = train_test.dtypes[train_test.dtypes != "object"].index

    # compute skew and do Box-Cox transformation
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    print("\nSkew in numeric features:")
    print(skewed_feats)
    # transform features with skew > 0.25 (this can be varied to find optimal value)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index
    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    features = train.columns
    cats = [feat for feat in features if 'cat' in feat]
    # factorize categorical features
    for feat in cats:
        train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]
    x_train = train_test.iloc[:ntrain, :]
    x_test = train_test.iloc[ntrain:, :]
    train_test_scaled, scaler = scale_data(train_test)
    train, _ = scale_data(x_train, scaler)
    test, _ = scale_data(x_test, scaler)

    train_labels = np.log(np.array(train_loader['loss']))
    train_ids = train_loader['id'].values.astype(np.int32)
    test_ids = test_loader['id'].values.astype(np.int32)

    return train, train_labels, test, train_ids, test_ids
