import pandas as pd
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.stats import skew, boxcox
from math import exp, log
import numpy as np
from datetime import datetime
from os.path import join
import matplotlib.pyplot as plt
import itertools

def scale_data(X, scaler=None):
    #cont = [cont for cont in X.columns if "cont" in cont]
    #X_tmp = X[cont]
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    #X_tmp = np.hstack((X_tmp,X.drop(cont,1).as_matrix()))
    return X, scaler

def feature_generation(train_loader):
    
    

    return train_loader

#Load the data and do some preprocessing
def load_data(path_train='data/train.csv', path_test='data/test.csv'):
    train_loader = pd.read_csv(path_train, dtype={'id': np.int32})
    #train_loader = continuous_feature_processing(train_loader)
    train = train_loader.drop(['id', 'loss'], axis=1)
    test_loader = pd.read_csv(path_test, dtype={'id': np.int32})
    #test_loader = continuous_feature_processing(test_loader)
    test = test_loader.drop(['id'], axis=1)
    ntrain = train.shape[0]
    ntest = test.shape[0]
    train_test = pd.concat((train, test)).reset_index(drop=True)
    numeric_feats = train_test.dtypes[train_test.dtypes != "object"].index
    #numeric_feats = ["cont"+str(x) for x in range(1,15)]

    
    # compute skew and do Box-Cox transformation
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    print("\nSkew in numeric features:")
    #skewed_feats = skewed_feats.drop(["lin_cont1*lin_cont12","lin_cont2*lin_cont12","lin_cont3*lin_cont12","lin_cont4*lin_cont12",
#"lin_cont5*lin_cont12","lin_cont6*lin_cont12","lin_cont7*lin_cont12",
#"lin_cont10*lin_cont12","lin_cont11*lin_cont12","lin_cont13*lin_cont12"])
    print(skewed_feats)
    # transform features with skew > 0.25 (this can be varied to find optimal value)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index
    for feats in skewed_feats:
        #if np.min(train_test[feats])<0:
         #   shift = np.min(train_test[feats])+1
       # else:
        shift = 1
        train_test[feats] = train_test[feats] + shift
        train_test[feats], lam = boxcox(train_test[feats])
       
    #train_test = continuous_feature_processing(train_test)
    train_test = categorical_feature_processing(train_test)
    

    #cats = [feat for feat in features if 'cat' in feat]
    cats = ["cat"+str(x) for x in range(1,117)]
    # factorize categorical features
    for feat in cats:
        train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]
        train_test[feat] = train_test[feat].astype('int32')
   # train_test.lin_cont2 = pd.factorize(train_test.lin_cont2, sort=True)[0]
    #cats = ["cat"+str(x) for x in [112]]
    #for feat in cats:
    #    tmp = pd.get_dummies(train_test[feat],prefix=feat,drop_first=True)
    #    tmp = tmp.astype('int32')
    #    train_test = train_test.join(tmp)
     #   train_test = train_test.drop(feat,1)

    features = train_test.columns
    x_train = train_test.iloc[:ntrain, :]
    x_test = train_test.iloc[ntrain:, :]

    
    

    train_test_scaled, scaler = scale_data(train_test)
    train, _ = scale_data(x_train, scaler)
    test, _ = scale_data(x_test, scaler)
    #train = x_train.as_matrix()
    #test = x_test.as_matrix()

    train_labels = np.log(np.array(train_loader['loss']))
    #train_labels = train_loader['loss']
    train_ids = train_loader['id'].values.astype(np.int32)
    test_ids = test_loader['id'].values.astype(np.int32)
    
    return train, train_labels, test, train_ids, test_ids, features


def categorical_feature_processing(data):
    merge = pd.read_csv('data/mergable_categories.csv',header=0,index_col=0)
    merge = merge.transpose()
    replace = {}
    for col in merge:
        replace[col] = {merge[col][0]:merge[col][1]}
    data = data.replace(to_replace=replace)

    state_means = pd.read_csv('data/state_mean.csv',index_col=0,header=None)
    state_means.columns = ["state_mean"]
    data = data.merge(state_means,left_on="cat112",right_index=True,how="left")

    return data

def continuous_feature_processing(data):
    poly = PolynomialFeatures(2,interaction_only=True,include_bias=False)
    numerical = [cont for cont in data.columns if "cont" in cont]
    tmp = data[numerical]
    for num in numerical:
        if(num not in ['cont14']):
            tmp.loc[:,num+"*"+"cont14"]=tmp[num]*tmp["cont14"]
    #tmp.loc[:,"lin_cont11*lin_cont14"] = tmp.lin_cont11*tmp.lin_cont14
    #tmp.loc[:,"lin_cont2*lin_cont5"] = tmp.lin_cont2*tmp.lin_cont5
    #tmp.loc[:,"lin_cont2*lin_cont11"] = tmp.lin_cont2*tmp.lin_cont11
    #tmp.loc[:,"lin_cont3*lin_cont5"] = tmp.lin_cont3*tmp.lin_cont5
    #tmp.loc[:,"lin_cont1*lin_cont13"] = tmp.lin_cont1*tmp.lin_cont13
    #tmp.loc[:,"lin_cont3*lin_cont11"] = tmp.lin_cont3*tmp.lin_cont1
    
    #tmp2 = poly.fit_transform(tmp)
    #tmp2 = pd.DataFrame(tmp2)
    #tmp2.columns = itertools.chain(tmp.columns,[x[0]+"*"+x[1] for x in itertools.combinations(tmp.columns,2)])
    
    data = data.drop(numerical,1)
    #data = data.join(tmp2)
    data = data.join(tmp)

    return data

