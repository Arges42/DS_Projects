from hyperopt import hp
from hyperopt import  tpe, hp, STATUS_OK, Trials
from hyperopt import fmin
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox
from math import exp, log
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor,ExtraTreesRegressor,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import numpy as np
from datetime import datetime
from os.path import join,isfile

from data_setup import load_data


def score(params):
    print("Training with params: ")
    print(params)

    n_estimators = params['n_estimators']
    max_depth = params['max_depth']

    cv_score = cross_val_score(
        RandomForestRegressor
        (
            n_estimators = int(n_estimators),
            max_depth = int(max_depth),
            random_state = 1001
        ),
        train,
        target,
        'mean_absolute_error',
        cv=5
        ).mean()
    print("Current MAE Error: {}".format(-cv_score))

    return {'loss':-cv_score,'status':STATUS_OK}

def optimize(trials):
    space = {
        'n_estimators' : hp.quniform('n_estimators', 10, 11, 1),
        'max_depth' : hp.quniform('max_depth', 1, 2, 1)
        }
    if(isfile("trials.p")):
        trials = pickle.load( open("trials.p", "rb"))
    else:
        trials = Trials()
    n_iterations = 100

    for i in range(len(trials.trials)+1,n_iterations):
        print("Turn {}".format(i))
        best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=i)
        pickle.dump(trials, open("trials.p", "wb"))
    
    print(best)


#Load the data
train, target, test, _, ids = load_data()

trials = Trials()

optimize(trials)
print(trials.trials)
print(trials.results)
