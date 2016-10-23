from hyperopt import hp
from hyperopt import  tpe, hp, STATUS_OK, Trials
from hyperopt import fmin
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
from os.path import join

from data_setup import load_data


def score(params):
    print("Training with params: ")
    print(params)

    n_estimators = params['n_estimators']
    max_depth = params['max_depth']

    cv_score = cross_val_score(
        ExtraTreesRegressor
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

    return {'loss':cv_score,'status':STATUS_OK}

def optimize(trials):
    space = {
            'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
            'max_depth' : hp.quniform('max_depth', 1, 13, 1)
            }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

    print(best)

#Load the data
train, target, test, _, ids = load_data()

trials = Trials()

optimize(trials)
