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
from sklearn.linear_model import LogisticRegression,BayesianRidge
import numpy as np
from datetime import datetime
from os.path import join,isfile

from data_setup import load_data




def optimize(trials,clf):

    n_iterations = 300
    space = clf[1]

    def score(params):
        print("Training with params: ")
        print(params)

        #n_estimators = params['n_estimators']
        #max_depth = params['max_depth']
        #learning_rate = params['learning_rate']

        alpha_1 = params['alpha_1'] 
        alpha_2 = params['alpha_2']
        lambda_1 = params['lambda_1']
        lambda_2 = params['lambda_2']

        cv_score = cross_val_score(
            BayesianRidge
            (
                alpha_1 =alpha_1,
                alpha_2 = alpha_2,
                lambda_1 = lambda_1,
                lambda_2 = lambda_2
            ),            
            #AdaBoostRegressor
            #(
            #    n_estimators = int(n_estimators),
            #    #max_depth = int(max_depth),
            #    learning_rate = learning_rate,
            #    random_state = 1001
            #),
            train,
            target,
            'neg_mean_absolute_error',
            cv=5
            ).mean()
        print("Current MAE Error: {}".format(-cv_score))

        return {'loss':-cv_score,'status':STATUS_OK}

    for i in range(len(trials.trials)+1,n_iterations):
        print("Turn {}".format(i))
        best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=i)
        pickle.dump(trials, open("trials_"+clf[0]+".p", "wb"))
    
    print(best)


#Load the data
train, target, test, _, ids = load_data()

#Get classifier from input flag
clfs = {}
clfs['AdaBoostRegressor'] = {
        'n_estimators' : hp.quniform('n_estimators', 50, 500, 1),
        'learning_rate' : 0.5,
        'random_state' : 1001
}
clfs['RandomForestRegressor'] ={
        'n_estimators' : hp.quniform('n_estimators', 50, 500, 1),
        'max_depth' : hp.quniform('max_depth', 1, 7, 1),
        'random_state' : 1001
        }

clfs['BayesianRidge'] ={
        'alpha_1' : hp.uniform('alpha_1', 1e-07, 1e02), 
        'alpha_2' : hp.uniform('alpha_2', 1e-07, 1e02), 
        'lambda_1': hp.uniform('lambda_1', 1e-07, 1e02), 
        'lambda_2': hp.uniform('lambda_2', 1e-07, 1e02)
        }

clf_name = 'BayesianRidge'
#clf_name = 'AdaBoostRegressor'
#clf_name = 'RandomForestRegressor'
trial_file = "trials_"+clf_name+".p" 
print(isfile(trial_file))

if(isfile(trial_file)):
    trials = pickle.load( open(trial_file, "rb"))
else:
    trials = Trials()

optimize(trials,(clf_name,clfs[clf_name]))
#print(trials.trials)
#print(trials.results)
