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
from bayes_opt import BayesianOptimization

from data_setup import load_data

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' %
              (tmin, round(tsec, 2)))




def main():
    # enter the number of folds from xgb.cv
    folds = 5
    pfolds = 4
    cv_sum = 0
    early_stopping = 25
    fpred = []
    xgb_rounds = []

    start_time = timer(None)
    # Load data set and target values
    train, target, test, _, ids = load_data()

    optimize()

    #XGBoost params
    params = {}
    params['booster'] = 'gbtree'
    params['objective'] = "reg:linear"
    params['eval_metric'] = 'mae'
    params['eta'] = 0.1
    params['gamma'] = 0.5290
    params['min_child_weight'] = 4.2922
    params['colsample_bytree'] = 0.3085
    params['subsample'] = 0.9930
    params['max_depth'] = 7
    params['max_delta_step'] = 0
    params['silent'] = 1
    params['random_state'] = 1001


    #Level 0 clf
    clfs = [
            XGBClassifier(),
            ExtraTreesRegressor(random_state=1001),
            AdaBoostRegressor(random_state=1001) 
          ]

    kf = KFold(train.shape[0], n_folds=folds)
    # Pre-allocate the data
    # Number of training data x Number of classifiers
    blend_train = np.zeros((train.shape[0], len(clfs))) 
    # Number of testing data x Number of classifiers
    blend_test = np.zeros((test.shape[0], len(clfs))) 

    # For each classifier, we train the number of fold times (=len(skf))
    for j, clf in enumerate(clfs):
        print('Training classifier {}'.format(str(j)))
        # Number of testing data x Number of folds , we will take the mean of the predictions later
        blend_test_j = np.zeros((test.shape[0], len(kf))) 
        for i, (train_index, cv_index) in enumerate(kf):
            print('Fold {}'.format(i))
            
            # This is the training and validation set
            X_train = train[train_index]
            Y_train = target[train_index]
            X_valid = train[cv_index]
            Y_valid = target[cv_index]
            if(j==0):
                d_train = xgb.DMatrix(X_train, label=Y_train)
                d_valid = xgb.DMatrix(X_valid, label=Y_valid)
                d_test = xgb.DMatrix(test)
                watchlist = [(d_train, 'train'), (d_valid, 'eval')]
                clf = xgb.train(params,d_train,100000,watchlist,
                    early_stopping_rounds=25,verbose_eval=50)

            else:
                clf.fit(X_train, Y_train)
            
            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            if(j==0):
                blend_train[cv_index, j] = clf.predict(d_valid)
                blend_test_j[:, i] = clf.predict(d_test)
            else:
                blend_train[cv_index, j] = clf.predict(X_valid)
                blend_test_j[:, i] = clf.predict(test)

        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)

    ####################### Blending #######################################
    pf = KFold(blend_train.shape[0], n_folds = pfolds)
    bclf = RandomForestRegressor()

    for i, (train_index, cv_index) in enumerate(pf):
        X_train = train[train_index]
        Y_train = target[train_index]
        X_valid = train[cv_index]
        Y_valid = target[cv_index]

        bclf.fit(X_train, Y_train)
        Y_valid_prediction = bclf.predict(X_valid)
        cv_score = mean_absolute_error(np.exp(Y_valid), np.exp(Y_valid_prediction))
        print(' eval-MAE: %.6f' % cv_score)
        Y_test_predict = np.exp(bclf.predict(blend_test))

        #  Add Predictions and Average Them
        if i > 0:
            fpred = pred + y_pred
        else:
            fpred = y_pred
        pred = fpred
        cv_sum = cv_sum + cv_score
    
    mpred = pred / pfolds
    score = cv_sum / pfolds
    print('\n Average eval-MAE: %.6f' % score)

    #Y_train_predict = bclf.predict(blend_train)
    #print("Train MAE {:.6f}".format(mean_absolute_error(np.exp(Y_train_predict),np.exp(target))))
    #Y_test_predict = np.exp(bclf.predict(blend_test))

    print("#\n Writing results")
    result = pd.DataFrame(mpred, columns=['loss'])
    result["id"] = ids
    result = result.set_index("id")
    print("\n %d-fold average prediction:\n" % pfolds)
    print(result.head())

    now = datetime.now()
    score = str(round((cv_sum / pfolds), 6))
    sub_file = 'submission_logistic-blending_' + str(score) + '_' + str(
        now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print("\n Writing submission: %s" % sub_file)
    result.to_csv(sub_file, index=True, index_label='id')

if __name__ == "__main__":
    main()

