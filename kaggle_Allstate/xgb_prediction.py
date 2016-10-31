import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from scipy.stats import skew, boxcox
from math import exp, log
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import numpy as np
from datetime import datetime
from os.path import join
import operator
from matplotlib import pylab as plt

from data_setup import load_data


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' %
              (tmin, round(tsec, 2)))

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

def feature_importance(clf):
    importance = clf.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(12, 40))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().savefig('feature_importance_xgb.png')

def main():

    # enter the number of folds from xgb.cv
    folds = 5
    cv_sum = 0
    early_stopping = 25
    fpred = []
    xgb_rounds = []
    use_cv = True  

    params = {}
    params['booster'] = 'gbtree'
    params['objective'] = "reg:linear"
    #params['objective'] = "multi:softprob"
    #params['num_class'] = 15
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

    start_time = timer(None)

    # Load data set and target values
    train, target, test, _, ids, features = load_data()
    print(train.shape,test.shape)
    #target = pd.cut(target,bins=15,labels=False)

    ceate_feature_map(features)

    #d_train_full = xgb.DMatrix(train, label=target)
    d_test = xgb.DMatrix(test)
    
    if(use_cv):
        # set up KFold that matches xgb.cv number of folds
        kf = KFold(train.shape[0], n_folds=folds)
        for i, (train_index, test_index) in enumerate(kf):
            print('Fold %d' % (i + 1))
            X_train, X_val = train[train_index], train[test_index]
            y_train, y_val = target[train_index], target[test_index]

        #######################################
        #
        # Define cross-validation variables
        #
        #######################################

            d_train = xgb.DMatrix(X_train, label=y_train)
            d_valid = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(d_train, 'train'), (d_valid, 'eval')]
      
        ####################################
        #  Build Model
        ####################################

            clf = xgb.train(params,
                            d_train,
                            100000,
                            watchlist,
                            verbose_eval=10,
                            early_stopping_rounds=early_stopping)

        ####################################
        #  Evaluate Model and Predict
        ####################################

            xgb_rounds.append(clf.best_iteration)
            scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
            cv_score = mean_absolute_error(np.exp(y_val), np.exp(scores_val))
            print(' eval-MAE: %.6f' % cv_score)
            y_pred = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit))

        ####################################
        #  Add Predictions and Average Them
        ####################################

            if i > 0:
                fpred = pred + y_pred
            else:
                fpred = y_pred
            pred = fpred
            cv_sum = cv_sum + cv_score

        feature_importance(clf)
        mpred = pred / folds
        score = cv_sum / folds
        print('\n Average eval-MAE: %.6f' % score)
        n_rounds = int(np.mean(xgb_rounds))
    else:
        # enter the number of iterations from xgb.cv with early_stopping turned on
        n_fixed = 376
        watchlist = [(d_train_full, 'train')]

        nfixed = int(n_fixed * (1 + (1. / folds)))
        print('\n Training full dataset for %d rounds ...\n' % n_fixed)
        clf_fixed = xgb.train(
            params, d_train_full,
            nfixed,
            watchlist,
            verbose_eval=False,)

        #######################
        # Feature Importance
        #######################
        feature_importance(clf_fixed)

        y_pred_fixed = np.exp(clf_fixed.predict(d_test))
        timer(start_time)
   
    print("#\n Writing results")
    if(use_cv):
        result = pd.DataFrame(mpred, columns=['loss'])
        result["id"] = ids
        result = result.set_index("id")
        print("\n %d-fold average prediction:\n" % folds)
        print(result.head())
    else:
        result_fixed = pd.DataFrame(y_pred_fixed, columns=['loss'])
        result_fixed["id"] = ids
        result_fixed = result_fixed.set_index("id")
        print("\n Full datset (at CV #iterations) prediction:\n")
        print(result_fixed.head())

    now = datetime.now()
    if(use_cv):
        score = str(round((cv_sum / folds), 6))
        sub_file = 'submission_5fold-average-xgb_' + str(score) + '_' + str(
            now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
        print("\n Writing submission: %s" % sub_file)
        result.to_csv(sub_file, index=True, index_label='id')
    else:
        sub_file = 'submission_full-CV-xgb_' + str(now.strftime(
        "%Y-%m-%d-%H-%M")) + '.csv'
        print("\n Writing submission: %s" % sub_file)
        result_fixed.to_csv(sub_file, index=True, index_label='id')


if __name__ == "__main__":
    main()

