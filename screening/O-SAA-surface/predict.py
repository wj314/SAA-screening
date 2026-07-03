#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
from itertools import combinations
from mpi4py import MPI


def printres(filename, g, x_train, x_test, y_train, y_test, iscv=0):
    fp = open(filename, 'a+')
    fp.write('name= '+str(g)+'\n')
    y_train_pre = g.predict(x_train)
    y_test_pre = g.predict(x_test)

    fp.write('train score : {0:<15.6f} test score : {1:<15.6f} train r2_score : {2:<15.6f} test r2_score : {3:<15.6f} train MAE : {4:<15.6f} test MAE : {5:<15.6f} train RMSE : {6:<15.6f} test RMSE : {7:<15.6f}\n'.format(g.score(x_train, y_train), g.score(
        x_test, y_test), r2_score(y_train, y_train_pre), r2_score(y_test, y_test_pre), mean_absolute_error(y_train, y_train_pre), mean_absolute_error(y_test, y_test_pre), mean_squared_error(y_train, y_train_pre)**0.5, mean_squared_error(y_test, y_test_pre)**0.5))

    if iscv:
        fp.write('best_estimator_ = '+str(g.best_estimator_)+'\n')
        str0 = 'feature_importances_ = ['
        for i in g.best_estimator_.feature_importances_:
            str0 = str0+str(i)+' '
        str0 = str0+']\n'
        fp.write(str0)
        fp.write('best_score_ = '+str(g.best_score_)+'\n')
        fp.write('best_params_ = '+str(g.best_params_)+'\n\n')
    else:
        str0 = 'feature_importances_ = ['
        for i in g.feature_importances_:
            str0 = str0+str(i)+' '
        str0 = str0+']\n\n'
        fp.write(str0)
    fp.close()


def write_true_pred_value(filename_model, filename_t_p, m):
    joblib.dump(m, filename_model)
    Y_TRAIN_P = m.predict(X_TRAIN)
    Y_TEST_P = m.predict(X_TEST)
    with open(filename_t_p, 'w') as fp:
        fp.write('#train\n')

        for i in range(len(Y_TRAIN)):
            fp.write('{0:>10.6f} {1:>10.6f}\n'.format(Y_TRAIN.iloc[i], Y_TRAIN_P[i]))

        fp.write('\n#test\n')
        for i in range(len(Y_TEST)):
            fp.write('{0:>10.6f} {1:>10.6f}\n'.format(Y_TEST.iloc[i], Y_TEST_P[i]))


def train(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, rank, id0, fn):
    os.mkdir(str(id0))
    os.chdir(str(id0))
    seed = 2000
    logfile = 'log'
    fp = open(logfile, 'w')
    fp.write(str(id0)+' '+' '.join(fn)+'\n')
    fp.close()
    for lr in [0.01, 0.05, 0.1, 0.2]:
        with open(logfile,'a+') as fp:
            fp.write('############################### lr = '+str(lr)+' #########################################\n')
        m1 = GradientBoostingRegressor(random_state=seed, learning_rate=lr, validation_fraction=0.1, n_iter_no_change=10, tol=0.0001)
        m1.fit(X_TRAIN, Y_TRAIN)
        printres(logfile, m1, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, 0)

        m1 = GradientBoostingRegressor(random_state=seed, learning_rate=lr, max_features='sqrt',
                                       subsample=0.8, validation_fraction=0.1, n_iter_no_change=10, tol=0.0001)
        m1.fit(X_TRAIN, Y_TRAIN)
        printres(logfile, m1, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, 0)

        param = {'n_estimators': list(range(10, 1001, 10))}
        g1 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=lr, min_samples_split=2, min_samples_leaf=1, max_depth=3,
                          max_features='sqrt', subsample=0.8, random_state=seed, validation_fraction=0.1, n_iter_no_change=10, tol=0.0001), param_grid=param, cv=5)
        g1.fit(X_TRAIN, Y_TRAIN)
        printres(logfile, g1, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, 1)

        best_n_estimators = g1.best_params_['n_estimators']

        param = {'max_depth': list(range(2, 9, 1)), 'min_samples_split': list(range(2, 11, 1))}
        g2 = GridSearchCV(estimator=GradientBoostingRegressor(n_estimators=best_n_estimators, learning_rate=lr, min_samples_leaf=1,
                          max_features='sqrt', subsample=0.8, random_state=seed, validation_fraction=0.1, n_iter_no_change=10, tol=0.0001),
                          param_grid=param, cv=5)
        g2.fit(X_TRAIN, Y_TRAIN)
        printres(logfile, g2, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, 1)

        best_max_depth = g2.best_params_['max_depth']
        best_min_samples_split = g2.best_params_['min_samples_split']

        param = {'min_samples_leaf': list(range(1, 16, 2))}
        g3 = GridSearchCV(estimator=GradientBoostingRegressor(n_estimators=best_n_estimators, learning_rate=lr, min_samples_split=best_min_samples_split,
                                                              max_depth=best_max_depth, max_features='sqrt', subsample=0.8, random_state=seed, validation_fraction=0.1, n_iter_no_change=10, tol=0.0001),
                          param_grid=param, cv=5)
        g3.fit(X_TRAIN, Y_TRAIN)
        printres(logfile, g3, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, 1)

        best_min_samples_leaf = g3.best_params_['min_samples_leaf']

        if len(fn)<7:
            best_max_features=len(fn)
        else:
            param = {'max_features': list(range(7, len(fn)+1, 2))}
            g4 = GridSearchCV(estimator=GradientBoostingRegressor(n_estimators=best_n_estimators, learning_rate=lr, min_samples_split=best_min_samples_split,
                                                              max_depth=best_max_depth,  min_samples_leaf=best_min_samples_leaf, subsample=0.8, random_state=seed, validation_fraction=0.1, n_iter_no_change=10, tol=0.0001),
                          param_grid=param, cv=5)
            g4.fit(X_TRAIN, Y_TRAIN)
            printres(logfile, g4, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, 1)

            best_max_features = g4.best_params_['max_features']

        param = {'subsample': [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]}
        g5 = GridSearchCV(estimator=GradientBoostingRegressor(n_estimators=best_n_estimators, learning_rate=lr, min_samples_split=best_min_samples_split,
                                                              max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf, max_features=best_max_features, random_state=seed, validation_fraction=0.1, n_iter_no_change=10, tol=0.0001),
                          param_grid=param, cv=5)
        g5.fit(X_TRAIN, Y_TRAIN)
        printres(logfile, g5, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, 1)

        best_subsample = g5.best_params_['subsample']
        with open(logfile,'a+') as fp:
            fp.write('best_n_estimators: {0:<5d} best_max_depth: {1:<5d} best_min_samples_split: {2:<5d} best_min_samples_leaf: {3:<5d} best_max_features: {4:<5d} best_subsample: {5:<10.2f}\n'.format(
            best_n_estimators, best_max_depth, best_min_samples_split, best_min_samples_leaf, best_max_features, best_subsample))

            fp.write('# search best begin #\n')
        m1 = GradientBoostingRegressor(n_estimators=best_n_estimators, learning_rate=lr, min_samples_split=best_min_samples_split, max_depth=best_max_depth,
                                       min_samples_leaf=best_min_samples_leaf, max_features=best_max_features, subsample=best_subsample, random_state=seed, validation_fraction=0.1, n_iter_no_change=10, tol=0.0001)
        m1.fit(X_TRAIN, Y_TRAIN)
        printres(logfile, m1, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, 0)
        write_true_pred_value('model1_'+str(lr), 'true_pred_value1_'+str(lr), m1)

        # only change n_estimators
        m1 = GradientBoostingRegressor(n_estimators=best_n_estimators, learning_rate=lr, random_state=seed,
                                       validation_fraction=0.1, n_iter_no_change=10, tol=0.0001)
        m1.fit(X_TRAIN, Y_TRAIN)
        printres(logfile, m1, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, 0)
        write_true_pred_value('model2_'+str(lr), 'true_pred_value2_'+str(lr), m1)
        with open(logfile,'a+') as fp:
            fp.write('# search best end #\n')

    os.chdir('..')



target_name = 'Ea'

feature_names = ['DE', 'd_OM', 'OCN', 'Eb_aver_OM', 'MMCN', 'MMGCN', 'XNv_M', 'X_M', 'Nd', 'Eaffi_M', 'Eion1_M',
            'd_aver_MM', 'XNv_aver_MM', 'X_aver_MM', 'Eaffi_aver_MM', 'Eion1_aver_MM', 'Diff_XNv_aver_MM', 'Diff_X_aver_MM']


nf = len(feature_names)
coms = []
for i in range(1, nf+1):
    coms.extend(list(combinations(range(nf), i)))
fnamess = []
for i in range(len(coms)):
    fnames = []
    for j in coms[i]:
        fnames.append(feature_names[j])
    fnamess.append(fnames)

num=int(sys.argv[1])
data = pd.read_csv('res', sep='\s+')
#Y = data[target_name]
X = data[fnamess[num]]
#X.loc[:,'DE']=-1.50

model=joblib.load('model_'+str(num))

Y_PRE=model.predict(X)

for i in range(len(Y_PRE)):
    print('{0:<50s} {1:>5d} {2:>5d} {3:>5d} {4:>20.6f}'.format(data['name'].iloc[i],data['siteid'].iloc[i],data['Oid'].iloc[i],data['Mid'].iloc[i],Y_PRE[i]))

