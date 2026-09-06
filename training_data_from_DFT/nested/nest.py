import os
import re
import sys
import time
import argparse
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
import json

SEED = 2000
np.random.seed(SEED)

def bootstrap_error_estimate_grouped(model_template, params, X, y, groups, n_bootstrap=1000, random_state=SEED):
    rng = np.random.RandomState(random_state)
    oob_rmse, oob_mae, oob_r2 = [], [], []

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    group_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)

        train_idx = np.concatenate([group_indices[g] for g in sampled_groups])
        oob_groups = np.setdiff1d(unique_groups, sampled_groups)
        if len(oob_groups) == 0:
            continue
        oob_idx = np.concatenate([group_indices[g] for g in oob_groups])

        X_boot = X.iloc[train_idx]
        y_boot = y.iloc[train_idx]
        X_oob = X.iloc[oob_idx]
        y_oob = y.iloc[oob_idx]

        model = clone(model_template)
        model.set_params(**params)
        model.fit(X_boot, y_boot)

        y_pred = model.predict(X_oob)
        oob_rmse.append(np.sqrt(mean_squared_error(y_oob, y_pred)))
        oob_mae.append(mean_absolute_error(y_oob, y_pred))
        oob_r2.append(r2_score(y_oob, y_pred))


    oob_rmse = np.array(oob_rmse)
    oob_mae = np.array(oob_mae)
    oob_r2 = np.array(oob_r2)

    results = {
        'RMSE_mean': oob_rmse.mean(),
        'RMSE_std': oob_rmse.std(),
        'RMSE_median': np.median(oob_rmse),
        'RMSE_95CI_lower': np.percentile(oob_rmse, 2.5),
        'RMSE_95CI_upper': np.percentile(oob_rmse, 97.5),
        'MAE_mean': oob_mae.mean(),
        'MAE_std': oob_mae.std(),
        'MAE_median': np.median(oob_mae),
        'MAE_95CI_lower': np.percentile(oob_mae, 2.5),
        'MAE_95CI_upper': np.percentile(oob_mae, 97.5),
        'r2_mean': oob_r2.mean(),
        'r2_std': oob_r2.std(),
        'r2_median': np.median(oob_r2),
        'r2_95CI_lower': np.percentile(oob_r2, 2.5),
        'r2_95CI_upper': np.percentile(oob_r2, 97.5),
        'n_bootstrap_used': len(oob_rmse),
        'oob_rmse':list(oob_rmse),
        'oob_mae':list(oob_mae),
        'oob_r2':list(oob_r2)
    }

    return results

def bootstrap_error_estimate(model_template, params, X, y, n_bootstrap=1000, random_state=SEED):
    rng = np.random.RandomState(random_state)
    oob_rmse, oob_mae, oob_r2 = [], [], []
    n_samples = len(y)

    for _ in range(n_bootstrap):
        bootstrap_idx = rng.choice(n_samples, size=n_samples, replace=True)
        oob_idx = np.setdiff1d(np.arange(n_samples), bootstrap_idx)
        if len(oob_idx) == 0:
            continue

        X_boot = X.iloc[bootstrap_idx]
        y_boot = y.iloc[bootstrap_idx]
        X_oob = X.iloc[oob_idx]
        y_oob = y.iloc[oob_idx]

        model = clone(model_template)
        model.set_params(**params)
        model.fit(X_boot, y_boot)
        y_pred = model.predict(X_oob)

        oob_rmse.append(np.sqrt(mean_squared_error(y_oob, y_pred)))
        oob_mae.append(mean_absolute_error(y_oob, y_pred))
        oob_r2.append(r2_score(y_oob, y_pred))

    oob_rmse = np.array(oob_rmse)
    oob_mae = np.array(oob_mae)
    oob_r2 = np.array(oob_r2)



    results = {
        'RMSE_mean': oob_rmse.mean(),
        'RMSE_std': oob_rmse.std(),
        'RMSE_median': np.median(oob_rmse),
        'RMSE_95CI_lower': np.percentile(oob_rmse, 2.5),
        'RMSE_95CI_upper': np.percentile(oob_rmse, 97.5),
        'MAE_mean': oob_mae.mean(),
        'MAE_std': oob_mae.std(),
        'MAE_median': np.median(oob_mae),
        'MAE_95CI_lower': np.percentile(oob_mae, 2.5),
        'MAE_95CI_upper': np.percentile(oob_mae, 97.5),
        'r2_mean': oob_r2.mean(),
        'r2_std': oob_r2.std(),
        'r2_median': np.median(oob_r2),
        'r2_95CI_lower': np.percentile(oob_r2, 2.5),
        'r2_95CI_upper': np.percentile(oob_r2, 97.5),
        'n_bootstrap_used': len(oob_rmse),
        'oob_rmse':list(oob_rmse),
        'oob_mae':list(oob_mae),
        'oob_r2':list(oob_r2)
    }
    return results




data = pd.read_csv('data', sep='\s+')

groups = data['metal']


target_name = 'Ea'

#features_to_use= [ 'DE', 'd_OM', 'MMGCN', 'XNv_M', 'X_aver_MM', 'Diff_XNv_aver_MM']
features_to_use= [ 'DE',
    'd_OM', 'OCN', 'Eb_aver_OM', 'MMCN', 'MMGCN', 'XNv_M', 'X_M', 'Nd',
    'Eaffi_M', 'Eion1_M', 'd_aver_MM', 'XNv_aver_MM', 'X_aver_MM',
    'Eaffi_aver_MM', 'Eion1_aver_MM', 'Diff_XNv_aver_MM', 'Diff_X_aver_MM'
]


X = data[features_to_use].copy()
y = data[target_name].copy()
groups=data['metal']

outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=5)

param_grid={
        'GBDT':{
            'selector__threshold': ['mean', 'median', 0.01, 0.05, 0.1],
            'regressor__n_estimators':[20, 40, 50, 100, 200, 300, 400, 500],
            'regressor__max_depth':[2, 3, 4],
            'regressor__min_samples_split':[2, 5, 10, 20],
            'regressor__min_samples_leaf':[1, 3, 5, 10],
            'regressor__max_features':['sqrt', 0.5, None],
            'regressor__subsample':[0.7, 0.8, 1.0],
            'regressor__learning_rate':[0.01, 0.05, 0.1, 0.2]
            },
        'GPR':{
            'selector__threshold': ['mean', 'median', 0.01, 0.05, 0.1],
            'regressor__alpha': [1e-10, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
            },
        'LASSO':{
            'selector__threshold': ['mean', 'median', 0.01, 0.05, 0.1],
            'regressor__alpha': np.logspace(-3, 1, 10)
            },
        'RF':{
            'selector__threshold': ['mean', 'median', 0.01, 0.05, 0.1],
            'regressor__n_estimators': [50, 100, 200, 300, 400, 500],
            'regressor__max_depth': [3, 5, 7],
            'regressor__min_samples_split': [2, 5, 10, 20],
            'regressor__min_samples_leaf': [1, 3, 5, 10],
            'regressor__max_features': ['sqrt', 0.5, None]
            },
        'SVR':{
            'selector__threshold': ['mean', 'median', 0.01, 0.05, 0.1],
            'regressor__kernel': ['rbf'],
            'regressor__C': np.logspace(-2, 3, 10),
            'regressor__gamma': np.logspace(-3, 1, 10),
            'regressor__epsilon': [0.01, 0.05, 0.1, 0.2]
            }
        }

gbdt_pipe=Pipeline([
    ('selector', SelectFromModel(estimator=GradientBoostingRegressor(loss='squared_error', random_state=SEED, validation_fraction=0.1, n_iter_no_change=10))),
    ('regressor', GradientBoostingRegressor(loss='squared_error', random_state=SEED, validation_fraction=0.1, n_iter_no_change=10 ))
    ])

gpr_selector_estimator = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=SEED)
gpr_regressor = GaussianProcessRegressor(
kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
alpha=1e-3,
random_state=SEED,
normalize_y=True
)
gpr_pipe= Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectFromModel(estimator=gpr_selector_estimator)),
        ('regressor', gpr_regressor)
    ])

lasso_selector_estimator = Lasso(alpha=0.1, random_state=SEED, max_iter=10000)
lasso_regressor = Lasso(random_state=SEED, max_iter=10000)
lasso_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(estimator=lasso_selector_estimator)),
    ('regressor', lasso_regressor)
])

rf_pipe = Pipeline([
    ('selector', SelectFromModel(estimator=RandomForestRegressor(random_state=SEED))),
    ('regressor', RandomForestRegressor(random_state=SEED))
    ])

svr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectFromModel(estimator=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=SEED))),
        ('regressor', SVR())
    ])


models={ 'GBDT': gbdt_pipe, 'GPR': gpr_pipe, 'LASSO': lasso_pipe, 'RF': rf_pipe, 'SVR': svr_pipe}
bootstrap_res={}

for modelname in models:
    model=models[modelname]

    outer_rmse, outer_mae, outer_r2 = [], [], []
    best_params_list = []

    print('####################'+modelname+'####################')
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        model_clone = clone(model)
        inner_grid = GridSearchCV(
                estimator=model_clone,
                param_grid=param_grid[modelname],
                cv=inner_cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
        inner_grid.fit(X_train, y_train, groups=groups_train)
        best_estimator = inner_grid.best_estimator_
        best_params = inner_grid.best_params_
        best_params_list.append(best_params)
        y_pred = best_estimator.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        outer_rmse.append(rmse)
        outer_mae.append(mae)
        outer_r2.append(r2)
        selector = best_estimator.named_steps['selector']
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"  Fold {fold+1}: selected {len(selected_features)} features: {selected_features}")
        print('RMSE: {0:>10.4f} MAE: {1:>10.4f} R2: {2:>10.4f}'.format(rmse,mae,r2))
        print('best_params: ',best_params)

    rmse=np.mean(outer_rmse)
    rmse_std=np.std(outer_rmse)
    mae=np.mean(outer_mae)
    mae_std=np.std(outer_mae)
    r2=np.mean(outer_r2)
    r2_std=np.std(outer_r2)
    print(modelname+' Nested CV Summary mean RMSE: {0:>.4f} ± {1:>.4f} eV, MAE: {2:>.4f} ± {3:>.4f} eV, R2: {4:>.4f} ± {5:.4f}'.format(rmse,rmse_std,mae,mae_std,r2,r2_std))

    modeln=clone(model)
    final_grid = GridSearchCV(
        estimator=modeln,
        param_grid=param_grid[modelname],
        cv=GroupKFold(n_splits=5),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    final_grid.fit(X, y, groups=groups)

    y_pred=final_grid.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    best_estimator=final_grid.best_estimator_
    selector = best_estimator.named_steps['selector']
    selected_features_final = X.columns[selector.get_support()].tolist()
    print(f"{modelname} Final selected {len(selected_features_final)} features: {selected_features_final}")

    print(modelname+" Final best hyperparameters:", final_grid.best_params_)
    print(modelname+" Final best CV RMSE:", np.sqrt(-final_grid.best_score_))
    print(modelname+' Final RMSE: {0:>.4f} eV, MAE: {1:>.4f} eV, R2: {2:>.4f}'.format(rmse,mae,r2))
    print(modelname+' cv_results_:',final_grid.cv_results_)

    if hasattr(final_grid.best_estimator_['regressor'], 'feature_importances_'):
        model_output = {
                'model': final_grid.best_estimator_,
                'selected_features': selected_features_final,
                'hyperparameters': final_grid.best_params_,
                'feature_importances': final_grid.best_estimator_['regressor'].feature_importances_,
                'scikit_learn_version': sklearn.__version__,
                'python_version': sys.version,
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    elif hasattr(final_grid.best_estimator_['regressor'], 'coef_'):
        model_output = {
                'model': final_grid.best_estimator_,
                'selected_features': selected_features_final,
                'hyperparameters': final_grid.best_params_,
                'feature_importances': final_grid.best_estimator_['regressor'].coef_,
                'scikit_learn_version': sklearn.__version__,
                'python_version': sys.version,
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        model_output = {
                'model': final_grid.best_estimator_,
                'selected_features': selected_features_final,
                'hyperparameters': final_grid.best_params_,
                'feature_importances': None,
                'scikit_learn_version': sklearn.__version__,
                'python_version': sys.version,
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    joblib.dump(model_output, modelname+'.joblib')

    with open('model_metadata.txt', 'a+') as f:
        f.write('####################'+modelname+'####################\n')
        f.write(modelname+" Final Model Metadata\n")
        f.write(f"Scikit-learn version: {sklearn.__version__}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Training date: {model_output['training_date']}\n")
        f.write(f"Selected features ({len(selected_features_final)}): {selected_features_final}\n")
        f.write(f"Hyperparameters: {final_grid.best_params_}\n")
        if model_output['feature_importances'] is None:
            f.write(f"Feature importances: None\n")
        else:
            f.write(f"Feature importances: {dict(zip(selected_features_final, model_output['feature_importances']))}\n")

    bootstrap_res[modelname]=bootstrap_error_estimate_grouped(final_grid.best_estimator_, final_grid.best_params_, X, y, groups, n_bootstrap=1000, random_state=SEED)
    with open('bootstrap.json','w') as fp:
        json.dump(bootstrap_res,fp)
