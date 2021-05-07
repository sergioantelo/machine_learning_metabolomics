import numpy as np
import optuna
import pandas as pd
import pickle
import pprint
from sklearn.metrics import mean_absolute_error, median_absolute_error

from utils.data import load_alvadesc_data, load_descriptors
from utils.train.model_selection import RegressionStratifiedKFold
from utils.train.param_search import param_search
from models.preprocessors.preprocessors import Preprocessor
from models.preprocessors.column_selectors import make_col_selector
from models.regressors.WeightedCatBoostRegressor import WeightedCatBoostRegressor

#######################################################################################################################
SEED = 42
RESULTS_FILENAME = 'data/results/catboost.pkl'
# FIXME: change this! a small part of the dataset is used for smoke test!
SMOKE_TEST = False

if SMOKE_TEST:
    OUT_CV = 2
    SEARCH_CV = 2
    N_TRIALS = 2
else:
    OUT_CV = 5
    SEARCH_CV = 5
    N_TRIALS = 50
######################################################################################################################

if __name__ == '__main__':
    common_cols = ['pid', 'rt']

    #Load fgps and descriptors
    fgp = load_alvadesc_data(split_as_np=False)
    descriptors = load_descriptors(split_as_np=False)
    descriptors = descriptors.drop_duplicates()

    descriptors_fgp = pd.merge(descriptors, fgp, on=common_cols)

    def get_feature_names(x):
        return x.drop(common_cols, axis=1).columns

    #Get fgps and descriptors' features
    X_fgp = descriptors_fgp[get_feature_names(fgp)].values.astype('float32')
    X_desc = descriptors_fgp[get_feature_names(descriptors)].values.astype('float32')
    X = np.concatenate([X_desc, X_fgp], axis=1)

    #Select either fgps or descriptors' features
    desc_col_selector = make_col_selector(X_desc.shape[1], return_first_half=True)
    fgp_col_selector = make_col_selector(X_desc.shape[1], return_first_half=False)

    #Get the target
    y = descriptors_fgp['rt'].values.astype('float32').flatten()

    #If test then take an X,y sample of length 1000 both
    if SMOKE_TEST:
        X, y = X[:1000, ...], y[:1000]

    #Preprocess data 
    #TODO: create a new pipeline that further improves preprocessing
    preprocessor = Preprocessor(desc_col_selector=desc_col_selector, fgp_col_selector=fgp_col_selector)
    estimator = WeightedCatBoostRegressor()

    outer_cv = RegressionStratifiedKFold(OUT_CV, random_seed=SEED)
    search_cv = RegressionStratifiedKFold(SEARCH_CV, random_seed=SEED)
    metrics = {'mae': mean_absolute_error, 'med_ae': median_absolute_error}
    results = []
    # We must use nested-cross-validation for evaluating the performance of the model. A new search must be done
    # withing each fold!
    for fold, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
        print("Hello")

        #Stopped here when REAL TEST
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if preprocessor:
            X_train = preprocessor.fit_transform(X_train, y_train)
            X_test = preprocessor.transform(X_test)

        study = optuna.create_study(
            study_name=f'CatBoost-CV-{fold}',
            direction='maximize',
            storage='sqlite:///data/optuna/catboost.db',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner()
        )
        best_params = param_search(estimator, X_train, y_train, search_cv, study, n_trials=N_TRIALS)

        estimator.set_params(**best_params)
        estimator.fit(X_train, y_train)

        cv_results = {k: [] for k in metrics.keys()}
        for key, metric in metrics.items():
            cv_results[key].append(metric(y_test, estimator.predict(X_test)).astype('float64'))
        results.append(cv_results)
        print(f"CV {fold} | {', '.join(['{} = {}'.format(k, v) for k, v in cv_results.items()])}")

    print('---------------------- Final results -------------------------')
    pprint.pprint(results)
    with open(RESULTS_FILENAME, 'wb') as fd:
        pickle.dump(results, fd)

