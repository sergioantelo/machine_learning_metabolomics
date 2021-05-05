from functools import singledispatch

import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.base import clone

from models.regressors.WeightedCatBoostRegressor import WeightedCatBoostRegressor
from utils.train.loss import truncated_medae_scorer, truncated_rmse_scorer

@singledispatch
def suggest_params(estimator, trial):
    raise NotImplementedError

@suggest_params.register
def _(estimator: WeightedCatBoostRegressor, trial):
    params = {
        "depth": trial.suggest_int("depth", 1, 10, 1),
        "iterations": trial.suggest_int("iterations", 10, 100, 10),
        #"use_best_model": trial.suggest_categorical("use_best_model",['True','False']), #provide non-empty eval_set
        "eval_metric": trial.suggest_categorical("eval_metric", ['RMSE','MAE','MedianAbsoluteError']),
        #"learning_rate": trial.suggest_int("learning_rate", 0.01, 0.1, 0.01),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 5, 30, 5)
    }
    return params

def create_objective(estimator, X, y, cv, scoring):
    estimator_factory = lambda: clone(estimator)

    def objective(trial):
        estim = estimator_factory()
        params = suggest_params(estim, trial)
        estim.set_params(**params)
        return cross_val_score_with_pruning(estim, X, y, cv=cv, scoring=scoring, trial=trial)

    return objective

def cross_val_score_with_pruning(estimator, X, y, cv, scoring, trial):
    cross_val_scores = []
    for step, (train_index, test_index) in enumerate(cv.split(X, y)):
        est = clone(estimator)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        est.fit(X_train, y_train)
        cross_val_scores.append(scoring(est, X_test, y_test))
        intermediate_value = np.mean(cross_val_scores)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(cross_val_scores)

@singledispatch
def param_search(estimator, X, y, cv, study, n_trials, scoring=truncated_rmse_scorer, keep_going=False):
    objective = create_objective(estimator, X, y, cv, scoring)
    trials = [trial for trial in study.get_trials() if trial.state in [TrialState.COMPLETE, TrialState.PRUNED]]
    if not keep_going:
        n_trials = n_trials - len(trials)
    if n_trials > 0:
        print(f"Starting {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)
    return load_best_params(estimator, study)

@singledispatch
def load_best_params(estimator, study):
    try:
        return study.best_params
    except Exception as e:
        print(f'Study for {type(estimator)} does not exist')
        raise e

@singledispatch
def set_best_params(estimator, study):
    if study is not None:
        best_params = load_best_params(estimator, study)
        estimator.set_params(**best_params)
    return estimator

