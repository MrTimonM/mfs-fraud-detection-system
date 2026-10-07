"""CPU estimators and small randomized chronological holdout searches."""
import logging
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from .resource_benchmark import ResourceMonitor


def estimators(cfg, ratio=19, multiclass=False):
    threads, seed = cfg['threads'], cfg['seed']
    return {
        'LogisticRegression': LogisticRegression(max_iter=500, class_weight='balanced', C=.1, solver='lbfgs'),
        'RandomForest': RandomForestClassifier(n_estimators=80, max_depth=13, min_samples_leaf=10,
            class_weight='balanced_subsample', n_jobs=threads, random_state=seed, max_samples=.8),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=23,
            l2_regularization=5, class_weight='balanced', early_stopping=False, random_state=seed),
        'XGBoost': XGBClassifier(n_estimators=120 if not multiclass else 70, max_depth=4,
            learning_rate=.08, subsample=.8, colsample_bytree=.85, tree_method='hist',
            n_jobs=threads, random_state=seed, **({} if multiclass else {'scale_pos_weight': ratio})),
        'LightGBM': LGBMClassifier(n_estimators=140 if not multiclass else 80, num_leaves=23,
            max_depth=-1, learning_rate=.06, reg_lambda=5, min_child_samples=30,
            class_weight='balanced', n_jobs=threads, random_state=seed, verbosity=-1,
            deterministic=True, force_col_wise=True),
    }


def fit_model(name, estimator, x, y, tx, ty, cfg):
    grids = {'RandomForest': {'max_depth': [10, 14], 'min_samples_leaf': [8, 18]},
        'LightGBM': {'num_leaves': [15, 23, 31], 'reg_lambda': [3, 8]},
        'XGBoost': {'max_depth': [3, 4, 5], 'reg_lambda': [3, 8]}}
    trials = list(ParameterSampler(grids[name], n_iter=cfg['tuning_trials'], random_state=cfg['seed'])) if name in grids else [{}]
    best, best_ap, records = None, -1, []
    with ResourceMonitor() as monitor:
        for params in trials:
            candidate = clone(estimator).set_params(**params)
            candidate.fit(x, y)
            score = average_precision_score(ty, candidate.predict_proba(tx)[:, 1])
            records.append({'parameters': params, 'validation_pr_auc': score})
            if score > best_ap:
                best, best_ap = candidate, score
    logging.info('%s fit %.1fs validation AP %.4f', name, monitor.seconds, best_ap)
    return best, {'training_time_seconds': monitor.seconds, 'peak_memory_mb': monitor.peak_mb,
        'cpu_seconds': monitor.cpu_seconds, 'tuning_trials': records,
        'best_validation_pr_auc': best_ap, 'best_parameters': best.get_params()}
