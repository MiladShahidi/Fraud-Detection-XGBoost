import pandas as pd
import numpy as np
# For Bayesian hyper-parameter optimization
import hyperopt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, precision_recall_curve, recall_score, precision_score
from functools import partial
# To supress a deprecation warning caused due to an issue between XGBoost and SciPy
import warnings

MIN_PRECISION = 0.05

# The current version of XGBoost uses a conditional statement that
# the current version SciPy doesn't like.
# This supresses SciPy's deprecation warning message
warnings.filterwarnings('ignore', category = DeprecationWarning)

def conditional_recall_score(y_true, pred_proba, min_prec = MIN_PRECISION):
    # Since the PR curve is discreet it might not contain the exact precision value given
    # So we first find the closest existing precision to the given level
    # Then return the highest recall acheiveable at that precision level
    # Taking max() helps in case PR curve is locally flat
    # with multiple recall values for the same precision
    pr, rc,_ = precision_recall_curve(y_true, pred_proba[:,1])
    return np.max(rc[pr >= min_prec])

def objective(params, X, y, X_early_stop, y_early_stop, scorer, n_folds = 10):

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    imbalance_ratio = neg_count / pos_count
    
    xgb_clf = XGBClassifier(**params, scale_pos_weight=imbalance_ratio,
                            n_estimators = 2000, n_jobs = 1)

    xgb_fit_params = {'early_stopping_rounds': 50,
                      'eval_metric': ['logloss'],
                      'eval_set': [(X_early_stop, y_early_stop)],
                      'verbose': False
                      }
    
    cv_score = np.mean(cross_val_score(xgb_clf, X_train, y_train, cv = n_folds,
                               fit_params = xgb_fit_params, n_jobs = -1,
                               scoring = scorer))
    
    # hypoeropt minimizes the loss, hence the minus sign behind cv_score
    return {'loss': -cv_score, 'status': hyperopt.STATUS_OK, 'params': params}

def tune_xgb(param_space, X_train, y_train, X_early_stop, y_early_stop, n_iter):    
    scorer = make_scorer(conditional_recall_score, needs_proba=True)

    # hyperopt.fmin will only pass the parameter values to objective. So we need to
    # create a partial function to bind the rest of the arguments we want to pass to objective
    obj = partial(objective, scorer = scorer, X = X_train, y = y_train,
                  X_early_stop = X_early_stop, y_early_stop = y_early_stop)

    # A trials object that will store the results of all iterations
    trials = hyperopt.Trials()
    
    hyperopt.fmin(fn = obj, space = param_space, algo = hyperopt.tpe.suggest,
                         max_evals = n_iter, trials = trials)
    
    # returns the values of parameters from the best trial
    return trials.best_trial['result']['params']

def optimal_threshold(estimator, X, y, n_folds = 10, min_prec = 0.05, fit_params = None):
    
    cv_pred_prob = cross_val_predict(estimator, X, y, method='predict_proba',
                                     cv = n_folds, fit_params=fit_params, n_jobs=-1)[:,1]

    # Once again, the PR curve is discreet and may not contain the exact precision level
    # we are looking for. So, we need to find the closest existing precision
    pr, _, threshold = precision_recall_curve(y, cv_pred_prob)
    # precision is always one element longer than threshold and the last one is always set to 1
    # So I drop the last element of precision so I can use it below to index threshold
    pr = pr[:-1]
    return min(threshold[pr >= min_prec])

def thresholded_predict(X, estimator, threshold):
    return np.array([1 if (p >= threshold) else 0 for p in estimator.predict_proba(X)[:,1]])

if __name__ == "__main__":    
    # Loading the data
    data = pd.read_csv('creditcard.csv')
    X = data.drop('Class', axis = 1)
    y = data['Class']
    
    # Train/test split, 80/20, random_state set for reproduibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 1)

    # Further splitting the initial training set so that 10% of all data(1/8 of 80%) 
    # can be used as the evaluation set by XGBoost for early stopping
    X_train, X_early_stop, y_train, y_early_stop = train_test_split(X_train, y_train, stratify = y_train, test_size = 1/8, random_state = 1)
    
    # The prior probability distribution of parameters for Bayesian optimization
    param_space = {
            'learning_rate': hyperopt.hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'max_depth': hyperopt.hp.choice('max_depth', [2, 4, 6, 8, 10]),
            'subsample': hyperopt.hp.uniform('subsample', 0.25, 1),
            'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.7, 1.0),
            'min_child_weight': hyperopt.hp.choice('min_child_weight', [1, 3, 5, 7]),
            'reg_alpha': hyperopt.hp.uniform('reg_alhpa', 0, 1.0),
            # Avoiding lambda = 0. There is a Github issue on strange behaviour with lambda = 0
            'reg_lambda': hyperopt.hp.uniform('reg_lambda', 0.01, 1.0),
            }

    # # # # # # # # #
    # Step 1: Tuning hyper-parameters of the XGBoost classifier
    # # # # # # # # #
    print('Step 1: Tuning hyper-parameters using Bayesian Optimization\n')

    best_params = tune_xgb(param_space, X_train, y_train, X_early_stop, y_early_stop, n_iter = 150)
    
    print('\tThe best hyper-parameters found:\n')
    print(*['\t\t%s = %s' % (k, str(round(v, 4))) for k, v in best_params.items()], sep='\n')

    # # # # # # # # #
    # Step 2: Empirical thresholding: finding optimal classification threshold
    # # # # # # # # #
    print('\nStep 2: Empirical Thresholding\n')
    
    # I use 1500 trees which is very close to optimal n_trees found by early stopping while tuning
    xgboost_clf = XGBClassifier(**best_params, n_estimators=1500)
    
    classification_cutoff = optimal_threshold(xgboost_clf, X_train, y_train, min_prec = MIN_PRECISION)
    
    print('\tOptimal classification threshold = %1.3f' % classification_cutoff)
    
    # # # # # # # # #
    # Setp 3: Training and testing the model
    # # # # # # # # #
    print('\nStep 3: Training and testing the model\n')
    
    # Training on all the training data (excluding the small validation set to avoid overfitting)
    xgboost_clf.fit(X_train, y_train, verbose = False)
    
    y_pred = thresholded_predict(X_test, xgboost_clf, threshold = classification_cutoff)
    
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    
    print('\tTest set performance:')
    print('\tRecall    = %2.3f' % test_recall)
    print('\tPrecision = %2.3f' % test_precision)