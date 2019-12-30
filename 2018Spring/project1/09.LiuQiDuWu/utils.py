from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LGR
from sklearn.svm import SVC as SVM
from sklearn.ensemble import RandomForestClassifier as RFC


def init_LDA(params):
    solver = getattr(params, 'solver', 'svd')
    shrinkage = getattr(params, 'shrinkage', None)
    priors = getattr(params, 'priors', None)
    n_components = getattr(params, 'n_components', None)
    store_covariance = getattr(params, 'store_covariance', False)
    tol = getattr(params, 'tol', 0.0001)
    return LDA(solver=solver, shrinkage=shrinkage, priors=priors, n_components=n_components, store_covariance=store_covariance, tol=tol)

def init_LGR(params):
    penalty = getattr(params, 'penalty', 'l2')
    dual = getattr(params, 'dual', False)
    tol = getattr(params, 'tol', 0.0001)
    C = getattr(params, 'C', 1.0)
    fit_intercept = getattr(params, 'fit_intercept', True)
    intercept_scaling = getattr(params, 'intercept_scaling', 1)
    class_weight = getattr(params, 'class_weight', None)
    random_state = getattr(params, 'random_state', None)
    solver = getattr(params, 'solver', 'liblinear')
    max_iter = getattr(params, 'max_iter', 100)
    multi_class = getattr(params, 'multi_class', 'ovr')
    verbose = getattr(params, 'verbose', 0)
    warm_start = getattr(params, 'warm_start', False)
    n_jobs = getattr(params, 'n_jobs', 1)
    return LGR(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

def init_SVM(params):
    C = getattr(params, 'C', 1.0)
    kernel = getattr(params, 'kernel', 'rbf')
    degree = getattr(params, 'degree', 3)
    gamma = getattr(params, 'gamma', 'auto')
    coef0 = getattr(params, 'coef0', 0.0)
    shrinking = getattr(params, 'shrinking', True)
    probability = getattr(params, 'probability', False)
    tol = getattr(params, 'tol', 0.001)
    cache_size = getattr(params, 'cache_size', 200)
    class_weight = getattr(params, 'class_weight', None)
    verbose = getattr(params, 'verbose', False)
    max_iter = getattr(params, 'max_iter', -1)
    decision_function_shape = getattr(params, 'decision_function_shape', 'ovr')
    random_state = getattr(params, 'random_state', None)
    return SVM(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, random_state=random_state)

def init_RFC(params):
    n_estimators = getattr(params, 'n_estimators', 10)
    criterion = getattr(params, 'criterion', 'gini')
    max_depth = getattr(params, 'max_depth', None)
    min_samples_split = getattr(params, 'min_samples_split', 2)
    min_samples_leaf = getattr(params, 'min_samples_leaf', 1)
    min_weight_fraction_leaf = getattr(params, 'min_weight_fraction_leaf', 0.0)
    max_features = getattr(params, 'max_features', 'auto')
    max_leaf_nodes = getattr(params, 'max_leaf_nodes', None)
    min_impurity_split = getattr(params, 'min_impurity_split', 1e-07)
    bootstrap = getattr(params, 'bootstrap', True)
    oob_score = getattr(params, 'oob_score', False)
    n_jobs = getattr(params, 'n_jobs', 1)
    random_state = getattr(params, 'random_state', None)
    verbose = getattr(params, 'verbose', 0)
    warm_start = getattr(params, 'warm_start', False)
    class_weight = getattr(params, 'class_weight', None)
    return RFC(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight)