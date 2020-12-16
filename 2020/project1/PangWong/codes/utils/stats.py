import time
import numpy as np
import scipy.linalg as la
import importlib

PINV_THRESHOLD = 1e-5


def load_classifiers(opt):
    cuml_lib = 'cuml' if opt.use_cuml else 'sklearn'
    thundersvm_lib = 'thundersvm' if opt.use_thundersvm else 'sklearn.svm'

    libs = {
        'log_reg': cuml_lib + '.linear_model',
        'svm': thundersvm_lib,
        'lda': 'sklearn.discriminant_analysis',
        'rand_forest': cuml_lib + '.ensemble'
    }

    modules = {k: importlib.import_module(libs[k]) for k in libs.keys()}
    log_reg_solver = 'qn' if cuml_lib == 'cuml' else 'saga'

    classifiers = {}
    if 'log_reg' in opt.classifiers:
        classifiers['log_reg'] = modules['log_reg'].LogisticRegression(solver=log_reg_solver, max_iter=1000)
    if 'svm' in opt.classifiers:
        # LinearSVC uses less time than SVC for sklearn
        classifiers['svm'] = (modules['svm'].SVC() if opt.use_thundersvm else modules['svm'].LinearSVC())
    if 'lda' in opt.classifiers:
        classifiers['lda'] = modules['lda'].LinearDiscriminantAnalysis()
    if 'rand_forest' in opt.classifiers:
        classifiers['rand_forest'] = modules['rand_forest'].RandomForestClassifier()

    return classifiers


def run_classifiers(classifiers, train_data, test_data):
    train_features, train_labels = train_data
    test_features, test_labels = test_data

    accs = {}

    for k in classifiers.keys():
        start_time = time.time()
        print('Running classifier "{}"'.format(k))

        classifier = classifiers[k].fit(train_features, train_labels)
        test_preds = classifier.predict(test_features)
        acc = get_accuracy(test_preds, test_labels)
        print('Accuracy: {:.3%}, time elapsed: {:.3f}'.format(acc, time.time() - start_time))
        accs[k] = acc

    return accs


def get_accuracy(preds, labels):
    acc = np.sum(preds == labels) / len(labels)
    return acc


def get_stats(data):
    features, labels = data
    D = features.shape[1]
    K = len(np.unique(labels))
    start_time = time.time()

    global_mean = np.mean(features, axis=0)
    class_means = np.zeros((K, D))

    for c in range(K):
        class_features = features[labels == c]
        class_means[c] = np.mean(class_features, axis=0)

    class_means_centered = class_means - global_mean
    between_cov = np.matmul(class_means_centered.T, class_means_centered) / K

    features_centered = features - np.take(class_means, labels, axis=0)
    within_cov = np.matmul(features_centered.T, features_centered) / len(labels)

    stats = {
        'global_mean': global_mean,
        'class_means': class_means,
        'between_class_cov': between_cov,
        'within_class_cov': within_cov
    }

    stats_finished_time = time.time()
    print('Calculated statistics in {:.3f}s'.format(stats_finished_time - start_time))

    between_cov_inv = la.pinv2(between_cov, cond=1e-5)
    within_cov_dispersion = np.trace(np.matmul(within_cov, between_cov_inv)) / K

    class_means_norms = np.linalg.norm(class_means_centered, axis=1)
    equinorm_measure = np.std(class_means_norms) / np.mean(class_means_norms)

    class_means_normalized = class_means_centered / class_means_norms.reshape((-1, 1))
    class_means_cov = np.matmul(class_means_normalized, class_means_normalized.T)
    class_means_cov = np.ma.masked_array(class_means_cov, mask=np.tril(np.ones((K, K))))
    equiangle_measure = class_means_cov.std()

    class_means_cov = np.abs(class_means_cov + 1/(K-1))
    max_angle_measure = class_means_cov.mean()

    measures = {
        'within_cov': np.log10(within_cov_dispersion),
        'equinorm': equinorm_measure,
        'equiangle': equiangle_measure,
        'max_angle': max_angle_measure
    }

    measures_finished_time = time.time()
    print('Calculated measurements in {:.3f}s'.format(measures_finished_time - stats_finished_time))

    return stats, measures
