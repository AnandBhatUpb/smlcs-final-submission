import json
import logging
import numpy as np
from smlcs.helper.read_data import ReadData
from joblib import load
from smlcs.evaluation.plotters import PlotResults
from smlcs.helper.preprocessing import Preprocessing


def get_data_estimator():
    try:
        logging.basicConfig(filename='../../logs/log_analyse.log', filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        logger = logging.getLogger('analyse')
        with open('../configurations/outer_fold_data_clf.txt') as json_file:
            data = json.load(json_file)

        X, Y, pgm_features = ReadData('local', logger).read_clf_data(logger)  # Read data from local/remote
        # Change here to load imputer
        X[:, 0:42], imp = Preprocessing().handle_missing_data(X[:, 0:42], logger)  # Handle missing data

        onehotcoded_data, config_features = Preprocessing().encode_categorical_data(X[:, 42:51],
                                                                                    logger)  # OneHotCode categorical data
        feature_names = pgm_features + config_features
        X = np.delete(X, np.s_[42:51], axis=1)
        logger.info('Shape of the onehotcoded data: {}'.format(onehotcoded_data.shape))
        logger.info('Shape of the program feature data: {}'.format(X.shape))
        logger.info('Feature names after onehotencoding: {}'.format(feature_names))

        X = np.concatenate((X, onehotcoded_data), axis=1)

        logger.info('Shape of the final processed data: {}'.format(X.shape))
        Y = Preprocessing().encode_labels(Y, logger)  # Encoding class labels

        for f in data['folds']:
            if 5 == int(f['foldId']):
                X_train, X_test = X[f['outer_train_index']], X[f['outer_test_index']]
                y_train, y_test = Y[f['outer_train_index']], Y[f['outer_test_index']]

        scalar = load('PATH TO saved scalar')
        X_train = scalar.transform(X_train)
        X_test = scalar.transform(X_test)

        learner = load('path to saved model')
        #learner = est.best_estimator_
        #return X_train, X_test, y_train, y_test, learner, feature_names, logger
        return learner, feature_names, logger

    except Exception as e:
        logger.error('Failed in cluster training: ' + str(e))


def plot_confusion_matrix():
    X_train, X_test, y_train, y_test, learner, feature_names, logger = get_data_estimator()
    plot = PlotResults(learner)
    plot.plot_confusion_matrix(X_test, y_test, logger, 1684718, 5)


def plot_feature_imp():
    learner, feature_names, logger = get_data_estimator()
    plot = PlotResults(learner)
    plot.plot_feature_imp(feature_names, logger, 7076778, 5)


#plot_confusion_matrix()
plot_feature_imp()
