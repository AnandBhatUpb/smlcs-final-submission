"""This script prepares dataset folds for regression and takes configurations values form the command line arguments.

Usage:
  regression.py --env=<run_environment> --datasource=<source> --outercvsplit=<split_strategy> --outercvfolds=<folds>
  regression.py (-h | --help)
  regression.py

Options:
  -h --help                             Show this screen.
  --env<run_environment>                specifies the running environment cluster/PC
  --datasource=<source>                 source of the dataset from which dataset to be read.
  --outercvsplit=<split_strategy>       data splitting strategy for cross validation.
  --outercvfolds=<folds>                number of folds for outer cross validation
"""

import logging
import datetime
from docopt import docopt
import numpy as np
import json

from smlcs.helper.read_data import ReadData
from smlcs.helper.preprocessing import Preprocessing
from smlcs.training.regressor import Regressor

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


class RegressionData:

    def prepare_regression_data(X, outercv, datasource, feature_names, split_strategy, logger):
        try:
            fold_count = 1
            outer_fold_data = dict()
            outer_fold_data['folds'] = []
            outer_fold_data['datasource'] = datasource
            outer_fold_data['feature_names'] = feature_names
            outer_fold_data['outer_split_strategy'] = split_strategy
            for outer_train_index, outer_test_index in outercv.split(X):
                outer_fold_data['folds'].append({
                    'foldId': fold_count,
                    'outer_train_index': outer_train_index.tolist(),
                    'outer_test_index': outer_test_index.tolist()
                })
                fold_count += 1

            with open('../configurations/outer_fold_data_reg.txt', 'w') as outfile:
                json.dump(outer_fold_data, outfile)
        except Exception as e:
            logger.error('Failed to create outer fold data: ' + str(e))

    if __name__ == '__main__':
        try:
            logging.basicConfig(filename='../../logs/reg_data_preparation.log', filemode='w',
                                format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

            arguments = docopt(__doc__, version=None)
            if arguments['--datasource'] is None:
                datasource = 'local'
            else:
                datasource = arguments['--datasource']

            if arguments['--outercvsplit'] is None:
                split_strategy = 'kfold'
            else:
                split_strategy = arguments['--outercvsplit']

            if arguments['--outercvfolds'] is None:
                folds = 5
            else:
                folds = int(arguments['--outercvfolds'])

            if arguments['--env'] is None:
                environment = 'pc'
            else:
                environment = arguments['--env']

            if environment == 'pc':
                logger = logging.getLogger('Experiment')
            else:
                logger = logging.getLogger('Data_preparation')

            start_time = datetime.datetime.now()
            logger.info('Started at : {}'.format(start_time))

            logger.info('Data source selected: {}'.format(datasource))
            logger.info('Split strategy selected: {}'.format(split_strategy))
            logger.info('Number of outer CV folds: {}'.format(folds))
            logger.info('Experiment environment: {}'.format(environment))

            X, Y, pgm_features = ReadData(datasource, logger).read_reg_data(logger)       # Read data from local/remote

            X[:, 0:42] = Preprocessing().handle_missing_data(X[:, 0:42], logger)    # Handle missing data

            onehotcoded_data, config_features = Preprocessing().encode_categorical_data(X[:, 42:51], logger)    # OneHotCode categorical data
            feature_names = pgm_features + config_features
            X = np.delete(X, np.s_[42:51], axis=1)

            logger.info('Shape of the onehotcoded data: {}'.format(onehotcoded_data.shape))
            logger.info('Shape of the program feature data: {}'.format(X.shape))
            logger.info('Feature names after onehotencoding: {}'.format(feature_names))

            X = np.concatenate((X, onehotcoded_data), axis=1)

            logger.info('Shape of the final processed data: {}'.format(X.shape))

            if split_strategy == 'shuffle':
                outercv = ShuffleSplit(n_splits=folds, test_size=0.25, random_state=0)
            elif split_strategy == 'kfold':
                outercv = KFold(n_splits=folds, shuffle=True, random_state=0)
            elif split_strategy == 'stratified':
                outercv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
            else:
                outercv = None
                logger.info('Wrong argument value for --outercvsplit {}'.format(split_strategy))

            if environment == 'cluster':
                prepare_regression_data(X, outercv, datasource, feature_names, split_strategy, logger)
                end_time = datetime.datetime.now()
                logger.info('Ended at : {}'.format(end_time))
                logger.info('Total time spent : {}'.format(end_time-start_time))
            elif environment == 'pc':
                p = Regressor().local_training(environment, 'rf', X, Y, outercv, logger)      # Training moddule with default classifier RF
                end_time = datetime.datetime.now()
                logger.info('Ended at : {}'.format(end_time))
                logger.info('Total time spent : {}'.format(end_time - start_time))
            else:
                logger.info('Wrong argument value for --env {}'.format(environment))

        except Exception as e:
            logger.error('Failed in the main of regression.py: ' + str(e))

