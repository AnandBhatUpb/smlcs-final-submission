"""This script trains --reg regressor to predict timeout values.

Usage:
  regressor.py --env=<run_environment> --job=<jobid> --subjob=<subjobid> --reg=<regressor>
  regressor.py (-h | --help)
  regressor.py

Options:
  -h --help                             Show this screen.
  --env<run_environment>                specifies the running environment cluster/PC
  --job=<jobid>                         specifies cluster job id
  --subjob=<subjobid>                   specifies cluster subjob id
  --reg=<regressor>                    specifies classifier to train
"""

import datetime
import logging
import json
from docopt import docopt
import numpy as np

from smlcs.helper.read_data import ReadData
from smlcs.evaluation.metrics import Metrics
from smlcs.evaluation.plotters import Plotters
from smlcs.helper.write_training_result import WriteToCSV

from smlcs.helper.preprocessing import Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import svm, ensemble
from skopt import BayesSearchCV


class Regressor:

    def local_training(environment, reg, X, Y, outercv, logger):
        try:
            print('Not Implemented')
        except Exception as e:
            print('Error')
            #logger.error('Failed in local training: ' + str(e))

    def cluster_training(environment, reg, job_id, subjob_id, logger):
        try:
            logger.info('Training environment: {}'.format(environment))
            logger.info('Regressor selected: {}'.format(reg))

            with open('../configurations/outer_fold_data_reg.txt') as json_file:
                data = json.load(json_file)
            datasource = data['datasource']
            feature_names = data['feature_names']
            outer_split_strategy = data['outer_split_strategy']
            logger.info('Data source selected for training: {}'.format(datasource))

            X, Y, pgm_features= ReadData(datasource, logger).read_reg_data(logger)  # Read data from local/remote

            X[:, 0:42] = Preprocessing().handle_missing_data(X[:, 0:42], logger)  # Handle missing data

            onehotcoded_data, config_features = Preprocessing().encode_categorical_data(X[:, 42:51],
                                                                                        logger)  # OneHotCode categorical data
            feature_names = pgm_features + config_features
            X = np.delete(X, np.s_[42:51], axis=1)

            logger.info('Shape of the onehotcoded data: {}'.format(onehotcoded_data.shape))
            logger.info('Shape of the program feature data: {}'.format(X.shape))
            logger.info('Feature names after onehotencoding: {}'.format(feature_names))

            X = np.concatenate((X, onehotcoded_data), axis=1)

            logger.info('Shape of the final processed data: {}'.format(X.shape))

            for f in data['folds']:
                if int(subjob_id) == int(f['foldId']):
                    X_train, X_test = X[f['outer_train_index']], X[f['outer_test_index']]
                    y_train, y_test = Y[f['outer_train_index']], Y[f['outer_test_index']]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            with open('../configurations/reg_config.txt') as json_file:
                reg_config = json.load(json_file)
            innercvfolds = int(reg_config['innercv_folds'])
            logger.info('Inner cross validation number of folds: {}'.format(innercvfolds))

            estimator = None
            tuning_parameters = None
            for c in reg_config['regressors']:
                if reg == c['reg_name']:
                    if reg == 'rf':
                        estimator = ensemble.RandomForestRegressor()
                        tuning_parameters = c['reg_parameters']
                        break
                    else:
                        estimator = svm.SVR()
                        tuning_parameters = c['reg_parameters']
                        break

            logger.info('estimator is : {}'.format(estimator))
            logger.info('Tunning parameters are: {}'.format(tuning_parameters))

            start_time = datetime.datetime.now()
            logger.info('Started Skopt CV at: {}'.format(start_time))

            opt_reg = BayesSearchCV(estimator, tuning_parameters, cv=innercvfolds)
            opt_reg.fit(X_train, y_train)

            end_time = datetime.datetime.now()
            logger.info('Ended Skopt CV at: {}'.format(end_time))
            logger.info('Total time for parameter search: {}'.format(end_time-start_time))

            metrics = Metrics(opt_reg)
            metrics.grid_models_metrics(logger, job_id, subjob_id)
            best_params = metrics.grid_best_params(logger)
            best_estimator = metrics.grid_best_estimator(logger)
            grid_score = metrics.grid_score(logger)
            test_score = metrics.test_score(X_test, y_test, logger)
            important_features = []
            if reg == 'rf':
                important_features = metrics.get_imprtant_features(logger)

            plot = Plotters(opt_reg)
            plot.plot_residuals_plot(X_train, y_train, X_test, y_test, logger, job_id, subjob_id)
            fi_path = ''
            if reg == 'rf':
                plot.plot_feature_imp(feature_names, logger, job_id, subjob_id)
                fi_path = './plot/fi_' + str(job_id) + "_" + str(subjob_id) + '.png'

            log_path = './logs/log_'+str(job_id)+'_'+str(subjob_id)+'.log'
            res_path = './plot/res_'+str(job_id)+'_'+str(subjob_id)+'.png'

            # dump all results into the training_result.csv file
            writer = WriteToCSV()
            writer.write_result_to_csv(logger, job_id, subjob_id, subjob_id, datetime.datetime.now(), reg, best_params,
                                       grid_score, test_score, innercvfolds, outer_split_strategy, 'none', datasource,
                                       start_time, end_time, end_time-start_time, X_train.shape, X_test.shape,
                                       log_path, res_path, fi_path)

        except Exception as e:
            logger.error('Failed in cluster training: ' + str(e))

    if __name__ == '__main__':
        try:
            arguments = docopt(__doc__, version=None)
            if arguments['--env'] is None:
                environment = 'cluster'
            else:
                environment = arguments['--env']

            if arguments['--job'] is None:
                job_id = -1
            else:
                job_id = arguments['--job']

            if arguments['--subjob'] is None:
                subjob_id = -1
            else:
                subjob_id = arguments['--subjob']

            if arguments['--reg'] is None:
                reg = 'rf'
            else:
                reg = arguments['--reg']
            logging.basicConfig(filename='../../logs/log_' + str(job_id) + '_' + str(subjob_id) + '.log', filemode='w',
                                format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
            logger = logging.getLogger('Reg_training')
            logger.info('Cluster job ID: {}'.format(job_id))
            logger.info('Cluster sub job ID: {}'.format(subjob_id))

            cluster_training(environment, reg, job_id, subjob_id, logger)

        except Exception as e:
            logger.error('Failed in the main of regressor.py: ' + str(e))