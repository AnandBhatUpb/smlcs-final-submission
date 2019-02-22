"""This script trains --clf classifier for multiclass classification.

Usage:
  classifier.py --env=<run_environment> --job=<jobid> --subjob=<subjobid> --clf=<classifier>
  classifier.py (-h | --help)
  classifier.py

Options:
  -h --help                             Show this screen.
  --env<run_environment>                specifies the running environment cluster/PC
  --job=<jobid>                         specifies cluster job id
  --subjob=<subjobid>                   specifies cluster subjob id
  --clf=<classifier>                    specifies classifier to train
"""

import datetime
import logging
import json
from docopt import docopt

from smlcs.helper.read_data import ReadData
from smlcs.evaluation.metrics import Metrics
from smlcs.evaluation.plotters import Plotters
from smlcs.helper.write_training_result import WriteToCSV

from sklearn import svm, ensemble
from skopt import BayesSearchCV


class Classifier:

    def local_training(self, environment, clf, X, Y, outercv, logger):
        try:
            print('Not Implemented')
        except Exception as e:
            print('Error')
            #logger.error('Failed in local training: ' + str(e))

    def cluster_training(self, environment, clf, job_id, subjob_id, logger):
        try:
            logger.info('Training environment: {}'.format(environment))
            logger.info('Classifier selected: {}'.format(clf))

            with open('../configurations/outer_fold_data_clf.txt') as json_file:
                data = json.load(json_file)
            datasource = data['datasource']
            feature_names = data['feature_names']
            outer_split_strategy = data['outer_split_strategy']
            logger.info('Data source selected for training: {}'.format(datasource))

            X, Y = ReadData(datasource, logger).read_clf_data(logger)  # Read data from local/remote

            for f in data['folds']:
                if int(subjob_id) == int(f['foldId']):
                    X_train, X_test = X[f['outer_train_index']], X[f['outer_test_index']]
                    y_train, y_test = Y[f['outer_train_index']], Y[f['outer_test_index']]

            with open('../configurations/clf_config.txt') as json_file:
                clf_config = json.load(json_file)
            innercvfolds = int(clf_config['innercv_folds'])
            class_weight = str(clf_config['class_weight'])
            logger.info('Inner cross validation number of folds: {}'.format(innercvfolds))

            for c in clf_config['classifiers']:
                if clf == c['clf_name']:
                    if clf == 'rf':
                        estimator = ensemble.RandomForestClassifier(class_weight=class_weight)
                    else:
                        estimator = svm.SVC(class_weight=class_weight)

                    tuning_parameters = c['clf_parameters']

            logger.info('estimator is : {}'.format(estimator))
            logger.info('Tunning parameters are: {}'.format(tuning_parameters))

            start_time = datetime.datetime.now()
            logger.info('Started Skopt CV at: {}'.format(start_time))

            opt_clf = BayesSearchCV(estimator, tuning_parameters, cv=innercvfolds)
            opt_clf.fit(X_train, y_train)

            end_time = datetime.datetime.now()
            logger.info('Ended Skopt CV at: {}'.format(end_time))
            logger.info('Total time for parameter search: {}'.format(end_time-start_time))

            metrics = Metrics(opt_clf)
            metrics.grid_models_metrics(logger, job_id, subjob_id)
            best_params = metrics.grid_best_params(logger)
            best_estimator = metrics.grid_best_estimator(logger)
            grid_score = metrics.grid_score(logger)
            test_score = metrics.test_score(X_test, y_test, logger)
            if clf == 'rf':
                important_features = metrics.get_imprtant_features(logger)

            plot = Plotters(opt_clf)
            plot.plot_confusion_matrix(X_test, y_test, logger, job_id, subjob_id)
            if clf == 'rf':
                plot.plot_feature_imp(feature_names, logger, job_id, subjob_id)
                fi_path = './plot/fi_' + str(job_id) + "_" + str(subjob_id) + '.png'

            log_path = './logs/log_'+str(job_id)+'_'+str(subjob_id)+'.log'
            cm_path = './plot/cm_'+str(job_id)+'_'+str(subjob_id)+'.png'

            # dump all results into the training_result.csv file
            writer = WriteToCSV()
            writer.write_result_to_csv(logger, job_id, subjob_id, subjob_id, datetime.datetime.now(), clf, best_params,
                                       grid_score, test_score, innercvfolds, outer_split_strategy, 'none', datasource,
                                       start_time, end_time, end_time-start_time, X_train.shape, X_test.shape,
                                       log_path, cm_path, fi_path)

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
                job_id = int(arguments['--jobid'])

            if arguments['--subjobid'] is None:
                subjob_id = -1
            else:
                subjob_id = int(arguments['--subjobid'])

            if arguments['--clf'] is None:
                clf = 'rf'
            else:
                clf = arguments['--clf']
            logging.basicConfig(filename='../../logs/log_' + str(job_id) + '_' + str(subjob_id) + '.log', filemode='w',
                                format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
            logger = logging.getLogger('Clf_training')
            logger.info('Cluster job ID: {}'.format(job_id))
            logger.info('Cluster sub job ID: {}'.format(subjob_id))

            cluster_training(environment, clf, job_id, subjob_id, logger)

        except Exception as e:
            logger.error('Failed in the main of classifier.py: ' + str(e))

