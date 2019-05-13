import pandas as pd
import numpy as np
from smlcs.helper.preprocessing import Preprocessing
from joblib import load


class Predictor:

    def predictor(self, path):
        logger = self.logger
        program_feature = pd.read_csv(path)
        program_list = list(program_feature['programs'])
        program_feature = program_feature.drop(['programs', 'THREAD_DESCR', 'FILE_DESCR', 'CHAR'], axis=1)

        predicate_feature = pd.read_csv('../dataset/configuration_feature/predicate_encoding.csv')
        predicate_config_list = list(predicate_feature['configurations'].values)
        predicate_feature = predicate_feature.drop(['configurations'], axis=1)

        value_feature = pd.read_csv('../dataset/configuration_feature/value_encoding.csv')
        value_config_list = list(value_feature['configurations'].values)
        value_feature = value_feature.drop(['configurations'], axis=1)

        bmc_feature = pd.read_csv('../dataset/configuration_feature/bmc_encoding.csv')
        bmc_config_list = list(bmc_feature['configurations'].values)
        bmc_feature = bmc_feature.drop(['configurations'], axis=1)

        configuration_list = predicate_config_list + value_config_list + bmc_config_list

        program_feature_names = program_feature.columns.values
        config_feature_names = predicate_feature.columns.values
        X = np.array([np.array(np.concatenate((program_feature_names, config_feature_names), axis=0))])
        for config_type in [predicate_feature, value_feature, bmc_feature]:
            for i in range(0, len(program_feature.values)):
                for j in range(0, len(config_type.values)):
                    y = np.concatenate((program_feature.values[i], config_type.values[j]), axis=0)
                    y = np.array([np.array(y)])
                    X = np.append(X, y, axis=0)

        X_data = np.delete(X, 0, 0)
        X_data[:, 0:42], imp = Preprocessing().handle_missing_data(X_data[:, 0:42], logger)
        onehotcoded_data, config_features = Preprocessing().encode_categorical_data(X_data[:, 42:51],
                                                                                    logger)
        feature_names = list(program_feature_names) + config_features
        X_data = np.delete(X_data, np.s_[42:51], axis=1)
        X_data = np.concatenate((X_data, onehotcoded_data), axis=1)

        scalar = load('../../models_persisted/clf_scalar_gb_7076798_1.joblib')
        X_data = scalar.transform(X_data)
        classifier = load('../../models_persisted/clf_gb_7076798_1.joblib')
        classifier = classifier.best_estimator_
        prediction = classifier.predict_proba(X_data)

        count = 0
        j = 0
        predict_count = []
        config_count = []
        program_count = []
        for i in range(0, len(prediction)):
            if count % 82 == 0:
                program = program_list[j]
                j += 1
                count = 0
            configuration = configuration_list[count]
            if prediction[i][0] > 0.80:
                #print('{} -- {} -- {}'.format(program, configuration, prediction[i][0]))
                predict_count.append(i)
                config_count.append(configuration)
                program_count.append(program)
            count += 1

        class_prediction = []
        for i in predict_count:
            class_prediction.append(prediction[i])

        regressor = load('../../models_persisted/reg_gb_7076756_1.joblib')
        regressor = regressor.best_estimator_
        runtime_prediction = []
        for i in predict_count:
            runtime_prediction.append(regressor.predict(X_data[i, :].reshape(1, -1)))

        #print(runtime_prediction)
        all_rtime =  regressor.predict(X_data)
        return program_list, program_count, config_count, class_prediction, runtime_prediction, prediction, all_rtime

    def __init__(self, log):
        self.logger = log
