import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Preprocessing:
    def handle_missing_data(self, X, logger):
        try:
            imputer = SimpleImputer(np.nan, strategy='mean')
            impute = imputer.fit(X)
            return impute.transform(X)
        except Exception as e:
            logger.error('Failed to handle missing data: ' + str(e))
            return None

    def encode_categorical_data(self, X, logger):
        try:
            onehotencoder = OneHotEncoder()
            ohe = onehotencoder.fit_transform(X.astype(str)).toarray()
            logger.info('Onehot encoded feature names: {}'.format(onehotencoder.get_feature_names()))
            return ohe, list(onehotencoder.get_feature_names())
        except Exception as e:
            logger.error('Failed to onehotcode the data: ' + str(e))
            return None

    def encode_labels(self, Y, logger):
        try:
            labelencoder_y = LabelEncoder()
            y = labelencoder_y.fit_transform(Y)
            logger.info('Encoded classes are : {}'.format(labelencoder_y.classes_))
            return y
        except Exception as e:
            logger.error('Failed to encode class labels: ' + str(e))
            return None
