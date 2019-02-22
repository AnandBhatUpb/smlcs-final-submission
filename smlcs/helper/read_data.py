import pandas as pd


class ReadData:

    def read_clf_data(self, logger):
        try:
            X = self.clf_dataset.iloc[:, :-1].values
            Y = self.clf_dataset.iloc[:, 51].values
            return X, Y, list(self.clf_dataset.iloc[:, :-10])
        except Exception as e:
            logger.error('Failed to read classification data: ' + str(e))
            return None, None

    def read_reg_data(self, logger):
        try:
            X = self.reg_dataset.iloc[:, :-1].values
            Y = self.reg_dataset.iloc[:, 51].values
            return X, Y, list(self.reg_dataset.iloc[:, :-10])
        except Exception as e:
            logger.error('Failed to read regression data: ' + str(e))
            return None, None

    def read_dataframe(self, logger):
        try:
            if self.source == 'local':
                clf_dataset = pd.read_csv('../dataset/clf_data.csv')
                reg_dataset = pd.read_csv('../dataset/reg_data_75.csv')
            elif self.source == 'remote':
                clf_url = "https://raw.github.com/AnandBhatUpb/CPAchecker_dataset/master/clf_data.csv"
                reg_url = "https://raw.github.com/AnandBhatUpb/CPAchecker_dataset/master/reg_data_75.csv"
                clf_dataset = pd.read_csv(clf_url)
                reg_dataset = pd.read_csv(reg_url)
            else:
                logger.error('Wrong argument value for source parameter{}'.format(self.source))

            return clf_dataset, reg_dataset
        except Exception as e:
            logger.error('Failed to read dataframe: ' + str(e))
            return None, None

    def __init__(self, source, logger):
        try:
            self.source = source
            logger.info('Dataset set is reading...')
            if source == 'local':
                self.clf_dataset = pd.read_csv('../dataset/clf_data.csv')
                self.clf_dataset = self.clf_dataset.drop(['THREAD_DESCR', 'FILE_DESCR', 'CHAR'], axis=1)

                self.reg_dataset = pd.read_csv('../dataset/reg_data_75.csv')
                self.reg_dataset = self.reg_dataset.drop(['THREAD_DESCR', 'FILE_DESCR', 'CHAR'], axis=1)
                logger.info('Dataset shape classification: {}'.format(self.clf_dataset.shape))
                logger.info('Dataset shape regression: {}'.format(self.reg_dataset.shape))
            elif source == 'remote':
                clf_url = "https://raw.github.com/AnandBhatUpb/CPAchecker_dataset/master/clf_data.csv"
                reg_url = "https://raw.github.com/AnandBhatUpb/CPAchecker_dataset/master/reg_data_75.csv"
                self.clf_dataset = pd.read_csv(clf_url)
                self.clf_dataset = self.clf_dataset.drop(['THREAD_DESCR', 'FILE_DESCR', 'CHAR'], axis=1)

                self.reg_dataset = pd.read_csv(reg_url)
                self.reg_dataset = self.reg_dataset.drop(['THREAD_DESCR', 'FILE_DESCR', 'CHAR'], axis=1)
                logger.info('Dataset shape classification: {}'.format(self.clf_dataset.shape))
                logger.info('Dataset shape regression: {}'.format(self.reg_dataset.shape))
            else:
                logger.error('Wrong argument value for --datasource {}'.format(source))
        except Exception as e:
            logger.error('Failed to read dataset: ' + str(e))


