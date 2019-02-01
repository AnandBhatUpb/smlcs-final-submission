import pandas as pd


class ReadData:

    def read_clf_data(self):
        X = self.clf_dataset.iloc[:, :-1]
        Y = self.clf_dataset.iloc[:, 50]
        return X, Y

    def read_reg_data(self):
        X = self.reg_dataset.iloc[:, :-1].values
        Y = self.reg_dataset.iloc[:, 50].values
        return X, Y

    def __init__(self):
        self.clf_dataset = pd.read_csv('../dataset/classification_data.csv')
        self.clf_dataset = self.clf_dataset.drop(['THREAD_DESCR', 'RECURSIVE_FUNC', 'FILE_DESCR', 'CHAR'], axis=1)

        self.reg_dataset = pd.read_csv('../dataset/regression_data.csv')
        self.reg_dataset = self.reg_dataset.drop(['THREAD_DESCR', 'RECURSIVE_FUNC', 'FILE_DESCR', 'CHAR'], axis=1)
