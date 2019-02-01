from smlcs.helper.read_data import ReadData
from smlcs.helper.preprocessing import Preprocessing

import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import ensemble
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

class ClassificationTraining:

    def calc_metrics(self, X_train, y_train, X_val, y_val, model):
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        validation_score = model.score(X_val, y_val)
        return train_score, validation_score

    def calc_test_score(self, X_test, y_test, model):
        score = model.score(X_test, y_test)
        return score

    def confusion_matrix_plotting(self, X_test, y_test, model, model_name):
        cm = confusion_matrix(y_target=y_test, y_predicted=model.predict(X_test), binary=False)
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        plt.savefig( model_name + '.png')
        plt.show()


    def train_model(self):
        X, Y = ReadData().read_clf_data()   # Read dataset
        X.iloc[:, 0:41] = Preprocessing().handle_missing_data(X.iloc[:, 0:41])    # Handle missing data
        df = Preprocessing().encode_categorical_data(X.iloc[:, 41:50])
        X = X.drop(columns=list(X.iloc[:, 41:50]))
        X = pd.concat([X, df], axis=1)
        Y = Preprocessing().encode_labels(Y)

        # scaling
        scaler = pp.MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X.values)

        # Split into training and test set
        X_fold, X_test, y_fold, y_test = train_test_split(X, Y, test_size=0.40, random_state=1)

        # 10K fold
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        random_forest_clf = ensemble.RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2,
                                                            random_state=0)

        train_scores = []
        validation_scores = []
        for train_index, val_index in kf.split(X_fold, y_fold):
            # split data
            X_train, X_val = X_fold[train_index], X_fold[val_index]
            y_train, y_val = y_fold[train_index], y_fold[val_index]

            # calculate scores
            train_score, val_score = self.calc_metrics(X_train, y_train, X_val, y_val, random_forest_clf)
            train_scores.append(train_score)
            validation_scores.append(val_score)

        print('Learner -> RF')

        # Score
        print('Training Score -> ', round(np.mean(train_scores), 4))
        print('Validation Score -> ', round(np.mean(validation_scores), 4))
        print('Test Score -> ', round(self.calc_test_score(X_test, y_test, random_forest_clf), 4))
        print('co_efficients: ', random_forest_clf.feature_importances_)
        # Draw residual plot
        self.confusion_matrix_plotting(X_test, y_test, random_forest_clf, 'RF')

