import numpy as np
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt


class PlotResults:

    def plot_confusion_matrix(self, x_test, y_test, logger, *argv):
        try:
            estimator = self.estimator.best_estimator_
            cm = confusion_matrix(y_target=y_test, y_predicted=estimator.predict(x_test), binary=False)
            fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(15, 15))
            plt.savefig('../../plots/cm_'+str(argv[0])+"_"+str(argv[1])+'.png')
        except Exception as e:
            logger.error('Failed in plot_confusion_matrix:' + str(e))

    def plot_residuals_plot(self, x_train, y_train, x_test, y_test, logger, *argv):
        try:
            estimator = self.estimator.best_estimator_
            f = plt.figure(figsize=(25, 10))
            plt.scatter(estimator.predict(x_test), y_test - estimator.predict(x_test), color='g', s=40, alpha=0.5)
            # plt.scatter(lm.predict(X_test), y_test  , color='g', s = 40, alpha=0.5)
            plt.ylabel('Residual')
            plt.title('Residual Vs predicted')
            plt.savefig('../../plots/res_'+str(argv[0])+"_"+str(argv[1])+'.png')
        except Exception as e:
            logger.error('Failed in plot_residuals_plot:' + str(e))

    def plot_feature_imp(self, feature_names, logger, *argv):
        try:
            fig, ax = plt.subplots(figsize=(18, 25))
            features = feature_names
            importances = self.estimator.best_estimator_.feature_importances_
            indices = np.argsort(importances)
            plt.title('Feature Importance')
            plt.barh(range(len(indices)), importances[indices], color='g', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.savefig('../../plots/fi_'+str(argv[0])+"_"+str(argv[1])+'.png')
        except Exception as e:
            logger.error('Failed in plot_feature_imp:' + str(e))

    def __init__(self, estimator):
        self.estimator = estimator
