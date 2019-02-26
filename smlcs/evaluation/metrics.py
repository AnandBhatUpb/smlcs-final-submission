class CalculateMetrics:

    def grid_models_metrics(self, logger, *argv):
        try:
            estimator = self.estimator
            means = estimator.cv_results_['mean_test_score']
            stds = estimator.cv_results_['std_test_score']
            grid_metric_file = open("../../results/grid_metric_"+str(argv[0])+"_"+str(argv[1])+".txt", "w+")
            logger.info('Grid  models metrics file : {}'.format(grid_metric_file))
            logger.info('Writing to: {}'.format(grid_metric_file))
            for mean, std, params in zip(means, stds, estimator.cv_results_['params']):
                grid_metric_file.write("%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params))
            logger.info('Writing completed')
        except Exception as e:
            logger.error('Failed in grid_model_metrics: :' + str(e))

    def grid_best_params(self, logger):
        try:
            best_params = self.estimator.best_params_
            logger.info('Best parameters: {}'.format(best_params))
            return best_params
        except Exception as e:
            logger.error('Failed in grid_best_params:' + str(e))

    def grid_best_estimator(self, logger):
        try:
            best_estimator = self.estimator.best_estimator_
            logger.info('Best estimator: {}'.format(best_estimator))
            return best_estimator
        except Exception as e:
            logger.error('Failed in grid_best_estimator:' + str(e))

    def grid_score(self, logger):
        try:
            best_score = self.estimator.best_score_
            logger.info('Best score in the grid: {}'.format(best_score))
            return best_score
        except Exception as e:
            logger.error('Failed in grid_score:' + str(e))

    def test_score(self, x_test, y_test, logger):
        try:
            test_score = self.estimator.score(x_test, y_test)
            logger.info('Test score: {}'.format(test_score))
            return test_score
        except Exception as e:
            logger.error('Failed in test_score:' + str(e))

    def get_imprtant_features(self, logger):
        try:
            imp_features = self.estimator.best_estimator_.feature_importances_
            logger.info('Important features: {}'.format(imp_features))
            return imp_features
        except Exception as e:
            logger.error('Failed in get_imprtant_features:' + str(e))

    def __init__(self, estimator):
        self.estimator = estimator
