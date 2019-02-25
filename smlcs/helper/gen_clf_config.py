import json


def gen_clf_config():
    classifier_config = dict()
    classifier_config['classifiers'] = []
    classifier_config['innercv_folds'] = 3
    classifier_config['class_weight'] = 'balanced'

    rf_grid = {'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
               'min_samples_split': [2, 5, 10, 20, 50],
               'min_samples_leaf': [1, 2, 4, 6, 8, 10],
               'bootstrap': [True, False]}

    rf_grid_test = {'n_estimators': [500, 100],
                    'max_features': ['auto'],
                    'max_depth': [10],
                    'min_samples_split': [2],
                    'min_samples_leaf': [1],
                    'bootstrap': [True]}

    classifier_config['classifiers'].append({
        'clf_name': 'rf',
        'clf_parameters': rf_grid_test
    })

    svc_grid = [{'kernel': ['rbf'], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
                 'C': [0.01, 0.1, 1, 10, 100, 1000]},
                {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]

    classifier_config['classifiers'].append({
        'clf_name': 'svc',
        'clf_parameters': svc_grid
    })

    xgb_grid = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'objective': ['multi:softprob'],
        'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'min_child_weight': [1, 3, 5, 7, 10],
        'gamma': [0.0, 0.1, 0.2 , 0.3, 0.4, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7],
        'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
        'eval_metric': "merror"
    }

    classifier_config['classifiers'].append({
        'clf_name': 'xgb',
        'clf_parameters': xgb_grid
    })

    with open('../configurations/clf_config.txt', 'w') as outfile:
        json.dump(classifier_config, outfile)


gen_clf_config()
