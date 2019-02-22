import json


def gen_clf_config():
    classifier_config = dict()
    classifier_config['classifiers'] = []
    classifier_config['innercv_folds'] = 3
    classifier_config['class_weight'] = 'balanced'

    rf_grid = {'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}

    classifier_config['classifiers'].append({
        'clf_name': 'rf',
        'clf_parameters': rf_grid
    })

    svc_grid = [{'kernel': ['rbf'], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
                 'C': [0.01, 0.1, 1, 10, 100, 1000]},
                {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]

    classifier_config['classifiers'].append({
        'clf_name': 'svc',
        'clf_parameters': svc_grid
    })

    with open('../configurations/clf_config.txt', 'w') as outfile:
        json.dump(classifier_config, outfile)


gen_clf_config()
