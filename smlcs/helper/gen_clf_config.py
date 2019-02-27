import json


def gen_clf_config():
    classifier_config = dict()
    classifier_config['classifiers'] = []
    classifier_config['innercv_folds'] = 2
    classifier_config['class_weight'] = 'balanced'

    rf_grid = {'n_estimators': [100, 200],
                    'max_features': ['sqrt'],
                    'max_depth': [10],
                    'min_samples_split': [2],
                    'min_samples_leaf': [1],
                    'bootstrap': [True]}

    classifier_config['classifiers'].append({
        'clf_name': 'rf',
        'clf_parameters': rf_grid
    })

    svc_grid = [{'kernel': ['rbf'], 'gamma': [0.01, 0.02],
                 'C': [0.1, 1]},
                {'kernel': ['linear'], 'C': [0.1, 1]}]

    classifier_config['classifiers'].append({
        'clf_name': 'svc',
        'clf_parameters': svc_grid
    })

    gb_grid = {
        'learning_rate': [0.01, 0.03],
        'n_estimators': [100, 200],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 3],
        'max_features': [5, 9],
        'subsample': [0.7, 0.75],
        'max_depth': [3, 4]
    }

    classifier_config['classifiers'].append({
        'clf_name': 'gb',
        'clf_parameters': gb_grid
    })

    with open('../configurations/clf_config.txt', 'w') as outfile:
        json.dump(classifier_config, outfile)


gen_clf_config()
