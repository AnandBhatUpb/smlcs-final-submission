import json


def gen_clf_config():
    classifier_config = dict()
    classifier_config['classifiers'] = []
    classifier_config['innercv_folds'] = 3

    rf_grid = {'n_estimators': (100, 1000),
               'max_features': (4, 13),
               'max_depth': (10, 100),
               'min_samples_split': (2, 50),
               'min_samples_leaf': (1, 10),
               'bootstrap': [True, False]
               }

    classifier_config['classifiers'].append({
        'clf_name': 'rf',
        'clf_parameters': rf_grid
    })

    svc_grid = {
        'C': (1e-4, 1e+2, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8),  # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    }

    classifier_config['classifiers'].append({
        'clf_name': 'svc',
        'clf_parameters': svc_grid
    })

    gb_grid = {
        'learning_rate': (0.01, 0.2),
        'n_estimators': (100, 2000),
        'min_samples_split': (2, 50),
        'min_samples_leaf': (1, 10),
        'max_features': (4, 13),
        'subsample': (0.6, 1.0),
        'max_depth': (3, 15)
    }

    classifier_config['classifiers'].append({
        'clf_name': 'gb',
        'clf_parameters': gb_grid
    })

    with open('../configurations/clf_config.txt', 'w') as outfile:
        json.dump(classifier_config, outfile)


gen_clf_config()
