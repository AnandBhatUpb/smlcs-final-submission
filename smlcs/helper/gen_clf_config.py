import json


def gen_clf_config():
    classifier_config = dict()
    classifier_config['classifiers'] = []
    classifier_config['innercv_folds'] = 2

    rf_grid = {'n_estimators': (100, 102),
               'bootstrap': [True, False]
               }

    classifier_config['classifiers'].append({
        'clf_name': 'rf',
        'clf_parameters': rf_grid
    })

    svc_grid = {
        'C': (1e-1, 1, 'log-uniform')
    }

    classifier_config['classifiers'].append({
        'clf_name': 'svc',
        'clf_parameters': svc_grid
    })

    gb_grid = {
        'n_estimators': (100, 102)
    }

    classifier_config['classifiers'].append({
        'clf_name': 'gb',
        'clf_parameters': gb_grid
    })

    with open('../configurations/clf_config.txt', 'w') as outfile:
        json.dump(classifier_config, outfile)


gen_clf_config()
