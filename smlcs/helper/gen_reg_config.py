import json


def gen_reg_config():
    regressor_config = dict()
    regressor_config['regressors'] = []
    regressor_config['innercv_folds'] = 2

    rf_grid = {'n_estimators': [100, 200],
               'max_features': ['sqrt'],
               'max_depth': [10],
               'min_samples_split': [2],
               'min_samples_leaf': [1],
               'bootstrap': [True]}

    regressor_config['regressors'].append({
        'reg_name': 'rf',
        'reg_parameters': rf_grid
    })

    svr_grid = [{'kernel': ['rbf'], 'gamma': [0.01, 0.02],
                 'C': [0.1, 1]},
                {'kernel': ['linear'], 'C': [0.1, 1]}]

    regressor_config['regressors'].append({
        'reg_name': 'svr',
        'reg_parameters': svr_grid
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

    regressor_config['regressors'].append({
        'reg_name': 'gb',
        'reg_parameters': gb_grid
    })

    with open('../configurations/reg_config.txt', 'w') as outfile:
        json.dump(regressor_config, outfile)


gen_reg_config()
