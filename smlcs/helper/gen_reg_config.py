import json


def gen_reg_config():
    regressor_config = dict()
    regressor_config['regressors'] = []
    regressor_config['innercv_folds'] = 3

    rf_grid = {'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
               'max_features': [4, 5, 7, 9, 10, 13],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
               'min_samples_split': [2, 5, 10, 20, 50],
               'min_samples_leaf': [1, 2, 4, 6, 8, 10],
               'bootstrap': [True, False]}

    regressor_config['regressors'].append({
        'reg_name': 'rf',
        'reg_parameters': rf_grid
    })

    svr_grid = [{'kernel': ['rbf'], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
                 'C': [0.01, 0.1, 1, 10, 100, 1000]},
                {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]

    regressor_config['regressors'].append({
        'reg_name': 'svr',
        'reg_parameters': svr_grid
    })

    gb_grid = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'min_samples_split': [2, 5, 10, 20, 50],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'max_features': [4, 5, 7, 9, 10, 13],
        'subsample': [0.6, 0.7, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6, 8, 10, 12, 15]
    }

    regressor_config['regressors'].append({
        'reg_name': 'gb',
        'reg_parameters': gb_grid
    })

    with open('../configurations/reg_config.txt', 'w') as outfile:
        json.dump(regressor_config, outfile)


gen_reg_config()
