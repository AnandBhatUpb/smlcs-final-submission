import json


def gen_reg_config():
    regressor_config = dict()
    regressor_config['regressors'] = []
    regressor_config['innercv_folds'] = 3

    rf_grid = {'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
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

    with open('../configurations/reg_config.txt', 'w') as outfile:
        json.dump(regressor_config, outfile)


gen_reg_config()
