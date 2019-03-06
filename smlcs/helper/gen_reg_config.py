import json


def gen_reg_config():
    regressor_config = dict()
    regressor_config['regressors'] = []
    regressor_config['innercv_folds'] = 3

    rf_grid = {'n_estimators': (100, 102),
               'bootstrap': [True, False]
               }

    regressor_config['regressors'].append({
        'reg_name': 'rf',
        'reg_parameters': rf_grid
    })

    svr_grid = {
        'C': (1e-1, 1, 'log-uniform'),
        }

    regressor_config['regressors'].append({
        'reg_name': 'svr',
        'reg_parameters': svr_grid
    })

    gb_grid = {
        'n_estimators': (100, 102),
    }

    regressor_config['regressors'].append({
        'reg_name': 'gb',
        'reg_parameters': gb_grid
    })

    with open('../configurations/reg_config.txt', 'w') as outfile:
        json.dump(regressor_config, outfile)


gen_reg_config()
