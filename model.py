import lightgbm as lgb
from sklearn.model_selection import train_test_split

def lightgbm_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_val, label=y_val)

    param = {'learning_rate': 0.2,
        'num_leaves': 256,
        'max_depth': 7,
        'feature_fraction': 1,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'objective':'binary',
    }
    param['metric'] = ['binary_logloss', 'auc']
    num_round = 500
    bst = lgb.train(param,
                    train_data,
                    num_round,
                    valid_sets=[test_data]
                    )

    return bst
