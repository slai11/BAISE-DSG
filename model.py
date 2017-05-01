import xgboost as xgb
import lightgbm as lgb
from sklearn.cross_validation import train_test_split


def xgboost_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    train = xgb.DMatrix(X_train, label=y_train, missing=-999.0)
    val = xgb.DMatrix(X_val, label = y_val, missing=-999.0)
    param = {
            'max_depth': 4,
            'eta': 0.1,
            'objective':'binary:logistic',
            'min_child_weight':100,
            'subsample': 0.5
            }
    param['eval_metric'] = 'auc'
    num_round=500
    evallist  = [(val,'eval'), (train,'train')]
    plst = param.items()
    bst = xgb.train(plst, train, num_round, evallist, verbose_eval=True)

    return bst

def lightgbm_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_val, label=y_val)

    param = {'learning_rate': 0.05,
        'num_leaves': 256,
        'max_depth': 7,
        'feature_fraction': 1,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'objective':'binary',
    }
    param['metric'] = ['binary_logloss', 'auc']
    num_round = 1000
    bst = lgb.train(param,
                    train_data,
                    num_round,
                    valid_sets=[test_data]
                    )

    return bst
