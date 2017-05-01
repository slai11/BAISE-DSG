import xgboost as xgb

def xgboost_model(train_X, train_y, val_X, val_Y):
    train = xgb.DMatrix(train_X, label=train_y, missing=-999.0)
    val = xgb.DMatrix(val_X, label = val_y, missing=-999.0)
    param = {
            'max_depth': 5,
            'eta': 0.1,
            'objective':'binary:logistic',
            'min_child_weight':100,
            'subsample': 0.5
            }
    param['eval_metric'] = 'auc'
    num_round=200
    evallist = [(val,'eval'), (train,'train')]
    
    bst = xgb.train(plst, train, num_round, evallist, early_stopping_rounds=10, verbose_eval=True)

    return bst
