import pandas as pd


def dummy(train, meta):
    _train=train.copy()
    ##to deal with the categorical variables, we create dummy variables
    v = meta[(meta.level == 'nominal') & (meta.keep)].index
    _train = pd.get_dummies(train, columns=v, drop_first=True)
    return _train

