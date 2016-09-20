# coding: utf-8

import logging

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier


Logger = logging.getLogger(__name__)

FNAME = 'data/cmc.data'
COLNAMES = ['age', 'education', 'husband_education', 'children', 'religion',
            'working', 'husband_occupation', 'living_index', 'media', 'method']


def main(data):
    X, y = data.drop('method', axis=1).values, data['method'].values
    Logger.info("split train/test data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)
    Logger.info("learn")
    clf = RandomForestClassifier(n_estimators=100)
    score = cross_val_score(clf, X_train, y_train, cv=5)
    Logger.info("fit")
    clf.fit(X_train, y_train)
    print("train mean score: %0.6f (+/- %0.3f)" % (score.mean(), score.std()))
    print("test score: %0.6f" % clf.score(X_test, y_test))
    return clf


if __name__ == '__main__':
    Logger.info("read input file '%s'", FNAME)
    data = pd.read_csv(FNAME, names=COLNAMES, header=None)
    # X, y = data.drop('method', axis=1).values, data['method'].values

    print("raw data")
    clf1 = main(data)

    df_2 = data.drop(['education', 'husband_education',
                      'husband_occupation', 'living_index'], axis=1)
    dummy = lambda x: pd.get_dummies(data[x], prefix=x, drop_first=True)
    df_2 = pd.concat([df_2, dummy('education'),
                      dummy('husband_education'),
                      dummy('husband_occupation'),
                      dummy('living_index')], axis=1)

    print("turn dummies data")
    clf2 = main(df_2)
