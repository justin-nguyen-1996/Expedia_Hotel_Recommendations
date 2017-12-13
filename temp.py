#! /usr/bin/env python

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import random
import ml_metrics as metrics
import xgboost as xgb
import matplotlib.pyplot as plt

#train = pd.read_csv("50k_with_extracted_features.csv")
train = pd.read_csv("50k_train.csv")
test = pd.read_csv("50k_test.csv")
#test = pd.read_csv("50k_test_with_extracted_features.csv")

train.shape
test.shape

train = train.drop(train.columns[0], axis=1)
test = test.drop(test.columns[0], axis=1)

train = train.drop("srch_ci", axis = 1)
train = train.drop("srch_co", axis = 1)
train = train.drop("date_time", axis = 1)
test = test.drop("srch_ci", axis = 1)
test = test.drop("srch_co", axis = 1)
test = test.drop("date_time", axis = 1)

most_common_clusters = list(train.hotel_cluster.value_counts().head().index)
most_common_clusters
common_predictions = [most_common_clusters for i in range(test.shape[0])]
common_target = [[l] for l in test["hotel_cluster"]]
print(metrics.mapk(common_target, common_predictions, k=5))

train_xgb = train.drop("hotel_cluster", axis = 1)
test_xgb = test.drop("hotel_cluster", axis = 1)
xgb_clf = XGBClassifier()
xgb_clf.fit(train_xgb, train['hotel_cluster'].values)
xgb_prediction = xgb_clf.predict_proba(test_xgb)
xgb_preds = []
for i in range(len(xgb_prediction)):
    xgb_preds.append(xgb_prediction[i].argsort()[-5:][::-1])
xgb_target = [[l] for l in test["hotel_cluster"]]
print(metrics.mapk(xgb_target, xgb_preds, k=5))

xgb_importances = xgb_clf.feature_importances_
xgb_indices = np.argsort(xgb_importances)[::-1]
for i in xgb_indices:
    print(str(train_xgb.columns[i]) + ': ' + str(xgb_importances[i]))
 
plt.figure()
plt.bar( np.arange(len(xgb_importances)), xgb_importances )
plt.xticks( np.arange(len(xgb_importances)), train_xgb.columns, rotation='vertical')
plt.show()

#######################################################################################

# train_rf = train.fillna(0)
# test_rf = test.fillna(0)
# 
# rf_clf = RandomForestClassifier(n_estimators = 100)
# rf_clf.fit(train_rf.drop("hotel_cluster", axis = 1), train_rf['hotel_cluster'].values)
# 
# rf_prediction = rf_clf.predict_proba(test_rf.drop("hotel_cluster", axis = 1))
# rf_preds = []
# for i in range(len(rf_prediction)):
#     rf_preds.append(rf_prediction[i].argsort()[-5:][::-1])
# rf_target = [[l] for l in test_rf["hotel_cluster"]]
# print(metrics.mapk(rf_target, rf_preds, k=5))
# 
# rf_importances = rf_clf.feature_importances_
# rf_indices = np.argsort(rf_importances)[::-1]
# for i in rf_indices:
#     print(str(train_rf.columns[i]) + ': ' + str(rf_importances[i]))
