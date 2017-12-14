'''
This code is only to be used with the full dataset.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import random
import ml_metrics as metrics
import xgboost as xgb
from itertools import chain, combinations
from collections import defaultdict
from heapq import nlargest
from operator import itemgetter
import math

test = pd.read_csv("full_test.csv")
test = test.drop("srch_ci", axis = 1)
test = test.drop("srch_co", axis = 1)
test = test.drop("date_time", axis = 1)
test = test.drop(test.columns[0], axis=1)

train = pd.read_csv("50k_train.csv")
train = train.drop(train.columns[0], axis=1)
train = train.drop("srch_ci", axis = 1)
train = train.drop("srch_co", axis = 1)
train = train.drop("date_time", axis = 1)
train = train.drop("is_booking", axis = 1)
train = train.drop("cnt", axis = 1)

print('Starting Base Model')
most_common_clusters = list(train.hotel_cluster.value_counts().head().index)
preds_file = [most_common_clusters for i in range(test.shape[0])]
path = 'base_model_submission.csv'
out = open(path, "w")
out.write("id,hotel_cluster\n")
for i in range(len(test['id'].values)):
    out.write(str(test['id'].values[i]) + ',' + ' ' + str(preds_file[i][0]) + ' ' + str(preds_file[i][1]) + ' ' + str(preds_file[i][2]) + ' ' + str(preds_file[i][3]) + ' ' + str(preds_file[i][4]))
    out.write("\n")
out.close()

print('Starting XGBoost')
train_xgb = train.drop("hotel_cluster", axis = 1)
test_xgb = test.drop("id", axis = 1)
xgb_clf = XGBClassifier(learning_rate = 0.1, max_depth = 5, n_estimators = 50)
xgb_clf.fit(train_xgb, train['hotel_cluster'].values)
prediction = xgb_clf.predict_proba(test_xgb)
xgb_preds = []
for i in range(len(prediction)):
    xgb_preds.append(prediction[i].argsort()[-5:][::-1])
path = 'xgb_submission.csv'
out = open(path, "w")
out.write("id,hotel_cluster\n")
for i in range(len(test['id'].values)):
    out.write(str(test['id'].values[i]) + ',' + ' ' + str(xgb_preds[i][0]) + ' ' + str(xgb_preds[i][1]) + ' ' + str(xgb_preds[i][2]) + ' ' + str(xgb_preds[i][3]) + ' ' + str(xgb_preds[i][4]))
    out.write("\n")
out.close()

print('Starting Random Forest')
train_rf = train.drop("hotel_cluster", axis = 1)
test_rf = test.drop("id", axis = 1)
train_rf = train.fillna(0)
test_rf = test.fillna(0)
rf_clf = RandomForestClassifier(n_estimators=5, max_depth=10, random_state=42)
rf_clf.fit(train_rf, train['hotel_cluster'].values)
prediction = rf_clf.predict_proba(test_rf)
rf_preds = []
for i in range(len(prediction)):
    rf_preds.append(prediction[i].argsort()[-5:][::-1])
path = 'rf_submission.csv'
out = open(path, "w")
out.write("id,hotel_cluster\n")
for i in range(len(test['id'].values)):
    out.write(str(test['id'].values[i]) + ',' + ' ' + str(rf_preds[i][0]) + ' ' + str(rf_preds[i][1]) + ' ' + str(rf_preds[i][2]) + ' ' + str(rf_preds[i][3]) + ' ' + str(rf_preds[i][4]))
    out.write("\n")
out.close()

print('Starting Logistic Regression')
train_lr = train.drop("hotel_cluster", axis = 1)
test_lr = test.drop("id", axis = 1)
train_lr = train.fillna(0)
test_lr = test.fillna(0)
lr_clf = LogisticRegression(penalty='l2', max_iter=300, C=6)
lr_clf.fit(train_lr, train['hotel_cluster'].values)
prediction = lr_clf.predict_proba(test_lr)
lr_preds = []
for i in range(len(prediction)):
    lr_preds.append(prediction[i].argsort()[-5:][::-1])
path = 'lr_submission.csv'
out = open(path, "w")
out.write("id,hotel_cluster\n")
for i in range(len(test['id'].values)):
    out.write(str(test['id'].values[i]) + ',' + ' ' + str(lr_preds[i][0]) + ' ' + str(lr_preds[i][1]) + ' ' + str(lr_preds[i][2]) + ' ' + str(lr_preds[i][3]) + ' ' + str(lr_preds[i][4]))
    out.write("\n")
out.close()

print('Starting Data Leak')
train = pd.read_csv("train.csv")
test = pd.read_csv("full_test.csv")
topclusters = list(train.hotel_cluster.value_counts().head().index)

ulc_odd = defaultdict(lambda: defaultdict(int))
all_columns = defaultdict(lambda: defaultdict(int))
sdi_hc_hm_by = defaultdict(lambda: defaultdict(int))
for index, row in train.iterrows():
    if (index % 100000 == 0):
        print('Read row: ' + str(index))
    values = []
    hotel_cluster = row['hotel_cluster']
    ulci = row['user_location_city']
    ulco = row['user_location_country']
    ulr = row['user_location_region']
    odd = row['orig_destination_distance']
    sdi = row['srch_destination_id']
    hm = row['hotel_market']
    hc = row['hotel_country']
    by = row['year']

    if ulci != '' and odd != '':
        ulc_odd[(ulci, odd)][hotel_cluster] += 1

    if ulci != '' and ulco != '' and ulr != '' and odd != '' and hm != '':
        all_columns[(ulci, ulco, ulr, odd, hm)][hotel_cluster] += 1

    if sdi != '' and hc != '' and hm != '' and by != '':
        sdi_hc_hm_by[(sdi, hc, hm, by)][hotel_cluster] += 1

preds = []
indicies = []
for index, row in test.iterrows():
    if (index % 100000 == 0):
        print('Predicted row: ' + str(index))
    output = []
    filled = []
    run_ml = True
    ulci = row['user_location_city']
    ulco = row['user_location_country']
    ulr = row['user_location_region']
    odd = row['orig_destination_distance']
    sdi = row['srch_destination_id']
    hm = row['hotel_market']
    hc = row['hotel_country']
    by = row['year']

    if (ulci, ulco, ulr, odd, hm) in all_columns:
        run_ml = False
        individual_outputs = []
        clusters = all_columns[ulci, ulco, ulr, odd, hm]
        topitems = nlargest(5, sorted(clusters.items()), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            output.append(str(topitems[i][0]))
            filled.append(topitems[i][0])

    if (ulci, odd) in ulc_odd:
        run_ml = False
        individual_outputs = []
        clusters = ulc_odd[ulci, odd]
        topitems = nlargest(5, sorted(clusters.items()), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            output.append(str(topitems[i][0]))
            filled.append(topitems[i][0])

    if (sdi, hc, hm, by) in sdi_hc_hm_by:
        run_ml = False
        individual_outputs = []
        clusters = sdi_hc_hm_by[sdi, hc, hm, by]
        topitems = nlargest(5, sorted(clusters.items()), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            output.append(str(topitems[i][0]))
            filled.append(topitems[i][0])

    for i in range(len(topclusters)):
        if topclusters[i] in filled:
            continue
        if len(filled) == 5:
            break
        output.append(str(topclusters[i]))
        filled.append(topclusters[i])
    if run_ml:
        indicies.append(index)
    preds.append(output)

preds_file = []
for i in range(len(preds)):
    row_pred = []
    row_pred.append(int(preds[i][0]))
    row_pred.append(int(preds[i][1]))
    row_pred.append(int(preds[i][2]))
    row_pred.append(int(preds[i][3]))
    row_pred.append(int(preds[i][4]))
    preds_file.append(row_pred)

path = 'data_leak_submission.csv'
out = open(path, "w")
out.write("id,hotel_cluster\n")
for i in range(len(test['id'].values)):
    if (i % 100000 == 0):
        print(i)
    out.write(str(test['id'].values[i]) + ',' + ' ' + str(preds_file[i][0]) + ' ' + str(preds_file[i][1]) + ' ' + str(preds_file[i][2]) + ' ' + str(preds_file[i][3]) + ' ' + str(preds_file[i][4]))
    out.write("\n")
out.close()

for i in indicies:
    preds_file[i][0] = xgb_preds[i][0]
    preds_file[i][1] = xgb_preds[i][1]
    preds_file[i][2] = xgb_preds[i][2]
    preds_file[i][3] = xgb_preds[i][3]
    preds_file[i][4] = xgb_preds[i][4]

path = 'data_leak_xgb_submission.csv'
out = open(path, "w")
out.write("id,hotel_cluster\n")
for i in range(len(test['id'].values)):
    if (i % 100000 == 0):
        print(i)
    out.write(str(test['id'].values[i]) + ',' + ' ' + str(preds_file[i][0]) + ' ' + str(preds_file[i][1]) + ' ' + str(preds_file[i][2]) + ' ' + str(preds_file[i][3]) + ' ' + str(preds_file[i][4]))
    out.write("\n")
out.close()

for i in indicies:
    preds_file[i][0] = rf_preds[i][0]
    preds_file[i][1] = rf_preds[i][1]
    preds_file[i][2] = rf_preds[i][2]
    preds_file[i][3] = rf_preds[i][3]
    preds_file[i][4] = rf_preds[i][4]

path = 'data_leak_rf_submission.csv'
out = open(path, "w")
out.write("id,hotel_cluster\n")
for i in range(len(test['id'].values)):
    if (i % 100000 == 0):
        print(i)
    out.write(str(test['id'].values[i]) + ',' + ' ' + str(preds_file[i][0]) + ' ' + str(preds_file[i][1]) + ' ' + str(preds_file[i][2]) + ' ' + str(preds_file[i][3]) + ' ' + str(preds_file[i][4]))
    out.write("\n")
out.close()

for i in indicies:
    preds_file[i][0] = lr_preds[i][0]
    preds_file[i][1] = lr_preds[i][1]
    preds_file[i][2] = lr_preds[i][2]
    preds_file[i][3] = lr_preds[i][3]
    preds_file[i][4] = lr_preds[i][4]

path = 'data_leak_lr_submission.csv'
out = open(path, "w")
out.write("id,hotel_cluster\n")
for i in range(len(test['id'].values)):
    if (i % 100000 == 0):
        print(i)
    out.write(str(test['id'].values[i]) + ',' + ' ' + str(preds_file[i][0]) + ' ' + str(preds_file[i][1]) + ' ' + str(preds_file[i][2]) + ' ' + str(preds_file[i][3]) + ' ' + str(preds_file[i][4]))
    out.write("\n")
out.close()
