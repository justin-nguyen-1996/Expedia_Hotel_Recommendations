'''
This code is only to be used with the datasets Hari has created.
Trains the ML model on 40k rows
Trains the data leak code on 160k rows
Tests on 10k rows
16:1 ratio
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

train = pd.read_csv("50k_train.csv")
test = pd.read_csv("50k_test.csv")
train = train.drop(train.columns[0], axis=1)
test = test.drop(test.columns[0], axis=1)
train = train.drop("srch_ci", axis = 1)
train = train.drop("srch_co", axis = 1)
train = train.drop("date_time", axis = 1)
test = test.drop("srch_ci", axis = 1)
test = test.drop("srch_co", axis = 1)
test = test.drop("date_time", axis = 1)

print('Starting Base Model')
most_common_clusters = list(train.hotel_cluster.value_counts().head().index)
predictions = [most_common_clusters for i in range(test.shape[0])]
target = [[l] for l in test["hotel_cluster"]]
print('Base Model Score: ' + str(metrics.mapk(target, predictions, k=5)))

print('Starting XGBoost')
train_xgb = train.drop("hotel_cluster", axis = 1)
test_xgb = test.drop("hotel_cluster", axis = 1)
xgb_clf = XGBClassifier()
xgb_clf.fit(train_xgb, train['hotel_cluster'].values)
prediction = xgb_clf.predict_proba(test_xgb)
xgb_preds = []
for i in range(len(prediction)):
    xgb_preds.append(prediction[i].argsort()[-5:][::-1])
target = [[l] for l in test["hotel_cluster"]]
print('XGBoost Score: ' + str(metrics.mapk(target, xgb_preds, k=5)))

print('Starting Random Forest')
train_rf = train.fillna(0)
test_rf = test.fillna(0)
rf_clf = RandomForestClassifier(n_estimators = 100)
rf_clf.fit(train_rf.drop("hotel_cluster", axis = 1), train_rf['hotel_cluster'].values)
prediction = rf_clf.predict_proba(test_rf.drop("hotel_cluster", axis = 1))
rf_preds = []
for i in range(len(prediction)):
    rf_preds.append(prediction[i].argsort()[-5:][::-1])
target = [[l] for l in test_rf["hotel_cluster"]]
print('Random Forest Score: ' + str(metrics.mapk(target, rf_preds, k=5)))

print('Starting Data Leak')
train = pd.read_csv("200k_train.csv")
test = pd.read_csv("50k_test.csv")
topclusters = list(train.hotel_cluster.value_counts().head().index)

ulc_odd = defaultdict(lambda: defaultdict(int))
all_columns = defaultdict(lambda: defaultdict(int))
sdi_hc_hm_by = defaultdict(lambda: defaultdict(int))
sdi_dict = defaultdict(lambda: defaultdict(int))
hc_dict = defaultdict(lambda: defaultdict(int))
popular_hotel_cluster = defaultdict(int)
all_minus_odd = defaultdict(lambda: defaultdict(int))
all_minus_hm = defaultdict(lambda: defaultdict(int))
all_minus_ulci = defaultdict(lambda: defaultdict(int))
all_minus_ulco = defaultdict(lambda: defaultdict(int))
all_minus_ulr = defaultdict(lambda: defaultdict(int))
for index, row in train.iterrows():
    #if (index % 10000 == 0):
        #print(str(index))
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

    if ulci != '' and ulco != '' and ulr != '' and sdi != '' and hm != '':
        all_minus_odd[(ulci, ulco, ulr, sdi, hm)][hotel_cluster] += 1

    if ulci != '' and ulco != '' and ulr != '' and sdi != '' and odd != '':
        all_minus_hm[(ulci, ulco, ulr, sdi, odd)][hotel_cluster] += 1

    if hm != '' and ulco != '' and ulr != '' and sdi != '' and hm != '':
        all_minus_ulci[(hm, ulco, ulr, sdi, odd)][hotel_cluster] += 1

    if ulci != '' and ulco != '' and ulr != '' and sdi != '' and hm != '':
        all_minus_ulco[(hm, ulci, ulr, sdi, odd)][hotel_cluster] += 1

    if ulci != '' and ulco != '' and ulr != '' and sdi != '' and hm != '':
        all_minus_ulr[(hm, ulci, ulco, sdi, odd)][hotel_cluster] += 1

    if sdi != '':
        sdi_dict[sdi][hotel_cluster] += 1

    if hc != '':
        hc_dict[hc][hotel_cluster] += 1

preds = []
indicies = []
for index, row in test.iterrows():
    #if (index % 100000 == 0):
        #print(index)
    output = []
    filled = []
    run_ml = True
    #hotel_cluster = row['hotel_cluster']
    ulci = row['user_location_city']
    ulco = row['user_location_country']
    ulr = row['user_location_region']
    odd = row['orig_destination_distance']
    sdi = row['srch_destination_id']
    hm = row['hotel_market']
    hc = row['hotel_country']
    by = row['year']

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
    '''
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

    if (ulci, ulco, ulr, sdi, hm) in all_columns:
        run_ml = False
        individual_outputs = []
        clusters = all_minus_odd[ulci, ulco, ulr, sdi, hm]
        topitems = nlargest(5, sorted(clusters.items()), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            output.append(str(topitems[i][0]))
            filled.append(topitems[i][0])

    if (ulci, ulco, ulr, sdi, odd) in all_columns:
        run_ml = False
        individual_outputs = []
        clusters = all_minus_hm[ulci, ulco, ulr, sdi, odd]
        topitems = nlargest(5, sorted(clusters.items()), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            output.append(str(topitems[i][0]))
            filled.append(topitems[i][0])

    if (hm, ulco, ulr, sdi, odd) in all_columns:
        run_ml = False
        individual_outputs = []
        clusters = all_minus_ulci[hm, ulco, ulr, sdi, odd]
        topitems = nlargest(5, sorted(clusters.items()), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            output.append(str(topitems[i][0]))
            filled.append(topitems[i][0])

    if (hm, ulci, ulr, sdi, odd) in all_columns:
        run_ml = False
        individual_outputs = []
        clusters = all_minus_ulco[hm, ulci, ulr, sdi, odd]
        topitems = nlargest(5, sorted(clusters.items()), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            output.append(str(topitems[i][0]))
            filled.append(topitems[i][0])

    if (hm, ulci, ulco, sdi, odd) in all_columns:
        run_ml = False
        individual_outputs = []
        clusters = all_minus_ulr[hm, ulci, ulco, sdi, odd]
        topitems = nlargest(5, sorted(clusters.items()), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            output.append(str(topitems[i][0]))
            filled.append(topitems[i][0])
    '''

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

    '''
    if sdi in sdi_dict:
        run_ml = False
        individual_outputs = []
        clusters = sdi_dict[sdi]
        topitems = nlargest(5, sorted(clusters.items()), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            output.append(str(topitems[i][0]))
            filled.append(topitems[i][0])

    if hc in hc_dict:
        run_ml = False
        individual_outputs = []
        clusters = hc_dict[hc]
        topitems = nlargest(5, sorted(clusters.items()), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            output.append(str(topitems[i][0]))
            filled.append(topitems[i][0])
    '''

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

target = [[l] for l in test["hotel_cluster"]]
print('Data Leak Score: ' + str(metrics.mapk(target, preds_file, k=5)))

for i in indicies:
    preds_file[i][0] = xgb_preds[i][0]
    preds_file[i][1] = xgb_preds[i][1]
    preds_file[i][2] = xgb_preds[i][2]
    preds_file[i][3] = xgb_preds[i][3]
    preds_file[i][4] = xgb_preds[i][4]
target = [[l] for l in test["hotel_cluster"]]
print('Data Leak + XGBoost Score: ' + str(metrics.mapk(target, preds_file, k=5)))

for i in indicies:
    preds_file[i][0] = rf_preds[i][0]
    preds_file[i][1] = rf_preds[i][1]
    preds_file[i][2] = rf_preds[i][2]
    preds_file[i][3] = rf_preds[i][3]
    preds_file[i][4] = rf_preds[i][4]
target = [[l] for l in test["hotel_cluster"]]
print('Data Leak + Random Forest Score: ' + str(metrics.mapk(target, preds_file, k=5)))
