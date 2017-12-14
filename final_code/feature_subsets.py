import pandas as pd
import numpy as np
import random
import ml_metrics as metrics
from itertools import chain, combinations
from collections import defaultdict
from heapq import nlargest
from operator import itemgetter
import math

train = pd.read_csv("500k_train.csv")
test = pd.read_csv("500k_test.csv")
match_cols = ['user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance']

subsets = []
def all_subsets(ss):
  return (chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1))))

for subset in all_subsets(match_cols):
  subsets.append(list(subset))
subsets = subsets[1:]

best_subset_name = ''
best_subset_value = 0
for j in range(len(subsets)):
    best_combo = defaultdict(lambda: defaultdict(int))
    popular_hotel_cluster = defaultdict(int)
    for index, row in train.iterrows():
        values = []
        skip = False
        hotel_cluster = row['hotel_cluster']
        length_of_subset = len(subsets[j])
        for i in range(length_of_subset):
            if math.isnan(row[subsets[j][i]]):
                skip = True
        if not skip:
            for i in range(length_of_subset):
                values.append(row[subsets[j][i]])
            if length_of_subset == 1:
                best_combo[(values[0])][hotel_cluster] += 1
            elif length_of_subset == 2:
                best_combo[(values[0], values[1])][hotel_cluster] += 1
            elif length_of_subset == 3:
                best_combo[(values[0], values[1], values[2])][hotel_cluster] += 1
            elif length_of_subset == 4:
                best_combo[(values[0], values[1], values[2], values[3])][hotel_cluster] += 1
            else:
                best_combo[(values[0], values[1], values[2], values[3], values[4])][hotel_cluster] += 1
            popular_hotel_cluster[hotel_cluster] += 1
    topclusters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))
    preds = []
    for index, row in test.iterrows():
        values = []
        filled = []
        skip = False
        hotel_cluster = row['hotel_cluster']
        length_of_subset = len(subsets[13])
        for i in range(length_of_subset):
            if math.isnan(row[subsets[13][i]]):
                skip = True
        if not skip:
            for i in range(length_of_subset):
                values.append(row[subsets[13][i]])
            if length_of_subset == 1:
                s1 = values[0]
            elif length_of_subset == 2:
                s1 = (values[0], values[1])
            elif length_of_subset == 3:
                s1 = (values[0], values[1], values[2])
            elif length_of_subset == 4:
                s1 = (values[0], values[1], values[2], values[3])
            else:
                s1 = (values[0], values[1], values[2], values[3], values[4])
            output = []
            if s1 in best_combo:
                individual_outputs = []
                clusters = best_combo[s1]
                topitems = nlargest(5, sorted(clusters.items()), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    output.append(str(topitems[i][0]))
                    filled.append(topitems[i][0])
            for i in range(len(topclusters)):
                if topclusters[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                output.append(str(topclusters[i][0]))
                filled.append(topclusters[i][0])
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
    print(str(subsets[j]) + ':' + str(metrics.mapk(target, preds_file, k=5)))
    if metrics.mapk(target, preds_file, k=5) >= best_subset_value:
        best_subset_value = metrics.mapk(target, preds_file, k=5)
        best_subset_name = str(subsets[j])

print(best_subset_name)
print(best_subset_value)
