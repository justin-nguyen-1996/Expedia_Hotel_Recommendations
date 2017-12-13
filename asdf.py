#! /usr/bin/env python

import numpy as np
import ml_metrics as metrics

# print metrics.mapk( [ [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5] ], 
#                       [range(1,6), range(1,6), range(1,6)], 
#                     5)
#                0.685185185185185)

print metrics.apk( [1],
                   [2,3,1,1,1],
                 5 )

# target = [ [1,2,3], [4,5,6], [7,7,7] ]
# pred   = [ []]
# print(metrics.mapk(target, preds, k=5))

print range(1,3)
