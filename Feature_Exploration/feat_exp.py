# -*- coding: utf-8 -*-
"""
This file details preliminary feature exploration for the 
Expedia Recommendations dataset. 

29 November 2017
"""
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 

#%%
train = pd.read_csv('/Users/mattjohnson/Desktop/Principles_Term_Projet/50k_train.csv')
train.head
