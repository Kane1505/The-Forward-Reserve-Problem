# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:11:03 2024

@author: kanev
"""

import pandas as pd
import numpy as np
import scipy

demand = pd.read_pickle('Demand Data Case Study Clean.pkl').to_numpy()
demand = demand[:,1:]

pval = np.zeros(len(demand))
for i in range(len(demand)):    
    pval[i] = scipy.stats.shapiro(demand[i])[1]

print('The lowest p-value of all SKUs equals: ' + str(np.min(pval)))
print('The highest p-value of all SKUs equals: ' + str(np.max(pval)))