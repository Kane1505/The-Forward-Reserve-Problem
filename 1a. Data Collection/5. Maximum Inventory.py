# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:49:11 2024

@author: kanev
"""

import pandas as pd
import numpy as np

# Set the number of inventory snap shots that are taken and the base name of the inventory files
num_snap_shots = 4
base_name = 'Inventory Snap Shot Case Study Clean'

# Import data
for i in range(num_snap_shots):
    globals()[f'inventory{i+1}'] = pd.read_pickle(base_name + ' ' + str(i+1) + '.pkl').to_numpy()
    
# Initialize a vector to store all maximum inventory
max_inv = np.zeros(len(globals()[f'inventory{1}']))

# For each SKU, find the maximum value over all inventory snap shots
for i in range(len(max_inv)):
    inv_levels = []
    for j in range(num_snap_shots):
        inv_levels.append(globals()[f'inventory{j+1}'][i,1])
    max_inv[i] = max(inv_levels)
    
# Add the SKU numbers to the average daily picks
SKUs = globals()[f'inventory{1}'][:,0]
max_inv = max_inv[:,np.newaxis]
SKUs = SKUs[:,np.newaxis]
max_inv = np.concatenate((SKUs, max_inv), axis = 1)

# Save the data to Excel and Pickle
columns = ['SKU Number', 'Maximum Inventory']
max_inv = pd.DataFrame(max_inv, columns = columns)
max_inv.to_excel('Maximum Inventory Case Study.xlsx', index = False)
max_inv.to_pickle('Maximum Inventory Case Study.pkl')