# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:37:20 2024

@author: kanev
"""

import pandas as pd

# Give the number of inventory snapshots
num_snap_shots = 4
# Give the base name of the inventory snapshot files
base_snap_name = 'Inventory Snap Shot Case Study'

# Store all inventory snapshots as pickle files
for i in range(num_snap_shots):
    snapshot = pd.read_excel(base_snap_name + ' ' + str(i+1) + '.xlsx', dtype = {'SKU Number': str, 'On Hand Qty': int})
    snapshot.to_pickle(base_snap_name + ' ' + str(i+1) + '.pkl')

# Fill out the correct file name to store physical SKU features as .pkl file
SKU_features = pd.read_excel('Physical SKU Features Case Study.xlsx', dtype = {'SKU Number': str, 'SKU Height': float, 'SKU Width': float, 'SKU Depth': float, 'Unit Weight': float, 'Hazmat Flag': str})
SKU_features.to_pickle('Physical SKU Features Case Study.pkl')

# Fill out the correct file name to store order data as .pkl file
Order_Data = pd.read_excel('Order Data Case Study.xlsx', dtype = {'Ord NO': str, 'SKU Number': str, 'Ord Qty': int, 'Create TStamp': str})
Order_Data.to_pickle('Order Data Case Study.pkl')
