# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:00:32 2024

@author: kanev
"""

import pandas as pd
import numpy as np

# State the number of inventory snapshots used and the base name of the files
num_snap_shots = 4
base_snap_shot_name = 'Inventory Snap Shot Case Study'

# If hazmat units need to be removed set this variable to 'Y', if not, set it to 'N'
hazmat_removal = 'Y'

# Import data
demand = pd.read_pickle('Demand Data Case Study.pkl').to_numpy()
location_information = pd.read_excel('Location Information.xlsx').to_numpy()
SKU_features = pd.read_pickle('Physical SKU Features Case Study.pkl').to_numpy()
for i in range(num_snap_shots):
    globals()[f'inventory{i+1}'] = pd.read_pickle(base_snap_shot_name + ' ' + str(i+1) + '.pkl').to_numpy()
    

# Create a vector with all possible SKUs from each data set
# Inialize the list
all_SKUs = []
# Add all SKU numbers found in the inventory data
for i in range(num_snap_shots):
    for j in range(len(globals()[f'inventory{i+1}'])):
        all_SKUs.append(globals()[f'inventory{i+1}'][j,0])

# Add all SKU numbers found in the demand data
for i in range(len(demand)):
    all_SKUs.append(demand[i,0])
    
# Add all SKU numbers found in the SKU features data
for i in range(len(SKU_features)):
    all_SKUs.append(SKU_features[i,0])

# Turn the list into an array
all_SKUs = np.array(all_SKUs)

# Find all unique SKUs
unique_SKUs = np.unique(all_SKUs, return_counts=False)

# Find how many times they occur
unique_SKUs_count = np.unique(all_SKUs, return_counts= True )[1]

# Initialize a list that decides which SKUs need to be removed
remove = []

# Remove the SKUs that are missing from one or more of the data sets
for i in range(len(unique_SKUs)):
    if unique_SKUs_count[i] != 2 + num_snap_shots:
        remove.append(i)
        
        
# Remove these SKUs first to speed up computation times
unique_SKUs = np.delete(unique_SKUs, remove)

remove_feat = []
for i in range(len(SKU_features)):
    if SKU_features[i,0] not in unique_SKUs:
        remove_feat.append(i)
SKU_features = np.delete(SKU_features, remove_feat, axis = 0)

remove_demand = []
for i in range(len(demand)):
    if demand[i,0] not in unique_SKUs:
        remove_demand.append(i)
demand = np.delete(demand, remove_demand, axis = 0)

for i in range(num_snap_shots):
    remove_inv = []
    for j in range(len(globals()[f'inventory{i+1}'])):
        if globals()[f'inventory{i+1}'][j,0] not in unique_SKUs:
            remove_inv.append(j)
    globals()[f'inventory{i+1}'] = np.delete(globals()[f'inventory{i+1}'], remove_inv, axis = 0)
    

# Start a new remove list
remove = []

# Remove the SKUs that have hazmat components if this is applicable
if hazmat_removal == 'Y':
    for i in range(len(SKU_features)):
        if SKU_features[i,5] != 'N':
            index = np.where(unique_SKUs == SKU_features[i,0])[0][0]
            remove.append(index)

# Remove SKUs that are missing dimensions or weight
for i in range(len(SKU_features)):
    for j in range(4):
        if SKU_features[i,j+1] == 0:
            index = np.where(unique_SKUs == SKU_features[i,0])[0][0]
            remove.append(index)

# Remove SKUs that are too heavy
for i in range(len(SKU_features)):
    if SKU_features[i,4] > location_information[0,4]:
        index = np.where(unique_SKUs == SKU_features[i,0])[0][0]
        remove.append(index)
        
# Remove SKUs that are too large
loc_dims = [location_information[0,1], location_information[0,2], location_information[0,3]]
loc_dims_sorted = sorted(loc_dims, reverse=True)
for i in range(len(SKU_features)):
    SKU_dims = []
    for j in range(3):
        SKU_dims.append(SKU_features[i,j+1])
    SKU_dims_sorted = sorted(SKU_dims, reverse=True)
    for j in range(3):
        if loc_dims_sorted[j] < SKU_dims_sorted[j]:
            index = np.where(unique_SKUs == SKU_features[i,0])[0][0]
            remove.append(index)
            break
            

# Remove SKUs that have not been ordered in the entire order history
for i in range(len(demand)):
    if sum(demand[i,1:]) == 0:
        index = np.where(unique_SKUs == demand[i,0])[0][0]
        remove.append(index)

# Remove SKUs that have no inventory in any of the inventory snap shots
# To do this, first collect all unique SKU numbers for which at least one of the snap shots has a record
inventory_SKUs = []
for i in range(num_snap_shots):
    for j in range(len(globals()[f'inventory{i+1}'])):
        inventory_SKUs.append(globals()[f'inventory{i+1}'][j,0])
inventory_SKUs = np.array(inventory_SKUs)
unique_inventory_SKUs = np.unique(inventory_SKUs, return_counts=False)
# For the SKUs that have inventory data, check if it is not equal to zero in all snap shots
for i in range(len(unique_inventory_SKUs)):
    # Set a variable to one that gets changed to zero if any of the snap shots have inventory
    all_zero = 1
    for j in range(num_snap_shots):
        index = np.where(globals()[f'inventory{j+1}'][:,0] == unique_inventory_SKUs[i])[0][0]
        if globals()[f'inventory{j+1}'][i,1] != 0:
            all_zero = 0
            break
    if all_zero == 1:
        index = np.where(unique_SKUs == unique_inventory_SKUs[j])[0][0]
        remove.append(index)

# Delete all SKU numbers that need to be removed from the unique SKUs vector
remaining_SKUs = np.delete(unique_SKUs, remove)

# Rewrite all data sets to have the same order of SKUs and only include the leftover SKUs
# Create new arrays
demand_clean = np.zeros([len(remaining_SKUs), len(demand[0])], dtype = object)
SKU_features_clean = np.zeros([len(remaining_SKUs), len(SKU_features[0])], dtype = object)
for i in range(num_snap_shots):
    globals()[f'inventory_clean{i+1}'] = np.zeros([len(remaining_SKUs), len(globals()[f'inventory{i+1}'][0])], dtype = object)
# Fill these arrays with the remaining SKUs in the same order for each data set
for i in range(len(remaining_SKUs)):
    index_demand = np.where(demand[:,0] == remaining_SKUs[i])[0][0]
    index_features = np.where(SKU_features[:,0] == remaining_SKUs[i])[0][0]
    demand_clean[i] = demand[index_demand]
    SKU_features_clean[i] = SKU_features[index_features]
    for j in range(num_snap_shots):
        index_inventory = np.where(globals()[f'inventory{j+1}'][:,0] == remaining_SKUs[i])[0][0]
        globals()[f'inventory_clean{j+1}'][i] = globals()[f'inventory{j+1}'][index_inventory]

# The column with the hazmat flag can be removed from the SKU features
if hazmat_removal == 'Y':
    SKU_features_clean = SKU_features_clean[:,:5]

# Create columns for all data sets
columns_demand = ['SKU Number']
for i in range(len(demand_clean[0,1:])):
    columns_demand.append('Day ' + str(i+1))
columns_features = ['SKU Number', 'SKU Height', 'SKU Width', 'SKU Depth', 'Unit Weight']
columns_inventory = ['SKU Number', 'On Hand Qty']

# Turn array into pandas data frame
demand_clean = pd.DataFrame(demand_clean, columns = columns_demand)
SKU_features_clean = pd.DataFrame(SKU_features_clean, columns = columns_features)
for i in range(num_snap_shots):
    globals()[f'inventory_clean{i+1}'] = pd.DataFrame(globals()[f'inventory_clean{i+1}'], columns = columns_inventory)
# Save the data to an Excel file and a Pickle file
demand_clean.to_excel('Demand Data Case Study Clean.xlsx', index = False)
demand_clean.to_pickle('Demand Data Case Study Clean.pkl')
SKU_features_clean.to_excel('Physical SKU Features Case Study Clean.xlsx', index = False)
SKU_features_clean.to_pickle('Physical SKU Features Case Study Clean.pkl')
for i in range(num_snap_shots):
    globals()[f'inventory_clean{i+1}'].to_excel(base_snap_shot_name + ' Clean  ' + str(i+1) + '.xlsx', index = False)
    globals()[f'inventory_clean{i+1}'].to_pickle(base_snap_shot_name + ' Clean ' + str(i+1) + '.pkl')









