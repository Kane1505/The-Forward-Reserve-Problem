# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:25:40 2024

@author: kanev
"""

import pandas as pd
import numpy as np

# Import SKU dimensions, weights and the location characteristics
SKU_feat = pd.read_pickle('SKU Dimensions and Weights Exponential 1000.pkl')
Loc_feat = pd.read_excel('Location Information.xlsx')

# Change the dataframes to a numpy arrays
SKU_feat = SKU_feat.to_numpy()
Loc_feat = Loc_feat.to_numpy()

# Split the SKU features matrix in the SKU features part and the SKU names
SKU_names = SKU_feat[:,0]
SKU_feat = SKU_feat[:,1:]

# Remove the Location Types column from the Location features matrix
Loc_feat = Loc_feat[:,1:]

# Initialize a matrix to store the results
Units_per_Location = np.zeros([len(SKU_feat),len(Loc_feat)])

# For each SKU...
for i in range(len(SKU_feat)):
    # ... and for each location type...
    for j in range(len(Loc_feat)):
        # ... calculate the maximum number of units that fit on a location dimensions wise
        max_units_dims = max([(Loc_feat[j,0]//SKU_feat[i,0]) * (Loc_feat[j,1]//SKU_feat[i,1]) * (Loc_feat[j,2]//SKU_feat[i,2]), (Loc_feat[j,0]//SKU_feat[i,0]) * (Loc_feat[j,1]//SKU_feat[i,2]) * (Loc_feat[j,2]//SKU_feat[i,1]), (Loc_feat[j,0]//SKU_feat[i,1]) * (Loc_feat[j,1]//SKU_feat[i,0]) * (Loc_feat[j,2]//SKU_feat[i,2]), (Loc_feat[j,0]//SKU_feat[i,1]) * (Loc_feat[j,1]//SKU_feat[i,2]) * (Loc_feat[j,2]//SKU_feat[i,0]), (Loc_feat[j,0]//SKU_feat[i,2]) * (Loc_feat[j,1]//SKU_feat[i,0]) * (Loc_feat[j,2]//SKU_feat[i,1]), (Loc_feat[j,0]//SKU_feat[i,2]) * (Loc_feat[j,1]//SKU_feat[i,1]) * (Loc_feat[j,2]//SKU_feat[i,0])])
        # Also calculate the maximum number of units that fit on a location weight wise
        max_units_weight = Loc_feat[j,3]//SKU_feat[i,3]
        # Take the minimum of these two values and store this as the maximum number of units of SKU i per location type j
        Units_per_Location[i,j] = min([max_units_dims, max_units_weight])
        
# Add the SKU names and Units per Location together
SKU_names = SKU_names[:,np.newaxis]
Units_per_Location = np.concatenate((SKU_names, Units_per_Location), axis = 1)

# Create Column names
columns = ['SKU number']
for j in range(len(Loc_feat)):
    columns.append('Max Units Location Type ' +str(j+1))

# Turn 'Units_per_Location' into a dataframe
Units_per_Location = pd.DataFrame(Units_per_Location, columns = columns)
 
# Save the data to Excel and to Pickle  
Units_per_Location.to_excel('Max Units per Location Type Exponential 1000.xlsx', index = False)
Units_per_Location.to_pickle('Max Units per Location Type Exponential 1000.pkl')