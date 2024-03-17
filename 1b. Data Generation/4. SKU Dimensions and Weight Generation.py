# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:48:10 2024

@author: kanev
"""

import numpy as np
import pandas as pd

# Set seed
np.random.seed(124)

# Choose the number of SKUs that need dimensions and weight
number_of_SKUs = 10000

# Choose ranges for dims and weights
height_lower = 0.01
height_upper = 30
width_lower = 0.01
width_upper = 30
depth_lower = 0.01
depth_upper = 30
weight_lower = 0.01
weight_upper = 5


# Initialize a matrix to store the dimensions and weight
dims_weights = np.zeros([number_of_SKUs,5], dtype = object)

# Set the first column to SKU names
for i in range(number_of_SKUs):
    dims_weights[i,0] = 'SKU ' + str(i+1)

# Select dimensions and weight from a Uniform distribution for each SKU
for i in range(number_of_SKUs):
    dims_weights[i,1] = np.random.uniform(height_lower, height_upper)
    dims_weights[i,2] = np.random.uniform(width_lower, width_upper)
    dims_weights[i,3] = np.random.uniform(depth_lower, depth_upper)
    dims_weights[i,4] = np.random.uniform(weight_lower, weight_upper)
    
# Convert the 'dims_weights' array to a dataframe
dims_weights = pd.DataFrame(dims_weights, columns = ['SKU Name', 'Unit Height', 'Unit Width', 'Unit Depth', 'Unit Weight'])
 
# Save the dimensions and weights to Excel and to Pickle  
dims_weights.to_excel('SKU Dimensions and Weights 10000.xlsx', index = False)
dims_weights.to_pickle('SKU Dimensions and Weights 10000.pkl')