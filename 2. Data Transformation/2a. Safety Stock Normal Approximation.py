# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:03:45 2024

@author: kanev
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import math

# Set a service level
alpha = 0.99

# Import demand data and data on the amount of units that fit on a location type
units_per_location = pd.read_pickle('Max Units per Location Type.pkl')
demand = pd.read_pickle('Normal demand.pkl')

# Change the dataframes to numpy arrays
units_per_location = units_per_location.to_numpy()
demand = demand.to_numpy()

# Split the demand matrix in the demand part and the SKU names
SKU_names = demand[:,0]
demand = demand[:,1:]

# Remove the SKU names column from the units_per_location matrix
units_per_location = units_per_location[:,1:]

# Initialize a vector to store the amount of units of safety stock needed for each SKU
safety_stock_units = np.zeros([len(demand)])

# Initialize a matrix to store safety stock in terms of locations needed for each SKU
safety_stock_locations = np.zeros([len(demand),len(units_per_location[0])])

# For each SKU...
for i in range(len(demand)):
    #... calculate the mean and standard devation
    mean = np.mean(demand[i])
    std = np.std(demand[i])
    # Use the formula derived in the thesis to set safety stock in units
    safety_stock_units[i] = std * norm.ppf(alpha) + mean
    # Initialize remaing stock as the total safety stock in terms of units
    remaining_stock = safety_stock_units[i]
    # Intialize a variable that tracks if the final feasible location type has been reached
    final = 0
    # For each location type...
    for j in range(len(units_per_location[i])):
        # ... check if units fit on the next location type if not already on the final type
        if j != len(units_per_location[i]) -1:
            if int(units_per_location[i,j+1]) == 0:
                final = 1
        # If the last location type is considered...
        if j == len(units_per_location[i])-1 or final == 1:
            # ... divide the remaining stock by the amount of stock that fits on the final location type and round this up
            safety_stock_locations[i,j] = math.ceil(remaining_stock/units_per_location[i,j])
            break
        # Else...
        else:
            #... divide the remaining stock by the amount of units that fit on the location and round down
            safety_stock_locations[i,j] = math.floor(remaining_stock/units_per_location[i,j])
            # Update remaining stock
            remaining_stock -= safety_stock_locations[i,j] * units_per_location[i,j]
            
# Add the SKU names and safety stock units together
SKU_names = SKU_names[:,np.newaxis]
safety_stock_units = safety_stock_units[:,np.newaxis]
safety_stock_units = np.concatenate((SKU_names, safety_stock_units), axis = 1)

# Turn 'safety_stock_units' into a dataframe
safety_stock_units = pd.DataFrame(safety_stock_units, columns = ['SKU Number', 'Safety Stock in Units'])
 
# Save the data to Excel and to Pickle  
safety_stock_units.to_excel('Safety Stock in Units Normal.xlsx', index = False)
safety_stock_units.to_pickle('Safety Stock in Units Normal.pkl')

# Add the SKU names and safety stock locations together
safety_stock_locations = np.concatenate((SKU_names, safety_stock_locations), axis = 1)

# Create Column names
columns = ['SKU number']
for j in range(len(safety_stock_locations[0])-1):
    columns.append('Location of Type ' +str(j+1) + ' for Safety Stock')

# Turn 'safety_stock_locations' into a dataframe
safety_stock_locations = pd.DataFrame(safety_stock_locations, columns = columns)

# Save the data to Excel and to Pickle  
safety_stock_locations.to_excel('Safety Stock in Locations Normal.xlsx', index = False)
safety_stock_locations.to_pickle('Safety Stock in Locations Normal.pkl')














