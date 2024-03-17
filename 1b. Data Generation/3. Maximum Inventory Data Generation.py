# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:27:08 2024

@author: kanev
"""

import numpy as np
import pandas as pd
import math

# Set Seed
np.random.seed(124)

# Define multiplier parameters
multiplier_lower = 5
multiplier_upper = 50

# Import demand data
demand = pd.read_pickle('Exponential Demand 10000.pkl')

# Change the dataframe to a numpy array
demand = demand.to_numpy()

# Split the demand matrix in the demand part and the SKU names
SKU_names = demand[:,0]
demand = demand[:,1:]

# Initialize a vector to store maximum inventory
max_inventory = np.zeros(len(demand))

# For each SKU...
for i in range(len(demand)):
    # ... find the mean demand
    mean_demand = np.mean(demand[i])
    # Draw a random multiplier from a Uniform distribution
    multiplier = np.random.uniform(multiplier_lower, multiplier_upper)
    highest_demand = np.amax(demand[i])
    # If the rounded down value of the product of the multiplier and the average demand is larger than or equal to the highest recorded demand...
    if math.floor(mean_demand*multiplier) >= highest_demand:
        # ...maximum inventory is set to the rounded down product of the mean demand and the multiplier
        max_inventory[i] = math.floor(mean_demand*multiplier)
    # Else...
    else:
        # ... maximum inventory is set to the highest recorded demand
        max_inventory[i] = highest_demand

# Add the SKU names and the maximum inventory together
SKU_names = SKU_names[:,np.newaxis]
max_inventory = max_inventory[:,np.newaxis]
max_inventory_per_SKU = np.concatenate((SKU_names, max_inventory), axis = 1)

# Turn 'max_inventory_per_SKU' into a dataframe
max_inventory_per_SKU = pd.DataFrame(max_inventory_per_SKU, columns = ['SKU Name', 'Maximum Inventory'])
 
# Save the demands to Excel and to Pickle  
max_inventory_per_SKU.to_excel('Maximum Inventory Exponential 10000.xlsx', index = False)
max_inventory_per_SKU.to_pickle('Maximum Inventory Exponential 10000.pkl')
