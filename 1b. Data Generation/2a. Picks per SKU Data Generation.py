# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:33:38 2024

@author: kanev
"""

import numpy as np
import pandas as pd
import math

# Set seed
np.random.seed(124)

# Import demand data
demand = pd.read_pickle('Poisson Demand 10000.pkl')

# Change the dataframe to a numpy array
demand = demand.to_numpy()

# Split the demand matrix in the demand part and the SKU names
SKU_names = demand[:,0]
demand = demand[:,1:]

# Initialize a matrix for picks per day
picks = np.zeros(np.shape(demand))

# Initialize an average picks vector
average_picks = np.zeros(len(demand))

# Loop over each SKU...
for i in range(len(demand)):
    # ... and over each day of demand
    for j in range(len(demand[i])):
        # If a SKU has zero demand on a certain day...
        if demand[i,j] == 0:
            # ...set the picks for that day to zero
            picks[i,j] = 0
        # else...
        else:
            #... round up a number drawn from a Uniform distribution between 1 and the demand of that day
            picks[i,j] = math.ceil(np.random.uniform(0,demand[i,j]))
    # Calculate the average picks over all days of a certain SKU
    average_picks[i] = np.mean(picks[i,:])

# Add the SKU names and average picks together
SKU_names = SKU_names[:,np.newaxis]
average_picks = average_picks[:,np.newaxis]
average_picks_per_SKU = np.concatenate((SKU_names, average_picks), axis = 1)

# Turn 'average_picks_per_SKU' into a dataframe
average_picks_per_SKU = pd.DataFrame(average_picks_per_SKU, columns = ['SKU Name', 'Average Daily Picks'])
 
# Save the demands to Excel and to Pickle  
average_picks_per_SKU.to_excel('Mean Daily Picks Poisson 10000.xlsx', index = False)
average_picks_per_SKU.to_pickle('Mean Daily Picks Poisson 10000.pkl')







