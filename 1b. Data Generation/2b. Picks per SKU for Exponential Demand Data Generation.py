# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:18:05 2024

@author: kanev
"""

import numpy as np
import pandas as pd
import math

# Set seed
np.random.seed(124)

# Import demand data
demand = pd.read_pickle('Exponential Demand 1000.pkl')

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
        # If demand is zero...
        if demand[i,j] == 0:
            # ... set the picks for that day to zero
            picks[i,j] = 0
        # Else...
        else:
            #... find the mean demand for this SKU
            mean_demand = np.mean(demand[i])
            # Sample from a poisson distribution with the demand of the current SKU and day divided by the mean demand and add one
            number_of_picks = np.random.poisson(demand[i,j]/mean_demand) + 1
            # Is this number is larger than demand...
            if number_of_picks > demand[i,j]:
                # ... set the number of picks for this SKU and day equal to the demand
                picks[i,j] = demand[i,j]
            # Else...
            else:
                # ... set the number of picks for this day and SKU equal to the sample from the poisson distribution
                picks[i,j] = number_of_picks
    # After all days have been considered for a SKU, add the average to the vector
    average_picks[i] = np.mean(picks[i,:])

# Add the SKU names and average picks together
SKU_names = SKU_names[:,np.newaxis]
average_picks = average_picks[:,np.newaxis]
average_picks_per_SKU = np.concatenate((SKU_names, average_picks), axis = 1)

# Turn 'average_picks_per_SKU' into a dataframe
average_picks_per_SKU = pd.DataFrame(average_picks_per_SKU, columns = ['SKU Number', 'Average Daily Picks'])
 
# Save the data to Excel and to Pickle  
average_picks_per_SKU.to_excel('Mean Daily Picks Exponential 1000.xlsx', index = False)
average_picks_per_SKU.to_pickle('Mean Daily Picks Exponential 1000.pkl')