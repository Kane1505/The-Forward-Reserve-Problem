# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:31:53 2024

@author: kanev
"""

import numpy as np
import pandas as pd

# Import data
order_data = pd.read_pickle('Order Data Case Study.pkl').to_numpy()
demand = pd.read_pickle('Demand Data Case Study Clean.pkl').to_numpy()

# Use the clean demand data to retrieve to set of feasible SKUs and their order
SKUs = demand[:,0]

# Split the time stamps to delete times and save dates
for i in range(len(order_data)):
    split = order_data[i,3].split() 
    order_data[i,3] = split[0]

# Find a list of unique dates
unique_dates = np.unique(order_data[:,3], return_counts=False)

# Initialize a vector to store the number of picks for each SKU
picks = np.zeros(len(SKUs))

# For each SKU, find how many times it was picked in the order data
for i in range(len(SKUs)):
    picks[i] = len(np.where(order_data[:,1] == SKUs[i])[0])
    
# Divide the number of picks by the number of days to find the average daily picks
daily_picks = picks/len(unique_dates)

# Add the SKU numbers to the average daily picks
daily_picks = daily_picks[:,np.newaxis]
SKUs = SKUs[:,np.newaxis]
daily_picks = np.concatenate((SKUs, daily_picks), axis = 1)

# Save the data to Excel and Pickle
columns = ['SKU Number', 'Mean Daily Picks']
daily_picks = pd.DataFrame(daily_picks, columns = columns)
daily_picks.to_excel('Mean Daily Picks Case Study.xlsx', index = False)
daily_picks.to_pickle('Mean Daily Picks Case Study.pkl')
