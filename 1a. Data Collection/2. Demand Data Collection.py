# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:05:49 2024

@author: kanev
"""

import pandas as pd
import numpy as np

# Import data
order_data = pd.read_pickle('Order Data Case Study.pkl')

# Change the dataframe to a numpy array
order_data = order_data.to_numpy()

# Split the time stamps to delete times and save dates
for i in range(len(order_data)):
    split = order_data[i,3].split() 
    order_data[i,3] = split[0]

# Find a list of unique dates
unique_dates = np.unique(order_data[:,3], return_counts=False)

# Sort by SKU numbers
skunum = np.sort(order_data[:,1])
# Initialize list for unique SKUs
unique_skus = []
# For every SKU, if it is different than the previous number, add it as a unique SKU
for i in range(len(skunum)):
    if skunum[i-1] != skunum[i]:
        unique_skus.append(skunum[i])
# Change the unique SKU list to an array
unique_skus = np.array(unique_skus)

# Initialize array with SKUs on the rows and days on the columns to store daily demand
daily_demand = np.zeros(shape=[len(unique_skus),len(unique_dates)],dtype = object)

# For every order find the SKU and day it belongs to and add the order quantity to the existing quantity for that day and SKU
for i in range(len(order_data)):
    c = np.where(unique_dates == order_data[i,3])[0][0]
    r = np.where(unique_skus == order_data[i,1])[0][0]
    daily_demand[r,c] += int(order_data[i,2])

# Create columns
columns = ['SKU Number']
for i in range(len(unique_dates)):
    columns.append('Day ' + str(i+1))

# Add the SKU numbers in front of the statistics
unique_skus = unique_skus[:,np.newaxis]
daily_demand = np.append(unique_skus, daily_demand, axis = 1)

# Turn array into pandas data frame
daily_demand = pd.DataFrame(daily_demand, columns = columns)
# Save the data to an Excel file and a Pickle file
daily_demand.to_excel('Demand Data Case Study.xlsx', index = False)
daily_demand.to_pickle('Demand Data Case Study.pkl')