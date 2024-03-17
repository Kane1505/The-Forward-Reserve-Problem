# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:28:31 2024

@author: kanev
"""

import numpy as np
import pandas as pd
import math

# Set a seed
np.random.seed(124)

# Set number of SKUs and days of demand
number_of_SKUs = 10000
days_of_demand = 500

# Set lower and upper limit for the mean of Normal distribution
mu_lower = 1
mu_upper = 50

# Set lower limit of the standard deviation of the Normal distribution
sigma_lower = 0

# Initialize a matrix to store demands
demands = np.zeros([number_of_SKUs,days_of_demand])

# For each SKU...
for i in range(number_of_SKUs):
    # ... draw a mean from the Uniform distribution...
    mu = np.random.uniform(mu_lower,mu_upper)
    # ... and draw a standard deviation from the Uniform distribution
    sigma = np.random.uniform(sigma_lower,mu/3)
    # Draw demand for each day from the Normal distribution
    demand = np.random.normal(mu,sigma, days_of_demand)
    # Round the demands down to the nearest integer for each day
    for j in range(days_of_demand):
        # If demand is negative, set it to zero
        if demand[j] < 0:
            demand[j] = 0
        demand[j] = math.floor(demand[j])
    # Store the demand in the matrix
    demands[i] = demand

# Initialize a list to store the column names of the matrix 'demands'
columns = ['SKU Number']
# Number each day
for i in range(days_of_demand):
    columns.append('Day ' + str(i+1))

# Create a column for SKU names
SKU_names = np.zeros(number_of_SKUs, dtype = object)
for i in range(number_of_SKUs):
    SKU_names[i] = 'SKU ' + str(i+1)
    
# Append the column to the demands matrix
SKU_names = SKU_names[:, np.newaxis]
demands = np.concatenate((SKU_names, demands), axis =1)
    
# Turn 'demands' into a dataframe
demands = pd.DataFrame(demands, columns = columns)
 
# Save the demands to Excel and to Pickle  
demands.to_excel('Normal Demand 10000.xlsx', index = False)
demands.to_pickle('Normal Demand 10000.pkl')







