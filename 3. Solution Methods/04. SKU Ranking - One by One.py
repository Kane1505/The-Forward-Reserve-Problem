# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:18:09 2024

@author: kanev
"""

import pandas as pd
import numpy as np
import time
import math
import copy
import sys

# Start timer
t1 = time.time()

# Define c vector contain location sizes in terms of fraction of full-sized location (ordered largest to smallest)
c = [1,0.5,0.25,0.125]

# Define the total available space
S = 16000

# Set epsilon and set the number of SKUs considered
epsilon = 0.0001
Number_of_SKUs = 8487

# Import all data
p = pd.read_pickle('Mean Daily Picks Case Study.pkl')
s = pd.read_pickle('Max Units per Location Type Case Study.pkl')
psi = pd.read_pickle('Safety Stock in Locations Empirical 99 Case Study.pkl')
demand = pd.read_pickle('Demand Data Case Study Clean.pkl')
theta = pd.read_pickle('Maximum Inventory Case Study.pkl')

# Transform all data frames into numpy arrays
p = p.to_numpy()
s = s.to_numpy()
psi = psi.to_numpy()
demand = demand.to_numpy()
theta = theta.to_numpy()

# Remove column with SKU names from all matrices and save SKU names
SKUs = p[:,0]
p = p[:,1:]
s = s[:,1:]
psi = psi[:,1:]
demand = demand[:,1:]
theta = theta[:,1:]

# Reduce the dimensions of the vectors
p = np.squeeze(p)
theta = np.squeeze(theta)

# Create mean demand in units vector
du = np.zeros(len(demand))
for i in range(len(demand)):
    du[i] = np.mean(demand[i])

# Create mean demand in locations vector
dl = np.zeros(len(demand))
# Initialize a matrix to store how many locations of each type are needed to store mean daily demand
locations = np.zeros([len(demand), len(c)])
# For each SKU
for i in range(len(demand)):
    remaining_stock = du[i]
    # Intialize a variable that tracks if the final feasible location type has been reached
    final = 0
    # For each location type...
    for j in range(len(s[i])):
        # ... check if units fit on the next location type if not already on the final type
        if j != len(s[i]) -1:
            if int(s[i,j+1]) == 0:
                final = 1
        # If the last location type is considered...
        if j == len(s[i])-1 or final == 1:
            # ... divide the remaining stock by the amount of stock that fits on the final location type and round this up
            locations[i,j] = math.ceil(remaining_stock/s[i,j])
            break
        # Else...
        else:
            #... divide the remaining stock by the amount of units that fit on the location and round down
            locations[i,j] = math.floor(remaining_stock/s[i,j])
            # Update remaining stock
            remaining_stock -= locations[i,j] * s[i,j]
    # Express the total space needed in terms of full-sized locations
    for j in range(len(c)):
        dl[i] += c[j]*locations[i,j]
        
# Remove safety stock from maximum inventory, if maximum inventory turn negative, delete the SKU
delete = []
for i in range(len(theta)):
    for j in range(len(c)):
        theta[i] -= psi[i,j]*s[i,j]
    if theta[i] <= 0:
        delete.append(i)
        
s = np.delete(s, delete, axis = 0)
p = np.delete(p, delete, axis = 0)
dl = np.delete(dl, delete, axis = 0)
du = np.delete(du, delete, axis = 0)
psi = np.delete(psi, delete, axis = 0)
theta = np.delete(theta, delete, axis = 0)
SKUs = np.delete(SKUs, delete, axis = 0)

# Rank SKUs according to descending picks per demand locations ratio
ratio = np.zeros(len(psi))
for i in range(len(p)):
    ratio[i] = p[i]/dl[i]
ranking = ratio.argsort()
des_ranking = ranking[::-1]

s = s[des_ranking]
p = p[des_ranking]
dl = dl[des_ranking]
du = du[des_ranking]
psi = psi[des_ranking]
theta = theta[des_ranking]
SKUs = SKUs[des_ranking]

# Remove all SKUs after the selected number of SKU
s = s[:Number_of_SKUs,:]
p = p[:Number_of_SKUs]
dl = dl[:Number_of_SKUs]
du = du[:Number_of_SKUs]
psi = psi[:Number_of_SKUs,:]
theta = theta[:Number_of_SKUs]
SKUs = SKUs[:Number_of_SKUs]

# Create a copy of theta to calcualte the total replenishments
theta_original = copy.deepcopy(theta)

# Set up a range for the total amount of location types
M = range(len(c))

# Check if a feasible solution is possible by subtracting safety stock from the available space
# And by subtracting the minimum amount of locations needed for a feasible solution
S_check = S
for i in range(len(SKUs)):
    for j in M:
        S_check -= psi[i,j] * c[j]
    for j in M:
        if s[i,len(c)-j-1] != 0:
            S_check -= c[len(c)-j-1]
            break

# If there is not enough space for a feasible solution, too many SKUs are selected
if S_check < 0:
    print('Too many SKUs were selected, there is not enough space for a feasible solution')
    sys.exit()
    

# Initialise a vector that tracks which SKUs have been assigned their fully capacity
max_stock = np.zeros(len(SKUs))
# Set up a range over all SKUs that are considered in the current assignment
N_sub = range(len(SKUs))
# Initialise vectors to keep track of the solution and to store the value of adding a new location to a SKU (add an extra row to the value matrix)
x = np.zeros([len(SKUs), len(c)])
v = np.zeros([len(SKUs)+1, len(c)])


# Initialize the value matrix
for i in N_sub:
    for k in M:
        # If location type k is considered that is not the smallest location on which units fit
        if k != len(c)-1 and int(s[i,k+1]) != 0:
            # If zero units fit, set value to a high number
            if int(s[i,k]) == 0:
                v[i,k] = 1000000
            # If the amount of units that fit exceed the remaining inventory that can be placed, set value to a high number
            elif s[i,k] > theta[i]:
                v[i,k] = 1000000
            # Else define value using the equation from the thesis
            else:
                v[i,k] = ((du[i]/(s[i,k] * (1/c[k]) + epsilon)) - (du[i]/(epsilon)))
        # If the smallest location on which units can be placed is considered...
        else:
            # Set to a high number if no units fit
            if int(s[i,k]) == 0:
                v[i,k] = 1000000
            # Otherwise use the value function provided in the thesis
            else:
                v[i,k] = ((du[i]/(s[i,k] * (1/c[k]) + epsilon)) - (du[i]/(epsilon)))
                
# Track the available space
S_prime = S
# For every SKUs that is considered in the current iteration
for i in N_sub:
    for j in M:
        # Subtract the safety stock from the total available space
        S_prime -= c[j]*psi[i,j]
# While there is still space available
while S_prime > 0:
    # Make a deep copy of c
    c_prime = copy.deepcopy(c)
    # Add a value larger than S_prime in front of the location sizes
    c_prime.insert(0,S_prime+1)
    # For each location type...
    for j in M:
        # ... check if the remaining space excludes some location types a options
        if S_prime >= c_prime[j+1] and S_prime < c_prime[j]:
            # Pick the best value, considering only the columns of locations types that are feasible for the remaining space
            best = np.argmin(np.concatenate((np.zeros([len(SKUs)+1,j]), v[:,j:]), axis = 1), axis = None)
            # Find the row and column of this best value
            row = best//len(c)
            column = best % len(c)
            # If the extra row of the value matrix is selected, it means all available SKUs have been filled up to their limit
            if row == len(SKUs):
                # So the iteration is ended by setting S_prime to a negative number
                S_prime = -10
            # If one of the SKUs is selected
            else:
                # The amount of space of the selection is subtracted from the available space
                S_prime -= c[column]
                # x vector is updated
                x[row,column] += 1
                # Reduce theta for the selected SKU by the number of extra units placed
                theta[row] -= s[row,column]
                # If theta is now negative, set all values to a high number
                if theta[row] < 0:
                    for k in M:
                        v[row,k] = 1000000
                # If theta is still positive...
                else:
                    sum_ = 0
                    for k in M:
                        sum_ += x[row,k]*s[row,k]
                    for k in M:
                        # ...for all location except the smallest on which units still fit...
                        if k != len(c)-1 and int(s[row,k+1]) != 0:
                            # If zero units fit set the value to a high number
                            if int(s[row,k]) == 0:
                                v[row,k] = 1000000
                            # If the amount of units that fit on the location type exceed theta, set the value to a high number
                            elif s[row,k] > theta[row]:
                                v[row,k] = 1000000
                            # Else, set the value using the formula from the thesis
                            else:
                                v[row,k] = ((du[row]/(sum_ + s[row,k] * (1/c[k]) + epsilon)) - (du[row]/(sum_+epsilon)))
                        # If the final feasible location is considered
                        else:
                            # Set value to a high number if zero units fit on the location
                            if int(s[row,k]) == 0:
                                v[row,k] = 1000000
                            # Else, use the function from the thesis
                            else:
                                v[row,k] = ((du[row]/(sum_ + s[row,k] * (1/c[k]) + epsilon)) - (du[row]/(sum_+epsilon)))
                break
        # If there is not enough space left to fill the smallest location, terminate this iteration
        elif S_prime < c[len(c)-1]:
            S_prime = -10
            break
        
# Calculate the number of replenishments needed
replenishments_fin = 0
for i in N_sub:
    sum_ = 0
    for j in M:
        sum_ += x[i,j]*s[i,j]
    if theta[i] < 0:
        replenishments_fin += du[i] / theta_original[i]
    else:
        replenishments_fin += du[i]/(sum_ + epsilon)

# Calculate the number of picks that are done
picks = 0
for i in N_sub:
    picks += p[i]

# Add SKU names to the matrix x
SKUs = SKUs[:,np.newaxis]
results = np.concatenate((SKUs, x), axis = 1)

# Create Column names
columns = ['SKU number']
for j in M:
    columns.append('Location Type ' +str(j+1) + ' for Cycle Stock')

results = pd.DataFrame(results, columns = columns)
# Save results to Excel and pkl
results.to_excel('Assignment and Allocation Heuristic 4.xlsx', index = False)
results.to_pickle('Assignment and Allocation Heuristic 4.pkl')

# Print the number of SKUs selected, the number of Replenishments, the number of Picks and the total runtime
print('The number of SKUs selected equals: ' + str(len(SKUs)))
print('The number of mean daily replenishments equals: ' + str(replenishments_fin))
print('The number of mean daily forward picks equals: ' + str(picks))
t2 = time.time()
time_elapsed = t2 - t1
print('The total elapsed time equals: ' + str(time_elapsed))