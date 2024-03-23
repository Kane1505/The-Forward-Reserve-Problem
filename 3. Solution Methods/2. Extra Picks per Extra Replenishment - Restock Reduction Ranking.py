# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 12:49:55 2024

@author: kanev
"""

import pandas as pd
import numpy as np
import time
import math
import sys
import copy

# Start timer
t1 = time.time()

# Define c vector contain location sizes in terms of fraction of full-sized location (ordered largest to smallest)
c = [1,0.5,0.25,0.125]

# Define the total available space
S = 10000

# Set epsilon and threshold
epsilon = 0.0001
threshold = 100

# Import all data
p = pd.read_pickle('Mean Daily Picks Normal 10000.pkl')
s = pd.read_pickle('Max Units per Location Type 10000.pkl')
psi = pd.read_pickle('Safety Stock in Locations Empirical 99 Normal 10000.pkl')
demand = pd.read_pickle('Normal Demand 10000.pkl')
theta = pd.read_pickle('Maximum Inventory Normal 10000.pkl')

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

# Set up a range for the total amount of location types
M = range(len(c))

# If there is not enough space to store the safety stock of the first SKU or if there is no space left after storing it, terminate the algorithm
sum_ = 0
for j in M:
    sum_ += psi[0,j] * c[j]
if sum_ >= S:
    print('There is not enough space to store the safety stock of the first SKU or there is no space left after storing the safety stock of the first SKU')
    sys.exit()


# Initialise vector to keep track of previous iterations
extra_picks_per_extra_replenishment = np.zeros(len(du))
replenishments = np.zeros(len(du))
not_selected = np.zeros(len(du))

# Set iteration counters, one that keeps deleted SKUs in mind and one that counts every iteration
iteration = 0
iteration_2 = 0

# While there are still SKUs left to add
while iteration < len(du):
    # Initialise a vector that tracks which SKUs have been assigned their fully capacity
    not_reached_max_stock = np.ones(iteration+1)
    # Set up a range over all SKUs that are considered in the current assignment
    N_sub = range(iteration+1)
    # Initialise a vector to keep track of the solution
    x = np.zeros([iteration+1,len(c)])
    # Initialise a counter to keep track of the available space
    S_prime = S
    # For every SKU that is considered in the current iteration
    for i in N_sub:
        for j in M:
            # Subtract the safety stock from the total available space
            S_prime -= c[j]*psi[i,j]
    
    # Initialize a matrix that stores the previous solution (with ones so it is not equal to the initialized cur_sol matrix)
    prev_sol = np.ones([iteration+1, len(c)])
    # Initialize a matrix that stores the current solution 
    cur_sol = np.zeros([iteration+1, len(c)])
    # Create a copy of S_prime
    S_prime_copy = S_prime
    
    while np.array_equal(prev_sol,cur_sol) == False and sum(not_reached_max_stock) != 0:
        # Copy the previous solution and set it as the current solution
        prev_sol = np.copy(cur_sol)
        # Initialize a vector to store the restock reduction
        restock_reduction = np.zeros(iteration+1)
        # For each SKU that is considered in the current iteration, calculate the restock reduction by going from one allocated location to two
        # Set this value to zero for the SKUs that are already on max capacity
        for i in N_sub:
            restock_reduction[i] = ((du[i]/(s[i,0])) - (du[i]/(2*s[i,0])))*not_reached_max_stock[i]
        # Normalize the vector, so the sum of entries equals 1 to find what percentage of space should be allocated to each SKU
        space_allocation_fraction = restock_reduction/restock_reduction.sum()
        # Multiply this vector with the available amount of space to find how many full-sized locations are allocated to each SKU
        space_allocation = space_allocation_fraction * S_prime
        
        # Initialize a matrix to store allocated locations
        location_allocation = np.zeros([iteration + 1, len(c)])
        # Calculate how many locations of each type this equals
        for i in N_sub:
            for j in M:
                # Round down as to not exceed the maximum capacity
                location_allocation[i,j] = space_allocation[i]//c[j]
                space_allocation[i] -= location_allocation[i,j] * c[j]
        
        # For each SKU
        for i in N_sub:
            # Find the amount of new units allocated
            units_allocated = 0
            # And the amount of units previously allocated
            prev_units_allocated = 0
            for j in M:
                units_allocated += s[i,j]*location_allocation[i,j]
                prev_units_allocated += s[i,j] * prev_sol[i,j]
            # Add these two numbers together to find the total units allocated
            total_units_allocated = prev_units_allocated + units_allocated
            # If this number exceeds the maximum inventory allowed
            if total_units_allocated > theta[i]:
                # Calculate how many locations should be allocated to store maximal inventory and set this as the current solution
                remaining_stock = theta[i]
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
                        cur_sol[i,j] = math.ceil(remaining_stock/s[i,j])
                        break
                    # Else...
                    else:
                        #... divide the remaining stock by the amount of units that fit on the location and round down
                        cur_sol[i,j] = math.floor(remaining_stock/s[i,j])
                        # Update remaining stock
                        remaining_stock -= cur_sol[i,j] * s[i,j]
                # And not_reached_max_stock to zero for this SKU
                not_reached_max_stock[i] = 0
            # If the newly allocated space does not lead to the SKU exceeding its capacity, add the newly allocated space to the solution
            else:
                for j in M:
                    cur_sol[i,j] = prev_sol[i,j] + location_allocation[i,j]
        
        # Update S_prime with the newly allocated stock
        S_prime = S_prime_copy
        for i in N_sub:
            for j in M:
                S_prime -= c[j]*cur_sol[i,j]
    # If the previous solution is equal to the new solution or all SKUs have max capacity, set x to the current solution
    x = cur_sol
    # Calculate the amount of replenishments needed
    replenishments_sub = 0
    for r in N_sub:
        sum_ = 0
        for m in M:
            sum_ += x[r,m]*s[r,m]
        if not_reached_max_stock[r] == 0: 
            replenishments_sub += du[r] / theta[r]
        else:
            replenishments_sub += du[r]/(sum_ + epsilon)
    # Store this value
    replenishments[iteration] = replenishments_sub    
    # Calculate the extra picks per extra replenishment
    extra_picks_per_extra_replenishment[iteration_2] = p[iteration]/(replenishments[iteration]-replenishments[iteration-1])
    # If the extra picks per extra replenishment are lower than the threshold, the SKU is deleted from all data and set to one in the 'not selected' list
    if extra_picks_per_extra_replenishment[iteration_2] < threshold:
        p = np.delete(p,iteration,axis = 0)
        s = np.delete(s,iteration,axis = 0)
        psi = np.delete(psi,iteration,axis = 0)
        du = np.delete(du,iteration,axis = 0)
        dl = np.delete(dl,iteration,axis = 0)
        theta = np.delete(theta,iteration,axis = 0)
        replenishments = np.delete(replenishments, iteration, axis = 0)
        not_selected[iteration_2] = 1
        iteration_2 += 1
    # If it is selected, we move to the next iteration
    else:
        iteration += 1
        iteration_2 += 1

# Delete the SKUs that have not been selected
delete = []
for i in range(len(not_selected)):
    if not_selected[i] == 1:
        delete.append(i)
SKUs = np.delete(SKUs, delete, axis = 0)

# Run the algorithm once more, just for the best selection

# Initialise a vector that tracks which SKUs have been assigned their fully capacity
not_reached_max_stock = np.ones(len(SKUs))
# Initialise a vector that tracks which SKUs have been assigned their fully capacity
max_stock = np.zeros(len(SKUs))
# Set up a range over all SKUs that are considered in the current assignment
N_sub = range(len(SKUs))
# Initialise vectors to keep track of the solution and to store the value of adding a new location to a SKU (add an extra row to the value matrix)
x = np.zeros([len(SKUs), len(c)])
# Initialise a counter to keep track of the available space
S_prime = S
# For every SKU that is considered in the current iteration
for i in N_sub:
    for j in M:
        # Subtract the safety stock from the total available space
        S_prime -= c[j]*psi[i,j]

# Initialize a matrix that stores the previous solution (with ones so it is not equal to the initialized cur_sol matrix)
prev_sol = np.ones([len(SKUs), len(c)])
# Initialize a matrix that stores the current solution 
cur_sol = np.zeros([len(SKUs), len(c)])
# Create a copy of S_prime
S_prime_copy = S_prime

while np.array_equal(prev_sol,cur_sol) == False and sum(not_reached_max_stock) != 0:
    # Copy the previous solution and set it as the current solution
    prev_sol = np.copy(cur_sol)
    # Initialize a vector to store the restock reduction
    restock_reduction = np.zeros(len(SKUs))
    # For each SKU that is considered in the current iteration, calculate the restock reduction by going from one allocated location to two
    # Set this value to zero for the SKUs that are already on max capacity
    for i in N_sub:
        restock_reduction[i] = ((du[i]/(s[i,0])) - (du[i]/(2*s[i,0])))*not_reached_max_stock[i]
    # Normalize the vector, so the sum of entries equals 1 to find what percentage of space should be allocated to each SKU
    space_allocation_fraction = restock_reduction/restock_reduction.sum()
    # Multiply this vector with the available amount of space to find how many full-sized locations are allocated to each SKU
    space_allocation = space_allocation_fraction * S_prime
    
    # Initialize a matrix to store allocated locations
    location_allocation = np.zeros([len(SKUs), len(c)])
    # Calculate how many locations of each type this equals
    for i in N_sub:
        for j in M:
            # Round down as to not exceed the maximum capacity
            location_allocation[i,j] = space_allocation[i]//c[j]
            space_allocation[i] -= location_allocation[i,j] * c[j]
    
    # For each SKU
    for i in N_sub:
        # Find the amount of new units allocated
        units_allocated = 0
        # And the amount of units previously allocated
        prev_units_allocated = 0
        for j in M:
            units_allocated += s[i,j]*location_allocation[i,j]
            prev_units_allocated += s[i,j] * prev_sol[i,j]
        # Add these two numbers together to find the total units allocated
        total_units_allocated = prev_units_allocated + units_allocated
        # If this number exceeds the maximum inventory allowed
        if total_units_allocated > theta[i]:
            # Calculate how many locations should be allocated to store maximal inventory and set this as the current solution
            remaining_stock = theta[i]

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
                    cur_sol[i,j] = math.ceil(remaining_stock/s[i,j])
                    break
                # Else...
                else:
                    #... divide the remaining stock by the amount of units that fit on the location and round down
                    cur_sol[i,j] = math.floor(remaining_stock/s[i,j])
                    # Update remaining stock
                    remaining_stock -= cur_sol[i,j] * s[i,j]
            # And not_reached_max_stock to zero for this SKU
            not_reached_max_stock[i] = 0
        # If the newly allocated space does not lead to the SKU exceeding its capacity, add the newly allocated space to the solution
        else:
            for j in M:
                cur_sol[i,j] = prev_sol[i,j] + location_allocation[i,j]
    
    # Update S_prime with the newly allocated stock
    S_prime = S_prime_copy
    for i in N_sub:
        for j in M:
            S_prime -= c[j]*cur_sol[i,j]
# If the previous solution is equal to the new solution or all SKUs have max capacity, set x to the current solution
x = cur_sol

# Calculate the amount of replenishments needed
replenishments_fin = 0
for r in N_sub:
    sum_ = 0
    for m in M:
        sum_ += x[r,m]*s[r,m]
    if not_reached_max_stock[r] == 0:
        replenishments_fin += du[r]/ theta[r]
    else:
        replenishments_fin += du[r]/(sum_ + epsilon)   


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
results.to_excel('Assignment and Allocation Heuristic 2.xlsx', index = False)
results.to_pickle('Assignment and Allocation Heuristic 2.pkl')

# Print the number of SKUs selected, the number of Replenishments, the number of Picks and the total runtime
print('The number of SKUs selected equals: ' + str(len(SKUs)))
print('The number of replenishments equals: ' + str(replenishments_fin))
print('The number of picks equals: ' + str(picks))
t2 = time.time()
time_elapsed = t2 - t1
print('The total elapsed time equals: ' + str(time_elapsed))