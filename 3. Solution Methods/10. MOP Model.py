# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:05:09 2024

@author: kanev
"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math


# Define c vector contain location sizes in terms of fraction of full-sized location (ordered largest to smallest)
c = [1,0.5,0.25,0.125]

# Define the total available space
S = 16000

# Set epsilon
epsilon = 0.0001

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

# Remove safety stock from maximum inventory, if maximum inventory turn negative, delete the SKU
delete = []
for i in range(len(theta)):
    for j in range(len(c)):
        theta[i] -= psi[i,j]*s[i,j]
    for j in range(len(c)):
        if int(s[i,j]) > 0:
            min_inv = s[i,j]
            break
    if theta[i] < min_inv:
        delete.append(i)
        
s = np.delete(s, delete, axis = 0)
p = np.delete(p, delete, axis = 0)
du = np.delete(du, delete, axis = 0)
psi = np.delete(psi, delete, axis = 0)
theta = np.delete(theta, delete, axis = 0)
SKUs = np.delete(SKUs, delete, axis = 0)

# Set up ranges
N = range(len(du))
M = range(len(c))
    
# Name the model
mdl1 = gp.Model("Cycle Stock")

# Allow for quadratic constraints
mdl1.params.NonConvex = 2

# Set a time limit
mdl1.setParam('TimeLimit', 1800)

# Introduce variables
x = mdl1.addVars([(i,j) for i in N for j in M], vtype=GRB.INTEGER)
z = mdl1.addVars([i for i in N], vtype=GRB.CONTINUOUS)

# Objective function
mdl1.setObjective(gp.quicksum(z[i] for i in N),GRB.MINIMIZE)

# Constraints according to model in thesis
for i in N:
    mdl1.addQConstr((z[i]*(gp.quicksum(x[i,j]*s[i,j] for j in M) + epsilon) == du[i]))
mdl1.addConstr((gp.quicksum(c[j]*x[i,j] for i in N for j in M) <= S - gp.quicksum(c[j]*psi[i,j] for i in N for j in M)))
mdl1.addConstrs((gp.quicksum(s[i,j] * x[i,j] for j in M) <= theta[i]) for i in N)
mdl1.addConstrs((x[i,j] >= 0) for i in N for j in M)

# Optimize the model
mdl1.optimize()

# Results
obj = mdl1.getObjective()
average_daily_restocks = obj.getValue()
x_sol = []
for v in x.values():
    x_sol.append([v.Varname, v.X])
print(x_sol)
