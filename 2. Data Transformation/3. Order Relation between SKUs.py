# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:01:26 2024

@author: kanev
"""

import pandas as pd
import numpy as np
import itertools
import math

# Define a function that return the unique combinations of elements of a list
def uniqueCombinations(list_elements):
    l = list(itertools.combinations(list_elements, 2))
    s = set(l)
    return list(s)

# Import data
Orders = pd.read_pickle('Order Data Case Study.pkl').to_numpy()
demand = pd.read_pickle('Demand Data Case Study Clean.pkl').to_numpy()

# Find the list of feasible SKUs from the demand data
feas_SKUs = demand[:,0]

# Order the order data by order number
Orders = Orders[np.argsort(Orders[:, 0])]

# Intialize old order number
ord_num_old = 0
# Initialize order length
ord_len = 0
# Initialize list to store indices of current order
ord_ind = []
# Initialize list to store all indices that are relevant to keep
all_ind = []
# For each order line
for i in range(len(Orders)):
    # Find the current order number
    ord_num = Orders[i,0]
    # If it is the same as the order number of the previous order
    if ord_num_old == ord_num:
        # Add 1 to the order length
        ord_len += 1
        # And add the index of this line to the list with indices of the current order
        ord_ind.append(i)
    # If a new order number is found
    else:
        # Check if the length of the previous order is longer than one (i.e. SKUs were ordered together)
        if ord_len != 1:
            # If this is the case, add the indices of this order to the list containing all indices
            for ind in ord_ind:
                all_ind.append(ind)
        # Set the old order number, to the current number
        ord_num_old = ord_num
        # Reset order length to one
        ord_len = 1
        # Add the index of this order to the list tracking the indices of the current order
        ord_ind = [i]
        
# Save all multiline orders
multi_orders = np.take(Orders, all_ind, axis = 0)

# Initialize a dictionary to store SKU pairs
sku_relation = {}
# Initialize a list to track what SKUs are in the current order
skus = []
# Initialize a variable to track the previous order number
ord_num_old = 0
# For each order line
for i in range(len(multi_orders)):
    # Find the order number
    ord_num = multi_orders[i,0]
    # If it is the same as the previous order number
    if ord_num == ord_num_old:
        # If the SKU number is in the set of feasible SKUs
        if multi_orders[i,1] in feas_SKUs:
            # Add the SKU number to the list of SKUs for this order
            skus.append(multi_orders[i,1])
    # If a new order number is found
    else:
        # And it is not the first iteration of the algorithm
        if i != 0:
            # If there are more than 1 SKUs in the list with SKUs
            if len(skus) > 1:
                # Sort the list with SKUs, to ensure the same SKU pair is not stored twice in different orders
                skus = sorted(skus)
                # Find the combinations of the SKUs in the list
                pairs_tup = uniqueCombinations(skus)
                # Initialize a list to store the pairs
                pairs = []
                # For each pair
                for j in range(len(pairs_tup)):
                    # Add the cleaned pair to the list
                    pairs.append(str(pairs_tup[j])[1:-1])
                # For each pair
                for j in range(len(pairs)):
                    # If the pair is already present in the dictionary
                    if pairs[j] in sku_relation:
                        # Add one to the counter
                        sku_relation[pairs[j]] += 1
                    # Otherwise, add it and set the counter to one
                    else:
                        sku_relation[pairs[j]] = 1 
        # After this, re-initialize the list of SKUs with the current SKU number
        if multi_orders[i,1] in feas_SKUs:
            skus = [multi_orders[i,1]]
        # Set the old order number to the current order number
        ord_num_old = ord_num

# Create a data frame with all data
all_data = pd.DataFrame.from_dict(sku_relation, orient="index")
all_data.columns = ['Number of Times Ordered Together']
all_data.index.name = 'SKU Pairs'

# Save the data to an Excel file
all_data.to_excel('SKU Order Relations.xlsx', index = True)
