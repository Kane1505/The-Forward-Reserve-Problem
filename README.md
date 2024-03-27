# The-Forward-Reserve-Problem
The process of solving the Forward-Reserve problem is split up in 3 different steps. Firstly, data needs to be made ready, then the data needs to be transformed, and finally solution methods can be applied. It is recommended to execute the programs in the order in which they are presented in the folders. When multiple methods are presented for the same step (i.e. 1a., 1b., etc.), one of the methods can be chosen.
# 1a. Data Collection
Folder 1a. can be chosen if a real life instance is solved. The programs in this folder require data to be provided. The necessary data can be found in the thesis to which this code belongs.

-Program 1. of this folder transforms all datasets to pickle files, which speeds up reading times. Furthermore, the data is stored as the correct data type in this file.

-Program 2. uses historical order data to collect the demand data of all SKUs.

-Program 3. removes all infeasible SKUs from consideration.

-Program 4. calculates the mean daily picks for each SKU.

-Program 5. Calculates the maximum inventory that can be stored for each SKU.

# 1b. Data Generation
Folder 1b. can be chosen if the aim is to test solution methods. This folder provides the means to generate data that can be used for testing. In each program the parameters of the distribution that is used to generate data can be customized.

-Program 1a. is used to generate demand data based on a Normal distribution.

-Program 1b. is used to generate demand data based on a Poisson distribution.

-Program 1c. is used to generate demand data based on an Exponential distribution.

-Program 2a. generates the mean daily picks for each SKU. This version should be used if either the Normal or the Poisson distribution were used to generate demand data.

-Program 2b. generates the mean daily picks for each SKU. This version should be used if the Exponential distribution was used to generate demand data.

-Program 3. generates the maximum inventory for each SKU.

-Program 4. generates the dimensions and weight of each SKU.

# 2. Data Transformation
Folder 2. uses the data from the previous folders to calculate parameters that are important for the solution methods.
-Program 1. calculates how many SKUs fit on each of the location types.

-Program 2. applies the Shapiro-Wilk test on each SKU. This test aims to find out if the demand data is similar to a Normal distribution. The program returns the highest and the lowest p-value. If the lowest value is larger than critical value, it can be assumed that the demand of all SKUs resembles a Normal distribution closely enough to use the Normal distribution in subsequent steps. If the highest p-value is lower than the critical value, all SKUs do not resemble the Normal distribution close enough to use the Normal distribution in subsequent steps. Finally, if the lowest p-value is lower than the critical value, and the highest p-value is higher than the critical value, one or more SKUs resemble a Normal distribution, but the others do not. In this case, the choice can be made to not use the Normal distribution at all, or to make a distinction between the SKUs that do resemble the Normal distribution and SKUs that do not.

-Program 3. is used to find how many times pairs of SKUs are co-ordered.

-Program 4a. sets safety stock using the Normal distribution. This program should be used if the results from program 2. indicate that the demand data of the SKUs resembles a Normal distribution. A service level needs to be set that indicates the probability of having no stockout within one day after the refill point has been triggered.

-Program 4b. sets safety stock using the empirical distribution. This program should be used if the results from program 2. indicate that the demand data of the SKUs does not resemble a Normal distribution. A service level needs to be set that indicates the probability of having no stockout within one day after the refill point has been triggered.

# 3. Solution methods
Folder 3. uses the data from the previous folders to find a solution for the forward-reserve problem. In these files the vector c, the total available space S, and either the EPER threshold or the size of the assignment need to be defined.

-Program 1. contains the main heuristic solution method presented in the thesis.

-Program 2. combines the EPER assignment method with the Restock Reduction allocation heuristic.

-Program 3. combines the EPER assignment method with the Uniform Allocation method.

-Program 4. combines the SKU Ranking assignment method with the One-by-One allocation heuristic.

-Program 5. combines the SKU Ranking assignment method with the Restock Reduction allocation heuristic.

-Program 6. combines the SKU Ranking assignment method with the Uniform Allocation method.

-Program 7. combines the Random Assignment method with the One-by-One allocation heuristic.

-Program 8. combines the Random Assignment method with the Restock Reduction allocation heuristic.

-Program 9. combines the Random Assignment method with the Uniform Allocation method.

-Program 10. contains the exact optimization model.

-Program 11. contains the optimization model with a reduced number of decision variables and relaxed constraints.

-Program 12. contains the main heuristic of the thesis without ranking the SKUs before initiating the iterative part of the heuristic.
