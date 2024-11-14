///Single Objetive

The user can choose 5 parameters:

HEURISTIC: 0-does not use heuristic
	   1-uses heuristic

CENTRAL: 0-corner warehouse
	 1-central warehouse

THIRTY: 0-runs 1 time
	1-runs 30 cycles 

FIFTY: 0-uses normal orders from file
       1-all costumers have 50 orders

NUMBER_OF_CITIES: 10/30/50

Once the code starts running it will perform the
EA, and prints on the terminal the values of: 
std, min, avg, max
for every generation.

When the program terminates it will plot the
graph for the best solution and for the cost
evolution throught the generations, and will
print on the terminal the mean and std for the
best runs (if multiple runs are performed).

-------------------------------------------------
///Multi Objective

The user can choose 2 parameters:

CENTRAL: 0-corner warehouse
	 1-central warehouse

NUMBER_OF_CITIES: 10/30/50

Once the code starts running it will perform the
EA, and prints on the terminal the values of: 
std, min, avg, max
for every generation.

When the program terminates it will plot the
pareto front and the best solution. It will also
print on the terminal the values of the (X,Y) 
coordinates for the pareto graph.