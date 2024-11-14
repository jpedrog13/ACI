import random
import numpy as np
from deap import base, creator, tools, algorithms
import pandas as pd
import matplotlib.pyplot as plt

cols = pd.read_csv('CustDist_WHCentral.csv')
distances_central = pd.read_csv('CustDist_WHCentral.csv', usecols = [i for i in cols if i != 'Distances between Customers and Warehouse'])
positions_central = pd.read_csv('CustXY_WHCentral.csv')
cols = pd.read_csv('CustDist_WHCorner.csv')
distances_corner = pd.read_csv('CustDist_WHCorner.csv', usecols = [i for i in cols if i != 'Distances between Customers and Warehouse'])
positions_corner = pd.read_csv('CustXY_WHCorner.csv')
orders = pd.read_csv('CustOrd.csv')



NUMBER_OF_CITIES = 10 #10, 30, 50

CENTRAL = 1

if CENTRAL:
    positions = positions_central
else:
    positions = positions_corner

MU = 50
NGEN = 200
CXPB = 0.9

path = []


def mutInversion(individual, indpb):
    size = len(individual)
    if random.random() < indpb:
        swap = random.randint(2, size - 2)
        first=individual[1:swap]
        second=individual[swap:size]
        individual[1:]=second+first
    return individual,     


def EVALUATE(individual):
    global path
    path = [0]
    start = individual[0] + 1
    path.append(start)
    total_distance = distances_central.iloc[start][0]
    current_trip = []
    current_trip.append(start)
    load = orders.iloc[start][1]
    cost = 0
    for i in range(1, len(individual)):
        end = individual[i] + 1
        if load + orders.iloc[end][1] > 1000:
            current_trip.append(0)
            current_trip.insert(0, 0)
            for i in range(len(current_trip)-1):
                cost += distances_central.iloc[current_trip[i]][current_trip[i+1]] * load  +10
                load -= orders.iloc[current_trip[i+1]][1]
            total_distance += distances_central.iloc[start][0] + distances_central.iloc[0][end]
            load = orders.iloc[end][1]
            current_trip = [end]
            path.append(0)
            path.append(end)
            continue
        path.append(end)
        load += orders.iloc[end][1]
        total_distance += distances_central.iloc[start][end]
        start = end
        current_trip.append(start)
    for i in range(len(current_trip)-1):
        cost += distances_central.iloc[current_trip[i]][current_trip[i+1]] * load  +10
        load -= orders.iloc[current_trip[i+1]][1]
    path.append(0)
    total_distance += distances_central.iloc[end][0]
    return (total_distance, cost)

def main(seed=None):
    random.seed(seed)

    random.seed(seed)
    global NUMBER_OF_CITIES
    global MU
    global NGEN
    global CXPB

    print(NUMBER_OF_CITIES)

    if NUMBER_OF_CITIES == 10:
        MU=200
        NGEN=50
    elif NUMBER_OF_CITIES == 30:
        MU=147
        NGEN=68
    elif NUMBER_OF_CITIES == 50:
        MU=105
        NGEN=95
    else:
        NUMBER_OF_CITIES = 50
        MU=105
        NGEN=95

    CXPB = 0.9

    ## Min.
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    ## permutation setup for individual,
    toolbox.register("indices", \
                    random.sample, \
                    range(NUMBER_OF_CITIES), 
                    NUMBER_OF_CITIES)
    toolbox.register("individual", \
                    tools.initIterate, \
                    creator.Individual, toolbox.indices)
    ## population setup,
    toolbox.register("population", \
                    tools.initRepeat, \
                    list, toolbox.individual)


    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", mutInversion, indpb=0.15)
    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", EVALUATE)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    pf = tools.ParetoFront()

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournament(pop, MU, tournsize=3)
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)
        pf.update(pop)

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    #print("Final population hypervolume is %f" % hypervolume(pop, [11.0]))
    
    #print(pop)
    return pop, logbook, pf
        
if __name__ == "__main__":
    pop, stats, pf = main()
    pop.sort(key=lambda x: x.fitness.values)
    
    #print("Convergence: ", convergence(pop, optimal_front))
    #print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))
    
    x, y = zip(*[ind.fitness.values for ind in pf])
    print(x)
    print(y)
    #optimal_front = numpy.array(optimal_front)
    #plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    plt.scatter(x, y, c="b")
    plt.xlabel("Total Distance")
    plt.ylabel("Cost")
    plt.axis("tight")
    plt.show()

    x=[]
    y=[]
    for i in path:
        x.append(positions.iloc[i][1])
        y.append(positions.iloc[i][2])

    fig, ax = plt.subplots()
    ax.scatter(x, y, c ="blue")
    plt.plot(x, y, '-r')

    for i in range(len(path)):
        plt.annotate(path[i], (x[i], y[i]))

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    plt.show()
    plt.clf()

    
