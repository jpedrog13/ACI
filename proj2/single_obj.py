import random
import numpy as np
from deap import base, creator, tools, algorithms
import pandas as pd
import matplotlib.pyplot as plt

HEURISTIC = 1
CENTRAL = 0
THIRTY = 0
FIFTY = 1


cols = pd.read_csv('CustDist_WHCentral.csv')
distances_central = pd.read_csv('CustDist_WHCentral.csv', usecols = [i for i in cols if i != 'Distances between Customers and Warehouse'])
positions_central = pd.read_csv('CustXY_WHCentral.csv')
cols = pd.read_csv('CustDist_WHCorner.csv')
distances_corner = pd.read_csv('CustDist_WHCorner.csv', usecols = [i for i in cols if i != 'Distances between Customers and Warehouse'])
positions_corner = pd.read_csv('CustXY_WHCorner.csv')
orders = pd.read_csv('CustOrd.csv')

if FIFTY:
    for i in range(1, 51):
        orders['Orders'][i] = 50

if CENTRAL:
    positions = positions_central
    dist = distances_central
else:
    positions = positions_corner
    dist = distances_corner

NUMBER_OF_CITIES = 50 # 10, 30, 50
MU = 50
NGEN = 200
CXPB = 0.9



path = []

#Heuristics central   
def heuristics(pop):
    heu=np.zeros(NUMBER_OF_CITIES, dtype=int)
    count=0
    for i in range(1, NUMBER_OF_CITIES+1):
        if positions_central.iloc[i][1]<=50:
            heu[count]=positions_central.iloc[i][0]
            k=count
            if k>0: 
                while positions_central.iloc[heu[k]][2]<positions_central.iloc[heu[k-1]][2] and k>0:
                    aux = heu[k]
                    heu[k]=heu[k-1]
                    heu[k-1]=aux
                    k=k-1
            
            count=count+1

    for i in range(1, NUMBER_OF_CITIES+1):
        if positions_central.iloc[i][1]>50:
            heu[count]=positions_central.iloc[i][0]
            k=count
            if k>0:            
                while positions_central.iloc[heu[k]][2]>positions_central.iloc[heu[k-1]][2] and positions_central.iloc[heu[k-1]][1]>50 and k>0:
                    aux = heu[k]
                    heu[k]=heu[k-1]
                    heu[k-1]=aux
                    k=k-1
            
            count=count+1
    
    x=[positions.iloc[0][1]]
    y=[positions.iloc[0][2]]
    for i in heu:
        x.append(positions.iloc[i][1])
        y.append(positions.iloc[i][2])
    x.append(positions.iloc[0][1])
    y.append(positions.iloc[0][2])
    

    # fig, ax = plt.subplots()
    # ax.scatter(x, y, c ="blue")
    # plt.plot(x, y, '-r')

    # for i in range(len(heu)):
    #     plt.annotate(heu[i]-1, (x[i], y[i]))

    # ax.set_xlabel("X coordinate")
    # ax.set_ylabel("Y coordinate")

    # plt.show()
    # plt.clf()

    heu = heu - 1
    pop[0][:] = heu[:]
    return pop


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
    total_distance = dist.iloc[start][0]
    load = orders.iloc[start][1]
    for i in range(1, len(individual)):
        end = individual[i] + 1

        if load + orders.iloc[end][1] > 1000:
            total_distance += dist.iloc[start][0] + dist.iloc[0][end]
            load = orders.iloc[end][1]
            path.append(0)
            path.append(end)
            continue
        path.append(end)
        load += orders.iloc[end][1]
        total_distance += dist.iloc[start][end]
        start = end
    path.append(0)
    total_distance += dist.iloc[end][0]
    return (total_distance,)



#%%
def main(seed=None):
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
    runs = 1
    
    if THIRTY:
        runs = 30
    bestdistance = np.inf
    bestrun = None
    distances = []
    for i in range(runs):
        print(i)
        random.seed(seed)

        ## Min.
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
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

        hofList = []
        
        pop = toolbox.population(n=MU)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))
        if HEURISTIC:
            pop = heuristics(pop)
        
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)

        # Begin the generational process
        oldfit = 0
        count = 0
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
            fittest = float(''.join(map(str, hof[0].fitness.values)))
            hofList.append(fittest)

            # Select the next generation population
            pop = toolbox.select(pop + offspring, MU)
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)
            

        distances.append(fittest)
        
        if fittest < bestdistance:
            bestdistance = fittest
            bestrun = hofList
            
    mean = np.mean(distances)
    std = np.std(distances)

    print("mean:", mean, "std:", std)

    return pop, logbook, bestrun

#%% 
if __name__ == "__main__":
    pop, stats, hofList = main()

    gens = []
    for i in range(1, NGEN):
        gens.append(i)
        if i > len(hofList):
            hofList.append(hofList[i-2])

    plt.plot(gens, hofList)
    plt.xlabel("Generations")
    plt.ylabel("Distance")
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




#%%