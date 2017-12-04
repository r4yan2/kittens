import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from database import Database
import helper
from recommender import Recommender
import subprocess

IND_SIZE = 77040

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_float", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual) #usare una funzione definita da noi; al posto di initRepeat in cui usiamo il 30% di uni
# meno uni possibili, inizialmente, per avere dei geni con meno ciarpame possibile; il 30% di uni e il resto di zeri

def evalOneMax(individual):
    kittens_args = ["0", "4", "4", "1"]
    kittens_args.append('[' + ','.join([str(item) for item in individual])  + ']')
    kittens = subprocess.Popen(["pypy", "kittens.py"] + kittens_args)
    kittens.wait()
    results = [elem for elem in helper.read("test_result1")]
    print results
    result = float(results[0][1])
    print result
    return result,


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)
population = toolbox.population(n=50)

NGEN=5

for gen in range(NGEN):
    print "percentage", gen * 100 / NGEN
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
top10 = tools.selBest(population, k=10)
helper.write("bestpop", top10, ',')
