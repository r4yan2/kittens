import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from database import Database
import helper
from recommender import Recommender
from multiprocessing import Manager, Queue, Process

IND_SIZE = 31900

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_float", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    db = Database(1, individual)
    rm = Recommender()
    results = []

    # trying to multiprocess everything

    #manager = Manager()
    #ns = manager.Namespace()
    #ns.db = db
    target_playlists = db.get_target_playlists()
    core = 2

    # Queue(s) for the process, one for input data to the process, the other for the output data to the main process
    q_in = Queue()
    q_out = Queue()
    [q_in.put((i, x)) for i, x in enumerate(target_playlists)]

    [q_in.put((-1, -1)) for _ in xrange(core)]

    proc = [Process(target=rm.run, args=(4, db, q_in, q_out, 1, i))
            for i in xrange(core)]
    for p in proc:
        p.daemon = True
        p.start()

    works = len(target_playlists)

    for i in xrange(works):
        r = q_out.get()
        if r == -1:
            continue
        results.append(r[0][0])
    [p.join() for p in proc]

    return helper.mean(results),


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
population = toolbox.population(n=300)

NGEN=40

for gen in range(NGEN):
    print "percentage", gen * 100 / NGEN
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
top10 = tools.selBest(population, k=10)
