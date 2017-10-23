from recommender import Recommender
from database import Database
import helper
from multiprocessing import Process, Queue, Manager
import sys
from operator import itemgetter
import logging

#sys.setcheckinterval(sys.maxint)

# A logfile to take info during the execution
logging.basicConfig(filename='kittens.log', level=logging.DEBUG, filemode='w')

# take input from command line sys.argv[0] is the program name
if eval(sys.argv[1]) == 0:
    test = True
    instance = sys.argv[4]
else:
    test = False
    instance = 0

choice = eval(sys.argv[2])
core = eval(sys.argv[3])

# Initializing the recommender instance
recommender_system = Recommender()

db = Database(instance) # the database is istancied accordingly to the number passed, 0 no test else test mode

# This list will store the result just before writing to file
to_write = []

# This list store the result from the worker process (identifier, playlist, [recommendation])
results = []

# The following lines are needed to pass the db object between all process (multiprocessing always on)
manager = Manager()
ns = manager.Namespace()
ns.db = db

target_playlists = db.get_target_playlists()

logging.debug("len of target playlist %i" % len(target_playlists))

# Queue(s) for the process, one for input data to the process, the other for the output data to the main process
q_in = Queue()
q_out = Queue()
[q_in.put((i, x)) for i, x in enumerate(target_playlists)]
# When a process see the -1 know that is the end of the processing
[q_in.put((-1, -1)) for _ in xrange(core)]

# Starting the process
proc = [Process(target=recommender_system.run, args=(choice, ns.db, q_in, q_out, test, i))
        for i in xrange(core)]
for p in proc:
    p.daemon = True
    p.start()

# Retrieve results from the out queue and display percentage
completion = 0
target_playlists_length = len(target_playlists)
for i in xrange(target_playlists_length):
    r = q_out.get()
    percentage = (i*100)/(target_playlists_length-1)
    if percentage > completion:
        sys.stdout.write("%i\n" % percentage)
        sys.stdout.flush()
        completion = percentage
    if test:
        logging.debug("worker number %i reporting result %s" % (r[1],r[0]))
        r = r[0]
    results.append(r)
    logging.debug("results length so far: %i" % len(results))

# Terminate worker process
[p.join() for p in proc]
logging.debug("All process terminated succesfully")
# Parse result, depending if test mode in on or off
if test:
    results_length = len(results)
    avg_map5 = sum([map5 for map5, precision, recall in results])/float(results_length)
    avg_precision = sum([precision for map5, precision, recall in results])/float(results_length)
    avg_recall = sum([recall for map5, precision, recall in results])/float(results_length)
    to_write = [["MAP@5", avg_map5], ["Precision", avg_precision], ["Recall", avg_recall]]
    logging.debug(to_write)
    helper.write("test_result"+str(instance), to_write, '\t')
else:
    result = [[x[1], x[2]] for x in sorted(results, key=itemgetter(0))]
    for playlist, recommendation in result:
        elem = [playlist, reduce(lambda x, y: str(x) + ' ' + str(y), recommendation)]
        to_write.append(elem)
    logging.debug(to_write)
    helper.write("result", to_write)
