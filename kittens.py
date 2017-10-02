from recommender import Recommender
from database import Database
import time
from helper import Helper
from multiprocessing import Process, Queue, cpu_count, Manager
import sys

disclaimer = """
    --> Kitt<3ns Recommendation ENGINE <--

    USAGE:

    [0] Random
    [1] Top Listened
    [2] Top Included

    Please wait until the Engine is ready, then select your choice
    """
print disclaimer

test = False # should be True when the test mode is tun
db = Database(test) # the database depends on which mode is on (Normal/Test)

# Initializing the recommender istance
recommender_system = Recommender()

# Getting input from user
choice = input("Please select one >  ")

# Just some debug information
start_time = time.time()

# This list will store the result just before writing to file
to_write = []

# This list store the result from the worker process (identifier, playlist, [recommendation])
recommendations = []

# The following lines are needed to pass the db object between all process (multiprocessing always on)
manager = Manager()
ns = manager.Namespace()
ns.db = db

core = cpu_count()
print "\nVROOOOOOOOOMMMMMMMMMMMMMMMMMMMMMMM\n" \
      "Parallel Engine ACTIVATION\n"+\
      "CORE TRIGGERED\n"*core+\
      "VROOOOOOOOOMMMMMMMMMMMMMMMMMMMMMMM\n"

target_playlists = db.get_target_playlists()

# Queue(s) for the process, one for input data to the process, the other for the output data to the main process
q_in = Queue()
q_out = Queue()
[q_in.put((i, x)) for i, x in enumerate(target_playlists)]
# When a process see the -1 know that is the end of the processing
[q_in.put((-1, -1)) for _ in xrange(core)]

# Starting the process
proc = [Process(target=recommender_system.run, args=(choice, ns.db, q_in, q_out))
        for i in xrange(core)]
for p in proc:
    p.daemon = True
    p.start()

# Retrieve results from the out queue and display percentage
for i in xrange(len(target_playlists)):
    completion = (i*100.0)/len(target_playlists)
    sys.stdout.write("\r%.2f%%" % completion)
    sys.stdout.flush()
    r = q_out.get()
    recommendations.append(r)

# Terminate worker process
[p.join() for p in proc]

# Parse result, from
result = map(lambda x: [x[1], x[2]], sorted(recommendations, key=lambda x: x[0]))
for playlist, recommendation in recommendations:
    elem = [playlist, reduce(lambda x, y: str(x) + ' ' + str(y), recommendation)]
    to_write.append(elem)

# Initialize the helper instance to write the csv
helper = Helper("result", ["playlist_id", "track_ids"])
helper.write(to_write)

# END
print "\nCompleted!" \
      "\nResult writed to file correctly!" \
      "\nCompletion time %f" \
      % (time.time() - start_time)
