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
test = False
db = Database(test)
manager = Manager()
ns = manager.Namespace()
ns.db = db
recommender_system = Recommender()
choice = input("Please select one >  ")
start_time = time.time()
to_write = []
result = []
helper = Helper("result", ["playlist_id", "track_ids"])
q_in = Queue()
q_out = Queue()
core = cpu_count()
print "\nVROOOOOOOOOMMMMMMMMMMMMMMMMMMMMMMM\n" \
      "Parallel Engine ACTIVATION\n"+\
      "CORE TRIGGERED\n"*core+\
      "VROOOOOOOOOMMMMMMMMMMMMMMMMMMMMMMM\n"
target = db.get_target_playlists()
[q_in.put((i, x)) for i, x in enumerate(target)]
[q_in.put((-1, -1)) for _ in xrange(core)]
proc = [Process(target=recommender_system.run, args=(choice, ns.db, q_in, q_out))
        for i in xrange(core)]
for p in proc:
    p.daemon = True
    p.start()

for i in xrange(len(target)):
    completion = (i*100.0)/len(target)
    sys.stdout.write("\r%.2f%%" % completion)
    sys.stdout.flush()
    r = q_out.get()
    result.append(r)
[p.join() for p in proc]

result = map(lambda x: [x[1], x[2]], sorted(result, key=lambda x: x[0]))
for playlist, recommendation in result:
    elem = [playlist, reduce(lambda x, y: str(x) + ' ' + str(y), recommendation)]
    to_write.append(elem)

helper.write(to_write)
print "\nCompleted!" \
      "\nResult writed to file correctly!" \
      "\nCompletion time %f" \
      % (time.time() - start_time)
