from recommender import Recommender
from database import Database
import helper
from multiprocessing import Process, Queue, Manager
import sys
from operator import itemgetter
import logging
from collections import Counter

#sys.setcheckinterval(sys.maxint)

# A logfile to take info during the execution
logging.basicConfig(filename='log/kittens.log', level=logging.DEBUG, filemode='w')

# take input from command line sys.argv[0] is the program name
if eval(sys.argv[1]) == 0:
    test = True
    instance = int(sys.argv[4])
else:
    test = False
    instance = 0

choice = int(sys.argv[2])
core = int(sys.argv[3])
individual = helper.parseIntList(helper.read("top1_gen").next()[0])

# Initializing the recommender instance
recommender_system = Recommender()

db = Database(instance) # the database is built accordingly to the number passed, 0 no test else test mode

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
proc = [Process(target=recommender_system.run, args=(choice, db, q_in, q_out, test, i))
        for i in xrange(core)]
for p in proc:
    p.daemon = True
    p.start()

# Retrieve results from the out queue and display percentage
completion = 0
target_playlists_length = len(target_playlists)

# placeholder for a running map@5
run_map5 = []
run_map5_n = 0
map_playlist = []
for i in xrange(target_playlists_length):
    r = q_out.get()
    if r == -1:
        continue
    percentage = (i*100)/(target_playlists_length-1)
    if percentage > completion:
        # write percentage in a feasible way for dialog/whiptail
        sys.stdout.write("%i\n" % percentage)
        sys.stdout.flush()
        completion = percentage
    if test: # if the test istance is enabled more logging is done
        logging.debug("worker number %i reporting result %s for playlist %i" % (r[1],r[0],r[2]))

        (map5, precision, recall) = r[0]
        map_playlist.append([map5, r[3]])

        # calculate a running map@5 value
        run_map5.append(map5)
        run_map5_n += 1
        avg = sum(run_map5)/run_map5_n

        logging.debug("running map5 average %f" % avg)
        logging.debug("map@5 distribution %s" % Counter(sorted(run_map5)).items())
        r=r[0]
    results.append(r)
    logging.debug("results length so far: %i" % len(results))

# Terminate worker process
[p.join() for p in proc]
logging.debug("All process terminated succesfully")
# Parse result, depending if test mode in on or off
if test:
    # write test results
    results_length = len(results)
    map5_res = [map5 for map5, precision, recall in results]
    precision_res = [precision for map5, precision, recall in results]
    recall_res = [recall for map5, precision, recall in results]

    avg_map5 = sum(map5_res)/float(results_length)
    avg_precision = sum(precision_res)/float(results_length)
    avg_recall = sum(recall_res)/float(results_length)

    to_write = [["MAP@5", avg_map5], ["Precision", avg_precision], ["Recall", avg_recall]]
    logging.debug("map@5 distribution %s" % Counter(map5_res).items())
    map_playlist.sort(key=itemgetter(1))
    logging.debug("map@5/playlists_length distribution %s" % map_playlist)

    old_value = map_playlist[0][1]
    cnt = 0.0
    res = 0.0
    map_playlist_mean = []
    for map5, value in map_playlist:
        if value != old_value:
            map_playlist_mean.append([res/cnt, old_value])
            res = map5
            cnt = 1.0
            old_value = value
        else:
            res += map5
            cnt += 1.0

    logging.debug("avg map@5 per playlist length %s" % map_playlist_mean)
    helper.write("map5distr"+str(choice), map_playlist_mean)

    logging.debug(to_write)
    helper.write("test_result"+str(instance), to_write, '\t')
else:
    #write normal results
    result = [[x[1], x[2]] for x in sorted(results, key=itemgetter(0))]
    for playlist, recommendation in result:
        elem = [playlist, reduce(lambda x, y: str(x) + ' ' + str(y), recommendation)]
        to_write.append(elem)
    logging.debug(to_write)
    helper.write("result", to_write)
