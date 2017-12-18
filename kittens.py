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
"""
sys.argv[0] kittens
sys.argv[1] Debug: 0 ON, 1 OFF 
sys.argv[2] choice of the recommendations algorithms
sys.argv[3] choice of the number of core to use
sys.argv[4] choice of the instance (0 for recommendation, 1-6 test)
sys.argv[5] individual enabled/disabled if using genetic algorithm
"""

if len(sys.argv) < 5:
    print "Less than 5 arguments passed"
    sys.exit(0)
if eval(sys.argv[1]) == 0:
    test = True
    instance = int(sys.argv[4])
else:
    test = False
    instance = 0

choice = int(sys.argv[2])
core = int(sys.argv[3])

if len(sys.argv) > 5:
    try:
        fp = open(sys.argv[5], "rb")
    except FileNotFoundError as e:
        logging.debug("Not found %s" % (e))
        sys.exit(0)
    individual = helper.parseIntList(fp.readline())
    fp.close()
    db = Database(instance, individual)
    suppress_output = True
    logging.debug("individual parsed, kittens output suppressed\n")
else:
    db = Database(instance)
    suppress_output = False

# This list will store the result just before writing to file
to_write = []

# Initializing the recommender instance
recommender_system = Recommender(db)

# This list store the result from the worker process (identifier, playlist, [recommendation])
results = []

target_playlists = db.get_target_playlists()
target_playlists_length = len(target_playlists)

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

# placeholder for a running map@5
run_map5 = []
run_auc = []
map_playlist = []
auc_playlist = []

done = set()
for i in xrange(target_playlists_length):
    try:
        r = q_out.get(timeout=300)
    except:
        missing = list(done.symmetric_difference(target_playlists))
        logging.debug("Missing: %s, Please request new recommendations manually" % missing)
        break
    if r == -1:
        continue
    if not suppress_output:
        percentage = (i*100)/(target_playlists_length-1)
        if percentage > completion:
            # write percentage in a feasible way for dialog/whiptail
            sys.stdout.write("%i\n" % percentage)
            sys.stdout.flush()
            completion = percentage
    if test: # if the test istance is enabled more logging is done
        map5, auc_score, precision, recall, worker, playlist, playlist_length = r
        logging.debug("worker number %i reporting map %f, auc %f, precision %f, recall %f for playlist %i" % (worker, map5, auc_score, precision, recall, playlist))
        done.add(playlist)

        map_playlist.append([map5, playlist_length])
        auc_playlist.append([auc_score, playlist_length])

        # calculate a running map@5 value
        run_map5.append(map5)
        avg_map5 = helper.mean(run_map5)

        run_auc.append(auc_score)
        avg_auc = helper.mean(run_auc)

        logging.debug("running map5 average %f" % avg_map5)
        logging.debug("running auc average %f" % avg_auc)
        logging.debug("map@5 distribution %s" % Counter(sorted(run_map5)).items())
        results.append([map5, auc_score, precision, recall])
    else:
        identifier, playlist, recommendations = r
        results.append([identifier, playlist, recommendations])
    logging.debug("results length so far: %i" % len(results))

# Terminate worker process
[p.join() for p in proc]
logging.debug("All process terminated succesfully")
# Parse result, depending if test mode in on or off
if test:
    # write test results
    results_length = len(results)
    map5_res, auc_res, precision_res, recall_res = zip(*results)

    avg_map5 = helper.mean(map5_res)
    avg_auc = helper.mean(auc_res)
    avg_precision = helper.mean(precision_res)
    avg_recall = helper.mean(recall_res)

    to_write = [["AVG MAP@5", avg_map5], ["AVG ROC_AUC", avg_auc], ["Precision", avg_precision], ["Recall", avg_recall]]
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
    result = [[playlist, recommendations] for identifier, playlist, recommendations in sorted(results, key=itemgetter(0))]
    for playlist, recommendations in result:
        elem = [playlist, ' '.join(str(x) for x in recommendations)]
        to_write.append(elem)
    logging.debug(to_write)
    helper.write("result", to_write)
