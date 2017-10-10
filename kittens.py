from recommender import Recommender
from database import Database
import helper
from multiprocessing import Process, Queue, cpu_count, Manager
import sys
from operator import itemgetter

# take input from command line sys.argv[0] is the program name
if eval(sys.argv[1]) == 0:
    test = True
    istances = 3
else:
    test = False
    istances = 1

choice = eval(sys.argv[2])

# Initializing the recommender istance
recommender_system = Recommender()

for istance in xrange(istances):
    db = Database(istance) # the database depends on which mode is on (Normal/Test)

    # This list will store the result just before writing to file
    to_write = []

    # This list store the result from the worker process (identifier, playlist, [recommendation])
    results = []

    # The following lines are needed to pass the db object between all process (multiprocessing always on)
    manager = Manager()
    ns = manager.Namespace()
    ns.db = db

    # Sadly in TestMode there is a memory issue which limit the parallel computation
    if test:
        core = 2
    else:
        core = cpu_count()

    target_playlists = db.get_target_playlists()

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
    print "instance", istance
    for i in xrange(len(target_playlists)):
        percentage = (i*100)/len(target_playlists)
        if percentage > completion:
            sys.stdout.write("%i\n" % percentage)
            sys.stdout.flush()
            completion = percentage
        r = q_out.get()
        results.append(r)

    # Terminate worker process
    [p.join() for p in proc]

    # Parse result, depending if test mode in on or off
    if test:
        results = filter(lambda x: x>=0, results)
        average = float(sum(results))/len(results)
        print average
        helper.write("test_result"+str(istance), [average])
    else:
        result = [[x[1], x[2]] for x in sorted(results, key=itemgetter(0))]
        for playlist, recommendation in result:
            elem = [playlist, reduce(lambda x, y: str(x) + ' ' + str(y), recommendation)]
            to_write.append(elem)

        # Initialize the helper instance to write the csv
        helper.write("result", to_write)
